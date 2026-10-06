# ============================================================================
# run_fit.jl — Kinetic parameter fit (§17), run after Stage 4.
# Fits (i0_CO2R, λ_MHC, log10 HER_prefactor, α_c) to the 5 Lees voltages using
# the Stage 4 residual (φ_L=0, φ_s=V_applied). Pre-fit polish at exact exp
# voltages with the §9.5.3 kernel (per-voltage `newton_affine!`, warm-started from
# the Stage 4 cache; aggressive retry alpha_init=1e-2); Nelder-Mead in log-space
# for HER_pf. After convergence, runs one final 89-V Stage 4 refit sweep at the
# fitted kinetics → output_fit/cache/fit_V*.bin — the Stage-4-physics reference at
# fitted kinetics that the Stage 5 checks compare against (§17.8). Stage 5 itself is
# seeded from the V_c = 0 anchor (§9.5.5), not from this cache.
# ============================================================================

const HERE = @__DIR__
include(joinpath(HERE, "params.jl"))
include(joinpath(HERE, "mesh.jl"))
include(joinpath(HERE, "chemistry.jl"))
include(joinpath(HERE, "kinetics.jl"))
include(joinpath(HERE, "transport.jl"))
include(joinpath(HERE, "gas_transport.jl"))
include(joinpath(HERE, "solver_sparse.jl"))
include(joinpath(HERE, "assembly.jl"))
include(joinpath(HERE, "cache.jl"))

using Optim

const STAGE4_CACHE = joinpath(HERE, "output_stage4", "cache")
const FIT_DIR   = joinpath(HERE, "output_fit")
const FIT_CACHE = joinpath(FIT_DIR, "cache")

const cl_mesh  = make_cl_mesh(N_CL, L_CL; q = 1.3)
const gdl_mesh = make_gdl_mesh(N_GDL, L_GDL; q = 1.3)
const gas_mesh = make_gas_mesh(gdl_mesh, cl_mesh)
const ly       = GasLayout(N_CL, N_GDL)
const c_eq_vec = equilibrium_vec(1000.0, 1000.0)
const Ndof     = ndof_stage3(ly)

# §9.5 workspace shared by every polish: fixed pattern, persistent LU, one AD assembler
# (Ref-backed kinetics ⇒ the assembler never needs rebuilding when the parameters change).
const V_ref     = Ref(0.0)
const res_fit!  = (F_, u_) -> residual_stage4!(F_, u_, V_ref[], cl_mesh, gas_mesh, c_eq_vec, ly)
const Jtemplate = let (r, c) = build_gas_pattern(ly); pattern_matrix(r, c, Ndof) end
const plu       = AffineLU()
const blocks    = dof_blocks(4; N_CL_ = ly.N_CL, N_GDL_ = ly.N_GDL)
const assemble_fit! = make_ad_assembler(res_fit!, Ndof; chunk = 12)

nearest_grid_u(V_rhe) = begin
    Vs = list_cache(STAGE4_CACHE, "s4")
    isempty(Vs) && error("No Stage 4 cache — run run_stage4.jl first")
    load_solution(STAGE4_CACHE, "s4", Vs[argmin(abs.(Vs .- V_rhe))])
end

# Polish Stage 4 at exact V_rhe from warm-start u0 (copy); return (converged, u).
# Same kernel as the sweeps, no continuation (§9.5.5 Fit row).
function solve_stage4_at(V_rhe, u0; aggressive::Bool = false)
    u = copy(u0)
    V_ref[] = V_rhe
    info = newton_affine!(u, res_fit!; blocks = blocks, plu = plu, jac_assembler = assemble_fit!,
                          J = Jtemplate, alpha_init = aggressive ? 1e-2 : 1.0,
                          max_iter = aggressive ? 200 : 40, n_log_dofs = 5*N_CL)
    return info.converged, u
end

# Persistent polish cache (warm-start carries across NM evals).
const exp_cache = Dict{Float64,Vector{Float64}}()
function init_exp_cache!()
    for V in E_exp_RHE; exp_cache[V] = copy(nearest_grid_u(V)); end
end

const neval = Ref(0)
function fitting_objective(p)
    neval[] += 1
    (10.0 ≤ p[1] ≤ 5000.0 && 0.5 ≤ p[2] ≤ 3.0 && 8.0 ≤ p[3] ≤ 18.0 && 0.01 ≤ p[4] ≤ 0.50) || return 1e6
    set_i0_CO2R!(p[1]); set_lambda_MHC!(p[2]); set_HER_prefactor!(10.0^p[3]); set_alpha_c_HER!(p[4])
    sse = 0.0
    for (k, V) in enumerate(E_exp_RHE)
        conv, u = solve_stage4_at(V, exp_cache[V])
        conv || return 1e6
        exp_cache[V] = u                                  # commit warm-start
        Vapp = V - V_RHE_to_SHE_offset
        iCO = compute_i_CO(u, Vapp, cl_mesh) / 10.0
        iH2 = compute_i_H2(u, Vapp, cl_mesh) / 10.0
        sse += (log10(max(iCO, 0.1)) - log10(i_CO_exp[k]))^2
        sse += (log10(max(iH2, 0.1)) - log10(i_H2_exp[k]))^2
    end
    return sse
end

function run_fit()
    println("="^70); println("=== Fit step (after Stage 4) ===")
    isdir(STAGE4_CACHE) && !isempty(readdir(STAGE4_CACHE)) || error("Run Stage 4 first")
    @printf("Initial kinetics: i0=%.0f, λ=%.2f, HER_pf=%.3e, α_c=%.4f\n",
            i0_CO2R(), lambda_MHC(), HER_prefactor(), alpha_c_HER())
    println("="^70)

    init_exp_cache!()
    params0 = [540.0, 1.8, log10(2.409e13), 0.0872]
    # Initial polish at default params (also triggers JIT)
    set_i0_CO2R!(params0[1]); set_lambda_MHC!(params0[2])
    set_HER_prefactor!(10.0^params0[3]); set_alpha_c_HER!(params0[4])
    println("Initial polish at default kinetics...")
    for V in E_exp_RHE
        conv, u = solve_stage4_at(V, exp_cache[V])
        conv || ((conv, u) = solve_stage4_at(V, exp_cache[V]; aggressive = true))
        @assert conv "initial polish failed at V_RHE=$V"
        exp_cache[V] = u
    end

    println("Nelder-Mead (≤200 iters)...")
    t0 = time()
    result = optimize(fitting_objective, params0, NelderMead(),
                      Optim.Options(iterations = 200, g_tol = 1e-6,
                                    f_reltol = 1e-6, show_trace = false))
    pbest = Optim.minimizer(result)
    @printf("NM done: %d objective evals, %.1f s, SSE=%.4f\n",
            neval[], time()-t0, Optim.minimum(result))
    set_i0_CO2R!(pbest[1]); set_lambda_MHC!(pbest[2])
    set_HER_prefactor!(10.0^pbest[3]); set_alpha_c_HER!(pbest[4])

    println("\n=== FITTED PARAMETERS ===")
    @printf("i0_CO2R       = %.2f A/m²   (init 540)\n", i0_CO2R())
    @printf("λ_MHC         = %.4f eV    (init 1.8)\n", lambda_MHC())
    @printf("HER_prefactor = %.4e      (init 2.409e13)\n", HER_prefactor())
    @printf("α_c_HER       = %.5f       (init 0.0872)\n", alpha_c_HER())

    # ── Fit quality at the 5 Lees points (§17.6) ──
    println("\n=== Fit quality (per-point) ===")
    @printf("%-8s %-8s %-8s %-7s | %-8s %-8s %-7s\n",
            "V_RHE","iCO_mod","iCO_exp","err%","iH2_mod","iH2_exp","err%")
    for (k, V) in enumerate(E_exp_RHE)
        conv, u = solve_stage4_at(V, exp_cache[V]); exp_cache[V] = u
        Vapp = V - V_RHE_to_SHE_offset
        iCO = compute_i_CO(u, Vapp, cl_mesh)/10.0; iH2 = compute_i_H2(u, Vapp, cl_mesh)/10.0
        eCO = 100*abs(iCO-i_CO_exp[k])/i_CO_exp[k]; eH2 = 100*abs(iH2-i_H2_exp[k])/i_H2_exp[k]
        @printf("%-8.3f %-8.1f %-8.1f %-7.1f | %-8.1f %-8.1f %-7.1f\n",
                V, iCO, i_CO_exp[k], eCO, iH2, i_H2_exp[k], eH2)
    end

    # ── Final 89-V refit sweep at fitted kinetics → output_fit/cache/fit_V*.bin (§17.8) ──
    # Per-voltage polish carrying the previous fitted state forward; on failure re-seed
    # from the Stage 4 grid cache at that voltage with the aggressive setting.
    println("\nRefit sweep (89 V at fitted kinetics) → output_fit/cache ...")
    Vs = sort(collect(V_RHE_grid), by = v -> -v)
    mkpath(FIT_CACHE)
    pol = NamedTuple[]; prof_rows = String[]; gas_rows = String[]
    nconv = 0
    u = copy(load_solution(STAGE4_CACHE, "s4", Vs[1]))
    for V_now in Vs
        conv, unew = solve_stage4_at(V_now, u)
        if !conv
            u4 = load_solution(STAGE4_CACHE, "s4", V_now)
            conv, unew = solve_stage4_at(V_now, u4 === nothing ? u : u4; aggressive = true)
        end
        conv || (@printf("refit FAILED at V_RHE=%.3f\n", V_now); continue)
        u = unew
        save_solution(FIT_CACHE, "fit", copy(u), V_now); nconv += 1
        Vapp = V_now - V_RHE_to_SHE_offset
        iCO = compute_i_CO(u, Vapp, cl_mesh)/10.0; iH2 = compute_i_H2(u, Vapp, cl_mesh)/10.0
        itot = iCO+iH2; FEco = itot>0 ? 100*iCO/itot : 0.0
        push!(pol, (V_RHE=V_now, V_SHE=Vapp, iCO=iCO, iH2=iH2, itot=itot, FE_CO=FEco, FE_H2=100-FEco))
        c_all = unpack_log_conc(u, N_CL); rco, rh2 = rate_profiles(u, Vapp, cl_mesh)
        for ix in 1:N_CL
            cv = @view c_all[:, ix]
            push!(prof_rows, @sprintf("%.6f,%.6f,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e",
                  V_now, cl_mesh.x_centers[ix]*1e6, -log10(cv[1]/1000.0), cv[1], cv[2], cv[3],
                  cv[4], cv[5], c_Cs_from_EN(cv), rco[ix], rh2[ix]))
        end
        for cc in 1:(N_GDL+N_CL)
            w = gas_cell_omega(u, ly, cc); x = omega_to_x(w); pp = gas_cell_p(u, ly, cc)
            push!(gas_rows, @sprintf("%.6f,%.6f,%.8e,%.8e,%.8e,%.8e",
                  V_now, gas_mesh.x_centers[cc]*1e6, x[1]*pp/101325, x[2]*pp/101325, x[3]*pp/101325, pp/101325))
        end
    end
    @printf("Refit: %d/%d converged\n", nconv, length(Vs))

    open(joinpath(FIT_DIR, "polarization.csv"), "w") do f
        println(f, "V_RHE,V_SHE,iCO_mAcm2,iH2_mAcm2,itot_mAcm2,FE_CO,FE_H2,iters,Finf")
        for p in sort(pol, by=r->r.V_RHE)
            @printf(f, "%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,0,0\n",
                    p.V_RHE, p.V_SHE, p.iCO, p.iH2, p.itot, p.FE_CO, p.FE_H2)
        end
    end
    open(joinpath(FIT_DIR, "profiles.csv"), "w") do f
        println(f, "V_RHE,x_um,pH,c_H,c_OH,c_HCO3,c_CO3,c_CO2,c_Cs,R_CO2R,R_HER")
        for row in prof_rows; println(f, row); end
    end
    open(joinpath(FIT_DIR, "gasprofiles.csv"), "w") do f
        println(f, "V_RHE,x_um,pCO2_atm,pCO_atm,pH2_atm,ptot_atm")
        for row in gas_rows; println(f, row); end
    end
    # Persist fitted parameters for run_stage5.jl
    open(joinpath(FIT_DIR, "fitted_params.txt"), "w") do f
        @printf(f, "i0_CO2R=%.10e\nlambda_MHC=%.10e\nHER_prefactor=%.10e\nalpha_c_HER=%.10e\n",
                i0_CO2R(), lambda_MHC(), HER_prefactor(), alpha_c_HER())
    end

    # FE_CO peak
    polV = sort(pol, by=r->-r.V_RHE)
    fe = [p.FE_CO for p in polV]
    @printf("\nFE_CO peak = %.1f%% at V_RHE=%.3f (target 70–85%%)\n",
            maximum(fe), polV[argmax(fe)].V_RHE)
    println("Fitted params written to output_fit/fitted_params.txt")
    println("Fit step complete. Plots: run plot_results.py fit")
    return pbest
end

run_fit()
