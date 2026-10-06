# run_fit.jl — Fit step (after Stage 4): Nelder-Mead on 4 kinetic parameters
# (i0_CO2R, λ_MHC, log₁₀ HER_pf, α_c) against the 5 Lees et al. current pairs,
# using the Stage 4 residual (φ_L=0, φ_s=V_applied). §17.
#
# Each NM evaluation polishes at the EXACT experimental voltages with the §9.5 kernel
# (per-voltage newton_affine!, no continuation), warm-started from the Stage-4 cache /
# the running exp_cache. After the fit converges, applies the fitted kinetics and runs
# a full 89-V Stage-4 refit sweep to output_fit/cache/fit_V*.bin — the Stage-4-physics
# reference at fitted kinetics that the Stage 5 checks compare against (§17.8).

include(joinpath(@__DIR__, "co2r.jl"))
include(joinpath(@__DIR__, "diagnostics.jl"))
using Optim

const OUT_DIR    = joinpath(HERE, "output_fit")
const CACHE_DIR  = joinpath(OUT_DIR, "cache")
const S4_CACHE   = joinpath(HERE, "output_stage4", "cache")
mkpath(CACHE_DIR)

const clm  = make_cl_mesh()
const gdlm = make_gdl_mesh()
const gm   = build_gas_mesh(gdlm, clm)
const c_eq = equilibrium_vec(1000.0, 1000.0)
const Ndof = 5*clm.N + 4*gm.Ng
const n_log    = 5*clm.N

# ── §9.5 solver workspace: fixed liquid|gas pattern, persistent LU, weight blocks. The residual
#    closure reads the voltage through V_ref and the kinetics through their Refs, so one AD
#    assembler serves every polish at every voltage and parameter set.
const Jt     = let (r, c) = build_gas_pattern(clm.N, gm.N_GDL); pattern_matrix(r, c, Ndof) end
const plu    = AffineLU()
const blocks = dof_blocks(4, clm.N, gm.N_GDL)
const V_ref  = Ref(0.0)
res_closure!(F, uu) = residual_stage3!(F, uu, V_ref[], clm, gm, c_eq; advect = true)
const assemble_J! = make_ad_assembler(res_closure!, Ndof)

# Fitting parameter bounds (§17.2): [i0_CO2R, λ_MHC, log10 HER_pf, α_c]
const P0    = [540.0, 1.8, log10(2.409e13), 0.0872]
const PLO   = [10.0, 0.5,  8.0, 0.01]
const PHI   = [5000.0, 3.0, 18.0, 0.50]

# per-voltage polish (§9.5.5 Fit row): the §9.5 kernel, no continuation; `aggressive` retries
# with cautious initial damping (alpha_init = 1e-2) and a longer iteration budget.
function newton_polish!(u::Vector{Float64}, V_rhe::Float64; aggressive::Bool = false)
    V_ref[] = V_rhe
    return newton_affine!(u, res_closure!; blocks = blocks, plu = plu, jac_assembler = assemble_J!, J = Jt,
                          alpha_init = aggressive ? 1e-2 : 1.0, max_iter = aggressive ? 200 : 40,
                          n_log_dofs = n_log)
end

# polish Stage-4 at an exact experimental voltage, warm-started from exp_cache/grid.
function polish!(exp_cache, V_rhe)
    u0 = if haskey(exp_cache, V_rhe)
        exp_cache[V_rhe]
    else
        u4, _ = load_nearest(S4_CACHE, "s4_", V_rhe); u4
    end
    u = copy(u0)
    info = newton_polish!(u, V_rhe)
    if !info.converged
        u = copy(u0); info = newton_polish!(u, V_rhe; aggressive = true)
    end
    info.converged || return false
    exp_cache[V_rhe] = u
    return true
end

function make_objective(exp_cache)
    return function (p::Vector{Float64})
        any(p .< PLO) && return 1e6
        any(p .> PHI) && return 1e6
        set_i0_CO2R!(p[1]); set_lambda_MHC!(p[2])
        set_HER_prefactor!(10.0^p[3]); set_alpha_c_HER!(p[4])
        for V in E_exp_RHE
            polish!(exp_cache, V) || return 1e6
        end
        sse = 0.0
        for (k, V) in enumerate(E_exp_RHE)
            u = exp_cache[V]; Vsh = V - V_RHE_to_SHE_offset
            iCO = compute_i_CO(u, Vsh, clm) / 10.0
            iH2 = compute_i_H2(u, Vsh, clm) / 10.0
            sse += (log10(max(iCO, 0.1)) - log10(i_CO_exp[k]))^2
            sse += (log10(max(iH2, 0.1)) - log10(i_H2_exp[k]))^2
        end
        return sse
    end
end

function report_fit(p)
    set_i0_CO2R!(p[1]); set_lambda_MHC!(p[2]); set_HER_prefactor!(10.0^p[3]); set_alpha_c_HER!(p[4])
    exp_cache = Dict{Float64,Vector{Float64}}()
    for V in E_exp_RHE; polish!(exp_cache, V); end
    println("\n── Fit vs Lees et al. (Stage-4 residual, apparent kinetics) ──")
    @printf("%8s | %8s %8s %6s | %8s %8s %6s\n", "V_RHE", "iCO_mdl","iCO_exp","err%","iH2_mdl","iH2_exp","err%")
    fecos = Float64[]
    for (k, V) in enumerate(E_exp_RHE)
        u = exp_cache[V]; Vsh = V - V_RHE_to_SHE_offset
        iCO = compute_i_CO(u, Vsh, clm)/10.0; iH2 = compute_i_H2(u, Vsh, clm)/10.0
        eCO = 100*(iCO - i_CO_exp[k])/i_CO_exp[k]; eH2 = 100*(iH2 - i_H2_exp[k])/i_H2_exp[k]
        push!(fecos, 100*iCO/(iCO+iH2))
        @printf("%8.3f | %8.1f %8.1f %+6.0f | %8.1f %8.1f %+6.0f\n",
                V, iCO, i_CO_exp[k], eCO, iH2, i_H2_exp[k], eH2)
    end
    @printf("FE_CO model at 5 pts: %s%%  (exp %s)\n",
            round.(fecos, digits=1), round.(FE_CO_exp, digits=1))
    return exp_cache
end

function main()
    @printf("=== Fit step (after Stage 4) ===  (N_CL=%d, N_GDL=%d)\n", N_CL, N_GDL)
    isdir(S4_CACHE) && !isempty(readdir(S4_CACHE)) || error("Stage 4 cache missing.")
    @printf("Initial guess: i0=%.0f λ=%.2f HER_pf=%.3e α_c=%.4f\n", P0[1], P0[2], 10^P0[3], P0[4])

    exp_cache = Dict{Float64,Vector{Float64}}()
    obj = make_objective(exp_cache)
    @printf("Initial SSE = %.4f\n", obj(P0))

    t0 = time()
    res = optimize(obj, P0, NelderMead(),
                   Optim.Options(iterations = 200, g_tol = 1e-6, f_reltol = 1e-6, show_trace = true, show_every = 10))
    t_nm = time() - t0
    pfit = Optim.minimizer(res)
    @printf("\nNM done: %d iterations, %d objective evaluations, %.1f s\n",
            Optim.iterations(res), Optim.f_calls(res), t_nm)
    @printf("Fitted: i0_CO2R=%.2f A/m², λ_MHC=%.3f eV, HER_pf=%.4e, α_c=%.4f  (SSE=%.4f)\n",
            pfit[1], pfit[2], 10^pfit[3], pfit[4], Optim.minimum(res))

    report_fit(pfit)

    # ── save fitted parameters to disk for Stage 5 ──
    open(joinpath(OUT_DIR, "fitted_params.txt"), "w") do io
        @printf(io, "i0_CO2R %.10e\nlambda_MHC %.10e\nHER_prefactor %.10e\nalpha_c %.10e\n",
                pfit[1], pfit[2], 10^pfit[3], pfit[4])
    end

    # ── final 89-V Stage-4 refit sweep with fitted kinetics → output_fit/cache/fit_V*.bin ──
    #    per-voltage polish, warm-started from the previous voltage's fitted solution (in basin);
    #    on failure fall back to the Stage-4 grid cache with cautious damping
    println("\nRunning 89-V refit sweep (fitted kinetics) → output_fit/cache/ …")
    Vs = sort(collect(V_RHE_grid), by = v -> -v)
    rows = Vector{NTuple{7,Float64}}(); ok = 0
    u = nothing
    t0 = time()
    for V in Vs
        u4, _ = load_nearest(S4_CACHE, "s4_", V)
        u === nothing && (u = copy(u4))
        info = newton_polish!(u, V)
        if !info.converged
            u = copy(u4); info = newton_polish!(u, V; aggressive = true)
        end
        if info.converged
            save_solution(CACHE_DIR, "fit_", copy(u), V)
            push!(rows, polarization_row(copy(u), V, clm, V - V_RHE_to_SHE_offset)); ok += 1
        else
            @printf("  refit V_RHE=%.3f FAILED (%s)\n", V, info.reason)
        end
    end
    @printf("refit sweep: %d/%d converged in %.1f s\n", ok, length(Vs), time() - t0)

    write_polarization(joinpath(OUT_DIR, "polarization.csv"), rows)
    cp(joinpath(HERE, "output_stage4", "polarization.csv"),
       joinpath(OUT_DIR, "prev_polarization.csv"); force = true)
    okV = [r[1] for r in rows]
    sel = select_profile_voltages(sort(okV, by = v -> -v); nsel = 9)
    Vsel = sort(okV, by = v -> -v)[sel]
    sols = Dict(r[1] => load_solution(CACHE_DIR, "fit_", r[1]) for r in rows)
    liq = [(V, sols[V], fill(V - V_RHE_to_SHE_offset, clm.N), zeros(clm.N)) for V in Vsel]
    write_liquid_profiles(joinpath(OUT_DIR, "liquid_profiles.csv"), clm, liq)
    write_gas_profiles(joinpath(OUT_DIR, "gas_profiles.csv"), gm, clm.N, [(V, sols[V]) for V in Vsel])

    fecos = [r[6] for r in rows]; ipk = argmax(fecos)
    @printf("peak FE_CO (89-V sweep) = %.1f%% at V_RHE=%.3f (exp 79.4%% @ −1.205)\n", fecos[ipk], rows[ipk][1])

    println("\nGenerating plots …")
    withenv("PREV_LABEL" => "Stage 4 (pre-fit)") do
        run(`python3 $(joinpath(HERE, "plot_results.py")) $(OUT_DIR) "Fit (Stage-4 residual, fitted kinetics)" --gas`)
    end
    @printf("\n=== Fit complete: %d refit solutions cached in %s ===\n", ok, CACHE_DIR)
    println("Fitted params written to output_fit/fitted_params.txt")
end

main()
