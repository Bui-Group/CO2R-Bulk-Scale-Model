# ============================================================================
# run_stage2.jl — Stage 2 driver (§11): replace electrolyte-side Dirichlet with
# the Sherwood–Reynolds convective Neumann BC. Same DOFs/solver as Stage 1.
# Self-anchoring continuation (§9.5.5): anchor at −0.300 V_RHE from the
# equilibrium guess, tangent predictor down the grid. The Stage 1 cache is only
# checked as a driver-order guard (§13); it is not needed as a seed (§10.5).
# ============================================================================

const HERE = @__DIR__
include(joinpath(HERE, "params.jl"))
include(joinpath(HERE, "mesh.jl"))
include(joinpath(HERE, "chemistry.jl"))
include(joinpath(HERE, "kinetics.jl"))
include(joinpath(HERE, "transport.jl"))
include(joinpath(HERE, "gas_transport.jl"))   # GasLayout type used by the Stage 3–5 residual signatures in assembly.jl
include(joinpath(HERE, "solver_sparse.jl"))
include(joinpath(HERE, "assembly.jl"))
include(joinpath(HERE, "cache.jl"))

const STAGE1_CACHE = joinpath(HERE, "output_stage1", "cache")
const STAGE2_DIR   = joinpath(HERE, "output_stage2")
const STAGE2_CACHE = joinpath(STAGE2_DIR, "cache")
const PREFIX  = "s2"

function build_equilibrium_u0(N::Int, c_eq_vec)
    ln_c_eq = log.(c_eq_vec)
    u0 = zeros(5 * N)
    for ix in 1:N
        u0[liquid_idx(ix, 1):liquid_idx(ix, 5)] .= ln_c_eq
    end
    return u0
end

function run_stage2()
    println("="^70)
    println("=== Stage 2 ===")
    @printf("i0_CO2R = %.0f A/m², λ_MHC = %.2f eV\n", i0_CO2R(), lambda_MHC())
    @printf("HER_pf = %.3e, α_c = %.4f\n", HER_prefactor(), alpha_c_HER())
    @printf("Mesh: N_CL = %d (graded q=1.3)\n", N_CL)
    @printf("k_MT [m/s] = %s\n", k_MT)
    @printf("Cache dir: %s\n", STAGE2_CACHE)
    isdir(STAGE1_CACHE) && !isempty(readdir(STAGE1_CACHE)) ||
        error("No Stage 1 cache — run run_stage1.jl first (§13 driver order)")
    println("="^70)

    cl_mesh  = make_cl_mesh(N_CL, L_CL; q = 1.3)
    c_eq_vec = equilibrium_vec(1000.0, 1000.0)

    # Biot numbers (§ Stage 2 check): Bi_i = k_MT_i × L_CL / D_eff_i
    Bi = [k_MT[i] * L_CL / D_eff[i] for i in 1:5]
    @printf("Biot numbers Bi_i = k_MT·L_CL/D_eff: %s  (expect O(0.1–10))\n",
            join([@sprintf("%.2f", b) for b in Bi], ", "))

    Ndof       = 5 * N_CL
    rows, cols = block_tridiag_pattern(N_CL, 5)
    Jtemplate  = pattern_matrix(rows, cols, Ndof)
    plu        = AffineLU()
    blocks     = dof_blocks(2)
    Vs = sort(collect(V_RHE_grid), by = v -> -v)

    V_ref = Ref(Vs[1])
    res!  = (F_, u_) -> residual_stage2!(F_, u_, V_ref[], cl_mesh, c_eq_vec)
    assemble! = make_ad_assembler(res!, Ndof; chunk = 12)

    pol = NamedTuple[]; prof_rows = String[]
    u  = build_equilibrium_u0(N_CL, c_eq_vec)    # §9.5.5 anchor guess
    t0 = time()
    sweep_log = continuation_sweep!(u, Vs, res!, V_ref;
        blocks = blocks, plu = plu, jac_assembler = assemble!, J = Jtemplate, n_log_dofs = Ndof,
        on_converged = (V_now, uu, info) -> begin
            save_solution(STAGE2_CACHE, PREFIX, copy(uu), V_now)

            V_app = V_now - V_RHE_to_SHE_offset
            iCO = compute_i_CO(uu, V_app, cl_mesh) / 10.0
            iH2 = compute_i_H2(uu, V_app, cl_mesh) / 10.0
            itot = iCO + iH2; FEco = itot > 0 ? 100*iCO/itot : 0.0
            push!(pol, (V_RHE=V_now, V_SHE=V_app, iCO=iCO, iH2=iH2, itot=itot,
                        FE_CO=FEco, FE_H2=100-FEco, iter=info.iterations, Finf=info.residual))

            c_all = unpack_log_conc(uu, N_CL); rco, rh2 = rate_profiles(uu, V_app, cl_mesh)
            for ix in 1:N_CL
                cv = @view c_all[:, ix]; cCs = c_Cs_from_EN(cv); pH = -log10(cv[1]/1000.0)
                push!(prof_rows, @sprintf("%.6f,%.6f,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e",
                      V_now, cl_mesh.x_centers[ix]*1e6, pH, cv[1], cv[2], cv[3], cv[4], cv[5], cCs, rco[ix], rh2[ix]))
            end
            @printf("V_RHE=%.3f: iCO=%7.2f iH2=%7.2f itot=%7.2f mA/cm²  FE_CO=%5.1f%%  iter=%d ‖F‖∞=%.1e\n",
                    V_now, iCO, iH2, itot, FEco, info.iterations, info.residual)
        end)
    nconv  = count(l -> l.converged, sweep_log)
    n_iter = sum(l.iterations for l in sweep_log)
    maxF   = maximum((l.residual for l in sweep_log if l.converged); init = 0.0)
    @printf("\nStage 2 sweep: %d/%d converged in %.1f s (%d Newton iterations, %.2f/step); max ‖F‖∞ = %.2e\n",
            nconv, length(Vs), time()-t0, n_iter, n_iter/length(Vs), maxF)

    mkpath(STAGE2_DIR)
    open(joinpath(STAGE2_DIR, "polarization.csv"), "w") do f
        println(f, "V_RHE,V_SHE,iCO_mAcm2,iH2_mAcm2,itot_mAcm2,FE_CO,FE_H2,iters,Finf")
        for p in sort(pol, by=r->r.V_RHE)
            @printf(f, "%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%d,%.3e\n",
                    p.V_RHE, p.V_SHE, p.iCO, p.iH2, p.itot, p.FE_CO, p.FE_H2, p.iter, p.Finf)
        end
    end
    open(joinpath(STAGE2_DIR, "profiles.csv"), "w") do f
        println(f, "V_RHE,x_um,pH,c_H,c_OH,c_HCO3,c_CO3,c_CO2,c_Cs,R_CO2R,R_HER")
        for row in prof_rows; println(f, row); end
    end

    # ── Physicality checks (§ Stage 2) ──
    println("\n--- Physicality checks ---")
    Vmax = Vs[end]                                  # most cathodic (Vs is mild→cathodic)
    u_hi = load_solution(STAGE2_CACHE, PREFIX, Vmax)
    c_all = unpack_log_conc(u_hi, N_CL)
    @printf("1. Biot numbers (above) O(0.1–10): %s\n",
            all(0.01 .< Bi .< 100) ? "in range" : "CHECK")
    @printf("2. c_CO2(L_CL) @ V_RHE=%.3f = %.4f mol/m³ (expect →0, depletion-limited)\n",
            Vmax, c_all[5, N_CL])
    @printf("3. c_OH(L_CL) @ same V = %.4e mol/m³ vs bulk c_eq_OH=%.4e (expect ABOVE bulk)\n",
            c_all[2, N_CL], c_eq_vec[2])
    # 4. electrolyte-face flux = k_MT*(c(L_CL)-c_eq)
    println("4. electrolyte-face flux k_MT·(c(L_CL)−c_eq) [mol/m²/s]:")
    for (i, nm) in enumerate(("H","OH","HCO3","CO3","CO2"))
        J = k_MT[i]*(c_all[i,N_CL]-c_eq_vec[i])
        @printf("     %-5s: %+.3e  (%s)\n", nm, J, J>0 ? "leaving CL" : "into CL")
    end
    polV = sort(pol, by=r->-r.V_RHE)
    @printf("5. FE_CO peak: %.1f%% at V_RHE=%.3f\n",
            maximum(p->p.FE_CO, polV), polV[argmax([p.FE_CO for p in polV])].V_RHE)

    println("\nStage 2 complete. Plots: run plot_results.py 2")
    return nconv, length(Vs)
end

run_stage2()
