# run_stage2.jl — Stage 2 driver (§Stage 2)
# Replaces the electrolyte-side Dirichlet with the Sherwood-Reynolds convective
# Neumann BC. Same 5×N_CL DOFs, same solver stack. Self-anchoring continuation
# (§9.5.5): anchor at −0.300 V_RHE from the equilibrium guess, tangent predictor
# down the grid (the Stage 1 cache is only checked as a driver-order guard, §13).
# Runs the §Stage-2 physicality checks, plots A1/A2 + overlay, STOPS.

const HERE = @__DIR__
include(joinpath(HERE, "params.jl"))
include(joinpath(HERE, "mesh.jl"))
include(joinpath(HERE, "chemistry.jl"))
include(joinpath(HERE, "kinetics.jl"))
include(joinpath(HERE, "transport.jl"))
include(joinpath(HERE, "solver.jl"))
include(joinpath(HERE, "cache.jl"))
include(joinpath(HERE, "assembly.jl"))
include(joinpath(HERE, "plots.jl"))

const STAGE1_CACHE = joinpath(HERE, "output_stage1", "cache")
const STAGE2_CACHE = joinpath(HERE, "output_stage2", "cache")
const N_CL  = parse(Int, get(ENV, "MESH_N_CL", "80"))

function main()
    println("="^70)
    println("=== Stage 2: Sherwood-Reynolds Neumann BC at electrolyte side ===")
    @printf("i0_CO2R = %.0f A/m²,  λ_MHC = %.2f eV\n", i0_CO2R(), lambda_MHC())
    @printf("HER_pf  = %.3e mA/cm²,  α_c = %.4f\n", HER_prefactor(), alpha_c_HER())
    @printf("Mesh: N_CL = %d  (DOFs = %d)\n", N_CL, 5*N_CL)
    @printf("Re = %.0f,  k_MT = %s m/s\n", Re_elec, round.(k_MT, sigdigits=3))
    println("="^70)

    cl_mesh = make_cl_mesh(N_CL)
    eq = solve_buffer_equilibrium(1000.0, 1000.0)
    c_eq = [eq.H, eq.OH, eq.HCO3, eq.CO3, eq.CO2]

    # Biot numbers Bi_i = k_MT,i × L_CL / D_eff,i  (§Stage-2 check; expect O(0.1–10))
    Bi = [k_MT[i]*L_CL/D_eff[i] for i in 1:5]
    spec = ["H⁺","OH⁻","HCO₃⁻","CO₃²⁻","CO₂"]
    println("Biot numbers (k_MT·L_CL/D_eff):")
    for i in 1:5; @printf("  %-6s Bi=%.3f\n", spec[i], Bi[i]); end

    if isempty(list_cache(STAGE1_CACHE, "s1_"))
        error("No Stage 1 cache found — run run_stage1.jl first.")
    end

    # ── §9.5.5 continuation workspace (same as Stage 1) ──
    Vs = sort(collect(V_RHE_grid), by = v -> -v)
    ND = 5 * N_CL
    rows, cols = block_tridiag_pattern(N_CL, 5)
    Jtemplate = pattern_matrix(rows, cols, ND)
    plu = AffineLU()
    blocks = dof_blocks(2, N_CL)
    V_ref = Ref(Vs[1])
    res_closure!(F, uu) = residual_stage2!(F, uu, V_ref[], cl_mesh, c_eq)
    assemble! = make_ad_assembler(res_closure!, ND; chunk=12)
    println("Solver: §9.5 stack — newton_affine! + AffineLU + ForwardDiff AD; §9.5.5 continuation, anchor at −0.300 V from equilibrium\n")

    results = Tuple{Float64, Vector{Float64}}[]
    nconv = Ref(0); maxF = Ref(0.0)
    u = repeat(log.(c_eq), N_CL)                  # anchor guess: bulk equilibrium in every cell
    t0 = time()
    sweep_log = continuation_sweep!(u, Vs, res_closure!, V_ref;
        blocks=blocks, plu=plu, jac_assembler=assemble!, J=Jtemplate, n_log_dofs=ND,
        on_converged = (V, uu, info) -> begin
            save_solution(STAGE2_CACHE, "s2_", copy(uu), V)
            push!(results, (V, copy(uu))); nconv[] += 1
            maxF[] = max(maxF[], info.residual)
        end)
    t_sweep = time() - t0
    n_iter = sum(l.iterations for l in sweep_log)
    @printf("\nConverged: %d/%d voltages in %.1f s\n", nconv[], length(Vs), t_sweep)
    @printf("  anchor V_RHE=%.3f: %d iters, ||F||inf=%.2e | %d Newton iterations total, %.2f/step | max ||F||inf over sweep = %.2e (§9.5.6 level 1e-6)\n",
            Vs[1], sweep_log[1].iterations, sweep_log[1].residual, n_iter, n_iter/length(Vs), maxF[])
    sort!(results, by = r -> r[1])

    if nconv[] == 0
        println("\n*** Stage 2 did NOT converge at any voltage. ***")
        println("    See BUILD_DIALOGUE.md (Sherwood BC sign) and test_literal_sign_stage2.jl.")
        println("    No plots (no converged solutions).")
    else
        # ── §Stage-2 boundary physicality checks ──
        println("\nBoundary values at the electrolyte face (cell N):")
        @printf("  bulk eq: c_OH=%.4e  c_CO2=%.4f mol/m³\n", c_eq[2], c_eq[5])
        for Vtarget in [-0.816, -1.205, -1.644, -2.079, -2.413]
            r = results[argmin(abs.([x[1] for x in results] .- Vtarget))]
            c = unpack_log_conc(r[2], N_CL)
            @printf("  V_RHE=%.3f : c_OH(L)=%.4e (%.2f× bulk)  c_CO2(L)=%.4f  pH(L)=%.2f\n",
                    r[1], c[2,end], c[2,end]/c_eq[2], c[5,end], -log10(c[1,end]/1000))
        end
        # sweep-wide ranges (the numbers cited in BUILD_DIALOGUE.md); results sorted ascending V_RHE
        ratio = [unpack_log_conc(r[2], N_CL)[2,end]/c_eq[2] for r in results]
        cco2L = [unpack_log_conc(r[2], N_CL)[5,end] for r in results]
        pHL   = [-log10(unpack_log_conc(r[2], N_CL)[1,end]/1000) for r in results]
        @printf("  sweep-wide (mild → cathodic): c_OH(L)/bulk %.1f× → %.0f×;  c_CO2(L) %.2f → %.2f mol/m³;  pH(L) %.2f → %.2f\n",
                ratio[end], ratio[1], cco2L[end], cco2L[1], pHL[end], pHL[1])
        iCOs = [compute_i_CO(r[2], r[1]-V_RHE_to_SHE_offset, cl_mesh)/10 for r in results]
        kpk = argmax(iCOs)
        @printf("  i_CO peak = %.1f mA/cm² at V_RHE = %.3f\n", iCOs[kpk], results[kpk][1])

        pdir = joinpath(HERE, "output_stage2"); mkpath(pdir)
        plot_A1(results, cl_mesh, joinpath(pdir, "stage2_A1_polarization.png"); stage_label="Stage 2")
        plot_A2(results, cl_mesh, joinpath(pdir, "stage2_A2_liquid.png"); stage_label="Stage 2")
        s1 = [(V, load_solution(STAGE1_CACHE,"s1_",V)) for V in list_cache(STAGE1_CACHE,"s1_")]
        plot_polarization_overlay([s1, results], ["S1","S2"], cl_mesh,
            joinpath(pdir, "stage2_overlay_vs_stage1.png"); title="Stage 1 (Dirichlet) vs Stage 2 (Sherwood)")
        println("\nPlots: stage2_A1_polarization.png, stage2_A2_liquid.png, stage2_overlay_vs_stage1.png")
    end

    println("\n" * "="^70)
    println("STAGE 2 COMPLETE — STOP GATE. Review plots before Stage 3.")
    println("="^70)
    return nconv[], length(Vs)
end

main()
