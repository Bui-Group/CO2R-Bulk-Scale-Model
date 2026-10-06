# run_stage5.jl — Stage 5 driver (§Stage 5, §9.5, §9.7, §11).
# Full electrochemistry with FITTED kinetics: φ_L (NP migration + charge balance,
# incl. Cs⁺) and φ_s (solid Ohm's law). Solver: the §9.5 stack with the unified
# continuation driver (§9.5.5) — anchor ONCE at V_c = 0 from a guess constructed from
# physics (equilibrium liquid, inlet gas, φ_s = φ_L = 0; never from a cache, §11
# gotcha #0/#9), pre-ramp in −0.025 V steps to the first output point, then walk the
# 89-V grid with the tangent predictor. Post-solve residual assertion ‖F‖∞ ≤ 1e-6 at
# every emitted voltage (§9.5.6). Only the 89 grid voltages are cached.

const HERE = @__DIR__
for f in ["params.jl","mesh.jl","chemistry.jl","kinetics.jl","transport.jl",
          "gas_transport.jl","solver.jl","cache.jl","assembly.jl",
          "assembly_stage3.jl","assembly_stage4.jl","assembly_stage5.jl","plots.jl"]
    include(joinpath(HERE, f))
end
include(joinpath(HERE, "output_fit", "fitted_params.jl"))

const FIT_CACHE    = joinpath(HERE, "output_fit", "cache")
const STAGE5_CACHE = joinpath(HERE, "output_stage5", "cache")
const N_CL  = parse(Int, get(ENV, "MESH_N_CL", "80"))
const N_GDL = parse(Int, get(ENV, "MESH_N_GDL", "120"))

function main()
    println("="^70)
    println("=== Stage 5: φ_L (NP migration) + φ_s (solid Ohm) — FITTED kinetics ===")
    @printf("i0_CO2R=%.2f A/m²  λ_MHC=%.4f eV  HER_pf=%.3e  α_c=%.5f\n",
            i0_CO2R(), lambda_MHC(), HER_prefactor(), alpha_c_HER())
    @printf("σ_s_eff GDL=%.1f CL=%.1f S/m | Mesh N_CL=%d N_GDL=%d DOFs=%d\n",
            sigma_s_eff_GDL, sigma_s_eff_CL, N_CL, N_GDL, n_dof_stage5(N_CL,N_GDL))
    println("Solver: §9.5 stack — newton_affine! + AffineLU + ForwardDiff AD; §9.5.5 continuation anchored at V_c = 0")
    println("="^70)

    cl_mesh = make_cl_mesh(N_CL); gdl_mesh = make_gdl_mesh(N_GDL)
    eq = solve_buffer_equilibrium(1000.0, 1000.0); c_eq = [eq.H,eq.OH,eq.HCO3,eq.CO3,eq.CO2]
    Vgrid = list_cache(FIT_CACHE, "fit_")
    isempty(Vgrid) && error("No fit cache — run run_fit.jl first (fitted kinetics + the fitted-kinetics Stage-4 reference sweep).")
    nlog = 5 * N_CL
    NDOF = n_dof_stage5(N_CL, N_GDL)
    Vs = sort(collect(V_RHE_grid), by = v -> -v)

    # ── anchor at V_c = 0 (§9.5.5): V_RHE = offset ⇒ V_applied_SHE = 0 ──
    V_anchor = V_RHE_to_SHE_offset
    V_ref = Ref(V_anchor)
    res_closure!(F, uu) = residual_stage5!(F, uu, V_ref[], cl_mesh, gdl_mesh, c_eq)
    u = make_u0_stage5_anchor(N_CL, N_GDL, c_eq)      # equilibrium liquid, inlet gas, φ_s = φ_L = 0

    # ── Fixed sparsity pattern by probing (perturb φ_s/φ_L so the migration cross-terms, which
    #    vanish exactly at φ_L = 0, appear); union of the pattern at the anchor voltage and at the
    #    first output voltage (kinetics active). Hand-deriving the Stage 5 pattern is error-prone. ──
    perturb = zeros(NDOF)
    for ix in 1:N_CL; perturb[phi_L_idx(ix,N_CL,N_GDL)] = 0.005; perturb[phi_s_cl_idx(ix,N_CL,N_GDL)] = 0.01; end
    for ig in 1:N_GDL; perturb[phi_s_gdl_idx(ig,N_CL,N_GDL)] = 0.01; end
    V_ref[] = V_anchor; r1, c1 = probe_sparsity(res_closure!, u; perturb=perturb)
    V_ref[] = Vs[1];    r2, c2 = probe_sparsity(res_closure!, u; perturb=perturb)
    V_ref[] = V_anchor
    Jtemplate = pattern_matrix(vcat(r1, r2), vcat(c1, c2), NDOF)
    @printf("Sparsity pattern: %d nnz (%.3f%% dense)\n", nnz(Jtemplate), 100*nnz(Jtemplate)/NDOF^2)
    assemble! = make_ad_assembler(res_closure!, NDOF; chunk=12)
    plu = AffineLU()
    blocks = dof_blocks(5, N_CL, N_GDL)

    # ── V chain: anchor at V_c = 0, pre-ramp in −0.025 V steps down to the first output point,
    #    then the 89 output points ──
    n_pre = floor(Int, (V_anchor - Vs[1]) / 0.025 - 1e-9)
    pre_ramp = [V_anchor - 0.025*s for s in 0:n_pre]
    chain = vcat(pre_ramp, Vs)
    is_output = Set(Vs)
    @printf("Continuation chain: %d points (%d pre-ramp + %d output); V_c = 0 → %.3f V\n",
            length(chain), length(pre_ramp), length(Vs), Vs[end] - V_RHE_to_SHE_offset)

    results = Tuple{Float64,Vector{Float64}}[]; nconv = Ref(0); maxF = Ref(0.0); t0 = time()
    sweep_log = continuation_sweep!(u, chain, res_closure!, V_ref;
        blocks=blocks, plu=plu, jac_assembler=assemble!, J=Jtemplate, n_log_dofs=nlog,
        shift_fallback! = (uu, dV) -> shift_phis!(uu, dV, N_CL, N_GDL),
        on_converged = (V, uu, info) -> begin
            if V in is_output
                save_solution(STAGE5_CACHE, "s5_", copy(uu), V)
                push!(results, (V, copy(uu))); nconv[] += 1
                maxF[] = max(maxF[], info.residual)
            end
        end)
    n_iter = sum(l.iterations for l in sweep_log)
    n_sub  = sum(l.substeps for l in sweep_log)
    @printf("\nSwept: %d/%d output voltages converged in %.1f s (%d chain points, %d Newton iterations, %.2f/step, %d sub-step recoveries)\n",
            nconv[], length(Vs), time()-t0, length(chain), n_iter, n_iter/length(chain), n_sub)
    @printf("  anchor V_c=0: %d iters, ||F||inf=%.2e | max ||F||inf over the %d output voltages = %.2e (§9.5.6 level 1e-6)\n",
            sweep_log[1].iterations, sweep_log[1].residual, nconv[], maxF[])
    sort!(results, by = r -> r[1])

    # ── §Stage-5 physicality checks ──
    if !isempty(results)
        println("\nStage-5 diagnostics (φ_s drop, Δφ_L, charge conservation):")
        for Vt in [-0.816, -1.205, -1.644, -2.079, -2.413]
            r = results[argmin(abs.([x[1] for x in results] .- Vt))]
            V = r[1]; u = r[2]; Vapp = V - V_RHE_to_SHE_offset; c = unpack_log_conc(u, N_CL)
            iCO, iH2 = s5_currents(u, cl_mesh, N_GDL); ifar = iCO + iH2
            φsN = u[phi_s_cl_idx(N_CL,N_CL,N_GDL)]; φs_drop = (φsN - Vapp)*1000
            φL1 = u[phi_L_idx(1,N_CL,N_GDL)]*1000
            is1 = -sigma_s_eff_GDL*(u[phi_s_gdl_idx(1,N_CL,N_GDL)] - Vapp)/gdl_mesh.dx_face_left
            # ionic current at right boundary (incl migration)
            iion = 0.0
            for i in 1:5
                Leff = D_eff[i]/k_MT[i]
                Ji = np_flux(c[i,N_CL], c_eq[i], u[phi_L_idx(N_CL,N_CL,N_GDL)], 0.0, D_eff[i], z[i], Leff)
                iion += z[i]*Ji
            end
            iion *= F_const
            # compare to the fitted-kinetics Stage-4 reference at this V (§17.8)
            uf = load_solution(FIT_CACHE, "fit_", V)
            dCO = uf === nothing ? NaN : 100*(iCO/10 - compute_i_CO(uf, Vapp, cl_mesh)/10)/(compute_i_CO(uf, Vapp, cl_mesh)/10)
            @printf("  V_RHE=%.3f iCO=%.0f iH2=%.0f mA/cm² (Δi_CO vs fit %+.0f%%) | φs_drop=%.1f mV Δφ_L=%.1f mV | charge:|is1|=%.0f ifar=%.0f rel=%.1e | i_ion_R rel=%.1e\n",
                    V, iCO/10, iH2/10, dCO, φs_drop, φL1, abs(is1), ifar, abs(abs(is1)-ifar)/ifar, abs(abs(iion)-ifar)/ifar)
        end
        iCOs = [s5_currents(r[2], cl_mesh, N_GDL)[1]/10 for r in results]
        kpk = argmax(iCOs)
        @printf("  Stage-5 i_CO peak = %.2f mA/cm² at V_RHE = %.3f\n", iCOs[kpk], results[kpk][1])
    end

    # ── Plots (A1, A2, B1, D1, D2) ──
    pdir = joinpath(HERE, "output_stage5"); mkpath(pdir)
    fit_results = [(V, load_solution(FIT_CACHE,"fit_",V)) for V in Vgrid]
    plot_A1_s5(results, cl_mesh, N_GDL, joinpath(pdir,"stage5_A1_polarization.png"), fit_results)
    plot_A2_s5(results, cl_mesh, N_GDL, joinpath(pdir,"stage5_A2_liquid.png"))
    plot_B1(results, cl_mesh, gdl_mesh, joinpath(pdir,"stage5_B1_gas.png"); stage_label="Stage 5")
    plot_D1(results, cl_mesh, gdl_mesh, N_GDL, joinpath(pdir,"stage5_D1_phi_s.png"))
    plot_D2(results, cl_mesh, N_GDL, joinpath(pdir,"stage5_D2_phi_L.png"))
    println("\nPlots: stage5_A1_polarization.png, stage5_A2_liquid.png, stage5_B1_gas.png, stage5_D1_phi_s.png, stage5_D2_phi_L.png")

    println("\n" * "="^70)
    println("STAGE 5 COMPLETE — FINAL STOP GATE. Review plots.")
    println("="^70)
end

main()
