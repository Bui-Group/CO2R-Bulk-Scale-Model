# run_stage3.jl — Stage 3 driver (§Stage 3)
# Adds Stefan-Maxwell gas transport (GDL+CL), volumetric CO₂ phase transfer R_PT,
# gas→liquid Henry left BC. u_G=0 (no Darcy yet), pressure pinned. Kinetics still
# simplified (φ_s=V_applied, φ_L=0).
#
# Solver: the §9.5 stack with the unified continuation driver (§9.5.5) — anchor at
# −0.300 V_RHE from equilibrium liquid + inlet gas (the §9.5.5 table), then the
# tangent predictor down the grid. The Stage 2 cache is only a driver-order guard.

const HERE = @__DIR__
for f in ["params.jl","mesh.jl","chemistry.jl","kinetics.jl","transport.jl",
          "gas_transport.jl","solver.jl","cache.jl",
          "assembly.jl","assembly_stage3.jl","plots.jl"]
    include(joinpath(HERE, f))
end

const STAGE2_CACHE = joinpath(HERE, "output_stage2", "cache")
const STAGE3_CACHE = joinpath(HERE, "output_stage3", "cache")
const N_CL  = parse(Int, get(ENV, "MESH_N_CL", "80"))
const N_GDL = parse(Int, get(ENV, "MESH_N_GDL", "120"))

"Number of derivative sign changes along a sampled curve (smoothness check)."
deriv_sign_changes(x) = (d = diff(x); count(k -> d[k]*d[k+1] < 0, 1:length(d)-1))

function main()
    println("="^70)
    println("=== Stage 3: Stefan-Maxwell gas transport + phase transfer (R_PT) ===")
    @printf("i0_CO2R=%.0f A/m²  λ_MHC=%.2f eV  HER_pf=%.3e  α_c=%.4f\n",
            i0_CO2R(), lambda_MHC(), HER_prefactor(), alpha_c_HER())
    @printf("k_GL=%.1e m/s  a_v_GL=%.2e m⁻¹  (a_v_GL·k_GL=%.0f s⁻¹)\n", k_GL(), a_v_GL, a_v_GL*k_GL())
    @printf("Mesh: N_CL=%d N_GDL=%d  DOFs=%d\n", N_CL, N_GDL, n_dof_stage34(N_CL,N_GDL))
    println("="^70)

    cl_mesh = make_cl_mesh(N_CL); gdl_mesh = make_gdl_mesh(N_GDL)
    eq = solve_buffer_equilibrium(1000.0, 1000.0)
    c_eq = [eq.H, eq.OH, eq.HCO3, eq.CO3, eq.CO2]
    isempty(list_cache(STAGE2_CACHE, "s2_")) && error("No Stage 2 cache — run run_stage2.jl first.")

    ND = n_dof_stage34(N_CL, N_GDL)
    nlog = 5 * N_CL
    Vs = sort(collect(V_RHE_grid), by = v -> -v)         # mild → cathodic

    # ── §9.5 workspace: fixed structural pattern (§9.3.3), persistent LU (§9.5.2), weight blocks (§9.5.3)
    rows, cols = build_gas_pattern(N_CL, N_GDL)
    Jtemplate = pattern_matrix(rows, cols, ND)
    plu = AffineLU()
    blocks = dof_blocks(3, N_CL, N_GDL)
    V_ref = Ref(Vs[1])
    res_closure!(F, uu) = residual_stage3!(F, uu, V_ref[], cl_mesh, gdl_mesh, c_eq)
    assemble! = make_ad_assembler(res_closure!, ND; chunk=12)
    @printf("Sparsity pattern: %d stored nnz (%.2f%% dense)\n", nnz(Jtemplate), 100*nnz(Jtemplate)/ND^2)
    println("Solver: §9.5 stack — newton_affine! + AffineLU + ForwardDiff AD; §9.5.5 continuation, anchor at −0.300 V from equilibrium liquid + inlet gas\n")

    # anchor guess (§9.5.5 table): equilibrium liquid, inlet gas at p_total
    u = make_u0_stage34_anchor(N_CL, N_GDL, c_eq)

    results = Tuple{Float64, Vector{Float64}}[]
    nconv = Ref(0); maxF = Ref(0.0)
    t0 = time()
    sweep_log = continuation_sweep!(u, Vs, res_closure!, V_ref;
        blocks=blocks, plu=plu, jac_assembler=assemble!, J=Jtemplate, n_log_dofs=nlog,
        on_converged = (V, uu, info) -> begin
            save_solution(STAGE3_CACHE, "s3_", copy(uu), V)
            push!(results, (V, copy(uu))); nconv[] += 1
            maxF[] = max(maxF[], info.residual)
        end)
    t_sweep = time() - t0
    n_iter = sum(l.iterations for l in sweep_log)
    @printf("\nConverged: %d/%d voltages in %.1f s\n", nconv[], length(Vs), t_sweep)
    @printf("  anchor V_RHE=%.3f: %d iters, ||F||inf=%.2e | %d Newton iterations total, %.2f/step | max ||F||inf over sweep = %.2e (§9.5.6 level 1e-6)\n",
            Vs[1], sweep_log[1].iterations, sweep_log[1].residual, n_iter, n_iter/length(Vs), maxF[])
    sort!(results, by = r -> r[1])

    # ── §Stage-3 physicality checks ──
    if !isempty(results)
        println("\nGas/liquid at the electrolyte face vs current (CL exit cell):")
        for Vt in [-0.816, -1.205, -1.644, -2.079, -2.413]
            r = results[argmin(abs.([x[1] for x in results] .- Vt))]
            u = r[2]; c = unpack_log_conc(u, N_CL)
            wg1,_  = gas_state(u, 1, N_CL, N_GDL)
            wgE,_  = gas_state(u, N_GDL+N_CL, N_CL, N_GDL)
            Vsh = r[1] - V_RHE_to_SHE_offset
            @printf("  V_RHE=%.3f iCO=%.0f iH2=%.0f mA/cm² | ωCO2 inlet=%.4f→CLexit=%.4f  ωCO=%.4f ωH2=%.4f | cCO2(L0)=%.2f\n",
                    r[1], compute_i_CO(u,Vsh,cl_mesh)/10, compute_i_H2(u,Vsh,cl_mesh)/10,
                    wg1[1], wgE[1], wgE[2], wgE[3], c[5,1])
        end

        # gas mass balance: species produced in the CL must leave through the GDL inlet face
        # (u_G = 0, zero flux at the electrolyte face). sm_face_flux is rightward-positive ⇒ outflux = −J.
        println("\nGas mass balance (produced in CL vs outflux at the GDL inlet face; rel. error):")
        for Vt in [-1.205, -1.644, -2.079]
            r = results[argmin(abs.([x[1] for x in results] .- Vt))]
            u = r[2]; V = r[1]; c = unpack_log_conc(u, N_CL); Vsh = V - V_RHE_to_SHE_offset
            prodCO = 0.0; prodH2 = 0.0
            for ix in 1:N_CL
                rco = R_CO2R(c[5,ix],Vsh,0.0,c[1,ix],a_v_CL,lambda_MHC(),i0_CO2R(),n_e)
                rh  = R_HER(Vsh,0.0,c[1,ix],a_v_CL,n_e)
                prodCO += rco*M_gas[2]*cl_mesh.dx[ix]; prodH2 += rh*M_gas[3]*cl_mesh.dx[ix]
            end
            w1, p1 = gas_state(u, 1, N_CL, N_GDL)
            Jin = sm_face_flux(_W_INLET, p_total, w1, p1, bf_GDL, gdl_mesh.dx_face_left)
            @printf("  V_RHE=%.3f  CO: produced %.3e vs outflux %.3e kg/m²/s (rel %.2e) | H2: produced %.3e vs outflux %.3e (rel %.2e)\n",
                    V, prodCO, -Jin[2], abs(-Jin[2]-prodCO)/max(prodCO,1e-30),
                    prodH2, -Jin[3], abs(-Jin[3]-prodH2)/max(prodH2,1e-30))
        end

        # sweep-wide numbers cited in BUILD_DIALOGUE.md
        iCOs = [compute_i_CO(r[2], r[1]-V_RHE_to_SHE_offset, cl_mesh)/10 for r in results]
        iH2s = [compute_i_H2(r[2], r[1]-V_RHE_to_SHE_offset, cl_mesh)/10 for r in results]
        itot = iCOs .+ iH2s
        kpk = argmax(iCOs)
        wgE,_ = gas_state(results[kpk][2], N_GDL+N_CL, N_CL, N_GDL)
        @printf("\nSweep-wide: i_CO peak = %.0f mA/cm² at V_RHE = %.3f (ω_CO2 at CL exit there = %.4f, inlet %.4f); derivative sign changes: i_total %d, i_CO %d, i_H2 %d\n",
                iCOs[kpk], results[kpk][1], wgE[1], omega_inlet[1],
                deriv_sign_changes(itot), deriv_sign_changes(iCOs), deriv_sign_changes(iH2s))
    end

    # ── Plots ──
    pdir = joinpath(HERE, "output_stage3"); mkpath(pdir)
    plot_A1(results, cl_mesh, joinpath(pdir, "stage3_A1_polarization.png"); stage_label="Stage 3")
    plot_A2(results, cl_mesh, joinpath(pdir, "stage3_A2_liquid.png"); stage_label="Stage 3")
    plot_B1(results, cl_mesh, gdl_mesh, joinpath(pdir, "stage3_B1_gas.png"); stage_label="Stage 3")
    s2 = [(V, load_solution(STAGE2_CACHE,"s2_",V)) for V in list_cache(STAGE2_CACHE,"s2_")]
    plot_polarization_overlay([s2, results], ["S2","S3"], cl_mesh,
        joinpath(pdir, "stage3_overlay_vs_stage2.png"); title="Stage 2 vs Stage 3 (gas transport on)")
    println("\nPlots: stage3_A1_polarization.png, stage3_A2_liquid.png, stage3_B1_gas.png, stage3_overlay_vs_stage2.png")

    println("\n" * "="^70)
    println("STAGE 3 COMPLETE — STOP GATE. Review plots before Stage 4.")
    println("="^70)
    return nconv[], length(Vs)
end

main()
