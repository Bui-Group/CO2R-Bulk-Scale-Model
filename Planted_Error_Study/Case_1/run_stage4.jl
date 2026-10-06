# run_stage4.jl — Stage 4 driver (§Stage 4)
# Finalizes the Darcy pressure equation (mass continuity) + activates gas convection.
# Kinetics still simplified (φ_s=V_applied, φ_L=0). Same 1200 DOFs as Stage 3.
#
# Solver: the §9.5 stack with the unified continuation driver (§9.5.5) — anchor at
# −0.300 V_RHE from equilibrium liquid + inlet gas at uniform p_total, then the
# tangent predictor down the grid (the pressure profile develops along the chain).
# The Stage 3 cache is only a driver-order guard and the comparison reference.

const HERE = @__DIR__
for f in ["params.jl","mesh.jl","chemistry.jl","kinetics.jl","transport.jl",
          "gas_transport.jl","solver.jl","cache.jl",
          "assembly.jl","assembly_stage3.jl","assembly_stage4.jl","plots.jl"]
    include(joinpath(HERE, f))
end

const STAGE3_CACHE = joinpath(HERE, "output_stage3", "cache")
const STAGE4_CACHE = joinpath(HERE, "output_stage4", "cache")
const N_CL  = parse(Int, get(ENV, "MESH_N_CL", "80"))
const N_GDL = parse(Int, get(ENV, "MESH_N_GDL", "120"))

function main()
    println("="^70)
    println("=== Stage 4: Darcy pressure finalized + gas convection ===")
    @printf("κ_GDL=%.2e  κ_CL=%.2e  κ_iface=%.2e m²  μ_gas=%.2e Pa·s\n",
            kappa_GDL, kappa_CL, kappa_interface, mu_gas)
    @printf("Mesh: N_CL=%d N_GDL=%d  DOFs=%d\n", N_CL, N_GDL, n_dof_stage34(N_CL,N_GDL))
    println("="^70)

    cl_mesh = make_cl_mesh(N_CL); gdl_mesh = make_gdl_mesh(N_GDL)
    eq = solve_buffer_equilibrium(1000.0, 1000.0)
    c_eq = [eq.H, eq.OH, eq.HCO3, eq.CO3, eq.CO2]
    isempty(list_cache(STAGE3_CACHE, "s3_")) && error("No Stage 3 cache — run run_stage3.jl first.")

    ND = n_dof_stage34(N_CL, N_GDL)
    nlog = 5 * N_CL
    Vs = sort(collect(V_RHE_grid), by = v -> -v)         # mild → cathodic

    # ── §9.5 workspace (same layout/pattern as Stage 3) ──
    rows, cols = build_gas_pattern(N_CL, N_GDL)
    Jtemplate = pattern_matrix(rows, cols, ND)
    plu = AffineLU()
    blocks = dof_blocks(4, N_CL, N_GDL)
    V_ref = Ref(Vs[1])
    res_closure!(F, uu) = residual_stage4!(F, uu, V_ref[], cl_mesh, gdl_mesh, c_eq)
    assemble! = make_ad_assembler(res_closure!, ND; chunk=12)
    println("Solver: §9.5 stack — newton_affine! + AffineLU + ForwardDiff AD; §9.5.5 continuation, anchor at −0.300 V from equilibrium liquid + inlet gas (uniform p_total)\n")

    u = make_u0_stage34_anchor(N_CL, N_GDL, c_eq)          # §9.5.5 table anchor guess
    results = Tuple{Float64, Vector{Float64}}[]
    nconv = Ref(0); maxF = Ref(0.0)
    t0 = time()
    sweep_log = continuation_sweep!(u, Vs, res_closure!, V_ref;
        blocks=blocks, plu=plu, jac_assembler=assemble!, J=Jtemplate, n_log_dofs=nlog,
        on_converged = (V, uu, info) -> begin
            save_solution(STAGE4_CACHE, "s4_", copy(uu), V)
            push!(results, (V, copy(uu))); nconv[] += 1
            maxF[] = max(maxF[], info.residual)
        end)
    t_sweep = time() - t0
    n_iter = sum(l.iterations for l in sweep_log)
    @printf("\nConverged: %d/%d voltages in %.1f s\n", nconv[], length(Vs), t_sweep)
    @printf("  anchor V_RHE=%.3f: %d iters, ||F||inf=%.2e | %d Newton iterations total, %.2f/step | max ||F||inf over sweep = %.2e (§9.5.6 level 1e-6)\n",
            Vs[1], sweep_log[1].iterations, sweep_log[1].residual, n_iter, n_iter/length(Vs), maxF[])
    sort!(results, by = r -> r[1])

    # ── §Stage-4 physicality checks ──
    if !isempty(results)
        println("\nDarcy diagnostics (pressure drop, mass balance) and current vs Stage 3:")
        for Vt in [-0.816, -1.205, -1.644, -2.079, -2.413]
            r = results[argmin(abs.([x[1] for x in results] .- Vt))]
            u = r[2]; V = r[1]
            # pressure range over CL gas cells
            pcl = [gas_state(u, NG_idx, N_CL, N_GDL)[2] for NG_idx in (N_GDL+1):(N_GDL+N_CL)]
            dP = maximum(pcl) - minimum(pcl)
            # net mass source vs Darcy outflux at GDL inlet (mass balance)
            c = unpack_log_conc(u, N_CL); Vsh = V - V_RHE_to_SHE_offset
            Smass_int = 0.0
            for ix in 1:N_CL
                rco = R_CO2R(c[5,ix],Vsh,0.0,c[1,ix],a_v_CL,lambda_MHC(),i0_CO2R(),n_e)
                rh  = R_HER(Vsh,0.0,c[1,ix],a_v_CL,n_e)
                H_loc = sechenov_H(@view c[:,ix]); rpt = a_v_GL*k_GL()*(H_loc*pCO2_atm(u,N_GDL+ix,N_CL,N_GDL)-c[5,ix])
                Smass_int += (-rpt*M_gas[1] + rco*M_gas[2] + rh*M_gas[3]) * cl_mesh.dx[ix]
            end
            w1,p1 = gas_state(u,1,N_CL,N_GDL)
            _, ru1 = sm_darcy_face_flux(_W_INLET, p_total, w1, p1, bf_GDL, kappa_GDL, gdl_mesh.dx_face_left)
            i_co = compute_i_CO(u,Vsh,cl_mesh)/10; i_h2 = compute_i_H2(u,Vsh,cl_mesh)/10
            s3 = load_solution(STAGE3_CACHE,"s3_",V)
            i_co3 = s3 === nothing ? NaN : compute_i_CO(s3,Vsh,cl_mesh)/10
            @printf("  V_RHE=%.3f ΔP_CL=%.2e Pa | massbal: ρu@inlet=%.3e net_src=%.3e relerr=%.1e | iCO=%.0f (S3 %.0f) iH2=%.0f\n",
                    V, dP, ru1, -Smass_int, abs(ru1+Smass_int)/max(abs(Smass_int),1e-30), i_co, i_co3, i_h2)
        end
        # sweep-wide numbers cited in BUILD_DIALOGUE.md
        dPs = [(pcl = [gas_state(r[2], g, N_CL, N_GDL)[2] for g in (N_GDL+1):(N_GDL+N_CL)]; maximum(pcl) - minimum(pcl)) for r in results]
        pall = [vcat([gas_state(r[2], g, N_CL, N_GDL)[2] for g in 1:(N_GDL+N_CL)]) for r in results]
        dPall = [maximum(p) - minimum(p) for p in pall]
        maxdCO = 0.0; maxdH2 = 0.0
        for r in results
            s3 = load_solution(STAGE3_CACHE, "s3_", r[1]); s3 === nothing && continue
            Vsh = r[1] - V_RHE_to_SHE_offset
            a = compute_i_CO(r[2],Vsh,cl_mesh); b = compute_i_CO(s3,Vsh,cl_mesh)
            maxdCO = max(maxdCO, 100*abs(a-b)/max(abs(b),1e-12))
            a = compute_i_H2(r[2],Vsh,cl_mesh); b = compute_i_H2(s3,Vsh,cl_mesh)
            maxdH2 = max(maxdH2, 100*abs(a-b)/max(abs(b),1e-12))
        end
        @printf("\nSweep-wide (mild → cathodic): ΔP_CL %.2f → %.2f Pa; ΔP over GDL+CL %.2f → %.2f Pa; max |Δi_CO| vs Stage 3 = %.3f%%, max |Δi_H2| = %.3f%%\n",
                dPs[end], dPs[1], dPall[end], dPall[1], maxdCO, maxdH2)
    end

    pdir = joinpath(HERE, "output_stage4"); mkpath(pdir)
    plot_A1(results, cl_mesh, joinpath(pdir, "stage4_A1_polarization.png"); stage_label="Stage 4")
    plot_A2(results, cl_mesh, joinpath(pdir, "stage4_A2_liquid.png"); stage_label="Stage 4")
    plot_B1(results, cl_mesh, gdl_mesh, joinpath(pdir, "stage4_B1_gas.png"); stage_label="Stage 4")
    s3 = [(V, load_solution(STAGE3_CACHE,"s3_",V)) for V in list_cache(STAGE3_CACHE,"s3_")]
    plot_polarization_overlay([s3, results], ["S3","S4"], cl_mesh,
        joinpath(pdir, "stage4_overlay_vs_stage3.png"); title="Stage 3 vs Stage 4 (Darcy on)")
    println("\nPlots: stage4_A1_polarization.png, stage4_A2_liquid.png, stage4_B1_gas.png, stage4_overlay_vs_stage3.png")

    println("\n" * "="^70)
    println("STAGE 4 COMPLETE — STOP GATE. Review plots before the Fit step.")
    println("="^70)
    return nconv[], length(Vs)
end

main()
