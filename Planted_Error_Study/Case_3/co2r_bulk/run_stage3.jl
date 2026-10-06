# run_stage3.jl — Stage 3: Stefan-Maxwell gas transport + R_PT phase transfer (§6, §7.3-7.4, §11).
# Adds 4×(N_GDL+N_CL) gas DOFs. u_G=0 (Darcy finalized in Stage 4); pressure pinned to p_total.
# Sweep: self-anchoring §9.5.5 continuation — anchor at −0.300 V_RHE from equilibrium liquid +
# inlet gas (ω_inlet, p_total in every gas cell); the Stage 2 cache is only a driver-order guard.
include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "plots.jl"))
using Printf

const STAGE2_CACHE = joinpath(HERE, "output_stage2", "cache")
const STAGE3_CACHE = joinpath(HERE, "output_stage3", "cache")
const STAGE3_PLOTS = joinpath(HERE, "output_stage3")

function main()
    println("="^70)
    println("=== Stage 3: Stefan-Maxwell gas transport + R_PT phase transfer ===")
    @printf("i0_CO2R = %.0f A/m², λ_MHC = %.2f eV, k_GL = %.1e m/s\n", i0_CO2R(), lambda_MHC(), k_GL())
    @printf("Mesh: N_CL=%d N_GDL=%d, DOFs = %d\n", N_CL, N_GDL, 5*N_CL + 4*(N_GDL+N_CL))
    @printf("D_bin CO2-H2O = %.3e m²/s, ω_inlet = %s\n", D_bin[1,4],
            join([@sprintf("%.3f",w) for w in omega_inlet], ","))
    println("="^70)

    isempty(list_cache(STAGE2_CACHE, "s2")) && error("No Stage 2 cache — run run_stage2.jl first")

    cl_mesh = make_cl_mesh(); gdl_mesh = make_gdl_mesh()
    gas_mesh = GasMesh(gdl_mesh, cl_mesh)
    c_eq_vec = equilibrium_c_vec(1000.0, 1000.0)
    N = 5*N_CL + 4*(N_GDL+N_CL)

    residual_at_V!(F, u, V) = residual_stage3!(F, u, V, cl_mesh, gdl_mesh, gas_mesh, c_eq_vec)
    # Anchor guess (§9.5.5 table): equilibrium liquid + inlet gas composition and p_total in every gas cell
    ln_c_eq = log.(c_eq_vec)
    u_liq = zeros(5*N_CL)
    for ix in 1:N_CL
        u_liq[liquid_idx(ix,1):liquid_idx(ix,5)] .= ln_c_eq
    end
    u0 = make_u0_gas(u_liq, N_CL, N_GDL)
    current_fn(u, V) = (compute_i_CO(u, V - V_RHE_to_SHE_offset, cl_mesh)/10,
                        compute_i_H2(u, V - V_RHE_to_SHE_offset, cl_mesh)/10)

    Jpat = probe_sparsity((F, u) -> residual_at_V!(F, u, -1.0), N; u_seed=u0)
    @printf("Sparsity: %d nonzeros (%.3f%% dense)\n", nnz(Jpat), 100*nnz(Jpat)/N^2)
    plu = AffineLU()
    blocks = dof_blocks(3)            # liquid | gas GDL | gas CL — the pressure DOFs (~1e5 Pa) get their own weight block

    save_fn(u, V) = save_solution(STAGE3_CACHE, "s3", u, V)
    results, n_conv, n_tot, maxres, _ = run_sweep!(residual_at_V!, u0, V_RHE_grid;
        Jpat=Jpat, plu=plu, blocks=blocks, n_log_dofs=5*N_CL,
        save_fn=save_fn, current_fn=current_fn, label="Stage3")

    # ── Plots ──
    Vr = [r[1] for r in results]; iCO = [r[2] for r in results]; iH2 = [r[3] for r in results]
    plot_A1(STAGE3_PLOTS, Vr, iCO, iH2; title_prefix="Stage 3")

    Vcache = sort(list_cache(STAGE3_CACHE, "s3"), by = v -> -v)
    profiles = Tuple{Float64,Matrix{Float64},Vector{Float64},Vector{Float64}}[]
    gas_series = Tuple{Float64,Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64}}[]
    for V in Vcache
        uu = load_solution(STAGE3_CACHE, "s3", V); c = unpack_log_conc(uu, N_CL)
        Vapp = V - V_RHE_to_SHE_offset
        rco2r = [R_CO2R(c[5,ix], Vapp, 0.0, c[1,ix], a_v_CL, lambda_MHC(), i0_CO2R(), n_e) for ix in 1:N_CL]
        rher  = [R_HER(Vapp, 0.0, c[1,ix], a_v_CL, n_e) for ix in 1:N_CL]
        push!(profiles, (V, c, rco2r, rher))
        pCO2,pCO,pH2,ptot = gas_profiles(uu, N_CL, N_GDL)
        push!(gas_series, (V, pCO2, pCO, pH2, ptot))
    end
    plot_A2(STAGE3_PLOTS, cl_mesh, profiles; title_prefix="Stage 3")
    x_gas_μm = vcat(gdl_mesh.xc, cl_mesh.xc) .* 1e6
    plot_B1(STAGE3_PLOTS, x_gas_μm, gas_series; title_prefix="Stage 3")

    # overlay vs Stage 1 & 2
    function stage_iV(cache, prefix)
        Vs = sort(list_cache(cache, prefix)); ico=Float64[]; ih2=Float64[]
        for V in Vs
            u = load_solution(cache, prefix, V); Va = V - V_RHE_to_SHE_offset
            push!(ico, compute_i_CO(u, Va, cl_mesh)/10); push!(ih2, compute_i_H2(u, Va, cl_mesh)/10)
        end
        (Vs, ico, ih2)
    end
    s1 = stage_iV(joinpath(HERE,"output_stage1","cache"), "s1")
    s2 = stage_iV(STAGE2_CACHE, "s2")
    plot_overlay(STAGE3_PLOTS, [("S1",s1...),("S2",s2...),("S3",Vr,iCO,iH2)];
                 fname="overlay_s1_s2_s3.png", title="Stage 3 vs Stages 1-2")

    # ── Physicality checks (§11 Stage 3) ──
    println("\n── Physicality checks ──")
    Vhi = Vcache[argmin(abs.(Vcache .- (-2.0)))]
    uu = load_solution(STAGE3_CACHE, "s3", Vhi)
    ω, p = unpack_gas(uu, N_CL, N_GDL)
    @printf("  At V_RHE=%.3f:\n", Vhi)
    @printf("    ω_CO2: GDL inlet=%.3f → CL exit=%.3f (expect decrease from ~0.947)\n", ω[1,1], ω[end,1])
    @printf("    ω_CO : GDL inlet=%.4f → CL exit=%.4f (expect increase)\n", ω[1,2], ω[end,2])
    @printf("    ω_H2 : GDL inlet=%.4f → CL exit=%.4f (expect increase)\n", ω[1,3], ω[end,3])
    pCO2,_,_,ptot = gas_profiles(uu, N_CL, N_GDL)
    @printf("    p_CO2: inlet=%.3f atm → CL exit=%.3f atm; p_total=%.4f atm (pinned)\n",
            pCO2[1], pCO2[end], ptot[end])
    println("\n>>> STAGE 3 COMPLETE. Plots in $STAGE3_PLOTS. STOP for human review. <<<")
    return n_conv, n_tot
end

main()
