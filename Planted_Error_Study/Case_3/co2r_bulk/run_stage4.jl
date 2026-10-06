# run_stage4.jl — Stage 4: finalize the Darcy pressure equation (§11). Darcy convection on;
# pressure DOF solved via continuity with GDL inlet Dirichlet (ghost) + CL outlet dp/dx=0.
# Sweep: self-anchoring §9.5.5 continuation — anchor at −0.300 V_RHE from equilibrium liquid +
# inlet gas at uniform p_total; the Stage 3 cache is only a driver-order guard.
include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "plots.jl"))
using Printf

const STAGE3_CACHE = joinpath(HERE, "output_stage3", "cache")
const STAGE4_CACHE = joinpath(HERE, "output_stage4", "cache")
const STAGE4_PLOTS = joinpath(HERE, "output_stage4")

function main()
    println("="^70)
    println("=== Stage 4: Darcy pressure finalized (convection on) ===")
    @printf("i0_CO2R = %.0f A/m², λ_MHC = %.2f eV, k_GL = %.1e m/s\n", i0_CO2R(), lambda_MHC(), k_GL())
    @printf("Mesh: N_CL=%d N_GDL=%d, DOFs = %d\n", N_CL, N_GDL, 5*N_CL + 4*(N_GDL+N_CL))
    @printf("κ_GDL=%.2e m², κ_CL=%.2e m², μ_gas=%.2e Pa·s\n", kappa_GDL, kappa_CL, mu_gas)
    println("="^70)

    isempty(list_cache(STAGE3_CACHE, "s3")) && error("No Stage 3 cache — run run_stage3.jl first")

    cl_mesh = make_cl_mesh(); gdl_mesh = make_gdl_mesh()
    gas_mesh = GasMesh(gdl_mesh, cl_mesh)
    c_eq_vec = equilibrium_c_vec(1000.0, 1000.0)
    N = 5*N_CL + 4*(N_GDL+N_CL)

    residual_at_V!(F, u, V) = residual_stage4!(F, u, V, cl_mesh, gdl_mesh, gas_mesh, c_eq_vec)
    # Anchor guess (§9.5.5 table): equilibrium liquid + inlet gas at uniform p_total in every gas cell
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
    blocks = dof_blocks(4)            # liquid | gas GDL | gas CL — the Darcy pressure DOFs (~1e5 Pa) get their own weight block (§9.5.3)

    save_fn(u, V) = save_solution(STAGE4_CACHE, "s4", u, V)
    results, n_conv, n_tot, maxres, _ = run_sweep!(residual_at_V!, u0, V_RHE_grid;
        Jpat=Jpat, plu=plu, blocks=blocks, n_log_dofs=5*N_CL,
        save_fn=save_fn, current_fn=current_fn, label="Stage4")

    # ── Plots ──
    Vr = [r[1] for r in results]; iCO = [r[2] for r in results]; iH2 = [r[3] for r in results]
    plot_A1(STAGE4_PLOTS, Vr, iCO, iH2; title_prefix="Stage 4")

    Vcache = sort(list_cache(STAGE4_CACHE, "s4"), by = v -> -v)
    profiles = Tuple{Float64,Matrix{Float64},Vector{Float64},Vector{Float64}}[]
    gas_series = Tuple{Float64,Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64}}[]
    darcy_series = Tuple{Float64,Vector{Float64},Vector{Float64}}[]
    x_gas_μm = vcat(gdl_mesh.xc, cl_mesh.xc) .* 1e6
    for V in Vcache
        uu = load_solution(STAGE4_CACHE, "s4", V); c = unpack_log_conc(uu, N_CL)
        Vapp = V - V_RHE_to_SHE_offset
        rco2r = [R_CO2R(c[5,ix], Vapp, 0.0, c[1,ix], a_v_CL, lambda_MHC(), i0_CO2R(), n_e) for ix in 1:N_CL]
        rher  = [R_HER(Vapp, 0.0, c[1,ix], a_v_CL, n_e) for ix in 1:N_CL]
        push!(profiles, (V, c, rco2r, rher))
        pCO2,pCO,pH2,ptot = gas_profiles(uu, N_CL, N_GDL)
        push!(gas_series, (V, pCO2, pCO, pH2, ptot))
        dp, uG, _, _, _ = darcy_diagnostics(uu, cl_mesh, gdl_mesh, gas_mesh, V)
        push!(darcy_series, (V, dp, uG))
    end
    plot_A2(STAGE4_PLOTS, cl_mesh, profiles; title_prefix="Stage 4")
    plot_B1(STAGE4_PLOTS, x_gas_μm, gas_series; title_prefix="Stage 4")
    plot_darcy(STAGE4_PLOTS, x_gas_μm, darcy_series; title_prefix="Stage 4")

    # overlay vs Stage 3
    s3V = sort(list_cache(STAGE3_CACHE, "s3")); iCO3=Float64[]; iH23=Float64[]
    for V in s3V
        u3 = load_solution(STAGE3_CACHE, "s3", V); Va = V - V_RHE_to_SHE_offset
        push!(iCO3, compute_i_CO(u3, Va, cl_mesh)/10); push!(iH23, compute_i_H2(u3, Va, cl_mesh)/10)
    end
    plot_overlay(STAGE4_PLOTS, [("S3",s3V,iCO3,iH23),("S4",Vr,iCO,iH2)];
                 fname="overlay_s3_s4.png", title="Stage 4 vs Stage 3 (Darcy is a tiny correction)")

    # ── Physicality checks (§11 Stage 4) ──
    println("\n── Physicality checks ──")
    for Vt in (-1.6, -2.0)
        Vhi = Vcache[argmin(abs.(Vcache .- Vt))]
        uu = load_solution(STAGE4_CACHE, "s4", Vhi)
        dp, uG, ρu_in, Smass_tot, mb = darcy_diagnostics(uu, cl_mesh, gdl_mesh, gas_mesh, Vhi)
        itot = (compute_i_CO(uu, Vhi - V_RHE_to_SHE_offset, cl_mesh) +
                compute_i_H2(uu, Vhi - V_RHE_to_SHE_offset, cl_mesh))/10
        @printf("  V_RHE=%.3f (i_tot≈%.0f mA/cm²): Δp_CL=%.3g Pa, |u_G|max=%.2e m/s\n",
                Vhi, itot, maximum(abs.(dp))-minimum(abs.(dp)), maximum(abs.(uG)))
        @printf("      mass balance: ρu_in=%.3e vs -ΣS·dx=%.3e (rel err %.2e)\n",
                ρu_in, -Smass_tot, mb)
    end
    # Stage 4 vs Stage 3 current difference (expect tiny)
    dmax = maximum(abs.(iCO .- iCO3[[argmin(abs.(s3V .- v)) for v in Vr]]))
    @printf("  max |i_CO(S4) - i_CO(S3)| = %.3g mA/cm² (expect small — Darcy is a tiny correction)\n", dmax)

    println("\n>>> STAGE 4 COMPLETE. Plots in $STAGE4_PLOTS. STOP for human review. <<<")
    return n_conv, n_tot
end

main()
