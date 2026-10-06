# run_stage2.jl — Stage 2: Sherwood-Reynolds Neumann BC at electrolyte side (§7.2, §11).
# Same DOFs/layout/solver as Stage 1; only the electrolyte-side flux changes.
# Sweep: self-anchoring §9.5.5 continuation (anchor at −0.300 V_RHE from the equilibrium guess);
# the Stage 1 cache is only checked as a driver-order guard.
include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "plots.jl"))
using Printf

const STAGE1_CACHE = joinpath(HERE, "output_stage1", "cache")
const STAGE2_CACHE = joinpath(HERE, "output_stage2", "cache")
const STAGE2_PLOTS = joinpath(HERE, "output_stage2")

function main()
    println("="^70)
    println("=== Stage 2: Sherwood-Reynolds Neumann BC at electrolyte side ===")
    @printf("i0_CO2R = %.0f A/m², λ_MHC = %.2f eV\n", i0_CO2R(), lambda_MHC())
    @printf("HER_pf = %.3e, α_c = %.4f\n", HER_prefactor(), alpha_c_HER())
    @printf("Mesh: N_CL = %d (DOFs = %d)\n", N_CL, 5*N_CL)
    @printf("k_MT [m/s] = %s\n", join([@sprintf("%.3e", k) for k in k_MT], ", "))
    println("="^70)

    isempty(list_cache(STAGE1_CACHE, "s1")) && error("No Stage 1 cache — run run_stage1.jl first")

    cl_mesh = make_cl_mesh()
    c_eq_vec = equilibrium_c_vec(1000.0, 1000.0)
    N = 5 * N_CL

    # Biot number check (§11 Stage 2): Bi = k_MT·L_CL/D_eff, expect O(0.1-10)
    Bi = [k_MT[i]*L_CL/D_eff[i] for i in 1:5]
    @printf("Biot numbers Bi_i = %s\n", join([@sprintf("%.2f", b) for b in Bi], ", "))

    residual_at_V!(F, u, V) = residual_stage2!(F, u, V, cl_mesh, c_eq_vec)
    # Anchor guess (§9.5.5 table): every cell at bulk equilibrium log-concentration
    ln_c_eq = log.(c_eq_vec)
    u0 = zeros(N)
    for ix in 1:N_CL
        u0[liquid_idx(ix,1):liquid_idx(ix,5)] .= ln_c_eq
    end
    current_fn(u, V) = (compute_i_CO(u, V - V_RHE_to_SHE_offset, cl_mesh)/10,
                        compute_i_H2(u, V - V_RHE_to_SHE_offset, cl_mesh)/10)

    Jpat = probe_sparsity((F, u) -> residual_at_V!(F, u, -1.0), N; u_seed=u0)
    @printf("Sparsity: %d nonzeros (%.2f%% dense)\n", nnz(Jpat), 100*nnz(Jpat)/N^2)
    plu = AffineLU()
    blocks = dof_blocks(2)

    save_fn(u, V) = save_solution(STAGE2_CACHE, "s2", u, V)
    results, n_conv, n_tot, maxres, _ = run_sweep!(residual_at_V!, u0, V_RHE_grid;
        Jpat=Jpat, plu=plu, blocks=blocks, n_log_dofs=N,
        save_fn=save_fn, current_fn=current_fn, label="Stage2")

    # ── Plots ──
    Vr  = [r[1] for r in results]; iCO = [r[2] for r in results]; iH2 = [r[3] for r in results]
    plot_A1(STAGE2_PLOTS, Vr, iCO, iH2; title_prefix="Stage 2")

    # A2 liquid profiles
    Vcache = sort(list_cache(STAGE2_CACHE, "s2"), by = v -> -v)
    profiles = Tuple{Float64,Matrix{Float64},Vector{Float64},Vector{Float64}}[]
    for V in Vcache
        uu = load_solution(STAGE2_CACHE, "s2", V); c = unpack_log_conc(uu, N_CL)
        Vapp = V - V_RHE_to_SHE_offset
        rco2r = [R_CO2R(c[5,ix], Vapp, 0.0, c[1,ix], a_v_CL, lambda_MHC(), i0_CO2R(), n_e) for ix in 1:N_CL]
        rher  = [R_HER(Vapp, 0.0, c[1,ix], a_v_CL, n_e) for ix in 1:N_CL]
        push!(profiles, (V, c, rco2r, rher))
    end
    plot_A2(STAGE2_PLOTS, cl_mesh, profiles; title_prefix="Stage 2")

    # Overlay vs Stage 1 (stop-gate requirement)
    s1V = sort(list_cache(STAGE1_CACHE, "s1"))
    iCO1 = Float64[]; iH21 = Float64[]
    for V in s1V
        u1 = load_solution(STAGE1_CACHE, "s1", V); Vapp = V - V_RHE_to_SHE_offset
        push!(iCO1, compute_i_CO(u1, Vapp, cl_mesh)/10); push!(iH21, compute_i_H2(u1, Vapp, cl_mesh)/10)
    end
    plot_overlay(STAGE2_PLOTS, [("S1", s1V, iCO1, iH21), ("S2", Vr, iCO, iH2)];
                 fname="overlay_stage1_stage2.png", title="Stage 2 vs Stage 1 polarization")

    # ── Physicality checks (§11 Stage 2) ──
    println("\n── Physicality checks ──")
    @printf("  Biot numbers O(0.1-10): %s → %s\n",
            join([@sprintf("%.2f",b) for b in Bi],", "), all(0.01 .< Bi .< 100) ? "ok range" : "check")
    Vhi = Vcache[argmin(abs.(Vcache .- (-2.0)))]
    uu = load_solution(STAGE2_CACHE, "s2", Vhi); c = unpack_log_conc(uu, N_CL)
    @printf("  At V_RHE=%.3f: c_CO2(L_CL)=%.3f mol/m³ (expect →0 depletion at high i)\n", Vhi, c[5,end])
    @printf("               c_OH(L_CL)=%.3e mol/m³ vs c_eq_OH=%.3e (expect above eq)\n", c[2,end], c_eq_vec[2])
    # flux at electrolyte face = k_MT*(c_local - c_eq)
    for (nm,i) in (("CO2",5),("OH",2))
        Jface = -k_MT[i]*(c_eq_vec[i]-c[i,end])
        @printf("  J_%s(L_CL)=%.3e mol/m²/s (%s)\n", nm, Jface, Jface<0 ? "into CL" : "out of CL")
    end
    println("\n>>> STAGE 2 COMPLETE. Plots in $STAGE2_PLOTS. STOP for human review. <<<")
    return n_conv, n_tot
end

main()
