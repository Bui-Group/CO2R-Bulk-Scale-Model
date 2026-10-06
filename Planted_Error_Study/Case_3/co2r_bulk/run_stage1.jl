# run_stage1.jl — Stage 1: liquid NP diffusion only + Dirichlet BCs + simplified kinetics.
# §11 Stage 1. Solver: §9.5 newton_affine! + persistent sparse LU + AD Jacobian (fixed pattern);
# driver: §9.5.5 continuation — anchor at −0.300 V_RHE from the equilibrium guess, tangent
# predictor down the 89-V grid, §9.5.6 post-solve assertion at every voltage.
include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "plots.jl"))
using Printf

const STAGE1_CACHE = joinpath(HERE, "output_stage1", "cache")
const STAGE1_PLOTS = joinpath(HERE, "output_stage1")

function main()
    println("="^70)
    println("=== Stage 1: Liquid NP diffusion only + Dirichlet BCs ===")
    @printf("i0_CO2R = %.0f A/m², λ_MHC = %.2f eV\n", i0_CO2R(), lambda_MHC())
    @printf("HER_pf = %.3e, α_c = %.4f\n", HER_prefactor(), alpha_c_HER())
    @printf("Mesh: N_CL = %d (DOFs = %d)\n", N_CL, 5*N_CL)
    @printf("Cache dir: %s\n", STAGE1_CACHE)
    println("="^70)

    cl_mesh = make_cl_mesh()
    c_eq_vec = equilibrium_c_vec(1000.0, 1000.0)

    # Equilibrium sanity (§9.6, §12.1)
    S_eq = buffer_sources(c_eq_vec)
    @printf("Equilibrium check: pH=%.3f, c_CO2=%.3f, max|S_buf|=%.2e\n",
            -log10(c_eq_vec[1]/1000), c_eq_vec[5], maximum(abs.(S_eq)))
    @assert maximum(abs.(S_eq)) < 1e-6 "Equilibrium buffer sources not near zero"

    N = 5 * N_CL
    # Anchor guess (§9.5.5 table): every cell at bulk equilibrium log-concentration
    ln_c_eq = log.(c_eq_vec)
    u0 = zeros(N)
    for ix in 1:N_CL
        u0[liquid_idx(ix,1):liquid_idx(ix,5)] .= ln_c_eq
    end

    # Residual at voltage V (closes over mesh + c_eq)
    residual_at_V!(F, u, V) = residual_stage1!(F, u, V, cl_mesh, c_eq_vec)

    # Probe the fixed sparsity pattern once (V-independent), §9.3.3 / §9.5.2
    Jpat = probe_sparsity((F, u) -> residual_at_V!(F, u, -1.0), N; u_seed=u0)
    @printf("Sparsity: %d nonzeros (%.2f%% dense)\n", nnz(Jpat), 100*nnz(Jpat)/N^2)
    plu = AffineLU()
    blocks = dof_blocks(1)

    # §9.5.5 continuation: anchor at −0.300 V_RHE from the equilibrium guess, tangent predictor down the grid
    Vs = sort(collect(V_RHE_grid), by = v -> -v)   # mild → cathodic
    V_ref = Ref(Vs[1])
    res_closure!(F, u) = residual_at_V!(F, u, V_ref[])
    assemble! = make_ad_assembler(res_closure!, N; chunk=12)
    u = copy(u0)
    n_conv = Ref(0); max_res = Ref(0.0)
    results = Tuple{Float64,Float64,Float64}[]   # (V_RHE, iCO_mAcm2, iH2_mAcm2)
    t0 = time()
    sweep_log = continuation_sweep!(u, Vs, res_closure!, V_ref;
        blocks=blocks, plu=plu, jac_assembler=assemble!, J=Jpat, n_log_dofs=N,
        progress_every=typemax(Int),
        on_converged = (V, uu, info) -> begin
            save_solution(STAGE1_CACHE, "s1", uu, V)          # cache FIRST (§10.4)
            n_conv[] += 1; max_res[] = max(max_res[], info.residual)
            Vapp = V - V_RHE_to_SHE_offset
            iCO = compute_i_CO(uu, Vapp, cl_mesh) / 10.0
            iH2 = compute_i_H2(uu, Vapp, cl_mesh) / 10.0
            push!(results, (V, iCO, iH2))
            @printf("V_RHE=%.3f  ‖F‖∞=%.2e iter=%2d  iCO=%.2f iH2=%.2f mA/cm²\n",
                    V, info.residual, info.iterations, iCO, iH2)
        end)
    n_iter = sum(l.iterations for l in sweep_log)
    @printf("\nStage 1 sweep: %d/%d converged in %.1fs (anchor %d iters, %d Newton iterations, %.2f/step), max‖F‖∞=%.2e\n",
            n_conv[], length(Vs), time()-t0, sweep_log[1].iterations, n_iter, n_iter/length(Vs), max_res[])

    # ── Plots ──
    sort!(results, by = r -> r[1])
    Vr  = [r[1] for r in results]
    iCO = [r[2] for r in results]
    iH2 = [r[3] for r in results]
    plot_A1(STAGE1_PLOTS, Vr, iCO, iH2; title_prefix="Stage 1")

    # A2 profiles: load every converged solution, color ramp mild→cathodic
    Vcache = sort(list_cache(STAGE1_CACHE, "s1"), by = v -> -v)
    profiles = Tuple{Float64,Matrix{Float64},Vector{Float64},Vector{Float64}}[]
    for V in Vcache
        uu = load_solution(STAGE1_CACHE, "s1", V)
        c  = unpack_log_conc(uu, N_CL)
        Vapp = V - V_RHE_to_SHE_offset
        rco2r = [R_CO2R(c[5,ix], Vapp, 0.0, c[1,ix], a_v_CL, lambda_MHC(), i0_CO2R(), n_e) for ix in 1:N_CL]
        rher  = [R_HER(Vapp, 0.0, c[1,ix], a_v_CL, n_e) for ix in 1:N_CL]
        push!(profiles, (V, c, rco2r, rher))
    end
    plot_A2(STAGE1_PLOTS, cl_mesh, profiles; title_prefix="Stage 1")

    # ── Physicality checks (§11 Stage 1) ──
    println("\n── Physicality checks ──")
    # pick a mid-high voltage solution
    Vmid = Vcache[argmin(abs.(Vcache .- (-1.7)))]
    uu = load_solution(STAGE1_CACHE, "s1", Vmid)
    c = unpack_log_conc(uu, N_CL)
    pH = [-log10(max(c[1,ix]/1000,1e-20)) for ix in 1:N_CL]
    @printf("At V_RHE=%.3f: pH range [%.2f, %.2f] (expect ~9-12)\n", Vmid, minimum(pH), maximum(pH))
    @printf("  CO2(aq): inlet-side=%.2f electrolyte-side=%.2f mol/m³ (expect depletion)\n",
            c[5,1], c[5,end])
    # Selectivity (§11 Stage 1 check #4). FE_CO at experimental onset (-0.816) and peak.
    iTot = iCO .+ iH2
    FE_CO = iCO ./ max.(iTot, 1e-30) .* 100
    j_onset = argmin(abs.(Vr .- (-0.816)))      # experimental first-current point
    @printf("  FE_CO at exp onset V=%.3f (iTot=%.2f) = %.1f%% (Lees exp = 54%%)\n",
            Vr[j_onset], iTot[j_onset], FE_CO[j_onset])
    @printf("  FE_CO peak = %.1f%% at V_RHE=%.3f (Lees exp peak 79.4%% at -1.205)\n",
            maximum(FE_CO), Vr[argmax(FE_CO)])
    # monotonicity of |total current| with cathodic potential (Vr ascending → iTot non-increasing)
    @printf("  i_total |V|-monotonic: %s (peak %.1f mA/cm² at V_RHE=%.3f)\n",
            all(diff(iTot) .<= 1e-3*maximum(iTot)) ? "yes" : "no", maximum(iTot), Vr[argmax(iTot)])
    # i_CO should rise, peak, fall (1 derivative sign change)
    sc = count(!=(0), diff(sign.(diff(iCO))))
    @printf("  i_CO rise→peak→fall: peak %.1f mA/cm² at V_RHE=%.3f, %d derivative sign change(s)\n",
            maximum(iCO), Vr[argmax(iCO)], sc)

    println("\n>>> STAGE 1 COMPLETE. Plots in $STAGE1_PLOTS. STOP for human review. <<<")
    return n_conv[], length(Vs)
end

main()
