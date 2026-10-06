# run_stage5.jl — Stage 5: φ_L (NP migration) + φ_s (solid Ohm's law), fitted kinetics (§11).
# Solver: §9.5 newton_affine! + persistent sparse LU + fixed AD sparsity pattern (probed at the
# anchor guess at V_c = 0 AND at the first output voltage, with a φ perturbation so kinetic +
# φ migration cross-terms are captured). Driver: §9.5.5 continuation — anchor ONCE at V_c = 0
# from a guess constructed from physics (never a cache), pre-ramp in −0.025 V steps to the first
# output point, then the 89 output voltages with the tangent predictor (§9.5.4); §9.5.6
# post-solve assertion ‖F‖∞ ≤ 1e-6 at every chain point; halve-dV sub-step recovery.
include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "plots.jl"))
using Printf

const FIT_CACHE    = joinpath(HERE, "output_fit", "cache")
const STAGE5_CACHE = joinpath(HERE, "output_stage5", "cache")
const STAGE5_PLOTS = joinpath(HERE, "output_stage5")

function load_fitted_kinetics()
    fn = joinpath(HERE, "output_fit", "fitted_params.txt")
    isfile(fn) || error("No fitted params — run run_fit.jl first")
    p = parse.(Float64, readlines(fn))
    set_i0_CO2R!(p[1]); set_lambda_MHC!(p[2]); set_HER_prefactor!(p[3]); set_alpha_c_HER!(p[4])
end

function main()
    println("="^70)
    println("=== Stage 5: φ_L migration + φ_s solid Ohm's law (fitted kinetics) ===")
    load_fitted_kinetics()
    @printf("FITTED kinetics: i0_CO2R=%.2f A/m², λ_MHC=%.4f eV, HER_pf=%.4e, α_c=%.5f\n",
            i0_CO2R(), lambda_MHC(), HER_prefactor(), alpha_c_HER())
    @printf("Mesh: N_CL=%d N_GDL=%d, Stage-5 DOFs = %d\n", N_CL, N_GDL, N_STAGE5)
    println("="^70)
    # the fit cache is the Stage-4-physics reference at fitted kinetics (§17.8) — not a Stage 5 seed
    isempty(list_cache(FIT_CACHE, "fit")) && error("No fit cache — run run_fit.jl first (fitted kinetics + the fitted-kinetics Stage-4 reference sweep)")

    cl_mesh = make_cl_mesh(); gdl_mesh = make_gdl_mesh(); gas_mesh = GasMesh(gdl_mesh, cl_mesh)
    c_eq_vec = equilibrium_c_vec(1000.0, 1000.0)
    N = N_STAGE5

    residual_at_V!(F, u, V) = residual_stage5!(F, u, V, cl_mesh, gdl_mesh, gas_mesh, c_eq_vec)

    # ── V_c chain: pre-anchor ramp 0 → first output, then exact output V_c = V_SHE_grid ──
    Vc_out = sort(V_SHE_grid, rev=true)                 # mild → cathodic
    pre = collect(0.0:-0.025:Vc_out[1])                 # 0 … just above first output
    Vc_chain = vcat(pre, Vc_out)
    @printf("Continuation chain: %d points (%d pre-anchor + %d output), V_c ∈ [%.3f, %.3f]\n",
            length(Vc_chain), length(pre), length(Vc_out), Vc_chain[1], Vc_chain[end])

    # ── Anchor guess at V_c = 0 (§9.5.5): equilibrium liquid, inlet gas, φ_s = φ_L = 0 — from physics, not a cache
    u = anchor_guess_stage5()

    # ── Fixed sparsity pattern (§9.3.3 / §9.5.2): union of the probes at the anchor voltage (V_c = 0) and
    #    at the first output voltage (kinetics active). probe_sparsity perturbs every DOF (incl. φ_s/φ_L),
    #    so the NP-migration cross-terms (∝ ∇φ_L, identically zero at the anchor) are in the pattern.
    Jpat = probe_sparsity((F, uu) -> residual_at_V!(F, uu, Vc_chain[1] + V_RHE_to_SHE_offset), N; u_seed=u, perturb=0.01) +
           probe_sparsity((F, uu) -> residual_at_V!(F, uu, Vc_out[1]   + V_RHE_to_SHE_offset), N; u_seed=u, perturb=0.01)
    @printf("Sparsity: %d nonzeros (%.3f%% dense)\n", nnz(Jpat), 100*nnz(Jpat)/N^2)
    plu = AffineLU()
    blocks = dof_blocks(5)            # liquid | gas GDL | gas CL | φ_s GDL | φ_s CL | φ_L

    # residual closure in the chain variable V_c (the residual takes V_RHE)
    Vc_ref = Ref(Vc_chain[1])
    res_closure!(F, uu) = residual_at_V!(F, uu, Vc_ref[] + V_RHE_to_SHE_offset)
    assemble! = make_ad_assembler(res_closure!, N; chunk=12)

    # ── Anchor + continuation (§9.5.5), φ_s-shift fallback when no tangent is available (§9.5.4) ──
    n_out = Ref(0); maxres = Ref(0.0)
    results = Tuple{Float64,Float64,Float64,Float64,Float64}[]   # V_RHE, iCO, iH2, charge_err, dphiL
    t0 = time()
    sweep_log = continuation_sweep!(u, Vc_chain, res_closure!, Vc_ref;
        blocks=blocks, plu=plu, jac_assembler=assemble!, J=Jpat, n_log_dofs=5*N_CL,
        shift_fallback! = shift_phis!, label="V_c", progress_every=typemax(Int),
        on_converged = (Vc, uu, info) -> begin
            # Save at output voltages only (the pre-ramp and the anchor are not cached)
            if any(abs.(V_SHE_grid .- Vc) .< 1e-9)
                V_RHE_now = Vc + V_RHE_to_SHE_offset
                save_solution(STAGE5_CACHE, "s5", uu, V_RHE_now)
                n_out[] += 1; maxres[] = max(maxres[], info.residual)
                d = stage5_diagnostics(uu, cl_mesh, gas_mesh, V_RHE_now)
                iCO = compute_i_CO_s5(uu, cl_mesh)/10; iH2 = compute_i_H2_s5(uu, cl_mesh)/10
                push!(results, (V_RHE_now, iCO, iH2, d.charge_err, d.dphiL))
                (n_out[] % 10 == 0) && @printf("  [out %2d] V_RHE=%.3f iter=%2d ‖F‖∞=%.1e | iCO=%.1f iH2=%.1f | chg_err=%.1e Δφ_L=%.1f mV\n",
                        n_out[], V_RHE_now, info.iterations, info.residual, iCO, iH2, d.charge_err, 1000*d.dphiL)
            end
        end)
    n_conv = count(l.converged for l in sweep_log[2:end])
    n_iter = sum(l.iterations for l in sweep_log[2:end])
    n_sub  = count(l.substeps > 0 for l in sweep_log[2:end])
    @printf("\nStage 5 continuation: anchor V_c=0 in %d iters (‖F‖∞=%.2e); %d/%d chain steps converged (%d Newton iterations, %.2f/step, %d sub-stepped), %d/89 output voltages saved in %.1fs, max‖F‖∞ over outputs=%.2e\n",
            sweep_log[1].iterations, sweep_log[1].residual, n_conv, length(Vc_chain)-1, n_iter,
            n_iter/(length(Vc_chain)-1), n_sub, n_out[], time()-t0, maxres[])

    # ── Plots + diagnostics ──
    Vcache = sort(list_cache(STAGE5_CACHE, "s5"))
    Vr = Float64[]; iCO = Float64[]; iH2 = Float64[]
    prof = Tuple{Float64,Matrix{Float64},Vector{Float64},Vector{Float64}}[]
    gasser = Tuple{Float64,Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64}}[]
    dser = Tuple{Float64,Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64}}[]
    x_s_μm = vcat(gdl_mesh.xc, cl_mesh.xc).*1e6; x_cl_μm = cl_mesh.xc.*1e6
    x_gas_μm = x_s_μm
    for V in Vcache
        uu = load_solution(STAGE5_CACHE, "s5", V); c = unpack_log_conc(uu, N_CL)
        φs = unpack_phi_s(uu); φL = unpack_phi_L(uu)
        push!(Vr, V); push!(iCO, compute_i_CO_s5(uu, cl_mesh)/10); push!(iH2, compute_i_H2_s5(uu, cl_mesh)/10)
        rco2r = [R_CO2R(c[5,ix], φs[N_GDL+ix], φL[ix], c[1,ix], a_v_CL, lambda_MHC(), i0_CO2R(), n_e) for ix in 1:N_CL]
        rher  = [R_HER(φs[N_GDL+ix], φL[ix], c[1,ix], a_v_CL, n_e) for ix in 1:N_CL]
        push!(prof, (V, c, rco2r, rher))
        pCO2,pCO,pH2,ptot = gas_profiles(uu, N_CL, N_GDL); push!(gasser,(V,pCO2,pCO,pH2,ptot))
        push!(dser, (V, x_s_μm, φs, x_cl_μm, φL))
    end
    plot_A1(STAGE5_PLOTS, Vr, iCO, iH2; title_prefix="Stage 5")
    plot_A2(STAGE5_PLOTS, cl_mesh, prof; title_prefix="Stage 5")
    plot_B1(STAGE5_PLOTS, x_gas_μm, gasser; title_prefix="Stage 5")
    plot_D(STAGE5_PLOTS, dser; title_prefix="Stage 5")

    # 5-stage polarization overlay
    function stage_iV(cache, prefix, s5::Bool)
        Vs = sort(list_cache(cache, prefix)); ico=Float64[]; ih2=Float64[]
        for V in Vs
            u = load_solution(cache, prefix, V)
            if s5
                push!(ico, compute_i_CO_s5(u, cl_mesh)/10); push!(ih2, compute_i_H2_s5(u, cl_mesh)/10)
            else
                Va = V - V_RHE_to_SHE_offset
                push!(ico, compute_i_CO(u, Va, cl_mesh)/10); push!(ih2, compute_i_H2(u, Va, cl_mesh)/10)
            end
        end
        (Vs, ico, ih2)
    end
    ov = [("S1",stage_iV(joinpath(HERE,"output_stage1","cache"),"s1",false)...),
          ("S2",stage_iV(joinpath(HERE,"output_stage2","cache"),"s2",false)...),
          ("S3",stage_iV(joinpath(HERE,"output_stage3","cache"),"s3",false)...),
          ("S4",stage_iV(joinpath(HERE,"output_stage4","cache"),"s4",false)...),
          ("S5",Vr,iCO,iH2)]
    plot_overlay(STAGE5_PLOTS, ov; fname="overlay_all_stages.png", title="All stages 1–5 polarization")

    # ── Physicality checks (§11 Stage 5) ──
    println("\n── Physicality checks ──")
    maxchg = maximum(r[4] for r in results); maxdphi = maximum(1000*r[5] for r in results)
    @printf("  Charge conservation |i_s(collector)|=i_far: max rel err over sweep = %.2e (target ≤1e-6)\n", maxchg)
    hi = [r for r in results if (r[2]+r[3]) > 50]
    if !isempty(hi)
        dphis = [1000*r[5] for r in hi]
        @printf("  Δφ_L at i>50 mA/cm² pts: range [%.1f, %.1f] mV (target 5-25 mV)\n",
                minimum(dphis), maximum(dphis))
    end
    for Vt in (-1.2, -2.0)
        V = Vcache[argmin(abs.(Vcache .- Vt))]; uu = load_solution(STAGE5_CACHE, "s5", V)
        d = stage5_diagnostics(uu, cl_mesh, gas_mesh, V)
        @printf("  V_RHE=%.3f: solid drop φ_s(cat)-V_app=%.1f mV, Δφ_L=%.1f mV, chg_err=%.1e\n",
                V, 1000*d.solid_drop, 1000*d.dphiL, d.charge_err)
    end
    # monotonicity
    sc = count(!=(0), diff(sign.(diff(iCO))))
    @printf("  i_CO: peak %.1f mA/cm² at V_RHE=%.3f, %d derivative sign change(s)\n", maximum(iCO), Vr[argmax(iCO)], sc)
    # Vr ascending (−2.5→−0.3) ⇒ |V| decreasing ⇒ i_H2 should be non-increasing along Vr
    nviol = count(x -> x > 1e-3, diff(iH2))
    @printf("  i_H2 strictly monotone in |V|: %s (%d violations)\n", nviol == 0 ? "yes" : "no", nviol)

    println("\n>>> STAGE 5 COMPLETE (final physics gate). Plots in $STAGE5_PLOTS. STOP for human review. <<<")
    return n_out[]
end

main()
