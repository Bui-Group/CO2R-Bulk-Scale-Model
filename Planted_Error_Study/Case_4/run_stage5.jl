# run_stage5.jl — Stage 5: full potential physics. φ_L (NP migration + Cs⁺,
# charge balance) + φ_s (solid Ohm's law on GDL+CL). Uses FITTED kinetics and the
# unified continuation driver (§9.5.5): anchor ONCE at V_c = 0 from a guess
# constructed from physics (never from a cache), pre-ramp in −0.025 V steps to the
# first output voltage, then walk the 89-V grid with the tangent predictor (§9.5.4;
# φ_s-shift as the fallback predictor). Post-solve assertion ‖F‖∞ ≤ 1e-6 at every
# emitted voltage (§9.5.6). §3.1/§3.4/§11.

include(joinpath(@__DIR__, "co2r.jl"))
include(joinpath(@__DIR__, "diagnostics.jl"))

const OUT_DIR   = joinpath(HERE, "output_stage5")
const CACHE_DIR = joinpath(OUT_DIR, "cache")
const FIT_PARAMS = joinpath(HERE, "output_fit", "fitted_params.txt")
mkpath(CACHE_DIR)

function load_fitted_kinetics()
    isfile(FIT_PARAMS) || error("Fitted params missing ($FIT_PARAMS) — run the Fit step first.")
    d = Dict{String,Float64}()
    for ln in readlines(FIT_PARAMS)
        k, v = split(strip(ln))
        d[k] = parse(Float64, v)
    end
    set_i0_CO2R!(d["i0_CO2R"]); set_lambda_MHC!(d["lambda_MHC"])
    set_HER_prefactor!(d["HER_prefactor"]); set_alpha_c_HER!(d["alpha_c"])
end

function main()
    load_fitted_kinetics()
    @printf("=== Stage 5 ===  (N_CL=%d, N_GDL=%d)\n", N_CL, N_GDL)
    @printf("FITTED kinetics: i0_CO2R=%.2f A/m², λ_MHC=%.3f eV, HER_pf=%.4e, α_c=%.4f\n",
            i0_CO2R(), lambda_MHC(), HER_prefactor(), alpha_c_HER())

    clm = make_cl_mesh(); gdlm = make_gdl_mesh(); gm = build_gas_mesh(gdlm, clm)
    ly = Stage5Layout(clm.N, gdlm.N)
    c_eq = equilibrium_vec(1000.0, 1000.0)
    @printf("DOFs: %d\n", ly.N_TOT)

    # ── §9.5.3 weight blocks: liquid | gas GDL | gas CL | φ_s GDL | φ_s CL | φ_L; persistent LU
    blocks = dof_blocks(ly)
    plu = AffineLU()

    # ── continuation chain (§9.5.5), all in V_RHE: anchor at V_c = 0 (V_RHE = offset), pre-ramp
    #    in −0.025 V steps down to the first output voltage, then the 89 output voltages
    Vs = sort(collect(V_RHE_grid), by = v -> -v)                  # −0.300 … −2.500 V_RHE
    V_anchor = V_RHE_to_SHE_offset                                 # V_c = V_RHE − offset = 0
    n_pre = floor(Int, (V_anchor - Vs[1]) / 0.025 - 1e-9)
    pre = [V_anchor - 0.025*s for s in 0:n_pre]                    # V_c = 0, −0.025, …
    chain = vcat(pre, Vs)
    is_output = Set(Vs)
    @printf("V_c chain: %d points (%d pre-anchor + %d output); V_c = 0 → %.3f V\n",
            length(chain), length(pre), length(Vs), Vs[end] - V_RHE_to_SHE_offset)

    V_ref = Ref(V_anchor)
    res_closure!(F, uu) = residual_stage5!(F, uu, V_ref[], clm, gm, c_eq, ly)

    # anchor guess at V_c = 0 from physics (never from a cache): equilibrium liquid, inlet gas,
    # φ_s = 0 (= V_applied_SHE at V_c = 0), φ_L = 0
    u = anchor_guess(ly, c_eq)

    # fixed sparsity pattern by probing (§9.3.3): union of the AD-Jacobian nonzeros at the anchor
    # voltage and at the first output voltage, each also at a φ_s/φ_L-perturbed point so the
    # state-dependent migration cross-terms (∝ ∇φ_L, which vanish at φ_L = 0) are captured
    perturb = zeros(ly.N_TOT)
    for ix in 1:clm.N; perturb[phi_L_idx5(ly, ix)] = 0.005; end
    for s in 1:gm.Ng;  perturb[phi_s_idx5(ly, s)]  = 0.01;  end
    V_ref[] = V_anchor; r1, c1 = probe_sparsity(res_closure!, u; perturb = perturb)
    V_ref[] = Vs[1];    r2, c2 = probe_sparsity(res_closure!, u; perturb = perturb)
    V_ref[] = V_anchor
    Jt = pattern_matrix(vcat(r1, r2), vcat(c1, c2), ly.N_TOT)
    @printf("Sparsity pattern: %d nnz (%.2f%% dense)\n", nnz(Jt), 100*nnz(Jt)/ly.N_TOT^2)
    assemble! = make_ad_assembler(res_closure!, ly.N_TOT)

    rows = Vector{NTuple{7,Float64}}(); ok_Vs = Float64[]; sols = Dict{Float64,Vector{Float64}}()
    dphiL_log = Tuple{Float64,Float64,Float64}[]     # (V_rhe, |Δφ_L| mV, charge-cons rel-err)

    t0 = time()
    sweep_log = continuation_sweep!(u, chain, res_closure!, V_ref;
        blocks = blocks, plu = plu, jac_assembler = assemble!, J = Jt, n_log_dofs = 5*clm.N,
        shift_fallback! = (uu, dV) -> shift_phis!(uu, dV, ly),
        on_converged = (V_rhe_now, uu, info) -> begin
            V_rhe_now in is_output || return                  # pre-ramp points are not saved
            i_far, i_ion_R, i_s_coll = stage5_currents(uu, V_rhe_now, clm, gm, c_eq, ly)
            φL = [uu[phi_L_idx5(ly, ix)] for ix in 1:clm.N]
            φs_cl = [uu[phi_s_cl_idx5(ly, ix)] for ix in 1:clm.N]
            save_solution(CACHE_DIR, "s5_", copy(uu), V_rhe_now)
            push!(rows, polarization_row_local(copy(uu), V_rhe_now, clm, φs_cl, φL))
            push!(ok_Vs, V_rhe_now); sols[V_rhe_now] = copy(uu)
            dphiL = abs(φL[1] - φL[clm.N]) * 1000.0
            cc_err = abs(abs(i_far) - abs(i_ion_R)) / max(abs(i_far), 1e-12)
            push!(dphiL_log, (V_rhe_now, dphiL, cc_err))
            @printf("  V_RHE=%.3f ✓ %2d it ‖F‖∞=%.2e  i_CO=%.1f i_H2=%.1f | |Δφ_L|=%.1f mV  |i_s|/i_far=%.3f\n",
                    V_rhe_now, info.iterations, info.residual, rows[end][3], rows[end][4],
                    dphiL, abs(i_s_coll)/max(abs(i_far),1e-12))
        end)
    t_sweep = time() - t0
    nfail  = count(l -> !l.converged, sweep_log)
    n_iter = sum(l.iterations for l in sweep_log)
    maxF   = maximum(l.residual for l in sweep_log if l.converged && (l.V in is_output))
    @printf("\nSweep done: %d/%d output voltages converged in %.1f s (%d chain points, %d Newton iterations, %.2f/step). max ‖F‖∞ (outputs) = %.2e\n",
            length(ok_Vs), length(Vs), t_sweep, length(chain), n_iter, n_iter/length(chain), maxF)

    # ── diagnostics dump ──
    write_polarization(joinpath(OUT_DIR, "polarization.csv"), rows)
    cp(joinpath(HERE, "output_fit", "polarization.csv"),
       joinpath(OUT_DIR, "prev_polarization.csv"); force = true)
    sel = select_profile_voltages(sort(ok_Vs, by = v -> -v); nsel = 9)
    Vsel = sort(ok_Vs, by = v -> -v)[sel]
    liq = [(V, sols[V], [sols[V][phi_s_cl_idx5(ly,ix)] for ix in 1:clm.N],
            [sols[V][phi_L_idx5(ly,ix)] for ix in 1:clm.N]) for V in Vsel]
    write_liquid_profiles(joinpath(OUT_DIR, "liquid_profiles.csv"), clm, liq)
    write_gas_profiles(joinpath(OUT_DIR, "gas_profiles.csv"), gm, clm.N, [(V, sols[V]) for V in Vsel])
    write_potentials(joinpath(OUT_DIR, "potentials.csv"), gm, ly, [(V, sols[V]) for V in Vsel])

    # ── physicality checks (§Stage 5) ──
    println("\n── Physicality checks ──")
    @printf("Converged: %d/%d output voltages (%d chain-point failures)\n", length(ok_Vs), length(Vs), nfail)
    hi = filter(t -> abs(t[1]) >= 1.5, dphiL_log)
    if !isempty(hi)
        dmax = maximum(t[2] for t in hi); dmin = minimum(t[2] for t in hi)
        @printf("|Δφ_L| at high current (|V_RHE|≥1.5): [%.1f, %.1f] mV  (expect ~5-25 mV) \n", dmin, dmax)
    end
    ccmax = maximum(t[3] for t in dphiL_log)
    @printf("max ionic charge-conservation rel-err |i_far−i_ionic_R|/i_far = %.2e (expect <0.01)\n", ccmax)
    # solid ohmic drop at highest current
    Vhi = Vsel[end]; uhi = sols[Vhi]
    drop = uhi[phi_s_cl_idx5(ly, clm.N)] - (Vhi - V_RHE_to_SHE_offset)
    @printf("@ V_RHE=%.3f: φ_s(CL exit) − V_applied = %.1f mV (solid ohmic drop, expect 5-80 mV)\n",
            Vhi, drop*1000)
    itot = [r[5] for r in sort(rows, by = r -> -r[1])]
    @printf("i_total sign-decreasing steps: %d\n", sum(diff(itot) .< -1e-6))
    fecos = [r[6] for r in rows]; ipk = argmax(fecos)
    @printf("peak FE_CO = %.1f%% at V_RHE=%.3f\n", fecos[ipk], rows[ipk][1])
    println("\nmodel vs Lees (fitted kinetics, full potential physics):")
    for (k, Ve) in enumerate(E_exp_RHE)
        j = argmin(abs.([r[1] for r in rows] .- Ve))
        @printf("  V_RHE≈%.3f: i_CO=%.1f (exp %.1f), i_H2=%.1f (exp %.1f) mA/cm²\n",
                rows[j][1], rows[j][3], i_CO_exp[k], rows[j][4], i_H2_exp[k])
    end

    println("\nGenerating plots …")
    withenv("PREV_LABEL" => "Stage 4 fit") do
        run(`python3 $(joinpath(HERE, "plot_results.py")) $(OUT_DIR) "Stage 5" --gas --pot`)
    end
    @printf("\n=== Stage 5 complete: %d solutions cached ===\n", length(ok_Vs))
end

main()
