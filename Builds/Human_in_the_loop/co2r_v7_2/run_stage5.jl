# =============================================================================
# run_stage5.jl — Stage 5: full coupled electrochemistry with FITTED kinetics.
#
#   + phi_L  : monolithic DOF, per-cell ionic charge balance, NP migration in
#              every liquid flux INCLUDING the electrolyte boundary, and the
#              Cs+ migration current in i_L (Cs+ has no residual row of its own)
#   + phi_s  : solid Ohm's law on [-L_GDL, L_CL], ghost-cell Dirichlet at the
#              current collector, Neumann zero at the electrolyte side
#   + kinetics driven by LOCAL (phi_s - phi_L), not by V_applied
#
# Seeding (§9.5.5, §11 gotcha #0): NO cache.  One anchor at V_c = 0 built from
# physics, then tangent-predictor continuation in -0.025 V steps.  The Stage-5
# residual is multi-rooted; cache-seeded solves land on a low-CO branch that
# passes every numerical check.
#
# STOPS after generating plots.  This is the final physics gate.
# =============================================================================
include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "export.jl"))
include(joinpath(@__DIR__, "diagnostics.jl"))
using Printf

# ── §11 gotcha #14: load the FITTED kinetics BEFORE anything else ───────────
const FITFILE = joinpath(@__DIR__, "fitted_kinetics.jl")
isfile(FITFILE) || error("fitted_kinetics.jl not found — run run_fit.jl first (§11 gotcha #14)")
include(FITFILE)

const STAGE     = 5
const CACHE_DIR = cache_dir_for(STAGE)
const PREFIX    = cache_prefix_for(STAGE)

print_stage_header(STAGE, CACHE_DIR;
    extra = "Physics: Stage 4 + phi_L (NP migration, monolithic DOF) + phi_s (solid Ohm's law)")
println("Kinetics loaded from fitted_kinetics.jl (§11 gotcha #14) — verify against run_fit.jl:")
@printf("  i0_CO2R = %.4f A/m^2 (pre-fit 540) | lambda_MHC = %.5f eV (pre-fit 1.8)\n", i0_CO2R(), lambda_MHC())
@printf("  HER_pf  = %.6e (pre-fit 2.409e13) | alpha_c    = %.6f (pre-fit 0.0872)\n", HER_prefactor(), alpha_c_HER())
i0_CO2R() == 540.0 && error("Stage 5 is running PRE-FIT kinetics — §14 pitfall / gotcha #14")

ctx = make_ctx(STAGE)
@printf("\nDOFs: %d = 5x%d liquid + 4x(%d+%d) gas + (%d+%d) phi_s + %d phi_L\n",
        ctx.ly.N_TOT, N_CL, N_GDL, N_CL, N_GDL, N_CL, N_CL)
@printf("sigma_s_eff: GDL %.3f S/m, CL %.3f S/m | D_eff_Cs = %.4e m^2/s, z_Cs = %+.0f\n",
        sigma_s_eff_GDL, sigma_s_eff_CL, D_eff_Cs, z_Cs)

# ── §9.5.5 V_c chain: pre-anchor ramp from V_c = 0, then the 89 output points ─
# V_c = V_RHE - V_RHE_to_SHE_offset, so V_c = 0 <=> V_RHE = +0.51305 and the
# phi_s block of the anchor guess is written as exactly V_applied_SHE = 0.
const V_c_pre  = collect(0.0:-DV_C_STEP:(V_SHE_grid[1] + 1e-9))
const V_chain  = vcat(V_c_pre .+ V_RHE_to_SHE_offset, collect(V_RHE_grid))
const emit     = Vector{Bool}(vcat(falses(length(V_c_pre)), trues(length(V_RHE_grid))))
@printf("\nContinuation chain: %d pre-anchor ramp points (V_c = %.3f -> %.3f) + %d output points = %d total\n",
        length(V_c_pre), V_c_pre[1], V_c_pre[end], length(V_RHE_grid), length(V_chain))
@printf("Anchor: V_c = 0.000 (V_RHE = %+.5f, V_applied_SHE = %.1f) — constructed from physics, never a cache\n\n",
        V_chain[1], V_chain[1] - V_RHE_to_SHE_offset)

u_state = make_anchor_guess(ctx, V_chain[1])   # §17.8: run_stage5.jl reads no warm-start cache
let Fa = zeros(ctx.ly.N_TOT)
    full_residual!(Fa, u_state, V_chain[1], ctx)
    @printf("||F||inf at the constructed V_c = 0 anchor guess = %.3e\n", maximum(abs, Fa))
    ps = unpack_phi_s_gdl(u_state, ctx.ly); pl = unpack_phi_L_cl(u_state, ctx.ly)
    @printf("anchor phi_s block identically zero: %s ; phi_L block identically zero: %s\n",
            all(ps .== 0), all(pl .== 0))
    println("  (§9.5.3: both blocks have rms exactly 0 at the anchor — the unit-weight fallback is exercised on the very first Newton step)")
end

res = run_continuation(ctx, u_state, V_chain, emit;
                       cache_dir = CACHE_DIR, prefix = PREFIX, record_predictor = true)

@printf("\nSweep complete: %d/%d output voltages converged and cached (chain length %d)\n",
        length(res.voltages), length(V_RHE_grid), res.n_chain)
@printf("  Newton iters/step: %.2f\n", res.iters_per_step)
@printf("  max ||F||inf     : %.3e   (assertion tol %.1e)\n", maximum(res.resids), F_INF_TOL)
@printf("  LU: %d symbolic / %d numeric (nnz-guard %d, lu!-pattern-catch %d)\n",
        res.lu_symbolic, res.lu_numeric, res.lu_nnz_change, res.lu_pattern_catch)
@printf("  sub-step recoveries: %d ; §9.5.6 assertion recovery fired: %d times\n",
        res.n_substep, res.n_assert_fire)
@printf("  wall time: %.2f s\n", res.wall)

# ── §9.5.9 step 4: predictor A/B ─────────────────────────────────────────────
if !isempty(res.pred_tangent_err)
    r = res.pred_carry_err ./ max.(res.pred_tangent_err, 1e-300)
    @printf("\n§9.5.9 step 4 — predictor A/B over %d continuation steps:\n", length(r))
    @printf("  ||u_pred - u*||: tangent median %.3e, carry-forward+phi_s-shift median %.3e\n",
            sort(res.pred_tangent_err)[cld(end,2)], sort(res.pred_carry_err)[cld(end,2)])
    @printf("  improvement factor: median %.1fx, min %.1fx, max %.1fx\n",
            sort(r)[cld(end,2)], minimum(r), maximum(r))
    @printf("  tangent better at EVERY step: %s   (criterion: median improvement > 10x)\n", all(r .> 1.0))
end

# ── Currents ─────────────────────────────────────────────────────────────────
Vs = res.voltages; sols = res.solutions
iCO = Float64[]; iH2 = Float64[]; itot = Float64[]
for (k, V) in enumerate(Vs)
    a, b, c = compute_currents(sols[k], V, ctx)
    push!(iCO, a/10); push!(iH2, b/10); push!(itot, c/10)
end
FE = 100 .* iCO ./ max.(itot, 1e-30)
kpk = argmax(FE)
nCO, nH2 = polarization_smoothness(Vs, iCO, iH2)
@printf("\n--- Polarization summary ---\n")
@printf("  i_CO  peak  %.2f mA/cm^2 at V_RHE = %.3f\n", maximum(iCO), Vs[argmax(iCO)])
@printf("  i_H2  max   %.2f mA/cm^2 at V_RHE = %.3f\n", maximum(iH2), Vs[argmax(iH2)])
@printf("  FE_CO peak  %.1f %% at V_RHE = %.3f\n", FE[kpk], Vs[kpk])
@printf("  §9.5.8 smoothness: sign changes di_CO/dV = %d (want exactly 1), di_H2/dV = %d (want 0)\n", nCO, nH2)
@printf("  i_H2 strictly monotone in |V_RHE|: %s\n", all(diff(iH2) .> 0))

# ── §11 Stage 5 physicality checks 1-4, 6 + §9.5.8 Δφ_L diagnostic ──────────
println("\n--- §11 Stage 5 physicality checks ---")
println("    V_RHE  i_tot[mA/cm2]  phi_s drop[mV]  GDL share  |dphi_L|[mV]  |di|/i solid   |di|/i ionic")
dmax_solid = 0.0; dmax_ionic = 0.0; dphis = Float64[]; dphiLs = Float64[]
for (k, V) in enumerate(Vs)
    d = stage5_diagnostics(sols[k], V, ctx)
    global dmax_solid = max(dmax_solid, d.d_solid)
    global dmax_ionic = max(dmax_ionic, d.d_ionic)
    push!(dphis, 1000*d.drop_tot); push!(dphiLs, 1000*d.dphiL)
    if k in (1, argmin(abs.(Vs .- (-1.200))), argmin(abs.(Vs .- (-1.650))),
             argmin(abs.(Vs .- (-2.075))), length(Vs))
        @printf("  %7.3f  %12.2f  %14.2f  %8.1f%%  %12.3f  %12.3e  %12.3e\n",
                V, itot[k], 1000*d.drop_tot, d.gdl_share, 1000*abs(d.dphiL), d.d_solid, d.d_ionic)
    end
end
@printf("\n  1. solid ohmic drop over the sweep: %.2f -> %.2f mV  (§11 expects 5-80 mV; ~50 mV at 1 A/cm2)\n",
        dphis[1], dphis[end])
let khi = argmin(abs.(itot .- 1000.0))
    @printf("     at i_tot = %.0f mA/cm2 (closest to 1 A/cm2): drop = %.1f mV\n", itot[khi], dphis[khi])
end
let d = stage5_diagnostics(sols[end], Vs[end], ctx)
    @printf("  2. GDL/CL split of the solid drop at V_RHE=%.3f: GDL %.1f %% / CL %.1f %%  (§11 expects GDL 80-90 %%)\n",
            Vs[end], d.gdl_share, 100 - d.gdl_share)
    @printf("  3. phi_L profile monotone across the CL: %s ; phi_L = (%.3f, %.3f, %.3f) mV at (1, N/2, N)\n",
            d.monotone_L, 1000*d.phiL1, 1000*d.phiLmid, 1000*d.phiLN)
    @printf("     max c_Cs in the reaction zone at this voltage: %.0f mol/m^3  (§3.1 quotes 2045 -> 6027 over the sweep)\n", d.cCs_max)
end
@printf("  4. current consistency |i_s_face[1]| = i_far_total : max |di|/i = %.3e  (§11 requires <= 1e-6)  PASS=%s\n",
        dmax_solid, dmax_solid <= 1e-6)
@printf("  6. ionic current at the electrolyte boundary       : max |di|/i = %.3e  (§11 requires < 0.01)  PASS=%s\n",
        dmax_ionic, dmax_ionic < 0.01)

# §9.5.8 Δφ_L diagnostic — the single most important Stage-5 check
println("\n--- §9.5.8 / §3.1 Δφ_L diagnostic (the Cs+-migration smoking gun) ---")
let hi = findall(itot .> 500.0)
    if isempty(hi)
        println("  no voltages above 500 mA/cm^2 — diagnostic not applicable")
    else
        a = abs.(dphiLs[hi])
        @printf("  %d voltages with i_tot > 500 mA/cm^2: |dphi_L| = %.2f .. %.2f mV (median %.2f)\n",
                length(hi), minimum(a), maximum(a), sort(a)[cld(end,2)])
        @printf("  in the 5-25 mV band at every one of them: %s\n", all(5.0 .<= a .<= 25.0))
        @printf("  none below 5 mV (would mean Cs+ missing from i_L, or the low-CO branch): %s\n", all(a .>= 5.0))
        @printf("  none above 30 mV (would mean a numerical artefact): %s\n", all(a .<= 30.0))
    end
    kv2 = argmin(abs.(Vs .- (-2.0 - V_RHE_to_SHE_offset + V_RHE_to_SHE_offset)))
    @printf("  sign of Δφ_L (must be negative, gauge phi_L[N]=0): phi_L[1] = %.3f mV at V_RHE=%.3f\n",
            1000*stage5_diagnostics(sols[end], Vs[end], ctx).phiL1, Vs[end])
    @printf("  phi_L never changes sign between neighbouring voltages: %s\n",
            all(sign.(dphiLs[2:end]) .== sign(dphiLs[end])) || all(dphiLs .<= 0))
end

stage3_checks(Vs, sols, ctx; pinned = false)
stage_liquid_checks(Vs, sols, ctx)
compare_to_experiment(Vs, iCO, iH2)

files = export_stage("stage5", Vs, sols, ctx)
println("\nExported:"); for f in files; println("  ", f); end

println("\n" * "="^78)
println("STAGE 5 COMPLETE — 🛑 FINAL PHYSICS STOP GATE.")
println("="^78)
