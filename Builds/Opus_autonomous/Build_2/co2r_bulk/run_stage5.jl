# run_stage5.jl — Stage 5: φ_L (NP migration) + φ_s (solid Ohm's law) as DOFs,
#                 with the FITTED kinetics from run_fit.jl.
#
# Seeding: NO cache.  Anchor once at V_c = 0 (zero overpotential) from a guess
# constructed from physics, then tangent-predictor continuation in −0.025 V
# steps down to the first output point and through the 89-voltage grid
# (§9.5.5, §11 Stage-5 gotchas #0 and #9).
include(joinpath(@__DIR__, "model.jl"))
using Printf

const N_CL  = N_CL_DEFAULT
const N_GDL = N_GDL_DEFAULT

# ── §11 gotcha #14: load the fitted kinetics BEFORE anything else ─────────
const FITFILE = joinpath(ROOT, "output_fit", "fitted_params.txt")
isfile(FITFILE) || error("No fitted parameters found at $FITFILE — run run_fit.jl first!")
let d = Dict{String,Float64}()
    for l in eachline(FITFILE)
        k, v = split(l, "="); d[strip(k)] = parse(Float64, strip(v))
    end
    set_i0_CO2R!(d["i0_CO2R"])
    set_lambda_MHC!(d["lambda_MHC"])
    set_HER_prefactor!(d["HER_prefactor"])
    set_alpha_c_HER!(d["alpha_c_HER"])
end

run5 = make_stage_run(Val(5), N_CL, N_GDL)
print_kinetics_banner(5, run5.cache_dir)
@printf("mesh: N_CL=%d N_GDL=%d ; DOFs=%d (expect %d)\n", N_CL, N_GDL, run5.ly.N_TOT,
        5*N_CL + 4*(N_GDL+N_CL) + N_GDL + N_CL + N_CL)
@printf("σ_s_eff: GDL %.2f S/m, CL %.2f S/m ; D_eff_Cs = %.3e m²/s\n",
        sigma_s_eff_GDL, sigma_s_eff_CL, D_eff_Cs)

# ── V_c chain (V_c is V vs SHE; V_cathode passed to the residual is V vs RHE) ─
const Vc_first = V_RHE_grid[1] - V_RHE_to_SHE_offset          # ≈ −0.81305
const n_ramp   = ceil(Int, abs(Vc_first) / 0.025)             # ≈ 33 steps
const Vc_ramp  = [Vc_first * m / n_ramp for m in 1:n_ramp]
const V_ramp   = Vc_ramp .+ V_RHE_to_SHE_offset               # → V vs RHE
const V_anchor = 0.0 + V_RHE_to_SHE_offset                    # V_c = 0
@printf("pre-anchor ramp: V_c = 0 → %.5f in %d steps of %.5f V; then the 89 grid points\n",
        Vc_first, n_ramp, Vc_first/n_ramp)

# anchor guess: bulk equilibrium liquid + inlet gas + φ_s = V_applied_SHE = 0,
# φ_L = 0 (§11 gotcha #9a: the SHE value, never the RHE value)
u0 = build_initial_guess(run5.ly, run5.cl, run5.gdl, run5.c_eq,
                         V_anchor - V_RHE_to_SHE_offset)
@printf("anchor guess: φ_s = %.6f V vs SHE on GDL+CL, φ_L = 0\n",
        u0[phi_s_gdl_idx(run5.ly, 1)])
Fchk = zeros(run5.ly.N_TOT)
res_at_V!(run5, Fchk, u0, V_anchor)
@printf("‖F‖∞ at the V_c = 0 anchor guess = %.3e\n", maximum(abs, Fchk))

# ── §9.5.9 step 3: ∂F/∂V structure probe ─────────────────────────────────
let Fp = zeros(run5.ly.N_TOT), Fm = zeros(run5.ly.N_TOT), h = 1e-6
    res_at_V!(run5, Fp, u0, V_anchor + h); res_at_V!(run5, Fm, u0, V_anchor - h)
    nz = count(x -> abs(x) > 1e-12, (Fp .- Fm) ./ (2h))
    @printf("∂F/∂V structure probe: %d nonzero entries (Stage 5 expects exactly 1)\n", nz)
end

# V_ramp[end] lands exactly on V_RHE_grid[1]; drop it so the grid point is
# visited once (a duplicate would emit 90 voltages and put a zero-length step
# into the polarization first difference).
chain = vcat(V_ramp[1:end-1], V_RHE_grid)
sw = run_sweep!(run5, u0, V_anchor, chain, V_RHE_grid)
s  = export_stage(run5, sw.Vs, sw.us)

@printf("\n=== Stage 5 summary ===\n")
@printf("converged %d / %d output voltages (%d chain points)\n",
        length(sw.Vs), length(V_RHE_grid), length(chain))
@printf("max ‖F‖∞ = %.3e ; mean Newton iters = %.2f ; wall = %.1f s ; sub-steps = %d\n",
        maximum(sw.resn), sum(sw.iters)/length(sw.iters), sw.wall, sw.nsub)
print_physicality(run5, s, sw.Vs, sw.us)

# ── §11 Stage-5 physicality checks ───────────────────────────────────────
println("\n── Stage 5 potential / charge checks ──")
println("  V_RHE  i_tot[mA/cm²]  Δφ_L[mV]  φ_s(L_CL)−V_app[mV]  GDL share  |Δi|/i solid  rel.err ionic")
NKtot = N_GDL + N_CL
for k in (1, 20, 37, 55, 73, 89)
    u = sw.us[k]; V = sw.Vs[k]
    Vapp = V - V_RHE_to_SHE_offset
    phis_gdl_end = u[phi_s_gdl_idx(run5.ly, N_GDL)]
    tot = u[phi_s_cl_idx(run5.ly, N_CL)] - Vapp
    gdl_share = tot == 0 ? NaN : 100*(phis_gdl_end - Vapp)/tot
    @printf("  %6.3f %12.2f %10.4f %20.3f %9.1f%% %13.2e %14.2e\n",
            V, s.itot[k]/10, s.dphiL[k]*1000, tot*1000, gdl_share,
            s.chg_solid[k], s.chg_ionic[k])
end
hi = findall(s.itot ./ 10 .> 500.0)
if !isempty(hi)
    dl = abs.(s.dphiL[hi]) .* 1000
    @printf("|Δφ_L| at i_tot > 500 mA/cm²: %.2f – %.2f mV (target 5–25 mV)\n",
            minimum(dl), maximum(dl))
end
@printf("max |Δi|/i solid  = %.3e (target ≤ 1e-6)\n", maximum(s.chg_solid))
@printf("max rel.err ionic = %.3e (target < 0.01, expect < 0.001)\n", maximum(s.chg_ionic))
@printf("sign of Δφ_L (φ_L[1] − φ_L[N_CL]) at the deepest V: %+.4f mV (expect negative)\n",
        s.dphiL[end]*1000)
@printf("sign changes di_CO/dV = %d (target 1) ; di_H2/dV = %d (target 0)\n",
        sign_changes(s.iCO), sign_changes(s.iH2))

# ── Stage 4→5 comparison (§11 Stage-5 check 5) ───────────────────────────
fitcsv = joinpath(ROOT, "output_fit", "summary.csv")
if isfile(fitcsv)
    ls = readlines(fitcsv)[2:end]
    iCO4 = [parse(Float64, split(l,",")[3]) for l in ls]
    iH24 = [parse(Float64, split(l,",")[4]) for l in ls]
    println("\n── Stage 4 (fitted kinetics) → Stage 5 current change ──")
    println("  V_RHE   i_CO_S4  i_CO_S5  ratio    i_H2_S4  i_H2_S5  ratio")
    for k in (20, 37, 55, 73, 89)
        @printf("  %6.3f %8.2f %8.2f %6.3f %9.2f %8.2f %6.3f\n", sw.Vs[k],
                iCO4[k], s.iCO[k]/10, (s.iCO[k]/10)/iCO4[k],
                iH24[k], s.iH2[k]/10, (s.iH2[k]/10)/iH24[k])
    end
end
compare_to_experiment(sw.Vs, s.iCO, s.iH2)
println("\nStage 5 complete — STOP for review.")
