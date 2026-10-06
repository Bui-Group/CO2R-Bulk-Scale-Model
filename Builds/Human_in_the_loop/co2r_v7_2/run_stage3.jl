# =============================================================================
# run_stage3.jl — Stage 3: Stefan-Maxwell multicomponent gas transport
# (CO2/CO/H2/H2O) coupled to the liquid through volumetric CO2 phase transfer.
#
#   * u_G = 0 (no Darcy convection) — the convective term is activated in Stage 4
#   * the pressure DOF exists so the Stage-3/4 layouts match, but its residual
#     is the placeholder p[ix] - p_total, so p is EXACTLY uniform at 1 atm
#   * the CO2(aq) left BC switches from the constant Henry value to
#     H_eff(Sechenov) * p_CO2(local gas DOF)
#   * still simplified kinetics: phi_s = V_applied_SHE, phi_L = 0
#
# Solver: §9.5 production stack with the unified §9.5.5 driver. 1200 DOFs.
# STOPS after generating plots.
# =============================================================================
include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "export.jl"))
include(joinpath(@__DIR__, "diagnostics.jl"))
using Printf

const STAGE     = 3
const CACHE_DIR = cache_dir_for(STAGE)
const PREFIX    = cache_prefix_for(STAGE)

print_stage_header(STAGE, CACHE_DIR;
    extra = "Physics: Stage 2 + Stefan-Maxwell gas (u_G = 0, p pinned) + volumetric R_PT")

ctx = make_ctx(STAGE)
@printf("DOFs: %d = 5x%d liquid + 4x(%d+%d) gas\n", ctx.ly.N_TOT, N_CL, N_GDL, N_CL)
@printf("Inlet: x = [%s] (mole)  ->  omega = [%s] (mass), M_avg = %.3f g/mol\n",
        join([@sprintf("%.2f", x) for x in x_inlet], ", "),
        join([@sprintf("%.4f", x) for x in omega_inlet], ", "), M_avg_inlet*1000)
@printf("FSG binary D (raw, m^2/s): CO2-CO %.4e  CO2-H2 %.4e  CO2-H2O %.4e\n",
        D_bin_raw[1,2], D_bin_raw[1,3], D_bin_raw[1,4])
@printf("a_v_GL = %.4e 1/m, k_GL = %.3e m/s\n\n", a_v_GL, k_GL())

# ── §11 Stage 3 bootstrap step 1: gas-only solve ─────────────────────────────
# With R_PT = R_CO2R = R_HER = 0 and every gas cell at the inlet composition and
# p_total, the gas residual must be identically zero (no gradients, no sources).
# This is also what triggers the JIT compilation of the whole SM chain cheaply.
println("--- Bootstrap step 1: gas-only check (all gas sources off) ---")
let
    i0s, hps, kgs = i0_CO2R(), HER_prefactor(), k_GL()
    set_i0_CO2R!(0.0); set_HER_prefactor!(0.0); set_k_GL!(0.0)
    u_gas = make_anchor_guess(ctx, V_RHE_grid[1])
    Fg = zeros(ctx.ly.N_TOT)
    t = @elapsed full_residual!(Fg, u_gas, V_RHE_grid[1], ctx)
    gas_rows = (5*N_CL+1):(5*N_CL + 4*(N_GDL+N_CL))
    @printf("  first residual evaluation (JIT): %.3f s\n", t)
    @printf("  max |F| over the %d gas rows = %.3e   PASS=%s\n",
            length(gas_rows), maximum(abs, Fg[gas_rows]), maximum(abs, Fg[gas_rows]) < 1e-14)
    set_i0_CO2R!(i0s); set_HER_prefactor!(hps); set_k_GL!(kgs)
end

# ── §11 Stage 3 bootstrap step 2: monolithic Newton, liquid warm-started ─────
V_chain = collect(V_RHE_grid)
emit    = fill(true, length(V_chain))

u0 = make_anchor_guess(ctx, V_chain[1])          # equilibrium liquid + inlet gas
u2 = load_solution(cache_dir_for(2), cache_prefix_for(2), V_chain[1])
if u2 !== nothing && length(u2) == 5*N_CL
    u0[1:5*N_CL] .= u2                            # liquid from the Stage-2 cache
    println("\nAnchor guess: Stage-2 liquid + inlet gas composition at p_total")
else
    println("\nAnchor guess: constructed equilibrium liquid + inlet gas at p_total")
end

res = run_continuation(ctx, u0, V_chain, emit; cache_dir = CACHE_DIR, prefix = PREFIX)

@printf("\nSweep complete: %d/%d voltages converged and cached\n", length(res.voltages), length(V_chain))
@printf("  Newton iters/step: %.2f\n", res.iters_per_step)
@printf("  max ||F||inf     : %.3e   (assertion tol %.1e)\n", maximum(res.resids), F_INF_TOL)
@printf("  LU: %d symbolic / %d numeric refreshes (nnz-guard rebuilds %d, lu!-pattern-catch rebuilds %d)\n",
        res.lu_symbolic, res.lu_numeric, res.lu_nnz_change, res.lu_pattern_catch)
@printf("  sub-step recoveries triggered: %d\n", res.n_substep)
@printf("  wall time: %.2f s\n", res.wall)

Vs = res.voltages; sols = res.solutions
iCO = Float64[]; iH2 = Float64[]; itot = Float64[]
for (k, V) in enumerate(Vs)
    a, b, c = compute_currents(sols[k], V, ctx)
    push!(iCO, a/10); push!(iH2, b/10); push!(itot, c/10)
end
FE = 100 .* iCO ./ max.(itot, 1e-30)
kpk = argmax(FE)
@printf("\n--- Polarization summary ---\n")
@printf("  i_CO  peak  %.2f mA/cm^2 at V_RHE = %.3f\n", maximum(iCO), Vs[argmax(iCO)])
@printf("  i_H2  max   %.2f mA/cm^2 at V_RHE = %.3f\n", maximum(iH2), Vs[argmax(iH2)])
@printf("  FE_CO peak  %.1f %% at V_RHE = %.3f\n", FE[kpk], Vs[kpk])
nCO, nH2 = polarization_smoothness(Vs, iCO, iH2)
@printf("  sign changes in di_CO/dV = %d (want 1), di_H2/dV = %d (want 0)\n", nCO, nH2)

stage3_checks(Vs, sols, ctx; pinned = true)
stage_liquid_checks(Vs, sols, ctx)
compare_to_experiment(Vs, iCO, iH2)

files = export_stage("stage3", Vs, sols, ctx)
println("\nExported:"); for f in files; println("  ", f); end

println("\n" * "="^78)
println("STAGE 3 COMPLETE — 🛑 STOP GATE.  Review the plots before Stage 4.")
println("="^78)
