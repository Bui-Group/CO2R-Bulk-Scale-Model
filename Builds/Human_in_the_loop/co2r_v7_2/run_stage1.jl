# =============================================================================
# run_stage1.jl — Stage 1: liquid Nernst-Planck, DIFFUSION ONLY, Dirichlet
# electrolyte BC, simplified kinetics (phi_s = V_applied_SHE, phi_L = 0).
#
# Solver: the §9.5 production stack via the unified §9.5.5 driver — affine-
# covariant Newton + persistent sparse LU + ForwardDiff AD Jacobian,
# self-anchoring at -0.300 V RHE from the exact equilibrium guess (§9.6).
#
# STOPS after generating plots — does NOT chain to Stage 2 (§13).
# =============================================================================
include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "export.jl"))
include(joinpath(@__DIR__, "diagnostics.jl"))
using Printf

const STAGE      = 1
const CACHE_DIR  = cache_dir_for(STAGE)
const PREFIX     = cache_prefix_for(STAGE)

print_stage_header(STAGE, CACHE_DIR;
    extra = "Physics: liquid diffusion + buffer + MHC/BV kinetics; Dirichlet c_eq at x=L_CL")

ctx = make_ctx(STAGE)
@printf("DOFs: %d (5 x N_CL)\n", ctx.ly.N_TOT)
@printf("Bulk equilibrium: pH=%.4f  c=[%s]\n",
        -log10(ctx.c_eq[1]/1000), join([@sprintf("%.4e", x) for x in ctx.c_eq], ", "))
@printf("c_CO2 at the GDL face (van 't Hoff only, §7.1) = %.4f mol/m^3\n\n", c_CO2_henry_stage12)

# ── §9.5.5 driver: anchor at the mildest grid point, then continue down ──────
V_chain = collect(V_RHE_grid)
emit    = fill(true, length(V_chain))

res = run_continuation(ctx, make_anchor_guess(ctx, V_chain[1]), V_chain, emit;
                       cache_dir = CACHE_DIR, prefix = PREFIX)

@printf("\nSweep complete: %d/%d voltages converged and cached\n", length(res.voltages), length(V_chain))
@printf("  chain points     : %d\n", res.n_chain)
@printf("  Newton iters/step: %.2f\n", res.iters_per_step)
@printf("  max ||F||inf     : %.3e   (assertion tol %.1e)\n", maximum(res.resids), F_INF_TOL)
@printf("  LU: %d symbolic / %d numeric refreshes (nnz-guard rebuilds %d, lu!-pattern-catch rebuilds %d)\n",
        res.lu_symbolic, res.lu_numeric, res.lu_nnz_change, res.lu_pattern_catch)
@printf("  sub-step recoveries triggered: %d\n", res.n_substep)
@printf("  wall time: %.2f s\n", res.wall)

# ── Currents + physicality checks ────────────────────────────────────────────
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
@printf("  FE_CO peak  %.1f %% at V_RHE = %.3f   (§11 Stage 1 expects a ~60-70 %% peak pre-fit)\n", FE[kpk], Vs[kpk])
@printf("  FE_CO at the mildest voltage: %.2f %%   (§11 Stage 1 expects <~15 %% near onset)\n", FE[1])
nCO, nH2 = polarization_smoothness(Vs, iCO, iH2)
@printf("  sign changes in di_CO/dV = %d (want 1), di_H2/dV = %d (want 0)\n", nCO, nH2)
@printf("  i_H2 strictly monotone in |V_RHE|: %s\n", all(diff(iH2) .> 0))

stage_liquid_checks(Vs, sols, ctx)
compare_to_experiment(Vs, iCO, iH2)

# ── Export for plotting ──────────────────────────────────────────────────────
files = export_stage("stage1", Vs, sols, ctx)
println("\nExported:"); for f in files; println("  ", f); end

println("\n" * "="^78)
println("STAGE 1 COMPLETE — 🛑 STOP GATE.  Review the plots before Stage 2.")
println("="^78)
