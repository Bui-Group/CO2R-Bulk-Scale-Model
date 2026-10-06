# =============================================================================
# run_stage2.jl — Stage 2: replace the Dirichlet at x = L_CL with the
# Sherwood-Reynolds convective Neumann BC.  No structural change to the DOF
# count, residual layout or solver — still 5 x N_CL DOFs, still pure diffusion
# on the GDL side, still simplified kinetics (phi_s = V_applied_SHE, phi_L = 0).
#
# Solver: §9.5 stack with the Stages 1-4 driver (§9.5.5).
# STOPS after generating plots.
# =============================================================================
include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "export.jl"))
include(joinpath(@__DIR__, "diagnostics.jl"))
using Printf

const STAGE     = 2
const CACHE_DIR = cache_dir_for(STAGE)
const PREFIX    = cache_prefix_for(STAGE)

print_stage_header(STAGE, CACHE_DIR;
    extra = "Physics: Stage 1 + Sherwood-Reynolds Neumann at x = L_CL (k_MT from D_w, §7.2)")

ctx = make_ctx(STAGE)
@printf("DOFs: %d\n", ctx.ly.N_TOT)
@printf("Re = %.1f, u_elec = %.4f m/s, L_elec = %.4f m\n", Re_elec, u_elec, L_elec)
@printf("k_MT (from bare-water D_w, NOT D_eff) = [%s] m/s\n\n",
        join([@sprintf("%.4e", x) for x in k_MT], ", "))

V_chain = collect(V_RHE_grid)
emit    = fill(true, length(V_chain))

# §9.5.5 anchor: the exact equilibrium guess is self-anchoring.  §11 Stage 2
# offers the Stage-1 cache as a warm start; §10.5 calls it "an optional
# convenience for the first voltage, not a requirement" — use it at the anchor
# when present, and fall back to the constructed equilibrium guess otherwise.
u0 = load_solution(cache_dir_for(1), cache_prefix_for(1), V_chain[1])
if u0 === nothing || length(u0) != ctx.ly.N_TOT
    u0 = make_anchor_guess(ctx, V_chain[1])
    println("Anchor guess: constructed bulk equilibrium (§9.6)")
else
    println("Anchor guess: warm-started from the Stage-1 cache at V_RHE = $(V_chain[1])")
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
@printf("  i_H2 strictly monotone in |V_RHE|: %s\n", all(diff(iH2) .> 0))

stage2_checks(Vs, sols, ctx)
stage_liquid_checks(Vs, sols, ctx)
compare_to_experiment(Vs, iCO, iH2)

# ── Overlay vs Stage 1 (§11 Stage 2 stop-gate asks for this) ────────────────
s1 = joinpath(outdir_for("stage1"), "polarization.csv")
if isfile(s1)
    println("\n--- Stage 2 vs Stage 1 (effect of the finite electrolyte mass transfer) ---")
    println("   V_RHE      i_CO  S1 ->  S2   (delta)        i_H2  S1 ->  S2   (delta)")
    rows = readlines(s1)[2:end]
    d1 = Dict{Float64,Tuple{Float64,Float64}}()
    for r in rows
        p = split(r, ",")
        d1[parse(Float64, p[1])] = (parse(Float64, p[3]), parse(Float64, p[4]))
    end
    for k in (1, argmin(abs.(Vs .- (-1.200))), argmin(abs.(Vs .- (-1.650))), length(Vs))
        V = Vs[k]
        haskey(d1, V) || continue
        a, b = d1[V]
        @printf("  %7.3f   %9.3f -> %9.3f (%+7.2f%%)   %9.3f -> %9.3f (%+7.2f%%)\n",
                V, a, iCO[k], 100*(iCO[k]-a)/max(a,1e-12), b, iH2[k], 100*(iH2[k]-b)/max(b,1e-12))
    end
end

files = export_stage("stage2", Vs, sols, ctx)
println("\nExported:"); for f in files; println("  ", f); end

println("\n" * "="^78)
println("STAGE 2 COMPLETE — 🛑 STOP GATE.  Review the plots before Stage 3.")
println("="^78)
