# =============================================================================
# run_stage4.jl — Stage 4: finalize the pressure DOF as a true Darcy momentum
# balance.  The pressure row stops being pinned and becomes the total-mass
# continuity balance; the species flux picks up the convective term
# rho*omega_j*u_G with u_G = -(kappa/mu) dp/dx; the GDL/CL interface uses the
# harmonic-mean permeability.  Kinetics still simplified.
#
# Solver: §9.5 stack with the Stages 1-4 driver (§9.5.5). 1200 DOFs.
# STOPS after generating plots.  The kinetic fit (§17) comes next.
# =============================================================================
include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "export.jl"))
include(joinpath(@__DIR__, "diagnostics.jl"))
using Printf

const STAGE     = 4
const CACHE_DIR = cache_dir_for(STAGE)
const PREFIX    = cache_prefix_for(STAGE)

print_stage_header(STAGE, CACHE_DIR;
    extra = "Physics: Stage 3 + Darcy pressure equation + convective species flux")

ctx = make_ctx(STAGE)
@printf("DOFs: %d (unchanged from Stage 3)\n", ctx.ly.N_TOT)
@printf("kappa_GDL = %.4e m^2, kappa_CL = %.4e m^2, harmonic mean at the interface = %.4e m^2\n",
        kappa_GDL, kappa_CL, ctx.gg.kappa[N_GDL+1])
@printf("mu_gas = %.3e Pa s\n\n", mu_gas)

V_chain = collect(V_RHE_grid)
emit    = fill(true, length(V_chain))

# §11 Stage 4: warm-start direct from the Stage-3 cache (no new DOFs).
u0 = load_solution(cache_dir_for(3), cache_prefix_for(3), V_chain[1])
if u0 === nothing || length(u0) != ctx.ly.N_TOT
    u0 = make_anchor_guess(ctx, V_chain[1])
    println("Anchor guess: constructed equilibrium liquid + inlet gas at p_total")
else
    println("Anchor guess: warm-started direct from the Stage-3 cache at V_RHE = $(V_chain[1])")
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

stage3_checks(Vs, sols, ctx; pinned = false)
stage4_checks(Vs, sols, ctx, iCO, iH2)
stage_liquid_checks(Vs, sols, ctx)
compare_to_experiment(Vs, iCO, iH2)

# ── §11 Stage 4: "currents should be almost identical to Stage 3" ───────────
function compare_to_stage3(Vs, iCO, iH2)
    s3 = joinpath(outdir_for("stage3"), "polarization.csv")
    isfile(s3) || return
    println("\n--- Stage 4 vs Stage 3 (Darcy is a small correction) ---")
    d3 = Dict{Float64,Tuple{Float64,Float64}}()
    for r in readlines(s3)[2:end]
        p = split(r, ",")
        d3[parse(Float64, p[1])] = (parse(Float64, p[3]), parse(Float64, p[4]))
    end
    dmaxCO = 0.0; dmaxH2 = 0.0
    for (k, V) in enumerate(Vs)
        haskey(d3, V) || continue
        a, b = d3[V]
        dmaxCO = max(dmaxCO, abs(iCO[k]-a)/max(a, 1e-12))
        dmaxH2 = max(dmaxH2, abs(iH2[k]-b)/max(b, 1e-12))
    end
    @printf("  max relative change over the 89 voltages: i_CO %.3f %%, i_H2 %.3f %%\n",
            100*dmaxCO, 100*dmaxH2)
    for k in (argmin(abs.(Vs .- (-1.200))), argmin(abs.(Vs .- (-2.500))))
        V = Vs[k]; haskey(d3, V) || continue
        a, b = d3[V]
        @printf("   V_RHE=%7.3f : i_CO %9.3f -> %9.3f (%+.3f%%)   i_H2 %9.3f -> %9.3f (%+.3f%%)\n",
                V, a, iCO[k], 100*(iCO[k]-a)/a, b, iH2[k], 100*(iH2[k]-b)/b)
    end
end
compare_to_stage3(Vs, iCO, iH2)

files = export_stage("stage4", Vs, sols, ctx)
println("\nExported:"); for f in files; println("  ", f); end

println("\n" * "="^78)
println("STAGE 4 COMPLETE — 🛑 STOP GATE.  Review the plots before the Fit step.")
println("="^78)
