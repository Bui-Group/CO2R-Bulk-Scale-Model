# run_stage1.jl — Stage 1: liquid NP diffusion only + Dirichlet electrolyte BC
#                 simplified kinetics (φ_L = 0, φ_s = V_applied_SHE)
include(joinpath(@__DIR__, "model.jl"))
using Printf

const N_CL  = N_CL_DEFAULT
const N_GDL = N_GDL_DEFAULT

run1 = make_stage_run(Val(1), N_CL, N_GDL)
print_kinetics_banner(1, run1.cache_dir)
@printf("mesh: N_CL=%d N_GDL=%d ; DOFs=%d\n", N_CL, N_GDL, run1.ly.N_TOT)

u0 = build_initial_guess(run1.ly, run1.cl, run1.gdl, run1.c_eq,
                         V_RHE_grid[1] - V_RHE_to_SHE_offset)
Fchk = zeros(run1.ly.N_TOT)
res_at_V!(run1, Fchk, u0, V_RHE_grid[1])
@printf("‖F‖∞ at the equilibrium guess (anchor V) = %.3e\n", maximum(abs, Fchk))

sw = run_sweep!(run1, u0, V_RHE_grid[1], V_RHE_grid[2:end], V_RHE_grid)
s  = export_stage(run1, sw.Vs, sw.us)

@printf("\n=== Stage 1 summary ===\n")
@printf("converged %d / %d voltages, V_RHE ∈ [%.3f, %.3f]\n",
        length(sw.Vs), length(V_RHE_grid), sw.Vs[1], sw.Vs[end])
@printf("max ‖F‖∞ over the sweep = %.3e ; mean Newton iters = %.2f ; wall = %.1f s\n",
        maximum(sw.resn), sum(sw.iters)/length(sw.iters), sw.wall)
print_physicality(run1, s, sw.Vs, sw.us)
compare_to_experiment(sw.Vs, s.iCO, s.iH2)
println("\nStage 1 complete — STOP for review.")
