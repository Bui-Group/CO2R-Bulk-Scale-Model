# run_stage1.jl — Stage 1: liquid NP diffusion only + Dirichlet electrolyte BC
include("model.jl")
include("checks.jl")
using Printf

const STAGE_TAG = "stage1"
const PREFIX    = "s1_"

print_kinetics_header("Stage 1 — liquid diffusion, Dirichlet electrolyte BC", STAGE_TAG)

N_CL = N_CL_DEFAULT; N_GDL = N_GDL_DEFAULT
dm = make_domain(N_CL, N_GDL)
ly = Layout(1, N_CL, N_GDL)
@printf("Mesh: N_CL=%d, N_GDL=%d  ->  %d DOFs\n", N_CL, N_GDL, ly.N_TOT)
ctx = make_ctx(ly, dm)

u0 = make_u0(ly, dm)
Vs = copy(V_RHE_grid)
emit = fill(true, length(Vs))

t = @elapsed Vout, Uout, st = continuation_sweep(ctx, u0, Vs; emit=emit,
        stage_tag=STAGE_TAG, prefix=PREFIX, quiet=true)

println()
@printf("Sweep complete: %d/%d voltages converged in %.2f s\n", length(Vout), length(Vs), t)
@printf("  Newton iterations/step: mean %.2f, max %d\n",
        sum(st.iters)/length(st.iters), maximum(st.iters))
@printf("  max ||F||inf over the sweep: %.3e (assertion tol %.0e)\n",
        maximum(st.resids), F_INF_TOL)
@printf("  sub-step recoveries fired: %d\n", st.n_substep)
@printf("  V_RHE range: [%.3f, %.3f]\n", maximum(Vout), minimum(Vout))

println("\n--- §11 Stage 1 physicality checks ---")
res = report_polarization(Vout, Uout, ly, dm)
report_liquid(Vout, Uout, ly, dm)
mb = [abs(co2_mass_balance(Uout[k], ly, dm, Vout[k])) for k in 1:length(Vout)]
@printf("  CO2 mass-balance residual: max %.3e mol/m2/s (tol 1e-6)\n", maximum(mb))
println()
compare_to_experiment(Vout, Uout, ly, dm)

export_polarization(STAGE_TAG, Vout, Uout, ly, dm)
export_profiles(STAGE_TAG, Vout, Uout, ly, dm)
println("\nExported polarization.csv and profiles.bin to output_stage1/")
println("STAGE 1 COMPLETE — STOP GATE")
