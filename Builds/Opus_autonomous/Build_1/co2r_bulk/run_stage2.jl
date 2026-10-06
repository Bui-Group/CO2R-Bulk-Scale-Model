# run_stage1.jl — Stage 2: + Sherwood-Reynolds Neumann BC at the electrolyte side
include("model.jl")
include("checks.jl")
using Printf

const STAGE_TAG = "stage2"
const PREFIX    = "s2_"

print_kinetics_header("Stage 2 — Sherwood-Reynolds Neumann electrolyte BC", STAGE_TAG)

N_CL = N_CL_DEFAULT; N_GDL = N_GDL_DEFAULT
dm = make_domain(N_CL, N_GDL)
ly = Layout(2, N_CL, N_GDL)
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

println("\n--- §11 Stage 2 physicality checks ---")
res = report_polarization(Vout, Uout, ly, dm)
report_liquid(Vout, Uout, ly, dm)
mb = [abs(co2_mass_balance(Uout[k], ly, dm, Vout[k])) for k in 1:length(Vout)]
@printf("  CO2 mass-balance residual: max %.3e mol/m2/s (tol 1e-6)\n", maximum(mb))
println()
compare_to_experiment(Vout, Uout, ly, dm)

export_polarization(STAGE_TAG, Vout, Uout, ly, dm)
export_profiles(STAGE_TAG, Vout, Uout, ly, dm)
println("\nExported polarization.csv and profiles.bin to output_stage2/")
println("STAGE 2 COMPLETE — STOP GATE")

# ── Stage-2 specific checks (§11 Stage 2)
println("\n--- Stage 2 specific checks ---")
@printf("  Biot numbers Bi_i = k_MT_i*L_CL/D_eff_i: %s (target O(0.1-10))\n",
        string(round.([k_MT[i]*L_CL/D_eff[i] for i in 1:5], sigdigits=4)))
let k = argmin(Vout), u = Uout[k]
    c = unpack_log_conc(u, ly.N_CL)
    @printf("  At V_RHE=%.3f: c_CO2(L_CL) = %.4g mol/m3 (should approach 0)\n", Vout[k], c[5,end])
    @printf("                 c_OH(L_CL)  = %.4g mol/m3 (bulk eq %.3e; must be well above)\n",
            c[2,end], dm.c_eq[2])
    names = ["H+","OH-","HCO3-","CO3^2-","CO2(aq)"]
    for i in 1:5
        J = k_MT[i]*(c[i,end] - dm.c_eq[i])
        @printf("    J_%-8s at EL face = %+.4e mol/m2/s  (%s)\n", names[i], J,
                J > 0 ? "leaving CL" : "entering CL")
    end
end
