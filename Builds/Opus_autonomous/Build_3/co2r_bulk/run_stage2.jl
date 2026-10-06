# run_stage2.jl — Stage 2: Sherwood-Reynolds Neumann BC at the electrolyte (§11 Stage 2)
include("common.jl")

const CACHE_DIR = joinpath(@__DIR__, "output_stage2", "cache")
const PREFIX    = "s2_"

m = build_model(2)
print_kinetics_header(2, CACHE_DIR)
@printf("mesh: N_CL = %d, DOFs = %d\n", m.cl.N, m.ly.N_TOT)
@printf("Biot numbers k_MT·L_CL/D_eff: %s\n\n",
        string(round.([k_MT[i]*L_CL/D_eff[i] for i in 1:5], digits=3)))

res_at_V!(F, u, V) = residual_stage12!(F, u, V, m.cl, m.c_eq, false)

# Self-anchoring (§9.5.5); the Stage-1 cache is an optional warm start only.
u0 = u0_stage(m.ly, m.c_eq)
chain = copy(V_RHE_grid); emit = trues(length(chain))

t0 = time()
Vs, Us, rows, plu = run_continuation(u0, chain, emit, res_at_V!;
        blocks = weight_blocks(m.ly), n_log = 5*m.cl.N,
        cache_dir = CACHE_DIR, prefix = PREFIX, verbose = false)
@printf("wall time: %.1f s\n", time() - t0)
export_all("stage2", Vs, Us, m.ly, m.cl, m.gm, rows)

@printf("\n=== Stage 2 summary ===\n")
@printf("converged %d/%d, mean iters = %.2f, max ||F||inf = %.3e\n",
        length(Vs), length(chain), sum(r.iters for r in rows)/length(rows),
        maximum(r.Finf for r in rows))

# ── §11 Stage 2 physicality checks
println("\n-- electrolyte-face fluxes  J_i,right = k_MT (c_i(L) − c_eq,i) --")
names = ["H+","OH-","HCO3-","CO3^2-","CO2(aq)"]
for V in (-1.200, -2.500)
    k = argmin(abs.(Vs .- V)); u = Us[k]
    c = unpack_log_conc(u, m.cl.N)
    @printf("V_RHE = %.3f\n", Vs[k])
    for i in 1:5
        J = -k_MT[i]*(m.c_eq[i] - c[i,end])
        @printf("   %-8s c(L)=%.4e  c_eq=%.4e  J=%+.4e mol/m²/s (%s)\n",
                names[i], c[i,end], m.c_eq[i], J, J > 0 ? "leaving CL" : "entering CL")
    end
end
for V in (-2.500,)
    k = argmin(abs.(Vs .- V)); c = unpack_log_conc(Us[k], m.cl.N)
    @printf("c_CO2(L_CL) = %.4e mol/m³ (should approach 0 at high current)\n", c[5,end])
    @printf("c_OH(L_CL)  = %.4e mol/m³ vs c_eq_OH = %.4e (should sit well above)\n",
            c[2,end], m.c_eq[2])
end

# ── overlay comparison with Stage 1
println("\n-- Stage 1 vs Stage 2 currents --")
for V in (-0.816, -1.205, -1.644, -2.079, -2.413)
    k = argmin(abs.(Vs .- V))
    u2 = Us[k]; u1 = load_solution(joinpath(@__DIR__,"output_stage1","cache"), "s1_", Vs[k])
    V_SHE = Vs[k] - V_RHE_to_SHE_offset
    a = compute_currents(u2, V_SHE, m.cl); b = compute_currents(u1, V_SHE, m.cl)
    @printf("V_RHE=%.3f  S1: i_CO=%8.2f i_H2=%8.2f | S2: i_CO=%8.2f i_H2=%8.2f  mA/cm²\n",
            Vs[k], b[1]/10, b[2]/10, a[1]/10, a[2]/10)
end
println("\nSTAGE 2 COMPLETE.")
