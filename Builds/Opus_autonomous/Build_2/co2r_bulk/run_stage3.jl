# run_stage3.jl — Stage 3: + Stefan-Maxwell gas transport (u_G = 0, p pinned)
#                          + volumetric CO₂ phase transfer R_PT
include(joinpath(@__DIR__, "model.jl"))
using Printf

const N_CL  = N_CL_DEFAULT
const N_GDL = N_GDL_DEFAULT

run3 = make_stage_run(Val(3), N_CL, N_GDL)
print_kinetics_banner(3, run3.cache_dir)
@printf("mesh: N_CL=%d N_GDL=%d ; DOFs=%d\n", N_CL, N_GDL, run3.ly.N_TOT)
@printf("ω_inlet = %s ; a_v_GL·k_GL = %.1f s⁻¹\n",
        string(round.(omega_inlet, sigdigits=4)), a_v_GL*k_GL())

u0 = build_initial_guess(run3.ly, run3.cl, run3.gdl, run3.c_eq,
                         V_RHE_grid[1] - V_RHE_to_SHE_offset)

# ── gas-only bootstrap (§11 Stage 3): all cells at the inlet composition are a
#    trivially converged gas state; the call also triggers JIT compilation of
#    the whole SM chain cheaply before the monolithic Newton.
Fchk = zeros(run3.ly.N_TOT)
res_at_V!(run3, Fchk, u0, V_RHE_grid[1])
gasrows = (_gas_start(run3.ly)+1):(_gas_start(run3.ly)+4*(N_GDL+N_CL))
@printf("gas-only bootstrap: max|F_gas| at inlet composition = %.3e\n",
        maximum(abs, Fchk[gasrows]))
@printf("‖F‖∞ at the anchor initial guess = %.3e\n", maximum(abs, Fchk))

sw = run_sweep!(run3, u0, V_RHE_grid[1], V_RHE_grid[2:end], V_RHE_grid)
s  = export_stage(run3, sw.Vs, sw.us)

@printf("\n=== Stage 3 summary ===\n")
@printf("converged %d / %d voltages, V_RHE ∈ [%.3f, %.3f]\n",
        length(sw.Vs), length(V_RHE_grid), sw.Vs[1], sw.Vs[end])
@printf("max ‖F‖∞ = %.3e ; mean Newton iters = %.2f ; wall = %.1f s\n",
        maximum(sw.resn), sum(sw.iters)/length(sw.iters), sw.wall)
print_physicality(run3, s, sw.Vs, sw.us)

# ── Stage-3 specific gas checks ──
println("\n── Stage 3 gas checks ──")
NK = N_GDL + N_CL
for k in (1, 40, 89)
    u = sw.us[k]; V = sw.Vs[k]
    w_in  = gas_omega(u, run3.ly, 1)
    w_out = gas_omega(u, run3.ly, NK)
    Mn_o  = Mn_from_omega(w_out...)
    p_out = gas_p(u, run3.ly, NK)
    c = unpack_log_conc(u, run3.ly)
    Hloc = sechenov_H_CO2(c[1,N_CL], c[2,N_CL], c[3,N_CL], c[4,N_CL])
    @printf("V=%.3f: ω_CO2 %.4f→%.4f ; ω_CO %.3e ; ω_H2 %.3e ; p=%.2f Pa\n",
            V, w_in[1], w_out[1], w_out[2], w_out[3], p_out)
    @printf("        p_CO2(CL exit)=%.4f atm ; H_eff=%.3f ; H·p=%.3f vs c_CO2(aq)=%.3f mol/m³\n",
            omega_to_x(w_out[1],Mn_o,1)*p_out/101325, Hloc,
            Hloc*omega_to_x(w_out[1],Mn_o,1)*p_out/101325, c[5,N_CL])
end
pmax = maximum(abs.([gas_p(sw.us[end], run3.ly, k) - p_total for k in 1:NK]))
@printf("max |p − p_total| over the CL+GDL (Stage-3 pin) = %.3e Pa\n", pmax)
compare_to_experiment(sw.Vs, s.iCO, s.iH2)
println("\nStage 3 complete — STOP for review.")
