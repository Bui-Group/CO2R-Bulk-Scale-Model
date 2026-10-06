# run_stage1.jl — Stage 4: + Darcy pressure finalized (convection + continuity)
include("model.jl")
include("checks.jl")
using Printf

const STAGE_TAG = "stage4"
const PREFIX    = "s4_"

print_kinetics_header("Stage 4 — Darcy pressure finalized", STAGE_TAG)

N_CL = N_CL_DEFAULT; N_GDL = N_GDL_DEFAULT
dm = make_domain(N_CL, N_GDL)
ly = Layout(4, N_CL, N_GDL)
@printf("Mesh: N_CL=%d, N_GDL=%d  ->  %d DOFs\n", N_CL, N_GDL, ly.N_TOT)
ctx = make_ctx(ly, dm)

u0 = make_u0(ly, dm)
# §11 Stage 3 bootstrap step 1: cheap gas-only evaluation to trigger JIT before
# the monolithic Newton (the compile cost, not the solve, is what this buys).
let Fw = zeros(ly.N_TOT)
    t = @elapsed ctx.res_at_V!(Fw, u0, -0.300)
    @printf("JIT warm-up: residual %.3f s, ||F||inf at the anchor guess = %.3e\n",
            t, maximum(abs.(Fw)))
    t = @elapsed ctx.jacfun!(ctx.Jdense, u0)
    @printf("JIT warm-up: first AD Jacobian %.3f s\n", t); flush(stdout)
end

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

println("\n--- §11 Stage 4 physicality checks ---")
res = report_polarization(Vout, Uout, ly, dm)
report_liquid(Vout, Uout, ly, dm)
mb = [abs(co2_mass_balance(Uout[k], ly, dm, Vout[k])) for k in 1:length(Vout)]
@printf("  CO2 mass-balance residual: max %.3e mol/m2/s (tol 1e-6)\n", maximum(mb))
println()
compare_to_experiment(Vout, Uout, ly, dm)

export_polarization(STAGE_TAG, Vout, Uout, ly, dm)
export_profiles(STAGE_TAG, Vout, Uout, ly, dm)
println("\nExported polarization.csv and profiles.bin to output_stage4/")

# ── Stage-3 specific checks (§11 Stage 3)
println("\n--- Stage 4 gas-phase checks ---")
let k = argmin(Vout), u = Uout[k]
    w0 = gas_omega(u, ly, 1); wI = gas_omega(u, ly, ly.N_GDL)
    wE = gas_omega(u, ly, ly.N_G)
    @printf("  At V_RHE=%.3f:\n", Vout[k])
    @printf("    omega_CO2: GDL inlet %.5f -> GDL/CL %.5f -> CL exit %.5f\n", w0[1], wI[1], wE[1])
    @printf("    omega_CO : GDL inlet %.3e -> CL exit %.3e (should rise)\n", w0[2], wE[2])
    @printf("    omega_H2 : GDL inlet %.3e -> CL exit %.3e (should rise)\n", w0[3], wE[3])
    x0 = omega_to_x(w0[1],w0[2],w0[3],w0[4]); xE = omega_to_x(wE[1],wE[2],wE[3],wE[4])
    @printf("    x_CO2    : %.4f -> %.4f  (inlet mole fraction 0.88)\n", x0[1], xE[1])
    pv = [gas_p(u, ly, kk) for kk in 1:ly.N_G]
    @printf("    pressure : [%.2f, %.2f] Pa\n", minimum(pv), maximum(pv))
    # Henry consistency at the GDL/CL boundary
    c = unpack_log_conc(u, ly.N_CL)
    wcl1 = gas_omega(u, ly, ly.N_GDL+1); pcl1 = gas_p(u, ly, ly.N_GDL+1)
    xcl1 = omega_to_x(wcl1[1],wcl1[2],wcl1[3],wcl1[4])
    Hloc = sechenov_H_CO2(view(c,:,1))
    @printf("    Henry BC : H_eff=%.3f mol/(m3 atm), p_CO2=%.4f atm -> c_CO2_eq=%.4f vs c_CO2(cell1)=%.4f\n",
            Hloc, xcl1[1]*pcl1/101325, Hloc*xcl1[1]*pcl1/101325, c[5,1])
    RPT = sum(a_v_GL*k_GL()*(sechenov_H_CO2(view(c,:,ix)) *
              (omega_to_x(gas_omega(u,ly,ly.N_GDL+ix)...)[1]*gas_p(u,ly,ly.N_GDL+ix)/101325)
              - c[5,ix]) * dm.cl.dx[ix] for ix in 1:ly.N_CL)
    @printf("    integral R_PT = %.4e mol/m2/s\n", RPT)
end

# ── Stage-4 specific checks (§11 Stage 4)
println("\n--- Stage 4 Darcy checks ---")
@printf("  %8s %10s %12s %12s %14s %12s\n", "V_RHE","i_tot", "dp_CL[Pa]","dp_GDL[Pa]","rho_u_in","massbal_rel")
for k in 1:length(Vout)
    (k % 11 == 1 || k == length(Vout)) || continue
    u = Uout[k]; V = Vout[k]
    V_SHE = V - V_RHE_to_SHE_offset
    Rco, Rher, c = compute_rates(u, ly, V_SHE)
    pv = [gas_p(u, ly, kk) for kk in 1:ly.N_G]
    dp_gdl = pv[1] - pv[ly.N_GDL]
    dp_cl  = pv[ly.N_GDL+1] - pv[ly.N_G]
    w1 = gas_omega(u, ly, 1)
    Mn1 = Mn_from_omega(w1[1],w1[2],w1[3],w1[4])
    Mng = Mn_from_omega(omega_inlet[1],omega_inlet[2],omega_inlet[3],omega_inlet[4])
    rho_f = 0.5*(pv[1]+p_total)*0.5*(Mn1+Mng)/(R_gas*T)
    uG = -(kappa_GDL/mu_gas)*(pv[1]-p_total)/dm.dxcc_g[1]
    rou_in = rho_f*uG
    Smass = 0.0
    for ix in 1:ly.N_CL
        wk = gas_omega(u, ly, ly.N_GDL+ix); pk = gas_p(u, ly, ly.N_GDL+ix)
        xk = omega_to_x(wk[1],wk[2],wk[3],wk[4])
        rpt = a_v_GL*k_GL()*(sechenov_H_CO2(view(c,:,ix))*xk[1]*pk/101325 - c[5,ix])
        Smass += (-rpt*M_gas[1] + Rco[ix]*M_gas[2] + Rher[ix]*M_gas[3])*dm.cl.dx[ix]
    end
    iCO,iH2,it = compute_currents(u, ly, dm, V)
    rel = abs(rou_in + Smass)/max(abs(rou_in),1e-30)
    @printf("  %8.3f %10.2f %12.2f %12.4f %14.4e %12.3e\n",
            V, it/10, dp_cl, dp_gdl, rou_in, rel)
end
let u = Uout[argmin(Vout)]
    pv = [gas_p(u, ly, kk) for kk in 1:ly.N_G]
    worst = 0.0
    for f in 2:ly.N_G
        kap = face_kappa(dm, f)
        uG = -(kap/mu_gas)*(pv[f]-pv[f-1])/dm.dxcc_g[f]
        lhs = (pv[f]-pv[f-1])/dm.dxcc_g[f]
        rhs = -mu_gas*uG/kap
        worst = max(worst, abs(lhs-rhs)/max(abs(lhs),1e-30))
    end
    @printf("  Darcy law |dp/dx + mu*u_G/kappa| max rel error: %.3e\n", worst)
end
println("STAGE 4 COMPLETE — STOP GATE")
