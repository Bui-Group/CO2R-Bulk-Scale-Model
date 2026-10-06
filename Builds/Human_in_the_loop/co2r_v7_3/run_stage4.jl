# ============================================================================
# run_stage4.jl — Stage 4: finalize the Darcy pressure equation. §11 Stage 4
#   Pressure row becomes the true continuity balance rho_u_L - rho_u_R + S_mass*dx
#   with p Dirichlet at the flow field (ghost cell) and dp/dx = 0 at x = L_CL.
#   Species flux now carries the convective term rho*omega_j*u_G.
#   Kinetics still simplified: phi_L = 0, phi_s = V_applied_SHE.
# ============================================================================
include("model.jl")
print_kinetics(4)
println("Cache dir: ", cache_dir(4))
if isdir(cache_dir(4)) && !isempty(readdir(cache_dir(4)))
    println("WARNING: Existing cache found. Verify kinetics match before reusing!")
end
println()

out = run_stage(4)
ctx = out.ctx; ly = ctx.ly; gm = ctx.gm
NG = gm.N; NGDL = ly.N_GDL; N = ly.N_CL

println("\n" * "="^72)
println("Stage 4 physicality checks")

"Recompute the gas face fluxes / Darcy mass flux from a converged state."
function gas_faces(u)
    rhou = zeros(NG+1)
    w_in = (omega_inlet[1], omega_inlet[2], omega_inlet[3], omega_inlet[4])
    _,_,_,_, ru = sm_face_flux(w_in, p_total, gas_omega(u,ly,1), gas_p(u,ly,1),
                               gm.dx_between[1], face_Dbin(gm,1), face_kappa(gm,1), true)
    rhou[1] = ru
    for f in 2:NG
        _,_,_,_, ru = sm_face_flux(gas_omega(u,ly,f-1), gas_p(u,ly,f-1),
                                   gas_omega(u,ly,f),   gas_p(u,ly,f),
                                   gm.dx_between[f], face_Dbin(gm,f), face_kappa(gm,f), true)
        rhou[f] = ru
    end
    return rhou            # rhou[NG+1] = 0 by the no-flow BC
end

println("\nDarcy mass balance:  rho_u(inlet) = -Sum(S_mass * dx)   [§11 Stage 4]")
println("  and the CO2-dissolution split (guide: 71/59/35/11 % to CO2R at -1.200/-1.650/-2.075/-2.500)")
for V in (-1.200, -1.650, -2.075, -2.500)
    k = argmin(abs.([r.V_RHE for r in out.recs] .- V)); r = out.recs[k]; u = r.u
    c = unpack_log_conc(u, ly)
    rco, rh = cell_rates(u, r.V_RHE, ctx)
    Rpt = [begin g=NGDL+ix; wg=gas_omega(u,ly,g); xg=omega_to_x(wg); pg=gas_p(u,ly,g)
                 a_v_GL*k_GL()*(sechenov_H_CO2(view(c,:,ix))*xg[1]*pg/101325 - c[5,ix]) end
           for ix in 1:N]
    dx = ctx.cl.dx
    m_pt = sum(Rpt.*dx)*M_gas[1]; m_co = sum(rco.*dx)*M_gas[2]; m_h2 = sum(rh.*dx)*M_gas[3]
    S_tot = -m_pt + m_co + m_h2
    rhou = gas_faces(u)
    iCO, iH2, itot = compute_currents(u, r.V_RHE, ctx)
    dp_CL  = gas_p(u,ly,NGDL+1) - gas_p(u,ly,NG)
    dp_GDL = gas_p(u,ly,1) - gas_p(u,ly,NGDL)
    @printf("  V=%7.3f i=%5.2f A/cm2 | rho_u(in)=%+.4e  -Sum(S*dx)=%+.4e  err=%.2e | in: R_PT %.3e  out: CO %.3e H2 %.3e | CO2->CO2R %5.1f%% | dp_CL=%7.1f Pa dp_GDL=%6.2f Pa\n",
            r.V_RHE, itot/1e4, rhou[1], -S_tot, abs(rhou[1]+S_tot)/max(abs(rhou[1]),1e-30),
            m_pt, m_co, m_h2, 100*sum(rco.*dx)/sum(Rpt.*dx), dp_CL, dp_GDL)
end

println("\nrho_u conservation through the mesh (max cell imbalance)")
for V in (-1.200, -2.500)
    k = argmin(abs.([r.V_RHE for r in out.recs] .- V)); r = out.recs[k]; u = r.u
    Fchk = zeros(ly.N_TOT); residual!(Fchk, u, r.V_RHE, ctx)
    prows = [gas_idx(ly,g,4) for g in 1:NG]
    @printf("  V=%7.3f  max|continuity residual| = %.3e kg/m^2/s   p(x=L_CL) = %.2f Pa\n",
            r.V_RHE, maximum(abs, Fchk[prows]), gas_p(u,ly,NG))
end

println("\nPressure profile and Darcy velocity")
for V in (-0.300, -1.200, -1.650, -2.075, -2.500)
    k = argmin(abs.([r.V_RHE for r in out.recs] .- V)); r = out.recs[k]; u = r.u
    ps = [gas_p(u,ly,g) for g in 1:NG]
    uG = -(kappa_CL/mu_gas)*(ps[NG]-ps[NGDL+1])/(gm.x_centers[NG]-gm.x_centers[NGDL+1])
    @printf("  V=%7.3f  p: %9.2f (flow field) %9.2f (GDL/CL) %9.2f Pa (x=L_CL) | dp/p = %.4f %% | u_G(CL) = %.3e m/s\n",
            r.V_RHE, ps[1], ps[NGDL+1], ps[NG], 100*(ps[1]-ps[NG])/p_total, uG)
end

println("\nStage 3 -> Stage 4 current comparison (expect near-identical)")
ctx3 = Ctx(3, N, NGDL, ctx.c_eq)
for V in (-0.800, -1.200, -1.650, -2.075, -2.425)
    Vg = V_RHE_grid[argmin(abs.(V_RHE_grid .- V))]
    u3 = load_solution(round(Vg, digits=9), cache_dir(3), cache_prefix(3))
    i3 = compute_currents(u3, Vg, ctx3)
    k = argmin(abs.([r.V_RHE for r in out.recs] .- Vg))
    i4 = compute_currents(out.recs[k].u, Vg, ctx)
    @printf("  V=%7.3f  i_CO: %8.2f -> %8.2f (%+.2f%%)   i_H2: %8.2f -> %8.2f (%+.2f%%)\n",
            Vg, i3[1]/10, i4[1]/10, 100*(i4[1]-i3[1])/i3[1],
            i3[2]/10, i4[2]/10, 100*(i4[2]-i3[2])/i3[2])
end

fe = [begin iCO,iH2,it = compute_currents(r.u, r.V_RHE, ctx); it>0 ? 100*iCO/it : 0.0 end
      for r in out.recs]
kmax = argmax(fe)
@printf("\npeak FE_CO = %.1f %% at V_RHE = %.3f\n", fe[kmax], out.recs[kmax].V_RHE)
println("\n>>> STAGE 4 COMPLETE — generate plots and STOP for human review.")
