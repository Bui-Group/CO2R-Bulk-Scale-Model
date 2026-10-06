# ============================================================================
# run_stage3.jl — Stage 3: Stefan-Maxwell gas transport (CO2/CO/H2/H2O) +
# volumetric phase transfer R_PT + Sechenov Henry BC.  §11 Stage 3 / §6 / §7.3-7.4
#   u_G = 0 (no Darcy convection yet); the pressure row is PINNED: F = p - p_total.
#   Kinetics still simplified: phi_L = 0, phi_s = V_applied_SHE.
# ============================================================================
include("model.jl")
print_kinetics(3)
println("Cache dir: ", cache_dir(3))
if isdir(cache_dir(3)) && !isempty(readdir(cache_dir(3)))
    println("WARNING: Existing cache found. Verify kinetics match before reusing!")
end

# ── §11 Stage 3 bootstrap step 1: gas-only warm-up.
# Its purpose is to JIT-compile the sm_residual/helper chain cheaply before the
# monolithic Newton (§14 "JIT compilation slow"). With the §9.5.5 anchor recipe
# the gas cells start AT the inlet composition, so this state is the trivial
# gas-only solution; we evaluate the residual + one AD Jacobian to force the
# compile, then hand the same state to the monolithic anchor.
println("\nBootstrap step 1: gas-only warm-up (JIT)")
let
    eq = solve_buffer_equilibrium(1000.0, 1000.0)
    c0 = [eq.H, eq.OH, eq.HCO3, eq.CO3, eq.CO2]
    c3 = Ctx(3, N_CL_DEFAULT, N_GDL_DEFAULT, c0)
    u0 = make_u0(c3); n = c3.ly.N_TOT
    w0 = SolveWS(c3, u0)
    w0.rf.V = -0.300
    Fg = zeros(n); w0.rf(Fg, u0)
    gasrows = [gas_idx(c3.ly, g, j) for g in 1:(c3.ly.N_GDL+c3.ly.N_CL) for j in 1:4]
    @printf("  gas-only ||F||inf over gas rows at inlet composition = %.3e\n",
            maximum(abs, Fg[gasrows]))
    t = @elapsed ForwardDiff.jacobian!(w0.Jd, w0.rf, zeros(n), u0, w0.cfg)
    @printf("  first AD Jacobian (incl. JIT) = %.2f s\n", t)
end

println("\nBootstrap step 2: monolithic Newton, anchor + continuation")
out = run_stage(3)
ctx = out.ctx; ly = ctx.ly; gm = ctx.gm

# ── §11 Stage 3 physicality checks ─────────────────────────────────────────
println("\n" * "="^72)
println("Stage 3 physicality checks")
NG = gm.N; NGDL = ly.N_GDL

function report3(V)
    k = argmin(abs.([r.V_RHE for r in out.recs] .- V))
    r = out.recs[k]; u = r.u
    c = unpack_log_conc(u, ly)
    w_in  = gas_omega(u, ly, 1)
    w_if  = gas_omega(u, ly, NGDL+1)      # first CL gas cell (GDL/CL interface)
    w_out = gas_omega(u, ly, NG)          # last CL gas cell (electrolyte side)
    x_out = omega_to_x(w_out)
    pmax = maximum(abs(gas_p(u, ly, g) - p_total) for g in 1:NG)
    iCO, iH2, itot = compute_currents(u, r.V_RHE, ctx)
    @printf("  V=%7.3f  w_CO2: %.4f(GDL in) %.4f(if) %.4f(CL out) | w_CO %.2e w_H2 %.2e (CL out) | max|p-p0|=%.1e Pa | c_CO2(aq) %.3f->%.3f | i_tot=%7.2f\n",
            r.V_RHE, w_in[1], w_if[1], w_out[1], w_out[2], w_out[3], pmax,
            c[5,1], c[5,end], itot/10)
end
for V in (-0.300, -0.800, -1.200, -1.650, -2.075, -2.500); report3(V); end

# Henry's-law consistency at the gas-liquid interface + CO2 mass balance (§12.6)
println("\nHenry's-law + CO2 mass balance (§7.3, §12.6)")
for V in (-0.800, -1.650, -2.500)
    k = argmin(abs.([r.V_RHE for r in out.recs] .- V))
    r = out.recs[k]; u = r.u; N = ly.N_CL
    c = unpack_log_conc(u, ly)
    w1 = gas_omega(u, ly, NGDL+1); x1 = omega_to_x(w1); p1 = gas_p(u, ly, NGDL+1)
    pCO2 = x1[1]*p1/101325
    Hl   = sechenov_H_CO2(view(c,:,1))
    cbc  = Hl*pCO2
    # CO2(aq) balance: J_in(x=0) - J_out(x=L) + int R_PT - int R_CO2R = 0
    Jin  = diffusion_flux(cbc, c[5,1], D_eff[5], ctx.cl.dx_between[1])
    Jout = -k_MT[5]*(ctx.c_eq[5] - c[5,N])
    rco, rh = cell_rates(u, r.V_RHE, ctx)
    Rpt = zeros(N)
    for ix in 1:N
        g = NGDL+ix; wg = gas_omega(u,ly,g); xg = omega_to_x(wg); pg = gas_p(u,ly,g)
        Rpt[ix] = a_v_GL*k_GL()*(sechenov_H_CO2(view(c,:,ix))*xg[1]*pg/101325 - c[5,ix])
    end
    # buffer consumption of CO2 must be included in the balance
    bufCO2 = sum(eps_I*buffer_sources(view(c,:,ix))[5]*ctx.cl.dx[ix] for ix in 1:N)
    bal = Jin - Jout + sum(Rpt.*ctx.cl.dx) - sum(rco.*ctx.cl.dx) + bufCO2
    @printf("  V=%7.3f  p_CO2(if)=%.4f atm  H_eff=%.3f  c_CO2_bc=%.4f  c_CO2(cell1)=%.4f | CO2 balance = %+.3e mol/m^2/s\n",
            r.V_RHE, pCO2, Hl, cbc, c[5,1], bal)
end

println("\nStage 2 -> Stage 3 current comparison")
ctx2 = Ctx(2, ly.N_CL, ly.N_GDL, ctx.c_eq)
for V in (-0.800, -1.200, -1.650, -2.075, -2.425)
    Vg = V_RHE_grid[argmin(abs.(V_RHE_grid .- V))]
    u2 = load_solution(round(Vg, digits=9), cache_dir(2), cache_prefix(2))
    i2 = compute_currents(u2, Vg, ctx2)
    k = argmin(abs.([r.V_RHE for r in out.recs] .- Vg))
    i3 = compute_currents(out.recs[k].u, Vg, ctx)
    @printf("  V=%7.3f  i_CO: %8.2f -> %8.2f   i_H2: %8.2f -> %8.2f  mA/cm2\n",
            Vg, i2[1]/10, i3[1]/10, i2[2]/10, i3[2]/10)
end

fe = [begin iCO,iH2,it = compute_currents(r.u, r.V_RHE, ctx); it>0 ? 100*iCO/it : 0.0 end
      for r in out.recs]
kmax = argmax(fe)
@printf("\npeak FE_CO = %.1f %% at V_RHE = %.3f\n", fe[kmax], out.recs[kmax].V_RHE)
println("\n>>> STAGE 3 COMPLETE — generate plots and STOP for human review.")
