# ============================================================================
# run_stage5.jl — Stage 5: phi_L (NP migration, monolithic DOF) + phi_s
# (solid Ohm's law on GDL+CL), with the FITTED kinetics from run_fit.jl.
#
# §9.5.5 driver: anchor ONCE at V_c = 0 from a guess constructed from physics
# (equilibrium liquid + inlet gas + phi_s = phi_L = 0), then tangent-predictor
# continuation in -0.025 V steps down the pre-anchor ramp and then the 89
# output voltages.  NO cache seeding (§11 Stage 5 gotcha #0).
# ============================================================================
include("model.jl")

# §11 Stage 5 gotcha #14 — fitted kinetics MUST be loaded before the anchor.
fitted = load_fitted_kinetics!()
print_kinetics(5)
@printf("(fitted values loaded from output_fit/fitted_params.txt)\n")
println("Cache dir: ", cache_dir(5))
if isdir(cache_dir(5)) && !isempty(readdir(cache_dir(5)))
    println("WARNING: Existing cache found. Verify kinetics match before reusing!")
end

# ── §9.5.5 V_c chain: V_c = 0 anchor -> pre-ramp -> the 89 output points ────
# V_c is V vs SHE, so the anchor sits at V_RHE = +V_RHE_to_SHE_offset.
const V_ANCHOR_RHE = V_RHE_to_SHE_offset          # V_c = 0
const N_RAMP = Int(ceil((V_ANCHOR_RHE - V_RHE_grid[1]) / 0.025))   # 33 steps <= 25 mV
ramp = [V_ANCHOR_RHE + k * (V_RHE_grid[1] - V_ANCHOR_RHE) / N_RAMP for k in 0:N_RAMP]
ramp[end] = V_RHE_grid[1]
V_chain = vcat(ramp, V_RHE_grid[2:end])
@printf("\nV_c chain: anchor at V_RHE=%.5f (V_c=0), %d pre-anchor ramp steps of %.6f V,\n",
        V_ANCHOR_RHE, N_RAMP, (V_RHE_grid[1] - V_ANCHOR_RHE)/N_RAMP)
@printf("           then the 89 output voltages — %d chain points total.\n\n", length(V_chain))

out = run_stage(5; V_chain = V_chain, emit_set = copy(V_RHE_grid), verbose = false)
ctx = out.ctx; ly = ctx.ly; gm = ctx.gm
N = ly.N_CL; NGDL = ly.N_GDL

println("\n" * "="^72)
println("Stage 5 physicality checks (§11) and §9.5.8 checklist diagnostics")

function s5row(V)
    k = argmin(abs.([r.V_RHE for r in out.recs] .- V)); r = out.recs[k]; u = r.u
    d = stage5_diagnostics(u, r.V_RHE, ctx)
    V_SHE = r.V_RHE - V_RHE_to_SHE_offset
    phis_gdl_end = u[phi_s_gdl_idx(ly, NGDL)]
    phis_cl_end  = u[phi_s_cl_idx(ly, N)]
    drop_tot = phis_cl_end - V_SHE
    drop_gdl = phis_gdl_end - V_SHE
    iCO, iH2, itot = compute_currents(u, r.V_RHE, ctx)
    phiL = [u[phi_L_idx(ly, ix)] for ix in 1:N]
    mono = all(diff(phiL) .>= -1e-12) || all(diff(phiL) .<= 1e-12)
    return (r=r, d=d, drop_tot=drop_tot, drop_gdl=drop_gdl, iCO=iCO, iH2=iH2,
            itot=itot, phiL=phiL, mono=mono)
end

println("\n1/2. Solid ohmic drop and its GDL/CL split (§11 checks 1-2)")
println("     expect 5-80 mV total (~50 mV at 1 A/cm2); GDL takes 80-90 %")
for V in (-0.816, -1.205, -1.644, -2.079, -2.413, -2.500)
    s = s5row(V)
    @printf("  V=%7.3f  i=%6.3f A/cm2  drop_tot=%6.2f mV  GDL share=%5.1f %%\n",
            s.r.V_RHE, s.itot/1e4, s.drop_tot*1e3, 100*s.drop_gdl/s.drop_tot)
end

println("\n3. phi_L profile: monotone across the CL, magnitude 5-50 mV (§11 check 3)")
println("   §9.5.8: |dphi_L| in 5-25 mV at i_total > 500 mA/cm2, sign NEGATIVE")
for V in (-0.816, -1.205, -1.644, -2.079, -2.413, -2.500)
    s = s5row(V)
    @printf("  V=%7.3f  i=%7.1f mA/cm2  phi_L: %+8.3f (x=0) %+8.3f (mid) %+8.3f mV (x=L)  dphi_L=%+7.3f mV  monotone=%s\n",
            s.r.V_RHE, s.itot/10, s.phiL[1]*1e3, s.phiL[div(N,2)]*1e3, s.phiL[N]*1e3,
            s.d.dphi_L_mV, s.mono)
end

println("\n4/6. Charge conservation (§11 checks 4 and 6)")
println("     |i_s_face[1]| = i_far_total to <= 1e-6 ;  ionic rel_err at x=L_CL < 0.01")
for V in (-0.816, -1.205, -1.644, -2.079, -2.413, -2.500)
    s = s5row(V)
    @printf("  V=%7.3f  i_far=%+.6e  i_s[1]=%+.6e  |di|/i=%.3e   i_ionic_R=%+.6e  rel=%.3e\n",
            s.r.V_RHE, s.d.i_far, s.d.i_s1, s.d.charge_bal_rel, s.d.i_ionic_R, s.d.ionic_bal_rel)
end
allrows = [stage5_diagnostics(r.u, r.V_RHE, ctx) for r in out.recs]
hi = [k for k in 1:length(out.recs)
      if compute_currents(out.recs[k].u, out.recs[k].V_RHE, ctx)[3] > 5000.0]
@printf("\n  worst |di|/i over all 89 voltages          = %.3e  (target <= 1e-6)\n",
        maximum(a.charge_bal_rel for a in allrows))
@printf("  worst ionic rel_err over all 89 voltages   = %.3e  (target < 0.01)\n",
        maximum(a.ionic_bal_rel for a in allrows))
@printf("  |dphi_L| range at i_total > 500 mA/cm2     = [%.2f, %.2f] mV  (target 5-25)\n",
        minimum(abs(allrows[k].dphi_L_mV) for k in hi),
        maximum(abs(allrows[k].dphi_L_mV) for k in hi))
@printf("  dphi_L sign at high current               : %s (expect all negative)\n",
        all(allrows[k].dphi_L_mV < 0 for k in hi) ? "all negative" : "MIXED — sign bug")

println("\n5. Stage 4 -> Stage 5 current change (§11 check 5)")
println("   expect i_H2 to fall 3-10 %; i_CO may fall or RISE (migration relieves Sechenov)")
ctx4 = Ctx(4, N, NGDL, ctx.c_eq)
for V in (-1.200, -1.650, -2.075, -2.425)
    Vg = V_RHE_grid[argmin(abs.(V_RHE_grid .- V))]
    u4 = load_solution(round(Vg, digits=9), joinpath(@__DIR__,"output_fit","cache"), "fit_V")
    i4 = compute_currents(u4, Vg, ctx4)
    s = s5row(Vg)
    c4 = unpack_log_conc(u4, ctx4.ly); c5 = unpack_log_conc(s.r.u, ly)
    cCs4 = c_Cs_from_electroneutrality(view(c4,:,1)); cCs5 = c_Cs_from_electroneutrality(view(c5,:,1))
    @printf("  V=%7.3f  i_CO %7.1f -> %7.1f (%+6.1f%%)  i_H2 %7.1f -> %7.1f (%+6.1f%%) | cell1: c_CO3 %7.0f->%7.0f  c_Cs %7.0f->%7.0f  H_eff %.2f->%.2f  c_CO2 %.2f->%.2f\n",
            Vg, i4[1]/10, s.iCO/10, 100*(s.iCO-i4[1])/i4[1],
            i4[2]/10, s.iH2/10, 100*(s.iH2-i4[2])/i4[2],
            c4[4,1], c5[4,1], cCs4, cCs5,
            sechenov_H_CO2(view(c4,:,1)), sechenov_H_CO2(view(c5,:,1)), c4[5,1], c5[5,1])
end

println("\n7 / §9.5.8. Polarization smoothness")
iCOs = [compute_currents(r.u, r.V_RHE, ctx)[1]/10 for r in out.recs]
iH2s = [compute_currents(r.u, r.V_RHE, ctx)[2]/10 for r in out.recs]
fes  = 100 .* iCOs ./ (iCOs .+ iH2s)
dCO = diff(iCOs); scCO = count(k -> sign(dCO[k]) != sign(dCO[k+1]), 1:length(dCO)-1)
dH2 = diff(iH2s); scH2 = count(k -> sign(dH2[k]) != sign(dH2[k+1]), 1:length(dH2)-1)
phiL1 = [r.u[phi_L_idx(ly,1)] for r in out.recs]
flips = count(k -> sign(phiL1[k]) != sign(phiL1[k+1]) && phiL1[k] != 0, 1:length(phiL1)-1)
kpk = argmax(fes)
@printf("  sign changes in di_CO/dV  = %d  (target exactly 1)  %s\n", scCO, scCO==1 ? "PASS" : "FAIL")
@printf("  sign changes in di_H2/dV  = %d  (target 0)          %s\n", scH2, scH2==0 ? "PASS" : "FAIL")
@printf("  phi_L sign flips between neighbouring V = %d (target 0) %s\n", flips, flips==0 ? "PASS" : "FAIL")
@printf("  FE_CO peak = %.1f %% at V_RHE = %.3f\n", fes[kpk], out.recs[kpk].V_RHE)

println("\n§9.5.9 step 8 — model vs experiment at the 5 experimental voltages")
for (k, V) in enumerate(E_exp_RHE)
    j = argmin(abs.([r.V_RHE for r in out.recs] .- V))
    @printf("  V=%7.3f  i_CO %7.1f vs %6.1f (%+6.1f%%)   i_H2 %7.1f vs %6.1f (%+6.1f%%)\n",
            out.recs[j].V_RHE, iCOs[j], i_CO_exp[k], 100*(iCOs[j]-i_CO_exp[k])/i_CO_exp[k],
            iH2s[j], i_H2_exp[k], 100*(iH2s[j]-i_H2_exp[k])/i_H2_exp[k])
end
println("\n>>> STAGE 5 COMPLETE — generate plots and STOP for human review.")
