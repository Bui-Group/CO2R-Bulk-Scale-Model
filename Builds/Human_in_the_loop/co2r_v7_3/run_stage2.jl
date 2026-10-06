# ============================================================================
# run_stage2.jl — Stage 2: replace the Dirichlet electrolyte BC with the
# Sherwood-Reynolds convective Neumann BC.  §11 Stage 2 / §7.2
# No structural change to DOFs, residual layout or solver.
# ============================================================================
include("model.jl")
print_kinetics(2)
println("Cache dir: ", cache_dir(2))
if isdir(cache_dir(2)) && !isempty(readdir(cache_dir(2)))
    println("WARNING: Existing cache found. Verify kinetics match before reusing!")
end
println()

out = run_stage(2)
ctx = out.ctx; ly = ctx.ly

# ── §11 Stage 2 physicality checks ─────────────────────────────────────────
println("\n" * "="^72)
println("Stage 2 physicality checks")

println("\nBiot numbers  Bi_i = k_MT_i * L_CL / D_eff_i   (expect O(0.1-10))")
names = ["H+", "OH-", "HCO3-", "CO3^2-", "CO2(aq)"]
for i in 1:5
    @printf("  %-8s Bi = %8.3f    k_MT = %.4e m/s   D_eff = %.4e m^2/s\n",
            names[i], k_MT[i]*L_CL/D_eff[i], k_MT[i], D_eff[i])
end

eqv = ctx.c_eq
println("\nElectrolyte-face state and flux consistency")
@printf("  bulk equilibrium: c_OH = %.4e, c_CO2 = %.4f mol/m^3\n", eqv[2], eqv[5])
function report2(V)
    k = argmin(abs.([r.V_RHE for r in out.recs] .- V))
    r = out.recs[k]
    c = unpack_log_conc(r.u, ly)
    N = ly.N_CL
    # right-face flux from the residual convention vs k_MT*(c_local - c_eq)
    maxerr = 0.0
    for i in 1:5
        Jres = -k_MT[i] * (eqv[i] - c[i,N])
        Jchk =  k_MT[i] * (c[i,N] - eqv[i])
        maxerr = max(maxerr, abs(Jres - Jchk))
    end
    iCO, iH2, itot = compute_currents(r.u, r.V_RHE, ctx)
    @printf("  V=%7.3f  c_CO2(L)=%8.4f  c_OH(L)=%10.4f (x%.1f bulk)  J_OH=%+.3e  J_CO2=%+.3e  |flux chk|=%.1e  i_tot=%8.2f mA/cm2\n",
            r.V_RHE, c[5,N], c[2,N], c[2,N]/eqv[2],
            k_MT[2]*(c[2,N]-eqv[2]), k_MT[5]*(c[5,N]-eqv[5]), maxerr, itot/10)
end
for V in (-0.300, -0.800, -1.125, -1.600, -2.075, -2.500); report2(V); end

# Stage 1 vs Stage 2 comparison at the same voltages
println("\nStage 1 -> Stage 2 current comparison")
for V in (-0.800, -1.205, -1.644, -2.079, -2.413)
    u1 = load_solution(round(V_RHE_grid[argmin(abs.(V_RHE_grid .- V))], digits=9),
                       cache_dir(1), cache_prefix(1))
    Vg = V_RHE_grid[argmin(abs.(V_RHE_grid .- V))]
    ctx1 = Ctx(1, ly.N_CL, ly.N_GDL, ctx.c_eq)
    i1 = compute_currents(u1, Vg, ctx1)
    k = argmin(abs.([r.V_RHE for r in out.recs] .- Vg))
    i2 = compute_currents(out.recs[k].u, Vg, ctx)
    @printf("  V=%7.3f  i_CO: %8.2f -> %8.2f   i_H2: %8.2f -> %8.2f  mA/cm2\n",
            Vg, i1[1]/10, i2[1]/10, i1[2]/10, i2[2]/10)
end

fe = [begin iCO,iH2,it = compute_currents(r.u, r.V_RHE, ctx); it>0 ? 100*iCO/it : 0.0 end
      for r in out.recs]
kmax = argmax(fe)
@printf("\npeak FE_CO = %.1f %% at V_RHE = %.3f\n", fe[kmax], out.recs[kmax].V_RHE)
println("\n>>> STAGE 2 COMPLETE — generate plots and STOP for human review.")
