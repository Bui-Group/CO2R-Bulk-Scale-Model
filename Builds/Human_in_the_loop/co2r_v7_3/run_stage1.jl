# ============================================================================
# run_stage1.jl — Stage 1: liquid NP diffusion only, Dirichlet electrolyte BC,
# simplified kinetics (φ_L = 0, φ_s = V_applied_SHE).   §11 Stage 1
# ============================================================================
include("model.jl")
print_kinetics(1)
println("Cache dir: ", cache_dir(1))
if isdir(cache_dir(1)) && !isempty(readdir(cache_dir(1)))
    println("WARNING: Existing cache found. Verify kinetics match before reusing!")
end
println()

out = run_stage(1)

# ── §11 Stage 1 physicality checks ─────────────────────────────────────────
println("\n" * "="^72)
println("Stage 1 physicality checks")
ctx = out.ctx
for r in out.recs
    if r.V_RHE in (-0.300, -1.200, -2.500) || abs(r.V_RHE + 2.500) < 1e-9
    end
end
function report(V)
    k = argmin(abs.([r.V_RHE for r in out.recs] .- V))
    r = out.recs[k]
    c = unpack_log_conc(r.u, ctx.ly)
    pH = [-log10(c[1,ix]/1000) for ix in 1:ctx.ly.N_CL]
    iCO, iH2, itot = compute_currents(r.u, r.V_RHE, ctx)
    @printf("V_RHE=%7.3f  pH: %.2f (x=0) -> %.2f (x=L)  c_CO2: %.3f -> %.3f  i_CO=%8.2f i_H2=%8.2f mA/cm2  FE_CO=%5.1f%%\n",
            r.V_RHE, pH[1], pH[end], c[5,1], c[5,end], iCO/10, iH2/10,
            itot > 0 ? 100*iCO/itot : 0.0)
end
for V in (-0.300, -0.800, -1.125, -1.600, -2.075, -2.500); report(V); end

fe = [begin iCO,iH2,it = compute_currents(r.u, r.V_RHE, ctx); it>0 ? 100*iCO/it : 0.0 end
      for r in out.recs]
kmax = argmax(fe)
@printf("\npeak FE_CO = %.1f %% at V_RHE = %.3f  (guide: ~64%% at ~-1.125 V)\n",
        fe[kmax], out.recs[kmax].V_RHE)
println("\n>>> STAGE 1 COMPLETE — generate plots and STOP for human review.")
