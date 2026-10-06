# =====================================================================
#  run_stage1.jl — Stage 1: liquid NP diffusion only, Dirichlet BCs
#  Simplified kinetics (φ_L = 0, φ_s = V_applied_SHE).  §11 Stage 1
#  Solver: §9.5 production stack, self-anchoring at −0.300 V RHE.
# =====================================================================
include("co2r_core.jl")
using Printf

const STAGE       = 1
const STAGE1_DIR   = joinpath(@__DIR__, "output_stage1")
const STAGE1_CACHE = joinpath(STAGE1_DIR, "cache")
save_stage1(u, V) = save_solution(STAGE1_CACHE, "s1", u, V)
load_stage1(V)    = load_solution(STAGE1_CACHE, "s1", V)

cache_banner("Stage 1", STAGE1_CACHE)

N_CL  = N_CL_DEFAULT; N_GDL = N_GDL_DEFAULT
cl    = make_cl_mesh(N_CL); gdl = make_gdl_mesh(N_GDL); gg = build_gas_geom(gdl, cl)
ly    = make_layout(STAGE, N_CL, N_GDL)
c_eq, eq = equilibrium_vector()
@printf("Mesh: N_CL=%d, N_GDL=%d, DOFs=%d\n", N_CL, N_GDL, ly.N_TOT)
@printf("Bulk equilibrium: pH=%.3f  c_CO2=%.3f  c_HCO3=%.2f mol/m3\n", eq.pH, eq.CO2, eq.HCO3)
@printf("Gas-side Henry BC (Stage 1-2, van't Hoff only): c_CO2 = %.3f mol/m3\n\n",
        c_CO2_henry_stage12)

u0 = make_u0_liquid(c_eq, ly)                       # §9.5.5 anchor guess

records, plu, ctx = continuation_sweep(Val(STAGE); ly=ly, cl=cl, gdl=gdl, gg=gg,
    c_eq=c_eq, u0=u0, V_anchor_rhe=V_RHE_grid[1], V_chain_rhe=V_RHE_grid[2:end],
    cache_dir=STAGE1_CACHE, prefix="s1", verbose=true)

write_summary(STAGE1_DIR, records)
export_all_profiles(STAGE1_DIR, records, STAGE1_CACHE, "s1", cl, gdl, ly)

# ── §11 Stage 1 physicality checks ───────────────────────────────────
println("\n" * "="^72); println("STAGE 1 SUMMARY"); println("="^72)
@printf("Voltages converged : %d / %d\n", length(records), length(V_RHE_grid))
@printf("V_RHE range        : %.3f  ->  %.3f\n", records[1].V_rhe, records[end].V_rhe)
@printf("max |F|inf         : %.3e   (assertion tol %.0e)\n",
        maximum(r.resid for r in records), F_INF_TOL)
@printf("mean Newton iters  : %.2f  (max %d)\n",
        sum(r.iters for r in records)/length(records), maximum(r.iters for r in records))
@printf("sub-steps triggered: %d\n", sum(r.substeps for r in records))

iCO = [r.i_CO/10 for r in records]; iH2 = [r.i_H2/10 for r in records]
itot = iCO .+ iH2; FE = 100 .* iCO ./ max.(itot, 1e-30)
@printf("i_CO  peak         : %.2f mA/cm2 at V_RHE = %.3f\n",
        maximum(iCO), records[argmax(iCO)].V_rhe)
@printf("i_tot at V_RHE=-2.5: %.2f mA/cm2\n", itot[end])
@printf("FE_CO peak         : %.1f %% at V_RHE = %.3f\n", maximum(FE), records[argmax(FE)].V_rhe)

u_last = load_stage1(records[end].V_rhe)
c_last = unpack_log_conc(u_last, N_CL)
pH_last = [-log10(max(c_last[1,ix]/1000,1e-30)) for ix in 1:N_CL]
@printf("\nCheck 1  pH across CL at V_RHE=-2.5 : %.2f (x=0) -> %.2f (x=L_CL)\n",
        pH_last[1], pH_last[end])
@printf("Check 2  CO2(aq) : %.4f (x=0) -> %.4f mol/m3 (x=L_CL)  [inlet Henry %.2f]\n",
        c_last[5,1], c_last[5,end], c_CO2_henry_stage12)
mono = all(diff(itot) .>= -1e-9)
@printf("Check 3  i_tot monotone increasing with cathodic V : %s\n", mono)
@printf("Check 4  FE_CO near onset (V_RHE=-0.300..-0.60)    : %.1f - %.1f %%\n",
        minimum(FE[1:13]), maximum(FE[1:13]))
println("Check 5  CO2R under-predicted vs experiment is EXPECTED at Stage 1")
println("="^72); flush(stdout)
