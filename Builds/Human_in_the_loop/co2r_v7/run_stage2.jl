# =====================================================================
#  run_stage2.jl — Stage 2: Sherwood-Reynolds Neumann BC at the
#  electrolyte side.  Only the last face flux changes (§11 Stage 2):
#      J[i, N+1] = -k_MT[i] * (c_eq[i] - c[i,N])
#  No structural change to DOF count, residual layout or solver.
# =====================================================================
include("co2r_core.jl")
using Printf

const STAGE        = 2
const STAGE2_DIR   = joinpath(@__DIR__, "output_stage2")
const STAGE2_CACHE = joinpath(STAGE2_DIR, "cache")
const STAGE1_CACHE = joinpath(@__DIR__, "output_stage1", "cache")
save_stage2(u, V) = save_solution(STAGE2_CACHE, "s2", u, V)
load_stage2(V)    = load_solution(STAGE2_CACHE, "s2", V)
load_warmstart_stage1(V) = load_solution(STAGE1_CACHE, "s1", V)   # §10.5, optional

cache_banner("Stage 2", STAGE2_CACHE)

N_CL  = N_CL_DEFAULT; N_GDL = N_GDL_DEFAULT
cl    = make_cl_mesh(N_CL); gdl = make_gdl_mesh(N_GDL); gg = build_gas_geom(gdl, cl)
ly    = make_layout(STAGE, N_CL, N_GDL)
c_eq, eq = equilibrium_vector()
@printf("Mesh: N_CL=%d, N_GDL=%d, DOFs=%d\n", N_CL, N_GDL, ly.N_TOT)

# §11 Stage 2 physicality check — Biot numbers (must be O(0.1-10))
println("\nBiot numbers  Bi_i = k_MT,i * L_CL / D_eff,i :")
for i in 1:5
    @printf("  %-8s k_MT=%.4e m/s   D_eff=%.4e m2/s   L_eff=%.4e m   Bi=%.3f\n",
            SPECIES_NAMES[i], k_MT[i], D_eff[i], L_eff_bc[i], k_MT[i]*L_CL/D_eff[i])
end
println()

u0 = make_u0_liquid(c_eq, ly)     # self-anchoring (§9.5.5); Stage-1 cache optional

records, plu, ctx = continuation_sweep(Val(STAGE); ly=ly, cl=cl, gdl=gdl, gg=gg,
    c_eq=c_eq, u0=u0, V_anchor_rhe=V_RHE_grid[1], V_chain_rhe=V_RHE_grid[2:end],
    cache_dir=STAGE2_CACHE, prefix="s2", verbose=true)

write_summary(STAGE2_DIR, records)
export_all_profiles(STAGE2_DIR, records, STAGE2_CACHE, "s2", cl, gdl, ly)

println("\n" * "="^72); println("STAGE 2 SUMMARY"); println("="^72)
@printf("Voltages converged : %d / %d\n", length(records), length(V_RHE_grid))
@printf("max |F|inf         : %.3e   (assertion tol %.0e)\n",
        maximum(r.resid for r in records), F_INF_TOL)
@printf("mean Newton iters  : %.2f  (max %d)\n",
        sum(r.iters for r in records)/length(records), maximum(r.iters for r in records))
@printf("sub-steps triggered: %d\n", sum(r.substeps for r in records))

iCO = [r.i_CO/10 for r in records]; iH2 = [r.i_H2/10 for r in records]
itot = iCO .+ iH2; FE = 100 .* iCO ./ max.(itot, 1e-30)
@printf("i_CO peak          : %.2f mA/cm2 at V_RHE = %.3f\n", maximum(iCO), records[argmax(iCO)].V_rhe)
@printf("FE_CO peak         : %.1f %% at V_RHE = %.3f\n", maximum(FE), records[argmax(FE)].V_rhe)
@printf("i_tot at V_RHE=-2.5: %.2f mA/cm2\n", itot[end])

# overlay vs Stage 1
s1 = joinpath(@__DIR__, "output_stage1", "summary.csv")
if isfile(s1)
    println("\nStage 1 -> Stage 2 comparison (Dirichlet -> Sherwood at x = L_CL):")
    l1 = readlines(s1)[2:end]
    d1 = Dict{Float64,NTuple{2,Float64}}()
    for ln in l1
        f = split(ln, ","); d1[parse(Float64,f[1])] = (parse(Float64,f[7]), parse(Float64,f[8]))
    end
    for V in (-0.816, -1.205, -1.644, -2.079, -2.413)
        k = argmin(abs.([r.V_rhe for r in records] .- V)); r = records[k]
        b = get(d1, r.V_rhe, (NaN,NaN))
        @printf("  V_RHE=%+.3f  i_CO %8.2f -> %8.2f (%+6.1f%%)   i_H2 %8.2f -> %8.2f (%+6.1f%%)\n",
                r.V_rhe, b[1], r.i_CO/10, 100*(r.i_CO/10-b[1])/max(b[1],1e-9),
                b[2], r.i_H2/10, 100*(r.i_H2/10-b[2])/max(b[2],1e-9))
    end
end

# §11 Stage 2 checks: boundary concentrations and fluxes
u_hi = load_stage2(records[end].V_rhe)
c_hi = unpack_log_conc(u_hi, N_CL)
println("\nElectrolyte-face state at V_RHE = -2.500 :")
@printf("  %-8s %-14s %-14s %-16s\n", "species", "c(L_CL)", "c_eq", "J_face [mol/m2/s]")
for i in 1:5
    J = -k_MT[i]*(c_eq[i] - c_hi[i,N_CL])
    @printf("  %-8s %-14.6e %-14.6e %+.6e  (%s)\n", SPECIES_NAMES[i], c_hi[i,N_CL], c_eq[i], J,
            J > 0 ? "leaving CL" : "entering CL")
end
@printf("\nCheck  c_CO2(L_CL) -> 0 at high current : %.4e mol/m3 (c_eq %.3f)\n",
        c_hi[5,N_CL], c_eq[5])
@printf("Check  c_OH(L_CL) above bulk equilibrium: %.4e vs %.4e -> %s\n",
        c_hi[2,N_CL], c_eq[2], c_hi[2,N_CL] > c_eq[2])
@printf("Check  i_tot monotone in V              : %s\n", all(diff(itot) .>= -1e-9))
println("="^72); flush(stdout)
