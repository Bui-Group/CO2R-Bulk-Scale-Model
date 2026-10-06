# =====================================================================
#  run_stage4.jl — Stage 4: Darcy pressure equation finalized.
#  The pressure DOF becomes a true Darcy continuity balance (Dirichlet
#  p_total at the flow field, dp/dx = 0 at the electrolyte side, harmonic
#  mean permeability at the GDL/CL interface), and the species flux picks
#  up the convective term rho*omega*u_G.  §11 Stage 4
# =====================================================================
include("co2r_core.jl")
using Printf

const STAGE        = 4
const STAGE4_DIR   = joinpath(@__DIR__, "output_stage4")
const STAGE4_CACHE = joinpath(STAGE4_DIR, "cache")
const STAGE3_CACHE = joinpath(@__DIR__, "output_stage3", "cache")
save_stage4(u, V) = save_solution(STAGE4_CACHE, "s4", u, V)
load_stage4(V)    = load_solution(STAGE4_CACHE, "s4", V)
load_warmstart_stage3(V) = load_solution(STAGE3_CACHE, "s3", V)   # §10.5, optional

cache_banner("Stage 4", STAGE4_CACHE)

N_CL  = N_CL_DEFAULT; N_GDL = N_GDL_DEFAULT
cl    = make_cl_mesh(N_CL); gdl = make_gdl_mesh(N_GDL); gg = build_gas_geom(gdl, cl)
ly    = make_layout(STAGE, N_CL, N_GDL)
c_eq, eq = equilibrium_vector()
@printf("Mesh: N_CL=%d, N_GDL=%d, DOFs=%d\n", N_CL, N_GDL, ly.N_TOT)
@printf("Darcy: kappa_GDL=%.3e  kappa_CL=%.3e  kappa_interface(harmonic)=%.3e m2\n",
        kappa_GDL, kappa_CL, kappa_interface)
@printf("       mu_gas = %.3e Pa s ; p BCs: Dirichlet %.0f Pa at x=-L_GDL, dp/dx=0 at x=L_CL\n\n",
        mu_gas, p_total)

u0 = make_u0_gasliquid(c_eq, ly)      # self-anchoring (§9.5.5)

records, plu, ctx = continuation_sweep(Val(STAGE); ly=ly, cl=cl, gdl=gdl, gg=gg,
    c_eq=c_eq, u0=u0, V_anchor_rhe=V_RHE_grid[1], V_chain_rhe=V_RHE_grid[2:end],
    cache_dir=STAGE4_CACHE, prefix="s4", verbose=true)

write_summary(STAGE4_DIR, records)
export_all_profiles(STAGE4_DIR, records, STAGE4_CACHE, "s4", cl, gdl, ly)

println("\n" * "="^72); println("STAGE 4 SUMMARY"); println("="^72)
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

# ── §11 Stage 4 physicality checks ───────────────────────────────────
function darcy_diag(u, V)
    c = unpack_log_conc(u, N_CL)
    wgh = (omega_inlet[1], omega_inlet[2], omega_inlet[3], omega_inlet[4])
    wR  = gas_omega(u, ly, 1); pR = gas_p(u, ly, 1)
    _,_,_, ru_in = sm_face_flux_full(wgh, p_total, wR, pR, gg.Dbin[1], gg.kappa[1], gg.dcc[1], true)
    RPT = [a_v_GL*k_GL()*(sechenov_H_CO2(view(c,:,ix)) *
           (gas_omega(u,ly,N_GDL+ix)[1]*Mn_from_omega(gas_omega(u,ly,N_GDL+ix))/M_gas[1]) *
           gas_p(u,ly,N_GDL+ix)/101325 - c[5,ix]) for ix in 1:N_CL]
    iCO_, iH2_, _ = compute_currents(u, V, cl, ly)
    Smass = -sum(RPT.*cl.dx)*M_gas[1] + (iCO_/(2*F_const))*M_gas[2] + (iH2_/(2*F_const))*M_gas[3]
    ps = [gas_p(u, ly, g) for g in 1:(N_GDL+N_CL)]
    return ru_in, Smass, ps, iCO_+iH2_
end

println("\nDarcy mass balance and pressure field:")
@printf("  %-9s %-11s %-13s %-13s %-9s %-11s %-11s\n",
        "V_RHE","i_tot","rho*u(inlet)","-int S_mass","rel err","dp GDL [Pa]","dp CL [Pa]")
for V in (-1.200, -1.650, -2.075, -2.500)
    u = load_stage4(V); ru, Sm, ps, it = darcy_diag(u, V)
    dp_gdl = p_total - ps[N_GDL]; dp_cl = ps[N_GDL] - ps[end]
    @printf("  %+.4f  %8.1f    %+.5e  %+.5e  %.2e  %+10.4f  %+10.4f\n",
            V, it/10, ru, -Sm, abs(ru+Sm)/max(abs(ru),1e-30), dp_gdl, dp_cl)
end

u_hi = load_stage4(-2.500); ru, Sm, ps, it = darcy_diag(u_hi, -2.500)
@printf("\nCheck A  Darcy inlet mass flux == net gas consumption : rel err %.2e\n",
        abs(ru + Sm)/abs(ru))
# dp/dx = mu*u_G/kappa consistency, checked face by face inside the GDL
# Darcy law is an identity by construction; the meaningful check is that the
# implied u_G is smooth and that rho*u is conserved face-to-face in the GDL
# (no gas sources there), i.e. rho(k)*u(k) is the same at every GDL face.
ru_faces = [ -(kappa_GDL/mu_gas)*(ps[k]-ps[k-1])/gdl.dx_between[k] *
             0.5*(gas_p(u_hi,ly,k)+gas_p(u_hi,ly,k-1)) *
             Mn_from_omega(gas_omega(u_hi,ly,k))/(R_gas*T) for k in 2:N_GDL ]
worst = (maximum(ru_faces)-minimum(ru_faces))/abs(ru)
@printf("Check B  rho*u constant across all GDL faces (no source): spread %.2e of inlet flux\n", worst)
@printf("Check C  total pressure drop across CL at i=%.2f A/cm2 : %.3f Pa\n", it/1e4, ps[N_GDL]-ps[end])
@printf("Check D  pressure stays physical (min p)               : %.2f Pa (%.4f atm)\n",
        minimum(ps), minimum(ps)/101325)

s3 = joinpath(@__DIR__, "output_stage3", "summary.csv")
if isfile(s3)
    d3 = Dict{Float64,NTuple{2,Float64}}()
    for ln in readlines(s3)[2:end]
        f = split(ln,","); d3[parse(Float64,f[1])] = (parse(Float64,f[7]), parse(Float64,f[8]))
    end
    println("\nCheck E  Stage 4 currents almost identical to Stage 3 (Darcy is a tiny correction):")
    mx = maximum(max(abs(r.i_CO/10-get(d3,r.V_rhe,(NaN,NaN))[1])/max(get(d3,r.V_rhe,(1.0,1.0))[1],1e-9),
                     abs(r.i_H2/10-get(d3,r.V_rhe,(NaN,NaN))[2])/max(get(d3,r.V_rhe,(1.0,1.0))[2],1e-9))
                 for r in records)
    @printf("         max relative change over all 89 voltages : %.4f %%\n", 100*mx)
    for V in (-1.200, -1.650, -2.500)
        k = findfirst(r -> r.V_rhe == V, records); r = records[k]; b = d3[V]
        @printf("         V_RHE=%+.3f  i_CO %9.3f -> %9.3f (%+.4f%%)   i_H2 %9.3f -> %9.3f (%+.4f%%)\n",
                V, b[1], r.i_CO/10, 100*(r.i_CO/10-b[1])/b[1],
                b[2], r.i_H2/10, 100*(r.i_H2/10-b[2])/b[2])
    end
end
println("="^72); flush(stdout)
