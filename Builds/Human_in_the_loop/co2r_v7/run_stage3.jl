# =====================================================================
#  run_stage3.jl — Stage 3: Stefan-Maxwell gas transport (u_G = 0)
#  + volumetric CO2 phase transfer R_PT.  Pressure DOF is a placeholder
#  pinned at p_total (Darcy is finalized in Stage 4).  §11 Stage 3
# =====================================================================
include("co2r_core.jl")
using Printf

const STAGE        = 3
const STAGE3_DIR   = joinpath(@__DIR__, "output_stage3")
const STAGE3_CACHE = joinpath(STAGE3_DIR, "cache")
const STAGE2_CACHE = joinpath(@__DIR__, "output_stage2", "cache")
save_stage3(u, V) = save_solution(STAGE3_CACHE, "s3", u, V)
load_stage3(V)    = load_solution(STAGE3_CACHE, "s3", V)
load_warmstart_stage2(V) = load_solution(STAGE2_CACHE, "s2", V)   # §10.5, optional

cache_banner("Stage 3", STAGE3_CACHE)

N_CL  = N_CL_DEFAULT; N_GDL = N_GDL_DEFAULT
cl    = make_cl_mesh(N_CL); gdl = make_gdl_mesh(N_GDL); gg = build_gas_geom(gdl, cl)
ly    = make_layout(STAGE, N_CL, N_GDL)
c_eq, eq = equilibrium_vector()
@printf("Mesh: N_CL=%d, N_GDL=%d, DOFs=%d  (liquid %d + gas %d)\n",
        N_CL, N_GDL, ly.N_TOT, 5*N_CL, 4*(N_GDL+N_CL))
@printf("Inlet gas: x = %s (mole)  ->  omega = [%.4f, %.4f, %.4f, %.4f] (mass)\n",
        x_inlet, omega_inlet...)
@printf("FSG binary diffusivities (raw, m2/s):  CO2-CO %.3e  CO2-H2 %.3e  CO2-H2O %.3e\n",
        D_bin_raw[1,2], D_bin_raw[1,3], D_bin_raw[1,4])
@printf("Bruggeman: GDL x%.4f (eps=%.2f)   CL x%.4f (eps_G=%.2f)\n",
        eps_GDL^1.5, eps_GDL, eps_G^1.5, eps_G)
@printf("Phase transfer: a_v_GL = %.3e 1/m, k_GL = %.1e m/s -> a_v_GL*k_GL = %.3e 1/s\n",
        a_v_GL, k_GL(), a_v_GL*k_GL())
@printf("Sechenov H_eff at bulk c_eq = %.3f mol/m3/atm (van't Hoff only: %.3f)\n\n",
        sechenov_H_CO2(c_eq), H_vanthoff)

# ── §11 Stage 3 bootstrap step 1: gas-only solve (kinetics + R_PT off) ──
println("Bootstrap 1/2 — gas-only solve (R_PT = R_CO = R_H2 = 0), triggers JIT:")
u_g = make_u0_gasliquid(c_eq, ly)
let i0s = i0_CO2R(), hps = HER_prefactor(), kgs = k_GL()
    set_i0_CO2R!(0.0); set_HER_prefactor!(0.0); set_k_GL!(0.0)
    plug = AffineLU()
    tjit = @elapsed ig = newton_affine!(u_g, (F_,u_) -> full_residual!(F_,u_,V_RHE_grid[1],
                    cl,gdl,c_eq,ly,gg,Val(STAGE)); ly=ly, plu=plug, rtol=RTOL_PROD,
                    max_iter=MAXIT_ANCHOR, alpha_init=ALPHA_ANCHOR, n_log_dofs=5*N_CL)
    Fg = zeros(ly.N_TOT); full_residual!(Fg,u_g,V_RHE_grid[1],cl,gdl,c_eq,ly,gg,Val(STAGE))
    gas_rows = (5*N_CL+1):ly.N_TOT
    @printf("  converged=%s it=%d |F|inf=%.3e (gas block %.3e) in %.1f s\n",
            ig.converged, ig.iterations, ig.residual, maximum(abs, Fg[gas_rows]), tjit)
    @printf("  all cells still at inlet composition: %s\n",
            all(abs(gas_omega(u_g, ly, g)[1] - omega_inlet[1]) < 1e-10 for g in 1:(N_GDL+N_CL)))
    set_i0_CO2R!(i0s); set_HER_prefactor!(hps); set_k_GL!(kgs)
end

println("\nBootstrap 2/2 — monolithic gas+liquid continuation sweep:")
u0 = make_u0_gasliquid(c_eq, ly)         # self-anchoring (§9.5.5)

records, plu, ctx = continuation_sweep(Val(STAGE); ly=ly, cl=cl, gdl=gdl, gg=gg,
    c_eq=c_eq, u0=u0, V_anchor_rhe=V_RHE_grid[1], V_chain_rhe=V_RHE_grid[2:end],
    cache_dir=STAGE3_CACHE, prefix="s3", verbose=true)

write_summary(STAGE3_DIR, records)
export_all_profiles(STAGE3_DIR, records, STAGE3_CACHE, "s3", cl, gdl, ly)

println("\n" * "="^72); println("STAGE 3 SUMMARY"); println("="^72)
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

# ── §11 Stage 3 physicality checks ───────────────────────────────────
u_hi = load_stage3(records[end].V_rhe); c_hi = unpack_log_conc(u_hi, N_CL)
w_in  = gas_omega(u_hi, ly, 1)                      # first GDL cell (inlet side)
w_int = gas_omega(u_hi, ly, N_GDL)                  # last GDL cell  (interface)
w_out = gas_omega(u_hi, ly, N_GDL+N_CL)             # last CL cell   (electrolyte side)
println("\nGas composition at V_RHE = -2.500 (mass fractions):")
@printf("  %-22s CO2=%.5f  CO=%.5f  H2=%.5f  H2O=%.5f\n","GDL cell 1 (inlet)", w_in...)
@printf("  %-22s CO2=%.5f  CO=%.5f  H2=%.5f  H2O=%.5f\n","GDL cell N (interface)", w_int...)
@printf("  %-22s CO2=%.5f  CO=%.5f  H2=%.5f  H2O=%.5f\n","CL  cell N (exit)", w_out...)
@printf("\nCheck 1  omega_CO2 decreases inlet -> CL exit : %.5f -> %.5f  (%s)\n",
        w_in[1], w_out[1], w_out[1] < w_in[1])
@printf("Check 2  omega_CO and omega_H2 rise through CL: CO %.3e -> %.3e, H2 %.3e -> %.3e (%s)\n",
        w_int[2], w_out[2], w_int[3], w_out[3], w_out[2] > w_int[2] && w_out[3] > w_int[3])

# Check 3 — Henry consistency at the gas-liquid interface, cell by cell
p1  = gas_p(u_hi, ly, N_GDL+1)
w1  = gas_omega(u_hi, ly, N_GDL+1)
x1  = w1[1]*Mn_from_omega(w1)/M_gas[1]
Hl  = sechenov_H_CO2(view(c_hi,:,1))
@printf("Check 3  CL cell 1: p_CO2 = %.5f atm, H_eff = %.3f -> H*p = %.4f mol/m3 vs c_CO2(aq) = %.4f\n",
        x1*p1/101325, Hl, Hl*x1*p1/101325, c_hi[5,1])
RPT = [a_v_GL*k_GL()*(sechenov_H_CO2(view(c_hi,:,ix)) *
       (gas_omega(u_hi,ly,N_GDL+ix)[1]*Mn_from_omega(gas_omega(u_hi,ly,N_GDL+ix))/M_gas[1]) *
       gas_p(u_hi,ly,N_GDL+ix)/101325 - c_hi[5,ix]) for ix in 1:N_CL]
intRPT  = sum(RPT .* cl.dx)
intRCO2R = sum(R_CO2R(c_hi[5,ix], records[end].V_rhe - V_RHE_to_SHE_offset, 0.0, c_hi[1,ix],
                      a_v_CL, lambda_MHC(), i0_CO2R(), n_e) * cl.dx[ix] for ix in 1:N_CL)
@printf("Check 4  int R_PT dx = %.6e vs int R_CO2R dx = %.6e mol/m2/s (ratio %.4f)\n",
        intRPT, intRCO2R, intRPT/intRCO2R)
# §12.6 CO2(aq) mass balance:  J_in(0) - J_out(L) + int R_PT - int R_CO2R + int S_buf = 0
Jin  = diffusion_flux(Hl*x1*p1/101325, c_hi[5,1], D_eff[5], cl.dx_between[1])
Jout = diffusion_flux(c_hi[5,N_CL], c_eq[5], D_eff[5], L_eff_bc[5])
buf5 = sum(eps_I*buffer_sources(view(c_hi,:,ix))[5]*cl.dx[ix] for ix in 1:N_CL)
@printf("Check 5  CO2(aq) mass balance residual = %.3e mol/m2/s (tol 1e-6)\n",
        Jin - Jout + intRPT - intRCO2R + buf5)
cCs_hi = [cCs_from_electroneutrality(view(c_hi,:,ix)) for ix in 1:N_CL]
@printf("         CL cell 1 ions: OH- %.4e  HCO3- %.4e  CO3-- %.4e  Cs+ %.4e mol/m3, pH %.2f\n",
        c_hi[2,1], c_hi[3,1], c_hi[4,1], cCs_hi[1], -log10(max(c_hi[1,1]/1000,1e-30)))
@printf("Check 6  pressure exactly uniform (Stage-3 placeholder): max|p - p_total| = %.3e Pa\n",
        maximum(abs(gas_p(u_hi,ly,g) - p_total) for g in 1:(N_GDL+N_CL)))
println("="^72); flush(stdout)
