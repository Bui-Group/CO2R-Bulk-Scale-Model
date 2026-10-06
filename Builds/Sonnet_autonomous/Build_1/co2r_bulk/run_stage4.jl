##############################################################################
# run_stage4.jl — Stage 4: Darcy pressure finalized (§11)
##############################################################################
include("common.jl")

const CACHE_DIR = joinpath(@__DIR__, "output_stage4", "cache")
const PREFIX = "s4"

@printf("=== Stage 4 ===\n")
@printf("i0_CO2R = %.1f A/m^2, lambda_MHC = %.3f eV\n", i0_CO2R(), lambda_MHC())
@printf("HER_pf = %.3e, alpha_c = %.4f\n", HER_prefactor(), alpha_c_HER())
@printf("N_CL = %d, N_GDL = %d\n", N_CL, N_GDL)
@printf("Cache dir: %s\n", CACHE_DIR)
if isdir(CACHE_DIR) && !isempty(readdir(CACHE_DIR))
    @printf("WARNING: existing cache found. Verify kinetics match before reusing!\n")
end
@printf("kappa_GDL=%.3e  kappa_CL=%.3e  kappa_int=%.3e m^2\n", kappa_GDL, kappa_CL, kappa_int)

cl_mesh = make_cl_mesh()
gdl_mesh = make_gdl_mesh()
c_eq = solve_buffer_equilibrium(1000.0, 1000.0)
c_eq_vec = c_eq_vector(c_eq)

ly = GasLayout(N_CL, N_GDL)
plu = AffineLU()
res_at_V! = (F_,u_,V_) -> residual_stage4!(F_, u_, V_, cl_mesh, gdl_mesh, c_eq_vec, ly)

u0 = make_u0_gas_equilibrium(c_eq_vec, ly)
V_anchor = V_RHE_grid[1]

t0 = time()
u, info_anchor = anchor_solve!(u0, V_anchor, res_at_V!, ly, plu; alpha_init=1e-2, max_iter=200, rtol=1e-8)
@printf("Anchor V=%.4f: converged=%s iters=%d ||F||=%.3e reason=%s (t=%.1fs)\n",
        V_anchor, info_anchor.converged, info_anchor.iterations, info_anchor.residual, info_anchor.reason, time()-t0)
@assert info_anchor.converged "Stage 4 anchor failed to converge"
post_solve_assert!(u, V_anchor, res_at_V!, ly, plu)
save_solution(CACHE_DIR, PREFIX, u, V_anchor)

Vs_rest, all_ok = sweep_voltages!(u, V_anchor, V_RHE_grid[2:end], res_at_V!, ly, plu;
                                   cache_dir=CACHE_DIR, prefix=PREFIX, rtol=1e-8, tag="[stage4]", verbose=false)
elapsed = time() - t0
n_conv = length(Vs_rest) + 1
@printf("Stage 4 sweep complete: %d/%d voltages converged in %.1f s\n", n_conv, length(V_RHE_grid), elapsed)

outdir = joinpath(@__DIR__, "output_stage4")
Vs_all = list_cache_voltages(CACHE_DIR, PREFIX)

iCO = Float64[]; iH2 = Float64[]
for V in Vs_all
    uV = load_solution(CACHE_DIR, PREFIX, V)
    Vshe = V - V_RHE_to_SHE_offset
    push!(iCO, compute_i_CO_simple(uV, Vshe, cl_mesh)/10.0)
    push!(iH2, compute_i_H2_simple(uV, Vshe, cl_mesh)/10.0)
end
export_polarization_csv(joinpath(outdir, "polarization.csv"), Vs_all, iCO, iH2)
rows_fn = V -> liquid_diagnostics_simple(load_solution(CACHE_DIR, PREFIX, V), cl_mesh, V - V_RHE_to_SHE_offset)
export_liquid_profiles_csv(joinpath(outdir, "liquid_profiles.csv"), cl_mesh, Vs_all, rows_fn)
gas_rows_fn = V -> gas_diagnostics(load_solution(CACHE_DIR, PREFIX, V), ly, gdl_mesh, cl_mesh)
export_gas_profiles_csv(joinpath(outdir, "gas_profiles.csv"), Vs_all, gas_rows_fn, N_GDL+N_CL)

# ── Physicality checks (§11 Stage 4) ────────────────────────────────────────
@printf("\n--- Physicality checks ---\n")
itot = iCO .+ iH2
for (target_i, label) in [(270.0,"~0.27 A/cm2"),(400.0,"~0.40 A/cm2"),(620.0,"~0.62 A/cm2"),(1150.0,"~1.15 A/cm2")]
    idx = argmin(abs.(itot .- target_i))
    uV = load_solution(CACHE_DIR, PREFIX, Vs_all[idx])
    (_,_,_,p1) = cl_gas(uV, ly, 1); (_,_,_,pN) = cl_gas(uV, ly, N_CL)
    (_,_,_,pgdl1) = gdl_gas(uV, ly, 1)
    dp_CL = pN - p1
    @printf("i_total=%.1f mA/cm2 (target %s) at V=%.3f: CL pressure drop = %.2f Pa (p_gdl_inlet=%.2f, p_cl1=%.2f, p_clN=%.2f)\n",
            itot[idx], label, Vs_all[idx], -dp_CL, pgdl1, p1, pN)
end

# mass balance: total Darcy mass flux at GDL inlet should equal net CL gas production (§11 Stage 4)
V_hi = Vs_all[argmax(itot)]
uV = load_solution(CACHE_DIR, PREFIX, V_hi)
N_CLl = cl_mesh.N
c = unpack_log_conc(uV, N_CLl)
Vshe_hi = V_hi - V_RHE_to_SHE_offset
totS = 0.0
for ix in 1:N_CLl
    c_ix = @view c[:,ix]
    r_co2r = R_CO2R(c_ix[5], Vshe_hi, 0.0, c_ix[1], a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
    r_her  = R_HER(Vshe_hi, 0.0, c_ix[1], a_v_CL, n_e)
    w1,w2,w3,p_g = cl_gas(uV, ly, ix)
    xCO2 = mole_fraction_CO2(w1,w2,w3); pCO2_local = xCO2*p_g/101325.0
    H_local = sechenov_H_CO2(c_ix)
    rpt = a_v_GL*k_GL()*(H_local*pCO2_local - c_ix[5])
    global totS += (-rpt*M_gas[1] + r_co2r*M_gas[2] + r_her*M_gas[3])*cl_mesh.dx[ix]
end
w1,w2,w3,p1 = gdl_gas(uV, ly, 1)
Mn1 = Mn_from_omega(w1,w2,w3); rho1 = p_total*Mn1/(R_gas*T)
u_G_inlet = -(kappa_GDL/mu_gas)*(p1-p_total)/gdl_mesh.dx_left
rhou_inlet = rho1*u_G_inlet
@printf("At V=%.3f (max current): net CL gas mass source integral = %.4e kg/m2/s; GDL-inlet Darcy mass flux = %.4e kg/m2/s\n",
        V_hi, totS, rhou_inlet)
@printf("(sign convention: inlet flux should be NEGATIVE if net source is negative i.e. net consumption pulls gas IN)\n")

V_test = -1.500
u3 = load_solution(joinpath(@__DIR__,"output_stage3","cache"), "s3", V_test)
u4 = load_solution(CACHE_DIR, PREFIX, V_test)
Vshe_test = V_test - V_RHE_to_SHE_offset
i3 = compute_i_CO_simple(u3, Vshe_test, cl_mesh)/10.0 + compute_i_H2_simple(u3, Vshe_test, cl_mesh)/10.0
i4 = compute_i_CO_simple(u4, Vshe_test, cl_mesh)/10.0 + compute_i_H2_simple(u4, Vshe_test, cl_mesh)/10.0
@printf("At V_RHE=%.3f: Stage3 i_total=%.2f, Stage4 i_total=%.2f mA/cm2 (should be ALMOST IDENTICAL, %.3f%% diff)\n",
        V_test, i3, i4, 100*(i4-i3)/i3)

@printf("\nStage 4 driver finished. Run plot_stage4.py to generate figures, then inspect.\n")
