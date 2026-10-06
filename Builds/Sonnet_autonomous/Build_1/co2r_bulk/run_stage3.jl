##############################################################################
# run_stage3.jl — Stage 3: Stefan-Maxwell gas transport (§11)
##############################################################################
include("common.jl")

const CACHE_DIR = joinpath(@__DIR__, "output_stage3", "cache")
const PREFIX = "s3"

@printf("=== Stage 3 ===\n")
@printf("i0_CO2R = %.1f A/m^2, lambda_MHC = %.3f eV\n", i0_CO2R(), lambda_MHC())
@printf("HER_pf = %.3e, alpha_c = %.4f\n", HER_prefactor(), alpha_c_HER())
@printf("N_CL = %d, N_GDL = %d\n", N_CL, N_GDL)
@printf("Cache dir: %s\n", CACHE_DIR)
if isdir(CACHE_DIR) && !isempty(readdir(CACHE_DIR))
    @printf("WARNING: existing cache found. Verify kinetics match before reusing!\n")
end

cl_mesh = make_cl_mesh()
gdl_mesh = make_gdl_mesh()
c_eq = solve_buffer_equilibrium(1000.0, 1000.0)
c_eq_vec = c_eq_vector(c_eq)
@printf("omega_inlet = %s (mole fractions x_inlet=%s, M_avg=%.3f g/mol)\n",
        string(round.(omega_inlet,digits=4)), string(x_inlet), M_avg_inlet*1000)

ly = GasLayout(N_CL, N_GDL)
@printf("Stage 3 DOFs = %d (expect 5*80+4*(120+80)=1200 at production mesh)\n", ndof(ly))
plu = AffineLU()
res_at_V! = (F_,u_,V_) -> residual_stage3!(F_, u_, V_, cl_mesh, gdl_mesh, c_eq_vec, ly)

u0 = make_u0_gas_equilibrium(c_eq_vec, ly)
V_anchor = V_RHE_grid[1]

t0 = time()
u, info_anchor = anchor_solve!(u0, V_anchor, res_at_V!, ly, plu; alpha_init=1e-2, max_iter=200, rtol=1e-8)
@printf("Anchor V=%.4f: converged=%s iters=%d ||F||=%.3e reason=%s (t=%.1fs)\n",
        V_anchor, info_anchor.converged, info_anchor.iterations, info_anchor.residual, info_anchor.reason, time()-t0)
@assert info_anchor.converged "Stage 3 anchor failed to converge"
post_solve_assert!(u, V_anchor, res_at_V!, ly, plu)
save_solution(CACHE_DIR, PREFIX, u, V_anchor)

Vs_rest, all_ok = sweep_voltages!(u, V_anchor, V_RHE_grid[2:end], res_at_V!, ly, plu;
                                   cache_dir=CACHE_DIR, prefix=PREFIX, rtol=1e-8, tag="[stage3]", verbose=false)
elapsed = time() - t0
n_conv = length(Vs_rest) + 1
@printf("Stage 3 sweep complete: %d/%d voltages converged in %.1f s\n", n_conv, length(V_RHE_grid), elapsed)

outdir = joinpath(@__DIR__, "output_stage3")
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

# ── Physicality checks (§11 Stage 3) ────────────────────────────────────────
@printf("\n--- Physicality checks ---\n")
V_hi = Vs_all[argmax(iCO .+ iH2)]
uV = load_solution(CACHE_DIR, PREFIX, V_hi)
w1_1,w2_1,w3_1,p_1 = cl_gas(uV, ly, 1)
w1_N,w2_N,w3_N,p_N = cl_gas(uV, ly, N_CL)
@printf("At V_RHE=%.3f: omega_CO2 at CL inlet(cell1)=%.4f, at CL exit(cellN)=%.4f (should decrease across CL at high current)\n",
        V_hi, w1_1, w1_N)
@printf("omega_CO, omega_H2 at CL exit: %.4e, %.4e (should be >0, increasing with current)\n", w2_N, w3_N)

# pressure should be pinned exactly at p_total everywhere in Stage 3
pmax = -Inf; pmin = Inf
for ig in 1:N_GDL
    (_,_,_,p) = gdl_gas(uV, ly, ig); global pmax=max(pmax,p); global pmin=min(pmin,p)
end
for ix in 1:N_CL
    (_,_,_,p) = cl_gas(uV, ly, ix); global pmax=max(pmax,p); global pmin=min(pmin,p)
end
@printf("Pressure range across GDL+CL = [%.4f, %.4f] Pa (should be EXACTLY p_total=%.4f, Stage-3 placeholder)\n", pmin, pmax, p_total)

# Compare Stage 2 vs Stage 3 liquid-only physics at same V (gas coupling should change CO2 supply)
V_test = -1.500
u2 = load_solution(joinpath(@__DIR__,"output_stage2","cache"), "s2", V_test)
u3 = load_solution(CACHE_DIR, PREFIX, V_test)
Vshe_test = V_test - V_RHE_to_SHE_offset
i2 = compute_i_CO_simple(u2, Vshe_test, cl_mesh)/10.0 + compute_i_H2_simple(u2, Vshe_test, cl_mesh)/10.0
i3 = compute_i_CO_simple(u3, Vshe_test, cl_mesh)/10.0 + compute_i_H2_simple(u3, Vshe_test, cl_mesh)/10.0
@printf("At V_RHE=%.3f: Stage2 i_total=%.2f, Stage3 i_total=%.2f mA/cm2\n", V_test, i2, i3)

@printf("\nStage 3 driver finished. Run plot_stage3.py to generate figures, then inspect.\n")
