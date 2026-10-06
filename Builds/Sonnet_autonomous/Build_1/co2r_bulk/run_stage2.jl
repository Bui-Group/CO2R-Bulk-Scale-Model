##############################################################################
# run_stage2.jl — Stage 2: Sherwood-Reynolds Neumann electrolyte BC (§11)
##############################################################################
include("common.jl")

const CACHE_DIR = joinpath(@__DIR__, "output_stage2", "cache")
const PREFIX = "s2"
const STAGE1_CACHE = joinpath(@__DIR__, "output_stage1", "cache")

@printf("=== Stage 2 ===\n")
@printf("i0_CO2R = %.1f A/m^2, lambda_MHC = %.3f eV\n", i0_CO2R(), lambda_MHC())
@printf("HER_pf = %.3e, alpha_c = %.4f\n", HER_prefactor(), alpha_c_HER())
@printf("N_CL = %d, N_GDL = %d\n", N_CL, N_GDL)
@printf("Cache dir: %s\n", CACHE_DIR)
if isdir(CACHE_DIR) && !isempty(readdir(CACHE_DIR))
    @printf("WARNING: existing cache found. Verify kinetics match before reusing!\n")
end

cl_mesh = make_cl_mesh()
c_eq = solve_buffer_equilibrium(1000.0, 1000.0)
c_eq_vec = c_eq_vector(c_eq)

# Biot number check (§11 Stage 2 physicality checks)
Bi = [k_MT_vec[i]*L_CL/D_eff_vec[i] for i in 1:5]
@printf("Biot numbers Bi_i = k_MT*L_CL/D_eff: %s\n", string(round.(Bi, digits=3)))

ly = LiquidLayout(N_CL)
plu = AffineLU()
res_at_V! = (F_,u_,V_) -> residual_stage2!(F_, u_, V_, cl_mesh, c_eq_vec)

u0 = make_u0_liquid_equilibrium(c_eq_vec, N_CL)
V_anchor = V_RHE_grid[1]

t0 = time()
u, info_anchor = anchor_solve!(u0, V_anchor, res_at_V!, ly, plu; alpha_init=1e-2, max_iter=200, rtol=1e-8)
@printf("Anchor V=%.4f: converged=%s iters=%d ||F||=%.3e reason=%s\n",
        V_anchor, info_anchor.converged, info_anchor.iterations, info_anchor.residual, info_anchor.reason)
@assert info_anchor.converged "Stage 2 anchor failed to converge"
post_solve_assert!(u, V_anchor, res_at_V!, ly, plu)
save_solution(CACHE_DIR, PREFIX, u, V_anchor)

Vs_rest, all_ok = sweep_voltages!(u, V_anchor, V_RHE_grid[2:end], res_at_V!, ly, plu;
                                   cache_dir=CACHE_DIR, prefix=PREFIX, rtol=1e-8, tag="[stage2]", verbose=false)
elapsed = time() - t0
n_conv = length(Vs_rest) + 1
@printf("Stage 2 sweep complete: %d/%d voltages converged in %.1f s\n", n_conv, length(V_RHE_grid), elapsed)

outdir = joinpath(@__DIR__, "output_stage2")
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

# ── Physicality checks (§11 Stage 2) ────────────────────────────────────────
@printf("\n--- Physicality checks ---\n")
V_hi = Vs_all[argmax(iCO .+ iH2)]
uV = load_solution(CACHE_DIR, PREFIX, V_hi)
N = cl_mesh.N
c = unpack_log_conc(uV, N)
@printf("At V_RHE=%.3f (max current): c_CO2(L_CL)=%.4f mol/m3 (should approach 0 at high current)\n", V_hi, c[5,N])
@printf("c_OH(L_CL)=%.4f mol/m3 vs bulk c_eq_OH=%.4e (should sit noticeably ABOVE bulk eq)\n", c[2,N], c_eq.OH)
flux_OH_check = -k_MT_vec[2]*(c_eq_vec[2]-c[2,N])
@printf("Flux consistency: J_OH_face(N+1) = %.6e (k_MT*(c_local-c_eq) form)\n", flux_OH_check)

# Compare Stage 1 vs Stage 2 polarization at same V for a sanity delta
V_test = -1.500
u1 = load_solution(STAGE1_CACHE, "s1", V_test)
u2 = load_solution(CACHE_DIR, PREFIX, V_test)
Vshe_test = V_test - V_RHE_to_SHE_offset
i1 = compute_i_CO_simple(u1, Vshe_test, cl_mesh)/10.0 + compute_i_H2_simple(u1, Vshe_test, cl_mesh)/10.0
i2 = compute_i_CO_simple(u2, Vshe_test, cl_mesh)/10.0 + compute_i_H2_simple(u2, Vshe_test, cl_mesh)/10.0
@printf("At V_RHE=%.3f: Stage1 i_total=%.2f, Stage2 i_total=%.2f mA/cm2 (finite k_MT should reduce current somewhat vs Dirichlet)\n",
        V_test, i1, i2)

@printf("\nStage 2 driver finished. Run plot_stage2.py to generate figures, then inspect.\n")
