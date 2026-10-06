##############################################################################
# run_stage1.jl — Stage 1: liquid NP diffusion-only + Dirichlet BC (§11)
##############################################################################
include("common.jl")

const CACHE_DIR = joinpath(@__DIR__, "output_stage1", "cache")
const PREFIX = "s1"

@printf("=== Stage 1 ===\n")
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
@printf("Bulk equilibrium: pH=%.4f  c_H=%.4e  c_OH=%.4e  c_HCO3=%.3f  c_CO3=%.3f  c_CO2=%.3f mol/m3\n",
        c_eq.pH, c_eq.H, c_eq.OH, c_eq.HCO3, c_eq.CO3, c_eq.CO2)
S_check = buffer_sources(c_eq_vec)
@printf("Buffer residual check: max|S_buf| = %.3e mol/m3/s (must be < 1e-6)\n", maximum(abs, S_check))
@assert maximum(abs, S_check) < 1e-6 "equilibrium solve did not satisfy all 5 buffer rates"
@printf("c_CO2_henry (GDL boundary, van't Hoff only) = %.4f mol/m3\n", c_CO2_henry)

ly = LiquidLayout(N_CL)
plu = AffineLU()
res_at_V! = (F_,u_,V_) -> residual_stage1!(F_, u_, V_, cl_mesh, c_eq_vec)

u0 = make_u0_liquid_equilibrium(c_eq_vec, N_CL)
V_anchor = V_RHE_grid[1]

t0 = time()
u, info_anchor = anchor_solve!(u0, V_anchor, res_at_V!, ly, plu; alpha_init=1e-2, max_iter=200, rtol=1e-8)
@printf("Anchor V=%.4f: converged=%s iters=%d ||F||=%.3e reason=%s\n",
        V_anchor, info_anchor.converged, info_anchor.iterations, info_anchor.residual, info_anchor.reason)
@assert info_anchor.converged "Stage 1 anchor failed to converge"
post_solve_assert!(u, V_anchor, res_at_V!, ly, plu)
save_solution(CACHE_DIR, PREFIX, u, V_anchor)

Vs_rest, all_ok = sweep_voltages!(u, V_anchor, V_RHE_grid[2:end], res_at_V!, ly, plu;
                                   cache_dir=CACHE_DIR, prefix=PREFIX, rtol=1e-8, tag="[stage1]")
elapsed = time() - t0
n_conv = length(Vs_rest) + 1
@printf("Stage 1 sweep complete: %d/%d voltages converged in %.1f s\n", n_conv, length(V_RHE_grid), elapsed)

# ── Export CSVs for plotting (§12.7) ────────────────────────────────────────
outdir = joinpath(@__DIR__, "output_stage1")
Vs_all = list_cache_voltages(CACHE_DIR, PREFIX)
@printf("Cache has %d voltages spanning [%.3f, %.3f]\n", length(Vs_all), Vs_all[1], Vs_all[end])

iCO = Float64[]; iH2 = Float64[]
for V in Vs_all
    uV = load_solution(CACHE_DIR, PREFIX, V)
    Vshe = V - V_RHE_to_SHE_offset
    push!(iCO, compute_i_CO_simple(uV, Vshe, cl_mesh)/10.0)   # A/m^2 -> mA/cm^2
    push!(iH2, compute_i_H2_simple(uV, Vshe, cl_mesh)/10.0)
end
export_polarization_csv(joinpath(outdir, "polarization.csv"), Vs_all, iCO, iH2)

rows_fn = V -> liquid_diagnostics_simple(load_solution(CACHE_DIR, PREFIX, V), cl_mesh, V - V_RHE_to_SHE_offset)
export_liquid_profiles_csv(joinpath(outdir, "liquid_profiles.csv"), cl_mesh, Vs_all, rows_fn)

# ── Physicality checks (§11 Stage 1, §12) ───────────────────────────────────
imax_idx = argmax(iCO .+ iH2)
@printf("\n--- Physicality checks ---\n")
d_mid = liquid_diagnostics_simple(load_solution(CACHE_DIR, PREFIX, Vs_all[imax_idx]), cl_mesh, Vs_all[imax_idx]-V_RHE_to_SHE_offset)
@printf("At V_RHE=%.3f (max current): pH range across CL = [%.2f, %.2f]\n", Vs_all[imax_idx], minimum(d_mid.pH), maximum(d_mid.pH))
@printf("CO2(aq) at GDL face = %.3f, at electrolyte face = %.3f mol/m3 (inlet Henry value = %.3f)\n",
        d_mid.cCO2[1], d_mid.cCO2[end], c_CO2_henry)
FE_CO_max = maximum(100 .* iCO ./ max.(iCO.+iH2, 1e-12))
@printf("Peak FE_CO across sweep = %.1f %% (guide expects ~60-70%% pre-fit, low near onset)\n", FE_CO_max)
@printf("i_total range: [%.3f, %.3f] mA/cm^2\n", minimum(iCO.+iH2), maximum(iCO.+iH2))
nsignchange = sum(diff(sign.(diff(iCO))) .!= 0)
@printf("Sign changes in di_CO/dV (first difference): %d\n", nsignchange)

@printf("\nStage 1 driver finished. Run plot_stage1.py to generate figures, then inspect before Stage 2.\n")
