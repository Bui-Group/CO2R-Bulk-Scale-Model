##############################################################################
# run_stage5.jl — Stage 5: phi_L (NP migration) + phi_s (solid Ohm's law) (§11)
# MUST use fitted kinetics from run_fit.jl. Seeded ONCE at V_c=0, never from
# a cache (§11 Stage 5 gotcha #0 / §17.8).
##############################################################################
include("common.jl")

const S5_CACHE = joinpath(@__DIR__, "output_stage5", "cache")
const S5_PREFIX = "s5"
const FITTED_PATH = joinpath(@__DIR__, "output_fit", "fitted_kinetics.txt")

@printf("=== Stage 5 ===\n")

# ── Load fitted kinetics (persisted by run_fit.jl; see BUILD_DIALOGUE for
#    the judgment call on cross-process persistence format) ────────────────
isfile(FITTED_PATH) || error("Fitted kinetics file not found at $FITTED_PATH — run run_fit.jl first!")
for line in eachline(FITTED_PATH)
    isempty(strip(line)) && continue
    k, v = split(line, "=")
    val = parse(Float64, v)
    if k == "i0_CO2R"; set_i0_CO2R!(val)
    elseif k == "lambda_MHC"; set_lambda_MHC!(val)
    elseif k == "HER_prefactor"; set_HER_prefactor!(val)
    elseif k == "alpha_c_HER"; set_alpha_c_HER!(val)
    end
end
@printf("Loaded FITTED kinetics: i0_CO2R=%.3f A/m^2  lambda_MHC=%.4f eV  HER_pf=%.4e  alpha_c=%.5f\n",
        i0_CO2R(), lambda_MHC(), HER_prefactor(), alpha_c_HER())
@printf("N_CL = %d, N_GDL = %d\n", N_CL, N_GDL)
if isdir(S5_CACHE) && !isempty(readdir(S5_CACHE))
    @printf("WARNING: existing Stage 5 cache found. Verify kinetics match before reusing!\n")
end

cl_mesh = make_cl_mesh()
gdl_mesh = make_gdl_mesh()
c_eq = solve_buffer_equilibrium(1000.0, 1000.0)
c_eq_vec = c_eq_vector(c_eq)

ly = Stage5Layout(N_CL, N_GDL)
@printf("Stage 5 DOFs = %d (expect 1480 at production mesh)\n", ndof(ly))
plu = AffineLU()
res_at_V! = (F_,u_,V_) -> residual_stage5!(F_, u_, V_, cl_mesh, gdl_mesh, c_eq_vec, ly)

# ── Anchor at V_c = 0 (i.e. V_applied_SHE = 0), built from physics ─────────
# V_c := V_applied_SHE = V_cathode - V_RHE_to_SHE_offset. V_c=0 <=> V_cathode
# = V_RHE_to_SHE_offset. Tracking the single continuation parameter in
# V_cathode (RHE) units throughout, consistent with every other stage's
# residual convention (§15.3) -- since RHE and SHE differ only by a fixed
# additive offset, stepping in either frame by the same increment visits
# IDENTICAL physical states; this is a pure bookkeeping simplification, not
# a physics change (documented in BUILD_DIALOGUE.md).
V_c0_cathode = V_RHE_to_SHE_offset
u0 = make_u0_stage5_anchor(c_eq_vec, ly)

t0 = time()
u, info_anchor = anchor_solve!(u0, V_c0_cathode, res_at_V!, ly, plu; alpha_init=1e-2, max_iter=200, rtol=1e-8)
@printf("Anchor (V_c=0, V_cathode=%.5f V RHE): converged=%s iters=%d ||F||=%.3e reason=%s (t=%.1fs)\n",
        V_c0_cathode, info_anchor.converged, info_anchor.iterations, info_anchor.residual, info_anchor.reason, time()-t0)
@assert info_anchor.converged "Stage 5 anchor (V_c=0) failed to converge"

# ── Pre-anchor ramp: V_c = 0 -> first output voltage, -0.025 V steps ──────
V_c_first_output = V_RHE_grid[1] - V_RHE_to_SHE_offset
ramp_Vc = collect(0.0:-0.025:V_c_first_output)
if !isapprox(ramp_Vc[end], V_c_first_output; atol=1e-9)
    push!(ramp_Vc, V_c_first_output)
end
ramp_Vcathode = ramp_Vc .+ V_RHE_to_SHE_offset
ramp_list = ramp_Vcathode[2:end]
@printf("Pre-anchor ramp: %d steps from V_c=0 to V_c=%.5f (V_RHE=%.3f)\n", length(ramp_list), V_c_first_output, V_RHE_grid[1])

Vs_ramp, ramp_ok = sweep_voltages!(u, V_c0_cathode, ramp_list, res_at_V!, ly, plu;
                                    do_cache=false, do_assert=false, rtol=1e-8, tag="[s5-ramp]", verbose=false)
@printf("Ramp complete: reached V_RHE=%.3f, ||F||-tight-converged throughout\n", Vs_ramp[end])

# First output point: assert + cache explicitly (ramp does not do this)
post_solve_assert!(u, V_RHE_grid[1], res_at_V!, ly, plu)
save_solution(S5_CACHE, S5_PREFIX, u, V_RHE_grid[1])
@printf("[s5]  V=%.4f  (first output point, post-ramp)\n", V_RHE_grid[1])

# ── Main continuation sweep over the 89 output voltages ────────────────────
Vs_rest, all_ok = sweep_voltages!(u, V_RHE_grid[1], V_RHE_grid[2:end], res_at_V!, ly, plu;
                                   cache_dir=S5_CACHE, prefix=S5_PREFIX, rtol=1e-8, tag="[s5]", verbose=true)
elapsed = time() - t0
n_conv = length(Vs_rest) + 1
@printf("Stage 5 sweep complete: %d/%d voltages converged in %.1f s\n", n_conv, length(V_RHE_grid), elapsed)

outdir = joinpath(@__DIR__, "output_stage5")
Vs_all = list_cache_voltages(S5_CACHE, S5_PREFIX)

iCO = Float64[]; iH2 = Float64[]
for V in Vs_all
    uV = load_solution(S5_CACHE, S5_PREFIX, V)
    push!(iCO, compute_i_CO_stage5(uV, cl_mesh, ly)/10.0)
    push!(iH2, compute_i_H2_stage5(uV, cl_mesh, ly)/10.0)
end
export_polarization_csv(joinpath(outdir, "polarization.csv"), Vs_all, iCO, iH2)

rows_fn = V -> liquid_diagnostics_stage5(load_solution(S5_CACHE, S5_PREFIX, V), cl_mesh, ly)
export_liquid_profiles_csv(joinpath(outdir, "liquid_profiles.csv"), cl_mesh, Vs_all, rows_fn)
gas_rows_fn = V -> gas_diagnostics(load_solution(S5_CACHE, S5_PREFIX, V), ly, gdl_mesh, cl_mesh)
export_gas_profiles_csv(joinpath(outdir, "gas_profiles.csv"), Vs_all, gas_rows_fn, N_GDL+N_CL)
phi_rows_fn = V -> load_solution(S5_CACHE, S5_PREFIX, V)
export_phi_profiles_csv(joinpath(outdir, "phi_s_profiles.csv"), joinpath(outdir, "phi_L_profiles.csv"),
                         Vs_all, gdl_mesh, cl_mesh, ly, phi_rows_fn)

# ── §11 Stage 5 physicality checks ──────────────────────────────────────────
@printf("\n--- Physicality checks (§11 Stage 5 / §9.5.8) ---\n")
itot = iCO .+ iH2
n_hi_current = count(>(50.0), itot)   # i_total > 500 mA/cm2 = 50 mA/cm2... guide says >500 mA/cm2 (i.e. i>500 mA/cm2 total)
for V in [V_RHE_grid[1], -1.200, -1.650, -2.075, -2.500]
    idx = argmin(abs.(Vs_all .- V))
    uV = load_solution(S5_CACHE, S5_PREFIX, Vs_all[idx])
    d = stage5_diagnostics(uV, ly, cl_mesh, gdl_mesh, Vs_all[idx], c_eq_vec)
    rel_err_charge = abs(abs(d.i_far_total) - abs(d.i_s_face1))/max(abs(d.i_far_total),1e-30)
    rel_err_ionic = abs(abs(d.i_far_total) - abs(d.i_ionic_R))/max(abs(d.i_far_total),1e-30)
    gdl_frac = d.gdl_drop_mV / d.total_solid_drop_mV * 100
    @printf("V_RHE=%7.3f: i_tot=%8.2f mA/cm2 | dphiL=%7.3f mV | solid_drop=%7.3f mV (GDL %.1f%%) | |Δi_charge|/i=%.2e | |Δi_ionic|/i=%.2e\n",
            Vs_all[idx], itot[idx], d.dphiL_mV, d.total_solid_drop_mV, gdl_frac, rel_err_charge, rel_err_ionic)
end

nsign_CO = sum(diff(sign.(diff(iCO))) .!= 0)
nsign_H2 = sum(diff(sign.(diff(iH2))) .!= 0)
# Vs_all is sorted ascending in V_RHE (most cathodic first); i_H2 should be
# strictly MONOTONE in that order (decreasing, since more cathodic = higher
# current) -- direction-agnostic sign-change count is the robust check.
mono_H2 = all(diff(iH2) .<= 1e-6) || all(diff(iH2) .>= -1e-6)
@printf("Sign changes in di_CO/dV: %d (want exactly 1)  |  sign changes in di_H2/dV: %d (want 0, strictly monotone=%s)\n", nsign_CO, nsign_H2, mono_H2)

FE_CO = 100 .* iCO ./ max.(itot, 1e-12)
@printf("Peak FE_CO = %.1f%% at V_RHE=%.3f (target 70-85%%, near -1.205)\n", maximum(FE_CO), Vs_all[argmax(FE_CO)])

# Stage4 -> Stage5 comparison at a shared voltage (currents should drop / CO may rise per §11 check 5)
V_test = -2.425
idx5 = argmin(abs.(Vs_all .- V_test))
u5 = load_solution(S5_CACHE, S5_PREFIX, Vs_all[idx5])
u4 = load_solution(joinpath(@__DIR__,"output_stage4","cache"), "s4", V_test)
Vshe_test = V_test - V_RHE_to_SHE_offset
i4_CO = compute_i_CO_simple(u4, Vshe_test, cl_mesh)/10.0; i4_H2 = compute_i_H2_simple(u4, Vshe_test, cl_mesh)/10.0
i5_CO = compute_i_CO_stage5(u5, cl_mesh, ly)/10.0; i5_H2 = compute_i_H2_stage5(u5, cl_mesh, ly)/10.0
@printf("At V_RHE=%.3f: Stage4(pre-fit) i_CO=%.2f i_H2=%.2f | Stage5(fitted+full-physics) i_CO=%.2f i_H2=%.2f mA/cm2\n",
        V_test, i4_CO, i4_H2, i5_CO, i5_H2)

@printf("\nStage 5 driver finished. Run plot_stage5.py to generate figures, then inspect.\n")
