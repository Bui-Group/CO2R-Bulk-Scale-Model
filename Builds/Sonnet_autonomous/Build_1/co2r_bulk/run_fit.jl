##############################################################################
# run_fit.jl — Fit step: Nelder-Mead kinetic parameter fit (§17)
# Uses the Stage-4 residual (simplified kinetics: phi_L=0, phi_s=V_applied_SHE)
##############################################################################
include("common.jl")
using Optim

const S4_CACHE = joinpath(@__DIR__, "output_stage4", "cache")
const FIT_CACHE = joinpath(@__DIR__, "output_fit", "cache")
const FIT_PREFIX = "fit"

@printf("=== Fit step ===\n")
@printf("Initial kinetics: i0_CO2R=%.1f  lambda_MHC=%.3f  HER_pf=%.3e  alpha_c=%.4f\n",
        i0_CO2R(), lambda_MHC(), HER_prefactor(), alpha_c_HER())

cl_mesh = make_cl_mesh()
gdl_mesh = make_gdl_mesh()
c_eq = solve_buffer_equilibrium(1000.0, 1000.0)
c_eq_vec = c_eq_vector(c_eq)
ly = GasLayout(N_CL, N_GDL)
plu = AffineLU()

grid_Vs = list_cache_voltages(S4_CACHE, "s4")
isempty(grid_Vs) && error("No Stage 4 cache found — run Stage 4 first!")

# ── §17.2 bounds and sloped penalty ─────────────────────────────────────────
const p_lower = [10.0, 0.5, 8.0, 0.01]
const p_upper = [5000.0, 3.0, 18.0, 0.50]
bounds_violation(params) = sum(begin
    if params[j] < p_lower[j]
        ((p_lower[j]-params[j])/(p_upper[j]-p_lower[j]))^2
    elseif params[j] > p_upper[j]
        ((params[j]-p_upper[j])/(p_upper[j]-p_lower[j]))^2
    else
        0.0
    end
end for j in 1:4)

# ── §17.3 pre-fitting voltage polish ────────────────────────────────────────
function nearest_grid_u(V_target, grid_Vs, cache_dir, prefix)
    idx = argmin(abs.(grid_Vs .- V_target))
    return copy(load_solution(cache_dir, prefix, grid_Vs[idx]))
end

function polish_all!(exp_cache::Dict{Float64,Vector{Float64}})
    ok_all = true
    for V_rhe in E_exp_RHE
        u_warm = copy(exp_cache[V_rhe])
        res! = (F_,u_) -> residual_stage4!(F_, u_, V_rhe, cl_mesh, gdl_mesh, c_eq_vec, ly)
        info = newton_affine!(u_warm, res!; ly=ly, plu=plu, rtol=1e-8, max_iter=40,
                               alpha_init=1.0, n_log_dofs=n_log_dofs(ly))
        if !info.converged
            ok_all = false
        else
            exp_cache[V_rhe] = u_warm
        end
    end
    return ok_all
end

exp_cache = Dict{Float64,Vector{Float64}}()
for V_rhe in E_exp_RHE
    exp_cache[V_rhe] = nearest_grid_u(V_rhe, grid_Vs, S4_CACHE, "s4")
end
ok0 = polish_all!(exp_cache)
@printf("Initial polish at 5 experimental voltages: all converged = %s\n", ok0)
@assert ok0 "initial polish (pre-fit kinetics) failed to converge at all 5 experimental voltages"

# ── §17.4 fitting objective (log-space SSE + sloped penalties) ─────────────
p_last = Ref([540.0, 1.8, log10(2.409e13), 0.0872])
n_evals = Ref(0)

function fitting_objective(params::Vector{Float64})
    n_evals[] += 1
    pen = bounds_violation(params)
    if pen > 0
        return 1e6*(1+pen)
    end
    set_i0_CO2R!(params[1])
    set_lambda_MHC!(params[2])
    set_HER_prefactor!(10.0^params[3])
    set_alpha_c_HER!(params[4])

    ok = polish_all!(exp_cache)
    if !ok
        return 1e6*(1 + sum(((params .- p_last[]) ./ (p_upper .- p_lower)).^2))
    end
    p_last[] = copy(params)

    sse = 0.0
    for V_rhe in E_exp_RHE
        u = exp_cache[V_rhe]
        V_applied_SHE = V_rhe - V_RHE_to_SHE_offset
        i_CO_model = compute_i_CO_simple(u, V_applied_SHE, cl_mesh) / 10.0
        i_H2_model = compute_i_H2_simple(u, V_applied_SHE, cl_mesh) / 10.0
        k = findfirst(==(V_rhe), E_exp_RHE)
        sse += (log10(max(i_CO_model,0.1)) - log10(i_CO_exp[k]))^2
        sse += (log10(max(i_H2_model,0.1)) - log10(i_H2_exp[k]))^2
    end
    return sse
end

params0 = [540.0, 1.8, log10(2.409e13), 0.0872]
sse0 = fitting_objective(params0)
@printf("SSE at initial guess (pre-fit kinetics): %.4f\n", sse0)

t0 = time()
result = optimize(fitting_objective, params0, NelderMead(),
                   Optim.Options(iterations=200, g_tol=1e-6, f_reltol=1e-6, show_trace=false))
@printf("Nelder-Mead finished: %d objective evals, %.1f s, converged=%s, SSE=%.6f\n",
        n_evals[], time()-t0, Optim.converged(result), Optim.minimum(result))

params_fit = Optim.minimizer(result)
set_i0_CO2R!(params_fit[1])
set_lambda_MHC!(params_fit[2])
set_HER_prefactor!(10.0^params_fit[3])
set_alpha_c_HER!(params_fit[4])
@printf("\nFitted kinetics: i0_CO2R=%.2f A/m^2  lambda_MHC=%.4f eV  HER_pf=%.4e  alpha_c=%.5f\n",
        i0_CO2R(), lambda_MHC(), HER_prefactor(), alpha_c_HER())

# ── §17.6 acceptance criteria ────────────────────────────────────────────────
@printf("\n--- Fit acceptance criteria (§17.6) ---\n")
polish_all!(exp_cache)   # final polish at the fitted optimum
all_pass = true
for (k, V_rhe) in enumerate(E_exp_RHE)
    u = exp_cache[V_rhe]
    V_applied_SHE = V_rhe - V_RHE_to_SHE_offset
    iCOm = compute_i_CO_simple(u, V_applied_SHE, cl_mesh)/10.0
    iH2m = compute_i_H2_simple(u, V_applied_SHE, cl_mesh)/10.0
    relCO = abs(iCOm - i_CO_exp[k])/i_CO_exp[k]
    relH2 = abs(iH2m - i_H2_exp[k])/i_H2_exp[k]
    tolCO = (k == 5) ? 0.80 : 0.30
    passCO = relCO <= tolCO; passH2 = relH2 <= 0.30
    global all_pass &= (passCO && passH2)
    @printf("V_RHE=%.3f: i_CO model=%.2f exp=%.2f (relerr=%.1f%%, tol=%.0f%%, %s) | i_H2 model=%.2f exp=%.2f (relerr=%.1f%%, %s)\n",
            V_rhe, iCOm, i_CO_exp[k], 100relCO, 100tolCO, passCO ? "PASS" : "FAIL",
            iH2m, i_H2_exp[k], 100relH2, passH2 ? "PASS" : "FAIL")
end
@printf("All per-point tolerances met: %s\n", all_pass)

# ── §17.8 diagnostic 89-voltage refit sweep (NOT the Stage 5 warm start) ───
@printf("\nRunning diagnostic 89-voltage Stage-4-physics sweep at fitted kinetics (output_fit/cache/)...\n")
u0 = make_u0_gas_equilibrium(c_eq_vec, ly)
res_at_V! = (F_,u_,V_) -> residual_stage4!(F_, u_, V_, cl_mesh, gdl_mesh, c_eq_vec, ly)
V_anchor = V_RHE_grid[1]
u, info_anchor = anchor_solve!(u0, V_anchor, res_at_V!, ly, plu; alpha_init=1e-2, max_iter=200, rtol=1e-8)
@assert info_anchor.converged "Fit-sweep anchor failed"
post_solve_assert!(u, V_anchor, res_at_V!, ly, plu)
save_solution(FIT_CACHE, FIT_PREFIX, u, V_anchor)
Vs_rest, all_ok = sweep_voltages!(u, V_anchor, V_RHE_grid[2:end], res_at_V!, ly, plu;
                                   cache_dir=FIT_CACHE, prefix=FIT_PREFIX, rtol=1e-8, tag="[fit-sweep]", verbose=false)
n_conv = length(Vs_rest)+1
@printf("Diagnostic fit sweep: %d/%d voltages converged\n", n_conv, length(V_RHE_grid))

outdir = joinpath(@__DIR__, "output_fit")
Vs_all = list_cache_voltages(FIT_CACHE, FIT_PREFIX)
iCO = Float64[]; iH2 = Float64[]
for V in Vs_all
    uV = load_solution(FIT_CACHE, FIT_PREFIX, V)
    Vshe = V - V_RHE_to_SHE_offset
    push!(iCO, compute_i_CO_simple(uV, Vshe, cl_mesh)/10.0)
    push!(iH2, compute_i_H2_simple(uV, Vshe, cl_mesh)/10.0)
end
export_polarization_csv(joinpath(outdir, "polarization.csv"), Vs_all, iCO, iH2)
rows_fn = V -> liquid_diagnostics_simple(load_solution(FIT_CACHE, FIT_PREFIX, V), cl_mesh, V - V_RHE_to_SHE_offset)
export_liquid_profiles_csv(joinpath(outdir, "liquid_profiles.csv"), cl_mesh, Vs_all, rows_fn)
gas_rows_fn = V -> gas_diagnostics(load_solution(FIT_CACHE, FIT_PREFIX, V), ly, gdl_mesh, cl_mesh)
export_gas_profiles_csv(joinpath(outdir, "gas_profiles.csv"), Vs_all, gas_rows_fn, N_GDL+N_CL)

# smoothness check
di_CO = diff(iCO)
nsign = sum(diff(sign.(di_CO)) .!= 0)
@printf("Sign changes in di_CO/dV across the 89-pt fit sweep: %d (guide requires exactly 1)\n", nsign)
FE_CO = 100 .* iCO ./ max.(iCO.+iH2, 1e-12)
@printf("Peak FE_CO = %.1f%% at V_RHE=%.3f (guide target: 70-85%%, near V_RHE=-1.205)\n", maximum(FE_CO), Vs_all[argmax(FE_CO)])

# ── Persist fitted kinetics to disk for run_stage5.jl (separate process) ───
# The guide's Ref-backed-parameter design (§13) assumes these setters are
# called within a session; since each run_stageN.jl is its own `julia`
# process (§13: "exit -- do NOT automatically chain to the next stage"), the
# guide does not specify a cross-process persistence format. Judgment call:
# write a small plain-text key=value file that run_stage5.jl reads at
# startup and applies via the same Ref setters, matching "print the active
# kinetic parameters at startup to confirm the fitted values were loaded"
# (§17.8).
fitted_path = joinpath(@__DIR__, "output_fit", "fitted_kinetics.txt")
open(fitted_path, "w") do f
    @printf(f, "i0_CO2R=%.10f\n", i0_CO2R())
    @printf(f, "lambda_MHC=%.10f\n", lambda_MHC())
    @printf(f, "HER_prefactor=%.10e\n", HER_prefactor())
    @printf(f, "alpha_c_HER=%.10f\n", alpha_c_HER())
end
@printf("Fitted kinetics written to %s\n", fitted_path)

@printf("\nFitted parameters saved. Run plot_fit.py, then proceed to Stage 5 with these kinetics.\n")
