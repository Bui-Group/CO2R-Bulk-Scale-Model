# run_fit.jl — Fit step (after Stage 4): i0_CO2R, lambda_MHC, HER_prefactor, alpha_c_HER against
# the 5 experimental points, using the Stage-4 (simplified-kinetics) residual for speed. §17.
#
# Note (see BUILD_DIALOGUE.md): §17.5 lists 4 manual/staged steps (1: sweep params by hand,
# 2: fit CO2R alone, 3: fit HER alone, 4: joint Nelder-Mead) as a workflow for a human building
# intuition. Steps 1-3 don't change any artifact the guide asks this stage to produce -- the
# deliverable is the Step-4 joint 4-parameter fit -- so this autonomous run goes straight to it.

include(joinpath(@__DIR__, "params.jl"))
include(joinpath(@__DIR__, "mesh.jl"))
include(joinpath(@__DIR__, "chemistry.jl"))
include(joinpath(@__DIR__, "kinetics.jl"))
include(joinpath(@__DIR__, "transport.jl"))
include(joinpath(@__DIR__, "gas_transport.jl"))
include(joinpath(@__DIR__, "assembly.jl"))
include(joinpath(@__DIR__, "newton_affine.jl"))
include(joinpath(@__DIR__, "driver.jl"))
include(joinpath(@__DIR__, "solver.jl"))
include(joinpath(@__DIR__, "postprocess.jl"))

using Printf, Optim

const CACHE_DIR4 = joinpath(@__DIR__, "output_stage4", "cache")
const CACHE_DIRFIT = joinpath(@__DIR__, "output_fit", "cache")

cl_mesh = make_cl_mesh(MESH_N_CL)
gdl_mesh = make_gdl_mesh(MESH_N_GDL)
ly4 = Layout14(MESH_N_CL, MESH_N_GDL, true)
c_eq_vec = equilibrium_c_vec()

grid_Vs = list_cache(CACHE_DIR4, "s4")
isempty(grid_Vs) && error("No Stage 4 cache found in $CACHE_DIR4 — run Stage 4 first!")
@printf("Stage 4 cache: %d solutions, V in [%.3f, %.3f]\n", length(grid_Vs), minimum(grid_Vs), maximum(grid_Vs))

const plu_fit = AffineLU()

function polish_one!(V_rhe::Float64, u_warm::Vector{Float64})
    res! = (F_, u_) -> residual_stage4!(F_, u_, V_rhe, cl_mesh, gdl_mesh, c_eq_vec, ly4)
    return newton_affine!(u_warm, res!; ly=ly4, plu=plu_fit, rtol=RTOL_PROD, max_iter=MAX_ITER_SWEEP,
                           alpha_init=1.0, n_log_dofs=5*MESH_N_CL)
end

# ── §17.3 pre-fitting voltage polish: solve at the exact experimental voltages ──
println("\n--- Pre-fitting polish at exact experimental voltages ---")
exp_cache = Dict{Float64,Vector{Float64}}()
for V_rhe in E_exp_RHE
    u_warm, V_near = load_nearest(CACHE_DIR4, "s4", V_rhe)
    info = polish_one!(V_rhe, u_warm)
    @printf("  V_RHE=%.3f (warm from %.3f): converged=%s iters=%d ||F||=%.2e\n",
            V_rhe, V_near, info.converged, info.iterations, info.residual)
    info.converged || error("Initial polish failed at V_RHE=$V_rhe")
    exp_cache[V_rhe] = u_warm
end

# ── §17.2 bounds + sloped penalties (flat penalties stall NelderMead at iteration 0 -- see the
# guide's explicit AffineSimplexer warning and BUILD_DIALOGUE.md) ──
const FIT_LOWER = [10.0, 0.5, 8.0, 0.01]
const FIT_UPPER = [5000.0, 3.0, 18.0, 0.50]
const PARAMS0 = [540.0, 1.8, log10(2.409e13), 0.0872]

function bounds_penalty(params::Vector{Float64}, lower::Vector{Float64}, upper::Vector{Float64})
    pen = 0.0
    for j in 1:4
        if params[j] < lower[j]
            pen += ((lower[j] - params[j]) / (upper[j] - lower[j]))^2
        elseif params[j] > upper[j]
            pen += ((params[j] - upper[j]) / (upper[j] - lower[j]))^2
        end
    end
    return pen
end

const p_last = Ref(copy(PARAMS0))
const n_evals = Ref(0)

function fitting_objective(params::Vector{Float64})
    n_evals[] += 1
    pen = bounds_penalty(params, FIT_LOWER, FIT_UPPER)
    pen > 0 && return 1e6 * (1 + pen)

    set_i0_CO2R!(params[1])
    set_lambda_MHC!(params[2])
    set_HER_prefactor!(10.0^params[3])
    set_alpha_c_HER!(params[4])

    for V_rhe in E_exp_RHE
        u_warm = copy(exp_cache[V_rhe])
        info = polish_one!(V_rhe, u_warm)
        if !info.converged
            return 1e6 * (1 + sum(((params .- p_last[]) ./ (FIT_UPPER .- FIT_LOWER)).^2))
        end
        exp_cache[V_rhe] = u_warm
    end
    p_last[] = copy(params)

    sse = 0.0
    for (k, V_rhe) in enumerate(E_exp_RHE)
        u = exp_cache[V_rhe]
        V_applied_SHE = V_rhe - V_RHE_to_SHE_offset
        phi_s_of, phi_L_of = const_phi_closures(V_applied_SHE)
        i_CO_model = compute_i_CO(u, cl_mesh, phi_s_of, phi_L_of) / 10.0
        i_H2_model = compute_i_H2(u, cl_mesh, phi_s_of, phi_L_of) / 10.0
        sse += (log10(max(i_CO_model, 0.1)) - log10(i_CO_exp[k]))^2
        sse += (log10(max(i_H2_model, 0.1)) - log10(i_H2_exp[k]))^2
    end
    return sse
end

println("\n--- Nelder-Mead fit (i0_CO2R, lambda_MHC, log10(HER_prefactor), alpha_c_HER) ---")
t0 = time()
result = optimize(fitting_objective, PARAMS0, NelderMead(),
                   Optim.Options(iterations=200, f_reltol=1e-6, show_trace=true, show_every=10))
elapsed = time() - t0
@printf("\nNelderMead done in %.1fs, %d objective evals, %d reported iterations\n",
        elapsed, n_evals[], Optim.iterations(result))

fitted = Optim.minimizer(result)
set_i0_CO2R!(fitted[1]); set_lambda_MHC!(fitted[2])
set_HER_prefactor!(10.0^fitted[3]); set_alpha_c_HER!(fitted[4])
@printf("\n=== FITTED KINETICS ===\n")
@printf("i0_CO2R      = %.3f A/m^2\n", i0_CO2R())
@printf("lambda_MHC   = %.4f eV\n", lambda_MHC())
@printf("HER_prefactor= %.4e\n", HER_prefactor())
@printf("alpha_c_HER  = %.5f\n", alpha_c_HER())
@printf("final SSE    = %.5f\n", Optim.minimum(result))

mkpath(joinpath(@__DIR__, "output_fit"))
open(joinpath(@__DIR__, "output_fit", "fitted_params.txt"), "w") do f
    @printf(f, "i0_CO2R=%.6f\nlambda_MHC=%.6f\nHER_prefactor=%.6e\nalpha_c_HER=%.6f\nSSE=%.6f\n",
            i0_CO2R(), lambda_MHC(), HER_prefactor(), alpha_c_HER(), Optim.minimum(result))
end

# ── §17.6 acceptance criteria (auto-verifiable) ──
println("\n--- Fit quality acceptance criteria ---")
tol_CO = [0.30, 0.30, 0.30, 0.30, 0.80]
tol_H2 = [0.30, 0.30, 0.30, 0.30, 0.30]
point_pass = falses(5)
for (k, V_rhe) in enumerate(E_exp_RHE)
    u = exp_cache[V_rhe]
    V_applied_SHE = V_rhe - V_RHE_to_SHE_offset
    phi_s_of, phi_L_of = const_phi_closures(V_applied_SHE)
    iCO = compute_i_CO(u, cl_mesh, phi_s_of, phi_L_of) / 10.0
    iH2 = compute_i_H2(u, cl_mesh, phi_s_of, phi_L_of) / 10.0
    relCO = abs(iCO - i_CO_exp[k]) / i_CO_exp[k]
    relH2 = abs(iH2 - i_H2_exp[k]) / i_H2_exp[k]
    okCO = relCO <= tol_CO[k]; okH2 = relH2 <= tol_H2[k]
    point_pass[k] = okCO && okH2
    @printf("  V_RHE=%.3f: i_CO model=%.2f exp=%.2f relerr=%.1f%% [%s]  i_H2 model=%.2f exp=%.2f relerr=%.1f%% [%s]\n",
            V_rhe, iCO, i_CO_exp[k], relCO*100, okCO ? "OK" : "FAIL", iH2, i_H2_exp[k], relH2*100, okH2 ? "OK" : "FAIL")
end
all_pass = all(point_pass)
FE_CO_peak_val, FE_CO_peak_idx = findmax([begin
    u = exp_cache[V_rhe]; V_applied_SHE = V_rhe - V_RHE_to_SHE_offset
    phi_s_of, phi_L_of = const_phi_closures(V_applied_SHE)
    iCO = compute_i_CO(u, cl_mesh, phi_s_of, phi_L_of); iH2 = compute_i_H2(u, cl_mesh, phi_s_of, phi_L_of)
    iCO/(iCO+iH2)*100
end for V_rhe in E_exp_RHE])
@printf("FE_CO at the 5 experimental points: peak %.1f%% at V_RHE=%.3f (target 70-85%%, near V_RHE=-1.205)\n",
        FE_CO_peak_val, E_exp_RHE[FE_CO_peak_idx])
println(all_pass ? "Per-point tolerances: ALL PASS" : "Per-point tolerances: SOME FAILED (see above)")

# ── Final 89-voltage Stage-4 sweep at fitted kinetics (§17.8 diagnostic artifact) ──
println("\n--- Final 89-voltage Stage 4 sweep at fitted kinetics -> output_fit/cache ---")
u0 = make_u0_stage34(ly4, c_eq_vec)
let F0 = zeros(ly4.N_TOT)
    residual_stage4!(F0, u0, V_RHE_grid[1], cl_mesh, gdl_mesh, c_eq_vec, ly4)
end
res_at_V! = (F, u, V) -> residual_stage4!(F, u, V, cl_mesh, gdl_mesh, c_eq_vec, ly4)
Vs_fit, sols_fit = run_continuation!(res_at_V!, u0, V_RHE_grid[1], collect(V_RHE_grid), ly4, 5*MESH_N_CL,
                                      CACHE_DIRFIT, "fit")
@printf("Fitted-kinetics sweep: %d/%d converged\n", length(Vs_fit), length(V_RHE_grid))

# Smoothness / branch checks (§17.6 criteria 3-4)
i_CO_fit = Float64[]; i_H2_fit = Float64[]
for (k, V) in enumerate(Vs_fit)
    V_SHE = V - V_RHE_to_SHE_offset
    phi_s_of, phi_L_of = const_phi_closures(V_SHE)
    push!(i_CO_fit, compute_i_CO(sols_fit[k], cl_mesh, phi_s_of, phi_L_of)/10.0)
    push!(i_H2_fit, compute_i_H2(sols_fit[k], cl_mesh, phi_s_of, phi_L_of)/10.0)
end
di_CO = diff(i_CO_fit)
sign_ch_CO = count(k -> sign(di_CO[k]) != sign(di_CO[k+1]) && di_CO[k]!=0 && di_CO[k+1]!=0, 1:length(di_CO)-1)
mono_H2 = all(diff(i_H2_fit) .> 0)
@printf("Sign changes in di_CO/dV: %d (target: exactly 1)\n", sign_ch_CO)
@printf("i_H2(V) strictly monotone increasing in |V_RHE|: %s\n", mono_H2)
peakFE, peakidx = findmax(i_CO_fit ./ (i_CO_fit .+ i_H2_fit) .* 100)
@printf("Peak FE_CO over full 89-V sweep: %.1f%% at V_RHE=%.3f\n", peakFE, Vs_fit[peakidx])

write_polarization_csv(joinpath(@__DIR__, "output_fit", "polarization.csv"), Vs_fit, sols_fit, cl_mesh, stage14_phi_of)
write_liquid_profiles_csv(joinpath(@__DIR__, "output_fit", "liquid_profiles.csv"), Vs_fit, sols_fit, cl_mesh, stage14_phi_of)
write_gas_profiles_csv(joinpath(@__DIR__, "output_fit", "gas_profiles.csv"), Vs_fit, sols_fit, gdl_mesh, cl_mesh, ly4)
println("\nFit step done.")
