# ============================================================================
# run_fit.jl — Fit step after Stage 4 (§17).
# Nelder-Mead on (i0_CO2R, lambda_MHC, log10 HER_prefactor, alpha_c_HER)
# against the 5 Romiluyi points, using the STAGE 4 residual (phi_L = 0,
# phi_s = V_applied_SHE) for speed (§11 "The fit comes after Stage 4").
# Per-voltage polish warm-started from the Stage-4 cache (§17.3), same
# newton_affine! kernel + persistent sparse LU + AD Jacobian.
# ============================================================================
include("model.jl")
using Optim

const FITDIR = joinpath(@__DIR__, "output_fit")
mkpath(joinpath(FITDIR, "cache"))

print_kinetics("Fit (starting from the default/initial-guess kinetics)")
println()

eq = solve_buffer_equilibrium(1000.0, 1000.0)
c_eq = [eq.H, eq.OH, eq.HCO3, eq.CO3, eq.CO2]
ctx  = Ctx(4, N_CL_DEFAULT, N_GDL_DEFAULT, c_eq)
ly   = ctx.ly
u_ref = make_u0(ctx)
ws   = SolveWS(ctx, u_ref)

# ── Stage-4 grid cache (§17.3 warm-start source) ───────────────────────────
s4_V = list_cache(cache_dir(4), cache_prefix(4))
@assert length(s4_V) == 89 "expected 89 Stage-4 cache entries, found $(length(s4_V))"
s4_cache = Dict(V => load_solution(V, cache_dir(4), cache_prefix(4)) for V in s4_V)
@printf("Stage-4 cache: %d solutions, V in [%.3f, %.3f]\n", length(s4_V), s4_V[1], s4_V[end])

# ── §17.2 bounds and the SLOPED penalties ──────────────────────────────────
const lower = [  10.0, 0.5,  8.0, 0.01]
const upper = [5000.0, 3.0, 18.0, 0.50]
const params0 = [540.0, 1.8, log10(2.409e13), 0.0872]

apply_params!(p) = (set_i0_CO2R!(p[1]); set_lambda_MHC!(p[2]);
                    set_HER_prefactor!(10.0^p[3]); set_alpha_c_HER!(p[4]))

function bound_penalty(p)
    pen = 0.0
    for j in 1:4
        v = max(lower[j] - p[j], p[j] - upper[j], 0.0)
        pen += (v / (upper[j] - lower[j]))^2
    end
    return pen
end

# ── §17.3 pre-fitting voltage polish ───────────────────────────────────────
const exp_cache = Dict{Float64,Vector{Float64}}()

function polish_all!(; alpha_init = 1.0)
    for V_rhe in E_exp_RHE
        u_warm = copy(exp_cache[V_rhe])
        # V_rhe is passed DIRECTLY: residual! converts RHE -> SHE itself (§15.3)
        info = solve_at!(ws, u_warm, V_rhe; alpha_init = alpha_init, max_iter = 40)
        info.converged || return false
        exp_cache[V_rhe] = u_warm
    end
    return true
end

function init_polish!()
    for V_rhe in E_exp_RHE
        idx = argmin(abs.(s4_V .- V_rhe))
        exp_cache[V_rhe] = copy(s4_cache[s4_V[idx]])
    end
    ok = polish_all!()
    @assert ok "initial polish at the default kinetics failed"
    println("\nPre-fit polish at the 5 experimental voltages (§17.3):")
    for V_rhe in E_exp_RHE
        iCO, iH2, _ = compute_currents(exp_cache[V_rhe], V_rhe, ctx)
        @printf("  V_RHE=%7.3f  i_CO=%8.2f  i_H2=%8.2f mA/cm2\n", V_rhe, iCO/10, iH2/10)
    end
end

# §9.5.7: when a direct polish fails, ramp the kinetics from the last feasible
# vector to the trial vector in 2 -> 4 -> 8 sub-steps, re-solving at each.
const p_last = Ref(copy(params0))
function polish_with_ramp!(p)
    apply_params!(p)
    polish_all!() && return true
    for nsub in (2, 4, 8)
        snapshot = Dict(k => copy(v) for (k, v) in exp_cache)
        ok = true
        for s in 1:nsub
            apply_params!(p_last[] .+ (s/nsub) .* (p .- p_last[]))
            if !polish_all!(alpha_init = 1e-2); ok = false; break; end
        end
        ok && return true
        for (k, v) in snapshot; exp_cache[k] = v; end
    end
    return false
end

# ── §17.4 objective ────────────────────────────────────────────────────────
n_eval = Ref(0)
function make_objective(; fit_CO = true, fit_H2 = true)
    return function (params::Vector{Float64})
        n_eval[] += 1
        pen = bound_penalty(params)
        pen > 0 && return 1e6 * (1 + pen)               # SLOPED, never flat (§17.2)
        if !polish_with_ramp!(params)
            apply_params!(p_last[])
            return 1e6 * (1 + sum(((params .- p_last[]) ./ (upper .- lower)).^2))
        end
        p_last[] = copy(params)
        sse = 0.0
        for (k, V_rhe) in enumerate(E_exp_RHE)
            iCO, iH2, _ = compute_currents(exp_cache[V_rhe], V_rhe, ctx)
            fit_CO && (sse += (log10(max(iCO/10, 0.1)) - log10(i_CO_exp[k]))^2)
            fit_H2 && (sse += (log10(max(iH2/10, 0.1)) - log10(i_H2_exp[k]))^2)
        end
        return sse
    end
end

init_polish!()
obj_full = make_objective()
@printf("\nSSE at the initial guess = %.6f\n", obj_full(copy(params0)))

opts(n) = Optim.Options(iterations = n, g_tol = 1e-6, f_reltol = 1e-6, show_trace = false)

# ── §17.5 Step 2: CO2R parameters first (HER fixed) ────────────────────────
println("\n" * "="^72)
println("Step 2 — fit i0_CO2R and lambda_MHC to i_CO (HER fixed)")
pcur = copy(params0)
obj_CO = p2 -> begin
    p = [p2[1], p2[2], pcur[3], pcur[4]]
    make_objective(fit_CO = true, fit_H2 = false)(p)
end
r2 = optimize(obj_CO, [pcur[1], pcur[2]], NelderMead(), opts(200))
pcur[1], pcur[2] = Optim.minimizer(r2)
@printf("  -> i0_CO2R = %.4f, lambda_MHC = %.4f   (SSE_CO = %.6f, %d iters)\n",
        pcur[1], pcur[2], Optim.minimum(r2), Optim.iterations(r2))

# ── §17.5 Step 3: HER parameters ───────────────────────────────────────────
println("\n" * "="^72)
println("Step 3 — fit HER_prefactor and alpha_c_HER to i_H2 (CO2R fixed)")
obj_H2 = p2 -> begin
    p = [pcur[1], pcur[2], p2[1], p2[2]]
    make_objective(fit_CO = false, fit_H2 = true)(p)
end
r3 = optimize(obj_H2, [pcur[3], pcur[4]], NelderMead(), opts(200))
pcur[3], pcur[4] = Optim.minimizer(r3)
@printf("  -> log10 HER_pf = %.4f, alpha_c = %.6f   (SSE_H2 = %.6f, %d iters)\n",
        pcur[3], pcur[4], Optim.minimum(r3), Optim.iterations(r3))

# ── §17.5 Step 4: joint 4-parameter refinement ─────────────────────────────
println("\n" * "="^72)
println("Step 4 — joint Nelder-Mead on all 4 parameters")
r4 = optimize(obj_full, copy(pcur), NelderMead(), Optim.Options(
        iterations = 200, g_tol = 1e-6, f_reltol = 1e-6, show_trace = false))
pfit = Optim.minimizer(r4)
@printf("  -> SSE = %.6f after %d iterations, %d objective evaluations total\n",
        Optim.minimum(r4), Optim.iterations(r4), n_eval[])

apply_params!(pfit)
polish_with_ramp!(pfit)
println("\n" * "="^72)
println("FITTED PARAMETERS")
@printf("  i0_CO2R       = %.4f A/m^2      (initial guess 540)\n", i0_CO2R())
@printf("  lambda_MHC    = %.4f eV         (initial guess 1.8)\n", lambda_MHC())
@printf("  HER_prefactor = %.6e mA/cm^2   (initial guess 2.409e13)\n", HER_prefactor())
@printf("  alpha_c_HER   = %.6f            (initial guess 0.0872)\n", alpha_c_HER())

open(joinpath(FITDIR, "fitted_params.txt"), "w") do io
    println(io, "# fitted kinetic parameters, run_fit.jl (Stage-4 residual, §17)")
    @printf(io, "i0_CO2R = %.10f\n", i0_CO2R())
    @printf(io, "lambda_MHC = %.10f\n", lambda_MHC())
    @printf(io, "HER_prefactor = %.10e\n", HER_prefactor())
    @printf(io, "alpha_c_HER = %.10f\n", alpha_c_HER())
    @printf(io, "k_GL = %.10e\n", k_GL())
end

# ── §17.6 acceptance criteria ──────────────────────────────────────────────
println("\n" * "="^72)
println("§17.6 fit-quality acceptance criteria")
tol_CO = [30.0, 30.0, 30.0, 30.0, 80.0]
allpass = Ref(true)
for (k, V_rhe) in enumerate(E_exp_RHE)
    iCO, iH2, _ = compute_currents(exp_cache[V_rhe], V_rhe, ctx)
    eCO = 100*abs(iCO/10 - i_CO_exp[k])/i_CO_exp[k]
    eH2 = 100*abs(iH2/10 - i_H2_exp[k])/i_H2_exp[k]
    okCO = eCO <= tol_CO[k]; okH2 = eH2 <= 30.0
    allpass[] &= (okCO && okH2)
    @printf("  V=%7.3f  i_CO %8.2f vs %6.1f (%5.1f%% %s, tol %.0f%%)   i_H2 %8.2f vs %6.1f (%5.1f%% %s, tol 30%%)\n",
            V_rhe, iCO/10, i_CO_exp[k], eCO, okCO ? "PASS" : "FAIL", tol_CO[k],
            iH2/10, i_H2_exp[k], eH2, okH2 ? "PASS" : "FAIL")
end

# ── §17.8 diagnostic refit sweep: Stage-4 physics at the fitted kinetics ────
println("\n" * "="^72)
println("§17.8 refit sweep — 89 voltages, Stage-4 residual at the FITTED kinetics")
println("      (a diagnostic artifact; Stage 5 does NOT warm-start from it)")
out = run_stage(4; tag = "fit_postfit", outdir = FITDIR,
                cdir = joinpath(FITDIR, "cache"), cpref = "fit_V", verbose = false)

csv = out.recs
iCOs = [compute_currents(r.u, r.V_RHE, ctx)[1]/10 for r in csv]
iH2s = [compute_currents(r.u, r.V_RHE, ctx)[2]/10 for r in csv]
fes  = 100 .* iCOs ./ (iCOs .+ iH2s)
dCO = diff(iCOs); sc = count(k -> sign(dCO[k]) != sign(dCO[k+1]), 1:length(dCO)-1)
dH2 = diff(iH2s); scH = count(k -> sign(dH2[k]) != sign(dH2[k+1]), 1:length(dH2)-1)
kpk = argmax(fes)
@printf("\n  sweep coverage                 : %d/89 %s\n", length(csv), length(csv)==89 ? "PASS" : "FAIL")
@printf("  sign changes in di_CO/dV       : %d  (target exactly 1) %s\n", sc, sc==1 ? "PASS" : "FAIL")
@printf("  sign changes in di_H2/dV       : %d  (target 0, strictly monotone) %s\n", scH, scH==0 ? "PASS" : "FAIL")
@printf("  FE_CO peak                     : %.1f %% at V_RHE = %.3f  (target 70-85%%, within 100 mV of -1.205) %s\n",
        fes[kpk], csv[kpk].V_RHE,
        (70 <= fes[kpk] <= 85 && abs(csv[kpk].V_RHE + 1.205) <= 0.100) ? "PASS" : "FAIL")
@printf("  overall per-point tolerances   : %s\n", allpass[] ? "PASS" : "FAIL")

println("\n>>> FIT COMPLETE — generate plots and STOP for human review.")
