# run_fit.jl — Fit step (§17), run AFTER Stage 4.
# Fits i0_CO2R, lambda_MHC, log10(HER_prefactor), alpha_c_HER against the
# 5 experimental points of §16.1 using the STAGE 4 residual (phi_L = 0,
# phi_s = V_applied_SHE) for speed.
include("model.jl")
include("checks.jl")
using Printf, Optim

const STAGE_TAG = "fit"
const PREFIX    = "fit_"

print_kinetics_header("Fit step — Nelder-Mead on the Stage 4 residual", STAGE_TAG)

N_CL = N_CL_DEFAULT; N_GDL = N_GDL_DEFAULT
dm = make_domain(N_CL, N_GDL)
ly = Layout(4, N_CL, N_GDL)
ctx = make_ctx(ly, dm)

# ── load the Stage-4 grid cache for warm-starting the polish (§17.3)
s4_V = list_cache("stage4", "s4_")
length(s4_V) == 89 || error("Stage 4 cache incomplete ($(length(s4_V))/89) — run run_stage4.jl first")
s4_U = [load_solution("stage4", "s4_", V) for V in s4_V]
@printf("Loaded %d Stage-4 cache entries, V_RHE in [%.3f, %.3f]\n",
        length(s4_V), minimum(s4_V), maximum(s4_V))

# ── §17.2 parameters, bounds, starting point
const params0 = [540.0, 1.8, log10(2.409e13), 0.0872]
const lower   = [ 10.0, 0.5,  8.0, 0.01]
const upper   = [5000.0, 3.0, 18.0, 0.50]
const PNAMES  = ["i0_CO2R", "lambda_MHC", "log10(HER_pf)", "alpha_c"]

set_params!(p) = (set_i0_CO2R!(p[1]); set_lambda_MHC!(p[2]);
                  set_HER_prefactor!(10.0^p[3]); set_alpha_c_HER!(p[4]))

# ── §17.3 pre-fitting polish at the exact experimental voltages
function initial_polish()
    cache = Dict{Float64,Vector{Float64}}()
    for V in E_exp_RHE
        j = argmin(abs.(s4_V .- V))
        u = copy(s4_U[j])
        info = solve_at!(ctx, u, V; alpha_init=ALPHA_SWEEP, max_iter=MAX_ITER_SWEEP)
        info.converged || error("Initial polish failed at V_RHE = $V")
        cache[V] = u
        @printf("  polish V_RHE=%.3f (warm start %.3f): %d iters, ||F||inf=%.2e\n",
                V, s4_V[j], info.iterations, info.residual)
    end
    return cache
end

println("\n§17.3 pre-fitting polish at the 5 experimental voltages:")
exp_cache = initial_polish()
p_last = copy(params0)

"""
Polish all 5 experimental voltages at `p_target`, warm-started from `base`.
On direct failure, ramp the kinetic vector from `p_last` in 2 → 4 → 8 sub-steps
(§9.5.7).  Returns `(ok, cache)`; `base` is left untouched.
"""
function try_polish(p_target::Vector{Float64}, base::Dict{Float64,Vector{Float64}})
    for nsub in (1, 2, 4, 8)
        cache = Dict(k => copy(v) for (k, v) in base)
        ok = true
        for s in 1:nsub
            ps = nsub == 1 ? p_target : p_last .+ (s/nsub) .* (p_target .- p_last)
            set_params!(ps)
            for V in E_exp_RHE
                u = cache[V]
                info = solve_at!(ctx, u, V; alpha_init=ALPHA_SWEEP, max_iter=MAX_ITER_SWEEP)
                if !info.converged
                    info = solve_at!(ctx, u, V; alpha_init=ALPHA_ANCHOR, max_iter=MAX_ITER_ANCHOR)
                end
                if !info.converged; ok = false; break; end
            end
            ok || break
        end
        ok && (set_params!(p_target); return (true, cache))
    end
    return (false, base)
end

function model_currents(cache)
    iCO = Float64[]; iH2 = Float64[]
    for V in E_exp_RHE
        a, b, _ = compute_currents(cache[V], ly, dm, V)
        push!(iCO, a/10); push!(iH2, b/10)
    end
    return iCO, iH2
end

const N_EVAL = Ref(0)

"""§17.4 objective.  `mode` = :both | :co | :h2 (Steps 2/3 fit one branch)."""
function make_objective(mode::Symbol)
    return function (params::Vector{Float64})
        N_EVAL[] += 1
        # sloped bound penalty (§17.2 — never a flat constant)
        pen = 0.0
        for j in 1:4
            v = max(lower[j] - params[j], params[j] - upper[j], 0.0)
            pen += (v / (upper[j] - lower[j]))^2
        end
        pen > 0 && return 1e6 * (1 + pen)

        ok, cache = try_polish(collect(params), exp_cache)
        if !ok
            set_params!(p_last)
            return 1e6 * (1 + sum(((params .- p_last) ./ (upper .- lower)).^2))
        end
        for (k, v) in cache; exp_cache[k] = v; end
        global p_last = collect(params)

        iCO, iH2 = model_currents(exp_cache)
        sse = 0.0
        for k in 1:5
            mode in (:both, :co) && (sse += (log10(max(iCO[k], 0.1)) - log10(i_CO_exp[k]))^2)
            mode in (:both, :h2) && (sse += (log10(max(iH2[k], 0.1)) - log10(i_H2_exp[k]))^2)
        end
        return sse
    end
end

obj_both = make_objective(:both)
obj_co   = make_objective(:co)
obj_h2   = make_objective(:h2)

const FITFILE = joinpath(@__DIR__, "fitted_params.txt")
const REUSE = get(ENV, "REUSE_FIT", "0") == "1" && isfile(FITFILE)

if REUSE
    v = parse.(Float64, readlines(FITFILE))
    global p_fit = [v[1], v[2], log10(v[3]), v[4]]
    global sse_fit = NaN
    @printf("\nREUSE_FIT=1 -> skipping Nelder-Mead, loaded fitted_params.txt\n")
else

# ── §17.5 Step 1: manual exploration
println("\n§17.5 Step 1 — one-at-a-time parameter exploration (SSE_total):")
@printf("  base params: i0=%.1f  lambda=%.2f  log10(HER_pf)=%.3f  alpha_c=%.4f\n", params0...)
@printf("  base SSE = %.5f\n", obj_both(copy(params0)))
for (j, vals) in zip(1:4, ([100.0, 540.0, 2000.0], [1.2, 1.8, 2.4],
                           [12.5, 13.382, 14.5], [0.05, 0.0872, 0.15]))
    for v in vals
        p = copy(params0); p[j] = v
        @printf("    %-14s = %-10.4g  SSE = %.5f\n", PNAMES[j], v, obj_both(p))
    end
end
set_params!(params0); p_last = copy(params0)

# ── §17.5 Step 2: CO2R parameters first (i_CO branch only)
println("\n§17.5 Step 2 — Nelder-Mead on (i0_CO2R, lambda_MHC), HER fixed:")
f2 = p2 -> obj_co([p2[1], p2[2], params0[3], params0[4]])
r2 = optimize(f2, [params0[1], params0[2]], NelderMead(),
              Optim.Options(iterations=200, g_tol=1e-6, f_reltol=1e-6, show_trace=false))
p_step2 = [Optim.minimizer(r2)[1], Optim.minimizer(r2)[2], params0[3], params0[4]]
@printf("  -> i0=%.3f  lambda=%.4f   SSE_CO=%.6f  (%d iters)\n",
        p_step2[1], p_step2[2], Optim.minimum(r2), Optim.iterations(r2))

# ── §17.5 Step 3: HER parameters
println("\n§17.5 Step 3 — Nelder-Mead on (log10 HER_pf, alpha_c), CO2R fixed:")
f3 = p3 -> obj_h2([p_step2[1], p_step2[2], p3[1], p3[2]])
r3 = optimize(f3, [p_step2[3], p_step2[4]], NelderMead(),
              Optim.Options(iterations=200, g_tol=1e-6, f_reltol=1e-6, show_trace=false))
p_step3 = [p_step2[1], p_step2[2], Optim.minimizer(r3)[1], Optim.minimizer(r3)[2]]
@printf("  -> log10(HER_pf)=%.4f  alpha_c=%.5f   SSE_H2=%.6f  (%d iters)\n",
        p_step3[3], p_step3[4], Optim.minimum(r3), Optim.iterations(r3))

# ── §17.5 Step 4: joint refinement
println("\n§17.5 Step 4 — joint 4-parameter Nelder-Mead:")
r4 = optimize(obj_both, p_step3, NelderMead(),
              Optim.Options(iterations=300, g_tol=1e-6, f_reltol=1e-6, show_trace=false))
global p_fit = collect(Optim.minimizer(r4))
global sse_fit = Optim.minimum(r4)
@printf("  -> SSE=%.6f after %d iterations, %d total objective evaluations\n",
        sse_fit, Optim.iterations(r4), N_EVAL[])
end  # !REUSE

println("\n" * "="^72)
println("FITTED KINETIC PARAMETERS")
println("="^72)
@printf("  i0_CO2R       = %.4f A/m^2      (initial guess 540)\n", p_fit[1])
@printf("  lambda_MHC    = %.4f eV         (initial guess 1.8)\n", p_fit[2])
@printf("  HER_prefactor = %.5e mA/cm^2  (initial guess 2.409e13)\n", 10.0^p_fit[3])
@printf("  alpha_c_HER   = %.5f            (initial guess 0.0872)\n", p_fit[4])
@printf("  SSE (log-space, 10 points) = %.6f\n", sse_fit)

# persist for run_stage5.jl
open(joinpath(@__DIR__, "fitted_params.txt"), "w") do f
    @printf(f, "%.10e\n%.10e\n%.10e\n%.10e\n", p_fit[1], p_fit[2], 10.0^p_fit[3], p_fit[4])
end
println("  written to fitted_params.txt")

# ── §17.8 diagnostic refit sweep: 89-V Stage-4 sweep at the fitted kinetics
# (self-anchoring, so it does not depend on the pre-fit cache at all)
set_params!(p_fit)
println("\n§17.8 diagnostic 89-V Stage-4 sweep at the fitted kinetics:")
u0 = make_u0(ly, dm)
Vs = copy(V_RHE_grid); emit = fill(true, length(Vs))
t = @elapsed Vout, Uout, st = continuation_sweep(ctx, u0, Vs; emit=emit,
        stage_tag=STAGE_TAG, prefix=PREFIX, quiet=true)
@printf("  %d/%d converged in %.2f s, %.2f iters/step, max ||F||inf=%.3e, %d recoveries\n",
        length(Vout), length(Vs), t, sum(st.iters)/length(st.iters),
        maximum(st.resids), st.n_substep)

# ── §17.6 acceptance criteria: polish the 5 exp voltages off the refit sweep
println("\n§17.6 polish at the 5 experimental voltages (warm start = refit sweep):")
fit_cache = Dict{Float64,Vector{Float64}}()
for V in E_exp_RHE
    j = argmin(abs.(Vout .- V)); u = copy(Uout[j])
    info = solve_at!(ctx, u, V; alpha_init=ALPHA_SWEEP, max_iter=MAX_ITER_SWEEP)
    info.converged || error("final polish failed at V_RHE=$V ($(info.reason))")
    fit_cache[V] = u
    @printf("  V_RHE=%.3f: %d iters, ||F||inf=%.2e\n", V, info.iterations, info.residual)
end
iCO, iH2 = model_currents(fit_cache)

println("\n§17.6 per-point acceptance check:")
@printf("  %8s %10s %10s %8s %6s | %10s %10s %8s %6s\n",
        "V_RHE","iCO_mod","iCO_exp","err%","tol","iH2_mod","iH2_exp","err%","tol")
pass = true
for k in 1:5
    tolCO = k == 5 ? 80.0 : 30.0
    eCO = 100*(iCO[k]-i_CO_exp[k])/i_CO_exp[k]
    eH2 = 100*(iH2[k]-i_H2_exp[k])/i_H2_exp[k]
    okCO = abs(eCO) <= tolCO; okH2 = abs(eH2) <= 30.0
    global pass = pass && okCO && okH2
    @printf("  %8.3f %10.2f %10.2f %8.1f %5.0f%s | %10.2f %10.2f %8.1f %5.0f%s\n",
            E_exp_RHE[k], iCO[k], i_CO_exp[k], eCO, tolCO, okCO ? " " : "X",
            iH2[k], i_H2_exp[k], eH2, 30.0, okH2 ? " " : "X")
end
@printf("  per-point tolerances all satisfied: %s\n", pass)

println("\n--- Fit-step physicality checks ---")
res = report_polarization(Vout, Uout, ly, dm)
println()
compare_to_experiment(Vout, Uout, ly, dm)

println("\n  Branch check (§17.6 crit. 4) — polished points vs the 89-V sweep:")
ordv = sortperm(Vout)
Vs_s = Vout[ordv]
iCOs = [compute_currents(Uout[i], ly, dm, Vout[i])[1]/10 for i in ordv]
iH2s = [compute_currents(Uout[i], ly, dm, Vout[i])[2]/10 for i in ordv]
lin(xq,x,y) = (j = searchsortedfirst(x,xq); j<=1 ? y[1] : j>length(x) ? y[end] :
               y[j-1] + (xq-x[j-1])/(x[j]-x[j-1])*(y[j]-y[j-1]))
for k in 1:5
    @printf("    V_RHE=%.3f: polished iCO=%.3f vs sweep %.3f (%.2f%%) | iH2=%.3f vs %.3f (%.2f%%)\n",
            E_exp_RHE[k], iCO[k], lin(E_exp_RHE[k],Vs_s,iCOs),
            100*abs(iCO[k]-lin(E_exp_RHE[k],Vs_s,iCOs))/max(iCO[k],1e-9),
            iH2[k], lin(E_exp_RHE[k],Vs_s,iH2s),
            100*abs(iH2[k]-lin(E_exp_RHE[k],Vs_s,iH2s))/max(iH2[k],1e-9))
end

export_polarization(STAGE_TAG, Vout, Uout, ly, dm)
export_profiles(STAGE_TAG, Vout, Uout, ly, dm)
println("\nExported polarization.csv and profiles.bin to output_fit/")
println("FIT STEP COMPLETE — STOP GATE")
