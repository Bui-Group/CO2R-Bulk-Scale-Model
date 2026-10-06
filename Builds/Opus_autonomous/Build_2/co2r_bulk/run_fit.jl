# run_fit.jl — Fit step (after Stage 4), §17.
# Fits i0_CO2R, λ_MHC, log10(HER_prefactor), α_c_HER against the 5 experimental
# points of §16.1 using the STAGE 4 residual (φ_L = 0, φ_s = V_applied_SHE).
include(joinpath(@__DIR__, "model.jl"))
using Printf, Optim

const N_CL  = N_CL_DEFAULT
const N_GDL = N_GDL_DEFAULT

runf = make_stage_run(Val(4), N_CL, N_GDL; outname="output_fit", prefix="fit")
print_kinetics_banner("Fit (Stage-4 residual)", runf.cache_dir)
@printf("mesh: N_CL=%d N_GDL=%d ; DOFs=%d\n", N_CL, N_GDL, runf.ly.N_TOT)

# ── Stage-4 grid cache (warm-start source for the pre-fit polish) ─────────
const S4CACHE = joinpath(ROOT, "output_stage4", "cache")
s4_V = list_cache(S4CACHE, "s4")
isempty(s4_V) && error("No Stage-4 cache found — run run_stage4.jl first!")
function load_s4(V)
    fn = joinpath(S4CACHE, @sprintf("s4_V%.6f.bin", V))
    open(fn, "r") do f; n = read(f, Int64); read!(f, Vector{Float64}(undef, n)); end
end
@printf("Stage-4 cache: %d solutions, V ∈ [%.3f, %.3f]\n", length(s4_V), s4_V[1], s4_V[end])

const params0 = [540.0, 1.8, log10(2.409e13), 0.0872]
const lower   = [ 10.0, 0.5,  8.0, 0.01]
const upper   = [5000.0, 3.0, 18.0, 0.50]
const NLOG    = 5 * N_CL

set_params!(p) = (set_i0_CO2R!(p[1]); set_lambda_MHC!(p[2]);
                  set_HER_prefactor!(10.0^p[3]); set_alpha_c_HER!(p[4]))

# ── §17.3 pre-fitting voltage polish ─────────────────────────────────────
exp_cache = Dict{Float64,Vector{Float64}}()
for V in E_exp_RHE
    exp_cache[V] = copy(load_s4(s4_V[argmin(abs.(s4_V .- V))]))
end

"""
    polish_all!(cache; alpha0) -> Bool

Re-solve the Stage-4 system at the 5 exact experimental voltages with the
current kinetic parameters, warm-started from `cache`.  Writes into a scratch
dict and only commits on full success.
"""
function polish_all!(cache::Dict{Float64,Vector{Float64}}; alpha0::Float64=1.0,
                     verbose::Bool=false)
    scratch = Dict{Float64,Vector{Float64}}()
    for V in E_exp_RHE
        u = copy(cache[V])
        runf.Vref[] = V
        info = newton_affine!(u, runf.ws; rtol=1e-8, max_iter=40,
                              alpha_init=alpha0, n_log_dofs=NLOG)
        info.converged || return false
        verbose && @printf("  polish V_RHE=%.3f → %d iters, ‖F‖∞=%.2e\n",
                           V, info.iterations, info.residual)
        scratch[V] = u
    end
    for V in E_exp_RHE; cache[V] = scratch[V]; end
    return true
end

set_params!(params0)
println("\n── pre-fit polish at the 5 experimental voltages (§17.3) ──")
@assert polish_all!(exp_cache; alpha0=1.0, verbose=true) "pre-fit polish failed"

# state of the last feasible parameter vector (for the sloped failure penalty
# and for the §9.5.7 parameter-direction ramp)
p_last = copy(params0)
cache_last = Dict(V => copy(exp_cache[V]) for V in E_exp_RHE)

function model_currents()
    out = zeros(2, 5)
    for (k, V) in enumerate(E_exp_RHE)
        iCO, iH2 = compute_currents(exp_cache[V], runf.ly, runf.cl, V)
        out[1,k] = iCO/10; out[2,k] = iH2/10          # A/m² → mA/cm²
    end
    return out
end

"""
    try_params(p) -> Bool

Apply `p` and re-polish.  On direct failure retry damped, then ramp the kinetic
parameters from `p_last` to `p` in 2 → 4 → 8 sub-steps (§9.5.7).
"""
function try_params(p::Vector{Float64})
    set_params!(p)
    polish_all!(exp_cache; alpha0=1.0) && return true
    for V in E_exp_RHE; exp_cache[V] = copy(cache_last[V]); end
    polish_all!(exp_cache; alpha0=1e-2) && return true
    for nsub in (2, 4, 8)
        for V in E_exp_RHE; exp_cache[V] = copy(cache_last[V]); end
        good = true
        for m in 1:nsub
            set_params!(p_last .+ (p .- p_last) .* (m / nsub))
            good = polish_all!(exp_cache; alpha0=1e-2)
            good || break
        end
        good && (set_params!(p); return polish_all!(exp_cache; alpha0=1e-2))
    end
    for V in E_exp_RHE; exp_cache[V] = copy(cache_last[V]); end
    return false
end

n_eval = Ref(0)
function objective(p::Vector{Float64}; use_CO::Bool=true, use_H2::Bool=true)
    n_eval[] += 1
    # §17.2 sloped bound penalty — never a flat constant
    pen = 0.0
    for j in 1:4
        v = max(lower[j] - p[j], 0.0) + max(p[j] - upper[j], 0.0)
        pen += (v / (upper[j] - lower[j]))^2
    end
    pen > 0 && return 1e6 * (1 + pen)
    if !try_params(p)
        return 1e6 * (1 + sum(((p .- p_last) ./ (upper .- lower)).^2))
    end
    global p_last = copy(p)
    for V in E_exp_RHE; cache_last[V] = copy(exp_cache[V]); end
    m = model_currents()
    sse = 0.0
    for k in 1:5
        use_CO && (sse += (log10(max(m[1,k], 0.1)) - log10(i_CO_exp[k]))^2)
        use_H2 && (sse += (log10(max(m[2,k], 0.1)) - log10(i_H2_exp[k]))^2)
    end
    return sse
end

function report(tag, p)
    set_params!(p); try_params(p)
    m = model_currents()
    @printf("\n%s  i0=%.3f  λ=%.4f  log10(HER_pf)=%.4f  α_c=%.5f\n",
            tag, p[1], p[2], p[3], p[4])
    println("  V_RHE    i_CO_mod  i_CO_exp   err%     i_H2_mod  i_H2_exp   err%")
    for k in 1:5
        @printf("  %6.3f  %9.3f %9.3f  %+7.1f  %9.3f %9.3f  %+7.1f\n",
                E_exp_RHE[k], m[1,k], i_CO_exp[k], 100*(m[1,k]-i_CO_exp[k])/i_CO_exp[k],
                m[2,k], i_H2_exp[k], 100*(m[2,k]-i_H2_exp[k])/i_H2_exp[k])
    end
    return m
end

t0 = time()
report("[Step 0: initial guess]", params0)

# ── §17.5 Step 1: manual exploration ─────────────────────────────────────
println("\n── Step 1: one-at-a-time parameter exploration ──")
for (j, nm, vals) in ((1, "i0_CO2R", [50.0, 150.0, 540.0, 1500.0]),
                      (2, "λ_MHC",   [1.0, 1.4, 1.8, 2.4]),
                      (3, "log10 HER_pf", [12.5, 13.0, 13.38, 14.0]),
                      (4, "α_c_HER", [0.05, 0.0872, 0.15, 0.25]))
    for v in vals
        p = copy(params0); p[j] = v
        s = objective(p)
        m = model_currents()
        @printf("  %-14s = %8.4f → SSE=%8.4f  i_CO=[%s]  i_H2=[%s]\n", nm, v, s,
                join([@sprintf("%.0f", m[1,k]) for k in 1:5], ","),
                join([@sprintf("%.0f", m[2,k]) for k in 1:5], ","))
    end
end

# ── §17.5 Step 2: CO₂R parameters first (HER fixed) ──────────────────────
println("\n── Step 2: fit (i0_CO2R, λ_MHC) against i_CO only ──")
obj_CO(x) = objective([x[1], x[2], params0[3], params0[4]]; use_H2=false)
r2 = optimize(obj_CO, [params0[1], params0[2]], NelderMead(),
              Optim.Options(iterations=120, g_tol=1e-6, show_trace=false))
p2 = [Optim.minimizer(r2)[1], Optim.minimizer(r2)[2], params0[3], params0[4]]
@printf("Step 2 → i0=%.3f λ=%.4f (SSE_CO=%.5f, %d evals)\n",
        p2[1], p2[2], Optim.minimum(r2), n_eval[])

# ── §17.5 Step 3: HER parameters (CO₂R fixed) ────────────────────────────
println("\n── Step 3: fit (HER_prefactor, α_c) against i_H₂ only ──")
obj_H2(x) = objective([p2[1], p2[2], x[1], x[2]]; use_CO=false)
r3 = optimize(obj_H2, [p2[3], p2[4]], NelderMead(),
              Optim.Options(iterations=120, g_tol=1e-6, show_trace=false))
p3 = [p2[1], p2[2], Optim.minimizer(r3)[1], Optim.minimizer(r3)[2]]
@printf("Step 3 → log10 HER_pf=%.4f α_c=%.5f (SSE_H2=%.5f, %d evals)\n",
        p3[3], p3[4], Optim.minimum(r3), n_eval[])
report("[after Steps 2-3]", p3)

# ── §17.5 Step 4: joint 4-parameter Nelder-Mead ──────────────────────────
println("\n── Step 4: joint 4-parameter Nelder-Mead ──")
r4 = optimize(objective, p3, NelderMead(),
              Optim.Options(iterations=200, g_tol=1e-6, show_trace=false))
pfit = collect(Optim.minimizer(r4))
@printf("Step 4 → SSE=%.6f after %d NM iterations (%d total objective evals)\n",
        Optim.minimum(r4), Optim.iterations(r4), n_eval[])
m = report("[FITTED]", pfit)
@printf("\nfit wall time: %.1f s\n", time()-t0)

# ── §17.6 acceptance criteria ────────────────────────────────────────────
println("\n── §17.6 acceptance criteria ──")
tol_CO = [30.0, 30.0, 30.0, 30.0, 80.0]
passes = Bool[]
for k in 1:5
    eCO = abs(100*(m[1,k]-i_CO_exp[k])/i_CO_exp[k])
    eH2 = abs(100*(m[2,k]-i_H2_exp[k])/i_H2_exp[k])
    okCO = eCO <= tol_CO[k]; okH2 = eH2 <= 30.0
    push!(passes, okCO & okH2)
    @printf("  V=%.3f  |Δi_CO|=%5.1f%% (tol %2.0f%%) %s   |Δi_H2|=%5.1f%% (tol 30%%) %s\n",
            E_exp_RHE[k], eCO, tol_CO[k], okCO ? "PASS" : "FAIL",
            eH2, okH2 ? "PASS" : "FAIL")
end
@printf("per-point tolerances: %s\n", all(passes) ? "ALL PASS" : "SOME FAIL")

# ── §17.8 final 89-voltage Stage-4 sweep at the fitted kinetics ──────────
println("\n── final 89-V Stage-4 sweep at the fitted kinetics (diagnostic artifact) ──")
set_params!(pfit)
print_kinetics_banner("Fit sweep", runf.cache_dir)
u0 = build_initial_guess(runf.ly, runf.cl, runf.gdl, runf.c_eq,
                         V_RHE_grid[1] - V_RHE_to_SHE_offset)
sw = run_sweep!(runf, u0, V_RHE_grid[1], V_RHE_grid[2:end], V_RHE_grid; verbose=false)
s  = export_stage(runf, sw.Vs, sw.us)
@printf("sweep: %d/%d converged, max ‖F‖∞ = %.3e, %.2f iters/step\n",
        length(sw.Vs), length(V_RHE_grid), maximum(sw.resn),
        sum(sw.iters)/length(sw.iters))
print_physicality(runf, s, sw.Vs, sw.us)
kpk = argmax(s.FE_CO)
@printf("FE_CO peak: %.1f %% at V_RHE = %.3f (target 70–85 %%, within 100 mV of −1.205)\n",
        s.FE_CO[kpk], sw.Vs[kpk])
@printf("sign changes di_CO/dV = %d (target 1); di_H2/dV = %d (target 0)\n",
        sign_changes(s.iCO), sign_changes(s.iH2))
compare_to_experiment(sw.Vs, s.iCO, s.iH2)

open(joinpath(runf.out_dir, "fitted_params.txt"), "w") do f
    @printf(f, "i0_CO2R        = %.10f\n", pfit[1])
    @printf(f, "lambda_MHC     = %.10f\n", pfit[2])
    @printf(f, "log10_HER_pf   = %.10f\n", pfit[3])
    @printf(f, "HER_prefactor  = %.10e\n", 10.0^pfit[3])
    @printf(f, "alpha_c_HER    = %.10f\n", pfit[4])
    @printf(f, "SSE            = %.10f\n", Optim.minimum(r4))
end
println("\nfitted parameters written to output_fit/fitted_params.txt")
println("Fit step complete — STOP for review.")
