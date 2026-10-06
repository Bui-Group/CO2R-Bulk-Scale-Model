# run_fit.jl — Fit step after Stage 4 (§17).  Fits i0_CO2R, λ_MHC,
# log10(HER_prefactor), α_c against the 5 experimental points of §16.1 using
# the Stage-4 residual (φ_L = 0, φ_s = V_applied_SHE) for speed.
include("common.jl")
using Optim

const S4_CACHE  = joinpath(@__DIR__, "output_stage4", "cache")
const FIT_CACHE = joinpath(@__DIR__, "output_fit", "cache")

m = build_model(4)
ly = m.ly; N = ly.N_TOT
print_kinetics_header("Fit", FIT_CACHE)

# ── shared solver workspace (one closure object, voltage held in a Ref, so the
#    ForwardDiff tag is stable across every polish)
const Vref = Ref(-0.816)
res_at_V!(F, u, V) = residual_stage34!(F, u, V, m.cl, m.gm, m.c_eq, ly, true)
res!(F, u) = res_at_V!(F, u, Vref[])
const PLU  = AffineLU()
const JD   = zeros(N, N)
const CFG  = ForwardDiff.JacobianConfig(res!, zeros(N), zeros(N), ForwardDiff.Chunk{12}())
const BLK  = weight_blocks(ly)
const NLOG = 5 * m.cl.N

polish!(u, V; alpha0 = 1.0, maxit = 40) =
    (Vref[] = V; newton_affine!(u, res!; blocks = BLK, plu = PLU, rtol = 1e-8,
                                max_iter = maxit, alpha_init = alpha0,
                                n_log_dofs = NLOG, Jdense = JD, cfg = CFG))

# ── §17.3 pre-fitting polish at the exact experimental voltages
s4_V = list_cache(S4_CACHE, "s4_")
isempty(s4_V) && error("no Stage 4 cache — run run_stage4.jl first")
exp_cache = Dict{Float64,Vector{Float64}}()
println("\n── §17.3 pre-fitting polish at the 5 experimental voltages ──")
for V in E_exp_RHE
    u = load_solution(S4_CACHE, "s4_", s4_V[argmin(abs.(s4_V .- V))])
    info = polish!(u, V)
    @assert info.converged "Polish failed at V_RHE = $V"
    exp_cache[V] = u
    @printf("  V_RHE = %.3f → %d iters, ||F||inf = %.2e\n", V, info.iterations, info.residual)
end

const LOWER = [10.0, 0.5,  8.0, 0.01]
const UPPER = [5000.0, 3.0, 18.0, 0.50]
const P0    = [540.0, 1.8, log10(2.409e13), 0.0872]
p_last = copy(P0)

apply_params!(p) = (set_i0_CO2R!(p[1]); set_lambda_MHC!(p[2]);
                    set_HER_prefactor!(10.0^p[3]); set_alpha_c_HER!(p[4]))

"Sloped bound penalty (§17.2 — never a flat constant)."
function bound_penalty(p)
    pen = 0.0
    for j in 1:4
        v = max(LOWER[j] - p[j], p[j] - UPPER[j], 0.0)
        pen += (v / (UPPER[j] - LOWER[j]))^2
    end
    return pen
end

"Polish all 5 voltages at parameter vector p; returns nothing on failure.
 §9.5.7: on a direct-polish failure, ramp the kinetics from the last feasible
 vector toward p in 2 → 4 → 8 sub-steps before rejecting the trial."
function polish_all!(p)
    scratch = Dict{Float64,Vector{Float64}}()
    apply_params!(p)
    ok = true
    for V in E_exp_RHE
        u = copy(exp_cache[V])
        info = polish!(u, V)
        if !info.converged
            ok = false; break
        end
        scratch[V] = u
    end
    if !ok
        for nsub in (2, 4, 8)
            scratch = Dict{Float64,Vector{Float64}}()
            work = Dict(V => copy(exp_cache[V]) for V in E_exp_RHE)
            ok = true
            for j in 1:nsub
                pj = p_last .+ (j / nsub) .* (p .- p_last)
                apply_params!(pj)
                for V in E_exp_RHE
                    info = polish!(work[V], V; alpha0 = (j == 1 ? 1e-2 : 1.0))
                    if !info.converged; ok = false; break; end
                end
                ok || break
            end
            if ok
                scratch = work
                break
            end
        end
        ok || return nothing
    end
    for V in E_exp_RHE; exp_cache[V] = scratch[V]; end
    return scratch
end

n_eval = Ref(0)
function fitting_objective(params::Vector{Float64})
    n_eval[] += 1
    pen = bound_penalty(params)
    pen > 0 && return 1e6 * (1 + pen)
    got = polish_all!(params)
    got === nothing && return 1e6 * (1 + sum(((params .- p_last) ./ (UPPER .- LOWER)).^2))
    global p_last = copy(params)
    sse = 0.0
    for (k, V) in enumerate(E_exp_RHE)
        V_SHE = V - V_RHE_to_SHE_offset
        iCO, iH2 = compute_currents(got[V], V_SHE, m.cl)
        sse += (log10(max(iCO/10, 0.1)) - log10(i_CO_exp[k]))^2
        sse += (log10(max(iH2/10, 0.1)) - log10(i_H2_exp[k]))^2
    end
    return sse
end

report(p, tag) = begin
    apply_params!(p); polish_all!(p)
    @printf("\n%s: i0=%.3f  λ=%.4f  log10HERpf=%.4f (%.4e)  α_c=%.5f   SSE=%.6f\n",
            tag, p[1], p[2], p[3], 10.0^p[3], p[4], fitting_objective(p))
    @printf("  %8s %10s %10s %10s %10s %8s %8s\n",
            "V_RHE","iCO_mod","iCO_exp","iH2_mod","iH2_exp","errCO%","errH2%")
    for (k, V) in enumerate(E_exp_RHE)
        iCO, iH2 = compute_currents(exp_cache[V], V - V_RHE_to_SHE_offset, m.cl)
        @printf("  %8.3f %10.2f %10.2f %10.2f %10.2f %8.1f %8.1f\n",
                V, iCO/10, i_CO_exp[k], iH2/10, i_H2_exp[k],
                100*(iCO/10 - i_CO_exp[k])/i_CO_exp[k],
                100*(iH2/10 - i_H2_exp[k])/i_H2_exp[k])
    end
end

println("\n── §17.5 Step 1: one-at-a-time sensitivity around the initial guess ──")
let base = fitting_objective(P0)
    @printf("  base SSE = %.4f\n", base)
    for (j, name, fac) in ((1,"i0_CO2R",0.5),(2,"λ_MHC",0.9),(3,"log10 HER_pf",0.98),(4,"α_c",0.8))
        for s in (fac, 1/fac)
            p = copy(P0); p[j] *= s
            @printf("  %-14s ×%.3f → SSE %.4f\n", name, s, fitting_objective(p))
        end
    end
    apply_params!(P0); polish_all!(P0)
end

println("\n── §17.5 Step 2: CO₂R parameters (i0, λ) with HER fixed ──")
obj_co2r(q) = fitting_objective([q[1], q[2], P0[3], P0[4]])
r2 = optimize(obj_co2r, [P0[1], P0[2]], NelderMead(),
              Optim.Options(iterations = 120, g_tol = 1e-6, show_trace = false))
p_step2 = [Optim.minimizer(r2)[1], Optim.minimizer(r2)[2], P0[3], P0[4]]
report(p_step2, "step 2")

println("\n── §17.5 Step 3: HER parameters (log10 HER_pf, α_c) with CO₂R fixed ──")
obj_her(q) = fitting_objective([p_step2[1], p_step2[2], q[1], q[2]])
r3 = optimize(obj_her, [p_step2[3], p_step2[4]], NelderMead(),
              Optim.Options(iterations = 120, g_tol = 1e-6, show_trace = false))
p_step3 = [p_step2[1], p_step2[2], Optim.minimizer(r3)[1], Optim.minimizer(r3)[2]]
report(p_step3, "step 3")

println("\n── §17.5 Step 4: joint 4-parameter Nelder-Mead ──")
r4 = optimize(fitting_objective, p_step3, NelderMead(),
              Optim.Options(iterations = 200, g_tol = 1e-6, show_trace = false))
p_fit = Optim.minimizer(r4)
report(p_fit, "FITTED")
@printf("\nNelder-Mead: %d objective evaluations total, final SSE = %.6f\n",
        n_eval[], Optim.minimum(r4))

open(joinpath(@__DIR__, "fitted_params.jl"), "w") do io
    println(io, "# fitted kinetics from run_fit.jl (§17) — loaded by run_stage5.jl")
    @printf(io, "const FITTED_i0_CO2R  = %.10e\n", p_fit[1])
    @printf(io, "const FITTED_lambda   = %.10e\n", p_fit[2])
    @printf(io, "const FITTED_HER_pf   = %.10e\n", 10.0^p_fit[3])
    @printf(io, "const FITTED_alpha_c  = %.10e\n", p_fit[4])
end
println("wrote fitted_params.jl")

# ── §17.8 diagnostic refit sweep: Stage-4 physics at the fitted kinetics
println("\n── §17.8 89-voltage Stage-4 refit sweep at the fitted kinetics ──")
apply_params!(p_fit)
u0 = u0_stage(ly, m.c_eq)
Vs, Us, rows, _ = run_continuation(u0, copy(V_RHE_grid), trues(89), res_at_V!;
        blocks = BLK, n_log = NLOG, cache_dir = FIT_CACHE, prefix = "fit_",
        verbose = false)
@printf("refit sweep: %d/89 converged, mean iters = %.2f, max ||F||inf = %.3e\n",
        length(Vs), sum(r.iters for r in rows)/length(rows), maximum(r.Finf for r in rows))
export_all("fit", Vs, Us, ly, m.cl, m.gm, rows)

# ── §17.6 acceptance criteria
println("\n── §17.6 acceptance criteria ──")
let
    tolCO = [30.0, 30.0, 30.0, 30.0, 80.0]; pass = true
    for (k, V) in enumerate(E_exp_RHE)
        iCO, iH2 = compute_currents(exp_cache[V], V - V_RHE_to_SHE_offset, m.cl)
        eCO = 100*abs(iCO/10 - i_CO_exp[k])/i_CO_exp[k]
        eH2 = 100*abs(iH2/10 - i_H2_exp[k])/i_H2_exp[k]
        ok = eCO <= tolCO[k] && eH2 <= 30.0
        pass &= ok
        @printf("  V=%.3f  |err i_CO| = %5.1f%% (tol %2.0f)  |err i_H2| = %5.1f%% (tol 30)  %s\n",
                V, eCO, tolCO[k], eH2, ok ? "PASS" : "FAIL")
    end
    iCOv = [compute_currents(Us[k], Vs[k]-V_RHE_to_SHE_offset, m.cl)[1] for k in eachindex(Vs)]
    iH2v = [compute_currents(Us[k], Vs[k]-V_RHE_to_SHE_offset, m.cl)[2] for k in eachindex(Vs)]
    FE = 100 .* iCOv ./ (iCOv .+ iH2v)
    kp = argmax(FE)
    dCO = diff(iCOv); dH2 = diff(iH2v)
    scCO = count(k -> sign(dCO[k]) != sign(dCO[k+1]), 1:length(dCO)-1)
    scH2 = count(k -> sign(dH2[k]) != sign(dH2[k+1]), 1:length(dH2)-1)
    @printf("  FE_CO peak = %.1f%% at V_RHE = %.3f (target 70–85%%, within 100 mV of −1.205)\n",
            FE[kp], Vs[kp])
    @printf("  sign changes in di_CO/dV = %d (target 1); in di_H2/dV = %d (target 0)\n", scCO, scH2)
    @printf("  sweep coverage = %d/89\n", length(Vs))
    @printf("  OVERALL per-point tolerances: %s\n", pass ? "PASS" : "FAIL")
end
println("\nFIT STEP COMPLETE.")
