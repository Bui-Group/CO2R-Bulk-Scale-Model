# =============================================================================
# run_fit.jl — Fit step (§17), run AFTER Stage 4 and BEFORE Stage 5.
#
# Fits i0_CO2R, lambda_MHC, log10(HER_prefactor) and alpha_c_HER against the 5
# experimental points of §16.1 using the **Stage 4 residual** (phi_L = 0,
# phi_s = V_applied_SHE).  That is a deliberate speed choice (§11): the Stage-4
# system is single-rooted and each polish converges in ~2 Newton iterations, so
# the ~1000 solves Nelder-Mead needs take minutes instead of hours.  The fitted
# values are therefore *apparent* parameters that absorb the 50-150 mV of
# potential drop missing from Stages 1-4; Stage 5 then restores that physics.
#
# Solver: the §9.5 production kernel (newton_affine! + persistent sparse LU +
# AD Jacobian) restricted to the 5 experimental voltages per NM trial.
# STOPS after generating plots.
# =============================================================================
include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "export.jl"))
include(joinpath(@__DIR__, "diagnostics.jl"))
using Printf, Optim

const CACHE_DIR = joinpath(@__DIR__, "output_fit", "cache")
const PREFIX    = "fit_"

print_stage_header("Fit", CACHE_DIR;
    extra = "Fitting i0_CO2R, lambda_MHC, log10(HER_pf), alpha_c against §16.1 using the Stage-4 residual")

const ctx4 = make_ctx(4)
const NDOF = ctx4.ly.N_TOT

# ── §17.2 starting point and hard bounds ─────────────────────────────────────
const params0 = [540.0, 1.8, log10(2.409e13), 0.0872]
const lower   = [ 10.0, 0.5,  8.0, 0.01]
const upper   = [5000.0, 3.0, 18.0, 0.50]
const PNAMES  = ["i0_CO2R [A/m2]", "lambda_MHC [eV]", "log10(HER_pf)", "alpha_c_HER"]

set_params!(p) = (set_i0_CO2R!(p[1]); set_lambda_MHC!(p[2]);
                  set_HER_prefactor!(10.0^p[3]); set_alpha_c_HER!(p[4]))
get_params()   = [i0_CO2R(), lambda_MHC(), log10(HER_prefactor()), alpha_c_HER()]

# ── Solver workspace: one closure, one ForwardDiff config, reused throughout ──
mutable struct FitWS
    vh::VHolder
    plu::AffineLU
    Jd::Matrix{Float64}
    cfg::Any
    res!::Any
end
function FitWS(ctx)
    vh = VHolder(E_exp_RHE[1])
    r! = (F_, u_) -> full_residual!(F_, u_, vh.V, ctx)
    Jd = zeros(ctx.ly.N_TOT, ctx.ly.N_TOT)
    cfg = ForwardDiff.JacobianConfig(r!, zeros(ctx.ly.N_TOT),
                                     zeros(ctx.ly.N_TOT), ForwardDiff.Chunk{12}())
    FitWS(vh, AffineLU(), Jd, cfg, r!)
end
const ws = FitWS(ctx4)

function polish!(u::Vector{Float64}, V::Float64; alpha_init = ALPHA_SWEEP, max_iter = MAXIT_SWEEP)
    ws.vh.V = V
    return newton_affine!(u, ws.res!; ly = ctx4.ly, plu = ws.plu, rtol = RTOL_PROD,
                          max_iter = max_iter, alpha_init = alpha_init,
                          Jdense = ws.Jd, cfg = ws.cfg)
end

"Polish one voltage, retrying damped (§9.5.7 backtracking) before giving up."
function polish_try!(u::Vector{Float64}, V::Float64)
    info = polish!(u, V; alpha_init = ALPHA_SWEEP)
    info.converged && return true
    info = polish!(u, V; alpha_init = ALPHA_ANCHOR)
    return info.converged
end

# ── §17.3 Pre-fitting voltage polish (required) ──────────────────────────────
"""
Solve Stage 4 at the EXACT experimental voltages, warm-starting each from the
nearest grid-cache entry.  `residual_stage4!` takes V in V vs RHE@pH8 and does
the SHE conversion itself (§15.3), so V_rhe is passed DIRECTLY — converting
here as well would subtract the 0.513 V offset twice.
"""
function build_exp_cache()
    grid = collect(V_RHE_grid)
    cache = Dict{Float64,Vector{Float64}}()
    for V in E_exp_RHE
        j = argmin(abs.(grid .- V))
        u = load_solution(cache_dir_for(4), cache_prefix_for(4), grid[j])
        u === nothing && error("No Stage-4 cache at V_RHE = $(grid[j]) — run run_stage4.jl first")
        u = copy(u)
        ok = polish_try!(u, V)
        ok || error("Pre-fit polish failed at V_RHE = $V")
        info = polish!(u, V)
        cache[V] = u
        @printf("  polish V_RHE = %+.3f  <- grid %+.3f   ||F||inf = %.2e\n", V, grid[j], info.residual)
    end
    return cache
end

println("\n--- §17.3 pre-fitting voltage polish at the 5 experimental voltages ---")
const exp_cache = build_exp_cache()
const p_last = copy(params0)

# ── §9.5.7 parameter-direction continuation, for NM trials the polish rejects ─
"""
Ramp the kinetic parameters from `pa` to `pb` in 2 -> 4 -> 8 sub-steps,
re-solving all 5 voltages at each.  Same continuation idea as the voltage
sweep, applied along the parameter direction.  Trials that still fail are
genuinely extreme and are rejected.
"""
function param_ramp!(cache, pa::Vector{Float64}, pb::Vector{Float64})
    for nsub in (2, 4, 8)
        tmp = Dict(V => copy(cache[V]) for V in E_exp_RHE)
        ok = true
        for s in 1:nsub
            set_params!(pa .+ (s/nsub) .* (pb .- pa))
            for V in E_exp_RHE
                u = tmp[V]
                if !polish_try!(u, V); ok = false; break; end
                tmp[V] = u
            end
            ok || break
        end
        if ok
            for V in E_exp_RHE; cache[V] = tmp[V]; end
            return true
        end
    end
    return false
end

# ── §17.4 Objective ──────────────────────────────────────────────────────────
const NEVAL = Ref(0); const NFAIL = Ref(0); const NRAMP = Ref(0)
const TRACE = Ref(true)

function fitting_objective(params::Vector{Float64})
    NEVAL[] += 1
    # §17.2 bound penalty — SLOPED, never a flat constant: a plateau ties two
    # simplex vertices together and Nelder-Mead returns at iteration 0.
    pen = 0.0
    for j in 1:4
        v = max(lower[j] - params[j], params[j] - upper[j], 0.0)
        pen += (v / (upper[j] - lower[j]))^2
    end
    if pen > 0
        val = 1e6 * (1 + pen)
        TRACE[] && @printf("  eval %4d  OUT OF BOUNDS  pen=%.4e -> %.6e\n", NEVAL[], pen, val)
        return val
    end

    set_params!(params)
    ok = true
    for V in E_exp_RHE
        u = copy(exp_cache[V])
        if polish_try!(u, V)
            exp_cache[V] = u
        else
            ok = false; break
        end
    end
    if !ok                                   # §9.5.7 recovery along the parameter direction
        NRAMP[] += 1
        set_params!(p_last)
        ok = param_ramp!(exp_cache, copy(p_last), copy(params))
        set_params!(params)
    end
    if !ok
        NFAIL[] += 1
        set_params!(p_last)
        # sloped, like the bound penalty above
        val = 1e6 * (1 + sum(((params .- p_last) ./ (upper .- lower)).^2))
        TRACE[] && @printf("  eval %4d  POLISH FAILED -> %.6e\n", NEVAL[], val)
        return val
    end
    p_last .= params

    sse = 0.0
    for (k, V) in enumerate(E_exp_RHE)
        iCO, iH2, _ = compute_currents(exp_cache[V], V, ctx4)
        sse += (log10(max(iCO/10, 0.1)) - log10(i_CO_exp[k]))^2
        sse += (log10(max(iH2/10, 0.1)) - log10(i_H2_exp[k]))^2
    end
    TRACE[] && @printf("  eval %4d  i0=%8.2f  lam=%.4f  log10pf=%7.4f  ac=%.5f  SSE=%.6f\n",
                       NEVAL[], params[1], params[2], params[3], params[4], sse)
    return sse
end

"Model currents at the 5 experimental voltages for the current cache."
function model_currents()
    co = Float64[]; h2 = Float64[]
    for V in E_exp_RHE
        a, b, _ = compute_currents(exp_cache[V], V, ctx4)
        push!(co, a/10); push!(h2, b/10)
    end
    return co, h2
end

# ── §17.5 Step 1: manual exploration ─────────────────────────────────────────
println("\n" * "="^78)
println("§17.5 STEP 1 — manual exploration (one parameter at a time)")
println("="^78)
TRACE[] = false
let base = copy(params0), cache_backup = Dict(V => copy(exp_cache[V]) for V in E_exp_RHE)
    sweeps = [(1, [100.0, 250.0, 540.0, 1500.0, 3000.0]),
              (2, [1.0, 1.4, 1.8, 2.2, 2.6]),
              (3, [12.0, 12.7, 13.382, 14.0, 15.0]),
              (4, [0.03, 0.06, 0.0872, 0.12, 0.20])]
    for (j, vals) in sweeps
        @printf("\n  %s:\n", PNAMES[j])
        println("       value        SSE      i_CO @ the 5 exp voltages [mA/cm2]        i_H2 @ same")
        for v in vals
            p = copy(base); p[j] = v
            for V in E_exp_RHE; exp_cache[V] = copy(cache_backup[V]); end
            p_last .= base
            s = fitting_objective(p)
            co, h2 = model_currents()
            @printf("   %10.4f  %9.4f   [%s]  [%s]\n", v, s,
                    join([@sprintf("%7.1f", x) for x in co], " "),
                    join([@sprintf("%7.1f", x) for x in h2], " "))
        end
    end
    for V in E_exp_RHE; exp_cache[V] = copy(cache_backup[V]); end
    set_params!(base); p_last .= base
    println("\n  exp targets:                       [", join([@sprintf("%7.1f", x) for x in i_CO_exp], " "),
            "]  [", join([@sprintf("%7.1f", x) for x in i_H2_exp], " "), "]")
end

# ── §17.5 Steps 2-4: staged Nelder-Mead ──────────────────────────────────────
"NM over a subset of the 4 parameters, holding the rest at their current values."
function nm_subset(idx::Vector{Int}, x0::Vector{Float64}, iters::Int; label::String)
    base = get_params()
    obj = function (x)
        p = copy(base)
        for (m, j) in enumerate(idx); p[j] = x[m]; end
        return fitting_objective(p)
    end
    @printf("\n--- %s: NM over %s from %s ---\n", label,
            join([PNAMES[j] for j in idx], ", "), string(round.(x0, sigdigits=6)))
    r = optimize(obj, x0, NelderMead(),
                 Optim.Options(iterations = iters, g_tol = 1e-6, f_reltol = 1e-6,
                               show_trace = false))
    p = copy(base)
    for (m, j) in enumerate(idx); p[j] = Optim.minimizer(r)[m]; end
    fitting_objective(p)                      # re-establish the cache at the minimizer
    @printf("  %s: %d NM iterations, %d f-evals, minimum SSE = %.6f, converged = %s\n",
            label, Optim.iterations(r), Optim.f_calls(r), Optim.minimum(r), Optim.converged(r))
    @printf("  -> %s\n", join([@sprintf("%s = %.6g", PNAMES[j], p[j]) for j in 1:4], ",  "))
    return p, r
end

println("\n" * "="^78)
println("§17.5 STEPS 2-4 — staged Nelder-Mead")
println("="^78)
TRACE[] = false

set_params!(params0); p_last .= params0
fitting_objective(params0)
@printf("\nStarting SSE at the initial guess = %.6f\n", fitting_objective(params0))

# Step 2: CO2R parameters first, HER held fixed.
p2, r2 = nm_subset([1, 2], [params0[1], params0[2]], 200; label = "STEP 2 (CO2R)")
set_params!(p2); p_last .= p2

# Step 3: HER parameters, CO2R held at the Step-2 values.
# §17.2: this is the step where a FLAT out-of-bounds penalty makes Optim return
# at iteration 0 (the AffineSimplexer(0.025, 0.5) vertex log10 HER_pf = 20.10
# is above the bound).  The sloped penalty above is what keeps it running.
p3, r3 = nm_subset([3, 4], [p2[3], p2[4]], 200; label = "STEP 3 (HER)")
set_params!(p3); p_last .= p3
@printf("  §17.2 check — Step 3 ran %d NM iterations (a flat penalty returns 0)\n",
        Optim.iterations(r3))

# Step 4: joint refinement over all four.
p4, r4 = nm_subset([1, 2, 3, 4], p3, 200; label = "STEP 4 (joint)")
set_params!(p4); p_last .= p4

const pfit = copy(p4)
println("\n" * "="^78)
println("FITTED KINETIC PARAMETERS")
println("="^78)
@printf("  i0_CO2R        = %.4f A/m^2      (initial guess 540)\n", pfit[1])
@printf("  lambda_MHC     = %.5f eV         (initial guess 1.8)\n", pfit[2])
@printf("  HER_prefactor  = %.6e            (initial guess 2.409e13, log10 = %.5f)\n",
        10.0^pfit[3], pfit[3])
@printf("  alpha_c_HER    = %.6f            (initial guess 0.0872)\n", pfit[4])
@printf("  final SSE      = %.6f  over %d objective evaluations\n", fitting_objective(pfit), NEVAL[])
@printf("  polish rescues via parameter ramp: %d ; trials rejected: %d\n", NRAMP[], NFAIL[])

# ── §17.6 Fit quality acceptance criteria ────────────────────────────────────
println("\n" * "="^78)
println("§17.6 FIT QUALITY ACCEPTANCE CRITERIA")
println("="^78)
set_params!(pfit)
let (co, h2) = model_currents()
    println("  per-point current errors (tolerance 30 %, except i_CO at -2.413 V: 80 %):")
    println("     V_RHE      i_CO model/exp   err     tol    ok      i_H2 model/exp   err     tol    ok")
    allok = true
    for (k, V) in enumerate(E_exp_RHE)
        eCO = 100*(co[k]-i_CO_exp[k])/i_CO_exp[k]
        eH2 = 100*(h2[k]-i_H2_exp[k])/i_H2_exp[k]
        tCO = (k == 5) ? 80.0 : 30.0
        okC = abs(eCO) <= tCO; okH = abs(eH2) <= 30.0
        allok &= (okC & okH)
        @printf("   %7.3f   %8.2f/%6.1f  %+7.1f%%  %4.0f%%   %s   %8.2f/%6.1f  %+7.1f%%  %4.0f%%   %s\n",
                V, co[k], i_CO_exp[k], eCO, tCO, okC ? "OK " : "MISS",
                h2[k], i_H2_exp[k], eH2, 30.0, okH ? "OK " : "MISS")
    end
    @printf("  all per-point tolerances met: %s\n", allok)
end

# ── §17.8 Final 89-voltage Stage-4 sweep at the fitted kinetics ──────────────
# A DIAGNOSTIC ARTIFACT: it isolates the effect of the fit from the effect of
# adding the potential fields.  Stage 5 does NOT warm-start from it (§17.8,
# §11 Stage 5 gotcha #0).
println("\n" * "="^78)
println("§17.8 refit sweep — 89-voltage Stage-4 sweep at the fitted kinetics")
println("  (diagnostic artifact; Stage 5 is seeded at V_c = 0, never from this cache)")
println("="^78)
isdir(CACHE_DIR) && rm(CACHE_DIR; recursive = true)     # §17.8 cache invalidation

V_chain = collect(V_RHE_grid)
emit    = fill(true, length(V_chain))
u0 = make_anchor_guess(ctx4, V_chain[1])
resf = run_continuation(ctx4, u0, V_chain, emit; cache_dir = CACHE_DIR, prefix = PREFIX)

@printf("\nRefit sweep: %d/%d converged | iters/step %.2f | max ||F||inf %.3e | %.2f s\n",
        length(resf.voltages), length(V_chain), resf.iters_per_step,
        maximum(resf.resids), resf.wall)

Vs = resf.voltages; sols = resf.solutions
iCO = Float64[]; iH2 = Float64[]; itot = Float64[]
for (k, V) in enumerate(Vs)
    a, b, c = compute_currents(sols[k], V, ctx4)
    push!(iCO, a/10); push!(iH2, b/10); push!(itot, c/10)
end
FE = 100 .* iCO ./ max.(itot, 1e-30)
kpk = argmax(FE)
nCO, nH2 = polarization_smoothness(Vs, iCO, iH2)
@printf("\n  i_CO peak %.2f mA/cm^2 at V_RHE = %.3f\n", maximum(iCO), Vs[argmax(iCO)])
@printf("  FE_CO peak %.1f %% at V_RHE = %.3f   (target 70-85 %%, exp 79.4 %% at -1.205)\n", FE[kpk], Vs[kpk])
@printf("  FE_CO peak location within 100 mV of -1.205 V: %s (|delta| = %.0f mV)\n",
        abs(Vs[kpk] + 1.205) <= 0.100, 1000*abs(Vs[kpk] + 1.205))
@printf("  sign changes di_CO/dV = %d (want exactly 1) ; di_H2/dV = %d (want 0)\n", nCO, nH2)
@printf("  i_H2 strictly monotone in |V_RHE|: %s\n", all(diff(iH2) .> 0))
@printf("  sweep coverage: %d/89 on one continuous branch\n", length(Vs))

# The 5 polished points must sit exactly on the 89-point sweep (§17.6 crit. 4).
println("\n  §17.6 criterion 4 — the 5 polished points vs the 89-point sweep:")
let (co, h2) = model_currents()
    ord = sortperm(Vs); Vso = Vs[ord]; cs = iCO[ord]; hs = iH2[ord]
    lin(x, xs, ys) = begin
        j = searchsortedfirst(xs, x)
        j <= 1 && return ys[1]; j > length(xs) && return ys[end]
        t = (x - xs[j-1])/(xs[j] - xs[j-1]); (1-t)*ys[j-1] + t*ys[j]
    end
    for (k, V) in enumerate(E_exp_RHE)
        @printf("    V=%+.3f  i_CO polished %8.2f vs sweep-interp %8.2f (%+.2f%%)   i_H2 %8.2f vs %8.2f (%+.2f%%)\n",
                V, co[k], lin(V, Vso, cs), 100*(co[k]-lin(V,Vso,cs))/max(lin(V,Vso,cs),1e-9),
                h2[k], lin(V, Vso, hs), 100*(h2[k]-lin(V,Vso,hs))/max(lin(V,Vso,hs),1e-9))
    end
end

stage3_checks(Vs, sols, ctx4; pinned = false)
stage_liquid_checks(Vs, sols, ctx4)
compare_to_experiment(Vs, iCO, iH2)

# ── Persist the fitted kinetics for run_stage5.jl (§17.8, §11 gotcha #14) ────
open(joinpath(@__DIR__, "fitted_kinetics.jl"), "w") do f
    println(f, "# Fitted kinetic parameters from run_fit.jl (§17).  Generated automatically.")
    println(f, "# Stage 5 MUST load these before its anchor solve (§11 Stage 5 gotcha #14).")
    @printf(f, "set_i0_CO2R!(%.10f)\n", pfit[1])
    @printf(f, "set_lambda_MHC!(%.10f)\n", pfit[2])
    @printf(f, "set_HER_prefactor!(%.10e)\n", 10.0^pfit[3])
    @printf(f, "set_alpha_c_HER!(%.10f)\n", pfit[4])
end
println("\nWrote fitted_kinetics.jl")

files = export_stage("fit", Vs, sols, ctx4)
println("Exported:"); for f in files; println("  ", f); end

println("\n" * "="^78)
println("FIT STEP COMPLETE — 🛑 STOP GATE.  Review before Stage 5.")
println("="^78)
