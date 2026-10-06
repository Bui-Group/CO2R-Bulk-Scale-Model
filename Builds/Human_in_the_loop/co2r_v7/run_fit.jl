# =====================================================================
#  run_fit.jl — Fit step (after Stage 4).  §17
#  Fits i0_CO2R, lambda_MHC, log10(HER_prefactor), alpha_c_HER against the
#  5 Romiluyi et al. 2023 points using the STAGE 4 residual (phi_L = 0,
#  phi_s = V_applied_SHE) for speed.  Ends with the §17.8 diagnostic
#  89-voltage Stage-4 sweep at the fitted kinetics.
# =====================================================================
include("co2r_core.jl")
using Printf, Optim

const FIT_DIR    = joinpath(@__DIR__, "output_fit")
const FIT_CACHE  = joinpath(FIT_DIR, "cache")
const S4_CACHE   = joinpath(@__DIR__, "output_stage4", "cache")
save_fit(u, V) = save_solution(FIT_CACHE, "fit", u, V)
load_fit(V)    = load_solution(FIT_CACHE, "fit", V)

cache_banner("Fit step (Stage-4 residual)", FIT_CACHE)

N_CL = N_CL_DEFAULT; N_GDL = N_GDL_DEFAULT
cl   = make_cl_mesh(N_CL); gdl = make_gdl_mesh(N_GDL); gg = build_gas_geom(gdl, cl)
ly   = make_layout(4, N_CL, N_GDL); N = ly.N_TOT
c_eq, eq = equilibrium_vector()

plu  = AffineLU(); Jd = zeros(N, N); Vref = Ref(E_exp_RHE[1])
res!      = (F_,u_)    -> full_residual!(F_,u_,Vref[],cl,gdl,c_eq,ly,gg,Val(4))
res_at_V! = (F_,u_,V_) -> full_residual!(F_,u_,V_,   cl,gdl,c_eq,ly,gg,Val(4))
cfg  = ForwardDiff.JacobianConfig(res!, zeros(N), zeros(N), ForwardDiff.Chunk{12}())
n_log = 5*N_CL

# ── §17.3 pre-fitting voltage polish at the EXACT experimental voltages ─
"Warm-start from the nearest Stage-4 grid cache entry, then solve at V_rhe."
function nearest_s4(V_rhe)
    Vs = list_cache(S4_CACHE, "s4")
    isempty(Vs) && error("No Stage-4 cache in $S4_CACHE — run run_stage4.jl first!")
    return copy(load_solution(S4_CACHE, "s4", Vs[argmin(abs.(Vs .- V_rhe))]))
end

"""
Polish at every experimental voltage with the current kinetics.
Returns (ok, cache) — the cache is committed atomically so a failed trial
cannot poison the warm start of the next one.
"""
function polish_all(seed::Dict{Float64,Vector{Float64}}; rtol = RTOL_PROD, verbose = false)
    out = Dict{Float64,Vector{Float64}}()
    for V_rhe in E_exp_RHE
        u = copy(seed[V_rhe]); Vref[] = V_rhe
        info = newton_affine!(u, res!; ly=ly, plu=plu, rtol=rtol, max_iter=MAXIT_SWEEP,
                              alpha_init=ALPHA_SWEEP, n_log_dofs=n_log, Jdense=Jd, cfg=cfg)
        if !info.converged
            info = newton_affine!(copy(seed[V_rhe]) |> (x->(u.=x; u)), res!; ly=ly, plu=plu,
                                  rtol=rtol, max_iter=MAXIT_ANCHOR, alpha_init=ALPHA_ANCHOR,
                                  n_log_dofs=n_log, Jdense=Jd, cfg=cfg)
        end
        info.converged || return (false, seed)
        verbose && @printf("  polish V_RHE=%+.3f -> %d iters, |F|inf=%.2e\n",
                           V_rhe, info.iterations, info.residual)
        out[V_rhe] = u
    end
    return (true, out)
end

println("\n§17.3 pre-fitting polish at the 5 experimental voltages (default kinetics):")
seed0 = Dict(V => nearest_s4(V) for V in E_exp_RHE)
ok, exp_cache = polish_all(seed0; verbose = true)
ok || error("Initial polish failed")

# ── §17.4 objective ──────────────────────────────────────────────────
const P_LOWER = [  10.0, 0.5,  8.0, 0.01]
const P_UPPER = [5000.0, 3.0, 18.0, 0.50]
const NEVAL   = Ref(0)

apply_params!(p) = (set_i0_CO2R!(p[1]); set_lambda_MHC!(p[2]);
                    set_HER_prefactor!(10.0^p[3]); set_alpha_c_HER!(p[4]))

function model_currents(cache)
    iCO = zeros(5); iH2 = zeros(5)
    for (k, V) in enumerate(E_exp_RHE)
        a, b, _ = compute_currents(cache[V], V, cl, ly)
        iCO[k] = a/10; iH2[k] = b/10          # A/m² → mA/cm²
    end
    return iCO, iH2
end

# Parameters that `exp_cache` currently corresponds to.
const CUR_P = Ref(copy([540.0, 1.8, log10(2.409e13), 0.0872]))

"""
Polish at the 5 experimental voltages for a trial parameter vector.  A direct
warm-started solve is tried first; if it fails (a large jump in alpha_c or
HER_pf can move i_H2 by >20x in one step and the corrector cannot absorb it),
the kinetics are ramped from `CUR_P` to the trial values in 2 -> 4 -> 8
sub-steps, re-solving at each.  Same kernel throughout — this is the §9.5
continuation idea applied along the parameter direction instead of V.
"""
function polish_ramped(target_p::Vector{Float64})
    apply_params!(target_p)
    ok, cache = polish_all(exp_cache)
    ok && return (true, cache)
    for nsub in (2, 4, 8)
        cache_try = exp_cache; good = true
        for s in 1:nsub
            t = s / nsub
            apply_params!(CUR_P[] .+ t .* (target_p .- CUR_P[]))
            ok2, cache_try = polish_all(cache_try)
            ok2 || (good = false; break)
        end
        if good
            apply_params!(target_p)
            return (true, cache_try)
        end
    end
    apply_params!(target_p)
    return (false, exp_cache)
end

"""
Bound violation measured in box-fractions, squared and summed.

§17.2 prescribes a FLAT `1e6` for a bound violation.  That stalls Optim's
Nelder-Mead at iteration 0 whenever two initial-simplex vertices land on the
plateau with the *same* value: NM reports `g_converged` immediately and returns
`minimum = 1e6` while `minimizer` is the good vertex (verified in
`test_nm_penalty.jl`).  Sloping the penalty so violating vertices differ fixes
it at every penalty magnitude tested (1e0 … 1e6).
"""
function bounds_penalty(p::Vector{Float64})
    pen = 0.0
    for j in 1:4
        lo, hi = P_LOWER[j], P_UPPER[j]; sc = hi - lo
        p[j] < lo && (pen += ((lo - p[j])/sc)^2)
        p[j] > hi && (pen += ((p[j] - hi)/sc)^2)
    end
    return pen
end

function fitting_objective(params::Vector{Float64})
    pen = bounds_penalty(params)
    pen > 0 && return 1e6 * (1.0 + pen)
    NEVAL[] += 1
    ok, cache = polish_ramped(params)
    # non-convergence penalty also sloped — back toward the last feasible point
    ok || return 1e6 * (1.0 + sum(((params .- CUR_P[]) ./ (P_UPPER .- P_LOWER)).^2))
    global exp_cache = cache                  # commit only on full success
    CUR_P[] = copy(params)
    iCO, iH2 = model_currents(cache)
    sse = 0.0
    for k in 1:5
        sse += (log10(max(iCO[k], 0.1)) - log10(i_CO_exp[k]))^2
        sse += (log10(max(iH2[k], 0.1)) - log10(i_H2_exp[k]))^2
    end
    return sse
end

const P0 = [540.0, 1.8, log10(2.409e13), 0.0872]
@printf("\nInitial objective at the default kinetics: SSE = %.5f\n", fitting_objective(P0))
let (a,b) = model_currents(exp_cache)
    println("  V_RHE     i_CO model/exp        i_H2 model/exp")
    for k in 1:5
        @printf("  %+.3f   %8.2f / %6.1f    %8.2f / %6.1f\n",
                E_exp_RHE[k], a[k], i_CO_exp[k], b[k], i_H2_exp[k])
    end
end

# ── §17.5 Step 1: manual exploration (one-at-a-time sensitivity) ──────
println("\n§17.5 Step 1 — one-at-a-time parameter sensitivity (SSE):")
const PNAME = ["i0_CO2R", "lambda_MHC", "log10(HER_pf)", "alpha_c_HER"]
for (j, mults) in enumerate(([0.1,0.3,1.0,3.0,10.0], [0.6,0.8,1.0,1.2,1.4],
                             [-1.0,-0.5,0.0,0.5,1.0], [0.5,0.75,1.0,1.5,2.0]))
    row = String[]
    for m in mults
        p = copy(P0); p[j] = (j == 3) ? P0[j] + m : P0[j] * m
        p = clamp.(p, P_LOWER, P_UPPER)
        push!(row, @sprintf("%s=%.3g:%.3f", j==3 ? "+" : "x", j==3 ? m : m, fitting_objective(p)))
    end
    @printf("  %-14s %s\n", PNAME[j], join(row, "  "))
end

# ── §17.5 Steps 2-4: staged then joint Nelder-Mead ───────────────────
const OPTS = Optim.Options(iterations = 200, g_abstol = 1e-6, f_reltol = 1e-6,
                           show_trace = false)

println("\n§17.5 Step 2 — fit CO2R (i0, lambda) with HER held at the initial guess:")
apply_params!(P0)
r2 = optimize(q -> fitting_objective([q[1], q[2], P0[3], P0[4]]), [P0[1], P0[2]],
              NelderMead(), OPTS)
p2 = [Optim.minimizer(r2)[1], Optim.minimizer(r2)[2], P0[3], P0[4]]
@printf("  -> i0=%.3f  lambda=%.4f   SSE=%.5f  (%d evals)\n",
        p2[1], p2[2], Optim.minimum(r2), NEVAL[])

println("\n§17.5 Step 3 — fit HER (log10 pf, alpha_c) with CO2R fixed from Step 2:")
r3 = optimize(q -> fitting_objective([p2[1], p2[2], q[1], q[2]]), [p2[3], p2[4]],
              NelderMead(), OPTS)
p3 = [p2[1], p2[2], Optim.minimizer(r3)[1], Optim.minimizer(r3)[2]]
@printf("  -> log10(HER_pf)=%.4f  alpha_c=%.5f   SSE=%.5f  (%d evals)\n",
        p3[3], p3[4], Optim.minimum(r3), NEVAL[])

println("\n§17.5 Step 4 — joint 4-parameter Nelder-Mead refinement:")
r4 = optimize(fitting_objective, p3, NelderMead(),
              Optim.Options(iterations = 200, g_abstol = 1e-6, f_reltol = 1e-6,
                            show_trace = true, show_every = 20))
pbest = Optim.minimizer(r4); sse_best = Optim.minimum(r4)
if sse_best > Optim.minimum(r3)                  # keep the better of the two
    pbest = p3; sse_best = Optim.minimum(r3)
end
apply_params!(pbest)
ok, exp_cache = polish_all(exp_cache; verbose = false)

println("\n" * "="^72); println("FITTED KINETIC PARAMETERS"); println("="^72)
@printf("i0_CO2R       = %.4f A/m^2      (initial guess 540)\n",      pbest[1])
@printf("lambda_MHC    = %.5f eV         (initial guess 1.8)\n",      pbest[2])
@printf("HER_prefactor = %.6e mA/cm^2  (initial guess 2.409e13)\n", 10.0^pbest[3])
@printf("alpha_c_HER   = %.6f           (initial guess 0.0872)\n",  pbest[4])
@printf("SSE (log10 space, 10 points) = %.6f     total objective evals = %d\n",
        sse_best, NEVAL[])
@printf("MHC pre-factor pf = a_v*i0/(n_e F c_ref) = %.4f 1/s\n",
        a_v_CL*pbest[1]/(n_e*F_const*c_ref))

open(joinpath(@__DIR__, "fitted_params.jl"), "w") do io
    println(io, "# Fitted kinetic parameters — written by run_fit.jl (§17).")
    println(io, "# Loaded by run_stage5.jl BEFORE the Stage 5 anchor (§11 gotcha #14).")
    @printf(io, "set_i0_CO2R!(%.10e)\n",      pbest[1])
    @printf(io, "set_lambda_MHC!(%.10e)\n",   pbest[2])
    @printf(io, "set_HER_prefactor!(%.10e)\n", 10.0^pbest[3])
    @printf(io, "set_alpha_c_HER!(%.10e)\n",  pbest[4])
    @printf(io, "set_k_GL!(%.10e)\n",         k_GL())
end
open(joinpath(FIT_DIR, "fitted_kinetics.txt"), "w") do io
    @printf(io, "i0_CO2R        = %.6f       A/m^2\n",   pbest[1])
    @printf(io, "lambda_MHC     = %.6f          eV\n",    pbest[2])
    @printf(io, "HER_prefactor  = %.6e   mA/cm^2\n",      10.0^pbest[3])
    @printf(io, "alpha_c_HER    = %.6f          -\n",     pbest[4])
    @printf(io, "k_GL           = %.6e        m/s\n",     k_GL())
end
println("Wrote fitted_params.jl and output_fit/fitted_kinetics.txt")

# ── §17.6 acceptance criteria ────────────────────────────────────────
iCO, iH2 = model_currents(exp_cache)
println("\n" * "="^72); println("§17.6 FIT QUALITY / ACCEPTANCE CRITERIA"); println("="^72)
@printf("%-9s %10s %10s %9s | %10s %10s %9s\n",
        "V_RHE","i_CO mod","i_CO exp","rel err","i_H2 mod","i_H2 exp","rel err")
tolCO = [0.30,0.30,0.30,0.30,0.80]
passCO = trues(5); passH2 = trues(5)
for k in 1:5
    eCO = abs(iCO[k]-i_CO_exp[k])/i_CO_exp[k]; eH2 = abs(iH2[k]-i_H2_exp[k])/i_H2_exp[k]
    passCO[k] = eCO <= tolCO[k]; passH2[k] = eH2 <= 0.30
    @printf("%+.3f %10.2f %10.1f %8.1f%%%s | %10.2f %10.1f %8.1f%%%s\n",
            E_exp_RHE[k], iCO[k], i_CO_exp[k], 100*eCO, passCO[k] ? " " : "*",
            iH2[k], i_H2_exp[k], 100*eH2, passH2[k] ? " " : "*")
end
FEm = 100 .* iCO ./ (iCO .+ iH2)
@printf("\nFE_CO model : %s\nFE_CO exp   : %s\n",
        join([@sprintf("%.1f",x) for x in FEm], "  "),
        join([@sprintf("%.1f",x) for x in FE_CO_exp], "  "))
@printf("\ni_CO within tol at points 1-4 (<=30%%)      : %s\n", all(passCO[1:4]))
@printf("i_CO within tol at point 5 (<=80%%)        : %s\n", passCO[5])
@printf("i_H2 within 30%% at all 5 points            : %s\n", all(passH2))

# ── §17.8 diagnostic 89-voltage Stage-4 sweep at the fitted kinetics ──
println("\n§17.8 — final 89-voltage Stage-4 sweep at the FITTED kinetics")
println("        (diagnostic artifact; Stage 5 does NOT warm-start from it)")
print_kinetics("Fit sweep")
u0 = make_u0_gasliquid(c_eq, ly)
records, _, _ = continuation_sweep(Val(4); ly=ly, cl=cl, gdl=gdl, gg=gg, c_eq=c_eq,
    u0=u0, V_anchor_rhe=V_RHE_grid[1], V_chain_rhe=V_RHE_grid[2:end],
    cache_dir=FIT_CACHE, prefix="fit", verbose=false)
write_summary(FIT_DIR, records)
export_all_profiles(FIT_DIR, records, FIT_CACHE, "fit", cl, gdl, ly)
for V in E_exp_RHE                       # also cache the polished exact-V solutions
    save_solution(FIT_CACHE, "fitexp", exp_cache[V], V)
end

iCOs = [r.i_CO/10 for r in records]; iH2s = [r.i_H2/10 for r in records]
itot = iCOs .+ iH2s; FEs = 100 .* iCOs ./ max.(itot,1e-30)
dsign = sum(diff(sign.(diff(iCOs))) .!= 0)
@printf("\nSweep: %d/89 converged, mean iters %.2f, max |F|inf %.2e\n",
        length(records), sum(r.iters for r in records)/length(records),
        maximum(r.resid for r in records))
@printf("FE_CO peak     : %.1f %% at V_RHE = %.3f  (target 70-85%% near -1.205)\n",
        maximum(FEs), records[argmax(FEs)].V_rhe)
@printf("i_CO derivative sign changes : %d  (target <= 1)\n", dsign)
@printf("i_H2 strictly monotone in |V|: %s\n", all(diff(iH2s) .> 0))
@printf("89/89 on one branch          : %s\n", length(records) == 89)
println("\nOverlay check — polished exact-V points vs the 89-point sweep:")
for k in 1:5
    j = argmin(abs.([r.V_rhe for r in records] .- E_exp_RHE[k]))
    @printf("  V_RHE=%+.3f (grid %+.3f): i_CO polish %8.2f vs sweep %8.2f | i_H2 %8.2f vs %8.2f\n",
            E_exp_RHE[k], records[j].V_rhe, iCO[k], iCOs[j], iH2[k], iH2s[j])
end
println("="^72); flush(stdout)
