# =============================================================================
# newton_affine.jl — the §9.5 production solver stack, used by EVERY stage
#   AffineLU / aff_factor_solve! / aff_resolve  persistent sparse LU  (§9.5.2)
#   newton_affine!                              affine-covariant Newton (§9.5.3)
#   tangent_predictor!                          continuation tangent   (§9.5.4)
#
# Design rules taken directly from §9.4 / §9.5:
#   * monolithic Newton, all DOFs in one residual / one Jacobian / one solve
#   * exact ForwardDiff AD Jacobian, never finite differences
#   * NO residual norm anywhere in the kernel: damping, the step-acceptance
#     test and termination all measure Newton CORRECTIONS in an
#     iterate-derived weighted norm
#   * NO Jacobian row scaling, ever
#   * termination is an error estimate, not a step-size heuristic or an
#     iteration count; the residual guarantee is a post-solve assertion
# =============================================================================
using LinearAlgebra, SparseArrays, ForwardDiff, Printf

# ── §9.5.2 Persistent sparse LU workspace ────────────────────────────────────
mutable struct AffineLU
    F::Union{Nothing, SparseArrays.UMFPACK.UmfpackLU{Float64,Int}}
    pattern_nnz::Int
    n_symbolic::Int      # bookkeeping: how many times the symbolic phase ran
    n_numeric::Int       # how many times only the numeric refresh ran
    n_nnz_change::Int    # rebuilds triggered by the pattern_nnz guard
    n_lu_catch::Int      # rebuilds triggered by lu! refusing a changed pattern
end
AffineLU() = AffineLU(nothing, -1, 0, 0, 0, 0)

"""
    aff_factor_solve!(plu, J, b) -> (x, ok)

Factorize `J` (reusing the symbolic analysis when the sparsity pattern is
unchanged) and solve `J x = b`.  Leaves the factorization live in `plu` so
`aff_resolve` can back-substitute against it without refactorizing.
"""
function aff_factor_solve!(plu::AffineLU, J::SparseMatrixCSC{Float64,Int},
                           b::Vector{Float64})
    if plu.F === nothing || plu.pattern_nnz != nnz(J)
        plu.F === nothing || (plu.n_nnz_change += 1)
        try
            plu.F = lu(J); plu.pattern_nnz = nnz(J); plu.n_symbolic += 1
        catch
            return (zeros(length(b)), false)
        end
    else
        try
            lu!(plu.F, J); plu.n_numeric += 1        # refresh values, reuse symbolic
        catch
            plu.n_lu_catch += 1        # lu! refused: sparsity pattern changed at equal nnz
            try
                plu.F = lu(J); plu.pattern_nnz = nnz(J); plu.n_symbolic += 1
            catch
                return (zeros(length(b)), false)
            end
        end
    end
    x = plu.F \ b
    return (x, all(isfinite, x))
end

"Back-substitute against the live factorization (valid right after aff_factor_solve!)."
@inline aff_resolve(plu::AffineLU, b::Vector{Float64}) = plu.F \ b

# ── §9.5.3 step 1: iterate-derived per-DOF weights ───────────────────────────
"""
    build_weights!(w, u, blocks, afrac)

`w_j = |u_j| + afrac * rms(|u| over j's DOF block)`.  The block floor keeps a
DOF passing through zero (the gauged phi_L, or phi_s at the Stage-5 anchor)
from blowing up the norm.  If a whole block is *identically* zero the floor
would be zero and `d_j/w_j` is 0/0 — fall back to a unit weight there.
This fallback is the only weight in the kernel that is not iterate-derived.
"""
function build_weights!(w::Vector{Float64}, u::Vector{Float64},
                        blocks::Vector{UnitRange{Int}}, afrac::Float64)
    @inbounds for blk in blocks
        s = 0.0
        for j in blk; s += u[j] * u[j]; end
        rms = sqrt(s / length(blk))
        fl = (rms == 0.0) ? 1.0 : afrac * rms
        for j in blk; w[j] = abs(u[j]) + fl; end
    end
    return w
end

"Weighted norm used throughout the kernel: sqrt(sum((d_j/w_j)^2)/N)."
@inline function wnorm(d::Vector{Float64}, w::Vector{Float64})
    s = 0.0
    @inbounds for j in eachindex(d)
        t = d[j] / w[j]
        s += t * t
    end
    return sqrt(s / length(d))
end

# ── §9.5.3 The affine-covariant damped Newton kernel ─────────────────────────
"""
    newton_affine!(u, res!; ly, plu, rtol=1e-8, max_iter=40, alpha_init=1.0, ...)
        -> (converged, iterations, residual, err_est, theta, alpha_final,
            n_res, n_bsub, reason)

Error-oriented affine-covariant damped Newton (Deuflhard NLEQ-ERR).  `rtol`
is the accuracy dial: stop when the estimated remaining solution error — the
weighted norm of the simplified Newton correction — falls below it.  Mutates
`u`.  `residual` in the return is ||F||_inf at exit, reported for the §9.5.6
assertion, never used as a termination or damping test.
"""
function newton_affine!(u::Vector{Float64}, res!::F1;
                        ly::Layout,
                        plu::AffineLU,
                        rtol::Float64      = RTOL_PROD,
                        max_iter::Int      = MAXIT_SWEEP,
                        alpha_init::Float64 = ALPHA_SWEEP,
                        alpha_min::Float64 = ALPHA_MIN,
                        afrac::Float64     = AFRAC,
                        max_log_step::Float64 = MAX_LOG_STEP,
                        n_log_dofs::Int    = -1,
                        Jdense::Union{Nothing,Matrix{Float64}} = nothing,
                        cfg = nothing,
                        final_correction::Bool = true,
                        max_damp::Int      = 60,
                        max_enlarge::Int   = 3,
                        verbose::Bool      = false) where {F1}

    N  = length(u)
    nl = n_log_dofs < 0 ? ly.n_log_dofs : n_log_dofs

    Fv     = zeros(N)
    F_tr   = zeros(N)
    u_tr   = zeros(N)
    scr    = zeros(N)
    dubar_prev = zeros(N)
    w      = ones(N)
    Jd     = Jdense === nothing ? zeros(N, N) : Jdense
    cfg_   = cfg === nothing ?
             ForwardDiff.JacobianConfig(res!, zeros(N), copy(u), ForwardDiff.Chunk{12}()) : cfg

    n_res = 0; n_bsub = 0
    alpha = alpha_init; alpha_prev = alpha_init
    du_prev_n = NaN; dubar_prev_n = NaN
    du_n = NaN; dubar_n = NaN; theta = NaN
    converged = false; reason = "max_iter"; clamp_fired = false
    it = 0

    while it < max_iter
        it += 1
        build_weights!(w, u, ly.blocks, afrac)

        # ── AD Jacobian (§9.3.2); jacobian! also fills Fv with F(u) ──────────
        ForwardDiff.jacobian!(Jd, res!, Fv, u, cfg_)
        if !all(isfinite, Fv) || !all(isfinite, Jd)
            return (converged = false, iterations = it, residual = Inf,
                    err_est = Inf, theta = NaN, alpha_final = alpha,
                    n_res = n_res, n_bsub = n_bsub, reason = "nonfinite_jacobian")
        end
        Jsp = sparse(Jd)

        # ── Ordinary Newton correction: J du = -F  (direct sparse LU) ────────
        du, ok = aff_factor_solve!(plu, Jsp, -Fv)
        ok || return (converged = false, iterations = it,
                      residual = maximum(abs, Fv), err_est = Inf, theta = NaN,
                      alpha_final = alpha, n_res = n_res, n_bsub = n_bsub,
                      reason = "linear_solve_failed")

        # §9.4 item 4: clamp the liquid log-concentration block only
        @inbounds for j in 1:nl
            if abs(du[j]) > max_log_step
                du[j] = sign(du[j]) * max_log_step
                clamp_fired = true
            end
        end
        du_n = wnorm(du, w)

        # ── NLEQ-ERR a-priori convergence test (Deuflhard Alg. 3.13) ─────────
        # The §9.5.3 step listing gives termination only as step 6, on the
        # SIMPLIFIED correction after a trial is accepted.  The cited algorithm
        # also tests the ORDINARY correction before entering the damping loop,
        # and that test is load-bearing: started from an already-converged
        # state (a repeated §17.3 polish at unchanged kinetics, or the §9.5.6
        # assertion recovery), ||du||_w sits at the round-off floor of the
        # residual evaluation, F(u + alpha*du) cannot shrink any further, and
        # the monotonicity test ||dubar||_w <= (1 - alpha/2)||du||_w can never
        # be satisfied — alpha halves to alpha_min and the solve reports
        # `damping_stalled` on a perfectly good solution.  Accept the (tiny)
        # full step and stop, exactly as Alg. 3.13 does.
        if du_n <= rtol
            @inbounds for j in 1:N; u[j] += du[j]; end
            res!(Fv, u); n_res += 1
            dubar_n = du_n
            converged = true
            reason = clamp_fired ? "err_estimate_apriori(log_clamp_fired)" : "err_estimate_apriori"
            break
        end

        # ── §9.5.3 step 3: a-priori damping from the previous iteration ──────
        if it > 1 && isfinite(du_prev_n) && isfinite(dubar_prev_n)
            @inbounds for j in 1:N; scr[j] = dubar_prev[j] - du[j]; end
            den = wnorm(scr, w) * du_n
            den > 0 && (alpha = min(1.0, (du_prev_n * dubar_prev_n) / den * alpha_prev))
        end
        alpha = clamp(alpha, alpha_min, 1.0)

        # ── §9.5.3 steps 4-5: trial, simplified correction, monotonicity ─────
        accepted = false; n_enlarge = 0
        for _ in 1:max_damp
            @inbounds for j in 1:N; u_tr[j] = u[j] + alpha * du[j]; end
            res!(F_tr, u_tr); n_res += 1
            if all(isfinite, F_tr)
                dubar = aff_resolve(plu, -F_tr); n_bsub += 1
                if all(isfinite, dubar)
                    dbn = wnorm(dubar, w)
                    if dbn <= (1.0 - alpha/2) * du_n      # natural monotonicity test
                        # a-posteriori estimate: retry only if it proposes >= 4x alpha
                        if alpha < 1.0 && n_enlarge < max_enlarge
                            @inbounds for j in 1:N; scr[j] = dubar[j] - (1.0 - alpha)*du[j]; end
                            dn2 = wnorm(scr, w)
                            if dn2 > 0
                                a_post = min(1.0, alpha^2 * du_n / (2.0 * dn2))
                                if a_post >= 4.0 * alpha
                                    alpha = a_post; n_enlarge += 1
                                    continue
                                end
                            end
                        end
                        @inbounds for j in 1:N; u[j] = u_tr[j]; Fv[j] = F_tr[j]; end
                        dubar_n = dbn
                        @inbounds for j in 1:N; dubar_prev[j] = dubar[j]; end
                        accepted = true
                        break
                    end
                end
            end
            alpha *= 0.5
            alpha < alpha_min && break
        end

        if !accepted
            return (converged = false, iterations = it,
                    residual = maximum(abs, Fv), err_est = dubar_n, theta = theta,
                    alpha_final = alpha, n_res = n_res, n_bsub = n_bsub,
                    reason = "damping_stalled")
        end

        theta = dubar_n / du_n
        verbose && @printf("      it %2d  alpha=%.3e  ||du||_w=%.3e  ||dubar||_w=%.3e  theta=%.3e\n",
                           it, alpha, du_n, dubar_n, theta)

        du_prev_n = du_n; dubar_prev_n = dubar_n; alpha_prev = alpha

        # ── §9.5.3 step 6: termination on the error estimate ─────────────────
        if dubar_n <= rtol
            if final_correction
                @inbounds for j in 1:N; u[j] += dubar_prev[j]; end
                res!(Fv, u); n_res += 1        # report the honest exit residual
            end
            converged = true; reason = clamp_fired ? "err_estimate(log_clamp_fired)" : "err_estimate"
            break
        end
    end

    return (converged = converged, iterations = it,
            residual = maximum(abs, Fv), err_est = dubar_n, theta = theta,
            alpha_final = alpha, n_res = n_res, n_bsub = n_bsub, reason = reason)
end

# ── §9.5.4 Tangent continuation predictor ────────────────────────────────────
"""
    tangent_predictor!(dudV, plu, res_at_V!, u, V; h=1e-6, Fp, Fm) -> ok

Fill `dudV` with du/dV = -J^-1 (dF/dV) using the factorization left live by
the last `aff_factor_solve!`.  `res_at_V!(F, u, V)` evaluates the residual at
an arbitrary V.  Cost: two residual evaluations (central difference for
dF/dV) plus one back-substitution — about 5 % of a Newton iteration.
"""
function tangent_predictor!(dudV::Vector{Float64}, plu::AffineLU, res_at_V!::F2,
                            u::Vector{Float64}, V::Float64;
                            h::Float64 = 1e-6,
                            Fp::Vector{Float64}, Fm::Vector{Float64}) where {F2}
    plu.F === nothing && return false
    res_at_V!(Fp, u, V + h)
    res_at_V!(Fm, u, V - h)
    @. Fp = (Fp - Fm) / (2h)
    (all(isfinite, Fp)) || return false
    d = try
        aff_resolve(plu, -Fp)
    catch
        return false
    end
    all(isfinite, d) || return false
    dudV .= d
    return true
end
