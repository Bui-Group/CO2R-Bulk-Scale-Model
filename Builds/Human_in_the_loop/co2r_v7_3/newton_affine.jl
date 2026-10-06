# ============================================================================
# newton_affine.jl — the §9.5 production solver stack
#
#   §9.5.2  AffineLU            persistent sparse LU workspace
#   §9.5.3  newton_affine!      affine-covariant damped Newton (NLEQ-ERR)
#   §9.5.4  tangent_predictor!  du/dV by one back-substitution
#
# NO residual norm is ever taken inside the kernel: damping, the step
# acceptance test and termination all measure Newton CORRECTIONS in an
# iterate-derived weighted norm.  Jacobian rows are never rescaled.
# ============================================================================
using SparseArrays, LinearAlgebra, ForwardDiff

# ── §9.5.2 Persistent sparse LU workspace ──────────────────────────────────
mutable struct AffineLU
    F::Union{Nothing,SparseArrays.UMFPACK.UmfpackLU{Float64,Int}}
    pattern_nnz::Int
end
AffineLU() = AffineLU(nothing, -1)

"""
    aff_factor_solve!(plu, J, b) -> (x, ok)

Factorize `J` (reusing the symbolic analysis when the sparsity pattern is
unchanged) and solve `J x = b`.  Leaves the factorization live in `plu` so
`aff_resolve` can back-substitute against it.
"""
function aff_factor_solve!(plu::AffineLU, J::SparseMatrixCSC{Float64,Int},
                           b::Vector{Float64})
    if plu.F === nothing || plu.pattern_nnz != nnz(J)
        try
            plu.F = lu(J); plu.pattern_nnz = nnz(J)
        catch
            return (zeros(length(b)), false)
        end
    else
        try
            lu!(plu.F, J)                      # refresh values, reuse symbolic
        catch
            try
                plu.F = lu(J); plu.pattern_nnz = nnz(J)
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

# ── residual closure with a mutable voltage (keeps the ForwardDiff tag fixed) ─
mutable struct ResFun{C}
    ctx::C
    V::Float64
end
(rf::ResFun)(F, u) = residual!(F, u, rf.V, rf.ctx)

# ── §9.5.3 weights and the weighted correction norm ────────────────────────
"""
    build_weights!(w, u, blocks, afrac)

`w_j = |u_j| + afrac · rms(|u| over j's DOF block)` (§9.5.3 step 1).
A block whose rms is EXACTLY zero (both potential blocks on the first
iteration of the Stage-5 V_c = 0 anchor) falls back to unit weights,
otherwise ‖·‖_w would be 0/0 = NaN.
"""
function build_weights!(w::Vector{Float64}, u::Vector{Float64},
                        blocks::Vector{UnitRange{Int}}, afrac::Float64)
    @inbounds for blk in blocks
        s = 0.0
        for j in blk; s += u[j]*u[j]; end
        rms = sqrt(s / length(blk))
        if rms == 0.0
            for j in blk; w[j] = 1.0; end
        else
            fl = afrac * rms
            for j in blk; w[j] = abs(u[j]) + fl; end
        end
    end
    return w
end

@inline function wnorm(d::Vector{Float64}, w::Vector{Float64})
    s = 0.0
    @inbounds for j in eachindex(d)
        t = d[j] / w[j]
        s += t*t
    end
    return sqrt(s / length(d))
end

# ── §9.5.3 the kernel ──────────────────────────────────────────────────────
"""
    newton_affine!(u, rf; ly, plu, Jd, cfg, rtol, max_iter, alpha_init, ...)
        -> (converged, iterations, residual, err_est, theta, alpha_final,
            n_res, n_bsub, reason)

Affine-covariant damped Newton, error-oriented (Deuflhard NLEQ-ERR).  `rtol`
is the accuracy dial: stop when the weighted norm of the simplified Newton
correction (the estimated remaining solution error) falls below it.  Mutates
`u`.  `residual` in the return is ‖F‖∞ at exit, reported for the §9.5.6
assertion, never used as a termination or damping test.
"""
function newton_affine!(u::Vector{Float64}, rf;
                        ly::Layout, plu::AffineLU,
                        Jd::Matrix{Float64}, cfg,
                        rtol::Float64 = RTOL_PROD,
                        max_iter::Int = 40,
                        alpha_init::Float64 = 1.0,
                        afrac::Float64 = AFRAC,
                        max_log_step::Float64 = MAX_LOG_STEP,
                        n_log_dofs::Int = 0,
                        alpha_min::Float64 = ALPHA_MIN,
                        max_damp::Int = 60,
                        final_correction::Bool = true)
    N    = length(u)
    blks = dof_blocks(ly)
    w    = ones(N)
    F    = zeros(N); F_tr = zeros(N); F_best = zeros(N)
    u_tr = zeros(N); u_best = zeros(N)
    scr  = zeros(N)
    du   = zeros(N); dubar = zeros(N); dubar_best = zeros(N); dubar_prev = zeros(N)

    fail(it, reason, nres, nbs) = (converged = false, iterations = it,
        residual = (all(isfinite, F) ? maximum(abs, F) : Inf), err_est = Inf,
        theta = NaN, alpha_final = NaN, n_res = nres, n_bsub = nbs, reason = reason)

    rf(F, u); n_res = 1; n_bsub = 0
    all(isfinite, F) || return fail(0, :nonfinite_initial_residual, n_res, n_bsub)

    alpha = alpha_init; alpha_prev = alpha_init
    du_prev_n = NaN; dubar_n = NaN; theta = NaN

    for it in 1:max_iter
        # ── ordinary Newton correction (§9.5.3 step 2) ─────────────────────
        ForwardDiff.jacobian!(Jd, rf, F_tr, u, cfg)
        all(isfinite, Jd) || return fail(it, :nonfinite_jacobian, n_res, n_bsub)
        Js = sparse(Jd)
        d, ok = aff_factor_solve!(plu, Js, -F)
        ok || return fail(it, :linear_solve_failed, n_res, n_bsub)
        copyto!(du, d)

        # log-step clamp on the liquid block only (§9.4 item 4)
        if n_log_dofs > 0
            m = 0.0
            @inbounds for j in 1:n_log_dofs; m = max(m, abs(du[j])); end
            if m > max_log_step
                s = max_log_step / m
                @inbounds for j in 1:n_log_dofs; du[j] *= s; end
            end
        end

        build_weights!(w, u, blks, afrac)
        du_n = wnorm(du, w)
        isfinite(du_n) || return fail(it, :nonfinite_step, n_res, n_bsub)
        if du_n == 0.0
            return (converged = true, iterations = it, residual = maximum(abs, F),
                    err_est = 0.0, theta = 0.0, alpha_final = alpha,
                    n_res = n_res, n_bsub = n_bsub, reason = :exact_step)
        end

        # ── a-priori damping from the previous iteration (§9.5.3 step 3) ───
        if it > 1 && isfinite(du_prev_n) && isfinite(dubar_n)
            @inbounds for j in 1:N; scr[j] = dubar_prev[j] - du[j]; end
            den = wnorm(scr, w) * du_n
            den > 0 && (alpha = min(1.0, (du_prev_n * dubar_n) / den * alpha_prev))
        end
        alpha = clamp(alpha, alpha_min, 1.0)

        # ── trial + simplified correction + monotonicity test (steps 4-5) ──
        accepted = false; dbn_best = NaN; alpha_best = NaN
        retries  = 0
        while true
            found = false
            while true
                @inbounds for j in 1:N; u_tr[j] = u[j] + alpha * du[j]; end
                rf(F_tr, u_tr); n_res += 1
                if all(isfinite, F_tr)
                    db = aff_resolve(plu, -F_tr); n_bsub += 1
                    if all(isfinite, db)
                        dbn = wnorm(db, w)
                        if dbn <= (1.0 - alpha/2) * du_n     # natural monotonicity test
                            copyto!(dubar, db); found = true
                            break
                        end
                    end
                end
                alpha *= 0.5
                alpha < alpha_min && break
            end
            if !found
                break                                   # damping search stalled
            end
            # keep this accepted trial
            copyto!(u_best, u_tr); copyto!(F_best, F_tr); copyto!(dubar_best, dubar)
            dbn_best = wnorm(dubar, w); alpha_best = alpha; accepted = true

            # a-posteriori α proposal; retry only if ≥ 4× larger (§9.5.3 step 5)
            @inbounds for j in 1:N; scr[j] = dubar[j] - (1.0 - alpha) * du[j]; end
            den2 = wnorm(scr, w)
            alpha_post = den2 > 0 ? min(1.0, 0.5 * alpha^2 * du_n / den2) : 1.0
            if retries < 2 && alpha < 1.0 && alpha_post >= 4 * alpha
                retries += 1
                alpha = min(1.0, alpha_post)
                continue                                # one more trial, larger α
            end
            break
        end
        accepted || return fail(it, :damping_stalled, n_res, n_bsub)

        copyto!(u, u_best); copyto!(F, F_best); copyto!(dubar, dubar_best)
        dubar_n = dbn_best; alpha = alpha_best
        theta = dubar_n / du_n

        # ── termination on the error estimate (§9.5.3 step 6) ──────────────
        if dubar_n <= rtol
            if final_correction
                @inbounds for j in 1:N; u[j] += dubar[j]; end
                rf(F, u); n_res += 1
            end
            return (converged = true, iterations = it, residual = maximum(abs, F),
                    err_est = dubar_n, theta = theta, alpha_final = alpha,
                    n_res = n_res, n_bsub = n_bsub, reason = :converged)
        end

        copyto!(dubar_prev, dubar)
        du_prev_n = du_n
        alpha_prev = alpha
    end
    return fail(max_iter, :max_iter, n_res, n_bsub)
end

# ── §9.5.4 tangent continuation predictor ──────────────────────────────────
"""
    tangent_predictor!(dudV, plu, rf, u, V; h = 1e-6, Fp, Fm) -> ok

Fill `dudV` with du/dV = -J⁻¹ (∂F/∂V) using the live factorization.
Cost: two residual evaluations (central difference for ∂F/∂V) + one
back-substitution.
"""
function tangent_predictor!(dudV::Vector{Float64}, plu::AffineLU, rf,
                            u::Vector{Float64}, V::Float64;
                            h::Float64 = 1e-6,
                            Fp::Vector{Float64} = zeros(length(u)),
                            Fm::Vector{Float64} = zeros(length(u)))
    plu.F === nothing && return false
    Vsave = rf.V
    rf.V = V + h; rf(Fp, u)
    rf.V = V - h; rf(Fm, u)
    rf.V = Vsave
    (all(isfinite, Fp) && all(isfinite, Fm)) || return false
    @. Fp = (Fp - Fm) / (2h)
    d = try
        aff_resolve(plu, -Fp)
    catch
        return false
    end
    all(isfinite, d) || return false
    copyto!(dudV, d)
    return true
end
