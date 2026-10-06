# newton_affine.jl — §9.5 production solver stack
#   AffineLU            persistent sparse LU workspace          (§9.5.2)
#   newton_affine!      affine-covariant damped Newton kernel   (§9.5.3)
#   tangent_predictor!  continuation tangent du/dV              (§9.5.4)
#
# No residual norm is ever used for damping, step acceptance, or termination.

using LinearAlgebra
using SparseArrays
using ForwardDiff

# ──────────────────────────────────────────────────────────────────────────
# §9.5.2 Persistent sparse LU workspace
# ──────────────────────────────────────────────────────────────────────────
mutable struct AffineLU
    F::Union{Nothing,SparseArrays.UMFPACK.UmfpackLU{Float64,Int}}
    colptr::Vector{Int}
    rowval::Vector{Int}
    n_symbolic::Int
    n_numeric::Int
end
AffineLU() = AffineLU(nothing, Int[], Int[], 0, 0)

"""
    aff_factor_solve!(plu, J, b) -> (x, ok)

Factorize `J`, reusing the symbolic analysis whenever the sparsity pattern is
unchanged, and solve `J x = b`.  The factorization is left live in `plu` so
`aff_resolve` can back-substitute against it without refactorizing.
"""
function aff_factor_solve!(plu::AffineLU, J::SparseMatrixCSC{Float64,Int},
                           b::Vector{Float64})
    same_pattern = plu.F !== nothing &&
                   length(plu.colptr) == length(J.colptr) &&
                   plu.colptr == J.colptr && plu.rowval == J.rowval
    if !same_pattern
        try
            plu.F = lu(J)
            plu.colptr = copy(J.colptr); plu.rowval = copy(J.rowval)
            plu.n_symbolic += 1; plu.n_numeric += 1
        catch
            return (zeros(length(b)), false)
        end
    else
        try
            lu!(plu.F, J)                       # refresh values, reuse symbolic
            plu.n_numeric += 1
        catch
            try
                plu.F = lu(J)
                plu.colptr = copy(J.colptr); plu.rowval = copy(J.rowval)
                plu.n_symbolic += 1; plu.n_numeric += 1
            catch
                return (zeros(length(b)), false)
            end
        end
    end
    local x
    try
        x = plu.F \ b
    catch
        return (zeros(length(b)), false)
    end
    return (x, all(isfinite, x))
end

"Back-substitute against the live factorization (valid right after aff_factor_solve!)."
@inline aff_resolve(plu::AffineLU, b::Vector{Float64}) = plu.F \ b

# ──────────────────────────────────────────────────────────────────────────
# Weighted correction norm (§9.5.3 step 1)
# ──────────────────────────────────────────────────────────────────────────
@inline function wnorm(d::Vector{Float64}, w::Vector{Float64})
    s = 0.0
    @inbounds for j in eachindex(d)
        t = d[j] / w[j]
        s += t * t
    end
    return sqrt(s / length(d))
end

"""
    build_weights!(w, u, blocks, afrac)

`w_j = |u_j| + afrac · rms(|u| over j's DOF block)`.  A block whose rms is
exactly zero (the Stage-5 anchor sets φ_s = φ_L = 0) falls back to unit
weights — §9.5.3 step 1.
"""
function build_weights!(w::Vector{Float64}, u::Vector{Float64},
                        blocks::Vector{UnitRange{Int}}, afrac::Float64)
    @inbounds for r in blocks
        s = 0.0
        for j in r; s += u[j] * u[j]; end
        rms = sqrt(s / length(r))
        if rms == 0.0
            for j in r; w[j] = 1.0; end
        else
            fl = afrac * rms
            for j in r; w[j] = abs(u[j]) + fl; end
        end
    end
    return w
end

"Scale the leading `n_log` (liquid log-concentration) entries of `du` so that
 their max magnitude does not exceed `max_log_step` (§9.4 item 4)."
function clamp_log_block!(du::Vector{Float64}, n_log::Int, max_log_step::Float64)
    n_log <= 0 && return du
    m = 0.0
    @inbounds for j in 1:n_log
        a = abs(du[j]); a > m && (m = a)
    end
    if m > max_log_step
        s = max_log_step / m
        @inbounds for j in 1:n_log; du[j] *= s; end
    end
    return du
end

# ──────────────────────────────────────────────────────────────────────────
# §9.5.3 Affine-covariant damped Newton (error-oriented, NLEQ-ERR family)
# ──────────────────────────────────────────────────────────────────────────
"""
    newton_affine!(u, res!; blocks, plu, ...) -> NamedTuple

Affine-covariant damped Newton.  `rtol` is the accuracy dial: stop when the
weighted norm of the simplified Newton correction (the estimate of the
remaining solution error) falls below it.  Mutates `u`.

Returns `(converged, iterations, residual, err_est, theta, alpha_final,
          n_res, n_bsub, reason)`; `residual` is ‖F‖∞ at exit, reported for the
§9.5.6 assertion and never used as a termination or damping test.
"""
function newton_affine!(u::Vector{Float64}, res!;
                        blocks::Vector{UnitRange{Int}},
                        plu::AffineLU,
                        rtol::Float64 = 1e-8,
                        max_iter::Int = 40,
                        alpha_init::Float64 = 1.0,
                        alpha_min::Float64 = 1e-8,
                        afrac::Float64 = 1e-3,
                        max_log_step::Float64 = 5.0,
                        n_log_dofs::Int = 0,
                        Jdense::Union{Nothing,Matrix{Float64}} = nothing,
                        cfg = nothing,
                        final_correction::Bool = true,
                        max_damp::Int = 60)
    N = length(u)
    Jd  = Jdense === nothing ? zeros(N, N) : Jdense
    ybuf = zeros(N)
    jcfg = cfg === nothing ?
           ForwardDiff.JacobianConfig(res!, ybuf, u, ForwardDiff.Chunk{12}()) : cfg

    F      = zeros(N); F_tr = zeros(N); u_tr = zeros(N)
    w      = ones(N);  scr  = zeros(N)
    du     = zeros(N); dubar = zeros(N); dubar_prev = zeros(N)
    n_res  = 0; n_bsub = 0

    res!(F, u); n_res += 1
    if !all(isfinite, F)
        return (converged = false, iterations = 0, residual = Inf, err_est = Inf,
                theta = NaN, alpha_final = NaN, n_res = n_res, n_bsub = n_bsub,
                reason = :nonfinite_residual)
    end

    alpha_prev = alpha_init
    du_prev_n  = NaN
    dubar_n    = NaN
    theta      = NaN
    alpha      = alpha_init

    for it in 1:max_iter
        build_weights!(w, u, blocks, afrac)
        ForwardDiff.jacobian!(Jd, res!, ybuf, u, jcfg)
        if !all(isfinite, Jd)
            return (converged = false, iterations = it, residual = maximum(abs, F),
                    err_est = NaN, theta = theta, alpha_final = alpha,
                    n_res = n_res, n_bsub = n_bsub, reason = :nonfinite_jacobian)
        end
        Js = sparse(Jd)
        (d, ok) = aff_factor_solve!(plu, Js, -F)
        if !ok
            return (converged = false, iterations = it, residual = maximum(abs, F),
                    err_est = NaN, theta = theta, alpha_final = alpha,
                    n_res = n_res, n_bsub = n_bsub, reason = :lu_failed)
        end
        du .= d
        clamp_log_block!(du, n_log_dofs, max_log_step)
        du_n = wnorm(du, w)

        if du_n == 0.0        # exact solution
            return (converged = true, iterations = it, residual = maximum(abs, F),
                    err_est = 0.0, theta = 0.0, alpha_final = 1.0,
                    n_res = n_res, n_bsub = n_bsub, reason = :exact)
        end

        # The ORDINARY correction is already inside the accuracy dial: the
        # iterate is converged in the affine-covariant sense and there is
        # nothing for the damping loop to do.  Without this guard, re-solving
        # an already-converged state (a Fit-step re-polish at an unchanged
        # parameter vector, or the §9.5.6 assertion's extra iteration) enters
        # the damping loop with du and dubar both at round-off, the
        # monotonicity test can never be satisfied, and the kernel reports a
        # spurious `damping_stalled`.
        if du_n <= rtol
            final_correction && (u .+= du)
            res!(F, u); n_res += 1
            return (converged = true, iterations = it, residual = maximum(abs, F),
                    err_est = du_n, theta = 0.0, alpha_final = 1.0,
                    n_res = n_res, n_bsub = n_bsub, reason = :rtol_ordinary)
        end

        # ── a-priori damping factor from the previous iteration
        if it > 1 && isfinite(du_prev_n) && isfinite(dubar_n)
            @inbounds for j in 1:N; scr[j] = dubar_prev[j] - du[j]; end
            den = wnorm(scr, w) * du_n
            alpha = den > 0 ? min(1.0, (du_prev_n * dubar_n) / den * alpha_prev) : 1.0
        else
            alpha = alpha_init
        end
        alpha = clamp(alpha, alpha_min, 1.0)

        # ── trial / simplified correction / monotonicity test
        accepted = false
        dbn = NaN
        for _ in 1:max_damp
            @inbounds for j in 1:N; u_tr[j] = u[j] + alpha * du[j]; end
            res!(F_tr, u_tr); n_res += 1
            if all(isfinite, F_tr)
                dbv = aff_resolve(plu, -F_tr); n_bsub += 1
                if all(isfinite, dbv)
                    dubar .= dbv
                    dbn = wnorm(dubar, w)
                    if dbn <= (1.0 - alpha / 2) * du_n
                        # a-posteriori proposal: retry only if ≥ 4× larger
                        @inbounds for j in 1:N
                            scr[j] = dubar[j] - (1.0 - alpha) * du[j]
                        end
                        sn = wnorm(scr, w)
                        if alpha < 1.0 && sn > 0
                            a_post = min(1.0, 0.5 * du_n * alpha^2 / sn)
                            if a_post >= 4 * alpha
                                alpha = a_post
                                continue
                            end
                        end
                        u .= u_tr; F .= F_tr
                        accepted = true
                        break
                    end
                end
            end
            alpha *= 0.5
            alpha < alpha_min && break
        end

        if !accepted
            return (converged = false, iterations = it, residual = maximum(abs, F),
                    err_est = NaN, theta = theta, alpha_final = alpha,
                    n_res = n_res, n_bsub = n_bsub, reason = :damping_stalled)
        end

        theta = dbn / du_n

        if dbn <= rtol
            final_correction && (u .+= dubar)
            res!(F, u); n_res += 1
            return (converged = true, iterations = it, residual = maximum(abs, F),
                    err_est = dbn, theta = theta, alpha_final = alpha,
                    n_res = n_res, n_bsub = n_bsub, reason = :rtol)
        end

        dubar_prev .= dubar
        du_prev_n = du_n
        dubar_n   = dbn
        alpha_prev = alpha
    end

    return (converged = false, iterations = max_iter, residual = maximum(abs, F),
            err_est = dubar_n, theta = theta, alpha_final = alpha,
            n_res = n_res, n_bsub = n_bsub, reason = :max_iter)
end

# ──────────────────────────────────────────────────────────────────────────
# §9.5.4 Tangent continuation predictor
# ──────────────────────────────────────────────────────────────────────────
"""
    tangent_predictor!(dudV, plu, res_at_V!, u, V; h = 1e-6) -> ok

Fill `dudV` with du/dV = −J⁻¹ ∂F/∂V using the live factorization left by the
last `aff_factor_solve!`.  Cost: two residual evaluations + one back-substitution.
"""
function tangent_predictor!(dudV::Vector{Float64}, plu::AffineLU, res_at_V!,
                            u::Vector{Float64}, V::Float64; h::Float64 = 1e-6)
    plu.F === nothing && return false
    N = length(u)
    Fp = zeros(N); Fm = zeros(N)
    res_at_V!(Fp, u, V + h)
    res_at_V!(Fm, u, V - h)
    @. Fp = (Fp - Fm) / (2h)
    (all(isfinite, Fp)) || return false
    local d
    try
        d = aff_resolve(plu, -Fp)
    catch
        return false
    end
    all(isfinite, d) || return false
    dudV .= d
    return true
end
