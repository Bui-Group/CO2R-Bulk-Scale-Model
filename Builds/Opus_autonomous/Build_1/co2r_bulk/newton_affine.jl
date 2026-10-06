# newton_affine.jl — §9.5 production solver stack
#   AffineLU              persistent sparse LU workspace          (§9.5.2)
#   newton_affine!        affine-covariant damped Newton (NLEQ-ERR) (§9.5.3)
#   tangent_predictor!    continuation tangent du/dV               (§9.5.4)
#
# NO residual norm is ever used for damping, acceptance, or termination.

using SparseArrays, LinearAlgebra, ForwardDiff, Printf

# ─────────────────────────────────────────────────────────────────────────────
# §9.5.2 Persistent sparse LU workspace
# ─────────────────────────────────────────────────────────────────────────────
mutable struct AffineLU
    F::Any                      # UmfpackLU or nothing
    colptr::Vector{Int}
    rowval::Vector{Int}
end
AffineLU() = AffineLU(nothing, Int[], Int[])

function aff_factor_solve!(plu::AffineLU, J::SparseMatrixCSC{Float64,Int},
                           b::Vector{Float64})
    same_pattern = (plu.F !== nothing) &&
                   (length(plu.colptr) == length(J.colptr)) &&
                   (length(plu.rowval) == length(J.rowval)) &&
                   (plu.colptr == J.colptr) && (plu.rowval == J.rowval)
    if !same_pattern
        try
            plu.F = lu(J)
            plu.colptr = copy(J.colptr); plu.rowval = copy(J.rowval)
        catch
            return (zeros(length(b)), false)
        end
    else
        try
            lu!(plu.F, J)
        catch
            try
                plu.F = lu(J)
                plu.colptr = copy(J.colptr); plu.rowval = copy(J.rowval)
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

# ─────────────────────────────────────────────────────────────────────────────
# Weighted norm and iterate-derived weights (§9.5.3 step 1)
# ─────────────────────────────────────────────────────────────────────────────
function wnorm(d::AbstractVector{Float64}, w::Vector{Float64})
    N = length(d)
    s = 0.0
    @inbounds for j in 1:N
        t = d[j] / w[j]
        s += t*t
    end
    return sqrt(s / N)
end

"""
    build_weights!(w, u, blocks, afrac)

`w_j = |u_j| + afrac * rms(|u| over j's DOF block)`.  A block whose rms is
exactly zero (e.g. the φ_s / φ_L blocks at the Stage-5 V_c = 0 anchor) falls
back to a unit weight — otherwise ‖·‖_w returns NaN (§9.5.3).
"""
function build_weights!(w::Vector{Float64}, u::Vector{Float64},
                        blocks::Vector{UnitRange{Int}}, afrac::Float64)
    @inbounds for rng in blocks
        s = 0.0
        for j in rng; s += u[j]*u[j]; end
        rms = sqrt(s / length(rng))
        if rms == 0.0
            for j in rng; w[j] = 1.0; end
        else
            fl = afrac * rms
            for j in rng; w[j] = abs(u[j]) + fl; end
        end
    end
    return w
end

# ─────────────────────────────────────────────────────────────────────────────
# §9.5.3 Affine-covariant damped Newton kernel
# ─────────────────────────────────────────────────────────────────────────────
"""
    newton_affine!(u, res!; blocks, plu, jacfun!, rtol, max_iter, alpha_init,
                   afrac, max_log_step, n_log_dofs, alpha_min, max_damp)
      -> (converged, iterations, residual, err_est, theta, alpha_final,
          n_res, n_bsub, reason)

`jacfun!(J, u)` must fill the dense `J` with the exact (AD) Jacobian.
Mutates `u`.  `residual` is ‖F‖∞ at exit — reported only, never used as a
termination or damping test.
"""
function newton_affine!(u::Vector{Float64}, res!;
                        blocks::Vector{UnitRange{Int}},
                        plu::AffineLU,
                        jacfun!,
                        Jdense::Matrix{Float64},
                        rtol::Float64=1e-8,
                        max_iter::Int=40,
                        alpha_init::Float64=1.0,
                        afrac::Float64=1e-3,
                        max_log_step::Float64=5.0,
                        n_log_dofs::Int=0,
                        alpha_min::Float64=1e-8,
                        max_damp::Int=60,
                        final_correction::Bool=true)
    N = length(u)
    F     = zeros(N); F_tr = zeros(N)
    u_tr  = zeros(N)
    w     = ones(N)
    scr   = zeros(N)
    dubar_prev = zeros(N)
    du_prev_n  = NaN
    dubar_prev_n = NaN
    alpha_prev = alpha_init
    n_res = 0; n_bsub = 0
    theta = NaN
    err_est = NaN

    res!(F, u); n_res += 1
    if !all(isfinite, F)
        return (converged=false, iterations=0, residual=Inf, err_est=NaN,
                theta=NaN, alpha_final=0.0, n_res=n_res, n_bsub=n_bsub,
                reason=:nonfinite_initial_residual)
    end

    alpha = alpha_init
    it = 0
    while it < max_iter
        it += 1
        build_weights!(w, u, blocks, afrac)

        jacfun!(Jdense, u)
        if !all(isfinite, Jdense)
            return (converged=false, iterations=it, residual=maximum(abs, F),
                    err_est=err_est, theta=theta, alpha_final=alpha,
                    n_res=n_res, n_bsub=n_bsub, reason=:nonfinite_jacobian)
        end
        Jsp = sparse(Jdense)
        @. scr = -F
        du, ok = aff_factor_solve!(plu, Jsp, copy(scr))
        if !ok
            return (converged=false, iterations=it, residual=maximum(abs, F),
                    err_est=err_est, theta=theta, alpha_final=alpha,
                    n_res=n_res, n_bsub=n_bsub, reason=:linear_solve_failed)
        end

        # log-step clamp on the liquid block only (§9.4 item 4)
        if n_log_dofs > 0
            mx = 0.0
            @inbounds for j in 1:n_log_dofs; mx = max(mx, abs(du[j])); end
            if mx > max_log_step
                sc = max_log_step / mx
                @inbounds for j in 1:n_log_dofs; du[j] *= sc; end
            end
        end

        du_n = wnorm(du, w)
        if !isfinite(du_n)
            return (converged=false, iterations=it, residual=maximum(abs, F),
                    err_est=err_est, theta=theta, alpha_final=alpha,
                    n_res=n_res, n_bsub=n_bsub, reason=:nonfinite_correction)
        end
        if du_n == 0.0
            return (converged=true, iterations=it, residual=maximum(abs, F),
                    err_est=0.0, theta=0.0, alpha_final=alpha,
                    n_res=n_res, n_bsub=n_bsub, reason=:zero_correction)
        end

        # ── a-priori damping from the previous iteration
        if it > 1 && isfinite(du_prev_n) && isfinite(dubar_prev_n)
            @inbounds for j in 1:N; scr[j] = dubar_prev[j] - du[j]; end
            den = wnorm(scr, w) * du_n
            if den > 0
                alpha = min(1.0, (du_prev_n * dubar_prev_n) / den * alpha_prev)
            end
        end
        alpha = clamp(alpha, alpha_min, 1.0)

        # ── trial + simplified correction + monotonicity test
        accepted = false
        dbn = NaN
        local dubar
        for _ in 1:max_damp
            @inbounds for j in 1:N; u_tr[j] = u[j] + alpha * du[j]; end
            res!(F_tr, u_tr); n_res += 1
            if all(isfinite, F_tr)
                @. scr = -F_tr
                dubar = aff_resolve(plu, copy(scr)); n_bsub += 1
                if all(isfinite, dubar)
                    dbn = wnorm(dubar, w)
                    if dbn <= (1.0 - alpha/2) * du_n
                        accepted = true
                        break
                    end
                end
            end
            alpha *= 0.5
            alpha < alpha_min && break
        end

        if !accepted
            return (converged=false, iterations=it, residual=maximum(abs, F),
                    err_est=err_est, theta=theta, alpha_final=alpha,
                    n_res=n_res, n_bsub=n_bsub, reason=:damping_stalled)
        end

        copyto!(u, u_tr); copyto!(F, F_tr)
        theta   = dbn / du_n
        err_est = dbn

        if dbn <= rtol
            if final_correction
                @inbounds for j in 1:N; u[j] += dubar[j]; end
                res!(F, u); n_res += 1
            end
            return (converged=true, iterations=it, residual=maximum(abs, F),
                    err_est=err_est, theta=theta, alpha_final=alpha,
                    n_res=n_res, n_bsub=n_bsub, reason=:converged)
        end

        copyto!(dubar_prev, dubar)
        du_prev_n    = du_n
        dubar_prev_n = dbn
        alpha_prev   = alpha
    end

    return (converged=false, iterations=it, residual=maximum(abs, F),
            err_est=err_est, theta=theta, alpha_final=alpha,
            n_res=n_res, n_bsub=n_bsub, reason=:max_iter)
end

# ─────────────────────────────────────────────────────────────────────────────
# §9.5.4 Tangent continuation predictor
# ─────────────────────────────────────────────────────────────────────────────
"""
    tangent_predictor!(dudV, plu, res_at_V!, u, V; h=1e-6) -> ok

Fill `dudV` with du/dV = -J⁻¹ ∂F/∂V using the live factorization in `plu`.
"""
function tangent_predictor!(dudV::Vector{Float64}, plu::AffineLU, res_at_V!,
                            u::Vector{Float64}, V::Float64; h::Float64=1e-6)
    plu.F === nothing && return false
    N = length(u)
    Fp = zeros(N); Fm = zeros(N)
    res_at_V!(Fp, u, V + h)
    res_at_V!(Fm, u, V - h)
    @. Fp = (Fp - Fm) / (2h)
    all(isfinite, Fp) || return false
    local d
    try
        d = aff_resolve(plu, -Fp)
    catch
        return false
    end
    all(isfinite, d) || return false
    copyto!(dudV, d)
    return true
end
