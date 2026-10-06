# newton_affine.jl — persistent sparse LU + affine-covariant damped Newton
# (error-oriented, NLEQ-ERR family) + tangent-predictor continuation, §9.5.
#
# NOTE on fidelity: the guide (§9.5.3) shows key excerpts of a ~180-line
# reference implementation but not the file verbatim (the a-posteriori
# "propose a larger alpha for an immediate retry" branch is elided with
# "..."). This file reproduces the required, testable behaviour listed in
# the §9.5.8 checklist exactly (correction-norm damping, error-estimate
# termination, no residual norm anywhere, block weight fallback for
# identically-zero blocks, final correction applied for free) and
# implements the a-priori damping estimate from the excerpt. The optional
# a-posteriori "retry with 4x larger alpha" performance refinement is
# NOT implemented — it only affects iteration count, not correctness or
# which branch is found — and this simplification is recorded in
# BUILD_DIALOGUE.md as a deliberate deviation.

using LinearAlgebra
using SparseArrays
using ForwardDiff

mutable struct AffineLU
    F::Any                 # live factorization (SparseArrays.UMFPACK.UmfpackLU)
    pattern_nnz::Int
end
AffineLU() = AffineLU(nothing, -1)

"""
    aff_factor_solve!(plu, J, b) -> (x, ok)

Factorize J (reusing symbolic analysis when nnz pattern is unchanged) and
solve J x = b. Leaves the factorization live in `plu`.
"""
function aff_factor_solve!(plu::AffineLU, J::SparseMatrixCSC{Float64,Int}, b::Vector{Float64})
    if plu.F === nothing || plu.pattern_nnz != nnz(J)
        try
            plu.F = lu(J); plu.pattern_nnz = nnz(J)
        catch
            return (zeros(length(b)), false)
        end
    else
        try
            lu!(plu.F, J)
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

# ── Weighted (block, iterate-derived) norm, §9.5.3 step 1 ────────────────

"""
    compute_weights(u, blocks, afrac) -> w

Per-DOF weight vector rebuilt from the current iterate: w_j = |u_j| +
afrac * rms(|u| over j's DOF block). Falls back to unit weight for any
block whose rms is exactly zero (the Stage-5 anchor 0/0 case, §9.5.3).
"""
function compute_weights(u::Vector{Float64}, blocks::Vector{UnitRange{Int}}, afrac::Float64)
    w = similar(u)
    for blk in blocks
        rms_b = sqrt(sum(abs2, @view u[blk]) / length(blk))
        if rms_b == 0.0
            @inbounds for j in blk; w[j] = 1.0; end
        else
            @inbounds for j in blk; w[j] = abs(u[j]) + afrac * rms_b; end
        end
    end
    return w
end

@inline function wnorm(d::Vector{Float64}, w::Vector{Float64})
    s = 0.0
    @inbounds for j in eachindex(d)
        s += (d[j] / w[j])^2
    end
    return sqrt(s / length(d))
end

# ── Affine-covariant damped Newton kernel, §9.5.3 ─────────────────────────

struct NewtonInfo
    converged::Bool
    iterations::Int
    residual::Float64      # ||F||_inf at exit (reported, never used for damping/termination)
    err_est::Float64
    alpha_final::Float64
    n_res::Int
    n_bsub::Int
    reason::Symbol
end

"""
    newton_affine!(u, res!; blocks, plu, rtol, max_iter, alpha_init, afrac,
                   max_log_step, n_log_dofs) -> NewtonInfo

Affine-covariant damped Newton (error-oriented). Mutates `u` in place.
`res!(F, u)` is the residual closure (type-generic for ForwardDiff). No
residual norm is ever computed inside this function except for the
returned diagnostic value.
"""
function newton_affine!(u::Vector{Float64}, res!::Function;
                         blocks::Vector{UnitRange{Int}},
                         plu::AffineLU,
                         rtol::Float64=RTOL_NEWTON,
                         max_iter::Int=40,
                         alpha_init::Float64=1.0,
                         afrac::Float64=AFRAC,
                         max_log_step::Float64=MAX_LOG_STEP,
                         n_log_dofs::Int=0,
                         alpha_min::Float64=ALPHA_MIN,
                         max_damp::Int=30,
                         final_correction::Bool=true)
    N = length(u)
    F = zeros(N)
    res!(F, u)
    n_res = 1

    cfg = ForwardDiff.JacobianConfig(res!, F, u, ForwardDiff.Chunk{min(12, N)}())
    Jdense = zeros(N, N)
    F0 = zeros(N)

    u_tr = similar(u)
    F_tr = similar(F)
    dubar = zeros(N)
    scr = zeros(N)

    alpha = alpha_init
    du_prev_n = NaN
    dubar_prev = zeros(N)
    dubar_prev_n = NaN
    alpha_prev = alpha_init

    n_bsub = 0
    converged = false
    reason = :max_iter
    it = 0
    dbn = Inf

    for outer_it in 1:max_iter
        it = outer_it
        w = compute_weights(u, blocks, afrac)

        # Exact AD Jacobian, §9.3.2
        ForwardDiff.jacobian!(Jdense, res!, F0, u, cfg)
        n_res += 1
        Jsp = sparse(Jdense)
        du, ok = aff_factor_solve!(plu, Jsp, -F)
        if !ok
            reason = :factor_failed
            break
        end
        if n_log_dofs > 0
            @inbounds for j in 1:n_log_dofs
                du[j] = clamp(du[j], -max_log_step, max_log_step)
            end
        end
        du_n = wnorm(du, w)

        # a-priori damping estimate from the previous iteration
        if it > 1 && isfinite(du_prev_n) && isfinite(dubar_prev_n)
            @inbounds for j in 1:N; scr[j] = dubar_prev[j] - du[j]; end
            den = wnorm(scr, w) * du_n
            if den > 0
                alpha = min(1.0, (du_prev_n * dubar_prev_n) / den * alpha_prev)
            end
        end
        alpha = clamp(alpha, alpha_min, 1.0)

        accepted = false
        for _ in 1:max_damp
            @inbounds for j in 1:N; u_tr[j] = u[j] + alpha * du[j]; end
            res!(F_tr, u_tr); n_res += 1
            if all(isfinite, F_tr)
                dubar = aff_resolve(plu, -F_tr); n_bsub += 1
                if all(isfinite, dubar)
                    dbn = wnorm(dubar, w)
                    if dbn <= (1.0 - alpha/2) * du_n
                        u .= u_tr; F .= F_tr
                        accepted = true
                        break
                    end
                end
            end
            alpha *= 0.5
            if alpha < alpha_min
                break
            end
        end

        if !accepted
            reason = :damping_stalled
            break
        end

        du_prev_n = du_n
        dubar_prev .= dubar
        dubar_prev_n = dbn
        alpha_prev = alpha

        if dbn <= rtol
            if final_correction
                u .+= dubar
                res!(F, u); n_res += 1
            end
            converged = true
            reason = :converged
            break
        end
    end

    return NewtonInfo(converged, it, maximum(abs, F), dbn, alpha, n_res, n_bsub, reason)
end

# ── Tangent continuation predictor, §9.5.4 ────────────────────────────────

"""
    tangent_predictor!(dudV, plu, res_at_V!, u, V; h=1e-6) -> ok

Fill `dudV` with du/dV = -J^-1 (dF/dV) via central difference + one
back-substitution against the live factorization.
"""
function tangent_predictor!(dudV::Vector{Float64}, plu::AffineLU, res_at_V!::Function,
                             u::Vector{Float64}, V::Float64; h::Float64=1e-6)
    N = length(u)
    Fp = zeros(N); Fm = zeros(N)
    res_at_V!(Fp, u, V + h)
    res_at_V!(Fm, u, V - h)
    @. Fp = (Fp - Fm) / (2h)
    d = aff_resolve(plu, -Fp)
    if !all(isfinite, d)
        return false
    end
    dudV .= d
    return true
end
