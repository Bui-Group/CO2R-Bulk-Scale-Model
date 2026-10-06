# =====================================================================
#  newton_affine.jl — production solver stack (§9.5)
#
#    §9.5.2  AffineLU            persistent sparse LU (symbolic reuse,
#                                factorization kept live for back-substitution)
#    §9.5.3  newton_affine!      error-oriented affine-covariant damped Newton
#                                (Deuflhard NLEQ-ERR, Alg. 3.13)
#    §9.5.4  tangent_predictor!  du/dV from one extra back-substitution
#
#  NO residual norm is ever taken inside the kernel.  Damping, the
#  step-acceptance test and termination all measure Newton *corrections* in an
#  iterate-derived weighted norm.  Jacobian rows are never rescaled.
# =====================================================================

using LinearAlgebra
using SparseArrays
using ForwardDiff

# ── §9.5.2 Persistent sparse LU workspace ────────────────────────────
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
            lu!(plu.F, J)                     # refresh values, reuse symbolic
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

# ── §9.5.3 weights and weighted norm ─────────────────────────────────
"DOF blocks used for the weight floor; every layout carries them."
dof_blocks(ly) = ly.blocks

"""
    build_weights!(w, u, blocks, afrac)

`w_j = |u_j| + afrac · rms(|u| over j's DOF block)` (§9.5.3 step 1).
The block floor keeps a DOF passing through zero from blowing up the norm.
When an entire block is identically zero (φ_s / φ_L at the Stage-5 anchor,
before the first step) the block has no scale of its own and falls back to a
unit weight — the only case in which the weights are not iterate-derived.
"""
function build_weights!(w::Vector{Float64}, u::Vector{Float64},
                        blocks::Vector{UnitRange{Int}}, afrac::Float64)
    @inbounds for blk in blocks
        s = 0.0
        for j in blk; s += u[j] * u[j]; end
        rms = sqrt(s / length(blk))
        fl  = afrac * rms
        base = rms > 0.0 ? 0.0 : 1.0
        for j in blk; w[j] = abs(u[j]) + fl + base; end
    end
    @inbounds for j in eachindex(w)
        (isfinite(w[j]) && w[j] > 0.0) || (w[j] = 1.0)
    end
    return w
end

@inline function wnorm(d::Vector{Float64}, w::Vector{Float64})
    s = 0.0
    @inbounds for j in eachindex(d)
        t = d[j] / w[j]
        s += t * t
    end
    return sqrt(s / length(d))
end

# ── §9.5.3 affine-covariant damped Newton kernel ─────────────────────
"""
    newton_affine!(u, res!; ly, plu, rtol=1e-8, max_iter=40, alpha_init=1.0,
                   afrac=1e-3, max_log_step=5.0, n_log_dofs=0,
                   Jdense=nothing, cfg=nothing, final_correction=true)
        -> (converged, iterations, residual, err_est, theta, alpha_final,
            n_res, n_bsub, reason)

Affine-covariant damped Newton (error-oriented).  `rtol` is the accuracy dial:
stop when the estimated remaining solution error (weighted norm of the
simplified Newton correction) falls below it.  Mutates `u`.  `residual` in the
return is ‖F‖∞ at exit — reported for the §9.5.6 check, never used as a
termination or damping test.
"""
function newton_affine!(u::Vector{Float64}, res!;
                        ly, plu::AffineLU,
                        rtol::Float64 = RTOL_PROD,
                        max_iter::Int = MAXIT_SWEEP,
                        alpha_init::Float64 = 1.0,
                        alpha_min::Float64 = ALPHA_MIN,
                        afrac::Float64 = AFRAC,
                        max_log_step::Float64 = MAX_LOG_STEP,
                        n_log_dofs::Int = 0,
                        Jdense::Union{Nothing,Matrix{Float64}} = nothing,
                        cfg = nothing,
                        final_correction::Bool = true,
                        max_damp::Int = 40)

    N      = length(u)
    blocks = dof_blocks(ly)
    Jd     = Jdense === nothing ? zeros(N, N) : Jdense
    cf     = cfg === nothing ?
             ForwardDiff.JacobianConfig(res!, zeros(N), u, ForwardDiff.Chunk{12}()) : cfg

    F          = zeros(N);  F_tr  = zeros(N)
    du         = zeros(N);  dubar = zeros(N)
    dubar_prev = zeros(N);  scr   = zeros(N)
    u_tr       = zeros(N);  w     = ones(N)

    n_res = 0; n_bsub = 0
    du_prev_n = NaN; dubar_prev_n = NaN
    alpha = alpha_init; alpha_prev = alpha_init
    theta = NaN; err_est = NaN
    converged = false; reason = "max_iter"; it_done = 0

    for it in 1:max_iter
        it_done = it
        build_weights!(w, u, blocks, afrac)

        # ── ordinary Newton correction (AD Jacobian §9.3.2, persistent LU §9.5.2)
        ForwardDiff.jacobian!(Jd, res!, F, u, cf); n_res += 1
        if !all(isfinite, F) || !all(isfinite, Jd)
            reason = "nonfinite_jacobian"; break
        end
        J = sparse(Jd)
        @inbounds for j in 1:N; scr[j] = -F[j]; end
        x, ok = aff_factor_solve!(plu, J, scr)
        if !ok
            reason = "lu_failure"; break
        end
        du .= x

        # §9.4 item 4 — clamp the liquid log-DOF block only
        if n_log_dofs > 0
            m = 0.0
            @inbounds for j in 1:n_log_dofs; m = max(m, abs(du[j])); end
            if m > max_log_step
                s = max_log_step / m
                @inbounds for j in 1:n_log_dofs; du[j] *= s; end
            end
        end

        du_n = wnorm(du, w)
        if !(du_n > 0.0)
            converged = true; err_est = 0.0; theta = 0.0; reason = "zero_step"; break
        end

        # ── a-priori damping from the previous iteration
        if it > 1 && isfinite(du_prev_n) && isfinite(dubar_prev_n)
            @inbounds for j in 1:N; scr[j] = dubar_prev[j] - du[j]; end
            den = wnorm(scr, w) * du_n
            den > 0 && (alpha = min(1.0, (du_prev_n * dubar_prev_n) / den * alpha_prev))
        else
            alpha = alpha_init
        end
        alpha = clamp(alpha, alpha_min, 1.0)

        # ── trial + simplified correction + natural monotonicity test
        accepted = false; enlarged = 0; dubar_n = NaN
        for _ in 1:max_damp
            @inbounds for j in 1:N; u_tr[j] = u[j] + alpha * du[j]; end
            res!(F_tr, u_tr); n_res += 1
            if all(isfinite, F_tr)
                @inbounds for j in 1:N; scr[j] = -F_tr[j]; end
                dbv = aff_resolve(plu, scr); n_bsub += 1     # back-substitute only
                if all(isfinite, dbv)
                    dubar .= dbv
                    dbn = wnorm(dubar, w)
                    if dbn <= (1.0 - alpha / 2) * du_n       # natural monotonicity test
                        # a-posteriori α proposal — retry only if ≥ 4× larger
                        @inbounds for j in 1:N; scr[j] = dubar[j] - (1.0 - alpha) * du[j]; end
                        hpost  = wnorm(scr, w)
                        a_post = hpost > 0 ? min(1.0, 0.5 * alpha^2 * du_n / hpost) : 1.0
                        if enlarged < 3 && alpha < 1.0 && a_post >= 4 * alpha
                            alpha = clamp(a_post, alpha_min, 1.0); enlarged += 1
                            continue
                        end
                        u .= u_tr; F .= F_tr
                        theta = dbn / du_n; dubar_n = dbn
                        accepted = true; break
                    end
                end
            end
            alpha *= 0.5
            alpha < alpha_min && break
        end
        if !accepted
            reason = "damping_stalled"; break
        end

        # ── termination on the error estimate; final correction applied free
        err_est = dubar_n
        if dubar_n <= rtol
            final_correction && (u .+= dubar)
            converged = true; reason = "rtol"; break
        end

        du_prev_n = du_n; dubar_prev_n = dubar_n
        dubar_prev .= dubar; alpha_prev = alpha
    end

    res!(F, u); n_res += 1
    resid = all(isfinite, F) ? maximum(abs, F) : Inf
    return (converged = converged, iterations = it_done, residual = resid,
            err_est = err_est, theta = theta, alpha_final = alpha,
            n_res = n_res, n_bsub = n_bsub, reason = reason)
end

# ── §9.5.4 tangent continuation predictor ────────────────────────────
"""
    tangent_predictor!(dudV, plu, res_at_V!, u, V; h=1e-6, Fp, Fm) -> ok

Fill `dudV` with du/dV = −J⁻¹ (∂F/∂V), using the live factorization.
`res_at_V!(F, u, V)` evaluates the residual at an arbitrary V.
Cost: two residual evaluations (central difference for ∂F/∂V) + one
back-substitution.  Never refactorizes.
"""
function tangent_predictor!(dudV::Vector{Float64}, plu::AffineLU, res_at_V!,
                            u::Vector{Float64}, V::Float64;
                            h::Float64 = 1e-6,
                            Fp::Vector{Float64} = zeros(length(u)),
                            Fm::Vector{Float64} = zeros(length(u)))
    plu.F === nothing && return false
    res_at_V!(Fp, u, V + h)
    res_at_V!(Fm, u, V - h)
    @. Fp = (Fp - Fm) / (2h)                     # ∂F/∂V
    (all(isfinite, Fp)) || return false
    @. Fm = -Fp
    d = aff_resolve(plu, Fm)
    all(isfinite, d) || return false
    dudV .= d
    return true
end
