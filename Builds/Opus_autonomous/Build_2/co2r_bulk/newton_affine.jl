# newton_affine.jl — the §9.5 production solver stack
#   §9.5.2  persistent sparse LU workspace
#   §9.5.3  affine-covariant (error-oriented) damped Newton kernel
#   §9.5.4  tangent continuation predictor
#
# NO residual norm is ever taken inside the kernel.  Damping, step acceptance
# and termination all measure Newton CORRECTIONS in an iterate-derived norm.

using LinearAlgebra, SparseArrays, ForwardDiff, Printf

# ── §9.5.2 persistent sparse LU ───────────────────────────────────────────
mutable struct AffineLU
    F::Union{Nothing,SparseArrays.UMFPACK.UmfpackLU{Float64,Int}}
    colptr::Vector{Int}
    rowval::Vector{Int}
end
AffineLU() = AffineLU(nothing, Int[], Int[])

"""
    aff_factor_solve!(plu, J, b) -> (x, ok)

Factorize `J` (reusing the symbolic analysis when the sparsity pattern is
unchanged) and solve `J x = b`.  Leaves the factorization live in `plu` so
`aff_resolve` can back-substitute against it.
"""
function aff_factor_solve!(plu::AffineLU, J::SparseMatrixCSC{Float64,Int},
                           b::Vector{Float64})
    same = plu.F !== nothing && plu.colptr == J.colptr && plu.rowval == J.rowval
    if !same
        try
            plu.F = lu(J)
            plu.colptr = copy(J.colptr); plu.rowval = copy(J.rowval)
        catch
            return (zeros(length(b)), false)
        end
    else
        try
            lu!(plu.F, J)                       # refresh values, reuse symbolic
        catch
            try
                plu.F = lu(J)
                plu.colptr = copy(J.colptr); plu.rowval = copy(J.rowval)
            catch
                return (zeros(length(b)), false)
            end
        end
    end
    x = try
        plu.F \ b
    catch
        return (zeros(length(b)), false)
    end
    return (x, all(isfinite, x))
end

"Back-substitute against the live factorization (valid right after aff_factor_solve!)."
@inline aff_resolve(plu::AffineLU, b::Vector{Float64}) = plu.F \ b

# ── weighted correction norm (§9.5.3 step 1) ──────────────────────────────
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
*exactly* zero (Stage-5 anchor: φ_s = φ_L = 0) falls back to a unit weight.
"""
function build_weights!(w::Vector{Float64}, u::Vector{Float64},
                        blocks::Vector{UnitRange{Int}}, afrac::Float64)
    @inbounds for blk in blocks
        s = 0.0
        for j in blk; s += u[j]^2; end
        rms = sqrt(s / length(blk))
        fl = rms == 0.0 ? 1.0 : afrac * rms
        for j in blk; w[j] = abs(u[j]) + fl; end
    end
    return w
end

# ── Newton workspace ──────────────────────────────────────────────────────
mutable struct NewtonWS{RF,CF}
    N::Int
    J::Matrix{Float64}
    Fv::Vector{Float64}
    F_tr::Vector{Float64}
    Fy::Vector{Float64}
    du::Vector{Float64}
    dubar::Vector{Float64}
    dubar_prev::Vector{Float64}
    u_tr::Vector{Float64}
    w::Vector{Float64}
    scr::Vector{Float64}
    blocks::Vector{UnitRange{Int}}
    plu::AffineLU
    res!::RF
    cfg::CF
    n_res::Int
    n_bsub::Int
    n_jac::Int
end

function NewtonWS(N::Int, blocks::Vector{UnitRange{Int}}, res!, u0::Vector{Float64};
                  chunk::Int=12)
    Fy  = zeros(N)
    cfg = ForwardDiff.JacobianConfig(res!, Fy, u0, ForwardDiff.Chunk{chunk}())
    return NewtonWS(N, zeros(N, N), zeros(N), zeros(N), Fy, zeros(N), zeros(N),
                    zeros(N), zeros(N), ones(N), zeros(N), blocks, AffineLU(),
                    res!, cfg, 0, 0, 0)
end

@inline function _res!(ws::NewtonWS, F::Vector{Float64}, u::Vector{Float64})
    ws.res!(F, u); ws.n_res += 1; return F
end

"""
    newton_affine!(u, ws; rtol=1e-8, max_iter=40, alpha_init=1.0, afrac=1e-3,
                   max_log_step=5.0, n_log_dofs=0, alpha_min=1e-8,
                   final_correction=true)
        -> (converged, iterations, residual, err_est, theta, alpha_final,
            n_res, n_bsub, reason)

Affine-covariant damped Newton (error-oriented, Deuflhard NLEQ-ERR).  `rtol` is
the accuracy dial: stop when the estimated remaining solution error (weighted
norm of the simplified Newton correction) falls below it.  Mutates `u`.
`residual` is ‖F‖∞ at exit — reported for the §9.5.6 assertion, never used as a
termination or damping test.
"""
function newton_affine!(u::Vector{Float64}, ws::NewtonWS;
                        rtol::Float64=1e-8, max_iter::Int=40,
                        alpha_init::Float64=1.0, afrac::Float64=1e-3,
                        max_log_step::Float64=5.0, n_log_dofs::Int=0,
                        alpha_min::Float64=1e-8, max_damp::Int=60,
                        final_correction::Bool=true)
    N = ws.N
    F, F_tr, du, u_tr, w, scr = ws.Fv, ws.F_tr, ws.du, ws.u_tr, ws.w, ws.scr
    dubar_prev = ws.dubar_prev
    _res!(ws, F, u)
    if !all(isfinite, F)
        return (converged=false, iterations=0, residual=Inf, err_est=Inf,
                theta=Inf, alpha_final=0.0, n_res=ws.n_res, n_bsub=ws.n_bsub,
                reason=:nonfinite_initial_residual)
    end

    alpha       = alpha_init
    alpha_prev  = alpha_init
    du_prev_n   = NaN
    dubar_prev_n= NaN
    theta       = NaN
    err_est     = Inf

    for it in 1:max_iter
        build_weights!(w, u, ws.blocks, afrac)

        ForwardDiff.jacobian!(ws.J, ws.res!, ws.Fy, u, ws.cfg); ws.n_jac += 1
        if !all(isfinite, ws.J)
            return (converged=false, iterations=it, residual=maximum(abs, F),
                    err_est=err_est, theta=theta, alpha_final=alpha,
                    n_res=ws.n_res, n_bsub=ws.n_bsub, reason=:nonfinite_jacobian)
        end
        Js = sparse(ws.J)
        @inbounds for j in 1:N; scr[j] = -F[j]; end
        (dsol, ok) = aff_factor_solve!(ws.plu, Js, scr)
        if !ok
            return (converged=false, iterations=it, residual=maximum(abs, F),
                    err_est=err_est, theta=theta, alpha_final=alpha,
                    n_res=ws.n_res, n_bsub=ws.n_bsub, reason=:linear_solve_failed)
        end
        copyto!(du, dsol)
        # §9.4 item 4: clamp the liquid log-DOF block
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
                    n_res=ws.n_res, n_bsub=ws.n_bsub, reason=:nonfinite_step)
        end
        if du_n == 0.0
            return (converged=true, iterations=it, residual=maximum(abs, F),
                    err_est=0.0, theta=0.0, alpha_final=alpha,
                    n_res=ws.n_res, n_bsub=ws.n_bsub, reason=:zero_step)
        end

        # ── a-priori damping from the previous iteration ──────────────────
        if it > 1 && isfinite(du_prev_n) && isfinite(dubar_prev_n)
            @inbounds for j in 1:N; scr[j] = dubar_prev[j] - du[j]; end
            den = wnorm(scr, w) * du_n
            den > 0 && (alpha = min(1.0, (du_prev_n * dubar_prev_n) / den * alpha_prev))
        else
            alpha = alpha_init
        end
        alpha = clamp(alpha, alpha_min, 1.0)

        # ── trial + simplified correction + monotonicity test ─────────────
        accepted = false
        dbn = Inf
        retries = 0
        for _ in 1:max_damp
            @inbounds for j in 1:N; u_tr[j] = u[j] + alpha * du[j]; end
            _res!(ws, F_tr, u_tr)
            if all(isfinite, F_tr)
                @inbounds for j in 1:N; scr[j] = -F_tr[j]; end
                dbar = aff_resolve(ws.plu, scr); ws.n_bsub += 1
                if all(isfinite, dbar)
                    dbn = wnorm(dbar, w)
                    if dbn <= (1.0 - alpha/2) * du_n
                        # a-posteriori proposal; retry only if ≥ 4× larger
                        @inbounds for j in 1:N; scr[j] = dbar[j] - (1.0-alpha)*du[j]; end
                        dn2 = wnorm(scr, w)
                        a_post = dn2 > 0 ? min(1.0, 0.5 * du_n * alpha^2 / dn2) : 1.0
                        if a_post >= 4*alpha && alpha < 1.0 && retries < 3
                            alpha = min(1.0, a_post); retries += 1
                            continue
                        end
                        copyto!(u, u_tr); copyto!(F, F_tr)
                        copyto!(ws.dubar, dbar)
                        theta = dbn / du_n
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
                    n_res=ws.n_res, n_bsub=ws.n_bsub, reason=:damping_stalled)
        end

        err_est = dbn
        if dbn <= rtol
            if final_correction
                @inbounds for j in 1:N; u[j] += ws.dubar[j]; end
                _res!(ws, F, u)
            end
            return (converged=true, iterations=it, residual=maximum(abs, F),
                    err_est=err_est, theta=theta, alpha_final=alpha,
                    n_res=ws.n_res, n_bsub=ws.n_bsub, reason=:err_est_below_rtol)
        end

        du_prev_n    = du_n
        dubar_prev_n = dbn
        copyto!(dubar_prev, ws.dubar)
        alpha_prev   = alpha
    end
    return (converged=false, iterations=max_iter, residual=maximum(abs, F),
            err_est=err_est, theta=theta, alpha_final=alpha,
            n_res=ws.n_res, n_bsub=ws.n_bsub, reason=:max_iter)
end

# ── §9.5.4 tangent continuation predictor ─────────────────────────────────
"""
    tangent_predictor!(dudV, ws, res_at_V!, u, V; h=1e-6) -> ok

Fill `dudV` with du/dV = −J⁻¹ (∂F/∂V), using the live factorization left by the
last `newton_affine!` linear solve.  Cost: two residual evaluations (central
difference for ∂F/∂V) + one back-substitution.
"""
function tangent_predictor!(dudV::Vector{Float64}, ws::NewtonWS, res_at_V!,
                            u::Vector{Float64}, V::Float64; h::Float64=1e-6)
    ws.plu.F === nothing && return false
    Fp = ws.F_tr; Fm = ws.scr
    res_at_V!(Fp, u, V + h); ws.n_res += 1
    res_at_V!(Fm, u, V - h); ws.n_res += 1
    @inbounds for j in 1:ws.N; Fp[j] = -(Fp[j] - Fm[j]) / (2h); end
    d = try
        aff_resolve(ws.plu, Fp)
    catch
        return false
    end
    ws.n_bsub += 1
    all(isfinite, d) || return false
    copyto!(dudV, d)
    return true
end
