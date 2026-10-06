##############################################################################
# newton_affine.jl — persistent sparse LU + affine-covariant damped Newton
#                     + tangent-predictor continuation (§9.5.2 - §9.5.4)
##############################################################################
using ForwardDiff, SparseArrays, LinearAlgebra, Printf

# ── §9.5.2 Persistent sparse LU workspace ───────────────────────────────────
mutable struct AffineLU
    F::Any            # holds the live sparse LU factorization (SparseArrays.UMFPACK.UmfpackLU)
    pattern_nnz::Int
end
AffineLU() = AffineLU(nothing, -1)

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
aff_resolve(plu::AffineLU, b::Vector{Float64}) = plu.F \ b

# ── §9.5.3 Affine-covariant weighted norm ───────────────────────────────────
function block_rms(u::AbstractVector, rng)
    s = 0.0
    @inbounds for j in rng
        s += u[j]^2
    end
    return sqrt(s/length(rng))
end

"""
Per-DOF weights from the current iterate (§9.5.3 step 1):
w_j = |u_j| + afrac*rms(|u| over j's block). Falls back to unit weight for a
block whose rms is exactly zero (only happens at the Stage-5 anchor where
phi_s and phi_L start at exactly 0 everywhere).
"""
function compute_weights!(w::Vector{Float64}, u::Vector{Float64}, blks; afrac::Float64=1e-3)
    for rng in blks
        rms = block_rms(u, rng)
        if rms == 0.0
            @inbounds for j in rng; w[j] = 1.0; end
        else
            @inbounds for j in rng; w[j] = abs(u[j]) + afrac*rms; end
        end
    end
    return w
end

wnorm(d::AbstractVector, w::AbstractVector) = sqrt(sum((d[j]/w[j])^2 for j in eachindex(d)) / length(d))

# ── §9.5.3 Affine-covariant damped Newton kernel ────────────────────────────
"""
    newton_affine!(u, res!; ly, plu, rtol=1e-8, max_iter=40, alpha_init=1.0,
                    afrac=1e-3, max_log_step=5.0, n_log_dofs=0,
                    alpha_min=1e-8, max_damp=60)

Error-oriented affine-covariant damped Newton (Deuflhard NLEQ-ERR, §9.5.3).
Mutates `u` in place. Never takes a residual norm; damping/termination use
weighted correction norms only.
"""
function newton_affine!(u::Vector{Float64}, res!::Function;
                         ly::AbstractLayout, plu::AffineLU,
                         rtol::Float64=1e-8, max_iter::Int=40, alpha_init::Float64=1.0,
                         afrac::Float64=1e-3, max_log_step::Float64=5.0,
                         n_log_dofs::Int=0, alpha_min::Float64=1e-8, max_damp::Int=60)
    N = length(u)
    blks = dof_blocks(ly)
    F    = zeros(N); F_tr = zeros(N)
    du   = zeros(N); u_tr = zeros(N)
    w    = zeros(N); scr  = zeros(N)
    dubar_prev = zeros(N); dubar = zeros(N)

    res!(F, u)
    cfg = ForwardDiff.JacobianConfig(res!, F, u, ForwardDiff.Chunk{min(12, N)}())
    Jdense = zeros(N, N)
    n_res = 1; n_bsub = 0

    alpha = alpha_init
    du_prev_n = NaN
    dubar_prev_n = NaN
    converged = false
    reason = :max_iter
    it_done = 0
    du_n = NaN
    dbn = NaN

    for it in 1:max_iter
        it_done = it
        compute_weights!(w, u, blks; afrac=afrac)

        ForwardDiff.jacobian!(Jdense, res!, F, u, cfg)
        Jsp = sparse(Jdense)
        (raw_du, ok) = aff_factor_solve!(plu, Jsp, -F)
        if !ok
            reason = :factorization_failed
            break
        end
        du .= raw_du
        if n_log_dofs > 0
            @inbounds for j in 1:n_log_dofs
                du[j] = clamp(du[j], -max_log_step, max_log_step)
            end
        end
        du_n = wnorm(du, w)

        if it > 1 && isfinite(du_prev_n) && isfinite(dubar_prev_n)
            @inbounds for j in 1:N; scr[j] = dubar_prev[j] - du[j]; end
            den = wnorm(scr, w) * du_n
            if den > 0
                alpha = min(1.0, (du_prev_n*dubar_prev_n)/den * alpha)
            end
        end
        alpha = clamp(alpha, alpha_min, 1.0)

        accepted = false
        for _ in 1:max_damp
            @inbounds for j in 1:N; u_tr[j] = u[j] + alpha*du[j]; end
            res!(F_tr, u_tr); n_res += 1
            if all(isfinite, F_tr)
                dubar .= aff_resolve(plu, -F_tr); n_bsub += 1
                if all(isfinite, dubar)
                    dbn = wnorm(dubar, w)
                    if dbn <= (1.0 - alpha/2)*du_n
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
            reason = :damping_stalled
            break
        end

        du_prev_n = du_n
        dubar_prev .= dubar
        dubar_prev_n = dbn

        if dbn <= rtol
            u .+= dubar             # final correction applied for free (§9.5.3 step 6)
            converged = true
            reason = :converged
            break
        end
    end

    theta = (isnan(du_n) || du_n == 0.0) ? NaN : dbn/du_n
    return (converged=converged, iterations=it_done, residual=maximum(abs, F),
            err_est=dubar_prev_n, theta=theta, alpha_final=alpha,
            n_res=n_res, n_bsub=n_bsub, reason=reason)
end

# ── §9.5.4 Tangent continuation predictor ───────────────────────────────────
"""
    tangent_predictor!(dudV, plu, res_at_V!, u, V; h=1e-6) -> ok

Fills dudV with du/dV = -J^-1 (dF/dV) using the live factorization from the
last newton_affine! call. `res_at_V!(F,u,V)` evaluates the residual at
arbitrary V. Cost: two residual evaluations + one back-substitution.
"""
function tangent_predictor!(dudV::Vector{Float64}, plu::AffineLU, res_at_V!::Function,
                             u::Vector{Float64}, V::Float64; h::Float64=1e-6)
    N = length(u)
    if plu.F === nothing
        return false
    end
    Fp = zeros(N); Fm = zeros(N)
    res_at_V!(Fp, u, V+h)
    res_at_V!(Fm, u, V-h)
    @. Fp = (Fp - Fm)/(2h)
    ok_lu = true
    local d
    try
        d = aff_resolve(plu, -Fp)
    catch
        ok_lu = false
    end
    (!ok_lu || !all(isfinite, d)) && return false
    dudV .= d
    return true
end
