# newton_affine.jl — persistent sparse LU + affine-covariant damped Newton + tangent predictor.
# §9.5.2 - §9.5.4. This is the ONE solver stack used by every stage (§9.5.1).
#
# Deviation note (see BUILD_DIALOGUE.md item 6): the guide's kernel code (§9.5.3) elides the
# a-posteriori "propose a larger alpha and retry the same iteration" refinement with `...`. That
# refinement is a convergence-speed optimization on top of the a-priori Kantorovich-style estimate
# (which already carries information from the previous iteration forward), not a correctness
# requirement -- omitting it still yields a complete, convergent affine-covariant damped Newton
# (Deuflhard's ordinary monotonicity test). Implemented the full a-priori-estimate -> trial ->
# monotonicity-test -> halve-on-reject cycle exactly as given.

using ForwardDiff, SparseArrays, LinearAlgebra

const AD_CHUNK = 12

mutable struct AffineLU
    F::Any             # holds the live `lu(::SparseMatrixCSC{Float64,Int})` factorization
    pattern_nnz::Int
end
AffineLU() = AffineLU(nothing, -1)

"""
    aff_factor_solve!(plu, J, b) -> (x, ok)

Factorize `J` (reusing the symbolic analysis when the sparsity pattern is unchanged) and solve
J x = b. Leaves the factorization live in `plu` so `aff_resolve` can back-substitute against it.
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
aff_resolve(plu::AffineLU, b::Vector{Float64}) = plu.F \ b

"""
    affine_weights(u, blocks, afrac)

Per-DOF weight vector (§9.5.3 step 1): w_j = |u_j| + afrac * rms(|u| over j's block). Falls back
to a unit weight for any block whose rms is exactly zero (the Stage-5 anchor has phi_s == phi_L
== 0 identically on the first iteration, which would otherwise be 0/0).
"""
function affine_weights(u::AbstractVector{Float64}, blocks::Vector{<:AbstractUnitRange{Int}}, afrac::Float64)
    w = similar(u)
    for blk in blocks
        rms_b = sqrt(sum(abs2, view(u, blk)) / length(blk))
        if rms_b == 0.0
            @inbounds for j in blk
                w[j] = 1.0
            end
        else
            @inbounds for j in blk
                w[j] = abs(u[j]) + afrac * rms_b
            end
        end
    end
    return w
end

wnorm(d::AbstractVector{Float64}, w::AbstractVector{Float64}) =
    sqrt(sum(k -> (d[k] / w[k])^2, eachindex(d)) / length(d))

"""
    newton_affine!(u, res!; ly, plu, rtol=1e-8, max_iter=40, alpha_init=1.0, afrac=1e-3,
                   max_log_step=5.0, n_log_dofs=0, cfg=nothing, Jdense=nothing,
                   final_correction=true) -> NamedTuple

Affine-covariant damped Newton (error-oriented, §9.5.3). Mutates `u` in place. `ly` is any
layout object with a `dof_block_ranges(ly)` method (defined in assembly.jl) used to build the
per-DOF weights. `residual` in the return is ||F||_inf at exit (for the §9.5.6 assertion, never
used as a termination/damping test).
"""
function newton_affine!(u::Vector{Float64}, res!::Function;
                         ly,
                         plu::AffineLU,
                         rtol::Float64=RTOL_PROD,
                         max_iter::Int=40,
                         alpha_init::Float64=1.0,
                         afrac::Float64=AFRAC,
                         max_log_step::Float64=MAX_LOG_STEP,
                         n_log_dofs::Int=0,
                         alpha_min::Float64=ALPHA_MIN,
                         max_damp::Int=50,
                         cfg=nothing,
                         Jdense::Union{Nothing,Matrix{Float64}}=nothing,
                         final_correction::Bool=true)
    N = length(u)
    blocks = dof_block_ranges(ly)

    F = zeros(N)
    res!(F, u)
    n_res = 1

    Jd = Jdense === nothing ? zeros(N, N) : Jdense
    jcfg = cfg === nothing ? ForwardDiff.JacobianConfig(res!, F, u, ForwardDiff.Chunk{min(N, AD_CHUNK)}()) : cfg

    u_tr = similar(u)
    F_tr = similar(F)
    scr  = similar(u)

    alpha = alpha_init
    alpha_prev = alpha_init
    du_prev_n = NaN
    dubar_prev = zeros(N)
    dubar_n = NaN
    dubar = zeros(N)
    n_bsub = 0
    converged = false
    it = 0
    reason = "max_iter"

    for outer_it in 1:max_iter
        it = outer_it
        ForwardDiff.jacobian!(Jd, res!, F, u, jcfg)   # also refreshes F = res!(u) (real, not dual)
        n_res += 1
        Jsp = sparse(Jd)
        du, ok = aff_factor_solve!(plu, Jsp, -F)
        if !ok
            reason = "factorization_failed"
            break
        end
        if n_log_dofs > 0
            @inbounds for j in 1:n_log_dofs
                du[j] = clamp(du[j], -max_log_step, max_log_step)
            end
        end

        w = affine_weights(u, blocks, afrac)
        du_n = wnorm(du, w)

        if it > 1 && isfinite(du_prev_n) && isfinite(dubar_n) && du_prev_n > 0
            @inbounds for j in 1:N
                scr[j] = dubar_prev[j] - du[j]
            end
            den = wnorm(scr, w) * du_n
            if den > 0
                alpha = min(1.0, (du_prev_n * dubar_n) / den * alpha_prev)
            end
        end
        alpha = clamp(alpha, alpha_min, 1.0)

        accepted = false
        dbn = NaN
        while true
            @inbounds for j in 1:N
                u_tr[j] = u[j] + alpha * du[j]
            end
            res!(F_tr, u_tr)
            n_res += 1
            if all(isfinite, F_tr)
                dubar = aff_resolve(plu, -F_tr)
                n_bsub += 1
                if all(isfinite, dubar)
                    dbn = wnorm(dubar, w)
                    if dbn <= (1.0 - alpha / 2) * du_n
                        u .= u_tr
                        F .= F_tr
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
            reason = "damping_stalled"
            break
        end

        alpha_prev = alpha
        du_prev_n = du_n
        dubar_prev .= dubar
        dubar_n = dbn

        if dbn <= rtol
            converged = true
            reason = "converged"
            if final_correction
                u .+= dubar
                res!(F, u)
                n_res += 1
            end
            break
        end
    end

    theta = (isfinite(dubar_n) && isfinite(du_prev_n) && du_prev_n > 0) ? dubar_n / du_prev_n : NaN
    residual = maximum(abs, F)
    return (converged=converged, iterations=it, residual=residual, err_est=dubar_n,
            theta=theta, alpha_final=alpha, n_res=n_res, n_bsub=n_bsub, reason=reason)
end

"""
    tangent_predictor!(dudV, plu, res_at_V!, u, V; h=1e-6) -> ok

Fill `dudV` with du/dV = -J^{-1}(dF/dV) using the live factorization (§9.5.4). `res_at_V!(F,u,V)`
evaluates the residual at an arbitrary V. Cost: two residual evaluations + one back-substitution.
"""
function tangent_predictor!(dudV::Vector{Float64}, plu::AffineLU, res_at_V!::Function,
                             u::Vector{Float64}, V::Float64; h::Float64=1e-6)
    Fp = similar(u); Fm = similar(u)
    res_at_V!(Fp, u, V + h)
    res_at_V!(Fm, u, V - h)
    @. Fp = (Fp - Fm) / (2h)
    d = aff_resolve(plu, -Fp)
    all(isfinite, d) || return false
    dudV .= d
    return true
end

"""
    post_solve_assert!(u, res_at_V!, V, ly, plu; F_inf_tol=1e-6) -> (ok, Finf)

§9.5.6 post-solve residual assertion, applied outside the newton_affine! termination test. On
failure, attempts one extra tight Newton iteration from the converged state before re-checking.
"""
function post_solve_assert!(u::Vector{Float64}, res_at_V!::Function, V::Float64, ly, plu::AffineLU;
                             F_inf_tol::Float64=F_INF_TOL, n_log_dofs::Int=0)
    Fchk = similar(u)
    res_at_V!(Fchk, u, V)
    Finf = maximum(abs, Fchk)
    if Finf > F_inf_tol
        res! = (F_, u_) -> res_at_V!(F_, u_, V)
        newton_affine!(u, res!; ly=ly, plu=plu, rtol=1e-10, max_iter=3, alpha_init=1.0,
                        n_log_dofs=n_log_dofs)
        res_at_V!(Fchk, u, V)
        Finf = maximum(abs, Fchk)
    end
    return (Finf <= F_inf_tol, Finf)
end
