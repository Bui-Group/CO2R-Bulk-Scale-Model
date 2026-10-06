# =============================================================================
# solver.jl — production solver stack (Case_4_guide.md §9.5)
#   * AffineLU + aff_factor_solve!/aff_resolve  (§9.5.2) — persistent sparse LU
#     (Julia stdlib UMFPACK via lu/lu!; symbolic analysis reused, factorization kept live)
#   * AD sparse-Jacobian assembler with a fixed sparsity pattern (§9.3.2 / §9.3.3)
#   * newton_affine!                             (§9.5.3) — affine-covariant damped Newton,
#     error-estimate termination; no residual norm is ever taken inside the kernel
#   * tangent_predictor!                         (§9.5.4) — du/dV from one back-substitution
#   * continuation_sweep!                        (§9.5.5 driver + §9.5.6 post-solve assertion)
# Used by ALL stages (1–5 + Fit).
# =============================================================================

using ForwardDiff, SparseArrays, LinearAlgebra, Printf

# ---------------------------------------------------------------------------
# §9.5.2 Persistent sparse LU workspace
# First call: lu(J) (symbolic + numeric). Later calls with the same pattern:
# lu!(F, J) refreshes the numeric factorization and reuses the symbolic analysis.
# The factorization stays live so the kernel and the tangent predictor can
# back-substitute against it without refactorizing.
# ---------------------------------------------------------------------------
mutable struct AffineLU
    F::Union{Nothing,SparseArrays.UMFPACK.UmfpackLU{Float64,Int}}
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
            lu!(plu.F, J)                          # refresh values, reuse symbolic analysis
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

# ---------------------------------------------------------------------------
# Fixed sparsity pattern: block-tridiagonal with full B×B blocks, cell-major.
# Guaranteed superset of the FV Jacobian sparsity (diffusion couples adjacent
# cells; buffer+kinetics couple within a cell) → the pattern never changes, so
# the symbolic LU analysis is reused for the whole sweep.
# ---------------------------------------------------------------------------
function block_tridiag_pattern(n_cells::Int, B::Int)
    rows = Int[]; cols = Int[]
    sizehint!(rows, 3*n_cells*B*B); sizehint!(cols, 3*n_cells*B*B)
    for I in 1:n_cells
        for Jb in max(1, I-1):min(n_cells, I+1)
            for a in 1:B, b in 1:B
                push!(rows, B*(I-1) + a)
                push!(cols, B*(Jb-1) + b)
            end
        end
    end
    return rows, cols
end

# Build a template sparse matrix with the given (rows,cols) pattern (values = 0).
function pattern_matrix(rows::Vector{Int}, cols::Vector{Int}, N::Int)
    return sparse(rows, cols, zeros(length(rows)), N, N)
end

# Probe the structural sparsity of a residual by unioning the dense AD-Jacobian
# nonzeros at the base point AND a perturbed point (so state-dependent cross-terms
# that vanish at the base point — e.g. NP migration ∝ ∇φ_L when φ_L=0 — are captured).
# Used for Stage 5, whose hand-derived pattern is error-prone.
function probe_sparsity(res!, u0::Vector{Float64}; perturb::Union{Nothing,Vector{Float64}}=nothing)
    N = length(u0)
    Jd = zeros(N, N); Fb = zeros(N)
    cfg = ForwardDiff.JacobianConfig(res!, Fb, u0, ForwardDiff.Chunk{min(12, N)}())
    ForwardDiff.jacobian!(Jd, res!, Fb, u0, cfg)
    P = abs.(Jd) .> 0
    if perturb !== nothing
        ForwardDiff.jacobian!(Jd, res!, Fb, u0 .+ perturb, cfg)
        P .|= (abs.(Jd) .> 0)
    end
    @inbounds for i in 1:N; P[i,i] = true; end
    rows = Int[]; cols = Int[]
    @inbounds for j in 1:N, i in 1:N
        P[i,j] && (push!(rows, i); push!(cols, j))
    end
    return rows, cols
end

# AD sparse-Jacobian assembler (§9.3.2). Returns a closure assemble!(Jout, u) that
# fills Jout.nzval (fixed pattern) from a dense ForwardDiff Jacobian.
function make_ad_assembler(res!, N::Int; chunk::Int=12)
    Jd = zeros(N, N)
    Fbuf = zeros(N)
    C = min(chunk, N)
    cfg = ForwardDiff.JacobianConfig(res!, Fbuf, zeros(N), ForwardDiff.Chunk{C}())
    function assemble!(Jout::SparseMatrixCSC{Float64,Int}, u::Vector{Float64})
        ForwardDiff.jacobian!(Jd, res!, Fbuf, u, cfg)
        @inbounds for col in 1:N
            for k in Jout.colptr[col]:(Jout.colptr[col+1]-1)
                Jout.nzval[k] = Jd[Jout.rowval[k], col]
            end
        end
        return Jout
    end
    return assemble!
end

# ---------------------------------------------------------------------------
# §9.5.3 step 1 — iterate-derived weights and the weighted RMS norm
#   w_j = |u_j| + afrac · rms(|u| over j's DOF block).  A block that is identically
#   zero (φ_s / φ_L at the Stage-5 anchor before the first step) has no scale of
#   its own and falls back to a unit weight.
# ---------------------------------------------------------------------------
function build_weights!(w::Vector{Float64}, u::Vector{Float64},
                        blocks::Vector{UnitRange{Int}}, afrac::Float64)
    @inbounds for blk in blocks
        s = 0.0
        for j in blk; s += u[j] * u[j]; end
        rms  = sqrt(s / length(blk))
        fl   = afrac * rms
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

# ---------------------------------------------------------------------------
# §9.5.3 Affine-covariant damped Newton kernel (error-oriented)
#   `rtol` is the accuracy dial: stop when the weighted norm of the simplified
#   Newton correction (the estimate of the remaining solution error) falls below
#   it. Damping and the step-acceptance test compare correction norms only.
#   `residual` in the return is ‖F‖∞ at exit — reported for the §9.5.6 check,
#   never used as a termination or damping test.
# ---------------------------------------------------------------------------
function newton_affine!(u::Vector{Float64}, res!;
        blocks::Vector{UnitRange{Int}}, plu::AffineLU,
        jac_assembler, J::SparseMatrixCSC{Float64,Int},
        rtol::Float64=1e-8, max_iter::Int=40,
        alpha_init::Float64=1.0, alpha_min::Float64=1e-8,
        afrac::Float64=1e-3, max_log_step::Float64=5.0, n_log_dofs::Int=0,
        final_correction::Bool=true, max_damp::Int=40, verbose::Bool=false)

    N = length(u)
    F  = zeros(N); F_tr  = zeros(N)
    du = zeros(N); dubar = zeros(N); dubar_prev = zeros(N)
    scr = zeros(N); u_tr = zeros(N); w = ones(N)

    n_res = 0; n_bsub = 0
    du_prev_n = NaN; dubar_prev_n = NaN
    alpha = alpha_init; alpha_prev = alpha_init
    theta = NaN; err_est = NaN
    converged = false; reason = "max_iter"; it_done = 0

    for it in 1:max_iter
        it_done = it
        build_weights!(w, u, blocks, afrac)

        # ── ordinary Newton correction: AD Jacobian (§9.3.2) + persistent LU (§9.5.2)
        res!(F, u); n_res += 1
        all(isfinite, F) || (reason = "nonfinite_residual"; break)
        jac_assembler(J, u)
        all(isfinite, J.nzval) || (reason = "nonfinite_jacobian"; break)
        @inbounds for j in 1:N; scr[j] = -F[j]; end
        x, ok = aff_factor_solve!(plu, J, scr)
        ok || (reason = "lu_failure"; break)
        du .= x

        # §9.4 — clamp the liquid log-DOF block only
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
                dbv = aff_resolve(plu, scr); n_bsub += 1      # back-substitute only
                if all(isfinite, dbv)
                    dubar .= dbv
                    dbn = wnorm(dubar, w)
                    if dbn <= (1.0 - alpha / 2) * du_n        # natural monotonicity test
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

        verbose && @printf("    it %3d: ||F||inf=%.3e  err_est=%.3e  theta=%.2e  alpha=%.3e\n",
                           it, maximum(abs, F), dubar_n, theta, alpha)

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

# ---------------------------------------------------------------------------
# §9.5.4 Tangent continuation predictor
#   du/dV = −J⁻¹ (∂F/∂V) using the live factorization. `res_at_V!(F, u, V)`
#   evaluates the residual at an arbitrary V. Cost: two residual evaluations
#   (central difference for ∂F/∂V) + one back-substitution. Never refactorizes.
# ---------------------------------------------------------------------------
function tangent_predictor!(dudV::Vector{Float64}, plu::AffineLU, res_at_V!,
                            u::Vector{Float64}, V::Float64;
                            h::Float64=1e-6,
                            Fp::Vector{Float64}=zeros(length(u)),
                            Fm::Vector{Float64}=zeros(length(u)))
    plu.F === nothing && return false
    res_at_V!(Fp, u, V + h)
    res_at_V!(Fm, u, V - h)
    @. Fp = (Fp - Fm) / (2h)                       # ∂F/∂V
    all(isfinite, Fp) || return false
    @. Fm = -Fp
    d = aff_resolve(plu, Fm)
    all(isfinite, d) || return false
    dudV .= d
    return true
end

# ---------------------------------------------------------------------------
# §9.5.5 Unified continuation driver + §9.5.6 post-solve residual assertion
#
#   continuation_sweep!(u, chain, res_closure!, V_ref; kw...) -> Vector{NamedTuple}
#
#   `u` holds the anchor guess (at chain[1]) on entry and the last converged state
#   on exit. `res_closure!(F, u)` evaluates the residual at the voltage V_ref[].
#   anchor:   newton_affine! (alpha_init = anchor_alpha, max_iter = anchor_max_iter)
#   sweep:    u .+= dV .* du/dV  →  newton_affine! (alpha_init = 1, max_iter = sweep_max_iter)
#             →  §9.5.6 assertion  →  on_converged(V, u, info)  →  new tangent
#   recovery: on a failed step roll back, halve dV, solve the two half-steps with
#             alpha_init = 1e-2 (recursive, depth-limited).
#   `shift_fallback!(u, dV)` is applied when no tangent is available (Stage 5:
#   add dV to every φ_s DOF; Stages 1–4: carry the previous state forward).
# ---------------------------------------------------------------------------
function continuation_sweep!(u::Vector{Float64}, chain::Vector{Float64}, res_closure!, V_ref::Ref{Float64};
        blocks::Vector{UnitRange{Int}}, plu::AffineLU, jac_assembler, J::SparseMatrixCSC{Float64,Int},
        n_log_dofs::Int, rtol::Float64=1e-8, f_inf_tol::Float64=1e-6,
        anchor_alpha::Float64=1e-2, anchor_max_iter::Int=200, sweep_max_iter::Int=40,
        shift_fallback! = nothing, on_converged = (V, u, info) -> nothing,
        max_substep_depth::Int=4, progress_every::Int=10, label::String="V_RHE")

    N = length(u)
    dudV = zeros(N); Fp = zeros(N); Fm = zeros(N); Fchk = zeros(N)
    kw = (blocks=blocks, plu=plu, jac_assembler=jac_assembler, J=J, rtol=rtol, n_log_dofs=n_log_dofs)
    res_at_V!(F, uu, V) = (Vold = V_ref[]; V_ref[] = V; res_closure!(F, uu); V_ref[] = Vold; F)

    # solve at V from the current uu, then the §9.5.6 assertion
    function solve_at!(uu, V, alpha, maxit)
        V_ref[] = V
        info = newton_affine!(uu, res_closure!; kw..., alpha_init=alpha, max_iter=maxit)
        info.converged || return info, false
        res_closure!(Fchk, uu); r = maximum(abs, Fchk)
        if r > f_inf_tol
            extra = newton_affine!(uu, res_closure!; kw..., rtol=1e-10, max_iter=3, alpha_init=1.0)
            res_closure!(Fchk, uu); r = maximum(abs, Fchk)
            info = merge(info, (iterations = info.iterations + extra.iterations, residual = r,
                                reason = r <= f_inf_tol ? "rtol+assert" : "assert_fail"))
            r <= f_inf_tol || return info, false
        end
        return merge(info, (residual = r,)), true
    end

    # fresh tangent at a converged (uu, V): one factorization + one back-substitution
    function refresh_tangent!(uu, V)
        V_ref[] = V
        jac_assembler(J, uu)
        aff_factor_solve!(plu, J, zeros(N))
        return tangent_predictor!(dudV, plu, res_at_V!, uu, V; Fp=Fp, Fm=Fm)
    end

    # predictor + corrector from a converged u_from at V_from to V_to; sub-steps on failure
    function advance!(uu, u_from, V_from, V_to, tangent_ok, alpha, maxit, depth)
        dV = V_to - V_from
        uu .= u_from
        if tangent_ok
            @. uu += dV * dudV
        elseif shift_fallback! !== nothing
            shift_fallback!(uu, dV)
        end
        info, ok = solve_at!(uu, V_to, alpha, maxit)
        (ok || depth >= max_substep_depth) && return info, ok, depth
        # recovery: roll back, halve dV, two half-steps with cautious damping
        V_mid = 0.5 * (V_from + V_to)
        u_mid = copy(u_from)
        t_ok = refresh_tangent!(u_from, V_from)
        info_m, ok_m, _ = advance!(u_mid, u_from, V_from, V_mid, t_ok, 1e-2, anchor_max_iter, depth + 1)
        ok_m || return info_m, false, depth + 1
        t_ok = refresh_tangent!(u_mid, V_mid)
        info, ok, d = advance!(uu, u_mid, V_mid, V_to, t_ok, 1e-2, anchor_max_iter, depth + 1)
        return info, ok, d
    end

    log = NamedTuple[]
    # ── anchor
    info, ok = solve_at!(u, chain[1], anchor_alpha, anchor_max_iter)
    ok || error(@sprintf("anchor solve failed at %s=%.4f (reason=%s, ||F||inf=%.3e)",
                         label, chain[1], info.reason, info.residual))
    on_converged(chain[1], u, info)
    push!(log, (V=chain[1], converged=true, iterations=info.iterations, residual=info.residual,
                err_est=info.err_est, substeps=0))
    @printf("  anchor %s=%.4f: %d iters, ||F||inf=%.2e\n", label, chain[1], info.iterations, info.residual)
    tangent_ok = tangent_predictor!(dudV, plu, res_at_V!, u, chain[1]; Fp=Fp, Fm=Fm)

    # ── sweep
    u_prev = copy(u)
    for k in 2:length(chain)
        u_prev .= u
        info, ok, depth = advance!(u, u_prev, chain[k-1], chain[k], tangent_ok, 1.0, sweep_max_iter, 0)
        if ok
            on_converged(chain[k], u, info)
            tangent_ok = tangent_predictor!(dudV, plu, res_at_V!, u, chain[k]; Fp=Fp, Fm=Fm)
        else
            @printf("  [FAIL] %s=%.4f reason=%s ||F||inf=%.3e — carrying previous state forward\n",
                    label, chain[k], info.reason, info.residual)
            u .= u_prev; tangent_ok = false
        end
        push!(log, (V=chain[k], converged=ok, iterations=info.iterations, residual=info.residual,
                    err_est=info.err_est, substeps=depth))
        if k % progress_every == 0 || k == length(chain)
            @printf("  swept %3d/%d  (%s=%.3f, conv=%s, it=%d, ||F||inf=%.2e)\n",
                    k, length(chain), label, chain[k], ok, info.iterations, info.residual)
        end
    end
    return log
end

nothing
