# driver.jl — the §9.5.5 unified continuation driver: anchor -> tangent
# predictor sweep -> §9.5.6 post-solve assertion -> cache. Shared by
# Stages 1-4 (and the Fit polish reuses newton_affine! directly). Stage 5
# has its own driver in run_stage5.jl because of the V_c=0 pre-ramp
# (§9.5.5 table) but calls the same newton_affine! / tangent_predictor!
# primitives from newton_affine.jl.

using Printf

struct SweepResult
    voltages::Vector{Float64}      # V_RHE, in the order solved (= V_grid order)
    solutions::Vector{Vector{Float64}}
    converged::Vector{Bool}
    iters::Vector{Int}
    resid::Vector{Float64}
end

"""
    post_solve_assert!(u, res_at_V!, V; F_inf_tol=F_INF_TOL, blocks, plu, n_log_dofs)

§9.5.6 — enforce ||F||_inf <= F_inf_tol as a post-solve assertion, separate
from the rtol termination test. One extra tight polish if it fails.
"""
function post_solve_assert!(u::Vector{Float64}, res_at_V!::Function, V::Float64;
                             blocks::Vector{UnitRange{Int}}, plu::AffineLU,
                             n_log_dofs::Int=0, F_inf_tol::Float64=F_INF_TOL)
    N = length(u)
    Fchk = zeros(N)
    res_at_V!(Fchk, u, V)
    if maximum(abs, Fchk) > F_inf_tol
        res! = (F_, u_) -> res_at_V!(F_, u_, V)
        info = newton_affine!(u, res!; blocks=blocks, plu=plu, rtol=1e-10, max_iter=3,
                               alpha_init=1.0, n_log_dofs=n_log_dofs)
        res_at_V!(Fchk, u, V)
        if maximum(abs, Fchk) > F_inf_tol
            @warn "post-solve residual assertion failed at V=$V: ||F||_inf = $(maximum(abs,Fchk))"
            return false
        end
    end
    return true
end

"""
    run_stage_sweep!(res_at_V!, V_grid, u0; blocks, n_log_dofs, save_fn, stage_name)

Generic §9.5.5 driver for the self-anchoring stages (1-4): anchor solve at
`V_grid[1]` (the mildest grid point) with `alpha_init = ALPHA_INIT_ANCHOR`,
then walk the remaining grid points with the tangent predictor and
`alpha_init = ALPHA_INIT_SWEEP`. `res_at_V!(F, u, V_RHE)` performs the
RHE->SHE conversion internally (§15.3). Caches every converged solution
immediately (§10.4) via `save_fn(u, V_RHE)`.
"""
function run_stage_sweep!(res_at_V!::Function, V_grid::Vector{Float64}, u0::Vector{Float64};
                           blocks::Vector{UnitRange{Int}}, n_log_dofs::Int,
                           save_fn::Function, stage_name::String="stage")
    N = length(u0)
    plu = AffineLU()
    u = copy(u0)

    V0 = V_grid[1]
    res0! = (F_, u_) -> res_at_V!(F_, u_, V0)
    @printf("[%s] anchor solve at V_RHE = %.4f ...\n", stage_name, V0)
    info = newton_affine!(u, res0!; blocks=blocks, plu=plu, rtol=RTOL_NEWTON,
                           max_iter=MAX_ITER_ANCHOR, alpha_init=ALPHA_INIT_ANCHOR,
                           n_log_dofs=n_log_dofs)
    @printf("[%s] anchor: converged=%s iters=%d ||F||_inf=%.3e reason=%s\n",
            stage_name, info.converged, info.iterations, info.residual, info.reason)
    if !info.converged
        error("[$stage_name] anchor failed to converge — fix the anchor guess / residual before sweeping")
    end
    post_solve_assert!(u, res_at_V!, V0; blocks=blocks, plu=plu, n_log_dofs=n_log_dofs)
    save_fn(u, V0)

    voltages = Float64[V0]; solutions = Vector{Vector{Float64}}([copy(u)])
    converged = Bool[true]; iters = Int[info.iterations]; resid = Float64[info.residual]

    dudV = zeros(N)
    have_tangent = tangent_predictor!(dudV, plu, res_at_V!, u, V0)

    for k in 2:length(V_grid)
        Vprev = V_grid[k-1]; Vcur = V_grid[k]
        dV = Vcur - Vprev
        if have_tangent
            u .+= dV .* dudV
        else
            # φ_s-shift fallback (Stages 1-4 have no explicit φ_s DOF block;
            # here the fallback is simply "carry the previous state forward",
            # since V enters only through the dense kinetics term, not a
            # single ghost-cell DOF — §9.5.4).
        end
        resV! = (F_, u_) -> res_at_V!(F_, u_, Vcur)
        info = newton_affine!(u, resV!; blocks=blocks, plu=plu, rtol=RTOL_NEWTON,
                               max_iter=MAX_ITER_SWEEP, alpha_init=ALPHA_INIT_SWEEP,
                               n_log_dofs=n_log_dofs)
        if !info.converged
            # sub-step recovery: halve dV, solve two half-steps at alpha_init=1e-2
            @printf("[%s] step FAILED at V=%.4f — sub-stepping\n", stage_name, Vcur)
            u_back = solutions[end]
            Vmid = Vprev + dV/2
            u_try = copy(u_back)
            if have_tangent; u_try .+= (dV/2) .* dudV; end
            resM! = (F_, u_) -> res_at_V!(F_, u_, Vmid)
            info_mid = newton_affine!(u_try, resM!; blocks=blocks, plu=plu, rtol=RTOL_NEWTON,
                                       max_iter=MAX_ITER_SWEEP, alpha_init=ALPHA_INIT_ANCHOR,
                                       n_log_dofs=n_log_dofs)
            if info_mid.converged
                u = u_try
                resV2! = (F_, u_) -> res_at_V!(F_, u_, Vcur)
                info = newton_affine!(u, resV2!; blocks=blocks, plu=plu, rtol=RTOL_NEWTON,
                                       max_iter=MAX_ITER_SWEEP, alpha_init=ALPHA_INIT_ANCHOR,
                                       n_log_dofs=n_log_dofs)
            end
        end
        ok = info.converged
        if ok
            post_solve_assert!(u, res_at_V!, Vcur; blocks=blocks, plu=plu, n_log_dofs=n_log_dofs)
            save_fn(u, Vcur)
            have_tangent = tangent_predictor!(dudV, plu, res_at_V!, u, Vcur)
        end
        push!(voltages, Vcur); push!(solutions, copy(u))
        push!(converged, ok); push!(iters, info.iterations); push!(resid, info.residual)
        if k % 10 == 0 || k == length(V_grid)
            @printf("[%s] step %d/%d: V=%.4f converged=%s iters=%d ||F||_inf=%.2e\n",
                    stage_name, k, length(V_grid), Vcur, ok, info.iterations, info.residual)
        end
    end
    return SweepResult(voltages, solutions, converged, iters, resid)
end

"""
    run_stage5_sweep!(res_at_V!, V_SHE_grid, u0_anchor; blocks, n_log_dofs,
                       save_fn, phi_s_shift!, stage_name)

Stage 5 driver (§9.5.5): anchor at V_c = 0 (V_applied_SHE = 0, i.e. zero
overpotential — the trivial, unique solution), pre-ramp in ~-0.025 V (SHE)
steps down to `V_SHE_grid[1]`, then continue through the remaining output
grid points at exactly -0.025 V steps. `res_at_V!(F, u, V_RHE)` keeps the
SAME uniform RHE-in convention as every other stage (§15.3 gotcha #9a) —
this driver works internally in the V_SHE frame and converts back to V_RHE
(`V_SHE + V_RHE_to_SHE_offset`) before every residual call, so a single
`residual_stage5!` implementation serves both the anchor/pre-ramp and the
output sweep. `phi_s_shift!(u, dV)` is the closed-form fallback predictor
(§9.5.4) used only if the tangent back-substitution is unavailable.
"""
function run_stage5_sweep!(res_at_V!::Function, V_SHE_grid_local::Vector{Float64}, u0_anchor::Vector{Float64};
                            blocks::Vector{UnitRange{Int}}, n_log_dofs::Int,
                            save_fn::Function, phi_s_shift!::Function, stage_name::String="Stage5")
    N = length(u0_anchor)
    plu = AffineLU()
    u = copy(u0_anchor)
    to_RHE(VSHE) = VSHE + V_RHE_to_SHE_offset
    res_she! = (F_, u_, VSHE_) -> res_at_V!(F_, u_, to_RHE(VSHE_))

    @printf("[%s] anchor solve at V_c = 0 (V_applied_SHE = 0) ...\n", stage_name)
    res0! = (F_, u_) -> res_she!(F_, u_, 0.0)
    info = newton_affine!(u, res0!; blocks=blocks, plu=plu, rtol=RTOL_NEWTON,
                           max_iter=MAX_ITER_ANCHOR, alpha_init=ALPHA_INIT_ANCHOR, n_log_dofs=n_log_dofs)
    @printf("[%s] anchor: converged=%s iters=%d ||F||_inf=%.3e reason=%s\n",
            stage_name, info.converged, info.iterations, info.residual, info.reason)
    info.converged || error("[$stage_name] V_c=0 anchor failed to converge")

    dudV = zeros(N)
    have_tangent = tangent_predictor!(dudV, plu, res_she!, u, 0.0)

    V_target = V_SHE_grid_local[1]
    n_pre = max(1, ceil(Int, abs(V_target) / 0.025))
    V_cur = 0.0
    for kpre in 1:n_pre
        V_next = kpre == n_pre ? V_target : V_cur - 0.025
        dV = V_next - V_cur
        if have_tangent
            u .+= dV .* dudV
        else
            phi_s_shift!(u, dV)
        end
        resV! = (F_, u_) -> res_she!(F_, u_, V_next)
        info = newton_affine!(u, resV!; blocks=blocks, plu=plu, rtol=RTOL_NEWTON,
                               max_iter=MAX_ITER_SWEEP, alpha_init=ALPHA_INIT_SWEEP, n_log_dofs=n_log_dofs)
        if !info.converged
            info = newton_affine!(u, resV!; blocks=blocks, plu=plu, rtol=RTOL_NEWTON,
                                   max_iter=MAX_ITER_ANCHOR, alpha_init=ALPHA_INIT_ANCHOR, n_log_dofs=n_log_dofs)
        end
        info.converged || error("[$stage_name] pre-ramp failed at V_SHE=$V_next")
        V_cur = V_next
        have_tangent = tangent_predictor!(dudV, plu, res_she!, u, V_cur)
    end
    @printf("[%s] pre-ramp complete: V_SHE=%.5f (target %.5f) in %d steps\n", stage_name, V_cur, V_target, n_pre)

    post_solve_assert!(u, res_she!, V_cur; blocks=blocks, plu=plu, n_log_dofs=n_log_dofs)
    save_fn(u, to_RHE(V_cur))
    voltages = Float64[to_RHE(V_cur)]; solutions = Vector{Vector{Float64}}([copy(u)])
    converged = Bool[true]; iters = Int[info.iterations]; resid = Float64[info.residual]

    for k in 2:length(V_SHE_grid_local)
        Vprev = V_SHE_grid_local[k-1]; Vcur2 = V_SHE_grid_local[k]
        dV = Vcur2 - Vprev
        if have_tangent
            u .+= dV .* dudV
        else
            phi_s_shift!(u, dV)
        end
        resV! = (F_, u_) -> res_she!(F_, u_, Vcur2)
        info = newton_affine!(u, resV!; blocks=blocks, plu=plu, rtol=RTOL_NEWTON,
                               max_iter=MAX_ITER_SWEEP, alpha_init=ALPHA_INIT_SWEEP, n_log_dofs=n_log_dofs)
        ok = info.converged
        if !ok
            @printf("[%s] step FAILED at V_SHE=%.4f — sub-stepping\n", stage_name, Vcur2)
            u_back = solutions[end]
            Vmid = Vprev + dV/2
            u_try = copy(u_back)
            if have_tangent; u_try .+= (dV/2) .* dudV; else; phi_s_shift!(u_try, dV/2); end
            resM! = (F_, u_) -> res_she!(F_, u_, Vmid)
            info_mid = newton_affine!(u_try, resM!; blocks=blocks, plu=plu, rtol=RTOL_NEWTON,
                                       max_iter=MAX_ITER_SWEEP, alpha_init=ALPHA_INIT_ANCHOR, n_log_dofs=n_log_dofs)
            if info_mid.converged
                u = u_try
                resV2! = (F_, u_) -> res_she!(F_, u_, Vcur2)
                info = newton_affine!(u, resV2!; blocks=blocks, plu=plu, rtol=RTOL_NEWTON,
                                       max_iter=MAX_ITER_SWEEP, alpha_init=ALPHA_INIT_ANCHOR, n_log_dofs=n_log_dofs)
            end
            ok = info.converged
        end
        if ok
            post_solve_assert!(u, res_she!, Vcur2; blocks=blocks, plu=plu, n_log_dofs=n_log_dofs)
            save_fn(u, to_RHE(Vcur2))
            have_tangent = tangent_predictor!(dudV, plu, res_she!, u, Vcur2)
        end
        push!(voltages, to_RHE(Vcur2)); push!(solutions, copy(u))
        push!(converged, ok); push!(iters, info.iterations); push!(resid, info.residual)
        if k % 10 == 0 || k == length(V_SHE_grid_local)
            @printf("[%s] step %d/%d: V_SHE=%.4f converged=%s iters=%d ||F||_inf=%.2e\n",
                    stage_name, k, length(V_SHE_grid_local), Vcur2, ok, info.iterations, info.residual)
        end
    end
    return SweepResult(voltages, solutions, converged, iters, resid)
end
