##############################################################################
# driver.jl — unified continuation driver, shared by all stages (§9.5.5)
##############################################################################
using Printf

const F_INF_TOL = 1e-6     # post-solve ||F||_inf assertion, §9.5.6

"phi_s-shift fallback predictor (Stage5Layout only; no-op for other layouts)."
apply_phi_s_shift!(u::Vector{Float64}, ly::AbstractLayout, dV::Float64) = u
function apply_phi_s_shift!(u::Vector{Float64}, ly::Stage5Layout, dV::Float64)
    for ig in 1:ly.N_GDL
        u[phi_s_gdl_idx(ly,ig)] += dV
    end
    for ix in 1:ly.N_CL
        u[phi_s_cl_idx(ly,ix)] += dV
    end
    return u
end

"""
    post_solve_assert!(u, V, res_at_V!, ly, plu) -> maxF

§9.5.6: enforce ||F||_inf <= F_INF_TOL as a post-solve assertion (separate
from the newton_affine! termination test). One extra tight polish if needed.
"""
function post_solve_assert!(u::Vector{Float64}, V::Float64, res_at_V!::Function,
                             ly::AbstractLayout, plu::AffineLU)
    N = length(u)
    Fchk = zeros(N)
    res_at_V!(Fchk, u, V)
    maxF = maximum(abs, Fchk)
    if maxF > F_INF_TOL
        res! = (F_,u_) -> res_at_V!(F_, u_, V)
        newton_affine!(u, res!; ly=ly, plu=plu, rtol=1e-10, max_iter=3, alpha_init=1.0,
                        n_log_dofs=n_log_dofs(ly))
        res_at_V!(Fchk, u, V)
        maxF = maximum(abs, Fchk)
        if maxF > F_INF_TOL
            error("residual assertion failed at V=$V: ||F||_inf=$maxF (halve dV and re-approach)")
        end
    end
    return maxF
end

"""
    step_one_voltage!(u, V_prev, V_target, res_at_V!, ly, plu; alpha_init, max_iter, rtol)

Tangent-predictor advance from V_prev to V_target, then newton_affine! corrector.
Returns the newton_affine! info NamedTuple.
"""
function step_one_voltage!(u::Vector{Float64}, V_prev::Float64, V_target::Float64,
                            res_at_V!::Function, ly::AbstractLayout, plu::AffineLU;
                            alpha_init::Float64=1.0, max_iter::Int=40, rtol::Float64=1e-8)
    N = length(u)
    dV = V_target - V_prev
    dudV = zeros(N)
    ok_tan = (dV == 0.0) ? false : tangent_predictor!(dudV, plu, res_at_V!, u, V_prev)
    if ok_tan
        u .+= dV .* dudV
    else
        apply_phi_s_shift!(u, ly, dV)
    end
    res! = (F_,u_) -> res_at_V!(F_, u_, V_target)
    info = newton_affine!(u, res!; ly=ly, plu=plu, rtol=rtol, max_iter=max_iter,
                           alpha_init=alpha_init, n_log_dofs=n_log_dofs(ly))
    return info
end

"""
    sweep_voltages!(u, V_start, V_list, res_at_V!, ly, plu;
                     cache_dir, prefix, rtol=1e-8, do_cache=true, do_assert=true,
                     verbose=true, tag="")

Walk V_list (in order) from V_start using the tangent-predictor continuation
driver (§9.5.5). On a failed step, halve dV and solve the two half-steps at
alpha_init=1e-2 (§9.5.5 recovery). Caches + asserts each point unless
do_cache/do_assert=false (used for the Stage-5 pre-anchor ramp, §11 Stage 5).
Returns (Vs::Vector{Float64}, converged_all::Bool).
"""
function sweep_voltages!(u::Vector{Float64}, V_start::Float64, V_list::Vector{Float64},
                          res_at_V!::Function, ly::AbstractLayout, plu::AffineLU;
                          cache_dir::String="", prefix::String="", rtol::Float64=1e-8,
                          do_cache::Bool=true, do_assert::Bool=true, verbose::Bool=true, tag::String="")
    V_prev = V_start
    Vs_done = Float64[]
    all_ok = true
    for V_target in V_list
        info = step_one_voltage!(u, V_prev, V_target, res_at_V!, ly, plu; alpha_init=1.0, rtol=rtol)
        if !info.converged
            verbose && @printf("%s  [recover] step FAILED at V=%.4f (reason=%s) -- halving dV\n", tag, V_target, info.reason)
            dV = V_target - V_prev
            Vmid = V_prev + dV/2
            info1 = step_one_voltage!(u, V_prev, Vmid, res_at_V!, ly, plu; alpha_init=1e-2, rtol=rtol)
            if !info1.converged
                all_ok = false
                error("$tag sub-step recovery failed at V=$Vmid (from $V_prev), reason=$(info1.reason)")
            end
            info2 = step_one_voltage!(u, Vmid, V_target, res_at_V!, ly, plu; alpha_init=1e-2, rtol=rtol)
            if !info2.converged
                all_ok = false
                error("$tag sub-step recovery failed at V=$V_target (from $Vmid), reason=$(info2.reason)")
            end
        end
        if do_assert
            post_solve_assert!(u, V_target, res_at_V!, ly, plu)
        end
        if do_cache
            save_solution(cache_dir, prefix, u, V_target)
        end
        push!(Vs_done, V_target)
        verbose && @printf("%s  V=%.4f  iters=%d  ||F||=%.2e  reason=%s\n",
                            tag, V_target, info.iterations, info.residual, info.reason)
        V_prev = V_target
    end
    return (Vs_done, all_ok)
end

"""
    anchor_solve!(u0, V_anchor, res_at_V!, ly, plu; alpha_init=1e-2, max_iter=200, rtol=1e-8)

Solve the stage's anchor point (§9.5.5). Returns (u, info).
"""
function anchor_solve!(u0::Vector{Float64}, V_anchor::Float64, res_at_V!::Function,
                        ly::AbstractLayout, plu::AffineLU; alpha_init::Float64=1e-2,
                        max_iter::Int=200, rtol::Float64=1e-8)
    res! = (F_,u_) -> res_at_V!(F_, u_, V_anchor)
    info = newton_affine!(u0, res!; ly=ly, plu=plu, rtol=rtol, max_iter=max_iter,
                           alpha_init=alpha_init, n_log_dofs=n_log_dofs(ly))
    return (u0, info)
end
