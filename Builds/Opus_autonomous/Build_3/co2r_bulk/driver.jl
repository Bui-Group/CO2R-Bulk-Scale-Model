# driver.jl — the §9.5.5 unified continuation driver, shared by ALL stages.
#   anchor → tangent-predictor sweep → §9.5.6 post-solve assertion → cache

const F_INF_TOL = 1e-6          # §9.5.7

"""
    run_continuation(u0, chain, emit, res_at_V!; ...)

Walk the voltage `chain` (V vs RHE@pH8) from `chain[1]` (the anchor) using the
tangent predictor of §9.5.4 and the affine-covariant kernel of §9.5.3.
`emit[k] == true` marks a voltage that is cached and reported.

`phi_s_range` is the φ_s DOF range used by the closed-form predictor fallback
(add dV to every φ_s DOF); pass `1:0` for stages without φ_s.

Returns `(Vs, Us, info_rows, plu)`.
"""
function run_continuation(u0::Vector{Float64}, chain::Vector{Float64},
                          emit::AbstractVector{Bool}, res_at_V!;
                          blocks::Vector{UnitRange{Int}},
                          n_log::Int,
                          cache_dir::String, prefix::String,
                          phi_s_range::UnitRange{Int} = 1:0,
                          rtol::Float64 = 1e-8,
                          anchor_alpha::Float64 = 1e-2,
                          sweep_alpha::Float64 = 1.0,
                          max_substep_depth::Int = 3,
                          verbose::Bool = true,
                          on_emit = nothing)
    N   = length(u0)
    u   = copy(u0)
    plu = AffineLU()
    Jd  = zeros(N, N)
    Vref = Ref(chain[1])
    res!  = (F_, u_) -> res_at_V!(F_, u_, Vref[])
    cfg   = ForwardDiff.JacobianConfig(res!, zeros(N), u, ForwardDiff.Chunk{12}())
    dudV  = zeros(N)
    Fchk  = zeros(N)
    have_tangent = false

    Vs = Float64[]; Us = Vector{Float64}[]
    rows = NamedTuple[]
    n_assert_fire = 0
    total_iters = 0; n_steps = 0

    "predictor + corrector for one step from V_from (current u) to V_to"
    function attempt(V_from::Float64, V_to::Float64, alpha0::Float64, maxit::Int)
        dV = V_to - V_from
        if have_tangent
            @inbounds for j in 1:N; u[j] += dV * dudV[j]; end
        elseif !isempty(phi_s_range)
            @inbounds for j in phi_s_range; u[j] += dV; end
        end
        Vref[] = V_to
        return newton_affine!(u, res!; blocks = blocks, plu = plu, rtol = rtol,
                              max_iter = maxit, alpha_init = alpha0,
                              n_log_dofs = n_log, Jdense = Jd, cfg = cfg)
    end

    function substep(u_back::Vector{Float64}, V_from::Float64, V_to::Float64,
                     depth::Int)
        depth > max_substep_depth && return nothing
        Vm = 0.5 * (V_from + V_to)
        for (Va, Vb) in ((V_from, Vm), (Vm, V_to))
            u_snap = copy(u)
            info = attempt(Va, Vb, 1e-2, 60)
            if !info.converged
                u .= u_snap
                info2 = substep(u_snap, Va, Vb, depth + 1)
                info2 === nothing && return nothing
                info = info2
            end
            have_tangent = tangent_predictor!(dudV, plu, res_at_V!, u, Vb)
        end
        Vref[] = V_to
        return newton_affine!(u, res!; blocks = blocks, plu = plu, rtol = rtol,
                              max_iter = 40, alpha_init = 1.0,
                              n_log_dofs = n_log, Jdense = Jd, cfg = cfg)
    end

    for (k, V) in enumerate(chain)
        if k == 1
            Vref[] = V
            info = newton_affine!(u, res!; blocks = blocks, plu = plu, rtol = rtol,
                                  max_iter = 200, alpha_init = anchor_alpha,
                                  n_log_dofs = n_log, Jdense = Jd, cfg = cfg)
            info.converged || error("ANCHOR FAILED at V_RHE = $V ($(info.reason))")
            verbose && @printf("anchor  V_RHE=%8.4f  iters=%3d  ||F||inf=%.3e  err=%.2e\n",
                               V, info.iterations, info.residual, info.err_est)
        else
            u_back = copy(u)
            info = attempt(chain[k-1], V, sweep_alpha, 40)
            if !info.converged
                verbose && @printf("  step %d at V_RHE=%.4f failed (%s) — sub-stepping\n",
                                   k, V, info.reason)
                u .= u_back
                have_tangent = tangent_predictor!(dudV, plu, res_at_V!, u, chain[k-1])
                info2 = substep(u_back, chain[k-1], V, 1)
                (info2 === nothing || !info2.converged) &&
                    error("STEP FAILED at V_RHE = $V even after sub-stepping")
                info = info2
            end
            total_iters += info.iterations; n_steps += 1
        end

        # §9.5.6 post-solve residual assertion
        res_at_V!(Fchk, u, V)
        if maximum(abs, Fchk) > F_INF_TOL
            n_assert_fire += 1
            Vref[] = V
            newton_affine!(u, res!; blocks = blocks, plu = plu, rtol = 1e-10,
                           max_iter = 3, alpha_init = 1.0, n_log_dofs = n_log,
                           Jdense = Jd, cfg = cfg)
            res_at_V!(Fchk, u, V)
            maximum(abs, Fchk) > F_INF_TOL &&
                error("residual assertion failed at V=$V (‖F‖∞ = $(maximum(abs,Fchk)))")
        end
        Finf = maximum(abs, Fchk)

        if emit[k]
            save_solution(cache_dir, prefix, u, V)
            push!(Vs, V); push!(Us, copy(u))
            push!(rows, (V = V, iters = info.iterations, Finf = Finf,
                         err = info.err_est, alpha = info.alpha_final))
            on_emit === nothing || on_emit(V, u, info, Finf)
            verbose && @printf("  V_RHE=%8.4f  iters=%2d  ||F||inf=%.3e\n",
                               V, info.iterations, Finf)
        end

        have_tangent = tangent_predictor!(dudV, plu, res_at_V!, u, V)
    end

    verbose && @printf("\nsweep done: %d emitted, mean iters/step = %.2f, assertion fired %d×, LU symbolic=%d numeric=%d\n",
                       length(Vs), n_steps > 0 ? total_iters/n_steps : 0.0,
                       n_assert_fire, plu.n_symbolic, plu.n_numeric)
    return Vs, Us, rows, plu
end
