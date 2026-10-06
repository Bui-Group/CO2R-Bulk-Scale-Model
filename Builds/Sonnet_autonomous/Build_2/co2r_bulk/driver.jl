# driver.jl — the §9.5.5 unified continuation driver (anchor -> tangent-predictor sweep ->
# §9.5.6 assertion -> §10 cache), shared by every stage, plus the §10.3 binary cache and the
# §9.6/anchor-guess builders. Written once, per the guide's own instruction not to duplicate it
# five times.

using Printf

# ── §10.3 binary cache ────────────────────────────────────────────────────

function save_solution(cache_dir::String, prefix::String, V::Float64, u::Vector{Float64})
    mkpath(cache_dir)
    fn = joinpath(cache_dir, @sprintf("%s_V%.6f.bin", prefix, V))
    open(fn, "w") do f
        write(f, Int64(length(u)))
        write(f, u)
    end
end

function load_solution(cache_dir::String, prefix::String, V::Float64)
    fn = joinpath(cache_dir, @sprintf("%s_V%.6f.bin", prefix, V))
    isfile(fn) || return nothing
    return open(fn, "r") do f
        n = read(f, Int64)
        u = Vector{Float64}(undef, n)
        read!(f, u)
        u
    end
end

function list_cache(cache_dir::String, prefix::String)
    isdir(cache_dir) || return Float64[]
    tag = prefix * "_V"
    files = filter(f -> startswith(f, tag) && endswith(f, ".bin"), readdir(cache_dir))
    isempty(files) && return Float64[]
    return sort([parse(Float64, f[length(tag)+1:end-4]) for f in files])
end

function load_nearest(cache_dir::String, prefix::String, V_target::Float64)
    Vs = list_cache(cache_dir, prefix)
    isempty(Vs) && error("No cache found in $cache_dir with prefix $prefix")
    idx = argmin(abs.(Vs .- V_target))
    return load_solution(cache_dir, prefix, Vs[idx]), Vs[idx]
end

# ── §9.6 equilibrium + per-stage anchor-guess builders ───────────────────

function equilibrium_c_vec()
    eq = solve_buffer_equilibrium(1000.0, 1000.0)
    return [eq.H, eq.OH, eq.HCO3, eq.CO3, eq.CO2]
end

function make_u0_liquid_eq(N_CL::Int, c_eq_vec::Vector{Float64})
    u0 = zeros(5 * N_CL)
    lnc = log.(c_eq_vec)
    for ix in 1:N_CL
        u0[liquid_idx(ix,1):liquid_idx(ix,5)] .= lnc
    end
    return u0
end

function _fill_gas_inlet!(u0::Vector{Float64}, ly)
    for ig in 1:ly.N_GDL
        u0[gas_gdl_idx(ly, ig, 1)] = omega_inlet[1]
        u0[gas_gdl_idx(ly, ig, 2)] = omega_inlet[2]
        u0[gas_gdl_idx(ly, ig, 3)] = omega_inlet[3]
        u0[gas_gdl_idx(ly, ig, 4)] = p_total
    end
    for ix in 1:ly.N_CL
        u0[gas_cl_idx(ly, ix, 1)] = omega_inlet[1]
        u0[gas_cl_idx(ly, ix, 2)] = omega_inlet[2]
        u0[gas_cl_idx(ly, ix, 3)] = omega_inlet[3]
        u0[gas_cl_idx(ly, ix, 4)] = p_total
    end
    return u0
end

function make_u0_stage34(ly::Layout14, c_eq_vec::Vector{Float64})
    u0 = zeros(ly.N_TOT)
    lnc = log.(c_eq_vec)
    for ix in 1:ly.N_CL
        u0[liquid_idx(ix,1):liquid_idx(ix,5)] .= lnc
    end
    ly.has_gas && _fill_gas_inlet!(u0, ly)
    return u0
end

"V_c = 0 anchor (§9.5.5 table): liquid at bulk equilibrium, gas at inlet composition/p_total,
phi_s = phi_L = 0 everywhere (left as zeros by construction)."
function make_u0_stage5_anchor(ly::Stage5Layout, c_eq_vec::Vector{Float64})
    u0 = zeros(ly.N_TOT)
    lnc = log.(c_eq_vec)
    for ix in 1:ly.N_CL
        u0[liquid_idx(ix,1):liquid_idx(ix,5)] .= lnc
    end
    _fill_gas_inlet!(u0, ly)
    return u0
end

# ── phi_s-shift continuation-predictor fallback (§9.5.4) ─────────────────

apply_phi_s_shift_fallback!(u::Vector{Float64}, ly::Layout14, dV::Float64) = nothing  # no phi_s DOF; carry forward
function apply_phi_s_shift_fallback!(u::Vector{Float64}, ly::Stage5Layout, dV::Float64)
    for ig in 1:ly.N_GDL
        u[phi_s_gdl_idx5(ly, ig)] += dV
    end
    for ix in 1:ly.N_CL
        u[phi_s_cl_idx5(ly, ix)] += dV
    end
    return u
end

"""
    attempt_step!(u, res_at_V!, V_prev, V_cur, dV, plu, dudV, have_tangent, ly, n_log_dofs,
                  alpha_init, max_iter, rtol) -> converged::Bool

Advance `u` from the converged state at V_prev to a trial corrector solve at V_cur, using the
tangent predictor (or the phi_s-shift/carry-forward fallback). Rolls `u` back to its pre-call
state on failure so the caller can retry with sub-stepping.
"""
function attempt_step!(u::Vector{Float64}, res_at_V!::Function, V_prev::Float64, V_cur::Float64,
                        dV::Float64, plu::AffineLU, dudV::Vector{Float64}, have_tangent::Bool,
                        ly, n_log_dofs::Int, alpha_init::Float64, max_iter::Int, rtol::Float64)
    u_backup = copy(u)
    if have_tangent
        u .+= dV .* dudV
    else
        apply_phi_s_shift_fallback!(u, ly, dV)
    end
    res! = (F, uu) -> res_at_V!(F, uu, V_cur)
    info = newton_affine!(u, res!; ly=ly, plu=plu, rtol=rtol, max_iter=max_iter,
                           alpha_init=alpha_init, n_log_dofs=n_log_dofs)
    @printf("    V=%.4f: converged=%s iters=%d ||F||=%.2e err=%.2e (%s)\n",
            V_cur, info.converged, info.iterations, info.residual, info.err_est, info.reason)
    if !info.converged
        u .= u_backup
        return false
    end
    return true
end

"""
    run_continuation!(res_at_V!, u0, V0, V_grid, ly, n_log_dofs, cache_dir, prefix) -> (Vs, sols)

Unified continuation driver (§9.5.5) for Stages 1-4 (and reused per-voltage in the Fit step):
anchor Newton solve at V0 (alpha_init=1e-2, generous max_iter), then tangent-predictor
continuation through V_grid in the given order, §9.5.6 post-solve assertion + §10 cache at every
step, sub-step recovery (halve dV) on a failed corrector.
"""
function run_continuation!(res_at_V!::Function, u0::Vector{Float64}, V0::Float64,
                            V_grid::Vector{Float64}, ly, n_log_dofs::Int,
                            cache_dir::String, prefix::String)
    mkpath(cache_dir)
    plu = AffineLU()
    u = copy(u0)

    res0! = (F, uu) -> res_at_V!(F, uu, V0)
    info = newton_affine!(u, res0!; ly=ly, plu=plu, rtol=RTOL_PROD, max_iter=MAX_ITER_ANCHOR,
                           alpha_init=ALPHA_INIT_ANCHOR, n_log_dofs=n_log_dofs)
    @printf("  anchor V=%.4f: converged=%s iters=%d ||F||=%.2e err=%.2e (%s)\n",
            V0, info.converged, info.iterations, info.residual, info.err_est, info.reason)
    info.converged || error("Anchor solve failed to converge at V=$V0 ($(info.reason))")
    okA, Finf = post_solve_assert!(u, res_at_V!, V0, ly, plu; n_log_dofs=n_log_dofs)
    okA || error("Post-solve assertion failed at anchor V=$V0: ||F||=$Finf")

    Vs = Float64[]; sols = Vector{Vector{Float64}}()
    if V0 in V_grid
        save_solution(cache_dir, prefix, V0, u)
        push!(Vs, V0); push!(sols, copy(u))
    end

    dudV = zeros(length(u))
    have_tangent = tangent_predictor!(dudV, plu, res_at_V!, u, V0)

    V_prev = V0
    for V_cur in V_grid
        V_cur == V0 && continue
        dV = V_cur - V_prev
        ok = attempt_step!(u, res_at_V!, V_prev, V_cur, dV, plu, dudV, have_tangent,
                            ly, n_log_dofs, ALPHA_INIT_SWEEP, MAX_ITER_SWEEP, RTOL_PROD)
        if !ok
            V_mid = V_prev + dV / 2
            ok1 = attempt_step!(u, res_at_V!, V_prev, V_mid, dV/2, plu, dudV, have_tangent,
                                 ly, n_log_dofs, ALPHA_INIT_ANCHOR, MAX_ITER_ANCHOR, RTOL_PROD)
            ok1 || error("Sub-step recovery failed (first half) approaching V=$V_cur from $V_prev")
            have_tangent = tangent_predictor!(dudV, plu, res_at_V!, u, V_mid)
            ok2 = attempt_step!(u, res_at_V!, V_mid, V_cur, dV/2, plu, dudV, have_tangent,
                                 ly, n_log_dofs, ALPHA_INIT_ANCHOR, MAX_ITER_ANCHOR, RTOL_PROD)
            ok2 || error("Sub-step recovery failed (second half) at V=$V_cur")
        end
        okA, Finf = post_solve_assert!(u, res_at_V!, V_cur, ly, plu; n_log_dofs=n_log_dofs)
        okA || error("Post-solve assertion failed at V=$V_cur: ||F||=$Finf")

        save_solution(cache_dir, prefix, V_cur, u)
        push!(Vs, V_cur); push!(sols, copy(u))

        have_tangent = tangent_predictor!(dudV, plu, res_at_V!, u, V_cur)
        V_prev = V_cur
    end
    return Vs, sols
end

"""
    run_stage5_sweep!(res_at_V!, u0, ly, cache_dir, prefix) -> (Vs, sols)

Stage 5 driver (§9.5.5, §11 Stage 5): one continuous continuation chain from the V_RHE=0 anchor
(kinetics negligible there -- see BUILD_DIALOGUE.md for the "V_c" ambiguity note) through a
-0.025 V pre-ramp down to the first canonical grid point, then the 89-point canonical grid;
only the 89 canonical points are cached/returned. Never seeds from another stage's cache.
"""
function run_stage5_sweep!(res_at_V!::Function, u0::Vector{Float64}, ly::Stage5Layout,
                            cache_dir::String, prefix::String)
    mkpath(cache_dir)
    plu = AffineLU()
    u = copy(u0)
    n_log_dofs = 5 * ly.N_CL

    n_pre = Int(round(0.300 / DV_STEP)) - 1     # pre-ramp points strictly between 0 and -0.300
    V_pre_ramp = [-DV_STEP * k for k in 0:n_pre]     # 0.000, -0.025, ..., -0.275
    V_full_chain = vcat(V_pre_ramp, V_RHE_grid)      # continues into the exact canonical grid
    n_pre_pts = length(V_pre_ramp)

    V_anchor = V_full_chain[1]
    res0! = (F, uu) -> res_at_V!(F, uu, V_anchor)
    info = newton_affine!(u, res0!; ly=ly, plu=plu, rtol=RTOL_PROD, max_iter=MAX_ITER_ANCHOR,
                           alpha_init=ALPHA_INIT_ANCHOR, n_log_dofs=n_log_dofs)
    @printf("  [Stage5] anchor V_RHE=%.4f: converged=%s iters=%d ||F||=%.2e err=%.2e (%s)\n",
            V_anchor, info.converged, info.iterations, info.residual, info.err_est, info.reason)
    info.converged || error("Stage 5 anchor failed to converge at V_RHE=$V_anchor ($(info.reason))")
    okA, Finf = post_solve_assert!(u, res_at_V!, V_anchor, ly, plu; n_log_dofs=n_log_dofs)
    okA || error("Stage 5 anchor post-solve assertion failed: ||F||=$Finf")

    dudV = zeros(length(u))
    have_tangent = tangent_predictor!(dudV, plu, res_at_V!, u, V_anchor)

    Vs = Float64[]; sols = Vector{Vector{Float64}}()
    V_prev = V_anchor
    for k in 2:length(V_full_chain)
        V_cur = V_full_chain[k]
        dV = V_cur - V_prev
        ok = attempt_step!(u, res_at_V!, V_prev, V_cur, dV, plu, dudV, have_tangent,
                            ly, n_log_dofs, ALPHA_INIT_SWEEP, MAX_ITER_SWEEP, RTOL_PROD)
        if !ok
            V_mid = V_prev + dV / 2
            ok1 = attempt_step!(u, res_at_V!, V_prev, V_mid, dV/2, plu, dudV, have_tangent,
                                 ly, n_log_dofs, ALPHA_INIT_ANCHOR, MAX_ITER_ANCHOR, RTOL_PROD)
            ok1 || error("Stage5 sub-step recovery failed (first half) near V_RHE=$V_cur")
            have_tangent = tangent_predictor!(dudV, plu, res_at_V!, u, V_mid)
            ok2 = attempt_step!(u, res_at_V!, V_mid, V_cur, dV/2, plu, dudV, have_tangent,
                                 ly, n_log_dofs, ALPHA_INIT_ANCHOR, MAX_ITER_ANCHOR, RTOL_PROD)
            ok2 || error("Stage5 sub-step recovery failed (second half) at V_RHE=$V_cur")
        end
        okA, Finf = post_solve_assert!(u, res_at_V!, V_cur, ly, plu; n_log_dofs=n_log_dofs)
        okA || error("Stage5 post-solve assertion failed at V_RHE=$V_cur: ||F||=$Finf")

        if k > n_pre_pts
            save_solution(cache_dir, prefix, V_cur, u)
            push!(Vs, V_cur); push!(sols, copy(u))
        end
        have_tangent = tangent_predictor!(dudV, plu, res_at_V!, u, V_cur)
        V_prev = V_cur
    end
    return Vs, sols
end
