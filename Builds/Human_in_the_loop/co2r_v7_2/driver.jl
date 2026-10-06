# =============================================================================
# driver.jl — the §9.5.5 unified continuation driver, shared by ALL stages.
# Written once: anchor -> tangent-predictor sweep -> §9.5.6 post-solve
# assertion -> cache.  Five per-stage copies would diverge (§13).
# =============================================================================
using Printf

"Holder so the residual closure (and therefore its ForwardDiff tag) is created once."
mutable struct VHolder
    V::Float64
end

# ── Anchor guesses (§9.5.5 table) ────────────────────────────────────────────
"""
    make_anchor_guess(ctx, V_cathode) -> u0

Stages 1-2: every cell at the exact bulk equilibrium (§9.6).
Stages 3-4: + gas at the inlet composition and `p_total` in every gas cell.
Stage 5   : + phi_s = V_applied_SHE on GDL+CL and phi_L = 0 on the CL.

`V_cathode` is V vs RHE@pH 8; the SHE conversion happens here, so at the
Stage-5 anchor `V_c = 0` (V_cathode = +0.51305 RHE) the phi_s block is
written as exactly 0 — matching §9.5.5 and §11 gotcha #9a.
"""
function make_anchor_guess(ctx::ModelCtx, V_cathode::Float64)
    ly = ctx.ly
    u0 = zeros(ly.N_TOT)
    lnc = log.(ctx.c_eq)
    for ix in 1:ly.N_CL, i in 1:5
        u0[liquid_idx(ix, i)] = lnc[i]
    end
    if ly.has_gas
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
    end
    if ly.has_pot
        V_applied_SHE = V_cathode - V_RHE_to_SHE_offset
        for ig in 1:ly.N_GDL; u0[phi_s_gdl_idx(ly, ig)] = V_applied_SHE; end
        for ix in 1:ly.N_CL;  u0[phi_s_cl_idx(ly, ix)]  = V_applied_SHE; end
        for ix in 1:ly.N_CL;  u0[phi_L_idx(ly, ix)]     = 0.0;           end
    end
    return u0
end

"§9.5.4 fallback predictor: the phi_s boundary value moves by exactly dV."
function phi_s_shift!(u::Vector{Float64}, ly::Layout, dV::Float64)
    ly.has_pot || return u
    for ig in 1:ly.N_GDL; u[phi_s_gdl_idx(ly, ig)] += dV; end
    for ix in 1:ly.N_CL;  u[phi_s_cl_idx(ly, ix)]  += dV; end
    return u
end

# ── The driver ───────────────────────────────────────────────────────────────
"""
    run_continuation(ctx, u_anchor, V_chain, emit; cache_dir, prefix, ...)

Anchor at `V_chain[1]`, then walk the chain with the tangent predictor and the
affine-covariant corrector.  Every emitted voltage passes the §9.5.6 post-solve
||F||_inf assertion and is cached before anything else happens with it.

`emit[k] == true` marks the chain points that are output voltages (for
Stages 1-4 every point is emitted; for Stage 5 the ~33-point pre-anchor ramp
from V_c = 0 is not).
"""
function run_continuation(ctx::ModelCtx, u_anchor::Vector{Float64},
                          V_chain::Vector{Float64}, emit::Vector{Bool};
                          cache_dir::String, prefix::String,
                          rtol::Float64 = RTOL_PROD,
                          f_inf_tol::Float64 = F_INF_TOL,
                          max_substep_depth::Int = 4,
                          record_predictor::Bool = false,
                          verbose::Bool = true)
    ly = ctx.ly; N = ly.N_TOT
    plu = AffineLU()
    vh  = VHolder(V_chain[1])
    res!      = (F_, u_)     -> full_residual!(F_, u_, vh.V, ctx)
    res_at_V! = (F_, u_, V_) -> full_residual!(F_, u_, V_, ctx)

    Jd  = zeros(N, N)
    cfg = ForwardDiff.JacobianConfig(res!, zeros(N), copy(u_anchor), ForwardDiff.Chunk{12}())

    u    = copy(u_anchor)
    dudV = zeros(N); Fp = zeros(N); Fm = zeros(N); Fchk = zeros(N)
    have_tangent = false

    voltages  = Float64[]; solutions = Vector{Vector{Float64}}()
    iters     = Int[];     resids    = Float64[]
    tot_iters = 0; tot_res = 0; tot_bsub = 0; n_substep = 0; n_assert_fire = 0
    # §9.5.9 step 4: predictor A/B — tangent vs carry-forward + phi_s shift
    pred_tan = Float64[]; pred_carry = Float64[]
    u_pred_t = zeros(N); u_pred_c = zeros(N); pending_ab = false
    t_start = time()

    "Corrector at one voltage from the current iterate; returns the info NamedTuple."
    function solve_at(V::Float64, a_init::Float64, mit::Int)
        vh.V = V
        return newton_affine!(u, res!; ly = ly, plu = plu, rtol = rtol,
                              max_iter = mit, alpha_init = a_init,
                              Jdense = Jd, cfg = cfg)
    end

    "Predictor from the converged state at V_from to V_to."
    function predict!(dV::Float64)
        if have_tangent
            @inbounds for j in 1:N; u[j] += dV * dudV[j]; end
        else
            phi_s_shift!(u, ly, dV)
        end
    end

    "§9.5.5 recovery: halve dV and take the two half-steps at alpha_init = 1e-2."
    function walk!(V_from::Float64, V_to::Float64, depth::Int)
        dV = V_to - V_from
        u_save = copy(u); tan_save = copy(dudV); have_save = have_tangent
        if record_predictor && have_tangent && depth == 0
            @inbounds for j in 1:N; u_pred_t[j] = u[j] + dV * dudV[j]; end
            copyto!(u_pred_c, u); phi_s_shift!(u_pred_c, ly, dV)
            pending_ab = true
        end
        predict!(dV)
        info = solve_at(V_to, depth == 0 ? ALPHA_SWEEP : ALPHA_ANCHOR, MAXIT_SWEEP)
        tot_iters += info.iterations; tot_res += info.n_res; tot_bsub += info.n_bsub
        if info.converged
            return info
        end
        if depth >= max_substep_depth
            return info
        end
        n_substep += 1
        verbose && @printf("    ! step to V=%.5f failed (%s) — halving dV (depth %d)\n",
                           V_to, info.reason, depth + 1)
        copyto!(u, u_save); copyto!(dudV, tan_save); have_tangent = have_save
        V_mid = 0.5 * (V_from + V_to)
        i1 = walk!(V_from, V_mid, depth + 1)
        i1.converged || return i1
        have_tangent = tangent_predictor!(dudV, plu, res_at_V!, u, V_mid; Fp = Fp, Fm = Fm)
        return walk!(V_mid, V_to, depth + 1)
    end

    "§9.5.6 post-solve residual assertion — outside the termination test."
    function assert_residual(V::Float64)
        full_residual!(Fchk, u, V, ctx)
        r = maximum(abs, Fchk)
        if r > f_inf_tol
            verbose && @printf("    . post-solve ||F||inf=%.3e > %.1e — polishing\n", r, f_inf_tol)
            vh.V = V
            n_assert_fire += 1
            newton_affine!(u, res!; ly = ly, plu = plu, rtol = 1e-10, max_iter = 3,
                           alpha_init = 1.0, Jdense = Jd, cfg = cfg)
            full_residual!(Fchk, u, V, ctx)
            r = maximum(abs, Fchk)
            r > f_inf_tol && error(@sprintf("residual assertion failed at V=%.5f (||F||inf=%.3e) — halve dV and re-approach", V, r))
        end
        return r
    end

    for (k, V) in enumerate(V_chain)
        if k == 1
            info = solve_at(V, ALPHA_ANCHOR, MAXIT_ANCHOR)
            tot_iters += info.iterations; tot_res += info.n_res; tot_bsub += info.n_bsub
            info.converged || error("Anchor solve failed at V=$V — reason: $(info.reason)")
            verbose && @printf("anchor  V=%8.5f  iters=%3d  ||F||inf=%.3e  err=%.2e  (%s)\n",
                               V, info.iterations, info.residual, info.err_est, info.reason)
        else
            info = walk!(V_chain[k-1], V, 0)
            info.converged || error("Continuation failed at V=$V after sub-stepping — reason: $(info.reason)")
        end

        r = assert_residual(V)

        if pending_ab
            et = 0.0; ec = 0.0
            @inbounds for j in 1:N
                et += (u_pred_t[j] - u[j])^2
                ec += (u_pred_c[j] - u[j])^2
            end
            push!(pred_tan, sqrt(et)); push!(pred_carry, sqrt(ec))
            pending_ab = false
        end

        if emit[k] || k == 1
            if emit[k]
                save_solution(cache_dir, prefix, u, V)
                push!(voltages, V); push!(solutions, copy(u))
                push!(iters, 0); push!(resids, r)
            end
        end
        verbose && (k % 10 == 0 || k == length(V_chain)) &&
            @printf("  chain %3d/%3d  V=%8.5f  ||F||inf=%.3e  cum.iters=%d  (%.1f s)\n",
                    k, length(V_chain), V, r, tot_iters, time() - t_start)
        flush(stdout)

        have_tangent = tangent_predictor!(dudV, plu, res_at_V!, u, V; Fp = Fp, Fm = Fm)
        tot_res += 2; tot_bsub += 1
    end

    return (voltages = voltages, solutions = solutions, resids = resids,
            n_chain = length(V_chain), tot_iters = tot_iters,
            iters_per_step = tot_iters / length(V_chain),
            n_res = tot_res, n_bsub = tot_bsub, n_substep = n_substep,
            n_assert_fire = n_assert_fire,
            pred_tangent_err = pred_tan, pred_carry_err = pred_carry,
            lu_symbolic = plu.n_symbolic, lu_numeric = plu.n_numeric,
            lu_nnz_change = plu.n_nnz_change, lu_pattern_catch = plu.n_lu_catch,
            wall = time() - t_start, plu = plu, ctx = ctx)
end
