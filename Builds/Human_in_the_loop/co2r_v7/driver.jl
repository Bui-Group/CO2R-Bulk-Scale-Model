# =====================================================================
#  driver.jl — unified continuation driver (§9.5.4 – §9.5.6)
#  anchor → tangent-predictor continuation → post-solve residual assertion
#  → cache.  Identical for every stage; only the residual and the anchor
#  recipe differ (§9.5.5).
# =====================================================================

using Printf

"Per-voltage record emitted by the sweep."
struct StepRecord
    V_rhe::Float64
    V_she::Float64
    iters::Int
    n_res::Int
    resid::Float64
    err_est::Float64
    i_CO::Float64        # A/m²
    i_H2::Float64
    dphiL::Float64       # φ_L[1] − φ_L[N_CL]   (Stage 5 only; 0 otherwise)
    substeps::Int
end

"""
    continuation_sweep(Val(S); ly, cl, gdl, gg, c_eq, u0, V_anchor_rhe,
                       V_chain_rhe, emit_set, cache_dir, prefix, verbose)

Runs the §9.5.5 driver:

    anchor:  newton_affine! (alpha_init = 1e-2, max_iter = 200) → tangent_predictor!
    sweep:   u .+= dV .* dudV;  newton_affine! (alpha_init = 1, max_iter = 40);
             §9.5.6 post-solve assertion;  cache;  tangent_predictor!
    recovery: roll back, halve dV, solve the two half-steps at alpha_init = 1e-2.

Voltages are in V vs RHE@pH 8 throughout (§11 gotcha #9a).  Only voltages in
`emit_set` are cached/recorded (lets Stage 5 ramp silently from V_c = 0).
"""
function continuation_sweep(::Val{S}; ly::ModelLayout, cl::Mesh1D, gdl::Mesh1D,
                            gg::GasGeom, c_eq::Vector{Float64},
                            u0::Vector{Float64},
                            V_anchor_rhe::Float64,
                            V_chain_rhe::Vector{Float64},
                            emit_set::Union{Nothing,Set{Float64}} = nothing,
                            cache_dir::AbstractString, prefix::AbstractString,
                            verbose::Bool = true,
                            anchor_alpha::Float64 = ALPHA_ANCHOR,
                            rtol::Float64 = RTOL_PROD) where {S}

    N   = ly.N_TOT
    plu = AffineLU()
    Jd  = zeros(N, N)
    Vref = Ref(V_anchor_rhe)

    res!      = (F_, u_)     -> full_residual!(F_, u_, Vref[], cl, gdl, c_eq, ly, gg, Val(S))
    res_at_V! = (F_, u_, V_) -> full_residual!(F_, u_, V_,    cl, gdl, c_eq, ly, gg, Val(S))
    cfg = ForwardDiff.JacobianConfig(res!, zeros(N), zeros(N), ForwardDiff.Chunk{12}())

    n_log = 5 * ly.N_CL
    Fchk  = zeros(N); Fp = zeros(N); Fm = zeros(N)
    dudV  = zeros(N); have_tan = false
    records = StepRecord[]
    mkpath(cache_dir)

    emits(V) = emit_set === nothing || any(abs(V - e) < 1e-9 for e in emit_set)

    function dphi_L(u)
        S == 5 || return 0.0
        return u[phi_L_idx5(ly, 1)] - u[phi_L_idx5(ly, ly.N_CL)]
    end

    function record!(V, info, nsub)
        iCO, iH2, _ = compute_currents(u_cur, V, cl, ly)
        push!(records, StepRecord(V, V - V_RHE_to_SHE_offset, info.iterations,
                                  info.n_res, info.residual, info.err_est,
                                  iCO, iH2, dphi_L(u_cur), nsub))
        save_solution(cache_dir, prefix, u_cur, V)
        if verbose
            @printf("  V_RHE=%+.5f  it=%d  |F|=%.2e  err=%.2e  i_CO=%8.2f  i_H2=%8.2f mA/cm2",
                    V, info.iterations, info.residual, info.err_est, iCO/10, iH2/10)
            S == 5 && @printf("  dphiL=%+.2f mV", dphi_L(u_cur)*1000)
            nsub > 0 && @printf("  [substeps=%d]", nsub)
            println(); flush(stdout)
        end
    end

    # ── anchor ────────────────────────────────────────────────────────
    u_cur = copy(u0)
    Vref[] = V_anchor_rhe
    t0 = time()
    info = newton_affine!(u_cur, res!; ly = ly, plu = plu, rtol = rtol,
                          max_iter = MAXIT_ANCHOR, alpha_init = anchor_alpha,
                          n_log_dofs = n_log, Jdense = Jd, cfg = cfg)
    verbose && @printf("ANCHOR V_RHE=%+.5f : converged=%s it=%d |F|=%.3e err=%.3e (%s)\n",
                       V_anchor_rhe, info.converged, info.iterations,
                       info.residual, info.err_est, info.reason)
    info.converged || error("Anchor solve failed at V_RHE = $V_anchor_rhe ($(info.reason))")
    res_at_V!(Fchk, u_cur, V_anchor_rhe)
    maximum(abs, Fchk) > F_INF_TOL &&
        error("Anchor residual assertion failed: |F|inf = $(maximum(abs,Fchk))")
    emits(V_anchor_rhe) && record!(V_anchor_rhe, info, 0)
    have_tan = tangent_predictor!(dudV, plu, res_at_V!, u_cur, V_anchor_rhe; Fp = Fp, Fm = Fm)

    # ── predictor + corrector for one step, with sub-step recovery ────
    function apply_predictor!(u, dV)
        if have_tan
            @inbounds for j in 1:N; u[j] += dV * dudV[j]; end
        elseif S == 5                     # exact φ_s-shift fallback (§9.5.4)
            @inbounds for ig in 1:ly.N_GDL; u[phi_s_gdl_idx5(ly, ig)] += dV; end
            @inbounds for ix in 1:ly.N_CL; u[phi_s_cl_idx5(ly, ix)]  += dV; end
        end
        return u
    end

    function advance!(V_prev::Float64, V_target::Float64, alpha0::Float64, halvings::Int)
        u_save = copy(u_cur); tan_save = have_tan; dudV_save = copy(dudV)
        apply_predictor!(u_cur, V_target - V_prev)
        Vref[] = V_target
        info = newton_affine!(u_cur, res!; ly = ly, plu = plu, rtol = rtol,
                              max_iter = MAXIT_SWEEP, alpha_init = alpha0,
                              n_log_dofs = n_log, Jdense = Jd, cfg = cfg)
        ok = info.converged
        if ok                                            # §9.5.6 post-solve assertion
            res_at_V!(Fchk, u_cur, V_target)
            if maximum(abs, Fchk) > F_INF_TOL
                newton_affine!(u_cur, res!; ly = ly, plu = plu, rtol = 1e-10,
                               max_iter = 3, alpha_init = 1.0, n_log_dofs = n_log,
                               Jdense = Jd, cfg = cfg)
                res_at_V!(Fchk, u_cur, V_target)
                ok = maximum(abs, Fchk) <= F_INF_TOL
            end
        end
        if ok
            have_tan = tangent_predictor!(dudV, plu, res_at_V!, u_cur, V_target; Fp = Fp, Fm = Fm)
            return (true, info, 0)
        end
        # recovery: roll back, halve dV, two half-steps at alpha_init = 1e-2
        copyto!(u_cur, u_save); have_tan = tan_save; dudV .= dudV_save
        if halvings <= 0
            return (false, info, 0)
        end
        Vmid = 0.5 * (V_prev + V_target)
        ok1, i1, s1 = advance!(V_prev, Vmid, ALPHA_ANCHOR, halvings - 1)
        ok1 || return (false, i1, s1 + 1)
        ok2, i2, s2 = advance!(Vmid, V_target, ALPHA_ANCHOR, halvings - 1)
        return (ok2, i2, s1 + s2 + 2)
    end

    # ── sweep ─────────────────────────────────────────────────────────
    V_prev = V_anchor_rhe
    for V_target in V_chain_rhe
        ok, info, nsub = advance!(V_prev, V_target, ALPHA_SWEEP, 4)
        if !ok
            @printf("  !! FAILED at V_RHE = %+.5f (%s, |F|=%.2e)\n",
                    V_target, info.reason, info.residual); flush(stdout)
            error("continuation step failed at V_RHE = $V_target")
        end
        emits(V_target) && record!(V_target, info, nsub)
        V_prev = V_target
    end
    verbose && @printf("Sweep done: %d emitted voltages, %.1f s\n", length(records), time() - t0)
    return records, plu, (cl = cl, gdl = gdl, gg = gg, ly = ly, c_eq = c_eq)
end

# ── anchor guess builders (§9.5.5 table) ─────────────────────────────
"Stages 1-2: every CL cell at the exact bulk equilibrium."
function make_u0_liquid(c_eq::Vector{Float64}, ly::ModelLayout)
    u0 = zeros(ly.N_TOT)
    lnc = log.(c_eq)
    for ix in 1:ly.N_CL, i in 1:5
        u0[liquid_idx(ix, i)] = lnc[i]
    end
    return u0
end

"Stages 3-4: equilibrium liquid + inlet gas composition at p_total everywhere."
function make_u0_gasliquid(c_eq::Vector{Float64}, ly::ModelLayout)
    u0 = make_u0_liquid(c_eq, ly)
    for g in 1:(ly.N_GDL + ly.N_CL)
        b = gas_base(ly, g)
        u0[b+1] = omega_inlet[1]; u0[b+2] = omega_inlet[2]
        u0[b+3] = omega_inlet[3]; u0[b+4] = p_total
    end
    return u0
end

"""
Stage 5 anchor at V_c = 0 — constructed from physics, never from a cache
(§9.5.5, §11 gotcha #0).  `V_cathode_rhe` is converted to SHE before being
written into the φ_s DOFs (§11 gotcha #9a).
"""
function make_u0_stage5_anchor(c_eq::Vector{Float64}, ly::ModelLayout,
                               V_cathode_rhe::Float64)
    u0 = make_u0_gasliquid(c_eq, ly)
    V_applied_SHE = V_cathode_rhe - V_RHE_to_SHE_offset
    for ig in 1:ly.N_GDL; u0[phi_s_gdl_idx5(ly, ig)] = V_applied_SHE; end
    for ix in 1:ly.N_CL;  u0[phi_s_cl_idx5(ly, ix)]  = V_applied_SHE; end
    for ix in 1:ly.N_CL;  u0[phi_L_idx5(ly, ix)]     = 0.0; end
    return u0
end

# ── summary / profile export for the plotting scripts ────────────────
function write_summary(dir, records::Vector{StepRecord})
    mkpath(dir)
    open(joinpath(dir, "summary.csv"), "w") do io
        println(io, "V_rhe,V_she,iters,n_res,resid,err_est,i_CO_mAcm2,i_H2_mAcm2,i_tot_mAcm2,FE_CO,FE_H2,dphiL_mV,substeps")
        for r in records
            it = r.i_CO + r.i_H2
            @printf(io, "%.6f,%.6f,%d,%d,%.6e,%.6e,%.8f,%.8f,%.8f,%.6f,%.6f,%.6f,%d\n",
                    r.V_rhe, r.V_she, r.iters, r.n_res, r.resid, r.err_est,
                    r.i_CO/10, r.i_H2/10, it/10,
                    it > 0 ? 100*r.i_CO/it : 0.0, it > 0 ? 100*r.i_H2/it : 0.0,
                    r.dphiL*1000, r.substeps)
        end
    end
end
