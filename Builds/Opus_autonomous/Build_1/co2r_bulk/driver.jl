# driver.jl — the §9.5.5 unified continuation driver, shared by ALL stages
#   anchor → tangent-predictor sweep → §9.5.6 post-solve assertion → cache

using Printf

# ─────────────────────────────────────────────────────────────────────────────
# §10 Per-stage binary cache
# ─────────────────────────────────────────────────────────────────────────────
cache_dir_for(stage_tag::String) =
    joinpath(@__DIR__, "output_$(stage_tag)", "cache")

function save_solution(stage_tag::String, prefix::String, u::Vector{Float64}, V::Float64)
    d = cache_dir_for(stage_tag); mkpath(d)
    fn = joinpath(d, @sprintf("%sV%.6f.bin", prefix, V))
    open(fn, "w") do f
        write(f, Int64(length(u)))
        write(f, u)
    end
    return fn
end

function load_solution(stage_tag::String, prefix::String, V::Float64)
    fn = joinpath(cache_dir_for(stage_tag), @sprintf("%sV%.6f.bin", prefix, V))
    isfile(fn) || return nothing
    open(fn, "r") do f
        n = read(f, Int64)
        read!(f, Vector{Float64}(undef, n))
    end
end

function list_cache(stage_tag::String, prefix::String)
    d = cache_dir_for(stage_tag)
    isdir(d) || return Float64[]
    pfx = prefix * "V"
    files = filter(f -> startswith(f, pfx) && endswith(f, ".bin"), readdir(d))
    return sort([parse(Float64, f[length(pfx)+1:end-4]) for f in files])
end

# ─────────────────────────────────────────────────────────────────────────────
# Initial-guess builders (§9.5.5 anchor table)
# ─────────────────────────────────────────────────────────────────────────────
"""
    make_u0(ly, dm; V_applied_SHE=0.0)

Anchor guess constructed from physics: every CL cell at the exact bulk buffer
equilibrium; every gas cell at the inlet composition and `p_total`; φ_s at
`V_applied_SHE` and φ_L = 0 (Stage 5).
"""
function make_u0(ly::Layout, dm::Domain; V_applied_SHE::Float64=0.0)
    u = zeros(ly.N_TOT)
    lnc = log.(dm.c_eq)
    for ix in 1:ly.N_CL, i in 1:5
        u[liquid_idx(ix,i)] = lnc[i]
    end
    if ly.stage >= 3
        for k in 1:ly.N_G
            u[gas_idx(ly,k,1)] = omega_inlet[1]
            u[gas_idx(ly,k,2)] = omega_inlet[2]
            u[gas_idx(ly,k,3)] = omega_inlet[3]
            u[gas_idx(ly,k,4)] = p_total
        end
    end
    if ly.stage >= 5
        for k in 1:ly.N_G
            u[phi_s_idx(ly,k)] = V_applied_SHE
        end
        for ix in 1:ly.N_CL
            u[phi_L_idx(ly,ix)] = 0.0
        end
    end
    return u
end

# ─────────────────────────────────────────────────────────────────────────────
# Solver bundle: residual closure + AD Jacobian + persistent LU
# ─────────────────────────────────────────────────────────────────────────────
mutable struct SolverCtx
    ly::Layout
    dm::Domain
    Vref::Base.RefValue{Float64}      # V_cathode in V vs RHE@pH8
    res_cl!::Function
    res_at_V!::Function
    jacfun!::Function
    Jdense::Matrix{Float64}
    plu::AffineLU
    blocks::Vector{UnitRange{Int}}
    Fwork::Vector{Float64}
end

function make_ctx(ly::Layout, dm::Domain)
    N = ly.N_TOT
    Vref = Ref(0.0)
    res_cl! = (Fo, uo) -> residual!(Fo, uo, Vref[], dm, ly)
    res_at_V! = (Fo, uo, Vv) -> residual!(Fo, uo, Vv, dm, ly)
    Fwork = zeros(N)
    cfg = ForwardDiff.JacobianConfig(res_cl!, zeros(N), zeros(N), ForwardDiff.Chunk{12}())
    Jdense = zeros(N, N)
    jacfun! = (Jd, uu) -> ForwardDiff.jacobian!(Jd, res_cl!, Fwork, uu, cfg)
    return SolverCtx(ly, dm, Vref, res_cl!, res_at_V!, jacfun!, Jdense,
                     AffineLU(), weight_blocks(ly), Fwork)
end

function solve_at!(ctx::SolverCtx, u::Vector{Float64}, V::Float64;
                   alpha_init::Float64, max_iter::Int, rtol::Float64=RTOL_PROD)
    ctx.Vref[] = V
    return newton_affine!(u, ctx.res_cl!;
                          blocks=ctx.blocks, plu=ctx.plu, jacfun! = ctx.jacfun!,
                          Jdense=ctx.Jdense, rtol=rtol, max_iter=max_iter,
                          alpha_init=alpha_init, afrac=AFRAC,
                          max_log_step=MAX_LOG_STEP, n_log_dofs=5*ctx.ly.N_CL,
                          alpha_min=ALPHA_MIN)
end

# ─────────────────────────────────────────────────────────────────────────────
# §9.5.6 Post-solve residual assertion
# ─────────────────────────────────────────────────────────────────────────────
function post_solve_assert!(ctx::SolverCtx, u::Vector{Float64}, V::Float64)
    Fchk = zeros(length(u))
    ctx.res_at_V!(Fchk, u, V)
    r = maximum(abs, Fchk)
    if r > F_INF_TOL
        solve_at!(ctx, u, V; alpha_init=1.0, max_iter=3, rtol=1e-10)
        ctx.res_at_V!(Fchk, u, V)
        r = maximum(abs, Fchk)
        r > F_INF_TOL && error(@sprintf("residual assertion failed at V=%.4f (||F||=%.3e) — halve dV and re-approach", V, r))
    end
    return r
end

# ─────────────────────────────────────────────────────────────────────────────
# §9.5.5 Unified continuation driver
# ─────────────────────────────────────────────────────────────────────────────
"""
    continuation_sweep(ctx, u0, Vs; emit, stage_tag, prefix, quiet=false)

Visit the voltages `Vs` (V vs RHE@pH8) in order.  `Vs[1]` is the anchor
(solved at `alpha_init = 1e-2`); every later point is reached by the tangent
predictor + `newton_affine!` at `alpha_init = 1`, then the §9.5.6 assertion,
then cached if `emit[k]`.  On a failed step: roll back, halve dV, solve the
two half-steps at `alpha_init = 1e-2`.

Returns `(Vout, Uout, stats)`.
"""
function continuation_sweep(ctx::SolverCtx, u0::Vector{Float64}, Vs::Vector{Float64};
                            emit::Vector{Bool}, stage_tag::String, prefix::String,
                            quiet::Bool=false)
    N = length(u0)
    u = copy(u0)
    dudV = zeros(N)
    Vout = Float64[]; Uout = Vector{Float64}[]
    iters = Int[]; resids = Float64[]
    n_substep = 0

    info = solve_at!(ctx, u, Vs[1]; alpha_init=ALPHA_ANCHOR, max_iter=MAX_ITER_ANCHOR)
    info.converged || error("ANCHOR FAILED at V=$(Vs[1]): $(info.reason), ||F||=$(info.residual)")
    r = post_solve_assert!(ctx, u, Vs[1])
    quiet || @printf("  anchor V_RHE=%+.4f: %d iters, ||F||inf=%.3e\n", Vs[1], info.iterations, r)
    if emit[1]
        save_solution(stage_tag, prefix, u, Vs[1])
        push!(Vout, Vs[1]); push!(Uout, copy(u))
        push!(iters, info.iterations); push!(resids, r)
    end
    have_tan = tangent_predictor!(dudV, ctx.plu, ctx.res_at_V!, u, Vs[1])

    for k in 2:length(Vs)
        Vprev = Vs[k-1]; Vcur = Vs[k]; dV = Vcur - Vprev
        u_back = copy(u)
        ok = false
        # predictor
        if have_tan
            @inbounds for j in 1:N; u[j] += dV * dudV[j]; end
        elseif ctx.ly.stage >= 5
            @inbounds for kk in 1:ctx.ly.N_G; u[phi_s_idx(ctx.ly,kk)] += dV; end
        end
        info = solve_at!(ctx, u, Vcur; alpha_init=ALPHA_SWEEP, max_iter=MAX_ITER_SWEEP)
        if info.converged
            ok = true
        else
            # sub-step recovery
            n_substep += 1
            quiet || @printf("  [recovery] step to V_RHE=%+.4f failed (%s) — halving dV\n",
                             Vcur, String(info.reason))
            copyto!(u, u_back)
            Vmid = Vprev + 0.5*dV
            if have_tan
                @inbounds for j in 1:N; u[j] += 0.5*dV*dudV[j]; end
            end
            i1 = solve_at!(ctx, u, Vmid; alpha_init=ALPHA_ANCHOR, max_iter=MAX_ITER_ANCHOR)
            if i1.converged
                have_tan = tangent_predictor!(dudV, ctx.plu, ctx.res_at_V!, u, Vmid)
                if have_tan
                    @inbounds for j in 1:N; u[j] += 0.5*dV*dudV[j]; end
                end
                i2 = solve_at!(ctx, u, Vcur; alpha_init=ALPHA_ANCHOR, max_iter=MAX_ITER_ANCHOR)
                ok = i2.converged
                info = i2
            end
        end
        ok || error("SWEEP FAILED at V_RHE=$(Vcur): $(info.reason), ||F||=$(info.residual)")
        r = post_solve_assert!(ctx, u, Vcur)
        if emit[k]
            save_solution(stage_tag, prefix, u, Vcur)
            push!(Vout, Vcur); push!(Uout, copy(u))
            push!(iters, info.iterations); push!(resids, r)
            quiet || @printf("  V_RHE=%+.4f: %d iters, ||F||inf=%.3e\n", Vcur, info.iterations, r)
        end
        have_tan = tangent_predictor!(dudV, ctx.plu, ctx.res_at_V!, u, Vcur)
    end
    stats = (iters=iters, resids=resids, n_substep=n_substep)
    return Vout, Uout, stats
end

# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics + export for the plotting layer
# ─────────────────────────────────────────────────────────────────────────────
out_dir_for(stage_tag::String) = joinpath(@__DIR__, "output_$(stage_tag)")

function export_polarization(stage_tag::String, Vs::Vector{Float64},
                             Us::Vector{Vector{Float64}}, ly::Layout, dm::Domain)
    d = out_dir_for(stage_tag); mkpath(d)
    open(joinpath(d, "polarization.csv"), "w") do f
        println(f, "V_RHE,V_SHE,i_CO_mAcm2,i_H2_mAcm2,i_tot_mAcm2,FE_CO_pct")
        for (k,V) in enumerate(Vs)
            iCO, iH2, itot = compute_currents(Us[k], ly, dm, V)
            fe = itot > 0 ? 100*iCO/itot : 0.0
            @printf(f, "%.6f,%.6f,%.6e,%.6e,%.6e,%.4f\n",
                    V, V - V_RHE_to_SHE_offset, iCO/10, iH2/10, itot/10, fe)
        end
    end
end

"""Binary profile dump read by plot_results.py."""
function export_profiles(stage_tag::String, Vs::Vector{Float64},
                         Us::Vector{Vector{Float64}}, ly::Layout, dm::Domain)
    d = out_dir_for(stage_tag); mkpath(d)
    nV = length(Vs); N_CL = ly.N_CL; N_G = ly.N_G
    has_gas = ly.stage >= 3 ? 1 : 0
    has_pot = ly.stage >= 5 ? 1 : 0
    open(joinpath(d, "profiles.bin"), "w") do f
        write(f, Int64(nV)); write(f, Int64(N_CL)); write(f, Int64(N_G))
        write(f, Int64(has_gas)); write(f, Int64(has_pot))
        write(f, collect(dm.cl.x_centers .* 1e6))     # μm
        write(f, collect(dm.xc_g .* 1e6))             # μm
        write(f, collect(Vs))
        for (k,V) in enumerate(Vs)
            u = Us[k]
            V_SHE = V - V_RHE_to_SHE_offset
            Rco, Rher, c = compute_rates(u, ly, V_SHE)
            pH  = [-log10(max(c[1,ix]/1000, 1e-30)) for ix in 1:N_CL]
            cCs = [cCs_from_electroneutrality(c[1,ix],c[2,ix],c[3,ix],c[4,ix]) for ix in 1:N_CL]
            write(f, pH)
            write(f, collect(c[5,:])); write(f, collect(c[2,:]))
            write(f, collect(c[3,:])); write(f, collect(c[4,:]))
            write(f, cCs);             write(f, collect(c[1,:]))
            write(f, Rco);             write(f, Rher)
            if has_gas == 1
                pco2 = zeros(N_G); pco = zeros(N_G); ph2 = zeros(N_G); ptot = zeros(N_G)
                for kk in 1:N_G
                    w = gas_omega(u, ly, kk); p = gas_p(u, ly, kk)
                    x = omega_to_x(w[1], w[2], w[3], w[4])
                    ptot[kk] = p
                    pco2[kk] = x[1]*p/101325; pco[kk] = x[2]*p/101325; ph2[kk] = x[3]*p/101325
                end
                write(f, pco2); write(f, pco); write(f, ph2); write(f, ptot)
            end
            if has_pot == 1
                write(f, [u[phi_s_idx(ly,kk)] for kk in 1:N_G])
                write(f, [u[phi_L_idx(ly,ix)] for ix in 1:N_CL])
            end
        end
    end
end

function print_kinetics_header(stage_name::String, stage_tag::String)
    @printf("=== %s ===\n", stage_name)
    @printf("i0_CO2R = %.4f A/m^2, lambda_MHC = %.4f eV\n", i0_CO2R(), lambda_MHC())
    @printf("HER_pf  = %.4e, alpha_c = %.5f, k_GL = %.3e m/s\n",
            HER_prefactor(), alpha_c_HER(), k_GL())
    d = cache_dir_for(stage_tag)
    @printf("Cache dir: %s\n", d)
    if isdir(d) && !isempty(readdir(d))
        @printf("WARNING: Existing cache found. Verify kinetics match before reusing!\n")
    end
    flush(stdout)
end
