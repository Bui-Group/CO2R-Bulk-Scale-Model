# ============================================================================
# driver.jl — §10 binary cache + the §9.5.5 unified continuation driver.
# Written ONCE and shared by every stage (§13).
# ============================================================================
using Printf

# ── §10.3 binary cache ─────────────────────────────────────────────────────
cache_dir(stage::Int) = joinpath(@__DIR__, "output_stage$(stage)", "cache")
cache_prefix(stage::Int) = "s$(stage)_V"

function save_solution(u::Vector{Float64}, V::Float64, dir::String, prefix::String)
    mkpath(dir)
    fn = joinpath(dir, @sprintf("%s%.6f.bin", prefix, V))
    open(fn, "w") do f
        write(f, Int64(length(u)))
        write(f, u)
    end
    return fn
end

function load_solution(V::Float64, dir::String, prefix::String)
    fn = joinpath(dir, @sprintf("%s%.6f.bin", prefix, V))
    isfile(fn) || return nothing
    open(fn, "r") do f
        n = read(f, Int64)
        read!(f, Vector{Float64}(undef, n))
    end
end

function list_cache(dir::String, prefix::String)
    isdir(dir) || return Float64[]
    files = filter(f -> startswith(f, prefix) && endswith(f, ".bin"), readdir(dir))
    return sort([parse(Float64, f[length(prefix)+1:end-4]) for f in files])
end

# ── solver workspace ───────────────────────────────────────────────────────
struct SolveWS{C}
    ctx::Ctx
    rf::ResFun{C}
    plu::AffineLU
    Jd::Matrix{Float64}
    cfg::Any
end

function SolveWS(ctx::Ctx, u_ref::Vector{Float64})
    rf  = ResFun(ctx, 0.0)
    N   = ctx.ly.N_TOT
    Jd  = zeros(N, N)
    cfg = ForwardDiff.JacobianConfig(rf, zeros(N), u_ref, ForwardDiff.Chunk{12}())
    return SolveWS(ctx, rf, AffineLU(), Jd, cfg)
end

nlog(ctx::Ctx) = 5 * ctx.ly.N_CL

function solve_at!(ws::SolveWS, u::Vector{Float64}, V::Float64;
                   alpha_init = 1.0, max_iter = 40, rtol = RTOL_PROD)
    ws.rf.V = V
    return newton_affine!(u, ws.rf; ly = ws.ctx.ly, plu = ws.plu, Jd = ws.Jd,
                          cfg = ws.cfg, rtol = rtol, max_iter = max_iter,
                          alpha_init = alpha_init, n_log_dofs = nlog(ws.ctx))
end

"§9.5.6 post-solve residual assertion."
function residual_assert!(ws::SolveWS, u::Vector{Float64}, V::Float64;
                          tol = F_INF_TOL)
    Fchk = zeros(length(u))
    ws.rf.V = V
    ws.rf(Fchk, u)
    r = maximum(abs, Fchk)
    if r > tol
        @printf("    [assert] ||F||inf = %.3e > %.1e at V=%.4f — one extra Newton\n", r, tol, V)
        solve_at!(ws, u, V; alpha_init = 1.0, max_iter = 3, rtol = 1e-10)
        ws.rf.V = V; ws.rf(Fchk, u)
        r = maximum(abs, Fchk)
        r > tol && error(@sprintf("residual assertion failed at V=%.4f (||F||inf=%.3e) — halve dV and re-approach", V, r))
    end
    return r
end

"φ_s-shift fallback predictor (§9.5.4): the φ_s boundary value moves by exactly dV."
function phis_shift!(u::Vector{Float64}, ly::Layout, dV::Float64)
    if ly.stage >= 5
        @inbounds for ig in 1:ly.N_GDL; u[phi_s_gdl_idx(ly, ig)] += dV; end
        @inbounds for ix in 1:ly.N_CL;  u[phi_s_cl_idx(ly, ix)]  += dV; end
    end
    return u
end

"""
    continuation_sweep!(ws, u, V_chain; emit, on_emit, ...)

The §9.5.5 unified driver: anchor at `V_chain[1]`, then walk the chain with the
tangent predictor + affine-covariant corrector, running the §9.5.6 assertion at
every emitted voltage and calling `on_emit(V, u, info, resid)` there.
Recovery on a failed step: roll back, halve dV, take two half-steps at
`alpha_init = 1e-2`.
"""
function continuation_sweep!(ws::SolveWS, u::Vector{Float64}, V_chain::Vector{Float64};
                             emit::Function = V -> true,
                             on_emit::Function = (V, u, info, r) -> nothing,
                             anchor_alpha = 1e-2, anchor_iter = 200,
                             verbose = true)
    N  = length(u)
    ly = ws.ctx.ly
    dudV = zeros(N); Fp = zeros(N); Fm = zeros(N)
    stats = Dict{Symbol,Any}(:iters => Int[], :resid => Float64[], :failed => Float64[],
                             :substeps => 0)

    V0 = V_chain[1]
    info = solve_at!(ws, u, V0; alpha_init = anchor_alpha, max_iter = anchor_iter)
    info.converged || error(@sprintf("ANCHOR failed at V=%.4f (%s)", V0, info.reason))
    verbose && @printf("anchor  V=%8.4f  iters=%2d  ||F||inf=%.3e  err=%.2e  (%s)\n",
                       V0, info.iterations, info.residual, info.err_est, info.reason)
    if emit(V0)
        r = residual_assert!(ws, u, V0)
        on_emit(V0, u, info, r)
        push!(stats[:iters], info.iterations); push!(stats[:resid], r)
    end
    have_tangent = tangent_predictor!(dudV, ws.plu, ws.rf, u, V0; Fp = Fp, Fm = Fm)

    u_prev = copy(u)
    V_prev = V0
    for k in 2:length(V_chain)
        V = V_chain[k]
        ok = step_to!(ws, u, u_prev, V_prev, V, dudV, have_tangent, Fp, Fm; depth = 0,
                      stats = stats, verbose = verbose)
        if !ok
            push!(stats[:failed], V)
            error(@sprintf("continuation failed at V=%.4f even after sub-stepping", V))
        end
        infoV = stats[:last_info]
        if emit(V)
            r = residual_assert!(ws, u, V)
            on_emit(V, u, infoV, r)
            push!(stats[:iters], infoV.iterations); push!(stats[:resid], r)
        end
        verbose && @printf("  V=%8.4f  iters=%2d  ||F||inf=%.3e  a=%.3f\n",
                           V, infoV.iterations, infoV.residual, infoV.alpha_final)
        copyto!(u_prev, u); V_prev = V
        have_tangent = tangent_predictor!(dudV, ws.plu, ws.rf, u, V; Fp = Fp, Fm = Fm)
    end
    return stats
end

"One continuation step with recursive sub-step recovery."
function step_to!(ws::SolveWS, u, u_start, V_from, V_to, dudV, have_tangent, Fp, Fm;
                  depth = 0, stats = nothing, verbose = true)
    dV = V_to - V_from
    copyto!(u, u_start)
    if have_tangent
        @inbounds for j in eachindex(u); u[j] += dV * dudV[j]; end
    else
        phis_shift!(u, ws.ctx.ly, dV)
    end
    ai = depth == 0 ? 1.0 : 1e-2
    info = solve_at!(ws, u, V_to; alpha_init = ai, max_iter = 40)
    if info.converged
        stats !== nothing && (stats[:last_info] = info)
        return true
    end
    depth >= 4 && return false
    verbose && @printf("    [recover] step to V=%.4f failed (%s) — halving dV (depth %d)\n",
                       V_to, info.reason, depth + 1)
    stats !== nothing && (stats[:substeps] += 1)
    V_mid = 0.5 * (V_from + V_to)
    u_mid = copy(u_start)
    ok1 = step_to!(ws, u_mid, u_start, V_from, V_mid, dudV, have_tangent, Fp, Fm;
                   depth = depth + 1, stats = stats, verbose = verbose)
    ok1 || return false
    ht2 = tangent_predictor!(dudV, ws.plu, ws.rf, u_mid, V_mid; Fp = Fp, Fm = Fm)
    return step_to!(ws, u, u_mid, V_mid, V_to, dudV, ht2, Fp, Fm;
                    depth = depth + 1, stats = stats, verbose = verbose)
end

# ── shared stage runner (§13: write the driver ONCE) ───────────────────────
"""
    run_stage(stage; V_chain, emit_set, u0, tag, outdir)

Anchor + tangent-predictor continuation for any stage, caching every emitted
solution immediately after the §9.5.6 assertion (§10.4) and exporting the
sweep for plotting.
"""
function run_stage(stage::Int; N_CL = N_CL_DEFAULT, N_GDL = N_GDL_DEFAULT,
                   V_chain::Vector{Float64} = copy(V_RHE_grid),
                   emit_set::Vector{Float64} = copy(V_RHE_grid),
                   u0::Union{Nothing,Vector{Float64}} = nothing,
                   tag::String = "stage$(stage)",
                   outdir::String = joinpath(@__DIR__, "output_stage$(stage)"),
                   cdir::String = cache_dir(stage),
                   cpref::String = cache_prefix(stage),
                   verbose = true)
    eq = solve_buffer_equilibrium(1000.0, 1000.0)
    c_eq = [eq.H, eq.OH, eq.HCO3, eq.CO3, eq.CO2]
    ctx  = Ctx(stage, N_CL, N_GDL, c_eq)
    u    = u0 === nothing ? make_u0(ctx) : copy(u0)
    ws   = SolveWS(ctx, u)

    emitset = Set(round.(emit_set, digits = 9))
    recs = SweepRecord[]
    t0 = time()
    stats = continuation_sweep!(ws, u, V_chain;
        emit = V -> round(V, digits = 9) in emitset,
        on_emit = (V, uu, info, r) -> begin
            save_solution(uu, V, cdir, cpref)
            push!(recs, SweepRecord(V, copy(uu), info.iterations, r))
        end, verbose = verbose)
    wall = time() - t0

    @printf("\nSweep complete: %d/%d emitted voltages converged in %.1f s\n",
            length(recs), length(emitset), wall)
    @printf("Newton iters/step: mean %.2f  max %d   max ||F||inf = %.3e   sub-steps: %d\n",
            sum(stats[:iters])/length(stats[:iters]), maximum(stats[:iters]),
            maximum(stats[:resid]), stats[:substeps])
    export_sweep(outdir, tag, ctx, recs)
    return (ctx = ctx, ws = ws, recs = recs, stats = stats, wall = wall)
end
