# =============================================================================
# sanity_tests.jl — §9.5.9 first-build sanity tests, steps 1-4.
# Steps 5-8 are Stage-5 tests and are run by run_stage5.jl.
# =============================================================================
include(joinpath(@__DIR__, "model.jl"))
using Printf, SparseArrays, LinearAlgebra

println("="^78); println("§9.5.9 FIRST-BUILD SANITY TESTS"); println("="^78)

# ── Step 1: persistent LU reuses the symbolic analysis ───────────────────────
println("\n[1] Persistent sparse LU")
ctx1 = make_ctx(1)
u1   = make_anchor_guess(ctx1, -0.300)
N    = ctx1.ly.N_TOT
res1! = (F_, u_) -> full_residual!(F_, u_, -0.300, ctx1)
Jd = zeros(N, N); Fo = zeros(N)
cfg = ForwardDiff.JacobianConfig(res1!, zeros(N), copy(u1), ForwardDiff.Chunk{12}())
ForwardDiff.jacobian!(Jd, res1!, Fo, u1, cfg)
Js = sparse(Jd)
plu = AffineLU()
t1 = @elapsed (x1, ok1) = aff_factor_solve!(plu, Js, -Fo)
t2 = @elapsed (x2, ok2) = aff_factor_solve!(plu, Js, -Fo)
t3 = @elapsed x3 = aff_resolve(plu, -Fo)
@printf("    factorize#1 %.4f ms (symbolic=%d)   factorize#2 %.4f ms (symbolic=%d, numeric=%d)\n",
        t1*1e3, plu.n_symbolic, t2*1e3, plu.n_symbolic, plu.n_numeric)
@printf("    back-substitute only %.4f ms ; ||x1-x2||inf=%.2e ; ||x1-x3||inf=%.2e\n",
        t3*1e3, maximum(abs, x1 .- x2), maximum(abs, x1 .- x3))
@printf("    PASS: second call skipped the symbolic phase = %s ; solves identical = %s\n",
        plu.n_symbolic == 1 && plu.n_numeric == 1, maximum(abs, x1 .- x2) == 0.0)
@printf("    Jacobian nnz = %d (%.3f %% dense), cond-ish ||J||inf=%.3e\n",
        nnz(Js), 100*nnz(Js)/N^2, maximum(sum(abs, Jd, dims=2)))

# ── AD validation: finite-difference spot check (§9.5.8 checklist) ───────────
println("\n[1b] AD Jacobian vs finite differences (mild-voltage state)")
function fd_column(res!, u, j; eps=1e-7)
    h = eps * max(abs(u[j]), 1.0)
    up = copy(u); up[j] += h
    Fp = zeros(length(u)); Fm = zeros(length(u))
    res!(Fp, up); up[j] -= 2h; res!(Fm, up)
    return (Fp .- Fm) ./ (2h)
end
maxrel = 0.0
for j in (1, 2, 3, 7, 200, 397, 400)
    fdc = fd_column(res1!, u1, j)
    adc = Jd[:, j]
    sc  = max(maximum(abs, adc), 1e-30)
    rel = maximum(abs, fdc .- adc) / sc
    global maxrel = max(maxrel, rel)
    @printf("    col %3d: max|FD-AD|/max|AD| = %.3e\n", j, rel)
end
@printf("    PASS (< 1e-5): %s\n", maxrel < 1e-5)

# ── Step 2: Stage-1 anchor from the equilibrium guess ────────────────────────
println("\n[2] Stage-1 anchor at -0.300 V RHE from the exact equilibrium guess")
F0 = zeros(N); full_residual!(F0, u1, -0.300, ctx1)
@printf("    ||F||inf at the constructed guess = %.3e  (§9.6 expects ~2e-3)\n", maximum(abs, F0))
plu2 = AffineLU()
u1b = copy(u1)
info = newton_affine!(u1b, res1!; ly = ctx1.ly, plu = plu2, rtol = RTOL_PROD,
                      max_iter = MAXIT_ANCHOR, alpha_init = ALPHA_ANCHOR,
                      Jdense = Jd, cfg = cfg, verbose = true)
@printf("    converged=%s iters=%d ||F||inf=%.3e err_est=%.3e theta=%.3e reason=%s\n",
        info.converged, info.iterations, info.residual, info.err_est, info.theta, info.reason)
@printf("    PASS (<= ~6 iters, ||F||inf <= 1e-9): %s\n",
        info.converged && info.iterations <= 8 && info.residual <= 1e-9)

# ── Step 3: dF/dV structure probe ────────────────────────────────────────────
println("\n[3] dF/dV structure probe at the converged state")
function dFdV_nnz(ctx, u, V; h = 1e-6, tol = 1e-14)
    n = ctx.ly.N_TOT; Fp = zeros(n); Fm = zeros(n)
    full_residual!(Fp, u, V + h, ctx); full_residual!(Fm, u, V - h, ctx)
    d = (Fp .- Fm) ./ (2h)
    return count(x -> abs(x) > tol * max(1.0, maximum(abs, d)), d), n
end
nz1, n1 = dFdV_nnz(ctx1, u1b, -0.300)
@printf("    Stage 1: %d / %d nonzero  (guide: ~160/400)\n", nz1, n1)

ctx5 = make_ctx(5)
u5 = make_anchor_guess(ctx5, V_RHE_to_SHE_offset)     # V_c = 0
nz5, n5 = dFdV_nnz(ctx5, u5, V_RHE_to_SHE_offset)
@printf("    Stage 5: %d / %d nonzero  (guide: exactly 1)\n", nz5, n5)
@printf("    PASS: %s\n", nz5 == 1)

# ── Buffer / electroneutrality sanity (§12.1) ────────────────────────────────
println("\n[extra] §12.1 equilibrium sanity at the converged Stage-1 anchor")
c = unpack_log_conc(u1b, ctx1.ly)
Sb = [maximum(abs, buffer_sources(c[:, ix])) for ix in 1:ctx1.ly.N_CL]
@printf("    max|S_buf| over cells at the anchor = %.3e mol/m3/s\n", maximum(Sb))
@printf("    pH range across CL: %.3f .. %.3f\n",
        minimum(-log10.(c[1,:]./1000)), maximum(-log10.(c[1,:]./1000)))
@printf("    c_CO2 range: %.4f .. %.4f mol/m3\n", minimum(c[5,:]), maximum(c[5,:]))
iCO, iH2, itot = compute_currents(u1b, -0.300, ctx1)
@printf("    i_CO = %.5f, i_H2 = %.4f, i_tot = %.4f mA/cm2\n", iCO/10, iH2/10, itot/10)
println("\n" * "="^78)
