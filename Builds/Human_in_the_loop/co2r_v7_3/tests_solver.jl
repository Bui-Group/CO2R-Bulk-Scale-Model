# §9.5.9 first-build sanity tests for the solver stack (steps 1, 3, 4)
include("model.jl")
using Printf, SparseArrays

eq = solve_buffer_equilibrium(1000.0, 1000.0)
c_eq = [eq.H, eq.OH, eq.HCO3, eq.CO3, eq.CO2]
ctx = Ctx(1, N_CL_DEFAULT, N_GDL_DEFAULT, c_eq)
u = make_u0(ctx); N = ctx.ly.N_TOT
ws = SolveWS(ctx, u)

println("="^72); println("Step 1 — persistent LU")
ws.rf.V = -0.300
F = zeros(N); ws.rf(F, u)
ForwardDiff.jacobian!(ws.Jd, ws.rf, zeros(N), u, ws.cfg)
J = sparse(ws.Jd)
t1 = @elapsed (x1, ok1) = aff_factor_solve!(ws.plu, J, -F)
t2 = @elapsed (x2, ok2) = aff_factor_solve!(ws.plu, J, -F)
@printf("first (symbolic+numeric) %.4f ms; second (lu! refresh) %.4f ms; speedup %.2fx\n",
        t1*1e3, t2*1e3, t1/t2)
@printf("identical solutions: %s ; nnz = %d (%.3f %% dense)\n",
        string(maximum(abs, x1 .- x2) == 0.0), nnz(J), 100*nnz(J)/N^2)
b = randn(N); xb = aff_resolve(ws.plu, b)
@printf("back-substitution against live factorization: resid = %.3e\n", maximum(abs, J*xb - b))

println("\n"*"="^72); println("Step: AD vs FD Jacobian spot check (§9.5.8)")
u_mild = load_solution(-0.800, cache_dir(1), cache_prefix(1))
ws.rf.V = -0.800
ForwardDiff.jacobian!(ws.Jd, ws.rf, zeros(N), u_mild, ws.cfg)
Jad = copy(ws.Jd)
F0 = zeros(N); ws.rf(F0, u_mild)
Jfd = zeros(N, N); up = copy(u_mild); Fp = zeros(N)
for j in 1:N
    h = 1e-7 * max(abs(u_mild[j]), 1.0)
    up[j] = u_mild[j] + h; ws.rf(Fp, up)
    @. Jfd[:, j] = (Fp - F0) / h
    up[j] = u_mild[j]
end
sc = maximum(abs, Jad)
@printf("max|J_AD - J_FD| / max|J_AD| = %.3e  (FD truncation-level agreement)\n",
        maximum(abs, Jad .- Jfd) / sc)

println("\n"*"="^72); println("Step 3 — dF/dV structure probe")
function dFdV_nnz(stage)
    c2 = Ctx(stage, N_CL_DEFAULT, N_GDL_DEFAULT, c_eq)
    uu = make_u0(c2); n = c2.ly.N_TOT
    w2 = SolveWS(c2, uu)
    V = stage == 5 ? V_RHE_to_SHE_offset : -0.300
    info = solve_at!(w2, uu, V; alpha_init = 1e-2, max_iter = 200)
    Fp = zeros(n); Fm = zeros(n); h = 1e-6
    w2.rf.V = V + h; w2.rf(Fp, uu)
    w2.rf.V = V - h; w2.rf(Fm, uu)
    d = (Fp .- Fm) ./ (2h)
    (count(x -> abs(x) > 1e-12 * max(maximum(abs, d), 1e-300), d), n, info.converged, info.iterations)
end
for s in (1, 3, 5)
    nz, n, cv, it = dFdV_nnz(s)
    @printf("stage %d: %d nonzero / %d DOFs  (anchor converged=%s, %d iters)\n", s, nz, n, cv, it)
end

println("\n"*"="^72); println("Step 4 — predictor A/B over the Stage-1 sweep")
ctx1 = Ctx(1, N_CL_DEFAULT, N_GDL_DEFAULT, c_eq)
u = make_u0(ctx1); ws = SolveWS(ctx1, u); N = ctx1.ly.N_TOT
dudV = zeros(N); Fp = zeros(N); Fm = zeros(N)
solve_at!(ws, u, V_RHE_grid[1]; alpha_init = 1e-2, max_iter = 200)
tangent_predictor!(dudV, ws.plu, ws.rf, u, V_RHE_grid[1]; Fp = Fp, Fm = Fm)
ratios = Float64[]
for k in 2:length(V_RHE_grid)
    dV = V_RHE_grid[k] - V_RHE_grid[k-1]
    u_carry = copy(u)                       # carry-forward (+ φ_s shift; no-op at stage 1)
    u_tan   = u .+ dV .* dudV
    u_star  = copy(u_tan)
    solve_at!(ws, u_star, V_RHE_grid[k]; alpha_init = 1.0, max_iter = 40)
    e_c = maximum(abs, u_carry .- u_star)
    e_t = maximum(abs, u_tan   .- u_star)
    push!(ratios, e_c / max(e_t, 1e-300))
    copyto!(u, u_star)
    tangent_predictor!(dudV, ws.plu, ws.rf, u, V_RHE_grid[k]; Fp = Fp, Fm = Fm)
end
sort!(ratios)
@printf("tangent better at every step: %s\n", string(all(ratios .> 1.0)))
@printf("improvement ratio: min %.2fx  median %.2fx  max %.2fx\n",
        ratios[1], ratios[div(end,2)+1], ratios[end])
