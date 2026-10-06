# sanity_tests.jl — §9.5.9 first-build sanity tests (steps 1–4)
include("common.jl")

m = build_model(1)
ly = m.ly; N = ly.N_TOT
res_at_V!(F,u,V) = residual_stage12!(F,u,V,m.cl,m.c_eq,true)

println("── Step 1: persistent LU reuses the symbolic analysis ──")
u = u0_stage(ly, m.c_eq)
Vref = Ref(-0.300); res!(F_,u_) = res_at_V!(F_,u_,Vref[])
Jd = zeros(N,N); cfg = ForwardDiff.JacobianConfig(res!, zeros(N), u, ForwardDiff.Chunk{12}())
ybuf = zeros(N); ForwardDiff.jacobian!(Jd, res!, ybuf, u, cfg)
Js = sparse(Jd); plu = AffineLU(); b = randn(N)
t1 = @elapsed aff_factor_solve!(plu, Js, b)
t2 = @elapsed aff_factor_solve!(plu, Js, b)
t3 = @elapsed aff_resolve(plu, b)
@printf("  1st factor %.4f ms | 2nd (lu! reuse) %.4f ms | back-sub only %.4f ms\n",
        t1*1e3, t2*1e3, t3*1e3)
@printf("  symbolic analyses = %d (must be 1), numeric = %d  -> %s\n",
        plu.n_symbolic, plu.n_numeric, plu.n_symbolic == 1 ? "PASS" : "FAIL")

println("\n── Step 2: Stage-1 anchor at −0.300 V from the equilibrium guess ──")
u = u0_stage(ly, m.c_eq)
plu2 = AffineLU()
info = newton_affine!(u, res!; blocks=weight_blocks(ly), plu=plu2, rtol=1e-8,
                      max_iter=200, alpha_init=1e-2, n_log_dofs=5*m.cl.N)
Fchk = zeros(N); res_at_V!(Fchk, u, -0.300)
@printf("  iters = %d (need ≤ ~6), ||F||inf = %.3e (need ≤ 1e-9) -> %s\n",
        info.iterations, maximum(abs,Fchk),
        (info.iterations <= 8 && maximum(abs,Fchk) <= 1e-9) ? "PASS" : "FAIL")
F0 = zeros(N); res_at_V!(F0, u0_stage(ly,m.c_eq), -0.300)
@printf("  ||F||inf at the equilibrium guess = %.3e (guide: ~2e-3)\n", maximum(abs,F0))

println("\n── AD validation: FD spot-check of the Jacobian at the converged state ──")
ForwardDiff.jacobian!(Jd, res!, ybuf, u, cfg)
maxrel = 0.0
for j in (1, 7, 200, 397)
    h = 1e-7*max(abs(u[j]),1.0)
    up = copy(u); up[j] += h; Fp = zeros(N); res_at_V!(Fp, up, -0.300)
    um = copy(u); um[j] -= h; Fm = zeros(N); res_at_V!(Fm, um, -0.300)
    col = (Fp .- Fm) ./ (2h)
    sc = maximum(abs, col); sc == 0 && continue
    rel = maximum(abs, col .- Jd[:,j]) / sc
    global maxrel = max(maxrel, rel)
    @printf("  col %3d: max|FD − AD|/scale = %.3e\n", j, rel)
end
@printf("  -> %s\n", maxrel < 1e-5 ? "PASS" : "FAIL")

println("\n── Step 3: ∂F/∂V structure probe (Stages 1–2: expect ~160/400 nonzero) ──")
h = 1e-6
Fp = zeros(N); Fm = zeros(N)
res_at_V!(Fp, u, -0.300+h); res_at_V!(Fm, u, -0.300-h)
dFdV = (Fp .- Fm)./(2h)
@printf("  nonzeros = %d / %d\n", count(x->abs(x)>1e-14, dFdV), N)

println("\n── Step 4: predictor A/B (tangent vs carry-forward) over the sweep ──")
plu3 = AffineLU(); u = u0_stage(ly, m.c_eq)
Vref[] = V_RHE_grid[1]
newton_affine!(u, res!; blocks=weight_blocks(ly), plu=plu3, rtol=1e-8, max_iter=200,
               alpha_init=1e-2, n_log_dofs=5*m.cl.N, Jdense=Jd, cfg=cfg)
dudV = zeros(N); ratios = Float64[]
ok = tangent_predictor!(dudV, plu3, res_at_V!, u, V_RHE_grid[1])
for k in 2:length(V_RHE_grid)
    dV = V_RHE_grid[k] - V_RHE_grid[k-1]
    u_carry = copy(u)                       # carry-forward predictor
    u_tan   = u .+ dV .* dudV               # tangent predictor
    u_star  = copy(u_tan)
    Vref[] = V_RHE_grid[k]
    newton_affine!(u_star, res!; blocks=weight_blocks(ly), plu=plu3, rtol=1e-12,
                   max_iter=40, alpha_init=1.0, n_log_dofs=5*m.cl.N, Jdense=Jd, cfg=cfg)
    e_carry = norm(u_carry .- u_star); e_tan = norm(u_tan .- u_star)
    e_tan > 0 && push!(ratios, e_carry/e_tan)
    u .= u_star
    ok = tangent_predictor!(dudV, plu3, res_at_V!, u, V_RHE_grid[k])
end
sort!(ratios)
@printf("  improvement ratio (carry-forward err / tangent err): median %.1f×, min %.1f×, max %.1f×\n",
        ratios[div(end,2)], ratios[1], ratios[end])
@printf("  tangent better at every step: %s ; median > 10×: %s\n",
        all(ratios .> 1.0) ? "yes (PASS)" : "no (FAIL)",
        ratios[div(end,2)] > 10 ? "yes (PASS)" : "no")

println("\n── Stage-1 check 4 detail: currents at V_RHE = −0.300 ──")
u1 = load_solution(joinpath(@__DIR__,"output_stage1","cache"), "s1_", -0.300)
if u1 !== nothing
    iCO,iH2 = compute_currents(u1, -0.300 - V_RHE_to_SHE_offset, m.cl)
    @printf("  i_CO = %.5f mA/cm² (guide ~0.002), i_H2 = %.3f mA/cm² (guide ~10)\n", iCO/10, iH2/10)
end
