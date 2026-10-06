# sanity.jl — §9.6 equilibrium check + §9.5.9 first-build sanity tests
include("model.jl")
using Printf, LinearAlgebra, SparseArrays

println("="^72)
println("§9.6 Equilibrium sanity check")
println("="^72)
eq = solve_buffer_equilibrium(1000.0, 1000.0)
@printf("pH      = %.4f  (expect ~8.23)\n", eq.pH)
@printf("c_H     = %.4e mol/m3\n", eq.H)
@printf("c_OH    = %.4e mol/m3\n", eq.OH)
@printf("c_HCO3  = %.4f mol/m3\n", eq.HCO3)
@printf("c_CO3   = %.4f mol/m3\n", eq.CO3)
@printf("c_CO2   = %.4f mol/m3\n", eq.CO2)
@printf("K_w check : c_H*c_OH = %.4e vs K_w = %.4e\n", eq.H*eq.OH, K_w)
@printf("K_2 check : %.4e vs %.4e\n", eq.H*eq.HCO3/eq.CO2, K_2)
@printf("K_3 check : %.4e vs %.4e\n", eq.H*eq.CO3/eq.HCO3, K_3)
@printf("charge bal: %.3e\n", eq.H + 1000 - eq.OH - eq.HCO3 - 2*eq.CO3)
cv = [eq.H, eq.OH, eq.HCO3, eq.CO3, eq.CO2]
S = buffer_sources(cv)
@printf("max|S_buf| = %.3e mol/m3/s  (must be < 1e-6)\n", maximum(abs.(S)))
@assert maximum(abs.(S)) < 1e-6

println()
println("="^72)
println("Derived parameter audit")
println("="^72)
@printf("V_RHE_to_SHE_offset = %.6f V (expect 0.51305)\n", V_RHE_to_SHE_offset)
@printf("a_v_CL   = %.4e 1/m   (expect ~1.54e7)\n", a_v_CL)
@printf("S_pore   = %.4f        (expect ~0.558)\n", S_pore)
@printf("a_v_GL   = %.4e 1/m   (expect ~4.08e7)\n", a_v_GL)
@printf("D_eff    = %s\n", string(round.(D_eff, sigdigits=4)))
@printf("D_eff_Cs = %.4e m2/s  (expect ~1.13e-10)\n", D_eff_Cs)
@printf("Re       = %.1f        (expect ~14648)\n", Re_elec)
@printf("k_MT     = %s\n", string(round.(k_MT, sigdigits=4)))
@printf("k_MT_OH  = %.4e m/s   (expect ~1.45e-4)\n", k_MT[2])
@printf("H_vantHoff = %.4f mol/(m3 atm) (expect ~18.24)\n", H_vantHoff)
@printf("c_CO2_bc_s12 = %.4f mol/m3 (expect ~16.0)\n", c_CO2_bc_s12)
@printf("omega_inlet = %s (expect ~[0.947,0,0,0.053])\n", string(round.(omega_inlet, sigdigits=4)))
@printf("M_avg_inlet = %.4f g/mol (expect ~40.89)\n", M_avg_inlet*1000)
@printf("sigma_s_eff_GDL = %.2f S/m (expect ~73)\n", sigma_s_eff_GDL)
@printf("sigma_s_eff_CL  = %.2f S/m (expect ~11)\n", sigma_s_eff_CL)
@printf("kappa_CL = %.4e m2 (expect ~7.0e-17)\n", kappa_CL)
@printf("D_bin_raw[CO2,H2O] = %.4e m2/s\n", D_bin_raw[1,4])
# MHC pre-factor check
pf = a_v_CL * 540.0 / (n_e * F_const * c_ref)
@printf("MHC pf @ i0=540 : %.3f 1/s (expect ~43.1)\n", pf)
# Sechenov at bulk equilibrium
Hb = sechenov_H_CO2(cv)
@printf("Sechenov H_eff at bulk eq: %.3f mol/(m3 atm) (expect ~14.5)\n", Hb)
@printf("  -> c_CO2 at GDL face: %.3f mol/m3 (expect ~12.8)\n", Hb*0.88)
@printf("  c_Cs from electroneutrality at bulk eq: %.1f mol/m3 (expect 570)\n",
        cCs_from_electroneutrality(cv[1],cv[2],cv[3],cv[4]))

println()
println("="^72)
println("Mesh check")
println("="^72)
N_CL = N_CL_DEFAULT; N_GDL = N_GDL_DEFAULT
dm = make_domain(N_CL, N_GDL)
@printf("CL : N=%d, sum(dx)=%.6e (L_CL=%.6e), dx[1]/dx[N]=%.4f\n",
        dm.cl.N, sum(dm.cl.dx), L_CL, dm.cl.dx[1]/dm.cl.dx[end])
@printf("GDL: N=%d, sum(dx)=%.6e (L_GDL=%.6e), dx[1]/dx[N]=%.4f\n",
        dm.gdl.N, sum(dm.gdl.dx), L_GDL, dm.gdl.dx[1]/dm.gdl.dx[end])
@printf("CL x_faces: [%.3e, %.3e]  GDL x_faces: [%.3e, %.3e]\n",
        dm.cl.x_faces[1], dm.cl.x_faces[end], dm.gdl.x_faces[1], dm.gdl.x_faces[end])
@printf("Biot numbers k_MT*L_CL/D_eff = %s\n",
        string(round.([k_MT[i]*L_CL/D_eff[i] for i in 1:5], sigdigits=4)))

println()
println("="^72)
println("§9.5.9 Test 1 — persistent LU")
println("="^72)
ly1 = Layout(1, N_CL, N_GDL)
ctx1 = make_ctx(ly1, dm)
u = make_u0(ly1, dm)
ctx1.Vref[] = -0.300
ctx1.jacfun!(ctx1.Jdense, u)
Jsp = sparse(ctx1.Jdense)
b = randn(ly1.N_TOT)
t1 = @elapsed (x1, ok1) = aff_factor_solve!(ctx1.plu, Jsp, copy(b))
t2 = @elapsed (x2, ok2) = aff_factor_solve!(ctx1.plu, Jsp, copy(b))
@printf("first  factor+solve: %.4f s (ok=%s)\n", t1, ok1)
@printf("second factor+solve: %.4f s (ok=%s), same answer: %s\n", t2, ok2, x1 ≈ x2)
xb = aff_resolve(ctx1.plu, b)
@printf("back-substitution residual ||Jx-b||inf = %.3e\n", maximum(abs.(Jsp*xb .- b)))

println()
println("="^72)
println("AD vs FD Jacobian spot check (Stage 1, mild V)")
println("="^72)
F0 = zeros(ly1.N_TOT); ctx1.res_cl!(F0, u)
@printf("||F||inf at the equilibrium guess, V_RHE=-0.300: %.4e\n", maximum(abs.(F0)))
# finite-difference a few random columns
Fp = similar(F0); up = copy(u)
maxrel = Ref(0.0)
for j in [1, 7, 23, 5*N_CL-2, 5*N_CL]
    h = 1e-7*max(abs(u[j]),1.0)
    up[j] = u[j]+h; ctx1.res_cl!(Fp, up); up[j] = u[j]
    col_fd = (Fp .- F0)./h
    col_ad = ctx1.Jdense[:, j]
    den = max(maximum(abs.(col_ad)), 1e-30)
    rel = maximum(abs.(col_fd .- col_ad))/den
    maxrel[] = max(maxrel[], rel)
    @printf("  col %5d: max rel diff = %.3e\n", j, rel)
end
@printf("worst AD-vs-FD column disagreement: %.3e\n", maxrel[])

println()
println("="^72)
println("§9.5.9 Test 2 — Stage-1 anchor at -0.300 V RHE")
println("="^72)
t = @elapsed info = solve_at!(ctx1, u, -0.300; alpha_init=ALPHA_ANCHOR, max_iter=MAX_ITER_ANCHOR)
@printf("converged=%s iters=%d ||F||inf=%.3e err=%.3e (%.3f s)\n",
        info.converged, info.iterations, info.residual, info.err_est, t)

println()
println("="^72)
println("§9.5.9 Test 3 — dF/dV structure probe")
println("="^72)
for (st, nm) in [(1,"Stage 1")]
    Fp = zeros(ly1.N_TOT); Fm = zeros(ly1.N_TOT)
    ctx1.res_at_V!(Fp, u, -0.300+1e-6); ctx1.res_at_V!(Fm, u, -0.300-1e-6)
    d = (Fp .- Fm)./2e-6
    @printf("%s: %d / %d nonzero entries in dF/dV\n", nm, count(abs.(d) .> 1e-14), ly1.N_TOT)
end
println("\nSANITY SCRIPT COMPLETE")
