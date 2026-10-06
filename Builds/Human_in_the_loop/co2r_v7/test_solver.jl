# §9.5.9 first-build sanity tests (steps 1-4) + AD validation
include("co2r_core.jl")
using Printf, SparseArrays, LinearAlgebra

N_CL = parse(Int, get(ENV,"MESH_N_CL","20")); N_GDL = parse(Int, get(ENV,"MESH_N_GDL","30"))
cl = make_cl_mesh(N_CL); gdl = make_gdl_mesh(N_GDL); gg = build_gas_geom(gdl, cl)
c_eq, eq = equilibrium_vector()

println("="^70); println("TEST 0 — AD vs FD Jacobian spot check (Stage 1, mild V)"); println("="^70)
ly1 = make_layout(1, N_CL, N_GDL); N1 = ly1.N_TOT
u = make_u0_liquid(c_eq, ly1)
res1! = (F_,u_) -> full_residual!(F_,u_,-0.300,cl,gdl,c_eq,ly1,gg,Val(1))
Jad = zeros(N1,N1); F0 = zeros(N1)
cfg1 = ForwardDiff.JacobianConfig(res1!, zeros(N1), u, ForwardDiff.Chunk{12}())
ForwardDiff.jacobian!(Jad, res1!, F0, u, cfg1)
Fv = zeros(N1); res1!(Fv, u)
@printf("jacobian! filled y correctly: %s\n", isapprox(F0, Fv; atol=0, rtol=0))
Jfd = zeros(N1,N1); Fp = zeros(N1); up = copy(u)
for j in 1:N1
    h = 1e-7*max(abs(u[j]),1.0); up[j]=u[j]+h; res1!(Fp,up); @. Jfd[:,j]=(Fp-Fv)/h; up[j]=u[j]
end
rel = norm(Jad-Jfd)/max(norm(Jad),eps())
@printf("||J_AD - J_FD||/||J_AD|| = %.3e   (expect ~1e-6, FD-limited)\n", rel)

println("\n"*"="^70); println("TEST 1 — persistent LU: symbolic reuse + live back-substitution"); println("="^70)
plu = AffineLU(); Js = sparse(Jad); b = randn(N1)
t1 = @elapsed (x1, ok1) = aff_factor_solve!(plu, Js, b)
t2 = @elapsed (x2, ok2) = aff_factor_solve!(plu, Js, b)
@printf("first  factor+solve: %.4f s (ok=%s)\n", t1, ok1)
@printf("second factor+solve: %.4f s (ok=%s, symbolic reused)\n", t2, ok2)
@printf("solutions identical : %s ; residual ||Jx-b||inf = %.3e\n",
        x1==x2, maximum(abs, Js*x1 - b))
xb = aff_resolve(plu, b)
@printf("aff_resolve matches (no refactorization): %s\n", xb == x1)
# pattern-change guard
Jbad = copy(Js); Jbad[1, N1] = 1.0; dropzeros!(Jbad)
(x3, ok3) = aff_factor_solve!(plu, Jbad, b)
@printf("pattern change handled: ok=%s, ||Jx-b||inf = %.3e\n", ok3, maximum(abs, Jbad*x3 - b))

println("\n"*"="^70); println("TEST 2 — Stage-1 anchor at −0.300 V RHE from the equilibrium guess"); println("="^70)
plu1 = AffineLU(); u1 = make_u0_liquid(c_eq, ly1)
info = newton_affine!(u1, res1!; ly=ly1, plu=plu1, rtol=RTOL_PROD,
                      max_iter=MAXIT_ANCHOR, alpha_init=ALPHA_ANCHOR,
                      n_log_dofs=5*N_CL, Jdense=zeros(N1,N1), cfg=cfg1)
@printf("converged=%s  iters=%d  |F|inf=%.3e  err_est=%.3e  theta=%.3e  reason=%s\n",
        info.converged, info.iterations, info.residual, info.err_est, info.theta, info.reason)
@printf("PASS: iters<=6 and |F|inf<=1e-9 -> %s\n",
        info.converged && info.iterations<=6 && info.residual<=1e-9)
iCO,iH2,it = compute_currents(u1, -0.300, cl, ly1)
@printf("i_CO=%.4f  i_H2=%.4f  i_tot=%.4f mA/cm2\n", iCO/10, iH2/10, it/10)

println("\n"*"="^70); println("TEST 3 — dF/dV structure probe at a converged state"); println("="^70)
function dFdV_nnz(::Val{S}, ly, u, V) where S
    n = ly.N_TOT; Fp=zeros(n); Fm=zeros(n); h=1e-6
    full_residual!(Fp,u,V+h,cl,gdl,c_eq,ly,gg,Val(S))
    full_residual!(Fm,u,V-h,cl,gdl,c_eq,ly,gg,Val(S))
    d = (Fp .- Fm)./(2h)
    return count(x->abs(x)>1e-14*max(1.0,maximum(abs,d)), d), n
end
nz,n = dFdV_nnz(Val(1), ly1, u1, -0.300); @printf("Stage 1: %d / %d nonzero  (guide: ~%d/%d)\n", nz,n, 2*N_CL, 5*N_CL)
for (S, mk) in ((3, make_u0_gasliquid), (4, make_u0_gasliquid))
    lyS = make_layout(S, N_CL, N_GDL); uS = mk(c_eq, lyS)
    resS! = (F_,u_) -> full_residual!(F_,u_,-0.300,cl,gdl,c_eq,lyS,gg,Val(S))
    pS = AffineLU()
    iS = newton_affine!(uS, resS!; ly=lyS, plu=pS, rtol=RTOL_PROD, max_iter=MAXIT_ANCHOR,
                        alpha_init=ALPHA_ANCHOR, n_log_dofs=5*N_CL)
    nzS,nS = dFdV_nnz(Val(S), lyS, uS, -0.300)
    @printf("Stage %d: anchor conv=%s it=%d |F|=%.2e ; dF/dV %d / %d nonzero\n",
            S, iS.converged, iS.iterations, iS.residual, nzS, nS)
end
ly5 = make_layout(5, N_CL, N_GDL)
u5 = make_u0_stage5_anchor(c_eq, ly5, V_RHE_to_SHE_offset)
res5! = (F_,u_) -> full_residual!(F_,u_,V_RHE_to_SHE_offset,cl,gdl,c_eq,ly5,gg,Val(5))
p5 = AffineLU()
i5 = newton_affine!(u5, res5!; ly=ly5, plu=p5, rtol=RTOL_PROD, max_iter=MAXIT_ANCHOR,
                    alpha_init=ALPHA_ANCHOR, n_log_dofs=5*N_CL)
@printf("Stage 5 anchor (V_c=0): conv=%s it=%d |F|inf=%.3e err=%.2e reason=%s\n",
        i5.converged, i5.iterations, i5.residual, i5.err_est, i5.reason)
nz5,n5 = dFdV_nnz(Val(5), ly5, u5, V_RHE_to_SHE_offset)
@printf("Stage 5: dF/dV %d / %d nonzero  (guide: exactly 1)\n", nz5, n5)
