include("co2r_core.jl"); using Printf, Optim
N_CL=N_CL_DEFAULT; N_GDL=N_GDL_DEFAULT
cl=make_cl_mesh(N_CL); gdl=make_gdl_mesh(N_GDL); gg=build_gas_geom(gdl,cl)
ly=make_layout(4,N_CL,N_GDL); N=ly.N_TOT; c_eq,_=equilibrium_vector()
plu=AffineLU(); Jd=zeros(N,N); Vref=Ref(-0.816)
res_f! = (F_,u_) -> full_residual!(F_,u_,Vref[],cl,gdl,c_eq,ly,gg,Val(4))
cfg=ForwardDiff.JacobianConfig(res_f!, zeros(N), zeros(N), ForwardDiff.Chunk{12}())
S4=joinpath(@__DIR__,"output_stage4","cache"); Vs=list_cache(S4,"s4")
function polish(seed)
    out=Dict{Float64,Vector{Float64}}()
    for V in E_exp_RHE
        u=copy(seed[V]); Vref[]=V
        i=newton_affine!(u,res_f!;ly=ly,plu=plu,rtol=RTOL_PROD,max_iter=40,alpha_init=1.0,
                         n_log_dofs=5*N_CL,Jdense=Jd,cfg=cfg)
        i.converged || return (false,seed)
        out[V]=u
    end
    (true,out)
end
cache = Dict(V=>copy(load_solution(S4,"s4",Vs[argmin(abs.(Vs.-V))])) for V in E_exp_RHE)
ok,cache = polish(cache)
NE=Ref(0)
function obj(p)
    (any(p.<[10.0,0.5,8.0,0.01]) || any(p.>[5000.0,3.0,18.0,0.50])) && return 1e6
    NE[]+=1
    set_i0_CO2R!(p[1]); set_lambda_MHC!(p[2]); set_HER_prefactor!(10.0^p[3]); set_alpha_c_HER!(p[4])
    ok,c2=polish(cache); ok || return 1e6
    global cache=c2
    s=0.0
    for (k,V) in enumerate(E_exp_RHE)
        a,b,_=compute_currents(c2[V],V,cl,ly)
        s+=(log10(max(a/10,0.1))-log10(i_CO_exp[k]))^2+(log10(max(b/10,0.1))-log10(i_H2_exp[k]))^2
    end
    s
end
function obj_pen(p, PEN)
    pen = 0.0
    for j in 1:4
        lo,hi = [10.0,0.5,8.0,0.01][j], [5000.0,3.0,18.0,0.50][j]
        sc = hi-lo
        p[j] < lo && (pen += ((lo-p[j])/sc)^2)
        p[j] > hi && (pen += ((p[j]-hi)/sc)^2)
    end
    pen > 0 && return PEN*(1+pen)
    v = obj(p)
    v >= 1e6 ? PEN : v
end
p2=[109.796,1.6065,log10(2.409e13),0.0872]
@printf("obj at Step-3 start = %.6f\n", obj(p2))
@printf("simplex vertex v1 (log10 pf -> %.4f) = %.6g\n", p2[3]*1.5+0.025, obj([p2[1],p2[2],p2[3]*1.5+0.025,p2[4]]))
@printf("simplex vertex v2 (alpha_c -> %.4f)  = %.6g\n", p2[4]*1.5+0.025, obj([p2[1],p2[2],p2[3],p2[4]*1.5+0.025]))
NE[]=0
r=optimize(q->obj([p2[1],p2[2],q[1],q[2]]), [p2[3],p2[4]], NelderMead(),
           Optim.Options(iterations=200, g_abstol=1e-6, f_reltol=1e-6, store_trace=true, show_trace=false))
@printf("\nNM result: minimizer=%s  minimum=%.6g  iters=%d  f_calls=%d  my_evals=%d\n",
        Optim.minimizer(r), Optim.minimum(r), Optim.iterations(r), Optim.f_calls(r), NE[])
@printf("converged=%s  reason: x_conv=%s f_conv=%s g_conv=%s\n", Optim.converged(r),
        Optim.x_converged(r), Optim.f_converged(r), Optim.g_converged(r))
tr=Optim.trace(r); @printf("trace length=%d, first f=%.6g last f=%.6g\n",
        length(tr), tr[1].value, tr[end].value)
for PEN in (1e6, 1e3, 10.0, 1.0)
    global cache
    NE[]=0
    rr=optimize(q->obj_pen([p2[1],p2[2],q[1],q[2]], PEN), [p2[3],p2[4]], NelderMead(),
                Optim.Options(iterations=200, g_abstol=1e-6, f_reltol=1e-6))
    @printf("PEN=%-8.0e  min=%-12.6g  minimizer=[%.4f, %.5f]  iters=%3d  evals=%3d  gconv=%s\n",
            PEN, Optim.minimum(rr), Optim.minimizer(rr)[1], Optim.minimizer(rr)[2],
            Optim.iterations(rr), NE[], Optim.g_converged(rr))
end
