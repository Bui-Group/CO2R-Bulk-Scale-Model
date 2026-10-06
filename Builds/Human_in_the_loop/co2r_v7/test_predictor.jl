# §9.5.9 step 4 — predictor A/B: tangent vs carry-forward (+ φ_s shift)
include("co2r_core.jl")
using Printf, LinearAlgebra

STG   = parse(Int, get(ENV, "AB_STAGE", "1"))
N_CL  = N_CL_DEFAULT; N_GDL = N_GDL_DEFAULT
cl = make_cl_mesh(N_CL); gdl = make_gdl_mesh(N_GDL); gg = build_gas_geom(gdl, cl)
c_eq, _ = equilibrium_vector()
ly = make_layout(STG, N_CL, N_GDL); N = ly.N_TOT
plu = AffineLU(); Jd = zeros(N, N); Vref = Ref(V_RHE_grid[1])
res!      = (F_,u_)     -> full_residual!(F_,u_,Vref[],cl,gdl,c_eq,ly,gg,Val(STG))
res_at_V! = (F_,u_,V_)  -> full_residual!(F_,u_,V_,   cl,gdl,c_eq,ly,gg,Val(STG))
cfg = ForwardDiff.JacobianConfig(res!, zeros(N), zeros(N), ForwardDiff.Chunk{12}())

u0 = STG <= 2 ? make_u0_liquid(c_eq, ly) :
     (STG <= 4 ? make_u0_gasliquid(c_eq, ly) : make_u0_stage5_anchor(c_eq, ly, V_RHE_to_SHE_offset))
u = copy(u0); Vref[] = V_RHE_grid[1]
newton_affine!(u, res!; ly=ly, plu=plu, rtol=RTOL_PROD, max_iter=MAXIT_ANCHOR,
               alpha_init=ALPHA_ANCHOR, n_log_dofs=5*N_CL, Jdense=Jd, cfg=cfg)

dudV = zeros(N); Fp=zeros(N); Fm=zeros(N)
ok = tangent_predictor!(dudV, plu, res_at_V!, u, V_RHE_grid[1]; Fp=Fp, Fm=Fm)
ratios = Float64[]
@printf("Stage %d predictor A/B over the %d-point grid (dV = %.3f V)\n",
        STG, length(V_RHE_grid), -DV_STEP)
for k in 2:length(V_RHE_grid)
    global ok, u, dudV
    Vt = V_RHE_grid[k]; dV = Vt - V_RHE_grid[k-1]
    u_cf = copy(u)                                  # carry-forward + φ_s shift
    if STG == 5
        for ig in 1:ly.N_GDL; u_cf[phi_s_gdl_idx5(ly,ig)] += dV; end
        for ix in 1:ly.N_CL;  u_cf[phi_s_cl_idx5(ly,ix)]  += dV; end
    end
    u_tan = copy(u); ok && (u_tan .+= dV .* dudV)    # tangent
    u_star = copy(u_tan); Vref[] = Vt
    newton_affine!(u_star, res!; ly=ly, plu=plu, rtol=1e-12, max_iter=MAXIT_SWEEP,
                   alpha_init=ALPHA_SWEEP, n_log_dofs=5*N_CL, Jdense=Jd, cfg=cfg)
    e_cf = norm(u_cf .- u_star); e_tan = norm(u_tan .- u_star)
    push!(ratios, e_cf / max(e_tan, 1e-300))
    u .= u_star
    ok = tangent_predictor!(dudV, plu, res_at_V!, u, Vt; Fp=Fp, Fm=Fm)
end
sort!(ratios)
@printf("improvement (carry-forward err / tangent err):\n")
@printf("  min = %.1fx   median = %.1fx   max = %.1fx\n",
        ratios[1], ratios[div(end,2)+1], ratios[end])
@printf("  tangent better at EVERY step: %s\n", all(ratios .> 1.0))
@printf("  PASS (all steps better, median > 10x): %s\n",
        all(ratios .> 1.0) && ratios[div(end,2)+1] > 10.0)
