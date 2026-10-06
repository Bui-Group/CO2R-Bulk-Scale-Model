# Telescoping mass-balance check on the Stage 4 cache (§11 Stage 4 check 1):
#   does ρu(inlet) = −net_src (global gas mass conservation), and does it telescope?
# Stage 4 cache = default (pre-fit) kinetics, so we DON'T load fitted params.
include("co2r_core.jl")
using Printf

cl_mesh = make_cl_mesh(); gdl_mesh = make_gdl_mesh()
eq = solve_buffer_equilibrium(1000.0,1000.0); c_eq_vec=[eq.H,eq.OH,eq.HCO3,eq.CO3,eq.CO2]
NDOF = NDOF_GAS

# ρu (Darcy mass flux) at the inlet face, recomputed from the Darcy law (same as gasface(1))
function rhou_inlet(u)
    omL = (omega_inlet[1],omega_inlet[2],omega_inlet[3],omega_inlet[4]); pL = p_total
    omR = gdl_omega(u,1); pR = gdl_p(u,1)
    uG = -(kappa_GDL/mu_gas)*(pR-pL)/gdl_mesh.dx_between[1]
    omf = ntuple(j->0.5*(omL[j]+omR[j]),4); rhof = 0.5*(pL+pR)*Mn_from_omega(omf)/(R_gas*T)
    return rhof*uG
end

# net gas mass source ∫S_mass dx over the CL (S_mass = −R_PT M_CO2 + R_CO2R M_CO + R_HER M_H2)
function net_src(u, Vapp)
    c = unpack_log_conc(u, N_CL); s = 0.0
    for ix in 1:N_CL
        cH=c[1,ix]; cCO2=c[5,ix]
        Rco = R_CO2R(cCO2,Vapp,0.0,cH,a_v_CL,lambda_MHC(),i0_CO2R(),n_e)
        Rh2 = R_HER(Vapp,0.0,cH,a_v_CL,n_e)
        Rpt = a_v_GL*k_GL()*(H_eff_sechenov(c[1,ix],c[2,ix],c[3,ix],c[4,ix])*pCO2_cl(u,ix)-cCO2)
        s += (-Rpt*M_gas[1] + Rco*M_gas[2] + Rh2*M_gas[3])*cl_mesh.dx[ix]
    end
    return s
end

@printf("Kinetics (should be pre-fit): i0=%.1f λ=%.2f HER_pf=%.3e α_c=%.4f\n\n",
        i0_CO2R(), lambda_MHC(), HER_prefactor(), alpha_c_HER())
println("  V_RHE |  net_src     ρu_inlet    ΣF[p]      telescope_err | massbal_rel | max|F[p]|/cell")
println("-"^104)
for V in (-0.816, -1.205, -1.644, -2.079, -2.413, -2.500)
    res = load_nearest(4, V); res === nothing && continue
    u, Vn = res; Vapp = Vn - V_RHE_to_SHE_offset
    F = zeros(NDOF); residual_stage4!(F, u, Vn, cl_mesh, gdl_mesh, c_eq_vec)
    # gas pressure rows
    Fp = vcat([F[gas_gdl_idx(ig,4)] for ig in 1:N_GDL], [F[gas_cl_idx(ix,4)] for ix in 1:N_CL])
    sumFp = sum(Fp); maxFp = maximum(abs, Fp)
    ru = rhou_inlet(u); ns = net_src(u, Vapp)
    # telescoping identity: ΣF[p] == ρu_inlet + net_src  (algebraic, should match to ~machine eps)
    tele_err = abs(sumFp - (ru + ns))
    # global mass balance relative error = |ρu_inlet + net_src| / |net_src|
    massbal_rel = abs(ru + ns)/max(abs(ns),1e-30)
    @printf("  %.3f | %+.3e  %+.3e  %+.3e   %.2e | %9.2e%% | %.2e\n",
            Vn, ns, ru, sumFp, tele_err, 100*massbal_rel, maxFp)
end
