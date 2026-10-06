# =============================================================================
# run_stage3.jl — Stage 3 driver (BaseCase.md §11 Stage 3)
#   Stefan-Maxwell multicomponent gas transport (CO₂/CO/H₂/H₂O) + volumetric CO₂
#   phase transfer R_PT. u_G = 0 (Darcy off), pressure pinned to p_total.
#   DOFs = 1200 (5×80 liquid + 4×200 gas). Simplified kinetics still.
#   Anchor (§9.5.5): equilibrium liquid + inlet gas at −0.300 V; tangent continuation.
# =============================================================================
include("co2r_core.jl")
using Printf, SparseArrays

const STAGE = 3

println("="^70)
@printf("=== Stage %d : Stefan-Maxwell gas transport + R_PT phase transfer ===\n", STAGE)
println("="^70)
@printf("Mesh: N_CL=%d N_GDL=%d (q=%.2f), DOFs=%d\n", N_CL, N_GDL, MESH_Q, NDOF_GAS)
@printf("Kinetics (initial-guess): i0_CO2R=%.1f A/m², lambda_MHC=%.2f eV, HER_pf=%.3e, alpha_c=%.4f\n",
        i0_CO2R(), lambda_MHC(), HER_prefactor(), alpha_c_HER())
@printf("k_GL=%.1e m/s, a_v_GL=%.3e m⁻¹\n", k_GL(), a_v_GL)
@printf("FSG D_bin_raw [m²/s]: CO2-H2O=%.3e CO-H2O=%.3e H2-H2O=%.3e\n",
        D_bin_raw[1,4], D_bin_raw[2,4], D_bin_raw[3,4])
@printf("omega_inlet (mass frac) = [%.4f, %.4f, %.4f, %.4f]\n",
        omega_inlet[1], omega_inlet[2], omega_inlet[3], omega_inlet[4])
@assert !isempty(list_cache(2)) "No Stage 2 cache — run run_stage2.jl first (§13 driver order)."

eq = solve_buffer_equilibrium(1000.0, 1000.0)
c_eq_vec = [eq.H, eq.OH, eq.HCO3, eq.CO3, eq.CO2]

cl_mesh = make_cl_mesh(); gdl_mesh = make_gdl_mesh()
const NDOF = NDOF_GAS
rows, cols = build_gas_pattern(N_CL, N_GDL)
Jtemplate = pattern_matrix(rows, cols, NDOF)
plu = AffineLU()
blocks = dof_blocks(STAGE)

Vs = sort(collect(V_RHE_grid), by = v -> -v)
V_ref = Ref(Vs[1])
res_closure!(F, uu) = residual_stage3!(F, uu, V_ref[], cl_mesh, gdl_mesh, c_eq_vec)
assemble! = make_ad_assembler(res_closure!, NDOF; chunk=12)

# anchor guess (§9.5.5): equilibrium liquid + inlet gas composition and p_total in every gas cell
u = make_u0_stage3(repeat(log.(c_eq_vec), N_CL))
n_conv = Ref(0); maxF = Ref(0.0)
t_start = time()
sweep_log = continuation_sweep!(u, Vs, res_closure!, V_ref;
    blocks=blocks, plu=plu, jac_assembler=assemble!, J=Jtemplate, n_log_dofs=5*N_CL,
    on_converged = (V, uu, info) -> begin
        save_solution(STAGE, copy(uu), V)
        n_conv[] += 1; maxF[] = max(maxF[], info.residual)
    end)
n_iter = sum(l.iterations for l in sweep_log)
@printf("\nSweep done: %d/%d converged in %.1f s (%d Newton iterations, %.2f/step). max ||F||inf = %.2e\n",
        n_conv[], length(Vs), time()-t_start, n_iter, n_iter/length(Vs), maxF[])

# ---------------------------------------------------------------------------
# Diagnostics CSVs
# ---------------------------------------------------------------------------
outdir = joinpath(@__DIR__, "output_stage$(STAGE)"); mkpath(outdir)
cached_Vs = list_cache(STAGE)

open(joinpath(outdir, "polarization.csv"), "w") do f
    println(f, "V_RHE,V_SHE,i_CO_mAcm2,i_H2_mAcm2,i_tot_mAcm2,FE_CO,FE_H2")
    for V in cached_Vs
        uu = load_solution(STAGE, V); Vapp = V - V_RHE_to_SHE_offset
        iCO = compute_i_CO(uu, Vapp, cl_mesh)/10; iH2 = compute_i_H2(uu, Vapp, cl_mesh)/10; itot=iCO+iH2
        @printf(f, "%.6f,%.6f,%.6e,%.6e,%.6e,%.4f,%.4f\n", V, Vapp, iCO, iH2, itot,
                itot>0 ? 100*iCO/itot : 0.0, itot>0 ? 100*iH2/itot : 0.0)
    end
end
open(joinpath(outdir, "profiles.csv"), "w") do f
    println(f, "V_RHE,x_um,pH,c_CO2,c_OH,c_HCO3,c_CO3,c_Cs,c_H,R_CO2R,R_HER")
    for V in cached_Vs
        uu = load_solution(STAGE, V); Vapp = V - V_RHE_to_SHE_offset
        c = unpack_log_conc(uu, N_CL)
        for ix in 1:N_CL
            cH=c[1,ix]; cOH=c[2,ix]; cHCO3=c[3,ix]; cCO3=c[4,ix]; cCO2=c[5,ix]
            cCs = cOH + cHCO3 + 2*cCO3 - cH - C_fix_eff
            pH = -log10(max(cH/1000.0,1e-20))
            rco = R_CO2R(cCO2, Vapp, 0.0, cH, a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
            rh2 = R_HER(Vapp, 0.0, cH, a_v_CL, n_e)
            @printf(f, "%.6f,%.5f,%.5f,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                    V, cl_mesh.x_centers[ix]*1e6, pH, cCO2, cOH, cHCO3, cCO3, cCs, cH, rco, rh2)
        end
    end
end
# gas profiles over GDL+CL (B1)
open(joinpath(outdir, "gas_profiles.csv"), "w") do f
    println(f, "V_RHE,x_um,region,p_CO2_atm,p_CO_atm,p_H2_atm,p_total_atm")
    for V in cached_Vs
        uu = load_solution(STAGE, V)
        for ig in 1:N_GDL
            om = gdl_omega(uu, ig); Mn=Mn_from_omega(om); x=omega_to_x(om,Mn); p=gdl_p(uu,ig)
            @printf(f, "%.6f,%.5f,GDL,%.6e,%.6e,%.6e,%.6f\n",
                    V, gdl_mesh.x_centers[ig]*1e6, x[1]*p/101325, x[2]*p/101325, x[3]*p/101325, p/101325)
        end
        for ix in 1:N_CL
            om = cl_omega(uu, ix); Mn=Mn_from_omega(om); x=omega_to_x(om,Mn); p=cl_p(uu,ix)
            @printf(f, "%.6f,%.5f,CL,%.6e,%.6e,%.6e,%.6f\n",
                    V, cl_mesh.x_centers[ix]*1e6, x[1]*p/101325, x[2]*p/101325, x[3]*p/101325, p/101325)
        end
    end
end

# ---------------------------------------------------------------------------
# Stage 3 physicality checks (§11 Stage 3)
# ---------------------------------------------------------------------------
println("\n--- Stage 3 physicality checks ---")
println("(ω_CO2 decreases GDL→CL-exit at high current; ω_CO, ω_H2 increase through CL)")
for Vchk in (-0.816, -1.205, -1.644, -2.079, -2.413)
    res = load_nearest(STAGE, Vchk); res === nothing && continue
    uu, Vnear = res; Vapp = Vnear - V_RHE_to_SHE_offset
    omIn = gdl_omega(uu, 1)                          # GDL inlet-ish cell
    omEx = cl_omega(uu, N_CL); Mn=Mn_from_omega(omEx); x=omega_to_x(omEx,Mn)
    iCO = compute_i_CO(uu, Vapp, cl_mesh)/10; iH2 = compute_i_H2(uu, Vapp, cl_mesh)/10
    @printf("  V_RHE=%.3f: ω_CO2[GDL1]=%.4f  CL-exit x: CO2=%.4f CO=%.4f H2=%.4f  iCO=%.1f iH2=%.1f\n",
            Vnear, omIn[1], x[1], x[2], x[3], iCO, iH2)
end
@printf("\nCache: %d solutions in %s\n", length(cached_Vs), cache_dir(STAGE))
println("\n>>> Stage 3 complete. Generate plots (A1, A2, B1, overlay), then STOP for review. <<<")
