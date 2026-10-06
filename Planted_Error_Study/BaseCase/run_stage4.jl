# =============================================================================
# run_stage4.jl — Stage 4 driver (BaseCase.md §11 Stage 4)
#   Finalize the Darcy pressure equation: u_G = −(κ/μ)dp/dx ≠ 0, real continuity
#   balance, harmonic-mean κ at the GDL/CL interface, p Dirichlet at flow-field,
#   dp/dx=0 at electrolyte. Same 1200 DOFs.
#   Anchor (§9.5.5): equilibrium liquid + inlet gas at −0.300 V; tangent continuation.
# =============================================================================
include("co2r_core.jl")
using Printf

const STAGE = 4

println("="^70)
@printf("=== Stage %d : Darcy pressure finalization ===\n", STAGE)
println("="^70)
@printf("Mesh: N_CL=%d N_GDL=%d, DOFs=%d\n", N_CL, N_GDL, NDOF_GAS)
@printf("Kinetics (initial-guess): i0_CO2R=%.1f A/m², lambda_MHC=%.2f eV, HER_pf=%.3e, alpha_c=%.4f\n",
        i0_CO2R(), lambda_MHC(), HER_prefactor(), alpha_c_HER())
@printf("κ_GDL=%.2e  κ_CL=%.2e  κ_IF(harmonic)=%.2e m²  μ_gas=%.2e Pa·s\n",
        kappa_GDL, kappa_CL, kappa_IF, mu_gas)
@assert !isempty(list_cache(3)) "No Stage 3 cache — run run_stage3.jl first (§13 driver order)."

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
res_closure!(F, uu) = residual_stage4!(F, uu, V_ref[], cl_mesh, gdl_mesh, c_eq_vec)
assemble! = make_ad_assembler(res_closure!, NDOF; chunk=12)

# anchor guess (§9.5.5): equilibrium liquid + inlet gas at p_total (uniform pressure) in every gas cell
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
# Diagnostics CSVs (same schema as Stage 3; gas_profiles now shows real p variation)
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
            cCs = cOH + cHCO3 + 2*cCO3 - cH - C_fix_eff; pH = -log10(max(cH/1000.0,1e-20))
            rco = R_CO2R(cCO2, Vapp, 0.0, cH, a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
            rh2 = R_HER(Vapp, 0.0, cH, a_v_CL, n_e)
            @printf(f, "%.6f,%.5f,%.5f,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                    V, cl_mesh.x_centers[ix]*1e6, pH, cCO2, cOH, cHCO3, cCO3, cCs, cH, rco, rh2)
        end
    end
end
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
# Stage 4 physicality checks (§11 Stage 4): pressure drop O(1–100 Pa); currents ≈ Stage 3
# ---------------------------------------------------------------------------
println("\n--- Stage 4 physicality checks ---")
for Vchk in (-0.816, -1.205, -1.644, -2.079, -2.413)
    res = load_nearest(STAGE, Vchk); res === nothing && continue
    uu, Vnear = res; Vapp = Vnear - V_RHE_to_SHE_offset
    ps = vcat([gdl_p(uu,ig) for ig in 1:N_GDL], [cl_p(uu,ix) for ix in 1:N_CL])
    dP = maximum(ps) - minimum(ps)
    iCO = compute_i_CO(uu, Vapp, cl_mesh)/10; iH2 = compute_i_H2(uu, Vapp, cl_mesh)/10
    # compare to Stage 3 at same V
    u3 = load_nearest(3, Vnear)
    if u3 !== nothing
        iCO3 = compute_i_CO(u3[1], Vapp, cl_mesh)/10; iH23 = compute_i_H2(u3[1], Vapp, cl_mesh)/10
        @printf("  V_RHE=%.3f: ΔP_gas=%.3e Pa  iCO=%.1f (S3 %.1f, Δ%.2f%%)  iH2=%.1f (S3 %.1f)\n",
                Vnear, dP, iCO, iCO3, 100*(iCO-iCO3)/iCO3, iH2, iH23)
    end
end
@printf("\nCache: %d solutions in %s\n", length(cached_Vs), cache_dir(STAGE))
println("\n>>> Stage 4 complete. Generate plots, then STOP for Fit step. <<<")
