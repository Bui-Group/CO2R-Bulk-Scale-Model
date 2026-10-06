# =============================================================================
# run_stage2.jl — Stage 2 driver (BaseCase.md §11 Stage 2)
#   Replace the electrolyte-side Dirichlet with the Sherwood-Reynolds convective
#   Neumann BC. Same 5×N_CL DOFs, solver, GDL BCs, simplified kinetics.
#   Self-anchoring continuation (§9.5.5): anchor at −0.300 V from the equilibrium
#   guess, tangent predictor down the grid.
# =============================================================================
include("co2r_core.jl")
using Printf

const STAGE = 2

println("="^70)
@printf("=== Stage %d : Sherwood-Reynolds Neumann electrolyte BC ===\n", STAGE)
println("="^70)
@printf("Mesh: N_CL=%d (q=%.2f), DOFs=%d\n", N_CL, MESH_Q, 5*N_CL)
@printf("Kinetics (initial-guess): i0_CO2R=%.1f A/m², lambda_MHC=%.2f eV\n", i0_CO2R(), lambda_MHC())
@printf("          HER_prefactor=%.3e mA/cm², alpha_c=%.4f\n", HER_prefactor(), alpha_c_HER())
@printf("Re_elec = %.0f ; u_elec = %.3f m/s\n", Re_elec, u_elec)
@printf("k_MT [m/s] (built from bare-water D_w): ")
for i in 1:5; @printf("%.3e ", k_MT[i]); end; println()
# Biot numbers Bi_i = k_MT_i * L_CL / D_eff_i  (§11 Stage 2 check; expect O(0.1–10))
@printf("Biot Bi_i = k_MT·L_CL/D_eff: ")
for i in 1:5; @printf("%.2f ", k_MT[i]*L_CL/D_eff[i]); end
println("  [H,OH,HCO3,CO3,CO2]")

let cdir = cache_dir(STAGE)
    if isdir(cdir) && !isempty(readdir(cdir))
        @printf("WARNING: existing cache in %s — verify kinetics match before reuse.\n", cdir)
    end
end
@assert !isempty(list_cache(1)) "No Stage 1 cache found — run run_stage1.jl first (§13 driver order)."

# ---------------------------------------------------------------------------
eq = solve_buffer_equilibrium(1000.0, 1000.0)
c_eq_vec = [eq.H, eq.OH, eq.HCO3, eq.CO3, eq.CO2]
@printf("\nBulk c_eq_OH = %.3e mol/m³ (electrolyte reference for the Sherwood BC)\n", eq.OH)

cl_mesh = make_cl_mesh()
const NDOF = 5 * N_CL
rows, cols = block_tridiag_pattern(N_CL, 5)
Jtemplate = pattern_matrix(rows, cols, NDOF)
plu = AffineLU()
blocks = dof_blocks(STAGE)

ln_c_eq = log.(c_eq_vec)
u0_equil() = repeat(ln_c_eq, N_CL)

Vs = sort(collect(V_RHE_grid), by = v -> -v)
V_ref = Ref(Vs[1])
res_closure!(F, uu) = residual_stage2!(F, uu, V_ref[], cl_mesh, c_eq_vec, c_CO2_bc_stage12)
assemble! = make_ad_assembler(res_closure!, NDOF; chunk=12)

# §9.5.5: anchor at −0.300 V from the equilibrium guess, tangent continuation down the grid
n_conv = Ref(0); maxF = Ref(0.0)
t_start = time()
u = u0_equil()
sweep_log = continuation_sweep!(u, Vs, res_closure!, V_ref;
    blocks=blocks, plu=plu, jac_assembler=assemble!, J=Jtemplate, n_log_dofs=NDOF,
    on_converged = (V, uu, info) -> begin
        save_solution(STAGE, copy(uu), V)
        n_conv[] += 1; maxF[] = max(maxF[], info.residual)
    end)
t_sweep = time() - t_start
n_iter = sum(l.iterations for l in sweep_log)
@printf("\nSweep done: %d/%d converged in %.1f s (%d Newton iterations, %.2f/step). max ||F||inf = %.2e\n",
        n_conv[], length(Vs), t_sweep, n_iter, n_iter/length(Vs), maxF[])

# ---------------------------------------------------------------------------
# Diagnostics CSV
# ---------------------------------------------------------------------------
outdir = joinpath(@__DIR__, "output_stage$(STAGE)")
mkpath(outdir)
cached_Vs = list_cache(STAGE)
open(joinpath(outdir, "polarization.csv"), "w") do f
    println(f, "V_RHE,V_SHE,i_CO_mAcm2,i_H2_mAcm2,i_tot_mAcm2,FE_CO,FE_H2")
    for V in cached_Vs
        uu = load_solution(STAGE, V); Vapp = V - V_RHE_to_SHE_offset
        iCO = compute_i_CO(uu, Vapp, cl_mesh)/10; iH2 = compute_i_H2(uu, Vapp, cl_mesh)/10
        itot = iCO+iH2
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
            pH  = -log10(max(cH/1000.0, 1e-20))
            rco = R_CO2R(cCO2, Vapp, 0.0, cH, a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
            rh2 = R_HER(Vapp, 0.0, cH, a_v_CL, n_e)
            @printf(f, "%.6f,%.5f,%.5f,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                    V, cl_mesh.x_centers[ix]*1e6, pH, cCO2, cOH, cHCO3, cCO3, cCs, cH, rco, rh2)
        end
    end
end

# ---------------------------------------------------------------------------
# Stage 2 physicality checks (§11 Stage 2)
# ---------------------------------------------------------------------------
println("\n--- Stage 2 physicality checks ---")
@printf("(c_OH(L_CL) should sit ABOVE bulk c_eq_OH=%.3e; c_CO2(L_CL) → 0 at high current)\n", eq.OH)
for Vchk in (-0.816, -1.205, -1.644, -2.079, -2.413)
    res = load_nearest(STAGE, Vchk); res === nothing && continue
    uu, Vnear = res; Vapp = Vnear - V_RHE_to_SHE_offset
    c = unpack_log_conc(uu, N_CL)
    # electrolyte-face (last cell) values and convective fluxes J = k_MT*(c_N - c_eq)
    fCO2 = k_MT[5]*(c[5,N_CL]-c_eq_vec[5]); fOH = k_MT[2]*(c[2,N_CL]-c_eq_vec[2])
    iCO = compute_i_CO(uu, Vapp, cl_mesh)/10; iH2 = compute_i_H2(uu, Vapp, cl_mesh)/10
    @printf("  V_RHE=%.3f: CO2(L_CL)=%.3f  OH(L_CL)=%.3e (×%.0f bulk)  J_CO2=%.2e J_OH=%.2e  iCO=%.1f iH2=%.1f\n",
            Vnear, c[5,N_CL], c[2,N_CL], c[2,N_CL]/eq.OH, fCO2, fOH, iCO, iH2)
end
@printf("\nCache: %d solutions in %s\n", length(cached_Vs), cache_dir(STAGE))
println("\n>>> Stage 2 complete. Generate plots (incl. overlay vs Stage 1), then STOP for review. <<<")
