# =============================================================================
# run_stage1.jl — Stage 1 driver (BaseCase.md §11 Stage 1)
#   Liquid NP DIFFUSION ONLY + Dirichlet BCs both sides + simplified kinetics
#   (phi_s = V_applied_SHE, phi_L = 0). DOFs = 5×N_CL.
# Anchors at the mildest grid voltage from the exact equilibrium guess, walks the
# 89-V grid by tangent-predictor continuation (§9.5.5), caches every converged
# solution, writes diagnostics CSV, prints a summary, and STOPS for human review.
# =============================================================================
include("co2r_core.jl")
using Printf

const STAGE = 1

println("="^70)
@printf("=== Stage %d : Liquid NP diffusion-only + Dirichlet + simplified kinetics ===\n", STAGE)
println("="^70)
@printf("Mesh: N_CL=%d (q=%.2f), DOFs=%d\n", N_CL, MESH_Q, 5*N_CL)
@printf("Kinetics (initial-guess): i0_CO2R=%.1f A/m², lambda_MHC=%.2f eV\n", i0_CO2R(), lambda_MHC())
@printf("          HER_prefactor=%.3e mA/cm², alpha_c=%.4f\n", HER_prefactor(), alpha_c_HER())
@printf("V_RHE_to_SHE_offset = %.5f V ; a_v_CL = %.3e m⁻¹\n", V_RHE_to_SHE_offset, a_v_CL)
let cdir = cache_dir(STAGE)
    if isdir(cdir) && !isempty(readdir(cdir))
        @printf("WARNING: existing cache in %s — verify kinetics match before reuse.\n", cdir)
    end
end

# ---------------------------------------------------------------------------
# §12.1 / §9.6 equilibrium sanity check
# ---------------------------------------------------------------------------
eq = solve_buffer_equilibrium(1000.0, 1000.0)
c_eq_vec = [eq.H, eq.OH, eq.HCO3, eq.CO3, eq.CO2]
@printf("\nEquilibrium (1 M CsHCO₃, 50 °C): pH=%.3f\n", eq.pH)
@printf("  c_H=%.3e  c_OH=%.3e  c_HCO3=%.2f  c_CO3=%.2f  c_CO2=%.2f mol/m³\n",
        eq.H, eq.OH, eq.HCO3, eq.CO3, eq.CO2)
let S = buffer_sources(c_eq_vec)
    smax = maximum(abs, S)
    @printf("  max|buffer_source(c_eq)| = %.3e mol/m³/s (must be < 1e-6)\n", smax)
    @assert smax < 1e-6 "Equilibrium buffer residual too large — check K/k unit conversions"
    cb = eq.H + 1000.0 - eq.OH - eq.HCO3 - 2*eq.CO3
    @printf("  charge balance = %.3e mol/m³\n", cb)
end
@printf("Stage 1 CO₂(aq) GDL-face Dirichlet (van 't Hoff): c_CO2_bc = %.3f mol/m³\n", c_CO2_bc_stage12)

# ---------------------------------------------------------------------------
# Mesh, solver workspace (§9.3.3 fixed pattern, §9.5.2 persistent LU)
# ---------------------------------------------------------------------------
cl_mesh = make_cl_mesh()
const NDOF = 5 * N_CL
rows, cols = block_tridiag_pattern(N_CL, 5)
Jtemplate = pattern_matrix(rows, cols, NDOF)
plu = AffineLU()
blocks = dof_blocks(STAGE)

# anchor guess (§9.5.5): every cell at bulk equilibrium log-concentration
ln_c_eq = log.(c_eq_vec)
u0_equil() = repeat(ln_c_eq, N_CL)        # cell-major: [lnH,lnOH,lnHCO3,lnCO3,lnCO2] × N_CL

# ---------------------------------------------------------------------------
# §9.5.5 continuation: anchor at −0.300 V_RHE, tangent predictor down the grid
# ---------------------------------------------------------------------------
Vs = sort(collect(V_RHE_grid), by = v -> -v)    # -0.300 → -2.500

# res! captures V through V_ref; the residual function identity is stable, so a
# single AD config/assembler is reused for the whole sweep.
V_ref = Ref(Vs[1])
res_closure!(F, uu) = residual_stage1!(F, uu, V_ref[], cl_mesh, c_eq_vec, c_CO2_bc_stage12)
assemble! = make_ad_assembler(res_closure!, NDOF; chunk=12)

n_conv = Ref(0); maxF = Ref(0.0)
t_start = time()
u = u0_equil()
sweep_log = continuation_sweep!(u, Vs, res_closure!, V_ref;
    blocks=blocks, plu=plu, jac_assembler=assemble!, J=Jtemplate, n_log_dofs=NDOF,
    on_converged = (V, uu, info) -> begin
        save_solution(STAGE, copy(uu), V)        # cache FIRST (§10.4)
        n_conv[] += 1; maxF[] = max(maxF[], info.residual)
    end)
t_sweep = time() - t_start
n_iter = sum(l.iterations for l in sweep_log)

@printf("\nSweep done: %d/%d converged in %.1f s (%d Newton iterations, %.2f/step). max ||F||inf = %.2e\n",
        n_conv[], length(Vs), t_sweep, n_iter, n_iter/length(Vs), maxF[])

# ---------------------------------------------------------------------------
# Diagnostics CSV (polarization + spatial profiles for ALL cached voltages)
# ---------------------------------------------------------------------------
outdir = joinpath(@__DIR__, "output_stage$(STAGE)")
mkpath(outdir)

cached_Vs = list_cache(STAGE)
open(joinpath(outdir, "polarization.csv"), "w") do f
    println(f, "V_RHE,V_SHE,i_CO_mAcm2,i_H2_mAcm2,i_tot_mAcm2,FE_CO,FE_H2")
    for V in cached_Vs
        uu = load_solution(STAGE, V)
        Vapp = V - V_RHE_to_SHE_offset
        iCO = compute_i_CO(uu, Vapp, cl_mesh) / 10.0     # mA/cm²
        iH2 = compute_i_H2(uu, Vapp, cl_mesh) / 10.0
        itot = iCO + iH2
        feco = itot > 0 ? 100*iCO/itot : 0.0
        feh2 = itot > 0 ? 100*iH2/itot : 0.0
        @printf(f, "%.6f,%.6f,%.6e,%.6e,%.6e,%.4f,%.4f\n", V, Vapp, iCO, iH2, itot, feco, feh2)
    end
end

open(joinpath(outdir, "profiles.csv"), "w") do f
    println(f, "V_RHE,x_um,pH,c_CO2,c_OH,c_HCO3,c_CO3,c_Cs,c_H,R_CO2R,R_HER")
    for V in cached_Vs
        uu = load_solution(STAGE, V)
        Vapp = V - V_RHE_to_SHE_offset
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
# Physicality checks (§11 Stage 1) — print at a few representative voltages
# ---------------------------------------------------------------------------
println("\n--- Stage 1 physicality checks ---")
for Vchk in (-0.816, -1.205, -1.644, -2.079, -2.413)
    res = load_nearest(STAGE, Vchk)
    res === nothing && continue
    uu, Vnear = res
    Vapp = Vnear - V_RHE_to_SHE_offset
    c = unpack_log_conc(uu, N_CL)
    pH_gdl = -log10(max(c[1,1]/1000.0,1e-20))
    pH_el  = -log10(max(c[1,N_CL]/1000.0,1e-20))
    iCO = compute_i_CO(uu, Vapp, cl_mesh)/10
    iH2 = compute_i_H2(uu, Vapp, cl_mesh)/10
    feco = 100*iCO/(iCO+iH2)
    @printf("  V_RHE=%.3f: pH[GDL]=%.2f pH[EL]=%.2f  CO2(EL)=%.3f mol/m³  iCO=%.1f iH2=%.1f mA/cm²  FE_CO=%.1f%%\n",
            Vnear, pH_gdl, pH_el, c[5,N_CL], iCO, iH2, feco)
end

@printf("\nCache: %d solutions in %s\n", length(cached_Vs), cache_dir(STAGE))
println("Wrote: $(joinpath(outdir,"polarization.csv")) and profiles.csv")
println("\n>>> Stage 1 complete. Generate plots, then STOP for human review. <<<")
