# =============================================================================
# run_stage5.jl — Stage 5 driver (BaseCase.md §11 Stage 5, §3.1, §3.4, §9.5)
#   Full coupled electrochemistry: φ_L (NP migration, monolithic) + φ_s (solid Ohm).
#   Uses FITTED kinetics (from output_fit/fitted_params.txt). 1480 DOFs.
#   Continuation (§9.5.5): anchor ONCE at V_c = 0 from a guess constructed from
#   physics (never from a cache — gotcha #0/#9), pre-ramp in −0.025 V steps to the
#   first output point, then walk the 89-V grid with the tangent predictor.
#   Post-solve residual assertion ‖F‖∞ ≤ 1e-6 at every emitted voltage (§9.5.6).
# =============================================================================
include("co2r_core.jl")
using Printf, SparseArrays, LinearAlgebra

const STAGE = 5
const FIT_CACHE = joinpath(@__DIR__, "output_fit", "cache")

# load fitted kinetics (§17.8, gotcha #14)
for line in eachline(joinpath(@__DIR__, "output_fit", "fitted_params.txt"))
    k, v = split(line); val = parse(Float64, v)
    k == "i0_CO2R" && set_i0_CO2R!(val); k == "lambda_MHC" && set_lambda_MHC!(val)
    k == "HER_prefactor" && set_HER_prefactor!(val); k == "alpha_c_HER" && set_alpha_c_HER!(val)
end
load_fit(V) = open(joinpath(FIT_CACHE, @sprintf("fit_V%.6f.bin", V)), "r") do f
    n = read(f, Int64); read!(f, Vector{Float64}(undef, n)) end

println("="^70)
println("=== Stage 5 : φ_L (NP migration) + φ_s (solid Ohm's law) — FITTED kinetics ===")
println("="^70)
@printf("Mesh N_CL=%d N_GDL=%d, DOFs=%d\n", N_CL, N_GDL, NDOF_STAGE5)
@printf("FITTED kinetics: i0_CO2R=%.2f A/m², λ_MHC=%.4f eV, HER_pf=%.4e, α_c=%.5f\n",
        i0_CO2R(), lambda_MHC(), HER_prefactor(), alpha_c_HER())
@assert isdir(FIT_CACHE) && !isempty(readdir(FIT_CACHE)) "No output_fit/cache — run run_fit.jl first (fitted kinetics + the fitted-kinetics Stage-4 reference sweep)."

eq = solve_buffer_equilibrium(1000.0, 1000.0); c_eq_vec = [eq.H, eq.OH, eq.HCO3, eq.CO3, eq.CO2]
cl_mesh = make_cl_mesh(); gdl_mesh = make_gdl_mesh()
const NDOF = NDOF_STAGE5
blocks = dof_blocks(STAGE)

Vs = sort(collect(V_RHE_grid), by = v -> -v)
V_anchor = V_RHE_to_SHE_offset                 # V_RHE at which V_c = V_RHE − offset = 0
V_ref = Ref(V_anchor)
res_closure!(F, uu) = residual_stage5!(F, uu, V_ref[], cl_mesh, gdl_mesh, c_eq_vec)

# anchor guess at V_c = 0 (§9.5.5): equilibrium liquid, inlet gas, φ_s = φ_L = 0 — from physics, not a cache
u = make_u0_stage5_anchor(c_eq_vec)

# fixed sparsity pattern by probing (perturb φ_s/φ_L so migration cross-terms appear); union of
# the pattern at the anchor voltage and at the first output voltage (kinetics active)
perturb = zeros(NDOF)
for ix in 1:N_CL; perturb[phi_L_idx(ix)] = 0.005; perturb[phi_s_cl_idx(ix)] = 0.01; end
for ig in 1:N_GDL; perturb[phi_s_gdl_idx(ig)] = 0.01; end
V_ref[] = V_anchor; r1, c1 = probe_sparsity(res_closure!, u; perturb=perturb)
V_ref[] = Vs[1];    r2, c2 = probe_sparsity(res_closure!, u; perturb=perturb)
V_ref[] = V_anchor
Jtemplate = pattern_matrix(vcat(r1, r2), vcat(c1, c2), NDOF)
@printf("Sparsity pattern: %d nnz (%.2f%% dense)\n", nnz(Jtemplate), 100*nnz(Jtemplate)/NDOF^2)
assemble! = make_ad_assembler(res_closure!, NDOF; chunk=12)
plu = AffineLU()

# V_c chain: anchor at V_c = 0, pre-ramp in −0.025 V steps down to the first output point, then the 89 output points
n_pre = floor(Int, (V_anchor - Vs[1]) / 0.025 - 1e-9)
pre_ramp = [V_anchor - 0.025*s for s in 0:n_pre]
chain = vcat(pre_ramp, Vs)
is_output = Set(Vs)
@printf("Continuation chain: %d points (%d pre-ramp + %d output); V_c = 0 → %.3f V\n",
        length(chain), length(pre_ramp), length(Vs), Vs[end] - V_RHE_to_SHE_offset)

# right-boundary ionic current check helper
function ionic_current_right(u)
    c = unpack_log_conc(u, N_CL); s = 0.0
    for i in 1:5
        L_eff = D_eff[i]/k_MT[i]
        Ji = np_flux(c[i,N_CL], c_eq_vec[i], u[phi_L_idx(N_CL)], 0.0, D_eff[i], z_charge[i], L_eff)
        s += z_charge[i]*Ji
    end
    return F_const*s
end

n_conv = Ref(0); maxF = Ref(0.0); t0 = time()
sweep_log = continuation_sweep!(u, chain, res_closure!, V_ref;
    blocks=blocks, plu=plu, jac_assembler=assemble!, J=Jtemplate, n_log_dofs=5*N_CL,
    shift_fallback! = shift_phis!,
    on_converged = (V, uu, info) -> begin
        if V in is_output
            save_solution(STAGE, copy(uu), V); n_conv[] += 1; maxF[] = max(maxF[], info.residual)
        end
    end)
n_iter = sum(l.iterations for l in sweep_log)
@printf("\nSweep done: %d/%d output voltages converged in %.1f s (%d chain points, %d Newton iterations, %.2f/step). max ||F||inf=%.2e\n",
        n_conv[], length(Vs), time()-t0, length(chain), n_iter, n_iter/length(chain), maxF[])

# ---------------------------------------------------------------------------
# Diagnostics CSVs
# ---------------------------------------------------------------------------
outdir = joinpath(@__DIR__, "output_stage$(STAGE)"); mkpath(outdir)
cached_Vs = list_cache(STAGE)
open(joinpath(outdir, "polarization.csv"), "w") do f
    println(f, "V_RHE,V_SHE,i_CO_mAcm2,i_H2_mAcm2,i_tot_mAcm2,FE_CO,FE_H2")
    for V in cached_Vs
        uu = load_solution(STAGE, V)
        iCO = compute_i_CO_s5(uu, cl_mesh)/10; iH2 = compute_i_H2_s5(uu, cl_mesh)/10; it = iCO+iH2
        @printf(f, "%.6f,%.6f,%.6e,%.6e,%.6e,%.4f,%.4f\n", V, V-V_RHE_to_SHE_offset, iCO, iH2, it,
                it>0 ? 100*iCO/it : 0.0, it>0 ? 100*iH2/it : 0.0)
    end
end
open(joinpath(outdir, "profiles.csv"), "w") do f
    println(f, "V_RHE,x_um,pH,c_CO2,c_OH,c_HCO3,c_CO3,c_Cs,c_H,R_CO2R,R_HER")
    for V in cached_Vs
        uu = load_solution(STAGE, V); c = unpack_log_conc(uu, N_CL)
        for ix in 1:N_CL
            cH=c[1,ix];cOH=c[2,ix];cHCO3=c[3,ix];cCO3=c[4,ix];cCO2=c[5,ix]
            cCs=cOH+cHCO3+2*cCO3-cH-C_fix_eff; pH=-log10(max(cH/1000.0,1e-20))
            rco=R_CO2R(cCO2,uu[phi_s_cl_idx(ix)],uu[phi_L_idx(ix)],cH,a_v_CL,lambda_MHC(),i0_CO2R(),n_e)
            rh2=R_HER(uu[phi_s_cl_idx(ix)],uu[phi_L_idx(ix)],cH,a_v_CL,n_e)
            @printf(f,"%.6f,%.5f,%.5f,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                    V,cl_mesh.x_centers[ix]*1e6,pH,cCO2,cOH,cHCO3,cCO3,cCs,cH,rco,rh2)
        end
    end
end
open(joinpath(outdir, "gas_profiles.csv"), "w") do f
    println(f, "V_RHE,x_um,region,p_CO2_atm,p_CO_atm,p_H2_atm,p_total_atm")
    for V in cached_Vs
        uu = load_solution(STAGE, V)
        for ig in 1:N_GDL
            om=gdl_omega(uu,ig);Mn=Mn_from_omega(om);x=omega_to_x(om,Mn);p=gdl_p(uu,ig)
            @printf(f,"%.6f,%.5f,GDL,%.6e,%.6e,%.6e,%.6f\n",V,gdl_mesh.x_centers[ig]*1e6,x[1]*p/101325,x[2]*p/101325,x[3]*p/101325,p/101325)
        end
        for ix in 1:N_CL
            om=cl_omega(uu,ix);Mn=Mn_from_omega(om);x=omega_to_x(om,Mn);p=cl_p(uu,ix)
            @printf(f,"%.6f,%.5f,CL,%.6e,%.6e,%.6e,%.6f\n",V,cl_mesh.x_centers[ix]*1e6,x[1]*p/101325,x[2]*p/101325,x[3]*p/101325,p/101325)
        end
    end
end
open(joinpath(outdir, "potentials.csv"), "w") do f      # D1/D2
    println(f, "V_RHE,x_um,region,phi_s_V,phi_L_V")
    for V in cached_Vs
        uu = load_solution(STAGE, V)
        for ig in 1:N_GDL
            @printf(f,"%.6f,%.5f,GDL,%.6e,NaN\n",V,gdl_mesh.x_centers[ig]*1e6,uu[phi_s_gdl_idx(ig)])
        end
        for ix in 1:N_CL
            @printf(f,"%.6f,%.5f,CL,%.6e,%.6e\n",V,cl_mesh.x_centers[ix]*1e6,uu[phi_s_cl_idx(ix)],uu[phi_L_idx(ix)])
        end
    end
end

# ---------------------------------------------------------------------------
# Stage 5 physicality checks (§11 Stage 5) — compare vs fit cache (Stage 4 phys, fitted kinetics)
# ---------------------------------------------------------------------------
println("\n--- Stage 5 physicality checks ---")
println("  V_RHE  φ_s_drop  GDL%   |Δφ_L|   |Δi|/i(coll)  ionicR_err  iCO   iH2   (Δi_CO vs fit)")
for Vchk in (-0.816, -1.205, -1.644, -2.079, -2.413)
    res = load_nearest(STAGE, Vchk); res === nothing && continue
    uu, Vn = res; Vapp = Vn - V_RHE_to_SHE_offset
    iCO = compute_i_CO_s5(uu, cl_mesh)/10; iH2 = compute_i_H2_s5(uu, cl_mesh)/10
    ifar = compute_i_CO_s5(uu,cl_mesh) + compute_i_H2_s5(uu,cl_mesh)   # A/m² (= n_e F ∫(R_CO2R+R_HER) dx)
    is1 = -sigma_s_eff_GDL*(uu[phi_s_gdl_idx(1)]-Vapp)/gdl_mesh.dx_between[1]
    chg = abs(abs(ifar)-abs(is1))/max(abs(ifar),1e-12)
    iion = ionic_current_right(uu); ionerr = abs(abs(ifar)-abs(iion))/max(abs(ifar),1e-12)
    drop = (uu[phi_s_cl_idx(N_CL)]-Vapp)*1000
    gdl_drop = (uu[phi_s_gdl_idx(N_GDL)]-Vapp)*1000
    gdlpct = 100*gdl_drop/drop
    dphiL = abs(uu[phi_L_idx(1)]-uu[phi_L_idx(N_CL)])*1000
    # compare to fit cache currents at this V
    uf = load_fit(Vn); iCOf = compute_i_CO(uf, Vapp, cl_mesh)/10
    dCO = 100*(iCO-iCOf)/iCOf
    @printf("  %.3f  %6.1fmV  %4.0f  %6.2fmV  %.2e   %.2e   %5.1f %5.1f  (%+.0f%%)\n",
            Vn, drop, gdlpct, dphiL, chg, ionerr, iCO, iH2, dCO)
end
@printf("\nCache: %d solutions in %s\n", length(cached_Vs), cache_dir(STAGE))
println("\n>>> Stage 5 complete. Generate plots (A1,A2,B1,D1,D2,overlay), then STOP — FINAL gate. <<<")
