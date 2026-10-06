# =============================================================================
# run_fit.jl — Kinetic parameter fit after Stage 4 (BaseCase.md §17)
#   Nelder-Mead on (i0_CO2R, λ_MHC, log10 HER_pf, α_c) against the 5 Lees points,
#   using the Stage 4 residual (φ_L=0, φ_s=V_applied) for speed (§11, §17).
#   Polishes at the EXACT experimental voltages each NM eval (§17.3) with the §9.5
#   kernel, warm-started from the Stage 4 cache; log-space SSE.
#   Writes the 89-V refit sweep to output_fit/cache/fit_V*.bin — the Stage-4-physics
#   reference at fitted kinetics that the Stage 5 checks compare against (§17.8).
#   STOPS for review.
# =============================================================================
include("co2r_core.jl")
using Printf, SparseArrays, Optim

println("="^70)
println("=== Fit step (after Stage 4): i0_CO2R, λ_MHC, HER_prefactor, α_c ===")
println("="^70)
@printf("Initial-guess kinetics: i0=%.1f, λ=%.2f, HER_pf=%.3e, α_c=%.4f\n",
        i0_CO2R(), lambda_MHC(), HER_prefactor(), alpha_c_HER())
@assert !isempty(list_cache(4)) "No Stage 4 cache — run run_stage4.jl first."

eq = solve_buffer_equilibrium(1000.0, 1000.0)
c_eq_vec = [eq.H, eq.OH, eq.HCO3, eq.CO3, eq.CO2]
cl_mesh = make_cl_mesh(); gdl_mesh = make_gdl_mesh()
const NDOF = NDOF_GAS
rows, cols = build_gas_pattern(N_CL, N_GDL)
Jtemplate = pattern_matrix(rows, cols, NDOF)
plu = AffineLU()
blocks = dof_blocks(4)

V_ref = Ref(0.0)
res_closure!(F, uu) = residual_stage4!(F, uu, V_ref[], cl_mesh, gdl_mesh, c_eq_vec)
assemble! = make_ad_assembler(res_closure!, NDOF; chunk=12)   # Ref-backed kinetics ⇒ no rebuild on param change

# per-voltage polish (§9.5.5 Fit row): same kernel, no continuation
function polish!(u::Vector{Float64}, V_rhe::Float64; aggressive::Bool=false)
    V_ref[] = V_rhe
    newton_affine!(u, res_closure!; blocks=blocks, plu=plu, jac_assembler=assemble!, J=Jtemplate,
                   alpha_init = aggressive ? 1e-2 : 1.0,
                   max_iter   = aggressive ? 200  : 40, n_log_dofs=5*N_CL)
end

# ── initial polish at the 5 exact experimental voltages (default kinetics), from nearest grid cache
exp_cache = Dict{Float64, Vector{Float64}}()
for V in E_exp_RHE
    u0, _ = load_nearest(4, V)
    u = copy(u0); info = polish!(u, V)
    info.converged || (u = copy(u0); info = polish!(u, V; aggressive=true))
    @assert info.converged "initial polish failed at V_RHE=$V"
    exp_cache[V] = u
end
@printf("Initial polish complete at exp voltages %s\n", string(E_exp_RHE))

# ── objective: re-polish at exp voltages (warm-start from exp_cache), log-space SSE
const LOWER = [10.0, 0.5, 8.0, 0.01]
const UPPER = [5000.0, 3.0, 18.0, 0.50]
neval = Ref(0); best_sse = Ref(Inf)
function objective(p)
    neval[] += 1
    any(p .< LOWER) && return 1e6
    any(p .> UPPER) && return 1e6
    set_i0_CO2R!(p[1]); set_lambda_MHC!(p[2]); set_HER_prefactor!(10.0^p[3]); set_alpha_c_HER!(p[4])
    local_cache = Dict{Float64, Vector{Float64}}()
    for V in E_exp_RHE
        u = copy(exp_cache[V]); info = polish!(u, V)
        info.converged || return 1e6
        local_cache[V] = u
    end
    for (V, u) in local_cache; exp_cache[V] = u; end      # commit only if all 5 converged
    sse = 0.0
    for (k, V) in enumerate(E_exp_RHE)
        u = exp_cache[V]; Vapp = V - V_RHE_to_SHE_offset
        iCO = compute_i_CO(u, Vapp, cl_mesh)/10; iH2 = compute_i_H2(u, Vapp, cl_mesh)/10
        sse += (log10(max(iCO,0.1)) - log10(i_CO_exp[k]))^2
        sse += (log10(max(iH2,0.1)) - log10(i_H2_exp[k]))^2
    end
    if sse < best_sse[]
        best_sse[] = sse
        @printf("  eval %3d: SSE=%.5f  i0=%.1f λ=%.3f HER_pf=%.3e α_c=%.4f\n",
                neval[], sse, p[1], p[2], 10.0^p[3], p[4])
    end
    return sse
end

params0 = [540.0, 1.8, log10(2.409e13), 0.0872]
println("\nRunning Nelder-Mead (max 200 iters)...")
t0 = time()
result = optimize(objective, params0, NelderMead(), Optim.Options(iterations=200))
pfit = Optim.minimizer(result)
set_i0_CO2R!(pfit[1]); set_lambda_MHC!(pfit[2]); set_HER_prefactor!(10.0^pfit[3]); set_alpha_c_HER!(pfit[4])
@printf("\nNM done: %d evals, %.1f s, final SSE=%.5f\n", neval[], time()-t0, Optim.minimum(result))
println("="^70)
@printf("FITTED KINETICS:\n  i0_CO2R       = %.2f A/m²\n  λ_MHC         = %.4f eV\n", i0_CO2R(), lambda_MHC())
@printf("  HER_prefactor = %.4e mA/cm²\n  α_c_HER       = %.5f\n", HER_prefactor(), alpha_c_HER())
println("="^70)

# ── per-point fit quality vs Lees (§17.6 acceptance criteria)
function report_fit_quality(exp_cache, cl_mesh)
    println("\n--- Fit quality at the 5 Lees voltages ---")
    println("  V_RHE    iCO_mod  iCO_exp  err%    iH2_mod  iH2_exp  err%   FE_CO_mod FE_CO_exp")
    allpass = true
    for (k, V) in enumerate(E_exp_RHE)
        u = exp_cache[V]; Vapp = V - V_RHE_to_SHE_offset
        iCO = compute_i_CO(u, Vapp, cl_mesh)/10; iH2 = compute_i_H2(u, Vapp, cl_mesh)/10
        eCO = 100*(iCO - i_CO_exp[k])/i_CO_exp[k]; eH2 = 100*(iH2 - i_H2_exp[k])/i_H2_exp[k]
        feco = 100*iCO/(iCO+iH2)
        tolCO = (k == 5 ? 80.0 : 30.0)   # point 5 (transport plateau): ±80% on iCO allowed (§17.6)
        pass = abs(eCO) <= tolCO && abs(eH2) <= 30.0
        allpass &= pass
        @printf("  %.3f   %7.1f  %7.1f  %+5.0f   %7.1f  %7.1f  %+5.0f   %6.1f    %6.1f  %s\n",
                V, iCO, i_CO_exp[k], eCO, iH2, i_H2_exp[k], eH2, feco, FE_CO_exp[k], pass ? "ok" : "OUT")
    end
    @printf("Per-point tolerance check (iCO ≤30%% [pt5 ≤80%%], iH2 ≤30%%): %s\n", allpass ? "PASS" : "see OUT rows")
end
report_fit_quality(exp_cache, cl_mesh)

# ── final 89-V refit sweep with fitted kinetics → output_fit/cache/fit_V*.bin (§17.8)
const FIT_DIR = joinpath(@__DIR__, "output_fit"); const FIT_CACHE = joinpath(FIT_DIR, "cache")
mkpath(FIT_CACHE)
function save_fit(u, V)
    open(joinpath(FIT_CACHE, @sprintf("fit_V%.6f.bin", V)), "w") do f
        write(f, Int64(length(u))); write(f, u)
    end
end
println("\nRefit 89-V sweep with fitted kinetics (per-voltage polish, warm-start from Stage 4 grid cache)...")
Vs = sort(collect(V_RHE_grid), by = v -> -v)
u = copy(load_solution(4, Vs[1])); nref = 0
for (k, V) in enumerate(Vs)
    global u, nref
    info = polish!(u, V)
    if !info.converged
        u4 = load_solution(4, V); u4 !== nothing && (u .= u4); info = polish!(u, V; aggressive=true)
    end
    info.converged ? (save_fit(copy(u), V); nref += 1) : @printf("  [refit FAIL] V_RHE=%.4f\n", V)
end
@printf("Refit sweep: %d/%d converged, cached to %s\n", nref, length(Vs), FIT_CACHE)

# ── diagnostics CSVs from the refit sweep (for plotting; mirror stage CSV schema)
function write_csvs(dir)
    mkpath(dir)
    fitVs = sort([parse(Float64, f[6:end-4]) for f in readdir(FIT_CACHE) if startswith(f,"fit_V")])
    load_fit(V) = open(joinpath(FIT_CACHE, @sprintf("fit_V%.6f.bin", V)),"r") do f
        n=read(f,Int64); read!(f,Vector{Float64}(undef,n)) end
    open(joinpath(dir, "polarization.csv"), "w") do f
        println(f, "V_RHE,V_SHE,i_CO_mAcm2,i_H2_mAcm2,i_tot_mAcm2,FE_CO,FE_H2")
        for V in fitVs
            uu=load_fit(V); Vapp=V-V_RHE_to_SHE_offset
            iCO=compute_i_CO(uu,Vapp,cl_mesh)/10; iH2=compute_i_H2(uu,Vapp,cl_mesh)/10; it=iCO+iH2
            @printf(f,"%.6f,%.6f,%.6e,%.6e,%.6e,%.4f,%.4f\n",V,Vapp,iCO,iH2,it,
                    it>0 ? 100*iCO/it : 0.0, it>0 ? 100*iH2/it : 0.0)
        end
    end
    open(joinpath(dir, "profiles.csv"), "w") do f
        println(f, "V_RHE,x_um,pH,c_CO2,c_OH,c_HCO3,c_CO3,c_Cs,c_H,R_CO2R,R_HER")
        for V in fitVs
            uu=load_fit(V); Vapp=V-V_RHE_to_SHE_offset; c=unpack_log_conc(uu,N_CL)
            for ix in 1:N_CL
                cH=c[1,ix];cOH=c[2,ix];cHCO3=c[3,ix];cCO3=c[4,ix];cCO2=c[5,ix]
                cCs=cOH+cHCO3+2*cCO3-cH-C_fix_eff; pH=-log10(max(cH/1000.0,1e-20))
                rco=R_CO2R(cCO2,Vapp,0.0,cH,a_v_CL,lambda_MHC(),i0_CO2R(),n_e); rh2=R_HER(Vapp,0.0,cH,a_v_CL,n_e)
                @printf(f,"%.6f,%.5f,%.5f,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                        V,cl_mesh.x_centers[ix]*1e6,pH,cCO2,cOH,cHCO3,cCO3,cCs,cH,rco,rh2)
            end
        end
    end
    open(joinpath(dir, "gas_profiles.csv"), "w") do f
        println(f, "V_RHE,x_um,region,p_CO2_atm,p_CO_atm,p_H2_atm,p_total_atm")
        for V in fitVs
            uu=load_fit(V)
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
end
write_csvs(FIT_DIR)

# save fitted params to a small text file for run_stage5.jl to load
open(joinpath(FIT_DIR, "fitted_params.txt"), "w") do f
    @printf(f, "i0_CO2R %.10e\nlambda_MHC %.10e\nHER_prefactor %.10e\nalpha_c_HER %.10e\n",
            i0_CO2R(), lambda_MHC(), HER_prefactor(), alpha_c_HER())
end
println("\nWrote fitted_params.txt and output_fit CSVs.")
println("\n>>> Fit step complete. Generate fit plots, then STOP for review (before Stage 5). <<<")
