# check_final.jl — §9.5.9 step 8 + §17.6 criterion 4: Stage 5 vs experiment,
# and the "5 polished fit points sit exactly on the 89-point sweep" branch check.
include("common.jl")
include("stage5_diag.jl")
include(joinpath(@__DIR__, "fitted_params.jl"))
set_i0_CO2R!(FITTED_i0_CO2R); set_lambda_MHC!(FITTED_lambda)
set_HER_prefactor!(FITTED_HER_pf); set_alpha_c_HER!(FITTED_alpha_c)

m = build_model(5); ly = m.ly
S5 = joinpath(@__DIR__, "output_stage5", "cache")
Vs = list_cache(S5, "s5_")
Us = [load_solution(S5, "s5_", V) for V in Vs]
iCO = [compute_currents5(u, ly, m.cl)[1]/10 for u in Us]
iH2 = [compute_currents5(u, ly, m.cl)[2]/10 for u in Us]

println("=== Stage 5 vs experimental data (§9.5.9 step 8) ===")
@printf("%9s %10s %10s %8s | %10s %10s %8s\n",
        "V_RHE","iCO_mod","iCO_exp","err%","iH2_mod","iH2_exp","err%")
for (k, V) in enumerate(E_exp_RHE)
    j = argmin(abs.(Vs .- V))
    # linear interpolation onto the exact experimental voltage
    j2 = Vs[j] > V ? j+1 : j-1
    j2 = clamp(j2, 1, length(Vs))
    w = j2 == j ? 0.0 : (V - Vs[j])/(Vs[j2] - Vs[j])
    ico = iCO[j] + w*(iCO[j2]-iCO[j]); ih2 = iH2[j] + w*(iH2[j2]-iH2[j])
    @printf("%9.3f %10.2f %10.2f %+8.1f | %10.2f %10.2f %+8.1f\n",
            V, ico, i_CO_exp[k], 100*(ico-i_CO_exp[k])/i_CO_exp[k],
            ih2, i_H2_exp[k], 100*(ih2-i_H2_exp[k])/i_H2_exp[k])
end

println("\n=== §17.6 criterion 4: one continuous branch ===")
# Independently re-solve Stage 5 at the 5 experimental voltages by continuing
# from the neighbouring cached sweep point, and check the results land on the
# sweep line (a wrong-branch sweep is off by ~20 %, not ~0.001 %).
plu = AffineLU(); Jd = zeros(ly.N_TOT, ly.N_TOT)
Vr = Ref(-0.816); rc!(F_,u_) = residual_stage5!(F_,u_,Vr[],m.cl,m.gm,m.c_eq,ly)
cfg = ForwardDiff.JacobianConfig(rc!, zeros(ly.N_TOT), zeros(ly.N_TOT), ForwardDiff.Chunk{12}())
for (k, V) in enumerate(E_exp_RHE)
    j = argmin(abs.(Vs .- V)); u = copy(Us[j]); Vr[] = V
    info = newton_affine!(u, rc!; blocks=weight_blocks(ly), plu=plu, rtol=1e-8,
                          max_iter=40, alpha_init=1.0, n_log_dofs=5*m.cl.N,
                          Jdense=Jd, cfg=cfg)
    a = compute_currents5(u, ly, m.cl)
    d = stage5_diagnostics(u, V, m.cl, m.gm, m.c_eq, ly)
    j2 = Vs[j] > V ? j+1 : j-1; j2 = clamp(j2,1,length(Vs))
    w = j2 == j ? 0.0 : (V - Vs[j])/(Vs[j2]-Vs[j])
    ico_interp = iCO[j] + w*(iCO[j2]-iCO[j])
    @printf("V=%.3f  polished i_CO=%7.2f vs sweep-interp %7.2f  (%.3f%% apart), Δφ_L=%.2f mV, conv=%s\n",
            V, a[1]/10, ico_interp, 100*abs(a[1]/10-ico_interp)/ico_interp,
            1e3*d.dphi_L, info.converged)
end

println("\n=== §12.1 electroneutrality per cell at a converged Stage 5 state ===")
let j = argmin(abs.(Vs .- (-1.650))), u = Us[j]
    c = unpack_log_conc(u, m.cl.N); worst = 0.0
    for ix in 1:m.cl.N
        cCs = c_Cs_from_electroneutrality(@view c[:,ix])
        r = c[1,ix] - c[2,ix] - c[3,ix] - 2*c[4,ix] + cCs + C_fix*eps_I
        worst = max(worst, abs(r))
    end
    @printf("max |Σ z_i c_i + c_Cs + C_fix ε_I| = %.3e mol/m³ (identity by construction)\n", worst)
end
