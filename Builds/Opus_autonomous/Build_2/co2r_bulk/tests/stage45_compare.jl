# §11 Stage-5 check 5: quantify the migration → Sechenov-relief mechanism
include(joinpath(@__DIR__, "..", "model.jl"))
using Printf
let d = Dict{String,Float64}()
    for l in eachline(joinpath(ROOT,"output_fit","fitted_params.txt"))
        k,v = split(l,"="); d[strip(k)] = parse(Float64, strip(v))
    end
    set_i0_CO2R!(d["i0_CO2R"]); set_lambda_MHC!(d["lambda_MHC"])
    set_HER_prefactor!(d["HER_prefactor"]); set_alpha_c_HER!(d["alpha_c_HER"])
end
N_CL = N_CL_DEFAULT; N_GDL = N_GDL_DEFAULT
r4 = make_stage_run(Val(4), N_CL, N_GDL; outname="output_fit", prefix="fit")
r5 = make_stage_run(Val(5), N_CL, N_GDL)
V = -2.425
u4 = load_stage(r4, V); u5 = load_stage(r5, V)
c4 = unpack_log_conc(u4, r4.ly); c5 = unpack_log_conc(u5, r5.ly)
cCs4 = cCs_from_electroneutrality(c4[1,1],c4[2,1],c4[3,1],c4[4,1])
cCs5 = cCs_from_electroneutrality(c5[1,1],c5[2,1],c5[3,1],c5[4,1])
H4 = sechenov_H_CO2(c4[1,1],c4[2,1],c4[3,1],c4[4,1])
H5 = sechenov_H_CO2(c5[1,1],c5[2,1],c5[3,1],c5[4,1])
oh4 = sum(c4[2,:] .* r4.cl.dx)/L_CL; oh5 = sum(c5[2,:] .* r5.cl.dx)/L_CL
i4 = compute_currents(u4, r4.ly, r4.cl, V); i5 = compute_currents(u5, r5.ly, r5.cl, V)
@printf("At V_RHE = %.3f, CL cell 1 (GDL face) unless noted — Stage 4 (fitted) → Stage 5\n", V)
@printf("  c_CO3   %8.1f → %8.1f mol/m³  (%+.0f %%)   [guide 4590 → 2845, −38 %%]\n",
        c4[4,1], c5[4,1], 100*(c5[4,1]-c4[4,1])/c4[4,1])
@printf("  c_Cs    %8.1f → %8.1f mol/m³  (%+.0f %%)   [guide 9369 → 5537, −41 %%]\n",
        cCs4, cCs5, 100*(cCs5-cCs4)/cCs4)
@printf("  c_OH (CL avg) %6.1f → %6.1f mol/m³ (%+.0f %%) [guide 410 → 197, −52 %%]\n",
        oh4, oh5, 100*(oh5-oh4)/oh4)
@printf("  H_eff   %8.3f → %8.3f mol/(m³·atm) (%.2f×) [guide 1.66 → 4.33, 2.6×]\n",
        H4, H5, H5/H4)
@printf("  c_CO2   %8.3f → %8.3f mol/m³      (%.2f×) [guide 1.10 → 3.26, 2.9×]\n",
        c4[5,1], c5[5,1], c5[5,1]/c4[5,1])
@printf("  i_CO    %8.1f → %8.1f mA/cm²      (%.2f×) [guide 86 → 164, 1.90×]\n",
        i4[1]/10, i5[1]/10, i5[1]/i4[1])
@printf("  i_H2    %8.1f → %8.1f mA/cm²      (%.3f×) [guide: falls 3–10 %%]\n",
        i4[2]/10, i5[2]/10, i5[2]/i4[2])
phiL = [u5[phi_L_idx(r5.ly, ix)] for ix in 1:N_CL]
@printf("  φ_L: %.3f mV at x=0 → %.3f mV at x=L_CL   [guide ≈ −17 mV → 0]\n",
        phiL[1]*1000, phiL[end]*1000)
# η comparison
ps4 = V - V_RHE_to_SHE_offset
pH4 = -log10(c4[1,1]/1000); pH5 = -log10(c5[1,1]/1000)
eta4 = ps4 - (U_CO_0 - nernst_fac*pH4)
eta5 = u5[phi_s_cl_idx(r5.ly,1)] - phiL[1] - (U_CO_0 - nernst_fac*pH5)
@printf("  η_CO2R (cell 1): %.4f → %.4f V (%.0f mV less cathodic) [guide −1.9912 → −1.9483, 43 mV]\n",
        eta4, eta5, 1000*(eta5-eta4))
