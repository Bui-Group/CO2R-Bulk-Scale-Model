include("common.jl")

println("── §8 derived constants ──")
@printf("V_RHE_to_SHE_offset = %.5f V   (guide 0.51305)\n", V_RHE_to_SHE_offset)
@printf("nernst_fac          = %.5f V/pH (guide 0.06413)\n", nernst_fac)
@printf("a_v_CL              = %.4e 1/m  (guide 1.54e7)\n", a_v_CL)
@printf("S_pore              = %.4f      (guide 0.558)\n", S_pore)
@printf("a_v_GL              = %.4e 1/m  (guide 4.08e7)\n", a_v_GL)
@printf("D_eff (×1e12)       = %s\n", string(round.(D_eff.*1e12, digits=4)))
@printf("D_eff_Cs            = %.4e      (guide 1.13e-10)\n", D_eff_Cs)
@printf("sigma_s_eff_GDL     = %.2f S/m  (guide ~73)\n", sigma_s_eff_GDL)
@printf("sigma_s_eff_CL      = %.2f S/m  (guide ~11)\n", sigma_s_eff_CL)
@printf("kappa_CL            = %.3e m²   (guide ~7.0e-17)\n", kappa_CL)
@printf("u_elec              = %.4f m/s  (guide 0.651)\n", u_elec)
@printf("Re                  = %.1f      (guide 14648)\n", Re_elec)
@printf("k_MT_OH             = %.4e m/s  (guide 1.45e-4)\n", k_MT[2])
@printf("pf (i0=540)         = %.2f 1/s  (guide 43.1)\n", a_v_CL*540/(2*F_const*c_ref))
@printf("H_vanthoff          = %.3f mol/(m³ atm) (guide 18.24)\n", H_vanthoff())
@printf("c_CO2_bc S1/S2      = %.3f mol/m³ (guide ~16.0)\n", H_vanthoff()*p_CO2_inlet_atm)
@printf("omega_inlet         = %s (guide ≈[0.947,0,0,0.053])\n", string(round.(omega_inlet, digits=4)))
@printf("M_avg_inlet         = %.3f g/mol (guide 40.89)\n", M_avg_inlet*1000)
println("D_bin_raw [m²/s] (FSG):")
for i in 1:4, j in 1:4
    i < j && @printf("   D[%d,%d] = %.4e\n", i, j, D_bin_raw[i,j])
end

println("\n── §9.6 buffer equilibrium (1 M CsHCO₃, 50 °C) ──")
e = solve_buffer_equilibrium(1000.0, 1000.0)
@printf("pH      = %.4f   (guide 8.23)\n", e.pH)
@printf("c_H     = %.4e   (guide 5.9e-6)\n", e.H)
@printf("c_OH    = %.4e   (guide 9.3e-3)\n", e.OH)
@printf("c_HCO3  = %.4f    (guide 978)\n", e.HCO3)
@printf("c_CO3   = %.4f    (guide 11)\n", e.CO3)
@printf("c_CO2   = %.4f    (guide 11)\n", e.CO2)
@printf("K_w check  cH*cOH        = %.4e vs %.4e\n", e.H*e.OH, K_w)
@printf("K_2 check  cH*cHCO3/cCO2 = %.4e vs %.4e\n", e.H*e.HCO3/e.CO2, K_2)
@printf("K_3 check  cH*cCO3/cHCO3 = %.4e vs %.4e\n", e.H*e.CO3/e.HCO3, K_3)
@printf("charge balance = %.3e\n", e.H + 1000 - e.OH - e.HCO3 - 2*e.CO3)

c_eq = equilibrium_vector()
S = buffer_sources(c_eq)
@printf("max |S_buf| at equilibrium = %.3e mol/m³/s  (need < 1e-6)\n", maximum(abs.(S)))
@printf("c_Cs from electroneutrality = %.2f mol/m³ (guide 570)\n",
        c_Cs_from_electroneutrality(c_eq))
@printf("H_eff (Sechenov, bulk eq)   = %.3f mol/(m³ atm) (guide ~14.5)\n",
        sechenov_H_CO2(c_eq))
@printf("c_CO2 at GDL face (S3+)     = %.3f mol/m³ (guide ~12.8)\n",
        sechenov_H_CO2(c_eq)*0.88)

println("\n── mesh ──")
cl = make_cl_mesh(); gdl = make_gdl_mesh(); gm = make_gas_mesh(gdl, cl)
@printf("CL  N=%d  dx[1]=%.4e  dx[end]=%.4e  ratio=%.4f  sum=%.6e (L=%.1e)\n",
        cl.N, cl.dx[1], cl.dx[end], cl.dx[1]/cl.dx[end], sum(cl.dx), L_CL)
@printf("GDL N=%d  dx[1]=%.4e  dx[end]=%.4e  ratio=%.4f  sum=%.6e (L=%.1e)\n",
        gdl.N, gdl.dx[1], gdl.dx[end], gdl.dx[1]/gdl.dx[end], sum(gdl.dx), L_GDL)
@printf("GDL x_faces: [%.3e, %.3e]; CL x_faces: [%.3e, %.3e]\n",
        gdl.x_faces[1], gdl.x_faces[end], cl.x_faces[1], cl.x_faces[end])
@printf("gas mesh N=%d, interface face=%d at x=%.3e\n", gm.N, gm.i_int, gm.x_faces[gm.i_int])

println("\n── DOF counts ──")
for s in 1:5
    ly = Layout(cl.N, gdl.N, s)
    @printf("stage %d: N_TOT = %d\n", s, ly.N_TOT)
end

println("\n── §7.2 Biot numbers (Stage 2 check) ──")
for i in 1:5
    @printf("  species %d: Bi = %.4f, k_MT = %.4e\n", i, k_MT[i]*L_CL/D_eff[i], k_MT[i])
end
