include("model.jl")
using Printf

println("="^72)
println("§9.6 / §12.1  Equilibrium sanity")
eq = solve_buffer_equilibrium(1000.0, 1000.0)
@printf("pH      = %.4f   (expect ~8.23)\n", eq.pH)
@printf("c_H     = %.4e  c_OH  = %.4e\n", eq.H, eq.OH)
@printf("c_HCO3  = %.4f   c_CO3 = %.4f   c_CO2 = %.4f\n", eq.HCO3, eq.CO3, eq.CO2)
@printf("K_w chk : %.4e vs %.4e\n", eq.H*eq.OH, K_w)
@printf("K_2 chk : %.4e vs %.4e\n", eq.H*eq.HCO3/eq.CO2, K_2)
@printf("K_3 chk : %.4e vs %.4e\n", eq.H*eq.CO3/eq.HCO3, K_3)
@printf("charge  : %.4e\n", eq.H + 1000 - eq.OH - eq.HCO3 - 2*eq.CO3)
c_eq = [eq.H, eq.OH, eq.HCO3, eq.CO3, eq.CO2]
S = buffer_sources(c_eq)
@printf("max|S_buf| = %.4e  (must be < 1e-6)\n", maximum(abs.(S)))
@assert maximum(abs.(S)) < 1e-6
@printf("c_Cs from electroneutrality = %.2f  (expect 570)\n", c_Cs_from_electroneutrality(c_eq))
@printf("H_eff vantHoff = %.4f  (expect 18.24)\n", H_vantHoff)
@printf("H_eff Sechenov = %.4f  (expect ~14.5)\n", sechenov_H_CO2(c_eq))
@printf("c_CO2 at GDL face (S3+) = %.4f (expect ~12.8)\n", sechenov_H_CO2(c_eq)*0.88)
@printf("c_CO2 at GDL face (S1-2) = %.4f (expect ~16.0)\n", H_vantHoff*0.88)

println("\n"*"="^72)
println("Derived constants")
@printf("nernst_fac = %.6f (0.06413)   offset = %.6f (0.51305)\n", nernst_fac, V_RHE_to_SHE_offset)
@printf("S_pore = %.4f  a_v_CL = %.4e  a_v_GL = %.4e\n", S_pore, a_v_CL, a_v_GL)
@printf("D_eff  = %s\n", string(round.(D_eff, sigdigits=4)))
@printf("D_eff_Cs = %.4e (expect 1.13e-10)\n", D_eff_Cs)
@printf("Re = %.1f (expect 14648)   k_MT_OH = %.4e (expect 1.45e-4)\n", Re_ch, k_MT[2])
@printf("k_MT = %s\n", string(round.(k_MT, sigdigits=4)))
@printf("sigma_eff GDL/CL = %.2f / %.2f  (73 / 11)\n", sigma_s_eff_GDL, sigma_s_eff_CL)
@printf("kappa_CL = %.4e (expect ~6.9e-17)\n", kappa_CL)
@printf("omega_inlet = %s (expect [0.947,0,0,0.053])\n", string(round.(omega_inlet, sigdigits=4)))
@printf("M_avg_inlet = %.4f g/mol (expect 40.89)\n", M_avg_inlet*1000)
@printf("D_bin_raw[CO2,H2O] = %.4e m^2/s\n", D_bin_raw[1,4])
pf = a_v_CL*540/(n_e*F_const*c_ref)
@printf("MHC prefactor pf = %.3f 1/s (expect 43.1)\n", pf)

println("\n"*"="^72)
println("Mesh")
cl = make_cl_mesh(80, L_CL); gdl = make_gdl_mesh(120, L_GDL); gm = GasMesh(gdl, cl)
@printf("CL:  sum(dx)=%.6e (L=%.1e) dx1/dxN=%.4f faces %.3e..%.3e\n",
        sum(cl.dx), L_CL, cl.dx[1]/cl.dx[end], cl.x_faces[1], cl.x_faces[end])
@printf("GDL: sum(dx)=%.6e (L=%.1e) dx1/dxN=%.4f faces %.3e..%.3e\n",
        sum(gdl.dx), L_GDL, gdl.dx[1]/gdl.dx[end], gdl.x_faces[1], gdl.x_faces[end])
@printf("GasMesh N=%d  sum(dx)=%.6e (expect %.6e)  dxb[NG+1]=%.4e\n",
        gm.N, sum(gm.dx), L_CL+L_GDL, gm.dx_between[gm.N_GDL+1])
@assert abs(sum(cl.dx) - L_CL) < 1e-18
@assert abs(sum(gdl.dx) - L_GDL) < 1e-16

println("\n"*"="^72)
println("DOF counts")
for s in 1:5
    ly = Layout(s, 80, 120)
    @printf("stage %d: N_TOT = %d\n", s, ly.N_TOT)
end
@assert Layout(5,80,120).N_TOT == 1480
@assert Layout(3,80,120).N_TOT == 1200
@assert Layout(1,80,120).N_TOT == 400
println("\nALL SANITY CHECKS PASSED")
