# stage5_diagnostics.jl — post-solve physicality diagnostics specific to Stage 5 (§9.5.8, §11
# Stage 5 checks 1-7): Delta phi_L, GDL/CL ohmic-drop split, current-conservation checks.
# Recomputes a handful of intermediate quantities that residual_stage5! computes internally but
# does not return (its signature is fixed by the guide's spec).

function stage5_diagnostics(u::Vector{Float64}, ly::Stage5Layout, cl_mesh::FVMesh, gdl_mesh::FVMesh,
                             c_eq_vec::Vector{Float64}, V_cathode::Float64)
    N_CL = ly.N_CL; N_GDL = ly.N_GDL
    c = unpack_log_conc(u, N_CL)
    phi_s_gdl = unpack_phi_s_gdl5(u, ly)
    phi_s_cl  = unpack_phi_s_cl5(u, ly)
    phi_L     = unpack_phi_L_cl5(u, ly)
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset

    omega_cl = [cl_omega(u, ly, ix) for ix in 1:N_CL]
    p_cl     = [cl_p(u, ly, ix) for ix in 1:N_CL]

    R_CO2R_cell = zeros(N_CL); R_HER_cell = zeros(N_CL)
    for ix in 1:N_CL
        c_ix = view(c, :, ix)
        R_CO2R_cell[ix] = R_CO2R(c_ix[5], phi_s_cl[ix], phi_L[ix], c_ix[1], a_v_CL, lambda_MHC(), i0_CO2R(), n_e_val)
        R_HER_cell[ix]  = R_HER(phi_s_cl[ix], phi_L[ix], c_ix[1], a_v_CL, n_e_val)
    end
    i_far_total = n_e_val * F_const * sum((R_CO2R_cell .+ R_HER_cell) .* cl_mesh.dx)

    i_s = solid_face_currents(phi_s_gdl, phi_s_cl, V_applied_SHE, sigma_s_eff_GDL, sigma_s_eff_CL,
                               gdl_mesh, cl_mesh)
    i_s_face1 = i_s[1]

    p_CO2_1 = mole_fraction_CO2(omega_cl[1]) * p_cl[1] / 101325.0
    c_CO2_ghost = sechenov_H_CO2(view(c, :, 1)) * p_CO2_1
    J = zeros(5, N_CL+1)
    # 2026-09-27 post-build change: no CO2(aq) flux at the GDL/CL face in Stage 5 (guide v7.1, Stage 5 note).
    J[5,1] = 0.0   # was: diffusion_flux(c_CO2_ghost, c[5,1], D_eff[5], cl_mesh.dx_between[1])
    for f in 2:N_CL, i in 1:5
        J[i,f] = np_flux(c[i,f-1], c[i,f], phi_L[f-1], phi_L[f], D_eff[i], z_liq[i], cl_mesh.dx_between[f])
    end
    for i in 1:5
        L_eff = D_eff[i] / k_MT[i]
        J[i,N_CL+1] = np_flux(c[i,N_CL], c_eq_vec[i], phi_L[N_CL], 0.0, D_eff[i], z_liq[i], L_eff)
    end
    # Cs+ carries no flux at either CL boundary by construction (§3.1) -> right-boundary i_ionic
    # is exactly the 5-species sum, no separate Cs+ term needed here.
    i_ionic_R = F_const * sum(z_liq[i] * J[i, N_CL+1] for i in 1:5)

    total_solid_drop = phi_s_cl[N_CL] - V_applied_SHE
    gdl_drop = phi_s_gdl[N_GDL] - V_applied_SHE
    cl_drop  = phi_s_cl[N_CL] - phi_s_gdl[N_GDL]
    dphi_L = phi_L[1] - phi_L[N_CL]

    return (i_far_total=i_far_total, i_s_face1=i_s_face1, i_ionic_R=i_ionic_R,
            total_solid_drop=total_solid_drop, gdl_drop=gdl_drop, cl_drop=cl_drop,
            dphi_L=dphi_L, phi_L=phi_L, phi_s_gdl=phi_s_gdl, phi_s_cl=phi_s_cl)
end
