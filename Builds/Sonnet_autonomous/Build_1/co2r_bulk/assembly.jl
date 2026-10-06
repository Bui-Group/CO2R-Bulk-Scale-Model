##############################################################################
# assembly.jl — full residual assembly for Stages 1-5 (§11), current extractors
##############################################################################

# ── Shared gas-phase residual assembly (Stages 3, 4, 5) ────────────────────
"""
    assemble_gas_residual!(F, u, ly, gdl_mesh, cl_mesh, S_CO2_cell, S_CO_cell,
                            S_H2_cell, S_mass_cell; with_darcy)

Writes the 4*(N_GDL+N_CL) gas-DOF residual rows (species j=1,2,3 + pressure).
`with_darcy=false` pins p ≡ p_total everywhere (Stage 3 placeholder, §11).
`with_darcy=true` solves the true Darcy continuity balance (Stage 4+).
"""
function assemble_gas_residual!(F::AbstractVector, u::AbstractVector, ly::GasCapableLayout,
                                 gdl_mesh::Mesh1D, cl_mesh::Mesh1D,
                                 S_CO2_cell::AbstractVector, S_CO_cell::AbstractVector,
                                 S_H2_cell::AbstractVector, S_mass_cell::AbstractVector;
                                 with_darcy::Bool)
    N_GDL = gdl_mesh.N; N_CL = cl_mesh.N
    ElT = eltype(u)
    Nt = N_GDL + N_CL

    J1 = zeros(ElT, Nt+1); J2 = zeros(ElT, Nt+1); J3 = zeros(ElT, Nt+1); rhou = zeros(ElT, Nt+1)

    # left boundary ghost (flow-field inlet, §7.3)
    w1,w2,w3,p1 = gdl_gas(u, ly, 1)
    (J1[1], J2[1], J3[1], rhou[1]) = gas_face_flux(omega_inlet[1], omega_inlet[2], omega_inlet[3], p_total,
                                                     w1, w2, w3, p1,
                                                     D_eff_GDL_mat, kappa_GDL, gdl_mesh.dx_left, with_darcy)
    # interior GDL faces
    for ig in 1:(N_GDL-1)
        wL = gdl_gas(u, ly, ig); wR = gdl_gas(u, ly, ig+1)
        (J1[ig+1], J2[ig+1], J3[ig+1], rhou[ig+1]) = gas_face_flux(wL[1],wL[2],wL[3],wL[4],
                                                                     wR[1],wR[2],wR[3],wR[4],
                                                                     D_eff_GDL_mat, kappa_GDL,
                                                                     gdl_mesh.dx_between[ig], with_darcy)
    end
    # GDL/CL interface (harmonic mean D and kappa, §8.7/§11 Stage 4)
    wL = gdl_gas(u, ly, N_GDL); wR = cl_gas(u, ly, 1)
    dx_int = gdl_mesh.dx_right + cl_mesh.dx_left
    (J1[N_GDL+1], J2[N_GDL+1], J3[N_GDL+1], rhou[N_GDL+1]) = gas_face_flux(wL[1],wL[2],wL[3],wL[4],
                                                                             wR[1],wR[2],wR[3],wR[4],
                                                                             D_eff_int_mat, kappa_int, dx_int, with_darcy)
    # interior CL faces
    for ix in 1:(N_CL-1)
        wL = cl_gas(u, ly, ix); wR = cl_gas(u, ly, ix+1)
        f = N_GDL + 1 + ix
        (J1[f], J2[f], J3[f], rhou[f]) = gas_face_flux(wL[1],wL[2],wL[3],wL[4],
                                                          wR[1],wR[2],wR[3],wR[4],
                                                          D_eff_CL_mat, kappa_CL,
                                                          cl_mesh.dx_between[ix], with_darcy)
    end
    # right boundary: zero flux (gas cannot penetrate electrolyte, §7.3/§11 Stage 4)
    J1[Nt+1] = zero(ElT); J2[Nt+1] = zero(ElT); J3[Nt+1] = zero(ElT); rhou[Nt+1] = zero(ElT)

    for ig in 1:N_GDL
        F[gas_gdl_idx(ly,ig,1)] = J1[ig] - J1[ig+1]
        F[gas_gdl_idx(ly,ig,2)] = J2[ig] - J2[ig+1]
        F[gas_gdl_idx(ly,ig,3)] = J3[ig] - J3[ig+1]
    end
    for ix in 1:N_CL
        f = N_GDL + ix
        F[gas_cl_idx(ly,ix,1)] = J1[f] - J1[f+1] + S_CO2_cell[ix]*cl_mesh.dx[ix]
        F[gas_cl_idx(ly,ix,2)] = J2[f] - J2[f+1] + S_CO_cell[ix]*cl_mesh.dx[ix]
        F[gas_cl_idx(ly,ix,3)] = J3[f] - J3[f+1] + S_H2_cell[ix]*cl_mesh.dx[ix]
    end

    if with_darcy
        for ig in 1:N_GDL
            F[gas_gdl_idx(ly,ig,4)] = rhou[ig] - rhou[ig+1]
        end
        for ix in 1:N_CL
            f = N_GDL + ix
            F[gas_cl_idx(ly,ix,4)] = rhou[f] - rhou[f+1] + S_mass_cell[ix]*cl_mesh.dx[ix]
        end
    else
        for ig in 1:N_GDL
            (_,_,_,p_ig) = gdl_gas(u, ly, ig)
            F[gas_gdl_idx(ly,ig,4)] = p_ig - p_total
        end
        for ix in 1:N_CL
            (_,_,_,p_ix) = cl_gas(u, ly, ix)
            F[gas_cl_idx(ly,ix,4)] = p_ix - p_total
        end
    end
    return F
end

# ── Stage 1: liquid NP diffusion-only, Dirichlet BCs (§11 Stage 1) ─────────
function residual_stage1!(F::AbstractVector, u::AbstractVector, V_cathode::Real,
                           cl_mesh::Mesh1D, c_eq_vec::Vector{Float64})
    N = cl_mesh.N
    ElT = eltype(u)
    c = unpack_log_conc(u, N)
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset

    J = zeros(ElT, 5, N+1)
    J[5,1] = diffusion_flux(c_CO2_henry, c[5,1], D_eff_vec[5], cl_mesh.dx_left)
    for i in 1:4; J[i,1] = zero(ElT); end
    for f in 2:N, i in 1:5
        J[i,f] = diffusion_flux(c[i,f-1], c[i,f], D_eff_vec[i], cl_mesh.dx_between[f-1])
    end
    for i in 1:5
        J[i,N+1] = diffusion_flux(c[i,N], c_eq_vec[i], D_eff_vec[i], cl_mesh.dx_right)
    end

    for ix in 1:N
        c_ix = @view c[:,ix]
        r_co2r = R_CO2R(c_ix[5], V_applied_SHE, zero(ElT), c_ix[1], a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        r_her  = R_HER(V_applied_SHE, zero(ElT), c_ix[1], a_v_CL, n_e)
        buf = buffer_sources(c_ix)
        for i in 1:5
            src = eps_I*buf[i]
            i==2 && (src += 2*r_co2r + 2*r_her)
            i==5 && (src -= r_co2r)
            F[liquid_idx(ix,i)] = J[i,ix] - J[i,ix+1] + src*cl_mesh.dx[ix]
        end
    end
    return F
end

# ── Stage 2: Sherwood-Reynolds Neumann electrolyte BC (§11 Stage 2) ────────
function residual_stage2!(F::AbstractVector, u::AbstractVector, V_cathode::Real,
                           cl_mesh::Mesh1D, c_eq_vec::Vector{Float64})
    N = cl_mesh.N
    ElT = eltype(u)
    c = unpack_log_conc(u, N)
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset

    J = zeros(ElT, 5, N+1)
    J[5,1] = diffusion_flux(c_CO2_henry, c[5,1], D_eff_vec[5], cl_mesh.dx_left)
    for i in 1:4; J[i,1] = zero(ElT); end
    for f in 2:N, i in 1:5
        J[i,f] = diffusion_flux(c[i,f-1], c[i,f], D_eff_vec[i], cl_mesh.dx_between[f-1])
    end
    for i in 1:5
        J[i,N+1] = -k_MT_vec[i]*(c_eq_vec[i] - c[i,N])         # convective Neumann, §7.2
    end

    for ix in 1:N
        c_ix = @view c[:,ix]
        r_co2r = R_CO2R(c_ix[5], V_applied_SHE, zero(ElT), c_ix[1], a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        r_her  = R_HER(V_applied_SHE, zero(ElT), c_ix[1], a_v_CL, n_e)
        buf = buffer_sources(c_ix)
        for i in 1:5
            src = eps_I*buf[i]
            i==2 && (src += 2*r_co2r + 2*r_her)
            i==5 && (src -= r_co2r)
            F[liquid_idx(ix,i)] = J[i,ix] - J[i,ix+1] + src*cl_mesh.dx[ix]
        end
    end
    return F
end

# ── Stages 3 & 4: + Stefan-Maxwell gas (+ Darcy in Stage 4) (§11) ──────────
function residual_stage34!(F::AbstractVector, u::AbstractVector, V_cathode::Real,
                            cl_mesh::Mesh1D, gdl_mesh::Mesh1D, c_eq_vec::Vector{Float64},
                            ly::GasLayout; with_darcy::Bool)
    N_CL = cl_mesh.N
    ElT = eltype(u)
    c = unpack_log_conc(u, N_CL)
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset

    # CO2(aq) left BC: Henry's law from the LOCAL CL cell-1 gas state (§7.3)
    w1_1,w2_1,w3_1,p_1 = cl_gas(u, ly, 1)
    xCO2_1 = mole_fraction_CO2(w1_1,w2_1,w3_1)
    pCO2_local_1 = xCO2_1*p_1/101325.0
    H_local_1 = sechenov_H_CO2(@view c[:,1])
    c_CO2_ghost = H_local_1 * pCO2_local_1

    J = zeros(ElT, 5, N_CL+1)
    J[5,1] = diffusion_flux(c_CO2_ghost, c[5,1], D_eff_vec[5], cl_mesh.dx_left)
    for i in 1:4; J[i,1] = zero(ElT); end
    for f in 2:N_CL, i in 1:5
        J[i,f] = diffusion_flux(c[i,f-1], c[i,f], D_eff_vec[i], cl_mesh.dx_between[f-1])
    end
    for i in 1:5
        J[i,N_CL+1] = -k_MT_vec[i]*(c_eq_vec[i] - c[i,N_CL])
    end

    R_CO2R_cell = zeros(ElT, N_CL); R_HER_cell = zeros(ElT, N_CL); R_PT_cell = zeros(ElT, N_CL)
    for ix in 1:N_CL
        c_ix = @view c[:,ix]
        r_co2r = R_CO2R(c_ix[5], V_applied_SHE, zero(ElT), c_ix[1], a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        r_her  = R_HER(V_applied_SHE, zero(ElT), c_ix[1], a_v_CL, n_e)
        R_CO2R_cell[ix] = r_co2r; R_HER_cell[ix] = r_her

        w1,w2,w3,p_g = cl_gas(u, ly, ix)
        xCO2 = mole_fraction_CO2(w1,w2,w3)
        pCO2_local = xCO2*p_g/101325.0
        H_local = sechenov_H_CO2(c_ix)
        R_PT_cell[ix] = a_v_GL * k_GL() * (H_local*pCO2_local - c_ix[5])

        buf = buffer_sources(c_ix)
        for i in 1:5
            src = eps_I*buf[i]
            i==2 && (src += 2*r_co2r + 2*r_her)
            if i==5
                src -= r_co2r
                src += R_PT_cell[ix]
            end
            F[liquid_idx(ix,i)] = J[i,ix] - J[i,ix+1] + src*cl_mesh.dx[ix]
        end
    end

    S_CO2_cell  = [-R_PT_cell[ix]*M_gas[1] for ix in 1:N_CL]
    S_CO_cell   = [ R_CO2R_cell[ix]*M_gas[2] for ix in 1:N_CL]
    S_H2_cell   = [ R_HER_cell[ix]*M_gas[3] for ix in 1:N_CL]
    S_mass_cell = S_CO2_cell .+ S_CO_cell .+ S_H2_cell

    assemble_gas_residual!(F, u, ly, gdl_mesh, cl_mesh, S_CO2_cell, S_CO_cell, S_H2_cell, S_mass_cell;
                            with_darcy=with_darcy)
    return F
end

residual_stage3!(F,u,V_cathode,cl_mesh,gdl_mesh,c_eq_vec,ly::GasLayout) =
    residual_stage34!(F,u,V_cathode,cl_mesh,gdl_mesh,c_eq_vec,ly; with_darcy=false)
residual_stage4!(F,u,V_cathode,cl_mesh,gdl_mesh,c_eq_vec,ly::GasLayout) =
    residual_stage34!(F,u,V_cathode,cl_mesh,gdl_mesh,c_eq_vec,ly; with_darcy=true)

# ── Stage 5: + phi_L (NP migration) + phi_s (solid Ohm's law) (§11 Stage 5) ─
function residual_stage5!(F::AbstractVector, u::AbstractVector, V_cathode::Real,
                           cl_mesh::Mesh1D, gdl_mesh::Mesh1D, c_eq_vec::Vector{Float64},
                           ly::Stage5Layout)
    N_CL = cl_mesh.N; N_GDL = gdl_mesh.N
    ElT = eltype(u)
    c = unpack_log_conc(u, N_CL)
    phi_s_gdl = unpack_phi_s_gdl(u, ly)
    phi_s_cl  = unpack_phi_s_cl(u, ly)
    phi_L     = unpack_phi_L(u, ly)
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset

    w1_1,w2_1,w3_1,p_1 = cl_gas(u, ly, 1)
    xCO2_1 = mole_fraction_CO2(w1_1,w2_1,w3_1)
    pCO2_local_1 = xCO2_1*p_1/101325.0
    H_local_1 = sechenov_H_CO2(@view c[:,1])
    c_CO2_ghost = H_local_1 * pCO2_local_1

    # ── liquid NP fluxes (diffusion + migration, §3.1/§9.1)
    J = zeros(ElT, 5, N_CL+1)
    # 2026-09-27 post-build change: no CO2(aq) flux at the GDL/CL face in Stage 5 (guide v7.1, Stage 5 note).
    J[5,1] = zero(ElT)   # was: diffusion_flux(c_CO2_ghost, c[5,1], D_eff_vec[5], cl_mesh.dx_left)
    for i in 1:4; J[i,1] = zero(ElT); end
    for f in 2:N_CL, i in 1:5
        J[i,f] = np_flux(c[i,f-1], c[i,f], phi_L[f-1], phi_L[f], D_eff_vec[i], z_vec[i], cl_mesh.dx_between[f-1])
    end
    for i in 1:5
        L_eff = D_eff_vec[i] / k_MT_vec[i]
        J[i,N_CL+1] = np_flux(c[i,N_CL], c_eq_vec[i], phi_L[N_CL], zero(ElT), D_eff_vec[i], z_vec[i], L_eff)
    end

    # ── Cs+ migration flux — MUST enter i_L even though Cs+ has no own DOF (§3.1, §9.4 item 8)
    JCs = zeros(ElT, N_CL+1)
    JCs[1] = zero(ElT)
    for f in 2:N_CL
        cCs_L = c_Cs_of(@view c[:,f-1]); cCs_R = c_Cs_of(@view c[:,f])
        JCs[f] = np_flux(cCs_L, cCs_R, phi_L[f-1], phi_L[f], D_eff_Cs, z_Cs, cl_mesh.dx_between[f-1])
    end
    JCs[N_CL+1] = zero(ElT)

    # ── faradaic rates + phase transfer, LOCAL phi_s/phi_L (§4, §15.3)
    R_CO2R_cell = zeros(ElT, N_CL); R_HER_cell = zeros(ElT, N_CL); R_PT_cell = zeros(ElT, N_CL)
    for ix in 1:N_CL
        c_ix = @view c[:,ix]
        phi_L_local = phi_L[ix]
        phi_s_local = phi_s_cl[ix]
        R_CO2R_cell[ix] = R_CO2R(c_ix[5], phi_s_local, phi_L_local, c_ix[1], a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        R_HER_cell[ix]  = R_HER(phi_s_local, phi_L_local, c_ix[1], a_v_CL, n_e)

        w1,w2,w3,p_g = cl_gas(u, ly, ix)
        xCO2 = mole_fraction_CO2(w1,w2,w3)
        pCO2_local = xCO2*p_g/101325.0
        H_local = sechenov_H_CO2(c_ix)
        R_PT_cell[ix] = a_v_GL * k_GL() * (H_local*pCO2_local - c_ix[5])
    end

    # ── liquid species residuals
    for ix in 1:N_CL
        c_ix = @view c[:,ix]
        buf = buffer_sources(c_ix)
        for i in 1:5
            src = eps_I*buf[i]
            i==2 && (src += 2*R_CO2R_cell[ix] + 2*R_HER_cell[ix])
            if i==5
                src -= R_CO2R_cell[ix]
                src += R_PT_cell[ix]
            end
            F[liquid_idx(ix,i)] = J[i,ix] - J[i,ix+1] + src*cl_mesh.dx[ix]
        end
    end

    # ── gas residual (same structure as Stage 4)
    S_CO2_cell  = [-R_PT_cell[ix]*M_gas[1] for ix in 1:N_CL]
    S_CO_cell   = [ R_CO2R_cell[ix]*M_gas[2] for ix in 1:N_CL]
    S_H2_cell   = [ R_HER_cell[ix]*M_gas[3] for ix in 1:N_CL]
    S_mass_cell = S_CO2_cell .+ S_CO_cell .+ S_H2_cell
    assemble_gas_residual!(F, u, ly, gdl_mesh, cl_mesh, S_CO2_cell, S_CO_cell, S_H2_cell, S_mass_cell; with_darcy=true)

    # ── phi_L residual: local ionic charge balance, incl. Cs+ (§3.1)
    i_L_face = zeros(ElT, N_CL+1)
    for f in 1:N_CL+1
        acc = zero(ElT)
        for i in 1:5
            acc += z_vec[i]*J[i,f]
        end
        acc += z_Cs*JCs[f]
        i_L_face[f] = F_const*acc
    end
    for ix in 1:N_CL
        i_far_vol = n_e*F_const*(R_CO2R_cell[ix] + R_HER_cell[ix])
        F[phi_L_idx(ly,ix)] = (i_L_face[ix+1] - i_L_face[ix]) + i_far_vol*cl_mesh.dx[ix]
    end
    F[phi_L_idx(ly,N_CL)] = phi_L[N_CL]          # Dirichlet gauge override (§3.1)

    # ── phi_s residual: solid Ohm's law on GDL+CL (§3.4)
    i_s = solid_face_currents(phi_s_gdl, phi_s_cl, V_applied_SHE, sigma_s_eff_GDL, sigma_s_eff_CL, gdl_mesh, cl_mesh)
    for ig in 1:N_GDL
        F[phi_s_gdl_idx(ly,ig)] = i_s[ig+1] - i_s[ig]
    end
    for ix in 1:N_CL
        i_far_vol = n_e*F_const*(R_CO2R_cell[ix] + R_HER_cell[ix])
        F[phi_s_cl_idx(ly,ix)] = (i_s[N_GDL+1+ix] - i_s[N_GDL+ix]) - i_far_vol*cl_mesh.dx[ix]
    end

    return F
end

# ── Current extractors (§17.4) ──────────────────────────────────────────────
"Simplified-kinetics current extractor (Stages 1-4 & Fit): phi_s=V_applied_SHE, phi_L=0."
function compute_i_CO_simple(u::AbstractVector, V_applied_SHE::Real, cl_mesh::Mesh1D)
    N = cl_mesh.N
    c = unpack_log_conc(u, N)
    i = 0.0
    for ix in 1:N
        r = R_CO2R(c[5,ix], V_applied_SHE, 0.0, c[1,ix], a_v_CL, lambda_MHC(), i0_CO2R(), 2)
        i += 2*F_const*r*cl_mesh.dx[ix]
    end
    return i
end
function compute_i_H2_simple(u::AbstractVector, V_applied_SHE::Real, cl_mesh::Mesh1D)
    N = cl_mesh.N
    c = unpack_log_conc(u, N)
    i = 0.0
    for ix in 1:N
        r = R_HER(V_applied_SHE, 0.0, c[1,ix], a_v_CL, 2)
        i += 2*F_const*r*cl_mesh.dx[ix]
    end
    return i
end

"Full-physics current extractor (Stage 5): local phi_s_cl, phi_L."
function compute_i_CO_stage5(u::AbstractVector, cl_mesh::Mesh1D, ly::Stage5Layout)
    N = cl_mesh.N
    c = unpack_log_conc(u, N)
    phi_s_cl = unpack_phi_s_cl(u, ly); phi_L = unpack_phi_L(u, ly)
    i = 0.0
    for ix in 1:N
        r = R_CO2R(c[5,ix], phi_s_cl[ix], phi_L[ix], c[1,ix], a_v_CL, lambda_MHC(), i0_CO2R(), 2)
        i += 2*F_const*r*cl_mesh.dx[ix]
    end
    return i
end
function compute_i_H2_stage5(u::AbstractVector, cl_mesh::Mesh1D, ly::Stage5Layout)
    N = cl_mesh.N
    c = unpack_log_conc(u, N)
    phi_s_cl = unpack_phi_s_cl(u, ly); phi_L = unpack_phi_L(u, ly)
    i = 0.0
    for ix in 1:N
        r = R_HER(phi_s_cl[ix], phi_L[ix], c[1,ix], a_v_CL, 2)
        i += 2*F_const*r*cl_mesh.dx[ix]
    end
    return i
end

# ── Anchor / initial-guess builders (§9.5.5, §9.6, §17.8) ──────────────────
function make_u0_liquid_equilibrium(c_eq_vec::Vector{Float64}, N_CL::Int)
    u0 = zeros(5*N_CL)
    lnc = log.(c_eq_vec)
    for ix in 1:N_CL, i in 1:5
        u0[liquid_idx(ix,i)] = lnc[i]
    end
    return u0
end

function make_u0_gas_equilibrium(c_eq_vec::Vector{Float64}, ly::GasLayout)
    u0 = zeros(ndof(ly))
    lnc = log.(c_eq_vec)
    for ix in 1:ly.N_CL, i in 1:5
        u0[liquid_idx(ix,i)] = lnc[i]
    end
    for ig in 1:ly.N_GDL
        u0[gas_gdl_idx(ly,ig,1)] = omega_inlet[1]
        u0[gas_gdl_idx(ly,ig,2)] = omega_inlet[2]
        u0[gas_gdl_idx(ly,ig,3)] = omega_inlet[3]
        u0[gas_gdl_idx(ly,ig,4)] = p_total
    end
    for ix in 1:ly.N_CL
        u0[gas_cl_idx(ly,ix,1)] = omega_inlet[1]
        u0[gas_cl_idx(ly,ix,2)] = omega_inlet[2]
        u0[gas_cl_idx(ly,ix,3)] = omega_inlet[3]
        u0[gas_cl_idx(ly,ix,4)] = p_total
    end
    return u0
end

"Stage 5 V_c=0 anchor: liquid+gas equilibrium, phi_s=phi_L=0 (§9.5.5/§17.8)."
function make_u0_stage5_anchor(c_eq_vec::Vector{Float64}, ly::Stage5Layout)
    gas_ly = GasLayout(ly.N_CL, ly.N_GDL)
    u0_prefix = make_u0_gas_equilibrium(c_eq_vec, gas_ly)
    u_full = zeros(ndof(ly))
    u_full[1:length(u0_prefix)] .= u0_prefix
    return u_full   # phi_s, phi_L default to 0.0
end
