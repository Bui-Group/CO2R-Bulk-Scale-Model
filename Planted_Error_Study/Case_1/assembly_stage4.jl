# assembly_stage4.jl — Stage 4 monolithic residual (§Stage 4).
# Finalizes the Darcy pressure equation (mass continuity, replaces the pinned p of
# Stage 3) and activates gas convection in the species flux. Liquid block and
# kinetics unchanged from Stage 3 (φ_s=V_applied, φ_L=0).
#  - GDL inlet: p = p_total (Dirichlet via ghost on the leftmost Darcy face).
#  - CL/electrolyte: dp/dx = 0 (no gas flow → ρu = 0 at the right face).
#  - GDL/CL interface: harmonic-mean permeability.

function residual_stage4!(F::AbstractVector{<:Real}, u::AbstractVector{<:Real},
                          V_cathode::Real, cl_mesh, gdl_mesh, c_eq::AbstractVector)
    ElT = eltype(u)
    N = cl_mesh.N; NG = gdl_mesh.N; Ngas = NG + N
    c = unpack_log_conc(u, N)
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset

    # ── Kinetics + phase transfer per CL cell ──
    R_co2r = zeros(ElT, N); R_her = zeros(ElT, N); R_pt = zeros(ElT, N)
    for ix in 1:N
        c_ix = @view c[:, ix]
        R_co2r[ix] = R_CO2R(c_ix[5], V_applied_SHE, zero(ElT), c_ix[1],
                            a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        R_her[ix]  = R_HER(V_applied_SHE, zero(ElT), c_ix[1], a_v_CL, n_e)
        H_loc = sechenov_H(c_ix)
        R_pt[ix] = a_v_GL * k_GL() * (H_loc * pCO2_atm(u, NG+ix, N, NG) - c_ix[5])
    end

    # ── Liquid block (identical to Stage 3) ──
    Jl = zeros(ElT, 5, N + 1)
    c_CO2_bc = sechenov_H(@view c[:, 1]) * pCO2_atm(u, NG+1, N, NG)
    Jl[5, 1] = diffusion_flux(c_CO2_bc, c[5, 1], D_eff[5], cl_mesh.dx_face_left)
    for f in 2:N, i in 1:5
        Jl[i, f] = diffusion_flux(c[i, f-1], c[i, f], D_eff[i], cl_mesh.dx_cc[f-1])
    end
    for i in 1:5
        Jl[i, N+1] = k_MT[i] * (c[i, N] - c_eq[i])
    end
    for ix in 1:N
        buf = buffer_sources(@view c[:, ix])
        for i in 1:5
            src = eps_I * buf[i]
            if i == 2; src += 2*R_co2r[ix] + 2*R_her[ix]; end
            if i == 5; src += -R_co2r[ix] + R_pt[ix];     end
            F[liquid_idx(ix, i)] = Jl[i, ix] - Jl[i, ix+1] + src * cl_mesh.dx[ix]
        end
    end

    # ── Gas: Stefan-Maxwell diffusion + Darcy convection + mass continuity ──
    Jg = zeros(ElT, 3, Ngas + 1)
    rhou = zeros(ElT, Ngas + 1)
    # f = 1: flow-field Dirichlet (inlet ω, p_total) → GDL cell 1
    w1, p1 = gas_state(u, 1, N, NG)
    J1, ru1 = sm_darcy_face_flux(_W_INLET, ElT(p_total), w1, p1, bf_GDL, kappa_GDL, gdl_mesh.dx_face_left)
    Jg[1,1], Jg[2,1], Jg[3,1] = J1[1], J1[2], J1[3]; rhou[1] = ru1
    for f in 2:Ngas
        wL, pL = gas_state(u, f-1, N, NG)
        wR, pR = gas_state(u, f,   N, NG)
        if f <= NG
            bf = bf_GDL; kap = kappa_GDL; dxc = gdl_mesh.dx_cc[f-1]
        elseif f == NG + 1
            bf = 2*bf_GDL*bf_CL/(bf_GDL + bf_CL); kap = kappa_interface
            dxc = gdl_mesh.dx_face_right + cl_mesh.dx_face_left
        else
            bf = bf_CL; kap = kappa_CL; dxc = cl_mesh.dx_cc[f - NG - 1]
        end
        Jf, ruf = sm_darcy_face_flux(wL, pL, wR, pR, bf, kap, dxc)
        Jg[1,f], Jg[2,f], Jg[3,f] = Jf[1], Jf[2], Jf[3]; rhou[f] = ruf
    end
    # f = Ngas+1: CL/electrolyte — no gas flow (dp/dx=0) and zero species flux (already 0)

    for g in 1:Ngas
        dxg = g <= NG ? gdl_mesh.dx[g] : cl_mesh.dx[g - NG]
        S1 = zero(ElT); S2 = zero(ElT); S3 = zero(ElT)
        if g > NG
            ix = g - NG
            S1 = -R_pt[ix]   * M_gas[1]
            S2 =  R_co2r[ix] * M_gas[2]
            S3 =  R_her[ix]  * M_gas[3]
        end
        F[gas_dof(g,1,N,NG)] = Jg[1,g] - Jg[1,g+1] + S1 * dxg
        F[gas_dof(g,2,N,NG)] = Jg[2,g] - Jg[2,g+1] + S2 * dxg
        F[gas_dof(g,3,N,NG)] = Jg[3,g] - Jg[3,g+1] + S3 * dxg
        # Darcy mass continuity: ρu_left − ρu_right + S_mass·dx
        Smass = S1 + S2 + S3
        F[gas_dof(g,4,N,NG)] = rhou[g] - rhou[g+1] + Smass * dxg
    end
    return F
end
