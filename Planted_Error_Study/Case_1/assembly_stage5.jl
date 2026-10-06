# assembly_stage5.jl — Stage 5 monolithic residual (§3.1, §3.4, §11).
# Full electrochemistry with FITTED kinetics: NP migration in the liquid fluxes,
# φ_L (liquid potential, charge balance per CL cell incl. Cs⁺ migration), and φ_s
# (solid Ohm's law on GDL+CL). Liquid kinetics use LOCAL (φ_s − φ_L).
# DOF layout: [ liquid 5N_CL ; gas 4(N_GDL+N_CL) ; φ_s_GDL N_GDL ; φ_s_CL N_CL ; φ_L_CL N_CL ].

# ── Index helpers (gas indices identical to Stage 3/4 → fit cache maps directly) ──
phi_s_gdl_idx(ig, N_CL, N_GDL) = 5*N_CL + 4*(N_GDL+N_CL) + ig
phi_s_cl_idx(ix, N_CL, N_GDL)  = 5*N_CL + 4*(N_GDL+N_CL) + N_GDL + ix
phi_L_idx(ix, N_CL, N_GDL)     = 5*N_CL + 4*(N_GDL+N_CL) + N_GDL + N_CL + ix
n_dof_stage5(N_CL, N_GDL)      = 5*N_CL + 4*(N_GDL+N_CL) + N_GDL + N_CL + N_CL

"Cs⁺ from electroneutrality at CL cell ix (§2): c_OH + c_HCO3 + 2c_CO3 − c_H − C_fix·ε_I."
@inline cCs_cell(c, ix) = c[2,ix] + c[3,ix] + 2*c[4,ix] - c[1,ix] - C_fix*eps_I

function residual_stage5!(F::AbstractVector{<:Real}, u::AbstractVector{<:Real},
                          V_cathode::Real, cl_mesh, gdl_mesh, c_eq::AbstractVector)
    ElT = eltype(u)
    N = cl_mesh.N; NG = gdl_mesh.N; Ngas = NG + N
    c = unpack_log_conc(u, N)
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset
    phi_L = ElT[u[phi_L_idx(ix, N, NG)] for ix in 1:N]
    phi_s_cl = ElT[u[phi_s_cl_idx(ix, N, NG)] for ix in 1:N]
    phi_s_gdl = ElT[u[phi_s_gdl_idx(ig, N, NG)] for ig in 1:NG]

    # ── Kinetics (LOCAL φ_s, φ_L) + phase transfer per CL cell ──
    R_co2r = zeros(ElT, N); R_her = zeros(ElT, N); R_pt = zeros(ElT, N)
    for ix in 1:N
        c_ix = @view c[:, ix]
        R_co2r[ix] = R_CO2R(c_ix[5], phi_s_cl[ix], phi_L[ix], c_ix[1],
                            a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        R_her[ix]  = R_HER(phi_s_cl[ix], phi_L[ix], c_ix[1], a_v_CL, n_e)
        R_pt[ix]   = a_v_GL * k_GL() * (sechenov_H(c_ix) * pCO2_atm(u, NG+ix, N, NG) - c_ix[5])
    end

    # ── Liquid Nernst-Planck fluxes (diffusion + migration) ──
    Jl = zeros(ElT, 5, N + 1)
    c_CO2_bc = sechenov_H(@view c[:, 1]) * pCO2_atm(u, NG+1, N, NG)
    Jl[5, 1] = diffusion_flux(c_CO2_bc, c[5, 1], D_eff[5], cl_mesh.dx_face_left)   # CO₂ Henry (neutral)
    # Jl[1:4,1] = 0 (ions no-flux at GDL side)
    for f in 2:N, i in 1:5
        Jl[i, f] = np_flux(c[i, f-1], c[i, f], phi_L[f-1], phi_L[f], D_eff[i], z[i], cl_mesh.dx_cc[f-1])
    end
    for i in 1:5      # right boundary: NP flux to equilibrium reservoir at φ_L=0, L_eff = D_eff/k_MT
        Leff = D_eff[i] / k_MT[i]
        Jl[i, N+1] = np_flux(c[i, N], c_eq[i], phi_L[N], zero(ElT), D_eff[i], z[i], Leff)
    end

    # ── Liquid species residuals ──
    for ix in 1:N
        buf = buffer_sources(@view c[:, ix])
        for i in 1:5
            src = eps_I * buf[i]
            if i == 2; src += 2*R_co2r[ix] + 2*R_her[ix]; end
            if i == 5; src += -R_co2r[ix] + R_pt[ix];     end
            F[liquid_idx(ix, i)] = Jl[i, ix] - Jl[i, ix+1] + src * cl_mesh.dx[ix]
        end
    end

    # ── Gas-phase residual (Stefan-Maxwell + Darcy, identical to Stage 4) ──
    Jg = zeros(ElT, 3, Ngas + 1); rhou = zeros(ElT, Ngas + 1)
    w1, p1 = gas_state(u, 1, N, NG)
    J1, ru1 = sm_darcy_face_flux(_W_INLET, ElT(p_total), w1, p1, bf_GDL, kappa_GDL, gdl_mesh.dx_face_left)
    Jg[1,1], Jg[2,1], Jg[3,1] = J1[1], J1[2], J1[3]; rhou[1] = ru1
    for f in 2:Ngas
        wL, pL = gas_state(u, f-1, N, NG); wR, pR = gas_state(u, f, N, NG)
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
    for g in 1:Ngas
        dxg = g <= NG ? gdl_mesh.dx[g] : cl_mesh.dx[g - NG]
        S1 = zero(ElT); S2 = zero(ElT); S3 = zero(ElT)
        if g > NG
            ix = g - NG
            S1 = -R_pt[ix]*M_gas[1]; S2 = R_co2r[ix]*M_gas[2]; S3 = R_her[ix]*M_gas[3]
        end
        F[gas_dof(g,1,N,NG)] = Jg[1,g] - Jg[1,g+1] + S1*dxg
        F[gas_dof(g,2,N,NG)] = Jg[2,g] - Jg[2,g+1] + S2*dxg
        F[gas_dof(g,3,N,NG)] = Jg[3,g] - Jg[3,g+1] + S3*dxg
        F[gas_dof(g,4,N,NG)] = rhou[g] - rhou[g+1] + (S1+S2+S3)*dxg
    end

    # ── φ_L residual: ionic charge balance (incl. Cs⁺ migration, §3.1 / §9.4 item 8) ──
    # J_Cs at interior faces; J_Cs = 0 at both CL boundaries (no Cs flux).
    i_L = zeros(ElT, N + 1)
    i_L[1] = zero(ElT)                                   # no ionic current at GDL/CL face
    for f in 2:N
        cCsL = cCs_cell(c, f-1); cCsR = cCs_cell(c, f)
        J_Cs = np_flux(cCsL, cCsR, phi_L[f-1], phi_L[f], D_eff_Cs, z_Cs, cl_mesh.dx_cc[f-1])
        i_L[f] = F_const * (sum(z[i]*Jl[i,f] for i in 1:5) + z_Cs*J_Cs)
    end
    i_L[N+1] = F_const * sum(z[i]*Jl[i,N+1] for i in 1:5)  # Cs no-flux at EL boundary ⇒ no Cs term
    for ix in 1:N
        i_far = n_e * F_const * (R_co2r[ix] + R_her[ix])
        F[phi_L_idx(ix, N, NG)] = (i_L[ix+1] - i_L[ix]) + i_far * cl_mesh.dx[ix]
    end
    F[phi_L_idx(N, N, NG)] = phi_L[N]                    # gauge override φ_L[N_CL] = 0

    # ── φ_s residual: solid Ohm's law on GDL+CL (§3.4) ──
    # Solid faces 1..Ngas+1; cell g (1..NG GDL, NG+1..Ngas CL). i_s_face = −σ_eff·Δφ_s/dx.
    is = zeros(ElT, Ngas + 1)
    is[1] = -sigma_s_eff_GDL * (phi_s_gdl[1] - V_applied_SHE) / gdl_mesh.dx_face_left   # collector ghost
    for f in 2:Ngas
        if f <= NG
            σf = sigma_s_eff_GDL; dxc = gdl_mesh.dx_cc[f-1]
            φL_ = phi_s_gdl[f-1]; φR_ = phi_s_gdl[f]
        elseif f == NG + 1
            σf = sigma_s_eff_iface; dxc = gdl_mesh.dx_face_right + cl_mesh.dx_face_left
            φL_ = phi_s_gdl[NG]; φR_ = phi_s_cl[1]
        else
            σf = sigma_s_eff_CL; dxc = cl_mesh.dx_cc[f - NG - 1]
            φL_ = phi_s_cl[f-1-NG]; φR_ = phi_s_cl[f-NG]
        end
        is[f] = -σf * (φR_ - φL_) / dxc
    end
    is[Ngas+1] = zero(ElT)                               # Neumann ∇φ_s=0 at electrolyte side
    for ig in 1:NG                                       # GDL: no faradaic source
        F[phi_s_gdl_idx(ig, N, NG)] = is[ig+1] - is[ig]
    end
    for ix in 1:N                                        # CL: −i_far_vol·dx (cathodic, ∇·i_s=+i_far)
        i_far = n_e * F_const * (R_co2r[ix] + R_her[ix])
        F[phi_s_cl_idx(ix, N, NG)] = (is[NG+ix+1] - is[NG+ix]) - i_far * cl_mesh.dx[ix]
    end
    return F
end

"Build the Stage-5 anchor initial guess (§9.5.5): equilibrium liquid,
inlet gas, φ_s = V_applied_SHE everywhere, φ_L = 0."
function build_anchor_u(N_CL, N_GDL, c_eq, V_applied_SHE)
    u = zeros(n_dof_stage5(N_CL, N_GDL))
    lnc = log.(c_eq)
    for ix in 1:N_CL; u[liquid_idx(ix,1):liquid_idx(ix,5)] .= lnc; end
    init_gas_inlet!(u, N_CL, N_GDL)
    for ig in 1:N_GDL; u[phi_s_gdl_idx(ig,N_CL,N_GDL)] = V_applied_SHE; end
    for ix in 1:N_CL;  u[phi_s_cl_idx(ix,N_CL,N_GDL)]  = V_applied_SHE; end
    # φ_L already 0
    return u
end

"Append φ_s = V_applied_SHE, φ_L = 0 to a Stage-4 (fit-cache) solution u4."
function make_u0_from_stage4(u4, N_CL, N_GDL, V_cathode)
    u = zeros(n_dof_stage5(N_CL, N_GDL))
    ngas_end = 5*N_CL + 4*(N_GDL+N_CL)
    u[1:ngas_end] .= u4[1:ngas_end]
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset
    for ig in 1:N_GDL; u[phi_s_gdl_idx(ig,N_CL,N_GDL)] = V_applied_SHE; end
    for ix in 1:N_CL;  u[phi_s_cl_idx(ix,N_CL,N_GDL)]  = V_applied_SHE; end
    return u
end

"Stage-5 anchor guess at V_c = 0 (§9.5.5): equilibrium liquid, inlet gas, φ_s = 0 on GDL+CL
(V_applied_SHE = 0 at zero overpotential) and φ_L = 0 — from physics, never from a cache."
make_u0_stage5_anchor(N_CL, N_GDL, c_eq) = build_anchor_u(N_CL, N_GDL, c_eq, 0.0)

"φ_s-shift predictor fallback (§9.5.4): when no tangent is available, move every φ_s DOF by
the voltage step dV (the collector boundary value moves by exactly dV)."
function shift_phis!(u::Vector{Float64}, dV::Float64, N_CL::Int, N_GDL::Int)
    for ig in 1:N_GDL; u[phi_s_gdl_idx(ig, N_CL, N_GDL)] += dV; end
    for ix in 1:N_CL;  u[phi_s_cl_idx(ix, N_CL, N_GDL)]  += dV; end
    return u
end

# ── Stage 5 current extractors (use local φ_s, φ_L) ──
function s5_currents(u, cl_mesh, N_GDL)
    N = cl_mesh.N; c = unpack_log_conc(u, N)
    iCO = 0.0; iH2 = 0.0
    for ix in 1:N
        ps = u[phi_s_cl_idx(ix,N,N_GDL)]; pL = u[phi_L_idx(ix,N,N_GDL)]
        rco = R_CO2R(c[5,ix], ps, pL, c[1,ix], a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        rh  = R_HER(ps, pL, c[1,ix], a_v_CL, n_e)
        iCO += 2*F_const*rco*cl_mesh.dx[ix]; iH2 += 2*F_const*rh*cl_mesh.dx[ix]
    end
    return iCO, iH2   # A/m²
end
