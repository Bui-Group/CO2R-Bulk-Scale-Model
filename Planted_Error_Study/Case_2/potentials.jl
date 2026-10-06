# ============================================================================
# potentials.jl — Stage 5: φ_L (NP migration, monolithic charge balance, §3.1) +
# φ_s (solid Ohm's law on GDL+CL, §3.4). Full coupled electrochemistry.
#
# DOF layout (§11): u = [ ln(c)_5N_CL ; gas_4(N_GDL+N_CL) ;
#                         φ_s_GDL_N_GDL ; φ_s_CL_N_CL ; φ_L_CL_N_CL ]
# The liquid+gas blocks share the Stage 3/4 layout, so the GasLayout gas helpers
# (gas_cell_omega/gas_cell_p/gas_resid_idx/pCO2_atm) apply unchanged.
# ============================================================================

@inline _phi_s_base(ly::GasLayout) = 5*ly.N_CL + 4*(ly.N_GDL + ly.N_CL)
@inline phi_s_gdl_idx5(ly::GasLayout, ig::Int) = _phi_s_base(ly) + ig
@inline phi_s_cl_idx5(ly::GasLayout, ix::Int)  = _phi_s_base(ly) + ly.N_GDL + ix
@inline phi_L_idx5(ly::GasLayout, ix::Int)     = _phi_s_base(ly) + ly.N_GDL + ly.N_CL + ix
@inline ndof_stage5(ly::GasLayout) = 5*ly.N_CL + 4*(ly.N_GDL + ly.N_CL) + ly.N_GDL + 2*ly.N_CL

@inline sigma_solid_cell(ly::GasLayout, c::Int) = c <= ly.N_GDL ? sigma_s_eff_GDL : sigma_s_eff_CL

# Cs⁺ effective diffusivity and Sherwood length (for its ionic current in i_L).
# Cs⁺ is recovered from electroneutrality but is a real current-carrying ion.
const D_eff_Cs = eps_I^1.5 * alpha_ion * D_w_Cs
const L_eff_Cs = D_eff_Cs / k_MT_of(D_w_Cs)

# Stage 5 residual. V_cathode in V vs RHE@pH8; V_applied_SHE = V_cathode − offset.
# gas_mesh is the combined GDL+CL mesh (also the φ_s solid mesh domain).
function residual_stage5!(F::AbstractVector{ElT}, u::AbstractVector{ElT},
                          V_cathode::Real, cl_mesh::Mesh, gas_mesh::Mesh,
                          c_eq_vec::AbstractVector{<:Real}, ly::GasLayout) where {ElT<:Real}
    N = cl_mesh.N; Ng = gas_mesh.N; NGDL = ly.N_GDL
    c = unpack_log_conc(u, N)
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset

    # Unpack potentials
    phi_s = Vector{ElT}(undef, Ng)
    for ig in 1:NGDL; phi_s[ig] = u[phi_s_gdl_idx5(ly, ig)]; end
    for ix in 1:N;    phi_s[NGDL + ix] = u[phi_s_cl_idx5(ly, ix)]; end
    phi_L = ElT[u[phi_L_idx5(ly, ix)] for ix in 1:N]

    # Per-CL-cell rates with LOCAL φ_s, φ_L (§4)
    R_co2r = zeros(ElT, N); R_her = zeros(ElT, N); R_pt = zeros(ElT, N)
    for ix in 1:N
        cix = @view c[:, ix]
        ps = u[phi_s_cl_idx5(ly, ix)]; pL = phi_L[ix]
        R_co2r[ix] = R_CO2R(cix[5], ps, pL, cix[1], a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        R_her[ix]  = R_HER(ps, pL, cix[1], a_v_CL, n_e)
        pco2 = pCO2_atm(u, ly, NGDL + ix)
        R_pt[ix] = a_v_GL * k_GL() * (sechenov_H_CO2(cix) * pco2 - cix[5])
    end

    # ── Liquid Nernst-Planck fluxes (diffusion + migration) ──
    dxb = cl_mesh.dx_between; dxfL = cl_mesh.dx_face_left; dx = cl_mesh.dx
    J = zeros(ElT, 5, N+1)
    c_CO2_bc = sechenov_H_CO2(@view c[:, 1]) * pCO2_atm(u, ly, NGDL + 1)
    J[5, 1] = diffusion_flux(c_CO2_bc, c[5, 1], D_eff[5], dxfL)   # CO₂ neutral; ions left no-flux
    for f in 2:N, i in 1:5
        J[i, f] = np_flux(c[i, f-1], c[i, f], phi_L[f-1], phi_L[f], D_eff[i], z_charge[i], dxb[f-1])
    end
    for i in 1:5                                                   # right face: NP to reservoir, gauge φ_L=0
        L_eff = D_eff[i] / k_MT[i]
        J[i, N+1] = np_flux(c[i, N], c_eq_vec[i], phi_L[N], zero(ElT), D_eff[i], z_charge[i], L_eff)
    end

    # ── Liquid species residuals ──
    for ix in 1:N
        buf = buffer_sources(@view c[:, ix])
        for i in 1:5
            src = eps_I * buf[i]
            i == 2 && (src += 2*R_co2r[ix] + 2*R_her[ix])
            i == 5 && (src += -R_co2r[ix] + R_pt[ix])
            F[liquid_idx(ix, i)] = J[i, ix] - J[i, ix+1] + src * dx[ix]
        end
    end

    # ── Gas-phase Stefan-Maxwell + Darcy (identical structure to Stage 4) ──
    Jg = zeros(ElT, 4, Ng+1); rhou = zeros(ElT, Ng+1)
    let p1 = gas_cell_p(u, ly, 1)
        uG = -(kappa_GDL / mu_gas) * (p1 - p_total) / gas_mesh.dx_face_left
        Jf, ru = sm_face_flux_conv(gas_inlet_tuple(), gas_cell_omega(u, ly, 1),
                                   0.5*(p_total + p1), eps_GDL^1.5, gas_mesh.dx_face_left, uG)
        for j in 1:4; Jg[j, 1] = Jf[j]; end
        rhou[1] = ru
    end
    for f in 2:Ng
        bf = harmonic_mean(gas_cell_eps(ly, f-1)^1.5, gas_cell_eps(ly, f)^1.5)
        κf = harmonic_mean(gas_cell_kappa(ly, f-1), gas_cell_kappa(ly, f))
        pL = gas_cell_p(u, ly, f-1); pR = gas_cell_p(u, ly, f)
        uG = -(κf / mu_gas) * (pR - pL) / gas_mesh.dx_between[f-1]
        Jf, ru = sm_face_flux_conv(gas_cell_omega(u, ly, f-1), gas_cell_omega(u, ly, f),
                                   0.5*(pL + pR), bf, gas_mesh.dx_between[f-1], uG)
        for j in 1:4; Jg[j, f] = Jf[j]; end
        rhou[f] = ru
    end
    dxg = gas_mesh.dx
    for cc in 1:Ng
        SCO2 = zero(ElT); SCO = zero(ElT); SH2 = zero(ElT)
        if cc > NGDL
            ix = cc - NGDL
            SCO2 = -R_pt[ix] * M_gas[1]; SCO = R_co2r[ix] * M_gas[2]; SH2 = R_her[ix] * M_gas[3]
        end
        F[gas_resid_idx(ly, cc, 1)] = Jg[1, cc] - Jg[1, cc+1] + SCO2 * dxg[cc]
        F[gas_resid_idx(ly, cc, 2)] = Jg[2, cc] - Jg[2, cc+1] + SCO  * dxg[cc]
        F[gas_resid_idx(ly, cc, 3)] = Jg[3, cc] - Jg[3, cc+1] + SH2  * dxg[cc]
        F[gas_resid_idx(ly, cc, 4)] = rhou[cc] - rhou[cc+1] + (SCO2 + SCO + SH2) * dxg[cc]
    end

    # ── φ_L residual: local ionic charge balance ∂i_L/∂x = −n_e F (R_CO2R+R_HER) (§3.1) ──
    # Cs⁺ ionic current is INCLUDED in i_L (correction to the §3.1 spec, approved
    # 2026-06-28). Cs⁺ is recovered from electroneutrality but is a real current-
    # carrying ion; excluding it makes the φ_L charge balance an exact linear
    # combination of the species mass balances (F_φL = −F Σz F_species), which makes
    # the Jacobian rank-deficient by ~N_CL and Stage 5 unsolvable. Including the Cs⁺
    # NP current makes the charge balance independent (rank deficiency → gauge only).
    c_Cs = ElT[c_Cs_from_EN(@view c[:, ix]) for ix in 1:N]
    c_Cs_eq = c_eq_vec[2] + c_eq_vec[3] + 2*c_eq_vec[4] - c_eq_vec[1] - C_fix * eps_I
    J_Cs = zeros(ElT, N+1)                       # J_Cs[1] = 0 (no-flux at GDL face)
    for f in 2:N
        J_Cs[f] = np_flux(c_Cs[f-1], c_Cs[f], phi_L[f-1], phi_L[f], D_eff_Cs, 1.0, dxb[f-1])
    end
    J_Cs[N+1] = np_flux(c_Cs[N], c_Cs_eq, phi_L[N], zero(ElT), D_eff_Cs, 1.0, L_eff_Cs)

    for ix in 1:N
        iL_R = F_const * (z_charge[1]*J[1,ix+1] + z_charge[2]*J[2,ix+1] + z_charge[3]*J[3,ix+1] +
                          z_charge[4]*J[4,ix+1] + z_charge[5]*J[5,ix+1] + J_Cs[ix+1])
        iL_L = F_const * (z_charge[1]*J[1,ix]   + z_charge[2]*J[2,ix]   + z_charge[3]*J[3,ix]   +
                          z_charge[4]*J[4,ix]   + z_charge[5]*J[5,ix]   + J_Cs[ix])
        i_far = n_e * F_const * (R_co2r[ix] + R_her[ix])
        F[phi_L_idx5(ly, ix)] = (iL_R - iL_L) + i_far * dx[ix]
    end
    # NOTE: no cell-N gauge override. With Cs⁺ included in i_L the φ_L charge balance
    # is no longer redundant with the species balances, and the electrolyte reservoir
    # reference (φ_L = 0 in the right-boundary NP flux) fixes the gauge — a uniform φ_L
    # shift changes the boundary migration current and the kinetics, so it is not a null
    # direction. Keeping cell N's charge balance enforced makes the boundary ionic
    # current carry the migration term and conserves charge at the electrolyte face
    # (the §3.1 override would zero φ_L[N], killing the boundary migration term).

    # ── φ_s residual: solid Ohm's law on GDL+CL (§3.4) ──
    i_s = zeros(ElT, Ng+1)
    i_s[1] = -sigma_s_eff_GDL * (phi_s[1] - V_applied_SHE) / gas_mesh.dx_face_left   # ghost Dirichlet
    for f in 2:Ng
        σf = harmonic_mean(sigma_solid_cell(ly, f-1), sigma_solid_cell(ly, f))
        i_s[f] = -σf * (phi_s[f] - phi_s[f-1]) / gas_mesh.dx_between[f-1]
    end
    # i_s[Ng+1] = 0 (Neumann at electrolyte side)
    for ig in 1:NGDL
        F[phi_s_gdl_idx5(ly, ig)] = i_s[ig+1] - i_s[ig]                              # no faradaic in GDL
    end
    for ix in 1:N
        cc = NGDL + ix
        i_far = n_e * F_const * (R_co2r[ix] + R_her[ix])
        F[phi_s_cl_idx5(ly, ix)] = (i_s[cc+1] - i_s[cc]) - i_far * dx[ix]            # ∇·i_s = +i_far
    end
    return F
end

# Current extractors reading φ_s/φ_L DOFs (Stage 5).
function compute_i_CO_s5(u, ly, cl_mesh)
    c = unpack_log_conc(u, ly.N_CL); i = 0.0
    for ix in 1:ly.N_CL
        ps = u[phi_s_cl_idx5(ly, ix)]; pL = u[phi_L_idx5(ly, ix)]
        r = R_CO2R(c[5,ix], ps, pL, c[1,ix], a_v_CL, lambda_MHC(), i0_CO2R(), 2)
        i += 2 * F_const * r * cl_mesh.dx[ix]
    end
    return i
end
function compute_i_H2_s5(u, ly, cl_mesh)
    c = unpack_log_conc(u, ly.N_CL); i = 0.0
    for ix in 1:ly.N_CL
        ps = u[phi_s_cl_idx5(ly, ix)]; pL = u[phi_L_idx5(ly, ix)]
        r = R_HER(ps, pL, c[1,ix], a_v_CL, 2)
        i += 2 * F_const * r * cl_mesh.dx[ix]
    end
    return i
end

# Stage 5 anchor guess at V_c = 0 (§9.5.5): constructed from physics, never from a
# cache — bulk buffer equilibrium in every CL cell, inlet gas composition and p_total
# in every gas cell, φ_s = 0 on GDL+CL (V_applied_SHE = 0 at zero overpotential), φ_L = 0.
function make_u0_stage5_anchor(ly::GasLayout, c_eq_vec)
    u = zeros(ndof_stage5(ly))
    ln = log.(c_eq_vec)
    for ix in 1:ly.N_CL; u[liquid_idx(ix, 1):liquid_idx(ix, 5)] .= ln; end
    for cc in 1:(ly.N_GDL + ly.N_CL)
        b = _gas_base(ly, cc)
        u[b+1] = omega_inlet[1]; u[b+2] = omega_inlet[2]; u[b+3] = omega_inlet[3]; u[b+4] = p_total
    end
    return u                                     # φ_s and φ_L already 0
end

# φ_s-shift predictor fallback (§9.5.4): the collector boundary value moves by exactly dV.
function shift_phis!(u::AbstractVector, dV::Real, ly::GasLayout)
    for ig in 1:ly.N_GDL; u[phi_s_gdl_idx5(ly, ig)] += dV; end
    for ix in 1:ly.N_CL;  u[phi_s_cl_idx5(ly, ix)]  += dV; end
    return u
end
