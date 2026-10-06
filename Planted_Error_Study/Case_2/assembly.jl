# ============================================================================
# assembly.jl — Residual assembly + current/diagnostic extractors
# Stage 1: liquid NP diffusion-only, Dirichlet both sides, simplified kinetics.
# ============================================================================

# DOF layout (Stage 1): 5 liquid log-concentrations per CL cell.
@inline liquid_idx(ix::Int, i::Int) = 5 * (ix - 1) + i

# Recover physical concentrations (5 × N) from log-DOFs. Type-generic.
function unpack_log_conc(u::AbstractVector{ElT}, N::Int) where {ElT<:Real}
    c = zeros(ElT, 5, N)
    @inbounds for ix in 1:N, i in 1:5
        c[i, ix] = exp(clamp(u[liquid_idx(ix, i)], -50.0, 50.0))
    end
    return c
end

# §9.5.3 weight blocks for each stage's DOF layout (one block per physical field;
# the kernel's iterate-derived weights are pooled block-wise). The layout is
# cell-major inside every block and the blocks are contiguous, in this order:
#   Stages 1–2 : [ liquid log-c : 5·N_CL ]
#   Stages 3–4 : [ liquid | gas GDL : 4·N_GDL | gas CL : 4·N_CL ]                 (GasLayout, gas_transport.jl)
#   Stage 5    : [ … | φ_s GDL : N_GDL | φ_s CL : N_CL | φ_L : N_CL ]             (potentials.jl)
# Written arithmetically so Stages 1–2 need neither gas_transport.jl nor
# potentials.jl; the Stage 3–5 drivers assert agreement with the index helpers.
function dof_blocks(stage::Int; N_CL_::Int = N_CL, N_GDL_::Int = N_GDL)
    nl = 5 * N_CL_
    blocks = UnitRange{Int}[1:nl]
    if stage >= 3
        push!(blocks, nl + 1 : nl + 4*N_GDL_)                                   # gas GDL
        push!(blocks, nl + 4*N_GDL_ + 1 : nl + 4*(N_GDL_ + N_CL_))               # gas CL
    end
    if stage >= 5
        b = nl + 4*(N_GDL_ + N_CL_)
        push!(blocks, b + 1 : b + N_GDL_)                                        # φ_s GDL
        push!(blocks, b + N_GDL_ + 1 : b + N_GDL_ + N_CL_)                       # φ_s CL
        push!(blocks, b + N_GDL_ + N_CL_ + 1 : b + N_GDL_ + 2*N_CL_)             # φ_L
    end
    return blocks
end

# Cs⁺ recovered from electroneutrality (§2). c is (c_H,c_OH,c_HCO3,c_CO3,c_CO2).
@inline c_Cs_from_EN(c::AbstractVector{<:Real}) =
    c[2] + c[3] + 2*c[4] - c[1] - C_fix * eps_I

# ── Stage 1 residual (§11) — integrated FV form ─────────────────────────────
function residual_stage1!(F::AbstractVector{ElT}, u::AbstractVector{ElT},
                          V_cathode::Real, cl_mesh::Mesh,
                          c_eq_vec::AbstractVector{<:Real}) where {ElT<:Real}
    N = cl_mesh.N
    c = unpack_log_conc(u, N)
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset

    dxb  = cl_mesh.dx_between
    dxfL = cl_mesh.dx_face_left
    dxfR = cl_mesh.dx_face_right
    dx   = cl_mesh.dx

    # Pure-diffusion fluxes (NO migration in Stage 1)
    J = zeros(ElT, 5, N+1)
    J[5, 1] = diffusion_flux(c_CO2_henry_stage12, c[5, 1], D_eff[5], dxfL)
    # J[1:4,1] = 0 (ions: no-flux at GDL side) — already zero
    for f in 2:N
        for i in 1:5
            J[i, f] = diffusion_flux(c[i, f-1], c[i, f], D_eff[i], dxb[f-1])
        end
    end
    for i in 1:5
        J[i, N+1] = diffusion_flux(c[i, N], c_eq_vec[i], D_eff[i], dxfR)
    end

    # Cell residuals: J_left − J_right + S × dx
    for ix in 1:N
        c_ix = @view c[:, ix]
        r_co2r = R_CO2R(c_ix[5], V_applied_SHE, 0.0, c_ix[1],
                        a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        r_her  = R_HER(V_applied_SHE, 0.0, c_ix[1], a_v_CL, n_e)
        buf = buffer_sources(c_ix)
        for i in 1:5
            src = eps_I * buf[i]
            i == 2 && (src += 2*r_co2r + 2*r_her)   # OH⁻ produced
            i == 5 && (src -= r_co2r)               # CO₂(aq) consumed
            F[liquid_idx(ix, i)] = J[i, ix] - J[i, ix+1] + src * dx[ix]
        end
    end
    return F
end

# ── Stage 2 residual (§11) — Sherwood–Reynolds Neumann at electrolyte side ──
# Identical to Stage 1 except the right-face flux at x = L_CL is the convective
# mass-transfer condition instead of a Dirichlet to c_eq. §7.2 / §11 Stage 2.
function residual_stage2!(F::AbstractVector{ElT}, u::AbstractVector{ElT},
                          V_cathode::Real, cl_mesh::Mesh,
                          c_eq_vec::AbstractVector{<:Real}) where {ElT<:Real}
    N = cl_mesh.N
    c = unpack_log_conc(u, N)
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset

    dxb  = cl_mesh.dx_between
    dxfL = cl_mesh.dx_face_left
    dx   = cl_mesh.dx

    J = zeros(ElT, 5, N+1)
    J[5, 1] = diffusion_flux(c_CO2_henry_stage12, c[5, 1], D_eff[5], dxfL)
    # J[1:4,1] = 0 (ions: no-flux at GDL side)
    for f in 2:N
        for i in 1:5
            J[i, f] = diffusion_flux(c[i, f-1], c[i, f], D_eff[i], dxb[f-1])
        end
    end
    # Right face: convective Neumann (§11 Stage 2, sign matches NP convention)
    for i in 1:5
        J[i, N+1] = -k_MT[i] * (c_eq_vec[i] - c[i, N])
    end

    for ix in 1:N
        c_ix = @view c[:, ix]
        r_co2r = R_CO2R(c_ix[5], V_applied_SHE, 0.0, c_ix[1],
                        a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        r_her  = R_HER(V_applied_SHE, 0.0, c_ix[1], a_v_CL, n_e)
        buf = buffer_sources(c_ix)
        for i in 1:5
            src = eps_I * buf[i]
            i == 2 && (src += 2*r_co2r + 2*r_her)
            i == 5 && (src -= r_co2r)
            F[liquid_idx(ix, i)] = J[i, ix] - J[i, ix+1] + src * dx[ix]
        end
    end
    return F
end

# ── Stage 3 residual (§11) — SM gas (u_G=0) + R_PT + Sechenov Henry ─────────
# Liquid: diffusion-only (φ_L=0), gas-coupled CO₂ left BC, Sherwood right BC,
#         +R_PT on CO₂(aq). Gas: SM mixture-average (u_G=0), pressure pinned to
#         p_total (placeholder; Darcy finalized in Stage 4).
function residual_stage3!(F::AbstractVector{ElT}, u::AbstractVector{ElT},
                          V_cathode::Real, cl_mesh::Mesh, gas_mesh::Mesh,
                          c_eq_vec::AbstractVector{<:Real}, ly::GasLayout) where {ElT<:Real}
    N = cl_mesh.N; Ng = gas_mesh.N
    c = unpack_log_conc(u, N)
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset

    # Per-CL-cell faradaic + phase-transfer rates (φ_s=V_applied, φ_L=0)
    R_co2r = zeros(ElT, N); R_her = zeros(ElT, N); R_pt = zeros(ElT, N)
    for ix in 1:N
        c_ix = @view c[:, ix]
        R_co2r[ix] = R_CO2R(c_ix[5], V_applied_SHE, 0.0, c_ix[1],
                            a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        R_her[ix]  = R_HER(V_applied_SHE, 0.0, c_ix[1], a_v_CL, n_e)
        p_co2 = pCO2_atm(u, ly, ly.N_GDL + ix)           # local gas CO₂ partial pressure [atm]
        H_loc = sechenov_H_CO2(c_ix)
        R_pt[ix] = a_v_GL * k_GL() * (H_loc * p_co2 - c_ix[5])
    end

    # ── Liquid diffusion fluxes ──
    dxb = cl_mesh.dx_between; dxfL = cl_mesh.dx_face_left; dx = cl_mesh.dx
    J = zeros(ElT, 5, N+1)
    # CO₂ left BC: Henry equilibrium with local gas at the first CL gas cell
    c_CO2_bc = sechenov_H_CO2(@view c[:, 1]) * pCO2_atm(u, ly, ly.N_GDL + 1)
    J[5, 1] = diffusion_flux(c_CO2_bc, c[5, 1], D_eff[5], dxfL)
    # ions: no-flux at GDL side (J[1:4,1]=0)
    for f in 2:N, i in 1:5
        J[i, f] = diffusion_flux(c[i, f-1], c[i, f], D_eff[i], dxb[f-1])
    end
    for i in 1:5
        J[i, N+1] = -k_MT[i] * (c_eq_vec[i] - c[i, N])    # Sherwood Neumann
    end

    # ── Liquid residuals ──
    for ix in 1:N
        buf = buffer_sources(@view c[:, ix])
        for i in 1:5
            src = eps_I * buf[i]
            i == 2 && (src += 2*R_co2r[ix] + 2*R_her[ix])
            i == 5 && (src += -R_co2r[ix] + R_pt[ix])
            F[liquid_idx(ix, i)] = J[i, ix] - J[i, ix+1] + src * dx[ix]
        end
    end

    # ── Gas-phase Stefan-Maxwell fluxes (u_G = 0) ──
    Jg = zeros(ElT, 4, Ng+1)
    # face 1: GDL inlet ghost (mole→mass omega_inlet, p_total)
    let bf = eps_GDL^1.5, wL = gas_inlet_tuple(), wR = gas_cell_omega(u, ly, 1),
        pf = 0.5*(p_total + gas_cell_p(u, ly, 1))
        Jf = sm_face_flux(wL, wR, pf, bf, gas_mesh.dx_face_left)
        for j in 1:4; Jg[j, 1] = Jf[j]; end
    end
    for f in 2:Ng
        bf = harmonic_mean(gas_cell_eps(ly, f-1)^1.5, gas_cell_eps(ly, f)^1.5)
        wL = gas_cell_omega(u, ly, f-1); wR = gas_cell_omega(u, ly, f)
        pf = 0.5*(gas_cell_p(u, ly, f-1) + gas_cell_p(u, ly, f))
        Jf = sm_face_flux(wL, wR, pf, bf, gas_mesh.dx_between[f-1])
        for j in 1:4; Jg[j, f] = Jf[j]; end
    end
    # face Ng+1: zero flux (gas does not penetrate electrolyte) — Jg[:,Ng+1]=0

    # ── Gas residuals ──
    dxg = gas_mesh.dx
    for cc in 1:Ng
        # species sources (kg/m³/s): nonzero only in CL cells
        SCO2 = zero(ElT); SCO = zero(ElT); SH2 = zero(ElT)
        if cc > ly.N_GDL
            ix = cc - ly.N_GDL
            SCO2 = -R_pt[ix] * M_gas[1]
            SCO  =  R_co2r[ix] * M_gas[2]
            SH2  =  R_her[ix]  * M_gas[3]
        end
        F[gas_resid_idx(ly, cc, 1)] = Jg[1, cc] - Jg[1, cc+1] + SCO2 * dxg[cc]
        F[gas_resid_idx(ly, cc, 2)] = Jg[2, cc] - Jg[2, cc+1] + SCO  * dxg[cc]
        F[gas_resid_idx(ly, cc, 3)] = Jg[3, cc] - Jg[3, cc+1] + SH2  * dxg[cc]
        # pressure placeholder (Stage 3): p pinned to p_total (uniform)
        F[gas_resid_idx(ly, cc, 4)] = gas_cell_p(u, ly, cc) - p_total
    end
    return F
end

# ── Stage 4 residual (§11) — finalize Darcy pressure + gas convection ───────
# Same as Stage 3 except: pressure DOF solves the Darcy continuity equation
# (∂(ρu_G)/∂x = S_mass), gas species flux includes the ρ ω u_G convective term,
# κ harmonic-mean at the GDL/CL interface. Liquid block unchanged from Stage 3.
function residual_stage4!(F::AbstractVector{ElT}, u::AbstractVector{ElT},
                          V_cathode::Real, cl_mesh::Mesh, gas_mesh::Mesh,
                          c_eq_vec::AbstractVector{<:Real}, ly::GasLayout) where {ElT<:Real}
    N = cl_mesh.N; Ng = gas_mesh.N
    c = unpack_log_conc(u, N)
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset

    R_co2r = zeros(ElT, N); R_her = zeros(ElT, N); R_pt = zeros(ElT, N)
    for ix in 1:N
        c_ix = @view c[:, ix]
        R_co2r[ix] = R_CO2R(c_ix[5], V_applied_SHE, 0.0, c_ix[1],
                            a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        R_her[ix]  = R_HER(V_applied_SHE, 0.0, c_ix[1], a_v_CL, n_e)
        p_co2 = pCO2_atm(u, ly, ly.N_GDL + ix)
        R_pt[ix] = a_v_GL * k_GL() * (sechenov_H_CO2(c_ix) * p_co2 - c_ix[5])
    end

    # ── Liquid (diffusion-only, φ_L=0; identical to Stage 3) ──
    dxb = cl_mesh.dx_between; dxfL = cl_mesh.dx_face_left; dx = cl_mesh.dx
    J = zeros(ElT, 5, N+1)
    c_CO2_bc = sechenov_H_CO2(@view c[:, 1]) * pCO2_atm(u, ly, ly.N_GDL + 1)
    J[5, 1] = diffusion_flux(c_CO2_bc, c[5, 1], D_eff[5], dxfL)
    for f in 2:N, i in 1:5
        J[i, f] = diffusion_flux(c[i, f-1], c[i, f], D_eff[i], dxb[f-1])
    end
    for i in 1:5
        J[i, N+1] = -k_MT[i] * (c_eq_vec[i] - c[i, N])
    end
    for ix in 1:N
        buf = buffer_sources(@view c[:, ix])
        for i in 1:5
            src = eps_I * buf[i]
            i == 2 && (src += 2*R_co2r[ix] + 2*R_her[ix])
            i == 5 && (src += -R_co2r[ix] + R_pt[ix])
            F[liquid_idx(ix, i)] = J[i, ix] - J[i, ix+1] + src * dx[ix]
        end
    end

    # ── Gas: SM diffusion + Darcy convection ──
    Jg  = zeros(ElT, 4, Ng+1)
    rhou = zeros(ElT, Ng+1)
    # face 1: GDL inlet ghost (p = p_total Dirichlet enters via Darcy velocity)
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
    # face Ng+1: dp/dx = 0 → u_G = 0, no flux (Jg[:,Ng+1]=0, rhou[Ng+1]=0)

    dxg = gas_mesh.dx
    for cc in 1:Ng
        SCO2 = zero(ElT); SCO = zero(ElT); SH2 = zero(ElT)
        if cc > ly.N_GDL
            ix = cc - ly.N_GDL
            SCO2 = -R_pt[ix] * M_gas[1]
            SCO  =  R_co2r[ix] * M_gas[2]
            SH2  =  R_her[ix]  * M_gas[3]
        end
        F[gas_resid_idx(ly, cc, 1)] = Jg[1, cc] - Jg[1, cc+1] + SCO2 * dxg[cc]
        F[gas_resid_idx(ly, cc, 2)] = Jg[2, cc] - Jg[2, cc+1] + SCO  * dxg[cc]
        F[gas_resid_idx(ly, cc, 3)] = Jg[3, cc] - Jg[3, cc+1] + SH2  * dxg[cc]
        # Darcy continuity (total mass): ρu_left − ρu_right + S_mass·dx
        F[gas_resid_idx(ly, cc, 4)] = rhou[cc] - rhou[cc+1] + (SCO2 + SCO + SH2) * dxg[cc]
    end
    return F
end

# ── Current extractors (§17.4) — A/m² of geometric CL cross-section ─────────
# Stages 1–4: pass V_applied_SHE as phi_s, 0.0 as phi_L.
function compute_i_CO(u::AbstractVector{<:Real}, phi_s, cl_mesh::Mesh; phi_L = 0.0)
    c_all = unpack_log_conc(u, cl_mesh.N)
    i = 0.0
    for ix in 1:cl_mesh.N
        ps = phi_s isa AbstractVector ? phi_s[ix] : phi_s
        pL = phi_L isa AbstractVector ? phi_L[ix] : phi_L
        r = R_CO2R(c_all[5, ix], ps, pL, c_all[1, ix],
                   a_v_CL, lambda_MHC(), i0_CO2R(), 2)
        i += 2 * F_const * r * cl_mesh.dx[ix]
    end
    return i
end

function compute_i_H2(u::AbstractVector{<:Real}, phi_s, cl_mesh::Mesh; phi_L = 0.0)
    c_all = unpack_log_conc(u, cl_mesh.N)
    i = 0.0
    for ix in 1:cl_mesh.N
        ps = phi_s isa AbstractVector ? phi_s[ix] : phi_s
        pL = phi_L isa AbstractVector ? phi_L[ix] : phi_L
        r = R_HER(ps, pL, c_all[1, ix], a_v_CL, 2)
        i += 2 * F_const * r * cl_mesh.dx[ix]
    end
    return i
end

# Per-cell volumetric rates (for A2 diagnostic plots).
function rate_profiles(u::AbstractVector{<:Real}, phi_s, cl_mesh::Mesh; phi_L = 0.0)
    c_all = unpack_log_conc(u, cl_mesh.N)
    rco = zeros(cl_mesh.N); rh2 = zeros(cl_mesh.N)
    for ix in 1:cl_mesh.N
        ps = phi_s isa AbstractVector ? phi_s[ix] : phi_s
        pL = phi_L isa AbstractVector ? phi_L[ix] : phi_L
        rco[ix] = R_CO2R(c_all[5, ix], ps, pL, c_all[1, ix],
                         a_v_CL, lambda_MHC(), i0_CO2R(), 2)
        rh2[ix] = R_HER(ps, pL, c_all[1, ix], a_v_CL, 2)
    end
    return rco, rh2
end
