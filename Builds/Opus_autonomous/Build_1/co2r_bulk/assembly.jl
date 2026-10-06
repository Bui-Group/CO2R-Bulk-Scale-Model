# assembly.jl — DOF layout, the unified staged residual, and current extractors
#
# One residual function covers Stages 1–5; `ly.stage` selects the physics:
#   stage 1 : liquid diffusion, Dirichlet c_eq at x = L_CL
#   stage 2 : + Sherwood-Reynolds Neumann at x = L_CL
#   stage 3 : + Stefan-Maxwell gas (u_G = 0, p pinned) + R_PT phase transfer
#   stage 4 : + Darcy continuity / convection
#   stage 5 : + φ_L (NP migration, monolithic DOF) + φ_s (solid Ohm's law)

# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────
struct Layout
    stage::Int
    N_CL::Int
    N_GDL::Int
    N_G::Int        # N_GDL + N_CL  (gas / solid cell count)
    gas0::Int       # offset before gas block  (0 if no gas)
    phis0::Int
    phiL0::Int
    N_TOT::Int
end

function Layout(stage::Int, N_CL::Int, N_GDL::Int)
    N_G  = N_GDL + N_CL
    nliq = 5 * N_CL
    gas0 = nliq
    ngas = stage >= 3 ? 4 * N_G : 0
    phis0 = gas0 + ngas
    nphis = stage >= 5 ? N_G : 0
    phiL0 = phis0 + nphis
    nphiL = stage >= 5 ? N_CL : 0
    return Layout(stage, N_CL, N_GDL, N_G, gas0, phis0, phiL0,
                  nliq + ngas + nphis + nphiL)
end

@inline liquid_idx(ix::Int, i::Int) = 5*(ix - 1) + i
@inline gas_idx(ly::Layout, k::Int, j::Int) = ly.gas0 + 4*(k - 1) + j   # k global gas cell
@inline gas_gdl_idx(ly::Layout, ig::Int, j::Int) = gas_idx(ly, ig, j)
@inline gas_cl_idx(ly::Layout, ix::Int, j::Int)  = gas_idx(ly, ly.N_GDL + ix, j)
@inline phi_s_idx(ly::Layout, k::Int)  = ly.phis0 + k
@inline phi_s_gdl_idx(ly::Layout, ig::Int) = ly.phis0 + ig
@inline phi_s_cl_idx(ly::Layout, ix::Int)  = ly.phis0 + ly.N_GDL + ix
@inline phi_L_idx(ly::Layout, ix::Int) = ly.phiL0 + ix

"""DOF blocks used by the affine-covariant weight construction (§9.5.3)."""
function weight_blocks(ly::Layout)
    b = UnitRange{Int}[]
    push!(b, 1:(5*ly.N_CL))
    if ly.stage >= 3
        push!(b, (ly.gas0+1):(ly.gas0 + 4*ly.N_GDL))                 # gas GDL
        push!(b, (ly.gas0 + 4*ly.N_GDL + 1):(ly.gas0 + 4*ly.N_G))    # gas CL
    end
    if ly.stage >= 5
        push!(b, (ly.phis0+1):(ly.phis0 + ly.N_GDL))                 # φ_s GDL
        push!(b, (ly.phis0 + ly.N_GDL + 1):(ly.phis0 + ly.N_G))      # φ_s CL
        push!(b, (ly.phiL0+1):(ly.phiL0 + ly.N_CL))                  # φ_L
    end
    return b
end

# ─────────────────────────────────────────────────────────────────────────────
# Domain — meshes + combined gas/solid grid + bulk equilibrium
# ─────────────────────────────────────────────────────────────────────────────
struct Domain
    cl::FVMesh
    gdl::FVMesh
    N_G::Int
    dx_g::Vector{Float64}        # combined cell widths (GDL then CL)
    dxcc_g::Vector{Float64}      # combined face centre-to-centre distances
    xc_g::Vector{Float64}        # combined cell centres
    c_eq::Vector{Float64}        # bulk equilibrium (5)
end

function make_domain(N_CL::Int, N_GDL::Int)
    cl  = make_cl_mesh(N_CL)
    gdl = make_gdl_mesh(N_GDL)
    N_G = N_CL + N_GDL
    dx_g  = vcat(gdl.dx, cl.dx)
    xc_g  = vcat(gdl.x_centers, cl.x_centers)
    dxcc_g = Vector{Float64}(undef, N_G + 1)
    for f in 1:N_GDL
        dxcc_g[f] = gdl.dx_cc[f]
    end
    dxcc_g[N_GDL+1] = gdl.dx_cc[N_GDL+1] + cl.dx_cc[1]
    for i in 2:N_CL
        dxcc_g[N_GDL + i] = cl.dx_cc[i]
    end
    dxcc_g[N_G+1] = cl.dx_cc[N_CL+1]
    eq = solve_buffer_equilibrium(1000.0, 1000.0)
    c_eq = [eq.H, eq.OH, eq.HCO3, eq.CO3, eq.CO2]
    return Domain(cl, gdl, N_G, dx_g, dxcc_g, xc_g, c_eq)
end

"""Binary-diffusivity matrix for combined-grid face `f` (§8.7)."""
@inline function face_Dbin(dm::Domain, f::Int)
    N_GDL = dm.gdl.N
    f <= N_GDL       && return D_bin_GDL
    f == N_GDL + 1   && return D_bin_INT
    return D_bin_CL
end

@inline function face_kappa(dm::Domain, f::Int)
    N_GDL = dm.gdl.N
    f <= N_GDL     && return kappa_GDL
    f == N_GDL + 1 && return kappa_INT
    return kappa_CL
end

@inline function face_sigma_s(dm::Domain, f::Int)
    N_GDL = dm.gdl.N
    f <= N_GDL     && return sigma_s_eff_GDL
    f == N_GDL + 1 && return sigma_s_eff_INT
    return sigma_s_eff_CL
end

# ─────────────────────────────────────────────────────────────────────────────
# Unpacking helpers (all eltype-preserving for ForwardDiff)
# ─────────────────────────────────────────────────────────────────────────────
function unpack_log_conc(u::AbstractVector{ElT}, N_CL::Int) where {ElT<:Real}
    c = zeros(ElT, 5, N_CL)
    @inbounds for ix in 1:N_CL, i in 1:5
        c[i, ix] = exp(clamp(u[liquid_idx(ix, i)], -50.0, 50.0))
    end
    return c
end

@inline function gas_omega(u::AbstractVector{ElT}, ly::Layout, k::Int) where {ElT<:Real}
    b = ly.gas0 + 4*(k - 1)
    w1 = u[b+1]; w2 = u[b+2]; w3 = u[b+3]
    return (w1, w2, w3, omega_H2O_closure(w1, w2, w3))
end

@inline gas_p(u::AbstractVector{ElT}, ly::Layout, k::Int) where {ElT<:Real} =
    u[ly.gas0 + 4*(k - 1) + 4]

# ─────────────────────────────────────────────────────────────────────────────
# Gas face flux (mixture-average SM + optional Darcy convection)
# ─────────────────────────────────────────────────────────────────────────────
"""
Returns `(J1, J2, J3, rho_u)` at a face given left/right (ω, p) states.
`with_darcy=false` forces u_G = 0 (Stage 3).
"""
@inline function sm_face_flux(wL::NTuple{4,ElT}, pL::ElT,
                              wR::NTuple{4,ElT}, pR::ElT,
                              dxcc::Float64, Dm::Matrix{Float64},
                              kap::Float64, with_darcy::Bool) where {ElT<:Real}
    MnL = Mn_from_omega(wL[1], wL[2], wL[3], wL[4])
    MnR = Mn_from_omega(wR[1], wR[2], wR[3], wR[4])
    Mbar = 0.5*(MnL + MnR)
    pbar = 0.5*(pL + pR)
    rho  = pbar * Mbar / (R_gas * T)

    wb1 = 0.5*(wL[1] + wR[1]); wb2 = 0.5*(wL[2] + wR[2])
    wb3 = 0.5*(wL[3] + wR[3]); wb4 = 0.5*(wL[4] + wR[4])
    x1 = wb1*Mbar/M_gas[1]; x2 = wb2*Mbar/M_gas[2]
    x3 = wb3*Mbar/M_gas[3]; x4 = wb4*Mbar/M_gas[4]

    uG = with_darcy ? -(kap / mu_gas) * (pR - pL) / dxcc : zero(ElT)
    dMn = (MnR - MnL) / dxcc

    D1 = D_MA_species(x1, x2, x3, x4, Dm, 1)
    D2 = D_MA_species(x1, x2, x3, x4, Dm, 2)
    D3 = D_MA_species(x1, x2, x3, x4, Dm, 3)

    J1 = -rho*D1*(wR[1]-wL[1])/dxcc - rho*D1*wb1*dMn/Mbar + rho*wb1*uG
    J2 = -rho*D2*(wR[2]-wL[2])/dxcc - rho*D2*wb2*dMn/Mbar + rho*wb2*uG
    J3 = -rho*D3*(wR[3]-wL[3])/dxcc - rho*D3*wb3*dMn/Mbar + rho*wb3*uG
    return (J1, J2, J3, rho*uG)
end

# ─────────────────────────────────────────────────────────────────────────────
# The unified staged residual
# ─────────────────────────────────────────────────────────────────────────────
function residual!(F::AbstractVector{ElT}, u::AbstractVector{ElT},
                   V_cathode::Real, dm::Domain, ly::Layout) where {ElT<:Real}
    N_CL  = ly.N_CL
    N_GDL = ly.N_GDL
    N_G   = ly.N_G
    stage = ly.stage
    cl    = dm.cl
    c_eq  = dm.c_eq
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset

    c = unpack_log_conc(u, N_CL)

    # ── potentials
    have_pot = stage >= 5
    phi_L = zeros(ElT, N_CL)
    phi_s_cl = zeros(ElT, N_CL)
    if have_pot
        @inbounds for ix in 1:N_CL
            phi_L[ix]    = u[phi_L_idx(ly, ix)]
            phi_s_cl[ix] = u[phi_s_cl_idx(ly, ix)]
        end
    else
        @inbounds for ix in 1:N_CL
            phi_s_cl[ix] = V_applied_SHE
        end
    end

    # ── gas-side CO₂(aq) boundary value at x = 0
    c_CO2_bc = zero(ElT)
    if stage <= 2
        c_CO2_bc = ElT(c_CO2_bc_s12)
    else
        wcl1 = gas_omega(u, ly, N_GDL + 1)
        pcl1 = gas_p(u, ly, N_GDL + 1)
        xg   = omega_to_x(wcl1[1], wcl1[2], wcl1[3], wcl1[4])
        p_CO2_atm = xg[1] * pcl1 / 101325.0
        c_CO2_bc  = sechenov_H_CO2(view(c, :, 1)) * p_CO2_atm
    end

    # ── Step 1: liquid fluxes, integrated FV
    J = zeros(ElT, 5, N_CL + 1)
    # 2026-09-27 post-build change: no CO2(aq) flux at the GDL/CL face in Stage 5 (guide v7.1, Stage 5 note).
    J[5, 1] = have_pot ? zero(ElT) :
              diffusion_flux(c_CO2_bc, c[5,1], D_eff[5], cl.dx_cc[1])
    # ions: no flux at the GDL/CL face (J[1:4,1] stay zero)
    if have_pot
        @inbounds for f in 2:N_CL, i in 1:5
            J[i,f] = np_flux(c[i,f-1], c[i,f], phi_L[f-1], phi_L[f],
                             D_eff[i], z_sp[i], cl.dx_cc[f])
        end
        @inbounds for i in 1:5
            L_eff = D_eff[i] / k_MT[i]
            J[i,N_CL+1] = np_flux(c[i,N_CL], ElT(c_eq[i]), phi_L[N_CL], zero(ElT),
                                  D_eff[i], z_sp[i], L_eff)
        end
    else
        @inbounds for f in 2:N_CL, i in 1:5
            J[i,f] = diffusion_flux(c[i,f-1], c[i,f], D_eff[i], cl.dx_cc[f])
        end
        if stage == 1
            @inbounds for i in 1:5
                J[i,N_CL+1] = diffusion_flux(c[i,N_CL], ElT(c_eq[i]), D_eff[i],
                                             cl.dx_cc[N_CL+1])
            end
        else
            @inbounds for i in 1:5
                J[i,N_CL+1] = -k_MT[i] * (c_eq[i] - c[i,N_CL])
            end
        end
    end

    # ── Step 2: faradaic rates + phase transfer per CL cell
    Rco = zeros(ElT, N_CL); Rher = zeros(ElT, N_CL); Rpt = zeros(ElT, N_CL)
    lam = lambda_MHC(); i0 = i0_CO2R()
    @inbounds for ix in 1:N_CL
        pl = have_pot ? phi_L[ix] : zero(ElT)
        ps = phi_s_cl[ix]
        Rco[ix]  = R_CO2R(c[5,ix], ps, pl, c[1,ix], a_v_CL, lam, i0, n_e)
        Rher[ix] = R_HER(ps, pl, c[1,ix], a_v_CL, n_e)
        if stage >= 3
            wk = gas_omega(u, ly, N_GDL + ix)
            pk = gas_p(u, ly, N_GDL + ix)
            xk = omega_to_x(wk[1], wk[2], wk[3], wk[4])
            p_CO2_atm = xk[1] * pk / 101325.0
            H_loc = sechenov_H_CO2(view(c, :, ix))
            Rpt[ix] = a_v_GL * k_GL() * (H_loc * p_CO2_atm - c[5,ix])
        end
    end

    # ── Step 3: liquid species residuals
    @inbounds for ix in 1:N_CL
        buf = buffer_sources(view(c, :, ix))
        for i in 1:5
            src = eps_I * buf[i]
            if i == 2; src += 2*Rco[ix] + 2*Rher[ix]; end
            if i == 5
                src -= Rco[ix]
                stage >= 3 && (src += Rpt[ix])
            end
            F[liquid_idx(ix,i)] = J[i,ix] - J[i,ix+1] + src * cl.dx[ix]
        end
    end

    # ── Step 4: gas-phase residual (Stage 3+)
    if stage >= 3
        with_darcy = stage >= 4
        Jg  = zeros(ElT, 3, N_G + 1)
        rou = zeros(ElT, N_G + 1)
        # face 1: Dirichlet flow-field ghost (ω_inlet, p_total)
        wghost = (ElT(omega_inlet[1]), ElT(omega_inlet[2]),
                  ElT(omega_inlet[3]), ElT(omega_inlet[4]))
        w1 = gas_omega(u, ly, 1); p1 = gas_p(u, ly, 1)
        j1, j2, j3, ru = sm_face_flux(wghost, ElT(p_total), w1, p1,
                                      dm.dxcc_g[1], face_Dbin(dm,1),
                                      face_kappa(dm,1), with_darcy)
        Jg[1,1] = j1; Jg[2,1] = j2; Jg[3,1] = j3; rou[1] = ru
        @inbounds for f in 2:N_G
            wL = gas_omega(u, ly, f-1); pL = gas_p(u, ly, f-1)
            wR = gas_omega(u, ly, f);   pR = gas_p(u, ly, f)
            j1, j2, j3, ru = sm_face_flux(wL, pL, wR, pR, dm.dxcc_g[f],
                                          face_Dbin(dm,f), face_kappa(dm,f),
                                          with_darcy)
            Jg[1,f] = j1; Jg[2,f] = j2; Jg[3,f] = j3; rou[f] = ru
        end
        # face N_G+1: zero species flux and no gas flow (dp/dx = 0)
        # (Jg[:,N_G+1] and rou[N_G+1] stay zero)

        @inbounds for k in 1:N_G
            if k <= N_GDL
                S1 = zero(ElT); S2 = zero(ElT); S3 = zero(ElT)
            else
                ix = k - N_GDL
                S1 = -Rpt[ix]  * M_gas[1]
                S2 =  Rco[ix]  * M_gas[2]
                S3 =  Rher[ix] * M_gas[3]
            end
            dxk = dm.dx_g[k]
            F[gas_idx(ly,k,1)] = Jg[1,k] - Jg[1,k+1] + S1 * dxk
            F[gas_idx(ly,k,2)] = Jg[2,k] - Jg[2,k+1] + S2 * dxk
            F[gas_idx(ly,k,3)] = Jg[3,k] - Jg[3,k+1] + S3 * dxk
            if stage == 3
                F[gas_idx(ly,k,4)] = gas_p(u, ly, k) - p_total
            else
                F[gas_idx(ly,k,4)] = rou[k] - rou[k+1] + (S1 + S2 + S3) * dxk
            end
        end
    end

    # ── Step 5 / 6: potentials (Stage 5)
    if have_pot
        # Cs⁺ NP flux — no flux at either CL boundary face (§9.4 item 8)
        JCs = zeros(ElT, N_CL + 1)
        @inbounds for f in 2:N_CL
            cCsL = cCs_from_electroneutrality(c[1,f-1], c[2,f-1], c[3,f-1], c[4,f-1])
            cCsR = cCs_from_electroneutrality(c[1,f],   c[2,f],   c[3,f],   c[4,f])
            JCs[f] = np_flux(cCsL, cCsR, phi_L[f-1], phi_L[f],
                             D_eff_Cs, z_Cs, cl.dx_cc[f])
        end

        i_L = zeros(ElT, N_CL + 1)
        @inbounds for f in 1:N_CL+1
            s = zero(ElT)
            for i in 1:5
                s += z_sp[i] * J[i,f]
            end
            i_L[f] = F_const * (s + z_Cs * JCs[f])
        end

        @inbounds for ix in 1:N_CL
            i_far_vol = n_e * F_const * (Rco[ix] + Rher[ix])
            F[phi_L_idx(ly,ix)] = (i_L[ix+1] - i_L[ix]) + i_far_vol * cl.dx[ix]
        end
        # gauge fix (Dirichlet override, §3.1)
        F[phi_L_idx(ly,N_CL)] = phi_L[N_CL]

        # φ_s solid Ohm's law on GDL + CL
        i_s = zeros(ElT, N_G + 1)
        phis1 = u[phi_s_idx(ly,1)]
        i_s[1] = -sigma_s_eff_GDL * (phis1 - V_applied_SHE) / dm.dxcc_g[1]
        @inbounds for f in 2:N_G
            i_s[f] = -face_sigma_s(dm,f) *
                     (u[phi_s_idx(ly,f)] - u[phi_s_idx(ly,f-1)]) / dm.dxcc_g[f]
        end
        # i_s[N_G+1] = 0  (Neumann at the electrolyte side)

        @inbounds for ig in 1:N_GDL
            F[phi_s_gdl_idx(ly,ig)] = i_s[ig+1] - i_s[ig]
        end
        @inbounds for ix in 1:N_CL
            i_far_vol = n_e * F_const * (Rco[ix] + Rher[ix])
            F[phi_s_cl_idx(ly,ix)] = (i_s[N_GDL+1+ix] - i_s[N_GDL+ix]) -
                                     i_far_vol * cl.dx[ix]
        end
    end

    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Current extractors (§17.4)
# ─────────────────────────────────────────────────────────────────────────────
"""Local (φ_s, φ_L) at CL cell ix for the given stage."""
@inline function local_potentials(u::Vector{Float64}, ly::Layout, ix::Int,
                                  V_applied_SHE::Float64)
    if ly.stage >= 5
        return (u[phi_s_cl_idx(ly,ix)], u[phi_L_idx(ly,ix)])
    else
        return (V_applied_SHE, 0.0)
    end
end

function compute_rates(u::Vector{Float64}, ly::Layout, V_applied_SHE::Float64)
    c = unpack_log_conc(u, ly.N_CL)
    Rco = zeros(ly.N_CL); Rher = zeros(ly.N_CL)
    for ix in 1:ly.N_CL
        ps, pl = local_potentials(u, ly, ix, V_applied_SHE)
        Rco[ix]  = R_CO2R(c[5,ix], ps, pl, c[1,ix], a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        Rher[ix] = R_HER(ps, pl, c[1,ix], a_v_CL, n_e)
    end
    return Rco, Rher, c
end

"""Partial CO current density [A/m²]."""
function compute_i_CO(u::Vector{Float64}, ly::Layout, dm::Domain, V_applied_SHE::Float64)
    Rco, _, _ = compute_rates(u, ly, V_applied_SHE)
    return sum(n_e * F_const * Rco[ix] * dm.cl.dx[ix] for ix in 1:ly.N_CL)
end

"""Partial H₂ current density [A/m²]."""
function compute_i_H2(u::Vector{Float64}, ly::Layout, dm::Domain, V_applied_SHE::Float64)
    _, Rher, _ = compute_rates(u, ly, V_applied_SHE)
    return sum(n_e * F_const * Rher[ix] * dm.cl.dx[ix] for ix in 1:ly.N_CL)
end

function compute_currents(u::Vector{Float64}, ly::Layout, dm::Domain, V_RHE::Float64)
    V_SHE = V_RHE - V_RHE_to_SHE_offset
    Rco, Rher, _ = compute_rates(u, ly, V_SHE)
    iCO = sum(n_e * F_const * Rco[ix] * dm.cl.dx[ix] for ix in 1:ly.N_CL)
    iH2 = sum(n_e * F_const * Rher[ix] * dm.cl.dx[ix] for ix in 1:ly.N_CL)
    return (iCO, iH2, iCO + iH2)
end
