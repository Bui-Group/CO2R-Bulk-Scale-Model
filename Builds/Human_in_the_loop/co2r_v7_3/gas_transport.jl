# ============================================================================
# gas_transport.jl — Stefan-Maxwell mixture-average + Darcy (§6)
# ============================================================================

"Mixture molar mass [kg/mol] from mass fractions."
@inline function Mn_from_omega(w::NTuple{4,ElT}) where {ElT<:Real}
    s = w[1]/M_gas[1] + w[2]/M_gas[2] + w[3]/M_gas[3] + w[4]/M_gas[4]
    return one(ElT) / s
end

"Mole fractions from mass fractions."
@inline function omega_to_x(w::NTuple{4,ElT}) where {ElT<:Real}
    Mn = Mn_from_omega(w)
    return (w[1]*Mn/M_gas[1], w[2]*Mn/M_gas[2], w[3]*Mn/M_gas[3], w[4]*Mn/M_gas[4])
end

"Gas mass density [kg/m³]: ρ = p M_n /(R T)."
@inline gas_rho(p::Real, Mn::Real) = p * Mn / (R_gas * T)

"""
    D_MA_species(x, Dbin) -> NTuple{4,ElT}

Mixture-average diffusivity D_j^MA = (1 - x_j) / Σ_{k≠j} (x_k / D_jk_eff) (§6.1).
"""
@inline function D_MA_species(x::NTuple{4,ElT}, Dbin::Matrix{Float64}) where {ElT<:Real}
    d1 = (one(ElT) - x[1]) / max(x[2]/Dbin[1,2] + x[3]/Dbin[1,3] + x[4]/Dbin[1,4], 1e-30)
    d2 = (one(ElT) - x[2]) / max(x[1]/Dbin[2,1] + x[3]/Dbin[2,3] + x[4]/Dbin[2,4], 1e-30)
    d3 = (one(ElT) - x[3]) / max(x[1]/Dbin[3,1] + x[2]/Dbin[3,2] + x[4]/Dbin[3,4], 1e-30)
    d4 = (one(ElT) - x[4]) / max(x[1]/Dbin[4,1] + x[2]/Dbin[4,2] + x[3]/Dbin[4,3], 1e-30)
    return (d1, d2, d3, d4)
end

"""
    sm_face_flux(wL, pL, wR, pR, dx_cc, Dbin, kappa, darcy_on)
        -> (J1, J2, J3, J4, rho_u)

Full mixture-average Stefan-Maxwell mass flux [kg/(m² s)] across one face
(§6.1) plus the Darcy mass flux ρu [kg/(m² s)] (§6.2).  Face properties are
arithmetic averages of the two cell states; `Dbin` is the domain-appropriate
effective binary matrix (harmonic mean at the GDL/CL interface, §8.7).
"""
@inline function sm_face_flux(wL::NTuple{4,ElT}, pL::ElT2,
                              wR::NTuple{4,ElT3}, pR::ElT4,
                              dx_cc::Float64, Dbin::Matrix{Float64},
                              kappa::Float64, darcy_on::Bool) where {ElT<:Real,ElT2<:Real,ElT3<:Real,ElT4<:Real}
    MnL = Mn_from_omega(wL); MnR = Mn_from_omega(wR)
    xL  = omega_to_x(wL);    xR  = omega_to_x(wR)
    rhoL = gas_rho(pL, MnL); rhoR = gas_rho(pR, MnR)
    rho_f = 0.5 * (rhoL + rhoR)
    Mn_f  = 0.5 * (MnL + MnR)
    xf = (0.5*(xL[1]+xR[1]), 0.5*(xL[2]+xR[2]), 0.5*(xL[3]+xR[3]), 0.5*(xL[4]+xR[4]))
    wf = (0.5*(wL[1]+wR[1]), 0.5*(wL[2]+wR[2]), 0.5*(wL[3]+wR[3]), 0.5*(wL[4]+wR[4]))
    Dma = D_MA_species(xf, Dbin)

    u_G = darcy_on ? -(kappa / mu_gas) * (pR - pL) / dx_cc : zero(pR)
    dMn = (MnR - MnL) / dx_cc

    J1 = -rho_f*Dma[1]*(wR[1]-wL[1])/dx_cc - rho_f*Dma[1]*wf[1]*dMn/Mn_f + rho_f*wf[1]*u_G
    J2 = -rho_f*Dma[2]*(wR[2]-wL[2])/dx_cc - rho_f*Dma[2]*wf[2]*dMn/Mn_f + rho_f*wf[2]*u_G
    J3 = -rho_f*Dma[3]*(wR[3]-wL[3])/dx_cc - rho_f*Dma[3]*wf[3]*dMn/Mn_f + rho_f*wf[3]*u_G
    J4 = -rho_f*Dma[4]*(wR[4]-wL[4])/dx_cc - rho_f*Dma[4]*wf[4]*dMn/Mn_f + rho_f*wf[4]*u_G
    return (J1, J2, J3, J4, rho_f * u_G)
end

"Effective binary diffusivity matrix for gas face `f` of the combined mesh."
@inline function face_Dbin(gm::GasMesh, f::Int)
    f <= gm.N_GDL      && return D_bin_GDL
    f == gm.N_GDL + 1  && return D_bin_INT
    return D_bin_CL
end

"Darcy permeability for gas face `f` of the combined mesh (§11 Stage 4)."
@inline function face_kappa(gm::GasMesh, f::Int)
    f <= gm.N_GDL      && return kappa_GDL
    f == gm.N_GDL + 1  && return kappa_INT
    return kappa_CL
end
