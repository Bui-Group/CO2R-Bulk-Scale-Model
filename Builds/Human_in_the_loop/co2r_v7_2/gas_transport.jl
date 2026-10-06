# =============================================================================
# gas_transport.jl — Stefan-Maxwell mixture-average diffusion + Darcy (§6)
# Gas species order: 1 CO2, 2 CO, 3 H2, 4 H2O (H2O is closure-only, not a DOF).
# Every function is type-generic so ForwardDiff.Dual flows through (§9.3.2).
# =============================================================================

"§6.1 H2O closure: omega_H2O = 1 - w1 - w2 - w3."
@inline omega_H2O_closure(w1::Real, w2::Real, w3::Real) = max(1.0 - w1 - w2 - w3, 0.0)

"Mixture molar mass [kg/mol] from the 4 mass fractions: M_n = 1/sum(w_j/M_j)."
@inline function Mn_from_omega(w::NTuple{4,<:Real})
    s = w[1]/M_gas[1] + w[2]/M_gas[2] + w[3]/M_gas[3] + w[4]/M_gas[4]
    return 1.0 / s
end

"Mole fractions from mass fractions: x_j = w_j * M_n / M_j."
@inline function omega_to_x(w::NTuple{4,ElT}) where {ElT<:Real}
    Mn = Mn_from_omega(w)
    return (w[1]*Mn/M_gas[1], w[2]*Mn/M_gas[2], w[3]*Mn/M_gas[3], w[4]*Mn/M_gas[4])
end

"Gas mass density [kg/m^3]: rho = p * M_n / (R T)."
@inline gas_density(p::Real, Mn::Real) = p * Mn / (R_gas * T)

"""
    D_MA_species(j, x, Dbin) -> D_j^MA [m^2/s]

Mixture-average diffusivity (§6.1): `D_j^MA = (1 - x_j) / sum_{k != j}(x_k / D_jk)`.
A tiny additive floor on the denominator keeps the expression finite in the
(never reached here) pure-species limit.
"""
@inline function D_MA_species(j::Int, x::NTuple{4,ElT}, Dbin::Matrix{Float64}) where {ElT<:Real}
    den = zero(ElT)
    @inbounds for k in 1:4
        k == j && continue
        den += x[k] / Dbin[j, k]
    end
    return (1.0 - x[j]) / (den + 1e-30)
end

"""
    gas_face_flux(wL, pL, wR, pR, Dbin, kap, dxc, darcy)
        -> (J1, J2, J3, rho_u)

Species mass fluxes [kg/(m^2 s)] for CO2/CO/H2 and the total Darcy mass flux
at one face (§6.1, §6.2):

    J_j   = -rho D_j^MA dw_j/dx - rho D_j^MA w_j (dM_n/dx)/M_n + rho w_j u_G
    u_G   = -(kappa/mu) dp/dx

`darcy = false` sets `u_G = 0` (Stage 3, §11).
"""
@inline function gas_face_flux(wL::NTuple{4,ElT}, pL::ElT,
                               wR::NTuple{4,ElT}, pR::ElT,
                               Dbin::Matrix{Float64}, kap::Float64,
                               dxc::Float64, darcy::Bool) where {ElT<:Real}
    MnL = Mn_from_omega(wL); MnR = Mn_from_omega(wR)
    rhoL = gas_density(pL, MnL); rhoR = gas_density(pR, MnR)
    wf = (0.5*(wL[1]+wR[1]), 0.5*(wL[2]+wR[2]), 0.5*(wL[3]+wR[3]), 0.5*(wL[4]+wR[4]))
    Mnf = Mn_from_omega(wf)
    rhof = 0.5 * (rhoL + rhoR)
    xf = omega_to_x(wf)
    dMn = (MnR - MnL) / dxc

    uG = darcy ? -(kap / mu_gas) * (pR - pL) / dxc : zero(ElT)
    rho_u = rhof * uG

    J1 = zero(ElT); J2 = zero(ElT); J3 = zero(ElT)
    @inbounds begin
        D1 = D_MA_species(1, xf, Dbin)
        D2 = D_MA_species(2, xf, Dbin)
        D3 = D_MA_species(3, xf, Dbin)
        J1 = -rhof*D1*(wR[1]-wL[1])/dxc - rhof*D1*wf[1]*dMn/Mnf + rhof*wf[1]*uG
        J2 = -rhof*D2*(wR[2]-wL[2])/dxc - rhof*D2*wf[2]*dMn/Mnf + rhof*wf[2]*uG
        J3 = -rhof*D3*(wR[3]-wL[3])/dxc - rhof*D3*wf[3]*dMn/Mnf + rhof*wf[3]*uG
    end
    return (J1, J2, J3, rho_u)
end

"Inlet ghost state at the flow-field face (§7.3): Dirichlet mole fractions -> mass."
@inline inlet_omega(::Type{ElT}) where {ElT<:Real} =
    (ElT(omega_inlet[1]), ElT(omega_inlet[2]), ElT(omega_inlet[3]), ElT(omega_inlet[4]))
