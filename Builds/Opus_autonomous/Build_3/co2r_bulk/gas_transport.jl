# gas_transport.jl — Stefan-Maxwell (mixture-average) + Darcy (§6)
#
# Gas domain = GDL ∥ CL (see GasMesh).  Per cell 4 DOFs: ω_CO₂, ω_CO, ω_H₂, p.
# H₂O is closure-only: ω_H₂O = max(1 − ω₁ − ω₂ − ω₃, 0)  (§6.1).

@inline omega_H2O_closure(w1::Real, w2::Real, w3::Real) = max(1.0 - w1 - w2 - w3, 0.0)

@inline function Mn_from_omega(w::NTuple{4,ElT}) where {ElT<:Real}
    s = zero(ElT)
    @inbounds for j in 1:4
        s += w[j] / M_gas[j]
    end
    return 1.0 / s
end

@inline function omega_to_x(w::NTuple{4,ElT}, Mn::ElT) where {ElT<:Real}
    return ntuple(j -> w[j] * Mn / M_gas[j], Val(4))
end

"Face transport coefficients: GDL faces, the single GDL/CL interface, CL faces."
@inline function face_Dbin(gm::GasMesh, f::Int)
    f <= gm.N_GDL      && return D_bin_GDL
    f == gm.i_int      && return D_bin_INT
    return D_bin_CL
end
@inline function face_kappa(gm::GasMesh, f::Int)
    f <= gm.N_GDL      && return kappa_GDL
    f == gm.i_int      && return kappa_INT
    return kappa_CL
end

"""
    gas_face_flux(wL, pL, wR, pR, Dbin, kappa, dxc, darcy) -> (J::NTuple{4}, rho_u)

Mixture-average Stefan-Maxwell mass fluxes [kg/(m² s)] plus the Darcy mass
flux ρu_G at one face (§6.1–6.2).  `darcy = false` sets u_G = 0 (Stage 3).
"""
@inline function gas_face_flux(wL::NTuple{4,ElT}, pL::ElT,
                               wR::NTuple{4,ElT}, pR::ElT,
                               Dbin::Matrix{Float64}, kappa::Float64,
                               dxc::Float64, darcy::Bool) where {ElT<:Real}
    MnL = Mn_from_omega(wL)
    MnR = Mn_from_omega(wR)
    wf  = ntuple(j -> 0.5 * (wL[j] + wR[j]), Val(4))
    pf  = 0.5 * (pL + pR)
    Mnf = Mn_from_omega(wf)
    rho = pf * Mnf / (R_gas * T)
    xf  = omega_to_x(wf, Mnf)
    uG  = darcy ? -(kappa / mu_gas) * (pR - pL) / dxc : zero(ElT)
    gradMn = (MnR - MnL) / dxc

    J = ntuple(Val(4)) do j
        den = zero(ElT)
        @inbounds for k in 1:4
            k == j && continue
            den += xf[k] / Dbin[j, k]
        end
        Dma = (1.0 - xf[j]) / max(den, 1e-30)
        (-rho * Dma * (wR[j] - wL[j]) / dxc
         - rho * Dma * wf[j] * gradMn / Mnf
         + rho * wf[j] * uG)
    end
    return J, rho * uG
end

"Local gas-phase CO₂ partial pressure [atm] from a cell's ω and p."
@inline function pCO2_atm(w::NTuple{4,ElT}, p::ElT) where {ElT<:Real}
    Mn = Mn_from_omega(w)
    x1 = w[1] * Mn / M_gas[1]
    return x1 * p / 101325.0
end
