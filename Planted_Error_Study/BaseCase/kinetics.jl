# =============================================================================
# kinetics.jl — electrochemical kinetics (BaseCase.md §4)
#   MHC (Marcus–Hush–Chidsey) for CO₂R, Butler–Volmer (Tafel) for HER.
# All potentials in V vs SHE. Type-generic for ForwardDiff.
# Driving force uses LOCAL (phi_s - phi_L); in Stages 1–4 caller passes
# phi_s = V_applied_SHE, phi_L = 0.
# =============================================================================

using SpecialFunctions: erfc

"""
    mhc_k_elec(eta, i0, a_v, lambda_eV, n_e) -> pseudo-first-order rate constant [1/s]

MHC integral rate constant (§4.1). `eta` is the overpotential (V vs SHE), `lambda_eV`
the reorganization energy in eV (converted to dimensionless internally).
"""
@inline function mhc_k_elec(eta::Real, i0::Real, a_v::Real, lambda_eV::Real, n_e_::Real)
    lambda_ND = lambda_eV / (k_B_eV * T)
    eta_ND = n_e_ * F_const * eta / (R_gas * T)
    eta_ND_clamp = clamp(eta_ND, -700.0, 700.0)
    fermi = 1.0 / (1.0 + exp(eta_ND_clamp))
    arg = (lambda_ND - sqrt(1.0 + sqrt(lambda_ND) + eta_ND^2)) / (2.0 * sqrt(lambda_ND))
    pf = a_v * i0 / (n_e_ * F_const * c_ref)
    return max(pf * sqrt(pi * lambda_ND) * fermi * erfc(arg), 0.0)
end

"""
    R_CO2R(c_CO2, phi_s, phi_L, c_H, a_v, lambda_eV, i0, n_e) -> volumetric rate [mol/m³/s]
"""
@inline function R_CO2R(c_CO2::Real, phi_s::Real, phi_L::Real, c_H::Real,
                        a_v::Real, lambda_eV::Real, i0::Real, n_e_::Real)
    pH_local = -log10(max(c_H / 1000.0, 1e-20))
    U_eq = U_CO_0 - nernst_fac * pH_local
    eta = (phi_s - phi_L) - U_eq                 # MUST subtract phi_L (§4.1)
    k = mhc_k_elec(eta, i0, a_v, lambda_eV, n_e_)
    return k * max(c_CO2, 0.0)
end

"""
    R_HER(phi_s, phi_L, c_H, a_v, n_e) -> volumetric rate [mol/m³/s]

Butler–Volmer cathodic branch with pH-dependent exchange current (§4.2).
Uses Ref-backed HER_prefactor() and alpha_c_HER().
"""
@inline function R_HER(phi_s::Real, phi_L::Real, c_H::Real, a_v::Real, n_e_::Real)
    pH_local = -log10(max(c_H / 1000.0, 1e-20))
    U_eq_HER = -nernst_fac * pH_local
    eta_HER = (phi_s - phi_L) - U_eq_HER          # MUST subtract phi_L (§4.2)
    i0_mA_cm2 = HER_prefactor() * exp(-(E_act_HER + dEact_dpH * pH_local) / (R_gas * T))
    i0 = i0_mA_cm2 * 10.0                          # mA/cm² -> A/m²
    exponent = clamp(-alpha_c_HER() * F_const * eta_HER / (R_gas * T), -700.0, 700.0)
    rate = (a_v / (n_e_ * F_const)) * i0 * exp(exponent)
    return max(rate, 0.0)
end
