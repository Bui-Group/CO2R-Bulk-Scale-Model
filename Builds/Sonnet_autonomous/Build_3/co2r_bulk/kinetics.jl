# kinetics.jl — R_CO2R (MHC), R_HER (BV), §4 Electrochemical Kinetics
using SpecialFunctions: erfc

"""
    mhc_k_elec(eta, i0, a_v, lambda_eV, n_e)

Marcus-Hush-Chidsey pseudo-first-order rate constant, §4.1.
"""
@inline function mhc_k_elec(eta::Real, i0::Real, a_v::Real, lambda_eV::Real, n_e::Real)
    lambda_ND = lambda_eV / (k_B_eV * T)
    eta_ND = n_e * F_const * eta / (R_gas * T)
    eta_ND_clamp = clamp(eta_ND, -700.0, 700.0)
    fermi = 1.0 / (1.0 + exp(eta_ND_clamp))
    arg = (lambda_ND - sqrt(1.0 + sqrt(lambda_ND) + eta_ND^2)) / (2.0 * sqrt(lambda_ND))
    pf = a_v * i0 / (n_e * F_const * c_ref)
    return max(pf * sqrt(pi * lambda_ND) * fermi * erfc(arg), 0.0)
end

"""
    R_CO2R(c_CO2, phi_s, phi_L, c_H, a_v, lambda_eV, i0, n_e)

Volumetric CO2R rate [mol/m3_CL/s], §4.1. `phi_s`, `phi_L` are LOCAL
potentials (V vs SHE) at the CL cell — never a global V_applied directly.
"""
@inline function R_CO2R(c_CO2::Real, phi_s::Real, phi_L::Real, c_H::Real,
                         a_v::Real, lambda_eV::Real, i0::Real, n_e::Real)
    pH_local = -log10(max(c_H / 1000.0, 1e-20))
    U_eq = U_CO_0 - (2.303 * R_gas * T / F_const) * pH_local
    eta = (phi_s - phi_L) - U_eq
    k = mhc_k_elec(eta, i0, a_v, lambda_eV, n_e)
    return k * max(c_CO2, 0.0)
end

"""
    R_HER(phi_s, phi_L, c_H, a_v, n_e)

Volumetric HER rate [mol/m3_CL/s], §4.2 (Tafel/BV, cathodic branch only).
"""
@inline function R_HER(phi_s::Real, phi_L::Real, c_H::Real, a_v::Real, n_e::Real)
    pH_local = -log10(max(c_H / 1000.0, 1e-20))
    U_eq_HER = -(2.303 * R_gas * T / F_const) * pH_local
    eta_HER = (phi_s - phi_L) - U_eq_HER
    i0_mA_cm2 = HER_prefactor() * exp(-(E_act_HER + dEact_dpH * pH_local) / (R_gas * T))
    i0 = i0_mA_cm2 * 10.0   # mA/cm^2 -> A/m^2
    exponent = clamp(-alpha_c_HER() * F_const * eta_HER / (R_gas * T), -700.0, 700.0)
    rate = (a_v / (n_e * F_const)) * i0 * exp(exponent)
    return max(rate, 0.0)
end
