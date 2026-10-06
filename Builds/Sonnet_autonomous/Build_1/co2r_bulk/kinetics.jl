##############################################################################
# kinetics.jl — MHC kinetics for CO2R (§4.1), Butler-Volmer for HER (§4.2)
##############################################################################
using SpecialFunctions: erfc

"""
    mhc_k_elec(eta, i0, a_v, lambda_eV, n_e)

Marcus-Hush-Chidsey pseudo-first-order rate constant [1/s].
`lambda_eV` is passed in eV; converted internally to dimensionless λ_ND.
"""
function mhc_k_elec(eta, i0, a_v, lambda_eV, n_e)
    lambda_ND = lambda_eV / (k_B_eV * T)
    eta_ND = n_e * F_const * eta / (R_gas * T)
    eta_ND_clamp = clamp(eta_ND, -700.0, 700.0)
    fermi = 1.0 / (1.0 + exp(eta_ND_clamp))
    arg = (lambda_ND - sqrt(1.0 + sqrt(lambda_ND) + eta_ND^2)) / (2.0 * sqrt(lambda_ND))
    pf = a_v * i0 / (n_e * F_const * c_ref)
    return max(pf * sqrt(pi * lambda_ND) * fermi * erfc(arg), zero(eta))
end

"""
    R_CO2R(c_CO2, phi_s, phi_L, c_H, a_v, lambda_eV, i0, n_e)

Volumetric CO2R rate [mol/m^3_CL/s].
"""
function R_CO2R(c_CO2, phi_s, phi_L, c_H, a_v, lambda_eV, i0, n_e)
    pH_local = -log10(max(c_H / 1000.0, 1e-20))
    U_eq = U_CO_0 - (2.303 * R_gas * T / F_const) * pH_local
    eta = (phi_s - phi_L) - U_eq
    k = mhc_k_elec(eta, i0, a_v, lambda_eV, n_e)
    return k * max(c_CO2, zero(c_CO2))
end

"""
    R_HER(phi_s, phi_L, c_H, a_v, n_e)

Volumetric HER rate [mol/m^3_CL/s], Butler-Volmer (cathodic branch only).
"""
function R_HER(phi_s, phi_L, c_H, a_v, n_e)
    pH_local = -log10(max(c_H / 1000.0, 1e-20))
    U_eq_HER = -(2.303 * R_gas * T / F_const) * pH_local
    eta_HER = (phi_s - phi_L) - U_eq_HER
    i0_mA_cm2 = HER_prefactor() * exp(-(E_act_HER + dEact_dpH * pH_local) / (R_gas * T))
    i0 = i0_mA_cm2 * 10.0
    exponent = clamp(-alpha_c_HER() * F_const * eta_HER / (R_gas * T), -700.0, 700.0)
    rate = (a_v / (n_e * F_const)) * i0 * exp(exponent)
    return max(rate, zero(rate))
end
