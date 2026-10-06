# =====================================================================
#  kinetics.jl — MHC CO₂R and Butler-Volmer HER (§4)
#  All potential arguments in V vs SHE.  φ_L must always be subtracted.
# =====================================================================

"""
    mhc_k_elec(eta, i0, a_v, lambda_eV, n_e) -> k [1/s]

Marcus-Hush-Chidsey pseudo-first-order rate constant (§4.1).
`lambda_eV` is passed in eV; λ_ND is computed internally.
"""
@inline function mhc_k_elec(eta::Real, i0::Real, a_v::Real, lambda_eV::Real, ne::Integer)
    lambda_ND = lambda_eV / (k_B_eV * T)
    eta_ND    = ne * F_const * eta / (R_gas * T)
    eta_ND_clamp = clamp(eta_ND, -700.0, 700.0)
    fermi = 1.0 / (1.0 + exp(eta_ND_clamp))
    arg   = (lambda_ND - sqrt(1.0 + sqrt(lambda_ND) + eta_ND^2)) / (2.0 * sqrt(lambda_ND))
    pf    = a_v * i0 / (ne * F_const * c_ref)
    return max(pf * sqrt(pi * lambda_ND) * fermi * erfc(arg), 0.0)
end

"""
    R_CO2R(c_CO2, phi_s, phi_L, c_H, a_v, lambda_eV, i0, n_e) -> [mol/m³/s]
"""
@inline function R_CO2R(c_CO2::Real, phi_s::Real, phi_L::Real, c_H::Real,
                        a_v::Real, lambda_eV::Real, i0::Real, ne::Integer)
    pH_local = -log10(max(c_H / 1000.0, 1e-20))
    U_eq = U_CO_0 - nernst_fac * pH_local
    eta  = (phi_s - phi_L) - U_eq
    k    = mhc_k_elec(eta, i0, a_v, lambda_eV, ne)
    return k * max(c_CO2, 0.0)
end

"""
    R_HER(phi_s, phi_L, c_H, a_v, n_e) -> [mol/m³/s]
"""
@inline function R_HER(phi_s::Real, phi_L::Real, c_H::Real, a_v::Real, ne::Integer)
    pH_local = -log10(max(c_H / 1000.0, 1e-20))
    U_eq_HER = -nernst_fac * pH_local
    eta_HER  = (phi_s - phi_L) - U_eq_HER
    i0_mA_cm2 = HER_prefactor() * exp(-(E_act_HER + dEact_dpH * pH_local) / (R_gas * T))
    i0 = i0_mA_cm2 * 10.0                       # mA/cm² → A/m²
    expo = clamp(-alpha_c_HER() * F_const * eta_HER / (R_gas * T), -700.0, 700.0)
    rate = (a_v / (ne * F_const)) * i0 * exp(expo)
    return max(rate, 0.0)
end
