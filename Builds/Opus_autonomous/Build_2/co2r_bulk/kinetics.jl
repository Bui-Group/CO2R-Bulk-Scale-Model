# kinetics.jl — MHC CO₂R (§4.1) and Butler-Volmer HER (§4.2)
#
# All potential arguments are V vs SHE.  Both functions take phi_s_local AND
# phi_L_local explicitly; Stages 1–4 pass (V_applied_SHE, 0.0).

using SpecialFunctions: erfc

"""
    mhc_k_elec(eta, i0, a_v, lambda_eV, ne) -> k [1/s]

Marcus-Hush-Chidsey pseudo-first-order rate constant (§4.1).  `lambda_eV` is in
eV; λ_ND is formed internally.
"""
@inline function mhc_k_elec(eta::Real, i0::Real, a_v::Real, lambda_eV::Real, ne::Real)
    lambda_ND = lambda_eV / (k_B_eV * T)
    eta_ND    = ne * F_const * eta / (R_gas * T)
    eta_ND_c  = clamp(eta_ND, -700.0, 700.0)
    fermi     = 1.0 / (1.0 + exp(eta_ND_c))
    arg       = (lambda_ND - sqrt(1.0 + sqrt(lambda_ND) + eta_ND^2)) / (2.0 * sqrt(lambda_ND))
    pf        = a_v * i0 / (ne * F_const * c_ref)
    return max(pf * sqrt(pi * lambda_ND) * fermi * erfc(arg), 0.0)
end

"""
    R_CO2R(c_CO2, phi_s, phi_L, c_H, a_v, lambda_eV, i0, ne) -> mol/m³_CL/s
"""
@inline function R_CO2R(c_CO2::Real, phi_s::Real, phi_L::Real, c_H::Real,
                        a_v::Real, lambda_eV::Real, i0::Real, ne::Real)
    pH_local = -log10(max(c_H / 1000.0, 1e-20))
    U_eq     = U_CO_0 - nernst_fac * pH_local
    eta      = (phi_s - phi_L) - U_eq          # MUST subtract phi_L
    k        = mhc_k_elec(eta, i0, a_v, lambda_eV, ne)
    return k * max(c_CO2, 0.0)
end

"""
    R_HER(phi_s, phi_L, c_H, a_v, ne) -> mol/m³_CL/s
"""
@inline function R_HER(phi_s::Real, phi_L::Real, c_H::Real, a_v::Real, ne::Real)
    pH_local  = -log10(max(c_H / 1000.0, 1e-20))
    U_eq_HER  = -nernst_fac * pH_local
    eta_HER   = (phi_s - phi_L) - U_eq_HER     # MUST subtract phi_L
    i0_mAcm2  = HER_prefactor() * exp(-(E_act_HER + dEact_dpH * pH_local) / (R_gas * T))
    i0        = i0_mAcm2 * 10.0                # mA/cm² → A/m²
    expo      = clamp(-alpha_c_HER() * F_const * eta_HER / (R_gas * T), -700.0, 700.0)
    rate      = (a_v / (ne * F_const)) * i0 * exp(expo)
    return max(rate, 0.0)
end
