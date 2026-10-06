# =============================================================================
# kinetics.jl — MHC for CO2R (§4.1) and Butler-Volmer for HER (§4.2)
# All potential arguments are V vs SHE.  Both functions take phi_L explicitly;
# in Stages 1-4 the caller passes phi_s = V_applied_SHE and phi_L = 0.
# =============================================================================
using SpecialFunctions      # erfc; ForwardDiff differentiates it via DiffRules

"""
    mhc_k_elec(eta, i0, a_v, lambda_eV, n_elec) -> k [1/s]

Marcus-Hush-Chidsey pseudo-first-order rate constant (§4.1).  `lambda_eV` is
in eV; the dimensionless reorganisation energy is computed internally.
"""
@inline function mhc_k_elec(eta::Real, i0::Real, a_v::Real,
                            lambda_eV::Real, n_elec::Integer)
    lambda_ND = lambda_eV / (k_B_eV * T)
    eta_ND    = n_elec * F_const * eta / (R_gas * T)
    eta_ND_c  = clamp(eta_ND, -700.0, 700.0)
    fermi     = 1.0 / (1.0 + exp(eta_ND_c))
    arg       = (lambda_ND - sqrt(1.0 + sqrt(lambda_ND) + eta_ND^2)) /
                (2.0 * sqrt(lambda_ND))
    pf        = a_v * i0 / (n_elec * F_const * c_ref)
    return max(pf * sqrt(pi * lambda_ND) * fermi * erfc(arg), 0.0)
end

"""
    R_CO2R(c_CO2, phi_s, phi_L, c_H, a_v, lambda_eV, i0, n_elec) -> mol/m^3/s

Volumetric CO2R rate.  `eta = (phi_s - phi_L) - U_eq(pH_local)` — phi_L MUST
be subtracted (§4.1, §15.3).
"""
@inline function R_CO2R(c_CO2::Real, phi_s::Real, phi_L::Real, c_H::Real,
                        a_v::Real, lambda_eV::Real, i0::Real, n_elec::Integer)
    pH_local = -log10(max(c_H / 1000.0, 1e-20))
    U_eq = U_CO_0 - nernst_fac * pH_local
    eta  = (phi_s - phi_L) - U_eq
    k    = mhc_k_elec(eta, i0, a_v, lambda_eV, n_elec)
    return k * max(c_CO2, 0.0)
end

"""
    R_HER(phi_s, phi_L, c_H, a_v, n_elec) -> mol/m^3/s

Volumetric HER rate, cathodic Butler-Volmer branch only (§4.2).
`i0_HER(pH) = HER_prefactor * exp(-(83000 + 1000*pH)/(RT))` in mA/cm^2, x10
to A/m^2.
"""
@inline function R_HER(phi_s::Real, phi_L::Real, c_H::Real,
                       a_v::Real, n_elec::Integer)
    pH_local  = -log10(max(c_H / 1000.0, 1e-20))
    U_eq_HER  = -nernst_fac * pH_local
    eta_HER   = (phi_s - phi_L) - U_eq_HER
    i0_mA_cm2 = HER_prefactor() * exp(-(E_act_HER + dEact_dpH * pH_local) / (R_gas * T))
    i0        = i0_mA_cm2 * 10.0                       # A/m^2
    expo      = clamp(-alpha_c_HER() * F_const * eta_HER / (R_gas * T), -700.0, 700.0)
    rate      = (a_v / (n_elec * F_const)) * i0 * exp(expo)
    return max(rate, 0.0)
end
