# kinetics.jl — MHC CO₂R, Butler-Volmer HER, Sechenov Henry's law
# See Case_4_guide.md §4 (kinetics), §7.4 (Henry+Sechenov).
#
# All functions type-generic (accept ForwardDiff.Dual). Potentials in V vs SHE.
# c_H etc. in mol/m³. Kinetics take BOTH phi_s and phi_L explicitly; driving
# force is (phi_s - phi_L) - U_eq(local pH).

# ── MHC pseudo-first-order rate constant [1/s] ────────────────────────────────
function mhc_k_elec(eta::Real, i0::Real, a_v::Real, lambda_eV::Real, ne::Real)
    lambda_ND = lambda_eV / (k_B_eV * T)
    eta_ND = ne * F_const * eta / (R_gas * T)
    eta_ND_clamp = clamp(eta_ND, -700.0, 700.0)
    fermi = 1.0 / (1.0 + exp(eta_ND_clamp))
    arg = (lambda_ND - sqrt(1.0 + sqrt(lambda_ND) + eta_ND^2)) / (2.0 * sqrt(lambda_ND))
    pf = a_v * i0 / (ne * F_const * c_ref)
    return max(pf * sqrt(π * lambda_ND) * fermi * erfc(arg), 0.0)
end

# ── CO₂R volumetric rate [mol/m³_CL/s] (MHC) ──────────────────────────────────
function R_CO2R(c_CO2::Real, phi_s::Real, phi_L::Real, c_H::Real,
                a_v::Real, lambda_eV::Real, i0::Real, ne::Real)
    pH_local = -log10(max(c_H / 1000.0, 1e-20))
    U_eq = U_CO_0 - nernst_fac * pH_local
    eta = (phi_s - phi_L) - U_eq
    k = mhc_k_elec(eta, i0, a_v, lambda_eV, ne)
    return k * max(c_CO2, 0.0)
end

# ── HER volumetric rate [mol/m³_CL/s] (Butler-Volmer, cathodic branch) ────────
function R_HER(phi_s::Real, phi_L::Real, c_H::Real, a_v::Real, ne::Real)
    pH_local = -log10(max(c_H / 1000.0, 1e-20))
    U_eq_HER = -nernst_fac * pH_local
    eta_HER = (phi_s - phi_L) - U_eq_HER
    i0_mA_cm2 = HER_prefactor() * exp(-(E_act_HER + dEact_dpH * pH_local) / (R_gas * T))
    i0 = i0_mA_cm2 * 10.0                       # mA/cm² → A/m²
    exponent = clamp(-alpha_c_HER() * F_const * eta_HER / (R_gas * T), -700.0, 700.0)
    rate = (a_v / (ne * F_const)) * i0 * exp(exponent)
    return max(rate, 0.0)
end

# ── Cs⁺ from electroneutrality (§2/§3.1) ──────────────────────────────────────
@inline function cCs_from_neutrality(c::AbstractVector{<:Real})
    return c[2] + c[3] + 2*c[4] - c[1] - C_fix * eps_I
end

# ── Combined Henry constant: van't Hoff (T) + Sechenov (salting-out) §7.4 ──────
# Ion sum over the salting-out species Cs⁺, OH⁻, HCO₃⁻, CO₃²⁻ (c_Cs from
# electroneutrality). Returns H_eff [mol/(m³·atm)].
function sechenov_H_CO2(c::AbstractVector{<:Real})
    cH, cOH, cHCO3, cCO3 = c[1], c[2], c[3], c[4]
    cCs = cOH + cHCO3 + 2*cCO3 - cH - C_fix * eps_I
    expo = (h_CO2 + h_Cs)  * cCs   / 1000.0 +
           (h_CO2 + h_OH)  * cOH   / 1000.0 +
           (h_CO2 + h_HCO3)* cHCO3 / 1000.0 +
           (h_CO2 + h_CO3) * cCO3  / 1000.0
    Hv = H_CO2_0 * exp(dH_sol * (1/T - 1/298.15))
    return Hv * 10.0^(-expo)
end

nothing
