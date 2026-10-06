# transport.jl — face flux functions (§9.1) + Henry/Sechenov gas-liquid equilibrium (§7.4)

"""Pure diffusion face flux [mol/m²/s] (Stages 1–4). `dx` = center-to-center distance."""
diffusion_flux(c_L::Real, c_R::Real, D_eff_::Real, dx::Real) = -D_eff_ * (c_R - c_L) / dx

"""Nernst-Planck face flux (diffusion + migration), Stage 5 (§9.1)."""
function np_flux(c_L::Real, c_R::Real, phi_L_left::Real, phi_L_right::Real,
                 D_eff_::Real, z_i::Real, dx_cc::Real)
    c_avg = 0.5 * (c_L + c_R)
    diff  = -D_eff_ * (c_R - c_L) / dx_cc
    migr  = -(z_i * F_const / (R_gas * T)) * D_eff_ * c_avg * (phi_L_right - phi_L_left) / dx_cc
    return diff + migr
end

# ─────────────────────────────────────────────────────────────────────────────
# Henry's law (§7.1 van't Hoff) and combined Henry+Sechenov (§7.4)

"""van't Hoff Henry constant [mol/(m³·atm)] — used Stages 1–2 (no Sechenov)."""
H_vanthoff() = H_CO2_0 * exp(dH_sol * (1.0/T - 1.0/298.15))   # ≈ 18.24 at 50°C

"""Back-derive free Cs⁺ from electroneutrality (§2, §3.1) [mol/m³]."""
c_Cs_from_electroneutrality(c::AbstractVector{<:Real}) =
    c[2] + c[3] + 2*c[4] - c[1] - C_fix * eps_I

"""
    sechenov_H_CO2(c) -> H_eff [mol/(m³·atm)]

Combined van't Hoff + Sechenov effective Henry constant (§7.4). Salting-out by
the local ions (Cs⁺ from electroneutrality, OH⁻, HCO₃⁻, CO₃²⁻). Used Stage 3+.
"""
function sechenov_H_CO2(c::AbstractVector{<:Real})
    cCs = c_Cs_from_electroneutrality(c)
    exponent = -((h_CO2 + h_Cs)  * cCs   +
                 (h_CO2 + h_OH)  * c[2]  +
                 (h_CO2 + h_HCO3)* c[3]  +
                 (h_CO2 + h_CO3) * c[4]) / 1000.0
    return H_vanthoff() * 10.0^exponent
end
