# chemistry.jl — buffer chemistry sources and the exact equilibrium (§5, §9.6)

using Roots      # §9.6: MUST be at top level

"""
    buffer_sources(c) -> NTuple{5}

Net volumetric buffer source [mol/m³/s] per species for the concentration
5-vector `c = (c_H, c_OH, c_HCO3, c_CO3, c_CO2)` in mol/m³ (§5).
Type-generic for ForwardDiff.
"""
@inline function buffer_sources(c::AbstractVector{<:Real})
    cH, cOH, cHCO3, cCO3, cCO2 = c[1], c[2], c[3], c[4], c[5]

    # reaction rates (forward − reverse), each in mol/m³/s
    R1 = k1 - k1r * cH * cOH                          # H₂O ⇌ H⁺ + OH⁻
    R2 = k2 * cCO2 - k2r * cH * cHCO3                 # CO₂ + H₂O ⇌ H⁺ + HCO₃⁻
    R3 = k3 * cHCO3 - k3r * cH * cCO3                 # HCO₃⁻ ⇌ H⁺ + CO₃²⁻
    R4 = k4 * cCO2 * cOH - k4r * cHCO3                # CO₂ + OH⁻ ⇌ HCO₃⁻
    R5 = k5 * cHCO3 * cOH - k5r * cCO3                # HCO₃⁻ + OH⁻ ⇌ CO₃²⁻ + H₂O

    S_H    =  R1 + R2 + R3
    S_OH   =  R1 - R4 - R5
    S_HCO3 =  R2 + R4 - R3 - R5
    S_CO3  =  R3 + R5
    S_CO2  = -R2 - R4
    return (S_H, S_OH, S_HCO3, S_CO3, S_CO2)
end

"""
    solve_buffer_equilibrium(c_Cs = 1000, C_total = 1000) -> NamedTuple

Exact charge-balance solution of the carbonate system (§9.6).  All
concentrations in mol/m³.
"""
function solve_buffer_equilibrium(c_Cs_mol_m3 = 1000.0, C_total_mol_m3 = 1000.0)
    function charge_balance(log_cH)
        cH    = 10.0^log_cH
        cOH   = K_w / cH
        denom = 1.0 + K_2/cH + K_2*K_3/cH^2
        cCO2  = C_total_mol_m3 / denom
        cHCO3 = K_2 * cCO2 / cH
        cCO3  = K_3 * cHCO3 / cH
        return cH + c_Cs_mol_m3 - cOH - cHCO3 - 2*cCO3
    end

    log_cH = find_zero(charge_balance, (-6.0, 0.0), Bisection())
    cH    = 10.0^log_cH
    cOH   = K_w / cH
    denom = 1.0 + K_2/cH + K_2*K_3/cH^2
    cCO2  = C_total_mol_m3 / denom
    cHCO3 = K_2 * cCO2 / cH
    cCO3  = K_3 * cHCO3 / cH
    return (H = cH, OH = cOH, HCO3 = cHCO3, CO3 = cCO3, CO2 = cCO2,
            Cs = c_Cs_mol_m3, pH = -log10(cH / 1000.0))
end

"5-vector [H, OH, HCO3, CO3, CO2] of the 1 M CsHCO₃ bulk equilibrium."
function equilibrium_vector()
    e = solve_buffer_equilibrium(1000.0, 1000.0)
    return [e.H, e.OH, e.HCO3, e.CO3, e.CO2]
end

"§2 / §3.1 Cs⁺ recovered from electroneutrality [mol/m³]."
@inline c_Cs_from_electroneutrality(c::AbstractVector{<:Real}) =
    c[2] + c[3] + 2*c[4] - c[1] - C_fix * eps_I

"""
    sechenov_H_CO2(c) -> H_eff [mol/(m³ atm)]

Combined van 't Hoff (temperature) + Sechenov (salting-out) Henry constant
(§7.4).  `c_Cs` is the electroneutrality value, per §7.4.
"""
@inline function sechenov_H_CO2(c::AbstractVector{<:Real})
    cCs = c_Cs_from_electroneutrality(c)
    expo = ((h_CO2 + h_Cs)   * cCs   +
            (h_CO2 + h_OH)   * c[2]  +
            (h_CO2 + h_HCO3) * c[3]  +
            (h_CO2 + h_CO3)  * c[4]) / 1000.0
    return H_CO2_0 * exp(dH_sol * (1.0/T - 1.0/298.15)) * 10.0^(-expo)
end
