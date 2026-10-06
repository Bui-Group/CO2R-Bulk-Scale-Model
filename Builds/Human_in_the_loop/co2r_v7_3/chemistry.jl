# ============================================================================
# chemistry.jl — buffer chemistry (§5) and the exact equilibrium guess (§9.6)
# ============================================================================
using Roots     # §9.6: MUST be at top level

"""
    buffer_sources(c) -> NTuple{5,ElT}

Net volumetric buffer source for (H⁺, OH⁻, HCO₃⁻, CO₃²⁻, CO₂(aq)) in
mol/m³/s, from the five equilibria of §5.  Type-generic for ForwardDiff.
"""
@inline function buffer_sources(c::AbstractVector{ElT}) where {ElT<:Real}
    cH, cOH, cHCO3, cCO3, cCO2 = c[1], c[2], c[3], c[4], c[5]

    r1 = k_1 - k_1r * cH * cOH                 # H2O ⇌ H+ + OH-
    r2 = k_2 * cCO2 - k_2r * cH * cHCO3        # CO2 + H2O ⇌ H+ + HCO3-
    r3 = k_3 * cHCO3 - k_3r * cH * cCO3        # HCO3- ⇌ H+ + CO3^2-
    r4 = k_4 * cCO2 * cOH - k_4r * cHCO3       # CO2 + OH- ⇌ HCO3-
    r5 = k_5 * cHCO3 * cOH - k_5r * cCO3       # HCO3- + OH- ⇌ CO3^2- + H2O

    S_H    = r1 + r2 + r3
    S_OH   = r1 - r4 - r5
    S_HCO3 = r2 + r4 - r3 - r5
    S_CO3  = r3 + r5
    S_CO2  = -r2 - r4
    return (S_H, S_OH, S_HCO3, S_CO3, S_CO2)
end

"""
    solve_buffer_equilibrium(c_Cs_mol_m3 = 1000.0, C_total_mol_m3 = 1000.0)

Exact charge-balance solve for the bulk electrolyte equilibrium (§9.6).
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
    return (H=cH, OH=cOH, HCO3=cHCO3, CO3=cCO3, CO2=cCO2,
            Cs=c_Cs_mol_m3, pH=-log10(cH/1000.0))
end

"c_Cs from local electroneutrality (§2 line 63)."
@inline c_Cs_from_electroneutrality(c::AbstractVector{<:Real}) =
    c[2] + c[3] + 2*c[4] - c[1] - C_fix * eps_I

"""
    sechenov_H_CO2(c) -> H_eff [mol/(m³ atm)]

Combined van 't Hoff + Sechenov effective Henry constant (§7.4), using the
electroneutrality-derived c_Cs.
"""
@inline function sechenov_H_CO2(c::AbstractVector{ElT}) where {ElT<:Real}
    cCs = c_Cs_from_electroneutrality(c)
    expo = (h_CO2 + h_Cs)   * cCs   / 1000.0 +
           (h_CO2 + h_OH)   * c[2]  / 1000.0 +
           (h_CO2 + h_HCO3) * c[3]  / 1000.0 +
           (h_CO2 + h_CO3)  * c[4]  / 1000.0
    return H_vantHoff * 10.0^(-expo)
end
