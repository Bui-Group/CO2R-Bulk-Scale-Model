# chemistry.jl — buffer chemistry (§5) and the exact equilibrium solve (§9.6)

using Roots

"""
    buffer_sources(c) -> NTuple{5,ElT}

Net volumetric buffer source [mol/m³/s] for
(H⁺, OH⁻, HCO₃⁻, CO₃²⁻, CO₂(aq)) — §5.  Type-generic for ForwardDiff.
"""
@inline function buffer_sources(c::AbstractVector{ElT}) where {ElT<:Real}
    cH, cOH, cHCO3, cCO3, cCO2 = c[1], c[2], c[3], c[4], c[5]

    r1 = k1        - k1r * cH * cOH
    r2 = k2 * cCO2 - k2r * cH * cHCO3
    r3 = k3 * cHCO3 - k3r * cH * cCO3
    r4 = k4 * cCO2 * cOH - k4r * cHCO3
    r5 = k5 * cHCO3 * cOH - k5r * cCO3

    S_H    = r1 + r2 + r3
    S_OH   = r1 - r4 - r5
    S_HCO3 = r2 - r3 + r4 - r5
    S_CO3  = r3 + r5
    S_CO2  = -r2 - r4
    return (S_H, S_OH, S_HCO3, S_CO3, S_CO2)
end

"""
    solve_buffer_equilibrium(c_Cs=1000.0, C_total=1000.0)

Exact charge-balance solve for the bulk electrolyte equilibrium (§9.6).
"""
function solve_buffer_equilibrium(c_Cs_mol_m3::Float64=1000.0,
                                  C_total_mol_m3::Float64=1000.0)
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

"""Back-derive free Cs⁺ from electroneutrality (§2 / §3.1)."""
@inline cCs_from_electroneutrality(cH, cOH, cHCO3, cCO3) =
    cOH + cHCO3 + 2*cCO3 - cH - C_fix_eff

"""Combined van 't Hoff + Sechenov Henry constant [mol/(m³·atm)] (§7.4)."""
@inline function sechenov_H_CO2(c::AbstractVector{ElT}) where {ElT<:Real}
    cH, cOH, cHCO3, cCO3 = c[1], c[2], c[3], c[4]
    cCs = cCs_from_electroneutrality(cH, cOH, cHCO3, cCO3)
    expo = -( (h_CO2 + h_Cs)   * cCs   / 1000.0 +
              (h_CO2 + h_OH)   * cOH   / 1000.0 +
              (h_CO2 + h_HCO3) * cHCO3 / 1000.0 +
              (h_CO2 + h_CO3)  * cCO3  / 1000.0 )
    return H_vantHoff * exp10(expo)
end
