# chemistry.jl — buffer_sources! / buffer_sources, solve_buffer_equilibrium (§5, §9.6)

using Roots

"c_Cs from electroneutrality (§2, §3.1): c_Cs = c_OH + c_HCO3 + 2 c_CO3 - c_H - C_fix*eps_I"
c_Cs_from_electroneutrality(cH, cOH, cHCO3, cCO3) = cOH + cHCO3 + 2 * cCO3 - cH - C_fix * eps_I

"""
    buffer_sources(c) -> NTuple{5,ElT}

Net volumetric buffer source [mol/m^3/s] for (H+, OH-, HCO3-, CO3^2-, CO2(aq)), §5.
Type-generic over `c`'s element type so ForwardDiff duals propagate.
"""
function buffer_sources(c::AbstractVector{ElT}) where {ElT<:Real}
    cH, cOH, cHCO3, cCO3, cCO2 = c[1], c[2], c[3], c[4], c[5]

    S_H = k1_f - k1_r * cH * cOH + k2_f * cCO2 - k2_r * cH * cHCO3 +
          k3_f * cHCO3 - k3_r * cH * cCO3
    S_OH = k1_f - k1_r * cH * cOH - k4_f * cCO2 * cOH + k4_r * cHCO3 -
           k5_f * cHCO3 * cOH + k5_r * cCO3
    S_HCO3 = k2_f * cCO2 - k2_r * cH * cHCO3 + k4_f * cCO2 * cOH - k4_r * cHCO3 -
             k3_f * cHCO3 + k3_r * cH * cCO3 - k5_f * cHCO3 * cOH + k5_r * cCO3
    S_CO3 = k3_f * cHCO3 - k3_r * cH * cCO3 + k5_f * cHCO3 * cOH - k5_r * cCO3
    S_CO2 = -k2_f * cCO2 + k2_r * cH * cHCO3 - k4_f * cCO2 * cOH + k4_r * cHCO3

    return (S_H, S_OH, S_HCO3, S_CO3, S_CO2)
end

"""
    solve_buffer_equilibrium(c_Cs_mol_m3=1000.0, C_total_mol_m3=1000.0)

Exact charge-balance solve for the bulk 1M CsHCO3 equilibrium (§9.6). Returns a NamedTuple
(H, OH, HCO3, CO3, CO2, Cs, pH). All concentrations in mol/m^3.
"""
function solve_buffer_equilibrium(c_Cs_mol_m3::Float64=1000.0, C_total_mol_m3::Float64=1000.0)
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
    cH = 10.0^log_cH
    cOH = K_w / cH
    denom = 1.0 + K_2/cH + K_2*K_3/cH^2
    cCO2  = C_total_mol_m3 / denom
    cHCO3 = K_2 * cCO2 / cH
    cCO3  = K_3 * cHCO3 / cH
    return (H=cH, OH=cOH, HCO3=cHCO3, CO3=cCO3, CO2=cCO2,
            Cs=c_Cs_mol_m3, pH=-log10(cH/1000.0))
end
