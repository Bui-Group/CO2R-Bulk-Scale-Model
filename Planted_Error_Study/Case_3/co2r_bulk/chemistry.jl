# chemistry.jl — buffer chemistry source terms (§5) and equilibrium initial guess (§9.7)
using Roots

"""
    buffer_sources(c) -> 5-vector  [mol/m³/s]

Net volumetric buffer source per species, indices 1:H⁺ 2:OH⁻ 3:HCO₃⁻ 4:CO₃²⁻ 5:CO₂(aq).
Transcribed verbatim from guide §5. Type-generic for ForwardDiff.
NOTE: per-ionomer-volume rate; the caller multiplies by eps_I (and dx) — see §5 / residual.
"""
function buffer_sources(c::AbstractVector{ElT}) where {ElT<:Real}
    c_H, c_OH, c_HCO3, c_CO3, c_CO2 = c[1], c[2], c[3], c[4], c[5]
    S = zeros(ElT, 5)
    S[1] = k1f - k1r*c_H*c_OH + k2f*c_CO2 - k2r*c_H*c_HCO3 + k3f*c_HCO3 - k3r*c_H*c_CO3
    S[2] = k1f - k1r*c_H*c_OH - k4f*c_CO2*c_OH + k4r*c_HCO3 - k5f*c_HCO3*c_OH + k5r*c_CO3
    S[3] = k2f*c_CO2 - k2r*c_H*c_HCO3 + k4f*c_CO2*c_OH - k4r*c_HCO3 -
           k3f*c_HCO3 + k3r*c_H*c_CO3 - k5f*c_HCO3*c_OH + k5r*c_CO3
    S[4] = k3f*c_HCO3 - k3r*c_H*c_CO3 + k5f*c_HCO3*c_OH - k5r*c_CO3
    S[5] = -k2f*c_CO2 + k2r*c_H*c_HCO3 - k4f*c_CO2*c_OH + k4r*c_HCO3
    return S
end

"""
    solve_buffer_equilibrium(c_Cs=1000, C_total=1000) -> NamedTuple

Exact charge-balance equilibrium for the CO₂/carbonate system (§9.7).
Returns concentrations [mol/m³] and pH.
"""
function solve_buffer_equilibrium(c_Cs_mol_m3=1000.0, C_total_mol_m3=1000.0)
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

# Equilibrium concentration vector [H, OH, HCO3, CO3, CO2] in mol/m³.
function equilibrium_c_vec(c_Cs=1000.0, C_total=1000.0)
    eq = solve_buffer_equilibrium(c_Cs, C_total)
    return [eq.H, eq.OH, eq.HCO3, eq.CO3, eq.CO2]
end
