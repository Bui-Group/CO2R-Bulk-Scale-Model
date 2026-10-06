# chemistry.jl — buffer chemistry sources (§5) and the exact bulk equilibrium (§9.6)

using Roots     # top level — a `using` inside a function body is a syntax error

"""
    buffer_sources(cH, cOH, cHCO3, cCO3, cCO2) -> NTuple{5}

Net volumetric buffer source per species [mol/m³/s], §5.  Order matches the DOF
layout: H⁺, OH⁻, HCO₃⁻, CO₃²⁻, CO₂(aq).  Reverse rates are always derived.
"""
@inline function buffer_sources(cH::Real, cOH::Real, cHCO3::Real, cCO3::Real, cCO2::Real)
    r1 = k1f - k1r * cH * cOH                 # H₂O ⇌ H⁺ + OH⁻
    r2 = k2f * cCO2 - k2r * cH * cHCO3        # CO₂ + H₂O ⇌ H⁺ + HCO₃⁻
    r3 = k3f * cHCO3 - k3r * cH * cCO3        # HCO₃⁻ ⇌ H⁺ + CO₃²⁻
    r4 = k4f * cCO2 * cOH - k4r * cHCO3       # CO₂ + OH⁻ ⇌ HCO₃⁻
    r5 = k5f * cHCO3 * cOH - k5r * cCO3       # HCO₃⁻ + OH⁻ ⇌ CO₃²⁻ + H₂O
    S_H    =  r1 + r2 + r3
    S_OH   =  r1 - r4 - r5
    S_HCO3 =  r2 + r4 - r3 - r5
    S_CO3  =  r3 + r5
    S_CO2  = -r2 - r4
    return (S_H, S_OH, S_HCO3, S_CO3, S_CO2)
end

"""
    solve_buffer_equilibrium(c_Cs=1000.0, C_total=1000.0)

Exact bulk carbonate equilibrium by bisection on the charge balance (§9.6).
Returns a NamedTuple in mol/m³.
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

"5-vector [H⁺, OH⁻, HCO₃⁻, CO₃²⁻, CO₂(aq)] of the bulk equilibrium, mol/m³"
function equilibrium_vector(c_Cs::Float64=1000.0, C_total::Float64=1000.0)
    e = solve_buffer_equilibrium(c_Cs, C_total)
    return [e.H, e.OH, e.HCO3, e.CO3, e.CO2]
end
