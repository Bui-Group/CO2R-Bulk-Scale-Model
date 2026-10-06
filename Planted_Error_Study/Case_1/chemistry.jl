# chemistry.jl — Buffer chemistry sources (§5) and exact equilibrium (§9.7)
# Species order: 1=H⁺, 2=OH⁻, 3=HCO₃⁻, 4=CO₃²⁻, 5=CO₂(aq).  Units: mol/m³, mol/m³/s.

using Roots

"""
    buffer_sources(c) -> NTuple{5}

Net volumetric reaction source for each species [mol/m³/s] from the five
equilibrium reactions (§5, lines 428–432). Type-generic for ForwardDiff.
"""
function buffer_sources(c::AbstractVector{<:Real})
    cH, cOH, cHCO3, cCO3, cCO2 = c[1], c[2], c[3], c[4], c[5]
    S_H    =  k1 - k1r*cH*cOH + k2*cCO2 - k2r*cH*cHCO3 + k3*cHCO3 - k3r*cH*cCO3
    S_OH   =  k1 - k1r*cH*cOH - k4*cCO2*cOH + k4r*cHCO3 - k5*cHCO3*cOH + k5r*cCO3
    S_HCO3 =  k2*cCO2 - k2r*cH*cHCO3 + k4*cCO2*cOH - k4r*cHCO3 -
              k3*cHCO3 + k3r*cH*cCO3 - k5*cHCO3*cOH + k5r*cCO3
    S_CO3  =  k3*cHCO3 - k3r*cH*cCO3 + k5*cHCO3*cOH - k5r*cCO3
    S_CO2  = -k2*cCO2 + k2r*cH*cHCO3 - k4*cCO2*cOH + k4r*cHCO3
    return (S_H, S_OH, S_HCO3, S_CO3, S_CO2)
end

"""
    solve_buffer_equilibrium(c_Cs=1000, C_total=1000)

Solve the bulk charge balance exactly for log10(c_H) (§9.7) and return the five
equilibrium concentrations + Cs⁺ + pH. All in mol/m³.
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

"Return the 5-vector [H, OH, HCO3, CO3, CO2] of bulk equilibrium concentrations."
function equilibrium_vector(c_Cs=1000.0, C_total=1000.0)
    eq = solve_buffer_equilibrium(c_Cs, C_total)
    return [eq.H, eq.OH, eq.HCO3, eq.CO3, eq.CO2]
end
