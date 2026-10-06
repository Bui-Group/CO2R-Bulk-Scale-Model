# chemistry.jl — buffer chemistry sources + exact equilibrium initial guess
# See Case_4_guide.md §5 (rates) and §9.7 (equilibrium).
#
# All concentrations mol/m³. Species order: [H⁺, OH⁻, HCO₃⁻, CO₃²⁻, CO₂(aq)].

using Roots

# Net volumetric buffer source per species [mol/m³/s]. Type-generic for ForwardDiff.
# NOTE: this is the RAW rate (per unit ionomer+liquid volume basis is applied by the
# caller as eps_I × buffer_sources; see §5 "multiply by ε_I × dx").
function buffer_sources(c::AbstractVector{<:Real})
    cH, cOH, cHCO3, cCO3, cCO2 = c[1], c[2], c[3], c[4], c[5]

    r1 = k1f            - k1r * cH * cOH        # H₂O ⇌ H⁺ + OH⁻
    r2 = k2f * cCO2     - k2r * cH * cHCO3      # CO₂ + H₂O ⇌ H⁺ + HCO₃⁻
    r3 = k3f * cHCO3    - k3r * cH * cCO3       # HCO₃⁻ ⇌ H⁺ + CO₃²⁻
    r4 = k4f * cCO2*cOH - k4r * cHCO3           # CO₂ + OH⁻ ⇌ HCO₃⁻
    r5 = k5f * cHCO3*cOH - k5r * cCO3           # HCO₃⁻ + OH⁻ ⇌ CO₃²⁻ + H₂O

    S_H    =  r1 + r2 + r3
    S_OH   =  r1 - r4 - r5
    S_HCO3 =  r2 - r3 + r4 - r5
    S_CO3  =  r3 + r5
    S_CO2  = -r2 - r4
    return (S_H, S_OH, S_HCO3, S_CO3, S_CO2)
end

# Solve the carbonate/charge-balance equilibrium exactly (§9.7). Returns NamedTuple.
function solve_buffer_equilibrium(c_Cs::Float64 = 1000.0, C_total::Float64 = 1000.0)
    function charge_balance(log_cH)
        cH    = 10.0^log_cH
        cOH   = K_w / cH
        denom = 1.0 + K_2/cH + K_2*K_3/cH^2
        cCO2  = C_total / denom
        cHCO3 = K_2 * cCO2 / cH
        cCO3  = K_3 * cHCO3 / cH
        return cH + c_Cs - cOH - cHCO3 - 2*cCO3
    end
    log_cH = find_zero(charge_balance, (-6.0, 0.0), Bisection())
    cH    = 10.0^log_cH
    cOH   = K_w / cH
    denom = 1.0 + K_2/cH + K_2*K_3/cH^2
    cCO2  = C_total / denom
    cHCO3 = K_2 * cCO2 / cH
    cCO3  = K_3 * cHCO3 / cH
    return (H=cH, OH=cOH, HCO3=cHCO3, CO3=cCO3, CO2=cCO2,
            Cs=c_Cs, pH=-log10(cH/1000.0))
end

# Convenience: 5-vector [H,OH,HCO3,CO3,CO2] of the bulk equilibrium.
function equilibrium_vec(c_Cs::Float64 = 1000.0, C_total::Float64 = 1000.0)
    e = solve_buffer_equilibrium(c_Cs, C_total)
    return [e.H, e.OH, e.HCO3, e.CO3, e.CO2]
end

nothing
