# =============================================================================
# chemistry.jl — carbonate buffer chemistry (BaseCase.md §5, §9.7)
# All concentrations in mol/m³. Type-generic for ForwardDiff.
# =============================================================================

using Roots

"""
    buffer_sources(cH, cOH, cHCO3, cCO3, cCO2) -> (S_H,S_OH,S_HCO3,S_CO3,S_CO2)

Net volumetric production rates [mol/m³/s] for the 5 carbonate-system reactions (§5).
Net forward reaction rates r1..r5 (forward − reverse):
  r1: H₂O ⇌ H⁺+OH⁻ ; r2: CO₂+H₂O ⇌ H⁺+HCO₃⁻ ; r3: HCO₃⁻ ⇌ H⁺+CO₃²⁻
  r4: CO₂+OH⁻ ⇌ HCO₃⁻ ; r5: HCO₃⁻+OH⁻ ⇌ CO₃²⁻+H₂O
"""
@inline function buffer_sources(cH::Real, cOH::Real, cHCO3::Real, cCO3::Real, cCO2::Real)
    r1 = k1 - k1r * cH * cOH
    r2 = k2 * cCO2 - k2r * cH * cHCO3
    r3 = k3 * cHCO3 - k3r * cH * cCO3
    r4 = k4 * cCO2 * cOH - k4r * cHCO3
    r5 = k5 * cHCO3 * cOH - k5r * cCO3

    S_H    = r1 + r2 + r3
    S_OH   = r1 - r4 - r5
    S_HCO3 = r2 + r4 - r3 - r5
    S_CO3  = r3 + r5
    S_CO2  = -r2 - r4
    return (S_H, S_OH, S_HCO3, S_CO3, S_CO2)
end

# Convenience overload taking a 5-vector/view in order [H,OH,HCO3,CO3,CO2].
@inline buffer_sources(c) = buffer_sources(c[1], c[2], c[3], c[4], c[5])

"""
    solve_buffer_equilibrium(c_Cs=1000.0, C_total=1000.0)

Exact bulk-electrolyte equilibrium of 1 M CsHCO₃ at 50 °C by solving the charge
balance for c_H (§9.7). Returns a NamedTuple in mol/m³ plus pH.
Note: bulk electrolyte has NO ionomer fixed charge, so the balance is
      cH + c_Cs − cOH − cHCO3 − 2 cCO3 = 0.
"""
function solve_buffer_equilibrium(c_Cs::Float64=1000.0, C_total::Float64=1000.0)
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

# 5-vector [H,OH,HCO3,CO3,CO2] of the equilibrium composition (Dirichlet reservoir value).
function equilibrium_vector(c_Cs::Float64=1000.0, C_total::Float64=1000.0)
    eq = solve_buffer_equilibrium(c_Cs, C_total)
    return [eq.H, eq.OH, eq.HCO3, eq.CO3, eq.CO2]
end
