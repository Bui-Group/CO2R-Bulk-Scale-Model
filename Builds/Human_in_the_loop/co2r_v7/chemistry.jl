# =====================================================================
#  chemistry.jl — buffer chemistry sources and equilibrium (§5, §9.6)
# =====================================================================

using Roots

"""
    buffer_sources(c) -> NTuple{5}

Net volumetric buffer source for each liquid species [mol/m³/s] (§5).
`c` is a 5-vector (H⁺, OH⁻, HCO₃⁻, CO₃²⁻, CO₂(aq)) in mol/m³.
Type-generic for ForwardDiff (§9.3.2).
"""
@inline function buffer_sources(c::AbstractVector{<:Real})
    cH, cOH, cHCO3, cCO3, cCO2 = c[1], c[2], c[3], c[4], c[5]

    r1 = k1f            - k1r * cH * cOH            # H2O ⇌ H+ + OH-
    r2 = k2f * cCO2     - k2r * cH * cHCO3          # CO2 + H2O ⇌ H+ + HCO3-
    r3 = k3f * cHCO3    - k3r * cH * cCO3           # HCO3- ⇌ H+ + CO3^2-
    r4 = k4f * cCO2*cOH - k4r * cHCO3               # CO2 + OH- ⇌ HCO3-
    r5 = k5f * cHCO3*cOH- k5r * cCO3                # HCO3- + OH- ⇌ CO3^2- + H2O

    S_H    =  r1 + r2 + r3
    S_OH   =  r1 - r4 - r5
    S_HCO3 =  r2 - r3 + r4 - r5
    S_CO3  =  r3 + r5
    S_CO2  = -r2 - r4
    return (S_H, S_OH, S_HCO3, S_CO3, S_CO2)
end

"""
    solve_buffer_equilibrium(c_Cs = 1000, C_total = 1000)

Exact bulk equilibrium of the CO₂/carbonate system by bisection on the charge
balance (§9.6).  All concentrations in mol/m³.
"""
function solve_buffer_equilibrium(c_Cs_mol_m3::Float64 = 1000.0,
                                  C_total_mol_m3::Float64 = 1000.0)
    function speciate(log_cH)
        cH    = 10.0^log_cH
        cOH   = K_w / cH
        denom = 1.0 + K_2/cH + K_2*K_3/cH^2
        cCO2  = C_total_mol_m3 / denom
        cHCO3 = K_2 * cCO2 / cH
        cCO3  = K_3 * cHCO3 / cH
        return (cH, cOH, cHCO3, cCO3, cCO2)
    end
    charge_balance(l) = ((cH, cOH, cHCO3, cCO3, _) = speciate(l);
                         cH + c_Cs_mol_m3 - cOH - cHCO3 - 2*cCO3)

    log_cH = find_zero(charge_balance, (-6.0, 0.0), Bisection())
    cH, cOH, cHCO3, cCO3, cCO2 = speciate(log_cH)
    return (H = cH, OH = cOH, HCO3 = cHCO3, CO3 = cCO3, CO2 = cCO2,
            Cs = c_Cs_mol_m3, pH = -log10(cH / 1000.0))
end

"5-vector of equilibrium concentrations in DOF order."
function equilibrium_vector(c_Cs::Float64 = 1000.0, C_total::Float64 = 1000.0)
    e = solve_buffer_equilibrium(c_Cs, C_total)
    return [e.H, e.OH, e.HCO3, e.CO3, e.CO2], e
end

"§2/§3.1 Cs⁺ recovered from electroneutrality (never a transported DOF)."
@inline cCs_from_electroneutrality(c::AbstractVector{<:Real}) =
    c[2] + c[3] + 2*c[4] - c[1] - C_fix * eps_I

"""
    sechenov_H_CO2(c) -> H_eff [mol/(m³·atm)]

Combined van 't Hoff (T) + Sechenov (salting-out) effective Henry constant
evaluated at the local ion composition (§7.4).
"""
@inline function sechenov_H_CO2(c::AbstractVector{<:Real})
    cCs = cCs_from_electroneutrality(c)
    expo = (h_CO2 + h_Cs)  * cCs   / 1000.0 +
           (h_CO2 + h_OH)  * c[2]  / 1000.0 +
           (h_CO2 + h_HCO3)* c[3]  / 1000.0 +
           (h_CO2 + h_CO3) * c[4]  / 1000.0
    return H_vanthoff * 10.0^(-expo)
end
