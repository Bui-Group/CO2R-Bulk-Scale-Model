# =============================================================================
# chemistry.jl — buffer chemistry (§5) and the exact equilibrium guess (§9.6)
# =============================================================================
using Roots      # §9.6: MUST be at top level; `using` inside a function body
                 # is a Julia syntax error.

"""
    buffer_sources(c) -> NTuple{5}

Net volumetric buffer source for each of the 5 liquid species [mol/m^3/s],
per §5.  `c` is a 5-vector in mol/m^3 ordered (H+, OH-, HCO3-, CO3^2-, CO2aq).
Fully type-generic so ForwardDiff.Dual values flow through (§9.3.2).
"""
@inline function buffer_sources(c::AbstractVector{<:Real})
    cH, cOH, cHCO3, cCO3, cCO2 = c[1], c[2], c[3], c[4], c[5]

    r1 = k1f              - k1r * cH * cOH          # H2O <-> H+ + OH-
    r2 = k2f * cCO2       - k2r * cH * cHCO3        # CO2 + H2O <-> H+ + HCO3-
    r3 = k3f * cHCO3      - k3r * cH * cCO3         # HCO3- <-> H+ + CO3^2-
    r4 = k4f * cCO2 * cOH - k4r * cHCO3             # CO2 + OH- <-> HCO3-
    r5 = k5f * cHCO3*cOH  - k5r * cCO3              # HCO3- + OH- <-> CO3^2- + H2O

    S_H    =  r1 + r2 + r3
    S_OH   =  r1 - r4 - r5
    S_HCO3 =  r2 - r3 + r4 - r5
    S_CO3  =  r3 + r5
    S_CO2  = -r2 - r4
    return (S_H, S_OH, S_HCO3, S_CO3, S_CO2)
end

"""
    solve_buffer_equilibrium(c_Cs_mol_m3 = 1000.0, C_total_mol_m3 = 1000.0)

Exact charge balance solve for the bulk 1 M CsHCO3 electrolyte at 50 C (§9.6).
This is the single most important initialisation step: an approximate guess
leaves buffer sources O(1e3-1e5) mol/m^3/s and Newton will not converge.
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

"Equilibrium composition as the 5-vector used by the residuals."
function equilibrium_vector(c_Cs = 1000.0, C_tot = 1000.0)
    e = solve_buffer_equilibrium(c_Cs, C_tot)
    return [e.H, e.OH, e.HCO3, e.CO3, e.CO2]
end

# ── §2 / §3.1 Cs+ from electroneutrality (not a transported DOF) ─────────────
@inline cCs_from_electroneutrality(c::AbstractVector{<:Real}) =
    c[2] + c[3] + 2*c[4] - c[1] - C_fix * eps_I

# ── §7.4 Combined van 't Hoff + Sechenov Henry constant ──────────────────────
"""
    sechenov_H_CO2(c) -> H_eff [mol/(m^3 atm)]

Combined temperature- and ion-corrected Henry constant (§7.4).  The Cs+ used
is the electroneutrality value (~570 mol/m^3 at the bulk equilibrium, not the
1000 mol/m^3 of the free electrolyte, because the ionomer fixed charge
displaces `C_fix * eps_I = 430` mol/m^3 of free Cs+).
"""
@inline function sechenov_H_CO2(c::AbstractVector{<:Real})
    cCs = cCs_from_electroneutrality(c)
    expo = ((h_CO2 + h_Cs)   * cCs   +
            (h_CO2 + h_OH)   * c[2]  +
            (h_CO2 + h_HCO3) * c[3]  +
            (h_CO2 + h_CO3)  * c[4]) / 1000.0
    return H_eff_T * 10.0^(-expo)
end
