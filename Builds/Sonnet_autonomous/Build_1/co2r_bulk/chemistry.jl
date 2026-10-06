##############################################################################
# chemistry.jl — buffer reaction sources (§5) and equilibrium solve (§9.6)
##############################################################################
using Roots

"""
    buffer_sources(c)

c = [c_H, c_OH, c_HCO3, c_CO3, c_CO2] (mol/m^3, SPECIES order).
Returns net volumetric source [mol/m^3/s] in the same order.
Type-generic: works for Float64 or ForwardDiff.Dual via duck typing.
"""
function buffer_sources(c::AbstractVector)
    cH, cOH, cHCO3, cCO3, cCO2 = c[1], c[2], c[3], c[4], c[5]

    S_H    = k1 - k1r*cH*cOH + k2*cCO2 - k2r*cH*cHCO3 + k3*cHCO3 - k3r*cH*cCO3
    S_OH   = k1 - k1r*cH*cOH - k4*cCO2*cOH + k4r*cHCO3 - k5*cHCO3*cOH + k5r*cCO3
    S_HCO3 = k2*cCO2 - k2r*cH*cHCO3 + k4*cCO2*cOH - k4r*cHCO3 - k3*cHCO3 + k3r*cH*cCO3 - k5*cHCO3*cOH + k5r*cCO3
    S_CO3  = k3*cHCO3 - k3r*cH*cCO3 + k5*cHCO3*cOH - k5r*cCO3
    S_CO2  = -k2*cCO2 + k2r*cH*cHCO3 - k4*cCO2*cOH + k4r*cHCO3

    return [S_H, S_OH, S_HCO3, S_CO3, S_CO2]
end

"""
    solve_buffer_equilibrium(c_Cs_mol_m3=1000.0, C_total_mol_m3=1000.0)

Exact bulk buffer equilibrium via bisection on the charge balance (§9.6).
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

"c_eq as a 5-vector in SPECIES order [H, OH, HCO3, CO3, CO2]"
function c_eq_vector(c_eq_nt)
    return [c_eq_nt.H, c_eq_nt.OH, c_eq_nt.HCO3, c_eq_nt.CO3, c_eq_nt.CO2]
end

"Cs+ back-derived from electroneutrality (§2, §3.1): c_Cs = c_OH+c_HCO3+2c_CO3-c_H-C_fix*eps_I"
@inline function c_Cs_of(c::AbstractVector)
    return c[2] + c[3] + 2*c[4] - c[1] - C_fix*eps_I
end
