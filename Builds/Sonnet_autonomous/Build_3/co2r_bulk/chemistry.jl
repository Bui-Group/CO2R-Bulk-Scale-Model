# chemistry.jl — buffer_sources! function, §5 Buffer Chemistry
using Roots

"""
    buffer_sources(c) -> (S_H, S_OH, S_HCO3, S_CO3, S_CO2)

Net volumetric buffer reaction source for each species [mol/m3/s], §5.
`c = (c_H, c_OH, c_HCO3, c_CO3, c_CO2)` in mol/m3. Type-generic for
ForwardDiff compatibility (works for any <:Real element type).
"""
@inline function buffer_sources(c::AbstractVector{ElT}) where {ElT<:Real}
    c_H, c_OH, c_HCO3, c_CO3, c_CO2 = c[1], c[2], c[3], c[4], c[5]

    S_H = bk1 - bk1r*c_H*c_OH + bk2*c_CO2 - bk2r*c_H*c_HCO3 + bk3*c_HCO3 - bk3r*c_H*c_CO3
    S_OH = bk1 - bk1r*c_H*c_OH - bk4*c_CO2*c_OH + bk4r*c_HCO3 - bk5*c_HCO3*c_OH + bk5r*c_CO3
    S_HCO3 = bk2*c_CO2 - bk2r*c_H*c_HCO3 + bk4*c_CO2*c_OH - bk4r*c_HCO3 -
             bk3*c_HCO3 + bk3r*c_H*c_CO3 - bk5*c_HCO3*c_OH + bk5r*c_CO3
    S_CO3 = bk3*c_HCO3 - bk3r*c_H*c_CO3 + bk5*c_HCO3*c_OH - bk5r*c_CO3
    S_CO2 = -bk2*c_CO2 + bk2r*c_H*c_HCO3 - bk4*c_CO2*c_OH + bk4r*c_HCO3

    return (S_H, S_OH, S_HCO3, S_CO3, S_CO2)
end

"""
    solve_buffer_equilibrium(c_Cs_mol_m3=1000.0, C_total_mol_m3=1000.0)

Exact bulk buffer equilibrium via bisection on the charge balance, §9.6.
Returns a NamedTuple (H, OH, HCO3, CO3, CO2, Cs, pH).
"""
function solve_buffer_equilibrium(c_Cs_mol_m3::Float64=1000.0, C_total_mol_m3::Float64=1000.0)
    function charge_balance(log_cH)
        cH = 10.0^log_cH
        cOH = Kw / cH
        denom = 1.0 + K2/cH + K2*K3/cH^2
        cCO2 = C_total_mol_m3 / denom
        cHCO3 = K2 * cCO2 / cH
        cCO3 = K3 * cHCO3 / cH
        return cH + c_Cs_mol_m3 - cOH - cHCO3 - 2*cCO3
    end
    log_cH = find_zero(charge_balance, (-6.0, 0.0), Bisection())
    cH = 10.0^log_cH
    cOH = Kw / cH
    denom = 1.0 + K2/cH + K2*K3/cH^2
    cCO2 = C_total_mol_m3 / denom
    cHCO3 = K2 * cCO2 / cH
    cCO3 = K3 * cHCO3 / cH
    return (H=cH, OH=cOH, HCO3=cHCO3, CO3=cCO3, CO2=cCO2, Cs=c_Cs_mol_m3, pH=-log10(cH/1000.0))
end
