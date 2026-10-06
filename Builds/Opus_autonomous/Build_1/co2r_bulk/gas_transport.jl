# gas_transport.jl — Stefan-Maxwell (mixture-average) + Darcy (§6)
# Gas species order: 1 CO₂, 2 CO, 3 H₂, 4 H₂O (H₂O is closure-only)

"""Mixture molar mass [kg/mol] from mass fractions."""
@inline function Mn_from_omega(w1::Real, w2::Real, w3::Real, w4::Real)
    s = w1/M_gas[1] + w2/M_gas[2] + w3/M_gas[3] + w4/M_gas[4]
    return 1.0 / s
end

"""H₂O mass-fraction closure (§6.1)."""
@inline omega_H2O_closure(w1::Real, w2::Real, w3::Real) =
    max(1.0 - w1 - w2 - w3, zero(w1))

"""Mole fractions from mass fractions."""
@inline function omega_to_x(w1::Real, w2::Real, w3::Real, w4::Real)
    Mn = Mn_from_omega(w1, w2, w3, w4)
    return (w1*Mn/M_gas[1], w2*Mn/M_gas[2], w3*Mn/M_gas[3], w4*Mn/M_gas[4])
end

"""Mixture-average diffusivity of species j (§6.1)."""
@inline function D_MA_species(x1::Real, x2::Real, x3::Real, x4::Real,
                              Dm::Matrix{Float64}, j::Int)
    xs = (x1, x2, x3, x4)
    den = zero(x1)
    @inbounds for k in 1:4
        k == j && continue
        den += xs[k] / Dm[j, k]
    end
    den = max(den, 1e-30)
    return (1.0 - xs[j]) / den
end
