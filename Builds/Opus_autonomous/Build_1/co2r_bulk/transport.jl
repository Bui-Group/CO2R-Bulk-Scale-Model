# transport.jl — liquid face fluxes (§9.1)

"""Pure diffusive face flux [mol/m²/s]; `dx_cc` = centre-to-centre distance."""
@inline diffusion_flux(c_L::Real, c_R::Real, D::Real, dx_cc::Real) =
    -D * (c_R - c_L) / dx_cc

"""Nernst-Planck face flux (diffusion + migration) — §9.1."""
@inline function np_flux(c_L::Real, c_R::Real,
                         phi_L_left::Real, phi_L_right::Real,
                         D::Real, z_i::Real, dx_cc::Real)
    c_avg = 0.5 * (c_L + c_R)
    diff  = -D * (c_R - c_L) / dx_cc
    migr  = -(z_i * F_const / (R_gas * T)) * D * c_avg *
            (phi_L_right - phi_L_left) / dx_cc
    return diff + migr
end
