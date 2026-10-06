# transport.jl — liquid face fluxes, §9.1, §3.1

"""Pure diffusion face flux (Stages 1-4, no migration)."""
@inline function diffusion_flux(c_L::Real, c_R::Real, D_eff_i::Real, dx_cc::Real)
    return -D_eff_i * (c_R - c_L) / dx_cc
end

"""
    np_flux(c_L, c_R, phi_L_left, phi_L_right, D_eff_i, z_i, dx_cc)

Full Nernst-Planck face flux (diffusion + migration), §9.1 / §3.1. Used
for ALL Stage-5 liquid faces, interior and the right (electrolyte) boundary.
"""
@inline function np_flux(c_L::Real, c_R::Real, phi_L_left::Real, phi_L_right::Real,
                          D_eff_i::Real, z_i::Real, dx_cc::Real)
    c_avg = 0.5 * (c_L + c_R)
    diff = -D_eff_i * (c_R - c_L) / dx_cc
    migr = -(z_i * F_const / (R_gas * T)) * D_eff_i * c_avg * (phi_L_right - phi_L_left) / dx_cc
    return diff + migr
end
