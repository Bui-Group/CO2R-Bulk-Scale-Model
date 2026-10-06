# =============================================================================
# transport.jl — liquid face fluxes (§9.1)
#   diffusion_flux : Stages 1-4 (phi_L = 0, no migration)
#   np_flux        : Stage 5 full Nernst-Planck (diffusion + migration)
# `dx` / `dx_cc` is always the distance between the two states being
# differenced, NOT the cell width.
# =============================================================================

"Pure diffusive face flux [mol/m^2/s], +x positive (§9.1)."
@inline diffusion_flux(c_L::Real, c_R::Real, D::Real, dx::Real) =
    -D * (c_R - c_L) / dx

"""
    np_flux(c_L, c_R, phi_L_left, phi_L_right, D_eff, z_i, dx_cc)

Nernst-Planck face flux [mol/m^2/s] — diffusion + migration (§9.1).
Used for ALL Stage 5 liquid face fluxes: interior faces AND the right
boundary.  At the right boundary pass `phi_L_right = 0` (the gauge point) and
`dx_cc = L_eff = D_eff[i] / k_MT[i]`, which reduces to the Sherwood formula
when the potential gradient vanishes but correctly carries migration current
when it does not (§7.2, §11 Stage 5 gotcha #12).
"""
@inline function np_flux(c_L::Real, c_R::Real,
                         phi_L_left::Real, phi_L_right::Real,
                         D_eff_i::Real, z_i::Real, dx_cc::Real)
    c_avg = 0.5 * (c_L + c_R)
    diff  = -D_eff_i * (c_R - c_L) / dx_cc
    migr  = -(z_i * F_const / (R_gas * T)) * D_eff_i * c_avg *
            (phi_L_right - phi_L_left) / dx_cc
    return diff + migr
end
