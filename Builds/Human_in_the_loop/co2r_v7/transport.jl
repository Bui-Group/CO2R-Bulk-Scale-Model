# =====================================================================
#  transport.jl — liquid face fluxes (§9.1)
# =====================================================================

"Pure diffusive face flux [mol/m²/s]; `dx_cc` = distance between cell centres."
@inline diffusion_flux(c_L::Real, c_R::Real, D_eff_i::Real, dx_cc::Real) =
    -D_eff_i * (c_R - c_L) / dx_cc

"""
    np_flux(c_L, c_R, phi_L_left, phi_L_right, D_eff, z_i, dx_cc)

Nernst-Planck face flux — diffusion + migration (§9.1).  Used for ALL Stage 5
face fluxes including the right boundary (`phi_L_right = 0`,
`dx_cc = L_eff = D_eff/k_MT`).
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
