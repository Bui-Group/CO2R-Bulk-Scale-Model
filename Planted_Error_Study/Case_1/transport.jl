# transport.jl — Face fluxes (§9.1)
# diffusion_flux: pure Fickian (Stages 1–4).  np_flux: Nernst-Planck (Stage 5).

"Pure diffusion face flux [mol/m²/s]. dx is the distance between cell centers."
diffusion_flux(c_L::Real, c_R::Real, D_eff_::Real, dx::Real) =
    -D_eff_ * (c_R - c_L) / dx

"""
    np_flux(c_L, c_R, phi_L_left, phi_L_right, D_eff, z_i, dx_cc)

Nernst-Planck face flux (diffusion + migration), §9.1. Reduces to diffusion_flux
when ∇φ_L = 0. Used for all Stage 5 faces (interior + right boundary).
"""
function np_flux(c_L::Real, c_R::Real, phi_L_left::Real, phi_L_right::Real,
                 D_eff_::Real, z_i::Real, dx_cc::Real)
    c_avg = 0.5 * (c_L + c_R)
    diff  = -D_eff_ * (c_R - c_L) / dx_cc
    migr  = -(z_i * F_const / (R_gas * T)) * D_eff_ * c_avg *
            (phi_L_right - phi_L_left) / dx_cc
    return diff + migr
end
