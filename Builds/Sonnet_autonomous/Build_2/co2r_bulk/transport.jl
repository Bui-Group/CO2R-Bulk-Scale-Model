# transport.jl — liquid face fluxes: diffusion_flux (Stages 1-4), np_flux (Stage 5), §9.1.

"Pure diffusive face flux. `dx` is the distance between the two cell CENTERS (or the ghost-to-
center distance at a boundary), not a cell width."
function diffusion_flux(c_L::Real, c_R::Real, D_eff::Real, dx::Real)
    return -D_eff * (c_R - c_L) / dx
end

"""
    np_flux(c_L, c_R, phi_L_left, phi_L_right, D_eff, z_i, dx_cc)

Nernst-Planck face flux: diffusion + migration (§9.1, used for ALL Stage-5 liquid faces
including the right boundary with `dx_cc = D_eff/k_MT`, `phi_L_right = 0`).
"""
function np_flux(c_L::Real, c_R::Real, phi_L_left::Real, phi_L_right::Real,
                  D_eff::Real, z_i::Real, dx_cc::Real)
    c_avg = 0.5 * (c_L + c_R)
    diff  = -D_eff * (c_R - c_L) / dx_cc
    migr  = -(z_i * F_const / (R_gas * T)) * D_eff * c_avg *
            (phi_L_right - phi_L_left) / dx_cc
    return diff + migr
end
