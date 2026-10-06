##############################################################################
# transport.jl — liquid-phase face fluxes (§9.1): pure diffusion + NP (Stage 5)
##############################################################################

"Pure diffusion face flux (Stages 1-4). `dx` = distance between the two cell centers."
@inline function diffusion_flux(c_L, c_R, D_eff, dx)
    return -D_eff * (c_R - c_L) / dx
end

"""
    np_flux(c_L, c_R, phi_L_left, phi_L_right, D_eff, z_i, dx_cc)

Nernst-Planck face flux (diffusion + migration), Stage 5, §9.1.
"""
@inline function np_flux(c_L, c_R, phi_L_left, phi_L_right, D_eff, z_i, dx_cc)
    c_avg = 0.5 * (c_L + c_R)
    diff  = -D_eff * (c_R - c_L) / dx_cc
    migr  = -(z_i * F_const / (R_gas * T)) * D_eff * c_avg * (phi_L_right - phi_L_left) / dx_cc
    return diff + migr
end
