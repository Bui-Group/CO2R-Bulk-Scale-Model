# ============================================================================
# transport.jl — Liquid face fluxes (§9.1)
#   diffusion_flux : pure diffusion (Stages 1–4)
#   np_flux        : Nernst-Planck diffusion + migration (Stage 5)
# `dx`/`dx_cc` is the distance between cell centers (not cell width).
# ============================================================================

# Pure diffusion face flux [mol/m²/s] (Stages 1–4).
@inline function diffusion_flux(c_L::Real, c_R::Real, D_eff_::Real, dx::Real)
    return -D_eff_ * (c_R - c_L) / dx
end

# Nernst-Planck face flux (diffusion + migration) [mol/m²/s] (Stage 5).
@inline function np_flux(c_L::Real, c_R::Real,
                         phi_L_left::Real, phi_L_right::Real,
                         D_eff_::Real, z_i::Real, dx_cc::Real)
    c_avg = 0.5 * (c_L + c_R)
    diff  = -D_eff_ * (c_R - c_L) / dx_cc
    migr  = -(z_i * F_const / (R_gas * T)) * D_eff_ * c_avg *
            (phi_L_right - phi_L_left) / dx_cc
    return diff + migr
end
