# =============================================================================
# transport.jl — liquid-phase face fluxes (BaseCase.md §9.1)
#   diffusion_flux : pure diffusion (Stages 1–4)
#   np_flux        : Nernst–Planck diffusion + migration (Stage 5)
# `dx` / `dx_cc` is the distance between the two cell *centers* (or face->center
# at boundaries), NOT the cell width. Type-generic for ForwardDiff.
# =============================================================================

# Pure-diffusion face flux [mol/m²/s], +x-positive convention.
@inline diffusion_flux(c_L::Real, c_R::Real, D_eff_i::Real, dx::Real) =
    -D_eff_i * (c_R - c_L) / dx

# Nernst–Planck face flux = diffusion + migration (§3.1, §9.1).
@inline function np_flux(c_L::Real, c_R::Real,
                         phi_L_left::Real, phi_L_right::Real,
                         D_eff_i::Real, z_i::Real, dx_cc::Real)
    c_avg = 0.5 * (c_L + c_R)
    diff  = -D_eff_i * (c_R - c_L) / dx_cc
    migr  = -(z_i * F_const / (R_gas * T)) * D_eff_i * c_avg *
            (phi_L_right - phi_L_left) / dx_cc
    return diff + migr
end
