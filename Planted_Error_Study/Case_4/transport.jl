# transport.jl — liquid-phase face fluxes (§9.1)
#
# Sign convention: +x-positive flux. dx_cc is the distance between the two cell
# centers (or center-to-boundary at BCs), NOT the cell width.

# Pure diffusion face flux [mol/m²/s] — Stages 1–4 (φ_L = 0).
@inline function diffusion_flux(c_L::Real, c_R::Real, D_eff_i::Real, dx_cc::Real)
    return -D_eff_i * (c_R - c_L) / dx_cc
end

# Nernst-Planck face flux [mol/m²/s] — diffusion + migration (Stage 5).
# For the right (electrolyte) boundary pass phi_L_right = 0 (gauge) and
# dx_cc = L_eff = D_eff/k_MT; this reduces to the Sherwood form when φ_L = 0.
@inline function np_flux(c_L::Real, c_R::Real,
                         phi_L_left::Real, phi_L_right::Real,
                         D_eff_i::Real, z_i::Real, dx_cc::Real)
    c_avg = 0.5 * (c_L + c_R)
    diff  = -D_eff_i * (c_R - c_L) / dx_cc
    migr  = -(z_i * F_const / (R_gas * T)) * D_eff_i * c_avg *
            (phi_L_right - phi_L_left) / dx_cc
    return diff + migr
end

nothing
