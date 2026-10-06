##############################################################################
# gas_transport.jl — Stefan-Maxwell mixture-average diffusion + Darcy (§6)
##############################################################################

@inline omega4(w1, w2, w3) = max(1 - w1 - w2 - w3, zero(w1))

@inline function Mn_from_omega(w1, w2, w3)
    w4 = omega4(w1, w2, w3)
    return 1.0 / (w1/M_gas[1] + w2/M_gas[2] + w3/M_gas[3] + w4/M_gas[4])
end

@inline function omega_to_x(w1, w2, w3, Mn)
    w4 = omega4(w1, w2, w3)
    return (w1*Mn/M_gas[1], w2*Mn/M_gas[2], w3*Mn/M_gas[3], w4*Mn/M_gas[4])
end

"mole fraction of CO2 (species 1) from CL cell gas mass fractions"
@inline function mole_fraction_CO2(w1, w2, w3)
    Mn = Mn_from_omega(w1, w2, w3)
    return w1 * Mn / M_gas[1]
end

# ── §6.3 Fuller-Schettler-Giddings binary diffusivities ────────────────────
# NOTE: computed once at the reference pressure p_total (1 atm). Pressure in
# this model varies by <1.5% of 1 atm (§11 Stage-4 check), so D_bin_raw is
# kept as a plain Float64 constant matrix rather than a function of the local
# (AD-dual) pressure DOF -- exactly the simplification the guide anticipates
# in §9.3.2 item 5 ("Constant matrices like D_bin_raw can stay Matrix{Float64}").
function fsg_binary_D(Mi_kg, Mj_kg, vi, vj, Tval, p_atm)
    Mi_g = Mi_kg * 1000.0
    Mj_g = Mj_kg * 1000.0
    Dij_cm2s = 1.00e-3 * Tval^1.75 * sqrt(1/Mi_g + 1/Mj_g) / (p_atm * (vi^(1/3) + vj^(1/3))^2)
    return Dij_cm2s * 1e-4     # -> m^2/s
end

const D_bin_raw = let Dm = zeros(4,4)
    for i in 1:4, j in 1:4
        if i != j
            Dm[i,j] = fsg_binary_D(M_gas[i], M_gas[j], v_diff[i], v_diff[j], T, 1.0)
        end
    end
    Dm
end

# ── §8.7 Bruggeman corrections + GDL/CL interface harmonic mean ────────────
const D_eff_GDL_mat = eps_GDL^1.5 .* D_bin_raw
const D_eff_CL_mat  = eps_G^1.5   .* D_bin_raw
const D_eff_int_mat = let Dm = zeros(4,4)
    for i in 1:4, j in 1:4
        if i != j
            Dm[i,j] = 2*D_eff_GDL_mat[i,j]*D_eff_CL_mat[i,j] / (D_eff_GDL_mat[i,j] + D_eff_CL_mat[i,j])
        end
    end
    Dm
end
const kappa_int = 2*kappa_GDL*kappa_CL / (kappa_GDL + kappa_CL)   # §11 Stage 4

"Mixture-average diffusivity for species j (§6.1), evaluated at face-averaged mole fractions."
@inline function D_MA_species(x1, x2, x3, x4, Dmat::AbstractMatrix{Float64}, j::Int)
    xs = (x1, x2, x3, x4)
    xj = xs[j]
    s = zero(xj)
    for k in 1:4
        if k != j
            s += xs[k] / Dmat[j,k]
        end
    end
    denom = max(s, 1e-30)
    return (1 - xj) / denom
end

"""
    gas_face_flux(w1L,w2L,w3L,pL, w1R,w2R,w3R,pR, Dmat, kappa_face, dx_cc, with_convection)

Returns (J1, J2, J3, rho_u) -- species mass fluxes [kg/m^2/s] for CO2, CO, H2
and the Darcy mass flux rho*u_G [kg/m^2/s], §6.1/§6.2. `with_convection=false`
forces u_G = 0 (Stage 3); `true` activates Darcy convection (Stage 4+).
"""
function gas_face_flux(w1L, w2L, w3L, pL, w1R, w2R, w3R, pR,
                        Dmat::AbstractMatrix{Float64}, kappa_face::Float64,
                        dx_cc::Float64, with_convection::Bool)
    MnL = Mn_from_omega(w1L, w2L, w3L)
    MnR = Mn_from_omega(w1R, w2R, w3R)
    x1L, x2L, x3L, x4L = omega_to_x(w1L, w2L, w3L, MnL)
    x1R, x2R, x3R, x4R = omega_to_x(w1R, w2R, w3R, MnR)

    rhoL = pL * MnL / (R_gas * T)
    rhoR = pR * MnR / (R_gas * T)
    rho_avg = 0.5*(rhoL + rhoR)
    Mn_avg  = 0.5*(MnL + MnR)
    dMn_dx  = (MnR - MnL) / dx_cc

    x1a = 0.5*(x1L+x1R); x2a = 0.5*(x2L+x2R); x3a = 0.5*(x3L+x3R); x4a = 0.5*(x4L+x4R)
    w1a = 0.5*(w1L+w1R); w2a = 0.5*(w2L+w2R); w3a = 0.5*(w3L+w3R)

    D1 = D_MA_species(x1a, x2a, x3a, x4a, Dmat, 1)
    D2 = D_MA_species(x1a, x2a, x3a, x4a, Dmat, 2)
    D3 = D_MA_species(x1a, x2a, x3a, x4a, Dmat, 3)

    dw1 = (w1R-w1L)/dx_cc; dw2 = (w2R-w2L)/dx_cc; dw3 = (w3R-w3L)/dx_cc

    u_G = with_convection ? -(kappa_face/mu_gas)*(pR-pL)/dx_cc : zero(pR - pL)

    J1 = -rho_avg*D1*dw1 - rho_avg*D1*w1a*(dMn_dx/Mn_avg) + rho_avg*w1a*u_G
    J2 = -rho_avg*D2*dw2 - rho_avg*D2*w2a*(dMn_dx/Mn_avg) + rho_avg*w2a*u_G
    J3 = -rho_avg*D3*dw3 - rho_avg*D3*w3a*(dMn_dx/Mn_avg) + rho_avg*w3a*u_G
    rho_u = rho_avg * u_G

    return (J1, J2, J3, rho_u)
end

# ── §7.4 Combined van't Hoff + Sechenov Henry's constant ───────────────────
"""
    sechenov_H_CO2(c)

c = [c_H, c_OH, c_HCO3, c_CO3, c_CO2] (SPECIES order, mol/m^3) at one cell.
Returns H_eff(T, c_ions) [mol/(m^3 atm)], §7.4.
"""
function sechenov_H_CO2(c::AbstractVector)
    cCs   = c_Cs_of(c)
    cOH   = c[2]; cHCO3 = c[3]; cCO3 = c[4]
    expo = -(h_CO2+h_Cs)*cCs/1000 - (h_CO2+h_OH)*cOH/1000 -
            (h_CO2+h_HCO3)*cHCO3/1000 - (h_CO2+h_CO3)*cCO3/1000
    return H_eff_vantHoff(T) * 10.0^expo
end
