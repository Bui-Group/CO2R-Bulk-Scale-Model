# gas_transport.jl — Stefan-Maxwell mixture-average diffusion + Darcy flow, §6.
#
# Binary diffusivities are precomputed CONSTANTS at the reference p_total = 101325 Pa (§9.3.2
# explicitly allows `D_bin_raw` to stay Matrix{Float64} in the type-genericity checklist, which
# only makes sense if it is not re-evaluated at the local Dual-valued pressure DOF — see
# BUILD_DIALOGUE.md item 3). Species order throughout: (CO2, CO, H2, H2O).

"Fuller-Schettler-Giddings binary diffusivity [m^2/s] at T (K) and p_atm (atm). §6.3 — prefactor
1.00e-3, NOT 1.013e-2 (Perry's 9th ed.); M_i in g/mol here (NOT the kg/mol used elsewhere)."
function fsg_binary_D(i::Int, j::Int, p_atm::Float64)
    Mi = M_gas_gmol[i]; Mj = M_gas_gmol[j]
    vi = v_gas[i]; vj = v_gas[j]
    D_cm2_s = 1.00e-3 * T^1.75 * sqrt(1/Mi + 1/Mj) / (p_atm * (vi^(1/3) + vj^(1/3))^2)
    return D_cm2_s * 1e-4    # cm^2/s -> m^2/s
end

const D_bin_raw = [i == j ? 0.0 : fsg_binary_D(i, j, p_total/101325.0) for i in 1:4, j in 1:4]

const D_eff_GDL_mat = (eps_GDL^1.5) .* D_bin_raw
const D_eff_CL_mat  = (eps_G^1.5)   .* D_bin_raw
const D_eff_int_mat = [i == j ? 0.0 :
                        2*D_eff_GDL_mat[i,j]*D_eff_CL_mat[i,j] / (D_eff_GDL_mat[i,j]+D_eff_CL_mat[i,j])
                        for i in 1:4, j in 1:4]

const kappa_interface = 2 * kappa_GDL * kappa_CL / (kappa_GDL + kappa_CL)

omega_H2O_closure(w1::Real, w2::Real, w3::Real) = max(1.0 - w1 - w2 - w3, zero(w1))

"Mixture molar mass [kg/mol] from a 4-tuple of mass fractions (CO2,CO,H2,H2O)."
function Mn_from_omega(omega::NTuple{4,<:Real})
    return 1.0 / (omega[1]/M_gas[1] + omega[2]/M_gas[2] + omega[3]/M_gas[3] + omega[4]/M_gas[4])
end

omega_to_x(wj::Real, Mn::Real, Mj::Real) = wj * Mn / Mj

function omega4_to_x4(omega::NTuple{4,<:Real})
    Mn = Mn_from_omega(omega)
    return (omega_to_x(omega[1], Mn, M_gas[1]), omega_to_x(omega[2], Mn, M_gas[2]),
            omega_to_x(omega[3], Mn, M_gas[3]), omega_to_x(omega[4], Mn, M_gas[4]))
end

function mole_fraction_CO2(omega::NTuple{4,<:Real})
    Mn = Mn_from_omega(omega)
    return omega_to_x(omega[1], Mn, M_gas[1])
end

"Local CO2 partial pressure [atm] from mass fractions and the local total pressure [Pa]."
function pCO2_atm(omega::NTuple{4,<:Real}, p_Pa::Real)
    return mole_fraction_CO2(omega) * p_Pa / 101325.0
end

"Mixture-average diffusivity of species j (§6.1): (1-x_j) / sum_{k != j} (x_k / D_jk)."
function D_MA_species(j::Int, x::NTuple{4,<:Real}, Dmat::AbstractMatrix{Float64})
    s = zero(x[1])
    for k in 1:4
        k == j && continue
        s += x[k] / Dmat[j, k]
    end
    return (1.0 - x[j]) / s
end

"""
    sm_face_flux_full(omega_L, p_L, omega_R, p_R, Dmat_face, kappa_face, dx, convective) -> (J, rho_u)

Full Stefan-Maxwell species flux (diffusion + mixture-molar-mass correction + optional Darcy
convection, §6.1-6.2) at one face, plus the Darcy mass flux rho*u_G at the same face. `omega_*`
are 4-tuples (CO2, CO, H2, H2O-closure). `convective=false` pins u_G=0 (Stage 3); `true` uses the
real pressure gradient (Stage 4+, §11).
"""
function sm_face_flux_full(omega_L::NTuple{4,<:Real}, p_L::Real,
                            omega_R::NTuple{4,<:Real}, p_R::Real,
                            Dmat_face::AbstractMatrix{Float64}, kappa_face::Real, dx::Real,
                            convective::Bool)
    Mn_L = Mn_from_omega(omega_L)
    Mn_R = Mn_from_omega(omega_R)
    omega_face = ntuple(k -> 0.5 * (omega_L[k] + omega_R[k]), 4)
    Mn_face = Mn_from_omega(omega_face)
    x_face = omega4_to_x4(omega_face)
    p_face = 0.5 * (p_L + p_R)
    rho_face = p_face * Mn_face / (R_gas * T)

    u_G = convective ? -(kappa_face / mu_gas) * (p_R - p_L) / dx : zero(p_face)
    rho_u = rho_face * u_G

    J = ntuple(4) do j
        DMA_j = D_MA_species(j, x_face, Dmat_face)
        Jdiff = -rho_face * DMA_j * (omega_R[j] - omega_L[j]) / dx -
                rho_face * DMA_j * omega_face[j] * (Mn_R - Mn_L) / (Mn_face * dx)
        Jconv = rho_face * omega_face[j] * u_G
        Jdiff + Jconv
    end
    return J, rho_u
end

# ── §7.4 / §8.10 Henry's law with Sechenov salting-out ───────────────────

"Van't Hoff-only Henry's constant [mol/(m^3 atm)] — Stages 1-2, Sechenov not yet enabled (§7.1)."
H_eff_vanthoff() = H_CO2_25C * exp(dH_sol * (1/T - 1/298.15))

"""
    sechenov_H_CO2(c5)

Combined van't Hoff + Sechenov effective Henry's constant [mol/(m^3 atm)] using the LOCAL ion
concentrations at a CL cell (§7.4). `c5` = (cH, cOH, cHCO3, cCO3, cCO2) in mol/m^3.
"""
function sechenov_H_CO2(c5::AbstractVector{<:Real})
    cH, cOH, cHCO3, cCO3 = c5[1], c5[2], c5[3], c5[4]
    cCs = c_Cs_from_electroneutrality(cH, cOH, cHCO3, cCO3)
    exponent = -((h_CO2 + h_Cs) * cCs + (h_CO2 + h_OH) * cOH +
                 (h_CO2 + h_HCO3) * cHCO3 + (h_CO2 + h_CO3) * cCO3) / 1000.0
    return H_eff_vanthoff() * 10.0^exponent
end
