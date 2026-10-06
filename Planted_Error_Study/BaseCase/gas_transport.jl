# =============================================================================
# gas_transport.jl — Stefan-Maxwell mixture-average gas transport (BaseCase.md §6)
#   + FSG binary diffusivities (§6.3), Sechenov Henry (§7.4), gas DOF layout (§2).
# Gas species order: 1=CO₂, 2=CO, 3=H₂, 4=H₂O (closure). DOFs per gas cell:
#   (ω_CO₂, ω_CO, ω_H₂, p). H₂O mass fraction is the closure ω₄ = 1−ω₁−ω₂−ω₃.
# Type-generic for ForwardDiff.
# =============================================================================

# ---------------------------------------------------------------------------
# Gas DOF indexing (Stages 3–4 layout, §2): u = [liquid 5N_CL ; GDL 4N_GDL ; CL 4N_CL]
# ---------------------------------------------------------------------------
const _GAS0 = 5 * N_CL
@inline gas_gdl_idx(ig::Int, j::Int) = _GAS0 + 4*(ig - 1) + j           # j=1,2,3 (ω), 4 (p)
@inline gas_cl_idx(ix::Int, j::Int)  = _GAS0 + 4*N_GDL + 4*(ix - 1) + j
const NDOF_GAS = 5*N_CL + 4*(N_GDL + N_CL)

# ---------------------------------------------------------------------------
# Stage 5 potential DOFs (§11 Stage 5 layout): appended after the gas block.
#   u = [ liquid 5N_CL ; gas 4(N_GDL+N_CL) ; φ_s_GDL(N_GDL) ; φ_s_CL(N_CL) ; φ_L_CL(N_CL) ]
# ---------------------------------------------------------------------------
const _PHIS0 = NDOF_GAS
@inline phi_s_gdl_idx(ig::Int) = _PHIS0 + ig
@inline phi_s_cl_idx(ix::Int)  = _PHIS0 + N_GDL + ix
@inline phi_L_idx(ix::Int)     = _PHIS0 + N_GDL + N_CL + ix
const NDOF_STAGE5 = NDOF_GAS + (N_GDL + N_CL) + N_CL          # 1480 at 80/120

# ---------------------------------------------------------------------------
# Unpacking helpers (return element type of u for ForwardDiff)
# ---------------------------------------------------------------------------
@inline function gdl_omega(u, ig::Int)
    o1 = u[gas_gdl_idx(ig,1)]; o2 = u[gas_gdl_idx(ig,2)]; o3 = u[gas_gdl_idx(ig,3)]
    return (o1, o2, o3, max(1 - o1 - o2 - o3, zero(eltype(u))))
end
@inline gdl_p(u, ig::Int) = u[gas_gdl_idx(ig,4)]
@inline function cl_omega(u, ix::Int)
    o1 = u[gas_cl_idx(ix,1)]; o2 = u[gas_cl_idx(ix,2)]; o3 = u[gas_cl_idx(ix,3)]
    return (o1, o2, o3, max(1 - o1 - o2 - o3, zero(eltype(u))))
end
@inline cl_p(u, ix::Int) = u[gas_cl_idx(ix,4)]

# Mixture molar mass [kg/mol] and mole fractions from mass fractions.
@inline Mn_from_omega(om) = 1 / (om[1]/M_gas[1] + om[2]/M_gas[2] + om[3]/M_gas[3] + om[4]/M_gas[4])
@inline omega_to_x(om, Mn) =
    (om[1]/M_gas[1]*Mn, om[2]/M_gas[2]*Mn, om[3]/M_gas[3]*Mn, om[4]/M_gas[4]*Mn)

# ---------------------------------------------------------------------------
# §6.3 Fuller-Schettler-Giddings binary diffusivities at T, p_total [m²/s].
# Prefactor 1.00e-3 (Perry's 9th ed.), NOT 1.013e-2. M in g/mol, v in cm³/mol.
# ---------------------------------------------------------------------------
function _fsg_dbin()
    D = zeros(4, 4)
    patm = p_total / 101325.0                       # = 1.0
    for i in 1:4, j in 1:4
        i == j && continue
        Mi = M_gas[i]*1000; Mj = M_gas[j]*1000       # g/mol
        Dij_cm2 = 1.00e-3 * T^1.75 * sqrt(1/Mi + 1/Mj) /
                  (patm * (v_diff[i]^(1/3) + v_diff[j]^(1/3))^2)
        D[i,j] = Dij_cm2 * 1e-4                       # m²/s
    end
    for i in 1:4; D[i,i] = 1.0; end                   # diag unused (skipped in D^MA); avoid 0/0
    return D
end
const D_bin_raw = _fsg_dbin()
const D_bin_GDL = eps_GDL^1.5 .* D_bin_raw            # Bruggeman, GDL porosity
const D_bin_CL  = eps_G^1.5   .* D_bin_raw            # Bruggeman, CL gas void
const D_bin_IF  = 2 .* D_bin_GDL .* D_bin_CL ./ (D_bin_GDL .+ D_bin_CL)   # harmonic mean at interface

# ---------------------------------------------------------------------------
# §6.1 Stefan-Maxwell mixture-average face flux [kg/m²/s] for species 1,2,3.
#   J_j = −ρ D_j^MA ∇ω_j − ρ D_j^MA ω_j (∇M_n/M_n) + ρ ω_j u_G
# `Dbin` is the (Bruggeman-corrected) 4×4 binary diffusivity at the face.
# ---------------------------------------------------------------------------
@inline function sm_face_flux(omL, omR, pL::Real, pR::Real, Dbin, dx_cc::Real, uG::Real)
    MnL = Mn_from_omega(omL); MnR = Mn_from_omega(omR)
    omf = (0.5*(omL[1]+omR[1]), 0.5*(omL[2]+omR[2]), 0.5*(omL[3]+omR[3]), 0.5*(omL[4]+omR[4]))
    Mnf = Mn_from_omega(omf)
    pf  = 0.5*(pL + pR)
    rhof = pf * Mnf / (R_gas * T)
    xf  = omega_to_x(omf, Mnf)
    dMn_over_Mn = (MnR - MnL) / (Mnf * dx_cc)
    J = ntuple(3) do j
        s = zero(eltype(omf))
        @inbounds for k in 1:4
            k == j && continue
            s += xf[k] / Dbin[j,k]
        end
        s = max(s, 1e-300)
        DMA = (1 - xf[j]) / s
        dom = (omR[j] - omL[j]) / dx_cc
        -rhof*DMA*dom - rhof*DMA*omf[j]*dMn_over_Mn + rhof*omf[j]*uG
    end
    return J
end

# ---------------------------------------------------------------------------
# §7.4 Combined van 't Hoff + Sechenov effective Henry constant [mol/(m³·atm)].
# Sum over dominant ions (Cs⁺,OH⁻,HCO₃⁻,CO₃²⁻); Cs⁺ from electroneutrality.
# ---------------------------------------------------------------------------
@inline function H_eff_sechenov(cH::Real, cOH::Real, cHCO3::Real, cCO3::Real)
    cCs = cOH + cHCO3 + 2*cCO3 - cH - C_fix_eff
    Hvh = H_CO2_0 * exp(dH_sol * (1/T - 1/298.15))
    expo = (h_CO2 + h_Cs)*cCs/1000 + (h_CO2 + h_OH)*cOH/1000 +
           (h_CO2 + h_HCO3)*cHCO3/1000 + (h_CO2 + h_CO3)*cCO3/1000
    return Hvh * 10.0^(-expo)
end

# local CL gas-phase CO₂ partial pressure [atm] at CL cell ix
@inline function pCO2_cl(u, ix::Int)
    om = cl_omega(u, ix); Mn = Mn_from_omega(om); x = omega_to_x(om, Mn)
    return x[1] * cl_p(u, ix) / 101325.0
end
