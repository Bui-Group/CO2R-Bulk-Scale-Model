# gas_transport.jl — Stefan-Maxwell mixture-average gas transport (§6) + Sechenov
# Henry (§7.4) + gas DOF indexing. Stage 3 uses u_G = 0 (no Darcy convection);
# Darcy is finalized in Stage 4. Gas species order: 1=CO₂, 2=CO, 3=H₂, 4=H₂O.
# Gas DOFs per cell: (ω₁, ω₂, ω₃, p) — mass fractions (raw, not log) + pressure [Pa].

# ── §8.7 gas parameters ───────────────────────────────────────────────────
const M_gas      = [44.01e-3, 28.01e-3, 2.016e-3, 18.015e-3]   # kg/mol
const M_gas_gmol = [44.01, 28.01, 2.016, 18.015]               # g/mol (for FSG)
const v_diff     = [26.9, 18.9, 7.07, 12.7]                    # FSG diffusion volumes
const eps_GDL    = 0.52
const bf_GDL     = eps_GDL^1.5          # Bruggeman factor, GDL gas
const bf_CL      = eps_G^1.5            # Bruggeman factor, CL gas (eps_G from params)
const p_total    = 101325.0            # Pa
const kappa_GDL  = 8.4e-13             # m² (used Stage 4)
const kappa_CL   = 8.0e-16 * (1 - S_pore)^3   # m² (used Stage 4)
const mu_gas     = 1.59e-5             # Pa·s (used Stage 4)

# ── §6.3 Fuller-Schettler-Giddings binary diffusivities [m²/s] (raw) ──────
function _fsg_Dbin()
    D = zeros(4, 4)
    patm = 1.0
    for i in 1:4, j in 1:4
        i == j && continue
        Dij_cm2s = 1.00e-3 * T^1.75 * sqrt(1/M_gas_gmol[i] + 1/M_gas_gmol[j]) /
                   (patm * (v_diff[i]^(1/3) + v_diff[j]^(1/3))^2)
        D[i, j] = Dij_cm2s * 1e-4       # cm²/s → m²/s
    end
    return D
end
const D_bin = _fsg_Dbin()

# ── §7.3 inlet composition (mole → mass fractions) ────────────────────────
const x_inlet = [0.88, 0.0, 0.0, 0.12]     # MOLE fractions (CO₂, CO, H₂, H₂O)
mole_to_mass(x) = (Mavg = sum(x[j]*M_gas_gmol[j] for j in 1:4);
                   [x[j]*M_gas_gmol[j]/Mavg for j in 1:4])
const omega_inlet = mole_to_mass(x_inlet)  # ≈ [0.947, 0, 0, 0.053]

# ── §7.4 gas-liquid phase transfer parameters ─────────────────────────────
const R_pore = 25e-9
const a_v_GL = 3 * eps_G / R_pore          # ≈ 4.08e7 m⁻¹
# Sechenov coefficients (§8.10)
const h_CO2  = -0.02599
const h_Cs   =  0.0759
const h_OH   =  0.0893
const h_HCO3 =  0.0967
const h_CO3  =  0.1423

# ── Gas mixture helpers (type-generic for ForwardDiff) ────────────────────
Mn_from_omega(w) = 1.0 / (w[1]/M_gas[1] + w[2]/M_gas[2] + w[3]/M_gas[3] + w[4]/M_gas[4])
omega_to_x(w, Mn) = ntuple(j -> w[j]*Mn/M_gas[j], 4)
omega_H2O_closure(w1, w2, w3) = max(1 - w1 - w2 - w3, zero(w1))

"Raw (Bruggeman-free) mixture-average diffusivity for species j, mole fractions x."
function D_MA_raw(j::Int, x)
    s = zero(eltype(x))
    @inbounds for k in 1:4
        k == j && continue
        s += x[k] / D_bin[j, k]
    end
    return (1 - x[j]) / s
end

"""
    sm_face_flux(wL, pL, wR, pR, bf_face, dx_cc) -> NTuple{3}

Stefan-Maxwell mixture-average diffusive face flux [kg/m²/s] for species 1..3
(CO₂, CO, H₂), Stage 3 form (u_G = 0): −ρ D^MA ∇ω − ρ D^MA ω ∇M_n/M_n.
`w*` are 4-tuples (incl. H₂O closure). bf_face = face Bruggeman factor.
"""
function sm_face_flux(wL, pL::Real, wR, pR::Real, bf_face::Real, dx_cc::Real)
    wf = ntuple(j -> 0.5*(wL[j] + wR[j]), 4)
    pf = 0.5*(pL + pR)
    MnL = Mn_from_omega(wL); MnR = Mn_from_omega(wR); Mnf = Mn_from_omega(wf)
    rho_f = pf * Mnf / (R_gas * T)
    xf = omega_to_x(wf, Mnf)
    dMndx = (MnR - MnL) / dx_cc
    return ntuple(3) do j
        DMA  = bf_face * D_MA_raw(j, xf)
        dwdx = (wR[j] - wL[j]) / dx_cc
        -rho_f * DMA * dwdx - rho_f * DMA * wf[j] * dMndx / Mnf
    end
end

const kappa_interface = 2*kappa_GDL*kappa_CL/(kappa_GDL + kappa_CL)   # harmonic mean (§Stage 4)

# §8.8 Solid-phase electronic conductivities (Bruggeman on solid fraction), Stage 5
const sigma_s_GDL_raw = 220.0          # S/m, Sigracet 39BC
const sigma_s_CL_raw  = 100.0          # S/m, Ag/C percolating phase
const sigma_s_eff_GDL = sigma_s_GDL_raw * (1 - eps_GDL)^1.5   # ≈ 73 S/m
const sigma_s_eff_CL  = sigma_s_CL_raw  * eps_S^1.5           # ≈ 11 S/m
const sigma_s_eff_iface = 2*sigma_s_eff_GDL*sigma_s_eff_CL/(sigma_s_eff_GDL + sigma_s_eff_CL)

"""
    sm_darcy_face_flux(wL, pL, wR, pR, bf_face, kappa_face, dx_cc) -> (NTuple{3}, rhou)

Stage-4 gas face flux: Stefan-Maxwell diffusion + Darcy convection.
Species mass flux J_j = −ρ D^MA ∇ω_j − ρ D^MA ω_j ∇M_n/M_n + ρ ω_j u_G  [kg/m²/s];
total Darcy mass flux ρu = ρ u_G with u_G = −(κ/μ)(dp/dx).
"""
function sm_darcy_face_flux(wL, pL::Real, wR, pR::Real, bf_face::Real, kappa_face::Real, dx_cc::Real)
    wf = ntuple(j -> 0.5*(wL[j] + wR[j]), 4)
    pf = 0.5*(pL + pR)
    MnL = Mn_from_omega(wL); MnR = Mn_from_omega(wR); Mnf = Mn_from_omega(wf)
    rho_f = pf * Mnf / (R_gas * T)
    xf = omega_to_x(wf, Mnf)
    dMndx = (MnR - MnL) / dx_cc
    uG = -(kappa_face / mu_gas) * (pR - pL) / dx_cc
    rhou = rho_f * uG
    J = ntuple(3) do j
        DMA  = bf_face * D_MA_raw(j, xf)
        dwdx = (wR[j] - wL[j]) / dx_cc
        -rho_f*DMA*dwdx - rho_f*DMA*wf[j]*dMndx/Mnf + rho_f*wf[j]*uG
    end
    return J, rhou
end

"""
    sechenov_H(c) -> H_eff [mol/(m³·atm)]

Combined van 't Hoff + Sechenov effective Henry constant (§7.4) at local liquid
composition c = [H, OH, HCO3, CO3, CO2]. Cs⁺ from electroneutrality (§2).
"""
function sechenov_H(c)
    cH, cOH, cHCO3, cCO3 = c[1], c[2], c[3], c[4]
    cCs = cOH + cHCO3 + 2*cCO3 - cH - C_fix*eps_I        # electroneutrality
    H0  = H_CO2_0 * exp(dH_sol * (1/T - 1/298.15))        # van 't Hoff
    expo = -((h_CO2 + h_Cs)*cCs + (h_CO2 + h_OH)*cOH +
             (h_CO2 + h_HCO3)*cHCO3 + (h_CO2 + h_CO3)*cCO3) / 1000.0
    return H0 * 10.0^expo
end

# ── Gas DOF indexing (global vector: liquid 5·N_CL, then gas GDL, then gas CL) ──
gas_gdl_idx(ig::Int, j::Int, N_CL::Int)           = 5*N_CL + 4*(ig - 1) + j
gas_cl_idx(ix::Int, j::Int, N_CL::Int, N_GDL::Int) = 5*N_CL + 4*N_GDL + 4*(ix - 1) + j

"Global gas-cell index g (1..N_GDL = GDL, N_GDL+1..N_GDL+N_CL = CL) → DOF index for component j."
function gas_dof(g::Int, j::Int, N_CL::Int, N_GDL::Int)
    return g <= N_GDL ? gas_gdl_idx(g, j, N_CL) : gas_cl_idx(g - N_GDL, j, N_CL, N_GDL)
end

"Unpack the 4-tuple ω (incl. H₂O closure) and pressure p for global gas cell g."
function gas_state(u, g::Int, N_CL::Int, N_GDL::Int)
    w1 = u[gas_dof(g,1,N_CL,N_GDL)]; w2 = u[gas_dof(g,2,N_CL,N_GDL)]; w3 = u[gas_dof(g,3,N_CL,N_GDL)]
    p  = u[gas_dof(g,4,N_CL,N_GDL)]
    w4 = omega_H2O_closure(w1, w2, w3)
    return (w1, w2, w3, w4), p
end

"Local CO₂ partial pressure [atm] for gas cell g."
function pCO2_atm(u, g::Int, N_CL::Int, N_GDL::Int)
    w, p = gas_state(u, g, N_CL, N_GDL)
    Mn = Mn_from_omega(w)
    x  = omega_to_x(w, Mn)
    return x[1] * p / 101325.0
end
