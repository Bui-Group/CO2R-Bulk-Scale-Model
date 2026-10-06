# params.jl — all physical constants, geometry, kinetics (Ref-backed mutables)
# CO2R Bulk-Scale Julia Model — Implementation Guide V7.1, §8 / §15
#
# NOTE: `T` is the global temperature constant.  Never use `T` as a type
# parameter anywhere in the residual chain (guide §9.3.2 item 6).

# ── §8.1 Physical constants ────────────────────────────────────────────────
const F_const = 96485.332            # C/mol      CODATA 2018
const R_gas   = 8.314462618          # J/(mol·K)  CODATA 2018
const T       = 323.15               # K          50 °C
const k_B_eV  = 8.617333e-5          # eV/K       CODATA 2018

# ── §15.2 Potential referencing ────────────────────────────────────────────
const pH_anode            = 8.0
const nernst_fac          = 2.303 * R_gas * T / F_const     # 0.06413 V/pH @50°C
const V_RHE_to_SHE_offset = nernst_fac * pH_anode           # 0.51305 V

# ── §8.2 Geometry ──────────────────────────────────────────────────────────
const L_CL   = 10e-6                 # m
const L_GDL  = 325e-6                # m
const r_p    = 25e-9                 # m   catalyst particle radius
const R_pore = 25e-9                 # m   characteristic gas-pore radius

# ── §8.3 Volume fractions ──────────────────────────────────────────────────
const eps_S     = 0.23
const eps_G     = 0.34
const eps_I     = 0.43
const alpha_ion = 0.11
const C_fix     = 1000.0             # mol/m³ ionomer fixed charge
const S_pore    = eps_I / (eps_I + eps_G)          # ≈ 0.5584
const a_v_CL    = S_pore * 3 * eps_S / r_p         # ≈ 1.541e7 m⁻¹
const a_v_GL    = 3 * eps_G / R_pore               # ≈ 4.08e7 m⁻¹
const n_e       = 2

# ── §8.4 Liquid diffusivities at 50 °C ─────────────────────────────────────
# order: H⁺, OH⁻, HCO₃⁻, CO₃²⁻, CO₂(aq)
const D_w      = [1.33e-8, 8.19e-9, 2.09e-9, 1.63e-9, 3.39e-9]   # m²/s
const D_w_Cs   = 3.63e-9
const D_eff    = eps_I^1.5 * alpha_ion .* D_w                    # §3.1
const D_eff_Cs = eps_I^1.5 * alpha_ion * D_w_Cs                  # ≈ 1.13e-10
const z_sp     = [1.0, -1.0, -1.0, -2.0, 0.0]
const z_Cs     = 1.0

# ── §8.6 Buffer chemistry at 50 °C (mol/m³ units) ──────────────────────────
const K_w = 5.48e-8                  # (mol/m³)²
const K_2 = 5.16e-4                  # mol/m³
const K_3 = 6.74e-8                  # mol/m³
const K_4 = K_2 / K_w                # ALWAYS derived
const K_5 = K_3 / K_w                # ALWAYS derived

const k1f = 2.861e3;  const k1r = k1f / K_w
const k2f = 0.265;    const k2r = k2f / K_2
const k3f = 59.44;    const k3r = k3f / K_3
const k4f = 12.4;     const k4r = k4f / K_4
const k5f = 6.0e6;    const k5r = k5f / K_5

# ── §8.5 Kinetics ──────────────────────────────────────────────────────────
const U_CO_0 = -0.11                 # V vs SHE
const c_ref  = 1000.0                # mol/m³

# Ref-backed mutable parameters (§13) — fitting sweeps without recompiling
const _i0_ref     = Ref{Float64}(540.0)
const _lam_ref    = Ref{Float64}(1.8)
const _her_pf_ref = Ref{Float64}(2.409e13)
const _ac_ref     = Ref{Float64}(0.0872)
const _kGL_ref    = Ref{Float64}(1.0e-4)

i0_CO2R()      = _i0_ref[]
lambda_MHC()   = _lam_ref[]
HER_prefactor()= _her_pf_ref[]
alpha_c_HER()  = _ac_ref[]
k_GL()         = _kGL_ref[]

set_i0_CO2R!(v)      = (_i0_ref[]     = v)
set_lambda_MHC!(v)   = (_lam_ref[]    = v)
set_HER_prefactor!(v)= (_her_pf_ref[] = v)
set_alpha_c_HER!(v)  = (_ac_ref[]     = v)
set_k_GL!(v)         = (_kGL_ref[]    = v)

const E_act_HER  = 83000.0           # J/mol
const dEact_dpH  = 1000.0            # J/(mol·pH)

# ── §8.7 Gas transport ─────────────────────────────────────────────────────
const M_gas   = [44.01e-3, 28.01e-3, 2.016e-3, 18.015e-3]   # kg/mol CO₂,CO,H₂,H₂O
const M_g_gmol= [44.01, 28.01, 2.016, 18.015]               # g/mol for FSG
const v_fsg   = [26.9, 18.9, 7.07, 12.7]                    # cm³/mol
const x_inlet = [0.88, 0.0, 0.0, 0.12]                      # MOLE fractions
const p_total = 101325.0                                    # Pa
const eps_GDL = 0.52
const kappa_GDL = 8.4e-13                                   # m²
const kappa_CL  = 8.0e-16 * (1 - S_pore)^3                  # ≈ 6.9e-17 m²
const mu_gas    = 1.59e-5                                   # Pa·s

# mole → mass fraction conversion for the inlet (§7.3)
const M_avg_inlet   = sum(x_inlet[j] * M_gas[j] for j in 1:4)
const omega_inlet   = [x_inlet[j] * M_gas[j] / M_avg_inlet for j in 1:4]

# §6.3 Fuller-Schettler-Giddings binary diffusivities [m²/s] at T, 1 atm
# prefactor 1.00e-3 (Perry's 9th ed.), NOT 1.013e-2
function _fsg_binary()
    D = zeros(4, 4)
    for i in 1:4, j in 1:4
        i == j && continue
        Dcm2 = 1.00e-3 * T^1.75 * sqrt(1/M_g_gmol[i] + 1/M_g_gmol[j]) /
               (1.0 * (v_fsg[i]^(1/3) + v_fsg[j]^(1/3))^2)
        D[i, j] = Dcm2 * 1e-4
    end
    return D
end
const D_bin_raw = _fsg_binary()

# ── §8.8 Solid-phase electronic conductivity ───────────────────────────────
const sigma_s_GDL     = 220.0
const sigma_s_CL      = 100.0
const sigma_s_eff_GDL = sigma_s_GDL * (1 - eps_GDL)^1.5     # ≈ 73 S/m
const sigma_s_eff_CL  = sigma_s_CL  * eps_S^1.5             # ≈ 11 S/m

# ── §8.9 Electrolyte channel / Sherwood mass transfer ──────────────────────
const L_elec   = 0.0225              # m
const Q_elyte  = 25.0 / 60e6         # m³/s
const A_ch     = 0.8e-3 * 0.8e-3     # m²
const rho_elec = 1000.0              # kg/m³
const mu_elec  = 1.0e-3              # Pa·s
const u_elec   = Q_elyte / A_ch
const Re_elec  = u_elec * L_elec * rho_elec / mu_elec        # ≈ 14648

# Blasius/Pohlhausen flat-plate laminar correlation.  Pass the BARE-WATER D_w,
# NOT D_eff — the boundary layer lives in the free electrolyte channel (§7.2).
k_MT_of(D_i) = (D_i / L_elec) * 0.664 * sqrt(Re_elec) *
               (mu_elec / (rho_elec * D_i))^(1/3)
const k_MT = [k_MT_of(D_w[i]) for i in 1:5]

# ── §8.10 Gas-liquid interface ─────────────────────────────────────────────
const H_CO2_0 = 34.0                 # mol/(m³·atm) at 25 °C
const dH_sol  = 2400.0               # K
const h_CO2   = -0.02599             # Sechenov, gas
const h_Cs    =  0.0759
const h_OH    =  0.0893
const h_HCO3  =  0.0967
const h_CO3   =  0.1423

# van 't Hoff-only Henry constant (Stages 1–2; Sechenov turns on in Stage 3)
const H_vanthoff  = H_CO2_0 * exp(dH_sol * (1/T - 1/298.15))    # ≈ 18.24
const p_CO2_inlet = x_inlet[1] * 1.0                            # atm
const c_CO2_henry_s12 = H_vanthoff * p_CO2_inlet                # ≈ 16.0 mol/m³

"Free Cs⁺ from electroneutrality (§2): c_Cs = c_OH+c_HCO3+2c_CO3−c_H−C_fix·ε_I"
@inline cCs_from_electroneutrality(cH, cOH, cHCO3, cCO3) =
    cOH + cHCO3 + 2*cCO3 - cH - C_fix * eps_I

"""
    sechenov_H_CO2(cH, cOH, cHCO3, cCO3) -> H_eff [mol/(m³·atm)]

Combined van 't Hoff + Sechenov effective Henry constant (§7.4).  c_Cs is the
electroneutrality value, not the free-electrolyte 1000 mol/m³.
"""
@inline function sechenov_H_CO2(cH, cOH, cHCO3, cCO3)
    cCs = cCs_from_electroneutrality(cH, cOH, cHCO3, cCO3)
    expo = -((h_CO2 + h_Cs)   * cCs   / 1000.0 +
             (h_CO2 + h_OH)   * cOH   / 1000.0 +
             (h_CO2 + h_HCO3) * cHCO3 / 1000.0 +
             (h_CO2 + h_CO3)  * cCO3  / 1000.0)
    return H_vanthoff * 10.0^clamp(expo, -30.0, 30.0)
end

# ── §16.1 Experimental data — Romiluyi et al. 2023 ─────────────────────────
const E_exp_RHE = [-0.816, -1.205, -1.644, -2.079, -2.413]   # V vs RHE@pH8
const i_CO_exp  = [35.1, 189.6, 299.1, 204.3, 64.6]          # mA/cm²
const i_H2_exp  = [29.9, 49.3, 119.6, 449.9, 850.5]          # mA/cm²
const E_exp_SHE = E_exp_RHE .- V_RHE_to_SHE_offset
const FE_CO_exp = i_CO_exp ./ (i_CO_exp .+ i_H2_exp) .* 100
const FE_H2_exp = i_H2_exp ./ (i_CO_exp .+ i_H2_exp) .* 100

# ── §9.5.5 Canonical voltage grid ──────────────────────────────────────────
const V_RHE_grid = collect(range(-0.300, stop = -2.500, length = 89))
const V_SHE_grid = V_RHE_grid .- V_RHE_to_SHE_offset

# ── Mesh defaults (§8.2) — env-var overrides for development only ──────────
const N_CL_DEFAULT  = parse(Int, get(ENV, "MESH_N_CL",  "80"))
const N_GDL_DEFAULT = parse(Int, get(ENV, "MESH_N_GDL", "120"))
const MESH_Q        = 1.3
