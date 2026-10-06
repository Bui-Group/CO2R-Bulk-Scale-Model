# params.jl — all physical constants, geometry, kinetics (§8, §15)
# Guide: CO2R_BULK_MODEL_IMPLEMENTATION_GUIDE.md (V7.1)

# ── §8.1 Physical constants ────────────────────────────────────────────────
const F_const = 96485.332          # C/mol
const R_gas   = 8.314462618        # J/(mol K)
const T       = 323.15             # K  (50 °C)
const k_B_eV  = 8.617333e-5        # eV/K

# ── §15.2 Potential referencing ────────────────────────────────────────────
const pH_anode            = 8.0
const nernst_fac          = 2.303 * R_gas * T / F_const      # 0.06413 V/pH
const V_RHE_to_SHE_offset = nernst_fac * pH_anode            # 0.51305 V

# ── §8.2 Geometry ──────────────────────────────────────────────────────────
const L_CL  = 10.0e-6              # m
const L_GDL = 325.0e-6             # m
const r_p   = 25.0e-9              # m, catalyst particle radius

# Mesh (production default §8.2; env overrides for development only)
const N_CL_DEFAULT  = parse(Int, get(ENV, "MESH_N_CL",  "80"))
const N_GDL_DEFAULT = parse(Int, get(ENV, "MESH_N_GDL", "120"))
const MESH_Q        = 1.3

# ── §8.3 Volume fractions ──────────────────────────────────────────────────
const eps_S   = 0.23               # solid (Ag/C + carbon)
const eps_G   = 0.34               # gas-filled macro-pores (CL)
const eps_I   = 0.43               # ionomer + liquid
const alpha_ion = 0.11             # ion diffusivity correction
const C_fix   = 1000.0             # mol/m³ fixed ionomer charge
const S_pore  = eps_I / (eps_I + eps_G)          # ≈ 0.5584 pore saturation

# §3.3 specific catalyst area per CL volume (saturation corrected)
const a_v_CL = S_pore * 3.0 * eps_S / r_p        # ≈ 1.541e7 1/m

# ── §8.4 Liquid diffusivities at 50 °C (bare water) ────────────────────────
#         1: H⁺  2: OH⁻  3: HCO₃⁻  4: CO₃²⁻  5: CO₂(aq)
const D_w    = [1.33e-8, 8.19e-9, 2.09e-9, 1.63e-9, 3.39e-9]   # m²/s
const z_ion  = [ 1.0,   -1.0,    -1.0,    -2.0,     0.0]
const D_w_Cs = 3.63e-9
const z_Cs   = 1.0

# §3.1 / §8.3 effective (Bruggeman + ionomer) liquid diffusivities
const D_eff    = (eps_I^1.5 * alpha_ion) .* D_w
const D_eff_Cs = eps_I^1.5 * alpha_ion * D_w_Cs

# ── §8.5 Kinetics (Ref-backed so the fit can sweep without recompiling) ────
const U_CO_0 = -0.11               # V vs SHE
const c_ref  = 1000.0              # mol/m³
const n_e    = 2

const _i0_ref     = Ref{Float64}(540.0)
const _lam_ref    = Ref{Float64}(1.8)
const _her_pf_ref = Ref{Float64}(2.409e13)
const _ac_ref     = Ref{Float64}(0.0872)
const _kGL_ref    = Ref{Float64}(1.0e-4)

i0_CO2R()       = _i0_ref[]
lambda_MHC()    = _lam_ref[]
HER_prefactor() = _her_pf_ref[]
alpha_c_HER()   = _ac_ref[]
k_GL()          = _kGL_ref[]

set_i0_CO2R!(v)       = (_i0_ref[]     = v)
set_lambda_MHC!(v)    = (_lam_ref[]    = v)
set_HER_prefactor!(v) = (_her_pf_ref[] = v)
set_alpha_c_HER!(v)   = (_ac_ref[]     = v)
set_k_GL!(v)          = (_kGL_ref[]    = v)

const E_act_HER   = 83000.0        # J/mol at pH 0
const dEact_dpH   = 1000.0         # J/(mol pH)

# ── §8.6 Buffer chemistry at 50 °C ─────────────────────────────────────────
const K_w = 5.48e-8                # (mol/m³)²
const K_2 = 5.16e-4                # mol/m³
const K_3 = 6.74e-8                # mol/m³
const K_4 = K_2 / K_w
const K_5 = K_3 / K_w

const k1 = 2.861e3                 # mol/m³/s
const k2 = 0.265                   # 1/s
const k3 = 59.44                   # 1/s
const k4 = 12.4                    # m³/(mol s)
const k5 = 6.0e6                   # m³/(mol s)

const k1r = k1 / K_w
const k2r = k2 / K_2
const k3r = k3 / K_3
const k4r = k4 / K_4
const k5r = k5 / K_5

# ── §8.7 Gas transport ─────────────────────────────────────────────────────
#         gas index 1: CO₂  2: CO  3: H₂  4: H₂O
const M_gas = [44.01e-3, 28.01e-3, 2.016e-3, 18.015e-3]        # kg/mol
const v_fsg = [26.9, 18.9, 7.07, 12.7]                          # cm³/mol

const x_inlet = [0.88, 0.0, 0.0, 0.12]                          # MOLE fractions
const M_avg_inlet = sum(x_inlet[j] * M_gas[j] for j in 1:4)
const omega_inlet = [x_inlet[j] * M_gas[j] / M_avg_inlet for j in 1:4]

const p_total  = 101325.0          # Pa
const eps_GDL  = 0.52
const kappa_GDL = 8.4e-13          # m²
const kappa_CL  = 8.0e-16 * (1.0 - S_pore)^3     # ≈ 6.9e-17 m²
const mu_gas    = 1.59e-5          # Pa s

# §6.3 Fuller-Schettler-Giddings binary diffusivities  [m²/s]
function _fsg_binary()
    D = zeros(4, 4)
    p_atm = p_total / 101325.0
    for i in 1:4, j in 1:4
        i == j && continue
        Dij_cm2 = 1.00e-3 * T^1.75 * sqrt(1.0/(M_gas[i]*1e3) + 1.0/(M_gas[j]*1e3)) /
                  (p_atm * (v_fsg[i]^(1/3) + v_fsg[j]^(1/3))^2)
        D[i, j] = Dij_cm2 * 1e-4
    end
    return D
end
const D_bin_raw  = _fsg_binary()
const D_bin_GDL  = eps_GDL^1.5 .* D_bin_raw
const D_bin_CL   = eps_G^1.5   .* D_bin_raw
const D_bin_INT  = [ (D_bin_GDL[i,j] == 0.0 ? 0.0 :
                      2*D_bin_GDL[i,j]*D_bin_CL[i,j]/(D_bin_GDL[i,j]+D_bin_CL[i,j]))
                     for i in 1:4, j in 1:4 ]
const kappa_INT  = 2*kappa_GDL*kappa_CL/(kappa_GDL + kappa_CL)

# ── §8.8 Solid-phase electronic conductivity ───────────────────────────────
const sigma_s_GDL     = 220.0
const sigma_s_CL      = 100.0
const sigma_s_eff_GDL = sigma_s_GDL * (1.0 - eps_GDL)^1.5      # ≈ 73.2 S/m
const sigma_s_eff_CL  = sigma_s_CL  * eps_S^1.5                # ≈ 11.0 S/m
const sigma_s_eff_INT = 2*sigma_s_eff_GDL*sigma_s_eff_CL /
                        (sigma_s_eff_GDL + sigma_s_eff_CL)

# ── §8.9 Electrolyte channel / Sherwood mass transfer (§7.2) ───────────────
const L_elec   = 0.0225            # m
const Q_elyte  = 25.0 / 60e6       # m³/s
const A_ch     = 0.8e-3 * 0.8e-3   # m²
const rho_elec = 1000.0            # kg/m³
const mu_elec  = 1.0e-3            # Pa s

const u_elec = Q_elyte / A_ch
const Re_elec = u_elec * L_elec * rho_elec / mu_elec

k_MT_of(D_i) = (D_i / L_elec) * 0.664 * sqrt(Re_elec) *
               (mu_elec / (rho_elec * D_i))^(1/3)
# ⚠ §7.2: pass the BARE-WATER D_w, not D_eff.
const k_MT = [k_MT_of(D_w[i]) for i in 1:5]

# ── §8.10 Gas-liquid interface ─────────────────────────────────────────────
const H_CO2_0 = 34.0               # mol/(m³ atm) at 25 °C
const dH_sol  = 2400.0             # K
const R_pore  = 25.0e-9            # m
const a_v_GL  = 3.0 * eps_G / R_pore                          # ≈ 4.08e7 1/m

# Sechenov coefficients (h_CO2 for the gas; h_i per ion) [L/mol]
const h_CO2  = -0.02599
const h_Cs   =  0.0759
const h_OH   =  0.0893
const h_HCO3 =  0.0967
const h_CO3  =  0.1423

# van 't Hoff-only Henry constant (Stages 1–2, no Sechenov)
H_vanthoff() = H_CO2_0 * exp(dH_sol * (1.0/T - 1.0/298.15))    # ≈ 18.24
const p_CO2_inlet_atm = x_inlet[1] * (p_total / 101325.0)      # 0.88 atm

# ── §9.5.5 Canonical voltage grid ──────────────────────────────────────────
const V_RHE_grid = collect(range(-0.300, stop = -2.500, length = 89))
const V_SHE_grid = V_RHE_grid .- V_RHE_to_SHE_offset

# ── §16.1 Experimental data (Romiluyi et al. 2023) ─────────────────────────
const E_exp_RHE = [-0.816, -1.205, -1.644, -2.079, -2.413]
const i_CO_exp  = [35.1, 189.6, 299.1, 204.3, 64.6]            # mA/cm²
const i_H2_exp  = [29.9, 49.3, 119.6, 449.9, 850.5]            # mA/cm²
const E_exp_SHE = E_exp_RHE .- V_RHE_to_SHE_offset
const FE_CO_exp = i_CO_exp ./ (i_CO_exp .+ i_H2_exp) .* 100
const FE_H2_exp = i_H2_exp ./ (i_CO_exp .+ i_H2_exp) .* 100
