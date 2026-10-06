# ============================================================================
# params.jl — All physical constants, geometry, kinetics (Ref-backed mutables)
# Transcribed verbatim from Case_2_guide.md §8, §15, §16, §17.
# ============================================================================

# ── §8.1 Physical constants ────────────────────────────────────────────────
const F_const = 96485.332        # C/mol      (CODATA 2018)
const R_gas   = 8.314462618      # J/(mol·K)  (CODATA 2018)
const T       = 323.15           # K          (50 °C)
const k_B_eV  = 8.617333e-5      # eV/K       (CODATA 2018)

const n_e = 2                    # electrons per CO2R / HER reaction

# ── §8.2 Geometry ──────────────────────────────────────────────────────────
const L_CL  = 10e-6              # m  (catalyst layer thickness)
const L_GDL = 325e-6             # m  (gas diffusion layer thickness)
const r_p   = 25e-9              # m  (catalyst particle radius)

# Default production mesh (overridable via env for dev/debug only — §8.2)
const N_CL  = parse(Int, get(ENV, "MESH_N_CL",  "80"))
const N_GDL = parse(Int, get(ENV, "MESH_N_GDL", "120"))

# ── §8.3 Volume fractions ───────────────────────────────────────────────────
const eps_S   = 0.23             # solid volume fraction (Ag/C + carbon)
const eps_G   = 0.34             # gas-filled macro-pores
const eps_I   = 0.43             # ionomer + liquid combined phase
const alpha_ion = 0.11           # ion diffusivity correction factor
const C_fix   = 1000.0           # fixed charge concentration [mol/m³] (1 M)
const S_pore  = eps_I / (eps_I + eps_G)   # ≈ 0.558  pore saturation

# Specific catalyst area per CL volume (§3.3): a_v_CL = S × 3 × ε_S / r_p
const a_v_CL = S_pore * 3 * eps_S / r_p          # ≈ 1.54e7 m⁻¹

# ── §8.4 Liquid diffusivities at 50 °C (bare water D_w) ─────────────────────
#  index: 1 H⁺, 2 OH⁻, 3 HCO₃⁻, 4 CO₃²⁻, 5 CO₂(aq)
const D_w = [1.33e-8, 8.19e-9, 2.09e-9, 1.63e-9, 3.39e-9]   # m²/s
const D_w_Cs = 3.63e-9                                       # m²/s (Cs⁺, not a DOF)

# Charges (CO₂ neutral)
const z_charge = [1.0, -1.0, -1.0, -2.0, 0.0]

# Effective liquid diffusivity in CL: D_eff,i = ε_I^1.5 × α_ion × D_w,i  (§3.1, §8.3)
const D_eff = (eps_I^1.5 * alpha_ion) .* D_w

# ── §8.5 Kinetic parameters (Ref-backed mutables for in-process fitting §13/§17.2)
const U_CO_0 = -0.11             # V vs SHE  (standard CO₂/CO reduction)
const c_ref  = 1000.0            # mol/m³    (MHC normalization)
const E_act_HER     = 83000.0    # J/mol     (HER activation energy at pH=0)
const dEact_dpH_HER = 1000.0     # J/(mol·pH)

const _i0_CO2R_ref = Ref{Float64}(540.0)        # A/m²   (initial guess)
const _lambda_ref  = Ref{Float64}(1.8)          # eV     (initial guess)
const _her_pf_ref  = Ref{Float64}(2.409e13)     # mA/cm² (initial guess)
const _alpha_c_ref = Ref{Float64}(0.0872)       # —      (initial guess)
const _kGL_ref     = Ref{Float64}(1e-4)         # m/s    (GL mass transfer coeff)

i0_CO2R()       = _i0_CO2R_ref[]
lambda_MHC()    = _lambda_ref[]
HER_prefactor() = _her_pf_ref[]
alpha_c_HER()   = _alpha_c_ref[]
k_GL()          = _kGL_ref[]

set_i0_CO2R!(v)       = (_i0_CO2R_ref[] = v)
set_lambda_MHC!(v)    = (_lambda_ref[]  = v)
set_HER_prefactor!(v) = (_her_pf_ref[]  = v)
set_alpha_c_HER!(v)   = (_alpha_c_ref[] = v)
set_k_GL!(v)          = (_kGL_ref[]     = v)

# ── §8.6 Buffer chemistry equilibrium / rate constants at 50 °C ─────────────
const K_w = 5.48e-8              # (mol/m³)²
const K_2 = 5.16e-4              # mol/m³
const K_3 = 6.74e-8              # mol/m³
const K_4 = K_2 / K_w            # always derived
const K_5 = K_3 / K_w            # always derived

const k1f = 2.861e3              # mol/m³/s
const k2f = 0.265                # s⁻¹
const k3f = 59.44                # s⁻¹
const k4f = 12.4                 # m³/(mol·s)
const k5f = 6.0e6                # m³/(mol·s)

const k1r = k1f / K_w
const k2r = k2f / K_2
const k3r = k3f / K_3
const k4r = k4f / K_4
const k5r = k5f / K_5

# ── §8.7 Gas transport parameters ───────────────────────────────────────────
#  gas index: 1 CO₂, 2 CO, 3 H₂, 4 H₂O
const M_gas = [44.01e-3, 28.01e-3, 2.016e-3, 18.015e-3]   # kg/mol
const v_diff = [26.9, 18.9, 7.07, 12.7]                   # cm³/mol (FSG diffusion volumes)

const x_inlet = [0.88, 0.0, 0.0, 0.12]                    # MOLE fractions (CO₂,CO,H₂,H₂O)
const p_total = 101325.0                                  # Pa (1 atm)

const eps_GDL = 0.52             # Sigracet 39BC porosity
# eps_G (CL gas void) defined above as 0.34
const kappa_GDL = 8.4e-13        # m² (dry Darcy permeability GDL)
const kappa_CL  = 8.0e-16 * (1 - S_pore)^3   # ≈ 7.0e-17 m² (Leverett-corrected)
const mu_gas    = 1.59e-5        # Pa·s (Wilke mixing at 50°C)

# Inlet mass fractions (mole → mass, §7.3)
const M_avg_inlet  = sum(x_inlet[j] * M_gas[j] for j in 1:4)   # ≈ 0.04089 kg/mol
const omega_inlet  = [x_inlet[j] * M_gas[j] / M_avg_inlet for j in 1:4]  # ≈ [0.947,0,0,0.053]

# ── §8.8 Solid-phase electronic conductivity ────────────────────────────────
const sigma_s_GDL = 220.0        # S/m (raw)
const sigma_s_CL  = 100.0        # S/m (raw)
const sigma_s_eff_GDL = sigma_s_GDL * (1 - eps_GDL)^1.5   # ≈ 73 S/m
const sigma_s_eff_CL  = sigma_s_CL  * eps_S^1.5           # ≈ 11 S/m

# ── §8.9 Electrolyte channel / mass transfer parameters ─────────────────────
const L_elec   = 0.0225          # m   (electrode/characteristic length)
const Q_elyte  = 25.0 / 60e6     # m³/s (25 mL/min)
const A_ch     = 0.8e-3 * 0.8e-3 # m²  (channel cross-section)
const d_h      = 0.8e-3          # m   (hydraulic diameter, square channel)
const rho_elec = 1000.0          # kg/m³ (water)
const mu_elec  = 1.0e-3          # Pa·s (water)

const u_elec = Q_elyte / A_ch                                   # ≈ 0.651 m/s
const Re_elec = u_elec * L_elec * rho_elec / mu_elec            # ≈ 14648

# Sherwood/Blasius–Pohlhausen per-species mass transfer coefficient (§7.2).
# IMPORTANT: pass bare-water D_w, NOT D_eff.
function k_MT_of(D_i)
    return (D_i / L_elec) * 0.664 * sqrt(Re_elec) * (mu_elec / (rho_elec * D_i))^(1/3)
end
const k_MT = [k_MT_of(D_w[i]) for i in 1:5]                     # m/s, per liquid species

# ── §8.10 Gas-liquid interface parameters ───────────────────────────────────
const H_CO2_0 = 34.0             # mol/(m³·atm) at 25°C
const dH_sol  = 2400.0           # K (van't Hoff)
const R_pore  = 25e-9            # m (gas-pore radius)
const a_v_GL  = 3 * eps_G / R_pore             # ≈ 4.08e7 m⁻¹

# Sechenov coefficients (gas + ions); index aligns with liquid species 1..5
#   h for: CO₂(gas)=-0.02599, Cs⁺=+0.0759, OH⁻=+0.0893, HCO₃⁻=+0.0967, CO₃²⁻=+0.1423
const h_CO2  = -0.02599
const h_Cs   =  0.0759
const h_OH   =  0.0893
const h_HCO3 =  0.0967
const h_CO3  =  0.1423

# van't Hoff-only Henry constant at temperature T (mol/(m³·atm)) — used in Stages 1–2
H_eff_vantHoff() = H_CO2_0 * exp(dH_sol * (1/T - 1/298.15))     # ≈ 18.24 at 50°C

# Stage-1/2 CO₂(aq) Dirichlet boundary value at the GDL face (§7.1):
#   c_CO2_bc = H_eff(50°C) × p_CO2_inlet,  p_CO2_inlet = x_CO2 × p_total = 0.88 atm
const p_CO2_inlet = x_inlet[1] * 1.0                           # atm
const c_CO2_henry_stage12 = H_eff_vantHoff() * p_CO2_inlet     # ≈ 16.0 mol/m³

# ── §15 Potential referencing ───────────────────────────────────────────────
const pH_anode = 8.0
const nernst_fac = 2.303 * R_gas * T / F_const                 # 0.06413 V/pH at 50°C
const V_RHE_to_SHE_offset = nernst_fac * pH_anode              # 0.51305 V

# ── §9.5.5 Canonical voltage sweep grid ───────────────────────────────────────
const V_RHE_grid = collect(range(-0.300, stop = -2.500, length = 89))   # step 0.025 V
const V_SHE_grid = V_RHE_grid .- V_RHE_to_SHE_offset

# ── §16.1 Experimental data — Lees et al. 2024, Nat. Chem. Eng. ─────────────
const E_exp_RHE = [-0.816, -1.205, -1.644, -2.079, -2.413]    # V vs RHE@pH8
const i_CO_exp  = [35.1, 189.6, 299.1, 204.3, 64.6]           # mA/cm²
const i_H2_exp  = [29.9, 49.3, 119.6, 449.9, 850.5]           # mA/cm²
const E_exp_SHE = E_exp_RHE .- V_RHE_to_SHE_offset
const FE_CO_exp = i_CO_exp ./ (i_CO_exp .+ i_H2_exp) .* 100
const FE_H2_exp = i_H2_exp ./ (i_CO_exp .+ i_H2_exp) .* 100
