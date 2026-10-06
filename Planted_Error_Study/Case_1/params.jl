# params.jl — All physical constants, geometry, kinetics (Ref-backed mutables)
# Source: Case_1_guide.md §8 (parameter tables), §3.3 (a_v_CL), §15 (Nernst shift).
# All concentrations in mol/m³, all potentials in V vs SHE unless noted.

using SpecialFunctions   # erfc

# ── §8.1 Physical constants ───────────────────────────────────────────────
const F_const = 96485.332          # C/mol      (CODATA 2018)
const R_gas   = 8.314462618        # J/(mol·K)  (CODATA 2018)
const T       = 323.15             # K          (50 °C)
const k_B_eV  = 8.617333e-5        # eV/K       (CODATA 2018)
const n_e     = 2                  # electrons per CO₂R / HER step

# ── §8.2 Geometry ─────────────────────────────────────────────────────────
const L_CL  = 10e-6                # m
const L_GDL = 325e-6               # m
const r_p   = 25e-9                # m (catalyst particle radius)

# ── §8.3 Volume fractions ─────────────────────────────────────────────────
const eps_S    = 0.23              # solid volume fraction (Ag/C + carbon)
const eps_G    = 0.34              # gas-filled macro-pores
const eps_I    = 0.43              # ionomer + liquid combined phase
const alpha_ion = 0.11             # ion diffusivity correction factor
const C_fix    = 1000.0            # fixed charge concentration [mol/m³] (1 M)
const S_pore   = eps_I / (eps_I + eps_G)   # ≈ 0.558 pore saturation

# Specific catalyst area per CL volume (§3.3), saturation-corrected:
#   a_v_CL = S × 3 × ε_S / r_p  ≈ 1.54e7 m⁻¹
const a_v_CL = S_pore * 3 * eps_S / r_p

# ── §8.4 Liquid diffusivities at 50 °C [m²/s] ─────────────────────────────
# Species order: 1=H⁺, 2=OH⁻, 3=HCO₃⁻, 4=CO₃²⁻, 5=CO₂(aq)
const D_w = [1.33e-8, 8.19e-9, 2.09e-9, 1.63e-9, 3.39e-9]
const z   = [ 1.0,    -1.0,    -1.0,    -2.0,     0.0]      # charges

# Effective liquid diffusivity in CL (Bruggeman with ε_I + α_ion), §3.1/§8.3
#   D_eff,i = ε_I^1.5 × α_ion × D_w,i   (same α_ion applied to neutral CO₂)
const D_eff = [eps_I^1.5 * alpha_ion * D_w[i] for i in 1:5]

# Cs⁺ (recovered from electroneutrality; needed for migration current, §3.1)
const D_w_Cs   = 3.63e-9
const D_eff_Cs = eps_I^1.5 * alpha_ion * D_w_Cs
const z_Cs     = 1.0

# ── §15.2 Nernst shift / potential referencing ────────────────────────────
const pH_anode           = 8.0
const nernst_fac         = 2.303 * R_gas * T / F_const      # 0.06413 V/pH at 50 °C
const V_RHE_to_SHE_offset = nernst_fac * pH_anode           # 0.51305 V

# ── §8.5 Kinetic parameters (Ref-backed mutables for in-process fitting) ──
const _i0_ref     = Ref{Float64}(540.0)        # i0_CO2R   [A/m²]
const _lam_ref    = Ref{Float64}(1.8)          # λ_MHC     [eV]
const _her_pf_ref = Ref{Float64}(2.409e13)     # HER prefactor [mA/cm²]
const _ac_ref     = Ref{Float64}(0.0872)       # α_c_HER   [-]
const _kGL_ref    = Ref{Float64}(1e-4)         # k_GL      [m/s] (2nd-tier fit)

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

const c_ref       = 1000.0         # mol/m³ (MHC normalization)
const U_CO_0      = -0.11          # V vs SHE (CO₂/CO standard reduction)
const E_act_HER   = 83000.0        # J/mol
const dEact_dpH   = 1000.0         # J/(mol·pH)

# ── §8.6 Buffer chemistry at 50 °C (mol/m³ units) ─────────────────────────
const K_w = 5.48e-8                # (mol/m³)²
const K_2 = 5.16e-4                # mol/m³
const K_3 = 6.74e-8                # mol/m³
const K_4 = K_2 / K_w              # always derived
const K_5 = K_3 / K_w              # always derived

const k1 = 2.861e3                 # mol/m³/s
const k2 = 0.265                   # s⁻¹
const k3 = 59.44                   # s⁻¹
const k4 = 12.4                    # m³/(mol·s)
const k5 = 6.0e6                   # m³/(mol·s)

const k1r = k1 / K_w               # reverse rates (never set independently)
const k2r = k2 / K_2
const k3r = k3 / K_3
const k4r = k4 / K_4
const k5r = k5 / K_5

# ── §8.10 Gas-liquid interface: Henry's law (van 't Hoff) ─────────────────
const H_CO2_0 = 34.0               # mol/(m³·atm) at 25 °C
const dH_sol  = 2400.0             # K (van 't Hoff sensitivity)
const p_CO2_inlet = 0.88           # atm (x_CO2,inlet × p_total)

# van 't Hoff-only effective Henry constant (Stages 1–2; Sechenov off)
H_eff_vanthoff(Temp::Real) = H_CO2_0 * exp(dH_sol * (1.0/Temp - 1.0/298.15))
# Stage 1–2 CO₂(aq) Dirichlet/Henry boundary value [mol/m³]:
#   c_CO2_henry = H_eff(50 °C) × p_CO2_inlet ≈ 18.24 × 0.88 ≈ 16.0
const c_CO2_henry = H_eff_vanthoff(T) * p_CO2_inlet

# ── §8.9 Electrolyte channel / Sherwood-Reynolds mass transfer (Stage 2+) ─
const L_elec   = 0.0225            # m, electrode/characteristic length
const Q_elyte  = 25.0 / 60e6       # m³/s (25 mL/min)
const A_ch     = 0.8e-3 * 0.8e-3   # m² channel cross-section
const rho_elec = 1000.0            # kg/m³ (≈ water)
const mu_elec  = 1.0e-3            # Pa·s  (≈ water)

const u_elec = Q_elyte / A_ch
const Re_elec = u_elec * L_elec * rho_elec / mu_elec   # ≈ 14648

# Per-species mass transfer coefficient — pass bare-water D_w (NOT D_eff!), §7.2
k_MT_of(D_i::Real) = (D_i / L_elec) * 0.664 * sqrt(Re_elec) *
                     (mu_elec / (rho_elec * D_i))^(1/3)
const k_MT = [k_MT_of(D_w[i]) for i in 1:5]

# ── Experimental data — Lees et al. 2024 (§16.1) ──────────────────────────
const E_exp_RHE = [-0.816, -1.205, -1.644, -2.079, -2.413]   # V vs RHE@pH8
const i_CO_exp  = [35.1, 189.6, 299.1, 204.3, 64.6]          # mA/cm²
const i_H2_exp  = [29.9, 49.3, 119.6, 449.9, 850.5]          # mA/cm²
const E_exp_SHE = E_exp_RHE .- V_RHE_to_SHE_offset
const FE_CO_exp = i_CO_exp ./ (i_CO_exp .+ i_H2_exp) .* 100
const FE_H2_exp = i_H2_exp ./ (i_CO_exp .+ i_H2_exp) .* 100

# ── §9.6 Canonical voltage sweep grid ─────────────────────────────────────
const V_RHE_grid = collect(range(-0.300, stop = -2.500, length = 89))  # step 0.025 V
const V_SHE_grid = V_RHE_grid .- V_RHE_to_SHE_offset
