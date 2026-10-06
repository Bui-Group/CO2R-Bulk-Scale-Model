# =====================================================================
#  params.jl — all physical constants, geometry, kinetics
#  CO2R Bulk-Scale Julia Model, guide V7 (Affine Solver)  §8, §15, §16
# =====================================================================

using SpecialFunctions          # erfc
using Printf

# ── §8.1 Physical constants ──────────────────────────────────────────
const F_const = 96485.332          # C/mol          CODATA 2018
const R_gas   = 8.314462618        # J/(mol K)      CODATA 2018
const T       = 323.15             # K              50 °C
const k_B_eV  = 8.617333e-5        # eV/K           CODATA 2018

# ── §8.2 Geometry ────────────────────────────────────────────────────
const L_CL  = 10.0e-6              # m   catalyst layer thickness
const L_GDL = 325.0e-6             # m   GDL thickness
const r_p   = 25.0e-9              # m   catalyst particle radius

# Production default mesh (§8.2 / §9.1); env overrides for dev only.
const N_CL_DEFAULT  = parse(Int, get(ENV, "MESH_N_CL",  "80"))
const N_GDL_DEFAULT = parse(Int, get(ENV, "MESH_N_GDL", "120"))
const MESH_Q        = 1.3          # geometric grading ratio (largest/smallest cell)

# ── §8.3 Volume fractions ────────────────────────────────────────────
const eps_S   = 0.23               # solid volume fraction (Ag/C + carbon)
const eps_G   = 0.34               # gas-filled macro-pores
const eps_I   = 0.43               # ionomer + liquid combined phase
const alpha_ion = 0.11             # ion diffusivity correction factor
const C_fix   = 1000.0             # mol/m³ fixed (cationic) charge of the ionomer
const S_pore  = eps_I / (eps_I + eps_G)          # ≈ 0.5584 pore saturation

# §3.3 specific catalyst area per CL volume, saturation-corrected
const a_v_CL  = S_pore * 3.0 * eps_S / r_p       # ≈ 1.541e7 m⁻¹

# §7.4 ionomer/gas-pore contact area per CL volume
const R_pore  = 25.0e-9            # m  characteristic gas-pore radius
const a_v_GL  = 3.0 * eps_G / R_pore             # ≈ 4.08e7 m⁻¹

# ── §8.4 Liquid diffusivities at 50 °C (bare water) ──────────────────
#   order: 1 H⁺, 2 OH⁻, 3 HCO₃⁻, 4 CO₃²⁻, 5 CO₂(aq)
const D_w   = [1.33e-8, 8.19e-9, 2.09e-9, 1.63e-9, 3.39e-9]   # m²/s
const z_ion = [ 1.0,   -1.0,    -1.0,    -2.0,     0.0    ]   # charge numbers
const D_w_Cs = 3.63e-9             # m²/s   Cs⁺
const z_Cs   = 1.0

# §3.1 / §8.3 effective (Bruggeman + ionomer) liquid diffusivities
const D_eff    = (eps_I^1.5) .* alpha_ion .* D_w                 # m²/s
const D_eff_Cs = (eps_I^1.5)  * alpha_ion  * D_w_Cs              # ≈ 1.13e-10 m²/s

const SPECIES_NAMES = ["H+", "OH-", "HCO3-", "CO3^2-", "CO2(aq)"]

# ── §8.6 Buffer chemistry at 50 °C (mol/m³ units) ────────────────────
const K_w = 5.48e-8                # (mol/m³)²
const K_2 = 5.16e-4                # mol/m³
const K_3 = 6.74e-8                # mol/m³
const K_4 = K_2 / K_w              # derived, never independent
const K_5 = K_3 / K_w              # derived

const k1f = 2.861e3                # mol/m³/s
const k2f = 0.265                  # 1/s
const k3f = 59.44                  # 1/s
const k4f = 12.4                   # m³/(mol s)
const k5f = 6.0e6                  # m³/(mol s)

const k1r = k1f / K_w
const k2r = k2f / K_2
const k3r = k3f / K_3
const k4r = k4f / K_4
const k5r = k5f / K_5

# ── §8.5 Kinetic parameters (Ref-backed for in-process fitting, §13/§17.2)
const n_e     = 2
const U_CO_0  = -0.11              # V vs SHE, CO₂/CO standard reduction
const c_ref   = 1000.0             # mol/m³

const _i0_ref     = Ref{Float64}(540.0)      # A/m²   i0_CO2R  (initial guess)
const _lam_ref    = Ref{Float64}(1.8)        # eV     λ_MHC    (initial guess)
const _her_pf_ref = Ref{Float64}(2.409e13)   # mA/cm² HER prefactor (initial guess)
const _ac_ref     = Ref{Float64}(0.0872)     # –      α_c HER  (initial guess)
const _kGL_ref    = Ref{Float64}(1.0e-4)     # m/s    k_GL     (§17.9 second tier)

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

const E_act_HER   = 83000.0        # J/mol  activation energy at pH 0
const dEact_dpH   = 1000.0         # J/(mol pH)

# ── §8.7 Gas transport parameters ────────────────────────────────────
#   gas species order: 1 CO₂, 2 CO, 3 H₂, 4 H₂O
const M_gas = [44.01e-3, 28.01e-3, 2.016e-3, 18.015e-3]   # kg/mol
const v_fsg = [26.9, 18.9, 7.07, 12.7]                    # cm³/mol FSG volumes
const GAS_NAMES = ["CO2", "CO", "H2", "H2O"]

const p_total  = 101325.0          # Pa
const x_inlet  = [0.88, 0.0, 0.0, 0.12]                   # MOLE fractions (§7.3)
const M_avg_in = sum(x_inlet[j] * M_gas[j] for j in 1:4)  # ≈ 40.89 g/mol
const omega_inlet = [x_inlet[j] * M_gas[j] / M_avg_in for j in 1:4]   # ≈ [0.947,0,0,0.053]

const eps_GDL  = 0.52              # Sigracet 39BC porosity
const kappa_GDL = 8.4e-13          # m²  dry Darcy permeability
const kappa_CL  = 8.0e-16 * (1.0 - S_pore)^3     # m²  ≈ 6.9e-17 (Leverett)
const kappa_interface = 2.0 * kappa_GDL * kappa_CL / (kappa_GDL + kappa_CL)
const mu_gas   = 1.59e-5           # Pa s  Wilke mixing rule at 50 °C

# §6.3 Fuller-Schettler-Giddings binary diffusivities (prefactor 1.00e-3!)
function fsg_binary()
    D = zeros(4, 4)
    for i in 1:4, j in 1:4
        i == j && continue
        Mi = M_gas[i] * 1000.0     # g/mol
        Mj = M_gas[j] * 1000.0
        Dcm2 = 1.00e-3 * T^1.75 * sqrt(1.0/Mi + 1.0/Mj) /
               (1.0 * (v_fsg[i]^(1/3) + v_fsg[j]^(1/3))^2)
        D[i, j] = Dcm2 * 1.0e-4    # cm²/s → m²/s
    end
    return D
end
const D_bin_raw = fsg_binary()
const D_bin_GDL = (eps_GDL^1.5) .* D_bin_raw     # §8.7 Bruggeman
const D_bin_CL  = (eps_G^1.5)   .* D_bin_raw

# ── §8.8 Solid-phase electronic conductivity ─────────────────────────
const sigma_s_GDL = 220.0          # S/m raw
const sigma_s_CL  = 100.0          # S/m raw
const sigma_s_eff_GDL = sigma_s_GDL * (1.0 - eps_GDL)^1.5     # ≈ 73 S/m
const sigma_s_eff_CL  = sigma_s_CL  * eps_S^1.5               # ≈ 11 S/m
const sigma_s_eff_int = 2.0 * sigma_s_eff_GDL * sigma_s_eff_CL /
                        (sigma_s_eff_GDL + sigma_s_eff_CL)    # harmonic mean (§3.4)

# ── §7.2 / §8.9 Electrolyte channel + Sherwood mass transfer ─────────
const L_elec   = 0.0225            # m
const Q_elyte  = 25.0 / 60e6       # m³/s   (25 mL/min)
const A_ch     = 0.8e-3 * 0.8e-3   # m²
const d_h      = 0.8e-3            # m
const rho_elec = 1000.0            # kg/m³
const mu_elec  = 1.0e-3            # Pa s

const u_elec = Q_elyte / A_ch
const Re_elec = u_elec * L_elec * rho_elec / mu_elec           # ≈ 14648

# Pass bare-water D_w, NOT D_eff (§7.2 warning box)
k_MT_of(D_i) = (D_i / L_elec) * 0.664 * sqrt(Re_elec) *
               (mu_elec / (rho_elec * D_i))^(1/3)
const k_MT = [k_MT_of(D_w[i]) for i in 1:5]                    # m/s
const L_eff_bc = [D_eff[i] / k_MT[i] for i in 1:5]             # m (§9.1 np_flux)

# ── §8.10 Gas-liquid interface / Henry ───────────────────────────────
const H_CO2_0 = 34.0               # mol/(m³ atm) at 25 °C
const dH_sol  = 2400.0             # K  van 't Hoff
const H_vanthoff = H_CO2_0 * exp(dH_sol * (1.0/T - 1.0/298.15))  # ≈ 18.24

# Sechenov h coefficients [L/mol] — Schumpe 1993, Weisenberger & Schumpe 1996
const h_CO2  = -0.02599
const h_Cs   =  0.0759
const h_OH   =  0.0893
const h_HCO3 =  0.0967
const h_CO3  =  0.1423

const p_CO2_inlet = x_inlet[1] * 1.0          # atm (p_total = 1 atm)
const c_CO2_henry_stage12 = H_vanthoff * p_CO2_inlet   # ≈ 16.05 mol/m³ (§7.1)

# ── §15 Potential referencing ────────────────────────────────────────
const pH_anode   = 8.0
const nernst_fac = 2.303 * R_gas * T / F_const                 # 0.06413 V/pH
const V_RHE_to_SHE_offset = nernst_fac * pH_anode              # 0.51305 V

# ── §9.5.5 Canonical voltage grid ────────────────────────────────────
const V_RHE_grid = collect(range(-0.300, stop = -2.500, length = 89))
const V_SHE_grid = V_RHE_grid .- V_RHE_to_SHE_offset

# ── §16.1 Experimental data — Romiluyi et al. 2023 ───────────────────
const E_exp_RHE = [-0.816, -1.205, -1.644, -2.079, -2.413]     # V vs RHE@pH8
const i_CO_exp  = [35.1, 189.6, 299.1, 204.3, 64.6]            # mA/cm²
const i_H2_exp  = [29.9, 49.3, 119.6, 449.9, 850.5]            # mA/cm²
const E_exp_SHE = E_exp_RHE .- V_RHE_to_SHE_offset
const FE_CO_exp = i_CO_exp ./ (i_CO_exp .+ i_H2_exp) .* 100
const FE_H2_exp = i_H2_exp ./ (i_CO_exp .+ i_H2_exp) .* 100

# ── §9.5.7 Production solver settings ────────────────────────────────
const RTOL_PROD   = 1e-8
const F_INF_TOL   = 1e-6
const ALPHA_ANCHOR= 1e-2
const ALPHA_SWEEP = 1.0
const ALPHA_MIN   = 1e-8
const MAXIT_SWEEP = 40
const MAXIT_ANCHOR= 200
const MAX_LOG_STEP= 5.0
const AFRAC       = 1e-3
const DV_STEP     = 0.025

function print_kinetics(tag::AbstractString)
    @printf("=== %s ===\n", tag)
    @printf("i0_CO2R = %.4f A/m^2, lambda_MHC = %.4f eV\n", i0_CO2R(), lambda_MHC())
    @printf("HER_pf  = %.6e mA/cm^2, alpha_c = %.6f, k_GL = %.3e m/s\n",
            HER_prefactor(), alpha_c_HER(), k_GL())
    flush(stdout)
end
