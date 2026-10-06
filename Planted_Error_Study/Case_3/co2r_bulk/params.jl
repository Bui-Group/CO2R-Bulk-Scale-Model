# params.jl — physical constants, geometry, kinetics (Ref-backed mutables)
# All values transcribed verbatim from Case_3_guide.md §8, §15, §17.
# Concentrations in mol/m³, lengths in m, potentials in V vs SHE unless noted.

# ─────────────────────────────────────────────────────────────────────────────
# §8.1 Physical constants (CODATA 2018)
const F_const = 96485.332        # C/mol
const R_gas   = 8.314462618      # J/(mol·K)
const T       = 323.15           # K  (50°C)
const k_B_eV  = 8.617333e-5      # eV/K
const n_e     = 2                # electrons per reaction (CO₂R and HER both 2e⁻)

# ─────────────────────────────────────────────────────────────────────────────
# §8.2 Geometry
const L_CL  = 10e-6              # m
const L_GDL = 325e-6            # m
const r_p   = 25e-9             # m  (catalyst particle radius)

# Mesh sizes — production default 80/120; dev override via env vars (§8.2, §9.1)
const N_CL  = parse(Int, get(ENV, "MESH_N_CL",  "80"))
const N_GDL = parse(Int, get(ENV, "MESH_N_GDL", "120"))

# ─────────────────────────────────────────────────────────────────────────────
# §8.3 Volume fractions
const eps_S = 0.23              # solid volume fraction (Ag/C + carbon)
const eps_G = 0.34              # gas-filled macro-pores
const eps_I = 0.43              # ionomer + liquid combined phase
const alpha_ion = 0.11          # ion diffusivity correction factor
const C_fix = 1000.0           # fixed charge concentration [mol/m³] (1 M)
const S_pore = eps_I / (eps_I + eps_G)   # pore saturation ≈ 0.558

# ─────────────────────────────────────────────────────────────────────────────
# §8.4 Liquid diffusivities at 50°C [m²/s], indexed 1:H⁺ 2:OH⁻ 3:HCO₃⁻ 4:CO₃²⁻ 5:CO₂(aq)
const D_w = [1.33e-8, 8.19e-9, 2.09e-9, 1.63e-9, 3.39e-9]
const D_eff = alpha_ion .* D_w              # §3.1, §8.3
const D_w_Cs   = 3.63e-9                     # Cs⁺ (Stage 5 migration)
const D_eff_Cs = alpha_ion * D_w_Cs
const z_charge = [1.0, -1.0, -1.0, -2.0, 0.0]   # H⁺ OH⁻ HCO₃⁻ CO₃²⁻ CO₂
const z_Cs     = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# §8.6 Buffer chemistry at 50°C
const K_w = 5.48e-8            # (mol/m³)²
const K_2 = 5.16e-4           # mol/m³
const K_3 = 6.74e-8           # mol/m³
const K_4 = K_2 / K_w          # ALWAYS derived
const K_5 = K_3 / K_w          # ALWAYS derived
const k1f = 2.861e3            # mol/m³/s
const k2f = 0.265             # s⁻¹
const k3f = 59.44             # s⁻¹
const k4f = 12.4              # m³/(mol·s)
const k5f = 6.0e6             # m³/(mol·s)
const k1r = k1f / K_w
const k2r = k2f / K_2
const k3r = k3f / K_3
const k4r = k4f / K_4
const k5r = k5f / K_5

# ─────────────────────────────────────────────────────────────────────────────
# §8.7 Gas transport
const M_gas = [44.01e-3, 28.01e-3, 2.016e-3, 18.015e-3]   # CO₂ CO H₂ H₂O [kg/mol]
const v_diff = [26.9, 18.9, 7.07, 12.7]                    # FSG diffusion volumes [cm³/mol]
const x_inlet = [0.88, 0.0, 0.0, 0.12]                     # MOLE fractions (CO₂ CO H₂ H₂O)
const p_total = 101325.0       # Pa
const eps_GDL = 0.52           # Sigracet 39BC porosity
const kappa_GDL = 8.4e-13      # m²  (dry Darcy permeability)
const kappa_CL  = 8.0e-16 * (1.0 - S_pore)^3   # ≈ 7.0e-17 m²
const mu_gas = 1.59e-5         # Pa·s

# inlet mass fractions (mole → mass), §7.3
const M_avg_inlet = sum(x_inlet[j] * M_gas[j] for j in 1:4)
const omega_inlet = [x_inlet[j] * M_gas[j] / M_avg_inlet for j in 1:4]

# ─────────────────────────────────────────────────────────────────────────────
# §8.8 Solid-phase electronic conductivity (GDL + CL)
const sigma_s_GDL = 220.0      # S/m
const sigma_s_CL  = 100.0      # S/m
const sigma_s_eff_GDL = sigma_s_GDL * (1.0 - eps_GDL)^1.5   # ≈ 73 S/m
const sigma_s_eff_CL  = sigma_s_CL  * eps_S^1.5             # ≈ 11 S/m

# ─────────────────────────────────────────────────────────────────────────────
# §8.9 Electrolyte channel / mass transfer (Sherwood-Reynolds)
const L_elec   = 0.0225        # m
const Q_elyte  = 25.0 / 60e6   # m³/s  (25 mL/min)
const A_ch     = 0.8e-3 * 0.8e-3   # m²
const rho_elec = 1000.0        # kg/m³
const mu_elec  = 1.0e-3        # Pa·s
const u_elec = Q_elyte / A_ch
const Re_elec = u_elec * L_elec * rho_elec / mu_elec   # ≈ 14648

# k_MT per species — pass bare-water D_w[i], NOT D_eff (§7.2 warning)
k_MT_func(D_i) = (D_i / L_elec) * 0.664 * sqrt(Re_elec) * (mu_elec / (rho_elec * D_i))^(1/3)
const k_MT = [k_MT_func(D_w[i]) for i in 1:5]

# ─────────────────────────────────────────────────────────────────────────────
# §8.10 Gas-liquid interface
const H_CO2_0 = 34.0           # mol/(m³·atm) at 25°C
const dH_sol  = 2400.0         # K (van't Hoff)
const R_pore  = 25e-9          # m
const a_v_GL  = 3.0 * eps_G / R_pore   # ≈ 4.08e7 m⁻¹
# Sechenov coefficients [from Schumpe 1993]
const h_CO2  = -0.02599
const h_Cs   = 0.0759
const h_OH   = 0.0893
const h_HCO3 = 0.0967
const h_CO3  = 0.1423

# ─────────────────────────────────────────────────────────────────────────────
# §3.3 Volume-fraction-scaled catalyst area + standard reduction potential
const a_v_CL = S_pore * 3.0 * eps_S / r_p   # ≈ 1.54e7 m⁻¹
const c_ref  = 1000.0          # mol/m³
const U_CO_0 = -0.11           # V vs SHE  (CO₂/CO standard reduction, §15.4)
const E_act_HER = 83000.0      # J/mol
const dEact_dpH = 1000.0       # J/(mol·pH)

# ─────────────────────────────────────────────────────────────────────────────
# §15.2 Potential referencing
const pH_anode = 8.0
const nernst_fac = 2.303 * R_gas * T / F_const          # 0.06413 V/pH at 50°C
const V_RHE_to_SHE_offset = nernst_fac * pH_anode        # 0.51305 V

# ─────────────────────────────────────────────────────────────────────────────
# §13/§17.2 Ref-backed mutable kinetic parameters (initial guesses)
const _i0_ref     = Ref{Float64}(540.0)
const _lam_ref    = Ref{Float64}(1.8)
const _her_pf_ref = Ref{Float64}(2.409e13)
const _ac_ref     = Ref{Float64}(0.0872)
const _kGL_ref    = Ref{Float64}(1e-4)

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

# ─────────────────────────────────────────────────────────────────────────────
# §16.1 Experimental data — Lees et al. 2024, Nat. Chem. Eng.
const E_exp_RHE = [-0.816, -1.205, -1.644, -2.079, -2.413]   # V vs RHE@pH8
const i_CO_exp  = [35.1, 189.6, 299.1, 204.3, 64.6]          # mA/cm²
const i_H2_exp  = [29.9, 49.3, 119.6, 449.9, 850.5]          # mA/cm²
const E_exp_SHE = E_exp_RHE .- V_RHE_to_SHE_offset
const FE_CO_exp = i_CO_exp ./ (i_CO_exp .+ i_H2_exp) .* 100
const FE_H2_exp = i_H2_exp ./ (i_CO_exp .+ i_H2_exp) .* 100

# §9.6 Canonical voltage sweep grid (89 points, V vs RHE@pH8)
const V_RHE_grid = collect(range(-0.300, stop = -2.500, length = 89))
const V_SHE_grid = V_RHE_grid .- V_RHE_to_SHE_offset
