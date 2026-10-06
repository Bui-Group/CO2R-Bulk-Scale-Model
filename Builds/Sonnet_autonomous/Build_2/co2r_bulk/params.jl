# params.jl — physical constants, geometry, kinetics (Ref-backed mutables), §8 of the guide.

# ── §8.1 Physical constants ──────────────────────────────────────────────
const F_const = 96485.332       # C/mol
const R_gas   = 8.314462618     # J/(mol K)
const T       = 323.15          # K (50 C)
const k_B_eV  = 8.617333e-5     # eV/K

# ── §15.2 Potential referencing ──────────────────────────────────────────
const pH_anode = 8.0
const nernst_fac = 2.303 * R_gas * T / F_const        # 0.06413 V/pH at 50 C
const V_RHE_to_SHE_offset = nernst_fac * pH_anode      # 0.51305 V

# ── §8.2 Geometry ─────────────────────────────────────────────────────────
const L_CL = 10e-6      # m
const L_GDL = 325e-6    # m
const r_p  = 25e-9      # m, catalyst particle radius

const MESH_N_CL  = parse(Int, get(ENV, "MESH_N_CL", "80"))
const MESH_N_GDL = parse(Int, get(ENV, "MESH_N_GDL", "120"))
const MESH_Q     = 1.3

# ── §8.3 Volume fractions ─────────────────────────────────────────────────
const eps_S   = 0.23
const eps_G   = 0.34
const eps_I   = 0.43
const alpha_ion = 0.11
const C_fix   = 1000.0   # mol/m^3
const S_pore  = eps_I / (eps_I + eps_G)          # pore saturation ~0.558

const a_v_CL  = S_pore * 3 * eps_S / r_p         # ~1.541e7 m^-1
const R_pore  = 25e-9                             # m, characteristic gas-pore radius (== r_p)
const a_v_GL  = 3 * eps_G / R_pore                # ~4.08e7 m^-1

# ── §8.4 Liquid species: order (H+, OH-, HCO3-, CO3^2-, CO2(aq)) ─────────
const z_liq  = [1.0, -1.0, -1.0, -2.0, 0.0]
const D_w    = [1.33e-8, 8.19e-9, 2.09e-9, 1.63e-9, 3.39e-9]     # m^2/s at 50 C
const D_eff  = (eps_I^1.5) * alpha_ion .* D_w                     # Bruggeman + ionomer correction

const z_Cs   = 1.0
const D_w_Cs = 3.63e-9
const D_eff_Cs = (eps_I^1.5) * alpha_ion * D_w_Cs                 # ~1.126e-10 m^2/s

const n_e_val = 2.0   # both CO2R and HER are 2-electron reactions

# ── §8.5 Kinetic parameters — Ref-backed for in-process (re)fitting ──────
const U_CO_0 = -0.11    # V vs SHE, standard reduction potential CO2/CO
const c_ref  = 1000.0   # mol/m^3, MHC normalization reference
const E_act_HER = 83000.0        # J/mol
const dEact_dpH = 1000.0         # J/(mol pH)

const _i0_ref     = Ref{Float64}(540.0)
const _lam_ref    = Ref{Float64}(1.8)
const _her_pf_ref = Ref{Float64}(2.409e13)
const _ac_ref     = Ref{Float64}(0.0872)
const _kGL_ref    = Ref{Float64}(1.0e-4)

i0_CO2R()        = _i0_ref[]
lambda_MHC()     = _lam_ref[]
HER_prefactor()  = _her_pf_ref[]
alpha_c_HER()    = _ac_ref[]
k_GL()           = _kGL_ref[]

set_i0_CO2R!(v)        = (_i0_ref[] = v)
set_lambda_MHC!(v)     = (_lam_ref[] = v)
set_HER_prefactor!(v)  = (_her_pf_ref[] = v)
set_alpha_c_HER!(v)    = (_ac_ref[] = v)
set_k_GL!(v)           = (_kGL_ref[] = v)

# ── §8.6 Buffer chemistry at 50 C ─────────────────────────────────────────
const K_w = 5.48e-8      # (mol/m^3)^2
const K_2 = 5.16e-4      # mol/m^3
const K_3 = 6.74e-8      # mol/m^3
const K_4 = K_2 / K_w
const K_5 = K_3 / K_w

const k1_f = 2.861e3     # mol/m^3/s
const k2_f = 0.265       # 1/s
const k3_f = 59.44       # 1/s
const k4_f = 12.4        # m^3/(mol s)
const k5_f = 6.0e6       # m^3/(mol s)

const k1_r = k1_f / K_w
const k2_r = k2_f / K_2
const k3_r = k3_f / K_3
const k4_r = k4_f / K_4
const k5_r = k5_f / K_5

# ── §8.7 Gas transport parameters. Gas species order: (CO2, CO, H2, H2O) ─
const M_gas      = [44.01e-3, 28.01e-3, 2.016e-3, 18.015e-3]   # kg/mol
const M_gas_gmol = [44.01, 28.01, 2.016, 18.015]                # g/mol (FSG formula only, §6.3)
const v_gas      = [26.9, 18.9, 7.07, 12.7]                      # cm^3/mol, FSG diffusion volumes

const x_inlet = [0.88, 0.0, 0.0, 0.12]     # mole fractions (CO2, CO, H2, H2O)
const p_total = 101325.0                    # Pa

# Mass fractions corresponding to x_inlet (computed once, §7.3)
const M_avg_inlet   = sum(x_inlet[j] * M_gas[j] for j in 1:4)
const omega_inlet   = [x_inlet[j] * M_gas[j] / M_avg_inlet for j in 1:4]

const eps_GDL = 0.52
# eps_G (CL gas void fraction) already defined above as 0.34

const kappa_GDL = 8.4e-13                                   # m^2
const kappa_CL  = 8.0e-16 * (1.0 - S_pore)^3                # m^2, Leverett-style saturation correction
const mu_gas    = 1.59e-5                                    # Pa s (Wilke mixing rule at inlet composition)

# ── §8.8 Solid-phase electronic conductivity ─────────────────────────────
const sigma_s_GDL = 220.0    # S/m
const sigma_s_CL  = 100.0    # S/m
const sigma_s_eff_GDL = sigma_s_GDL * (1 - eps_GDL)^1.5      # ~73 S/m
const sigma_s_eff_CL  = sigma_s_CL  * eps_S^1.5               # ~11 S/m

# ── §8.9 Electrolyte channel / mass transfer parameters ──────────────────
const L_elec   = 0.0225                 # m
const Q_elyte  = 25.0 / 60e6            # m^3/s (25 mL/min)
const A_ch     = 0.8e-3 * 0.8e-3        # m^2
const rho_elec = 1000.0                 # kg/m^3
const mu_elec  = 1.0e-3                 # Pa s

const u_elec = Q_elyte / A_ch
const Re_elec = u_elec * L_elec * rho_elec / mu_elec

function k_MT_species(D_i::Real)
    return (D_i / L_elec) * 0.664 * sqrt(Re_elec) * (mu_elec / (rho_elec * D_i))^(1/3)
end
# Per-species k_MT computed from D_w (bare-water diffusivity), NOT D_eff — see §7.2 warning.
const k_MT = [k_MT_species(D_w[i]) for i in 1:5]

# ── §8.10 Gas-liquid interface parameters ────────────────────────────────
const H_CO2_25C = 34.0          # mol/(m^3 atm) at 25 C
const dH_sol    = 2400.0        # K, van't Hoff sensitivity
# Sechenov coefficients (species order matches liquid DOFs where relevant)
const h_CO2  = -0.02599
const h_Cs   =  0.0759
const h_OH   =  0.0893
const h_HCO3 =  0.0967
const h_CO3  =  0.1423

# ── §9.5.7 Production solver settings ────────────────────────────────────
const RTOL_PROD      = 1e-8
const F_INF_TOL       = 1e-6
const ALPHA_INIT_ANCHOR = 1e-2
const ALPHA_INIT_SWEEP  = 1.0
const ALPHA_MIN        = 1e-8
const MAX_ITER_SWEEP   = 40
const MAX_ITER_ANCHOR  = 200
const MAX_LOG_STEP     = 5.0
const AFRAC            = 1e-3
const DV_STEP          = 0.025     # V, matches the grid spacing

# ── §9.5.5 Canonical voltage grid ─────────────────────────────────────────
const V_RHE_grid = collect(range(-0.300, stop = -2.500, length = 89))
const V_SHE_grid = V_RHE_grid .- V_RHE_to_SHE_offset

# ── §16.1 Experimental data — Romiluyi et al. 2023 ───────────────────────
const E_exp_RHE = [-0.816, -1.205, -1.644, -2.079, -2.413]
const i_CO_exp  = [35.1, 189.6, 299.1, 204.3, 64.6]
const i_H2_exp  = [29.9, 49.3, 119.6, 449.9, 850.5]
const E_exp_SHE = E_exp_RHE .- V_RHE_to_SHE_offset
const FE_CO_exp = i_CO_exp ./ (i_CO_exp .+ i_H2_exp) .* 100
const FE_H2_exp = i_H2_exp ./ (i_CO_exp .+ i_H2_exp) .* 100
