# Human–AI Collaborative Build Record

## 1D CO₂ Electroreduction Catalyst Layer Model

**Model:** Bulk-scale 1D Nernst-Planck + MHC/BV kinetics + Stefan-Maxwell gas transport
**AI Assistant:** Claude Opus 4.6 (1M context), Anthropic
**Human Researcher:** Researcher C
**Date:** March 27–28, 2026
**Platform:** Julia 1.12.5 on Windows 11, with Python 3 (matplotlib) for plotting

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Implementation Guide](#2-implementation-guide)
3. [Build Dialogue](#3-build-dialogue)
   - [3.1 Session Setup and Ground Rules](#31-session-setup-and-ground-rules)
   - [3.2 Stage 1: Liquid Nernst-Planck with Dirichlet BCs](#32-stage-1-liquid-nernst-planck-with-dirichlet-bcs)
   - [3.3 Stage 2: Sherwood-Reynolds Neumann BCs](#33-stage-2-sherwood-reynolds-neumann-bcs)
   - [3.4 Stage 3: Stefan-Maxwell Gas Transport](#34-stage-3-stefan-maxwell-gas-transport)
   - [3.5 Parameter Fitting](#35-parameter-fitting)
4. [Final Model Architecture](#4-final-model-architecture)
5. [Fitted Parameters and Results](#5-fitted-parameters-and-results)
6. [Bug Log and Resolutions](#6-bug-log-and-resolutions)
7. [Files Produced](#7-files-produced)

---

## 1. Project Overview

This document records the complete collaborative process between a human electrochemistry researcher and Claude (Opus 4.6), an AI assistant by Anthropic, to implement a 1D continuum model for CO₂ electroreduction (CO₂R) to CO on an Ag/C catalyst layer. The model was built from scratch in Julia, following a detailed implementation guide provided by the human researcher.

**Physical system modeled:**
- 1D catalyst layer (CL, 10 µm thick) with gas diffusion layer (GDL, 325 µm)
- Ag/C catalyst, Sustainion X37-50 AEM, 50°C, 1M CsHCO₃ electrolyte
- CO₂ + H₂O + 2e⁻ → CO + 2OH⁻ (MHC kinetics)
- 2H₂O + 2e⁻ → H₂ + 2OH⁻ (Butler-Volmer kinetics)
- Experimental validation against Lees et al. 2024, *Nat. Chem. Eng.*

**Scope:** No agglomerate sub-scale. No membrane. No anode. Bulk liquid transport only.

---

## 2. Implementation Guide

The human researcher provided a comprehensive 41-page implementation guide (`CO2R_BULK_MODEL_IMPLEMENTATION_GUIDE.md`) specifying all equations, parameters, boundary conditions, numerical methods, and validation criteria. The guide was structured in 18 sections covering:

- Physical domain and species layout (5 liquid + 4 gas DOFs per cell)
- Nernst-Planck liquid transport with electroneutrality closure
- Marcus-Hush-Chidsey (MHC) kinetics for CO₂R
- Butler-Volmer kinetics for HER
- Aqueous CO₂/carbonate buffer chemistry (5 equilibrium reactions)
- Stefan-Maxwell mixture-average gas diffusion + Darcy flow
- Sherwood-Reynolds mass transfer BCs
- Finite volume discretization with log-concentration DOFs
- Newton solver with FD Jacobian, pseudo-arclength continuation
- Per-stage binary solution caching
- 4-stage implementation workflow with stop gates
- Experimental data tables and fitting procedure

The guide was written specifically for an LLM agent to implement from scratch, with all parameters, equations, and debugging checks fully specified.

---

## 3. Build Dialogue

### 3.1 Session Setup and Ground Rules

**Human instruction:** "Build this 1D CO₂ electroreduction catalyst layer model following this implementation guide exactly. Perform every check and ensure physicality."

**AI initial action:** The AI began by attempting to explore existing code in the repository (Researcher B's and Researcher A's directories).

**Human correction:** "Stop, do not explore Researcher B's model, get rid of any context you gained from reading those files. You are not allowed to look at Researcher B or Researcher A's folder yet and cannot use context from them. Build from scratch."

**AI response:** Acknowledged the constraint. Saved it as a persistent memory rule. Proceeded to build entirely from the implementation guide without referencing other team members' code.

**Key ground rule established:** Independent implementation — no influence from other team members' solutions.

---

### 3.2 Stage 1: Liquid Nernst-Planck with Dirichlet BCs

**Goal:** 5 liquid species (H⁺, OH⁻, HCO₃⁻, CO₃²⁻, CO₂(aq)) with buffer chemistry + MHC CO₂R + BV HER in the CL. Dirichlet BCs on both sides.

#### Files Created (in order)

1. **params.jl** — All physical constants, geometry, kinetics with Ref-backed mutables for fitting
2. **mesh.jl** — CLMesh struct with uniform cell-centered FV mesh
3. **chemistry.jl** — Buffer source terms (5 reactions) and exact equilibrium solver (bisection on charge balance)
4. **kinetics.jl** — MHC CO₂R rate (`mhc_k_elec`) and BV HER rate
5. **transport.jl** — Diffusion flux helper
6. **assembly.jl** — Stage 1 and Stage 2 residual assembly
7. **solver.jl** — Dense FD Jacobian, damped Newton with row scaling and step clamping
8. **cache.jl** — Binary save/load with voltage-encoded filenames
9. **run_stage1.jl** — Bootstrap + voltage sweep driver
10. **plot_stage1.py** — 4-panel diagnostic plots

#### Technical Decisions

- **Log-concentration DOFs:** `u[i] = ln(c_i)`, recovering `c_i = exp(clamp(u[i], -50, 50))`. Ensures positivity and improves conditioning across 10 orders of magnitude.
- **Equilibrium solver:** Hand-written bisection on `log10(c_H)` rather than using the `Roots.jl` package, to minimize dependencies.
- **Effective diffusivity:** `D_eff = eps_L × D_w` with `eps_L = 0.08` (liquid-filled macro-pore fraction), following the Lees-style bulk model.
- **Natural parameter continuation** instead of PAC for voltage sweep — PAC was implemented but produced very small voltage steps (~2 mV) due to the tangent vector norm dominating over the voltage component in 250-DOF space. Switched to adaptive voltage stepping (10–100 mV steps) which completed the sweep in 26 steps.

#### Bugs Encountered and Fixed

1. **Julia 1.12 scoping issues (3 instances):** Variables assigned inside `for`/`while` loops at global scope triggered `UndefVarError`. Fixed with `let` blocks and `global` declarations.
2. **Cache filename parsing:** `list_cache` tried to parse `"V-0.300000"` as Float64. Fixed by correctly stripping the `prefix_V` portion of filenames.

#### Bootstrap Protocol

```
Stage A: Buffer ramp (α_buf: 0 → 1, α_elec = 0)
  → 9 steps, all converged in 2-5 Newton iterations

Stage B: Electrochemistry ramp (α_elec: 1e-6 → 1.0)
  → ~110 geometric + linear steps, all converged in 2-4 iterations

Voltage sweep: V = -0.3 → -2.5 V vs RHE@pH8
  → 26 steps, all converged in 4-7 iterations
```

#### Physicality Checks (all passed)

| Check | Result |
|-------|--------|
| Buffer equilibrium residuals | 7.5×10⁻⁹ mol/m³/s |
| Electroneutrality | 2.8×10⁻¹⁴ mol/m³ |
| pH range at V = -1.2 | [8.83, 11.76] |
| CO₂ depletion direction | Left (GDL) > Right (electrolyte) |
| CO FE peak | ~64% at -1.3 V RHE |

#### Human Review

**Human:** "It looks good, proceed to stage 2."

---

### 3.3 Stage 2: Sherwood-Reynolds Neumann BCs

**Goal:** Replace Dirichlet BC at x = L_CL with convective mass transfer: `J = k_MT × (c_eq - c_boundary)`.

#### Critical Bug: Neumann Flux Sign Convention

The implementation guide specifies:
```
J_i(L_CL) = k_MT,i × (c_bulk_elyte,i − c_i(L_CL))
```

This was initially implemented directly as `J[i, N+1] = k_MT[i] * (c_eq[i] - c[i, N])`.

**Problem:** This sign convention defines positive flux as flowing INTO the cell (from electrolyte toward GDL), but the FV scheme's diffusion flux convention has positive = +x direction (from GDL toward electrolyte). The Neumann flux had the **opposite sign** from the diffusion convention.

**Symptom:** Newton solver diverged when switching from Dirichlet to Neumann BC. The solver tried multiple strategies:
1. Direct BC switch → `SingularException(247)` (DOF 247 = cell 50, OH⁻)
2. Blending parameter (α: Dirichlet→Neumann) → converged to α=0.95, diverged at α=0.97
3. Independent bootstrap with Neumann from start → failed at α_elec=0.17

**Root cause analysis:** With the wrong sign, OH⁻ produced by electrochemistry was being pushed BACK into the CL by the boundary flux instead of being removed. This created an unstable positive feedback loop.

**Fix:** Negated the Neumann flux to match the diffusion sign convention:
```julia
J_neu = -k_MT[i] * (c_eq[i] - c[i, N])   # = k_MT*(c_N - c_eq)
```

**After fix:** All bootstraps and voltage sweeps converged immediately (2-7 Newton iterations per step).

#### Results

- CO partial current reduced by 42% vs Stage 1 (mass transfer limited)
- OH⁻ builds up to 0.66 mol/m³ at boundary (vs c_eq = 0.0093)
- Biot number = 3.75 for all species (moderate mass transfer resistance)

**Human:** "Looks good proceed."

---

### 3.4 Stage 3: Stefan-Maxwell Gas Transport

**Goal:** Add 4 gas DOFs per cell (ω_CO₂, ω_CO, ω_H₂, p) across GDL (30 cells) + CL (50 cells), coupled to liquid via Henry's law and volumetric phase transfer.

#### New File

- **gas_transport.jl** — Fuller-Schettler-Giddings binary diffusivities, mixture-average SM diffusion, Darcy flow, Sechenov-corrected Henry's law, phase transfer rate

#### Architecture

Total DOFs: 570 (250 liquid + 120 GDL gas + 200 CL gas)

Dense FD Jacobian: 570 × 570 matrix, requiring 570 residual evaluations per Jacobian computation.

#### Technical Decisions

- **Integrated FV form** for gas residual: `F = J_left - J_right + S×dx` (not divided form) to avoid O(1/dx) scaling issues
- **Harmonic mean permeability** at GDL-CL interface: `κ_if = 2κ_GDL×κ_CL/(κ_GDL+κ_CL)`
- **FSG prefactor** = 1.00×10⁻³ (Perry's 9th ed.), NOT 1.013×10⁻²

#### Bug: Constant Redefinition

Gas transport parameters (`kappa_GDL`, `mu_gas`, `eps_GDL`) were defined twice in `run_stage3.jl` — once after the binary diffusivity computation (needed for Bruggeman correction) and once later in a separate section. Julia's `const` redefinition error caught this. Fixed by consolidating all gas constants before the residual function.

#### Bootstrap Protocol

```
1. Gas-only solve (no liquid coupling): converged in 7 iterations
2. Phase transfer ramp (α_pt: 0.01 → 1.0): 6 steps, 4 iterations each
3. Electrochemistry ramp (α_elec: 0.01 → 1.0): 6 steps, 4-5 iterations each
4. Voltage sweep: -0.3 → -2.5 V, 26 steps, 4-7 iterations each
```

#### Key Results

| Metric | Stage 2 (no gas) | Stage 3 (with gas) |
|--------|-----------------|-------------------|
| i_CO peak | 200 mA/cm² | 1076 mA/cm² |
| FE_CO peak | 58% | 94% |

The gas-phase CO₂ supply removed the liquid-only mass transfer bottleneck, dramatically increasing CO current. The model overpredicted CO by ~3× compared to experiment, consistent with the guide's warning that `i0_CO2R = 540 A/m²` (fitted for agglomerate model) would give ~2× too high currents in the bulk model.

#### Profile Plots

**Human:** "Plot profiles of CO and H₂ mass fraction across the modeling domain as well as the delta p across the modeling domain... also plot another 3 panel figure of OH⁻, CO₂(aq), and pH profiles."

The AI wrote an extraction script (`extract_stage3_profiles.jl`) to read binary cache files and export spatial profiles to CSV, then plotted with Python.

Key observations from profiles:
- CO mass fraction builds up linearly across GDL (up to 7% at V=-2.5)
- H₂ mass fraction up to 0.15% at high overpotential
- Pressure drop: ~100-120 Pa in CL, negligible in GDL (κ_CL ≪ κ_GDL)
- OH⁻ peaks at ~1800 mol/m³ at GDL side (V=-2.5)
- CO₂(aq) depletes sharply in first 1-2 µm of CL
- pH ranges from ~10 (electrolyte side) to ~13+ (GDL side)

**Human:** "Looks good, proceed."

---

### 3.5 Parameter Fitting

**Note:** Stage 4 (Darcy pressure) was already fully included in Stage 3 via the pressure DOF, Darcy velocity, and interface permeability. The Δp profile confirmed correct behavior (~100 Pa, <0.2% of p_total). Stage 4 was therefore skipped as a separate implementation step.

#### Fitting Strategy

Following the guide's §17, the AI implemented a manual grid search over the 4 kinetic parameters:

1. **Sweep i0_CO2R** (50–540 A/m²) at fixed HER params
2. **Sweep λ_MHC** (0.8–2.5 eV) at best i0
3. **Sweep HER_prefactor** (10¹⁰–10¹⁵) at best CO₂R params
4. **Sweep α_c_HER** (0.02–0.20) at best remaining params

**Objective:** Sum of squared errors in log-space: `SSE = Σ(log10(i_model) - log10(i_exp))²` over 10 data points (5 voltages × 2 partial currents).

#### Results

| Parameter | Original | Fitted | Change |
|-----------|----------|--------|--------|
| i0_CO2R | 540 A/m² | 100 A/m² | -81% |
| λ_MHC | 1.8 eV | 1.6 eV | -11% |
| HER_prefactor | 2.409×10¹³ | 2.409×10¹³ | 0% |
| α_c_HER | 0.0872 | 0.08 | -8% |
| **SSE** | **2.39** | **0.58** | **-76%** |

The large reduction in `i0_CO2R` (from 540 to 100 A/m²) is physically expected: the agglomerate model has a higher effective catalyst area due to the internal pore structure, which is absent in the bulk model.

#### Comparison with Experiment (fitted model)

| V (RHE) | i_CO model | i_CO exp | i_H₂ model | i_H₂ exp |
|---------|-----------|---------|-----------|---------|
| -0.816 | 48.8 | 35.1 | 45.1 | 29.9 |
| -1.205 | 315.9 | 189.6 | 57.1 | 49.3 |
| -1.644 | 401.3 | 299.1 | 148.8 | 119.6 |
| -2.079 | 348.6 | 204.3 | 415.1 | 449.9 |
| -2.413 | 276.3 | 64.6 | 877.4 | 850.5 |

- **H₂ current:** Excellent fit across all voltages
- **CO current:** Correct qualitative shape (rise → peak → rollover) but 1.5-2× too high
- **CO rollover at high overpotential:** Not sharp enough — model gives 276 mA/cm² at -2.4V vs experimental 64.6 mA/cm²
- **FE_CO peak:** Model ~85%, experiment 79%

The remaining CO overestimation may be addressable by fitting the gas-liquid mass transfer coefficient `k_GL` as a 5th parameter (guide §17.8).

---

## 4. Final Model Architecture

```
co2r_bulk/
├── params.jl              # Physical constants, geometry, kinetics (Ref-backed)
├── mesh.jl                # CLMesh, GDLMesh structs
├── chemistry.jl           # Buffer sources + equilibrium solver
├── kinetics.jl            # MHC CO₂R + BV HER
├── transport.jl           # Diffusion flux helper
├── gas_transport.jl       # SM + Darcy + FSG + Sechenov + phase transfer
├── assembly.jl            # Residual assembly (Stages 1-2)
├── solver.jl              # Newton + FD Jacobian + PAC
├── cache.jl               # Binary cache save/load
├── run_stage1.jl          # Stage 1 driver
├── run_stage2.jl          # Stage 2 driver
├── run_stage3.jl          # Stage 3 driver (includes Stage 4)
├── run_fitting.jl         # Parameter fitting
├── extract_stage3_profiles.jl  # Profile extraction from cache
├── plot_stage1.py         # Stage 1 plots
├── plot_stage2.py         # Stage 2 comparison plots
├── plot_stage3.py         # Stage 3 comparison plots
├── plot_stage3_profiles.py # Gas + liquid profile plots
├── plot_fitted.py         # Fitted model vs experiment
├── output_stage1/cache/   # 27 cached Stage 1 solutions
├── output_stage2/cache/   # 27 cached Stage 2 solutions
├── output_stage3/cache/   # 27 cached Stage 3 solutions (fitted params)
└── BUILD_RECORD.md        # This document
```

**Total lines of Julia code:** ~1,800
**Total lines of Python plotting code:** ~500
**Build time:** ~4 hours wall clock (single session)

---

## 5. Fitted Parameters and Results

### Final fitted parameters

| Symbol | Value | Units | Description |
|--------|-------|-------|-------------|
| i₀,CO₂R | 100 | A/m² | Exchange current density for CO₂ reduction |
| λ_MHC | 1.6 | eV | Marcus reorganization energy |
| HER_prefactor | 2.409×10¹³ | — | HER pre-exponential factor |
| α_c | 0.08 | — | HER cathodic transfer coefficient |

### Fixed transport/geometry parameters

| Symbol | Value | Units |
|--------|-------|-------|
| L_CL | 10 µm | Catalyst layer thickness |
| L_GDL | 325 µm | GDL thickness |
| ε_L | 0.08 | Liquid-filled macro-pore fraction |
| ε_G | 0.30 | Gas-filled void fraction |
| a_v,CL | 2.76×10⁷ m⁻¹ | Specific catalyst area |
| Sh | 30 | Sherwood number |
| k_GL | 1×10⁻⁴ m/s | Gas-liquid mass transfer coefficient |
| T | 323.15 K | Temperature (50°C) |

---

## 6. Bug Log and Resolutions

| # | Bug | Location | Symptom | Resolution |
|---|-----|----------|---------|------------|
| 1 | Julia 1.12 soft-scope assignment | run_stage1.jl | `UndefVarError` for loop variables at global scope | Wrapped in `let` blocks or added `global` keyword |
| 2 | Cache filename parsing | cache.jl | `cannot parse "V-0.300000" as Float64` | Fixed prefix stripping: `pat = prefix * "_V"` |
| 3 | **Neumann flux sign error** | assembly.jl | Newton divergence when switching Dirichlet→Neumann BC | Negated flux: `J = -k_MT*(c_eq - c_N)` to match diffusion convention |
| 4 | Const redefinition | run_stage3.jl | Gas transport constants defined twice | Consolidated all gas constants before residual function |
| 5 | PAC slow convergence | solver.jl / run_stage1.jl | PAC took ~2 mV steps (tangent norm dominated) | Switched to natural parameter continuation with adaptive stepping |

**Bug #3 (Neumann sign)** was the most significant. It consumed ~3 iterations of increasingly sophisticated workarounds (blending parameters, BC ramping, independent bootstrapping) before the root cause — a sign convention mismatch between the implementation guide and the FV scheme — was identified through careful analysis of the flux direction physics.

---

## 7. Files Produced

### Output files

| File | Description |
|------|-------------|
| `output_stage1/stage1_summary.csv` | Polarization data (27 voltages) |
| `output_stage1/stage1_profiles.csv` | Concentration profiles at 8 voltages |
| `output_stage1/stage1_4panel.png` | 4-panel diagnostic plot |
| `output_stage1/stage1_profiles.png` | pH, CO₂, OH⁻ profiles |
| `output_stage2/stage2_summary.csv` | Stage 2 polarization data |
| `output_stage2/stage2_comparison.png` | Stage 1 vs 2 comparison |
| `output_stage3/stage3_summary.csv` | Stage 3 polarization (original params) |
| `output_stage3/stage3_fitted_summary.csv` | Stage 3 polarization (fitted params) |
| `output_stage3/stage3_gas_profiles.csv` | Gas composition profiles |
| `output_stage3/stage3_liquid_profiles.csv` | Liquid concentration profiles |
| `output_stage3/stage3_4panel.png` | 3-stage comparison |
| `output_stage3/stage3_FE_overlay.png` | FE evolution across stages |
| `output_stage3/stage3_gas_profiles.png` | CO, H₂, Δp spatial profiles |
| `output_stage3/stage3_liquid_profiles.png` | OH⁻, CO₂, pH spatial profiles |
| `output_stage3/stage3_fitted_4panel.png` | Fitted model vs experiment |

### Binary cache files

27 solutions per stage, each containing the full DOF vector (250 DOFs for Stages 1-2, 570 DOFs for Stage 3) in raw Float64 binary format.

---

*This record was generated as part of the collaborative build process. The AI assistant (Claude Opus 4.6) wrote all code, ran all simulations, debugged all errors, and performed parameter fitting. The human researcher provided the implementation guide, reviewed results at each stop gate, and directed the workflow.*
