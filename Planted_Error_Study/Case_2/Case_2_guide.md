# CO₂R Bulk-Scale Julia Model — Implementation Guide

**Scope:** 1D Catalyst Layer (CL) with Nernst-Planck liquid transport (diffusion + migration), MHC kinetics for CO₂R and Butler-Volmer for HER, Stefan-Maxwell + Darcy gas transport, and Sherwood-Reynolds mass transfer at the electrolyte boundary. **No membrane. No anode.**

This guide is written for an LLM agent to implement from scratch. All parameters, equations, boundary conditions, numerical methods, and debugging checks are specified completely.

---

## Table of Contents
1. [Physical Domain](#1-physical-domain)
2. [Species and DOF Layout](#2-species-and-dof-layout)
3. [Governing Equations — Liquid Phase](#3-governing-equations--liquid-phase)
4. [Electrochemical Kinetics](#4-electrochemical-kinetics)
5. [Buffer Chemistry](#5-buffer-chemistry)
6. [Gas-Phase Transport — Stefan-Maxwell + Darcy](#6-gas-phase-transport--stefan-maxwell--darcy)
7. [Boundary Conditions by Stage](#7-boundary-conditions-by-stage)
8. [Complete Parameter Tables](#8-complete-parameter-tables)
9. [Numerical Methods](#9-numerical-methods)
10. [Solution Caching — Per-Stage Binary Cache](#10-solution-caching--per-stage-binary-cache)
11. [Implementation Stages](#11-implementation-stages)
12. [Physicality Checks and Debug Plots](#12-physicality-checks-and-debug-plots)
13. [Module Structure](#13-module-structure-recommended)
14. [Common Pitfalls and Solutions](#14-common-pitfalls-and-solutions)
15. [Potential Referencing — Nernst Shift, RHE, SHE](#15-potential-referencing--nernst-shift-rhe-she)
16. [Experimental Data Tables — Lees et al. 2024](#16-experimental-data-tables--lees-et-al-2024)
17. [Fitting Strategy and Procedure](#17-fitting-strategy-and-procedure)
18. [Experimental Comparison Plots](#18-experimental-comparison-plots-required)

---

## 1. Physical Domain

```
 FLOW-FIELD / GDL               CATALYST LAYER (CL)               ELECTROLYTE
 (gas feed)           |                                   |       (flowing)
                      |                                   |
 CO₂/H₂O inlet       |   Liquid NP transport             |   Sherwood-Reynolds
 (Dirichlet p, ω)     |   MHC CO₂R + BV HER kinetics     |   k_MT Neumann BC
                      |   SM+Darcy gas transport          |
  x = 0              x = 0                              x = L_CL
```

The CL occupies `x ∈ [0, L_CL]`. The gas-diffusion layer (GDL) is `x ∈ [-L_GDL, 0]`.

- **Gas feed side (x = 0, GDL/CL interface):** Dirichlet for gas (known composition from GDL); for liquid, CO₂ partial pressure determines CO₂(aq) via Henry's law; ions: no-flux.
- **Electrolyte side (x = L_CL):** Sherwood-Reynolds mass transfer Neumann BC for all liquid species (Stage 2+). Dirichlet in Stage 1.

---

## 2. Species and DOF Layout

### Liquid species (5 DOFs per cell)
| Index | Species | Charge z |
|-------|---------|----------|
| 1 | H⁺ | +1 |
| 2 | OH⁻ | −1 |
| 3 | HCO₃⁻ | −1 |
| 4 | CO₃²⁻ | −2 |
| 5 | CO₂(aq) | 0 |

Cs⁺ is NOT a DOF — it is computed from electroneutrality at every cell:
```
c_Cs = c_OH + c_HCO3 + 2*c_CO3 − c_H − C_fix × ε_I
```
where `C_fix = 1000 mol/m³` (1 M) is the fixed positive charge concentration of the ionomer and `ε_I = 0.43` is the ionomer+liquid volume fraction. The `−C_fix × ε_I` term accounts for the immobile cationic groups in the anion-exchange ionomer: more fixed positive charge requires *less* free Cs⁺ to satisfy local electroneutrality.

**Cs⁺ is recovered from electroneutrality, not transported.** This is OK physically because Cs⁺ is conserved on the cell scale: any local accumulation or depletion is balanced by an equal-and-opposite Cs⁺ flux from the anolyte side. The model captures the right pseudo-steady-state Cs⁺ distribution (creation/destruction is forbidden, only redistribution is allowed). This is a standard simplification for bulk-scale half-cell models with a large bulk Cs⁺ reservoir.

### Gas species (4 DOFs per cell, added in Stage 3)
| Index | Species | DOF? |
|-------|---------|------|
| 1 | CO₂ | ω₁ (mass fraction) |
| 2 | CO | ω₂ (mass fraction) |
| 3 | H₂ | ω₃ (mass fraction) |
| 4 | H₂O | closure: ω₄ = 1 − ω₁ − ω₂ − ω₃ |

Gas pressure `p` is the 4th DOF per cell (ω₁, ω₂, ω₃, p).

### Global DOF vector (Stages 3–4 — with gas DOFs, before potentials)
```
u = [  lnc_H(1), lnc_OH(1), lnc_HCO3(1), lnc_CO3(1), lnc_CO2(1),   # cell 1 liquid (5 DOFs)
       lnc_H(2), ..., lnc_CO2(2),                                     # cell 2 liquid
       ...
       lnc_H(N_CL), ..., lnc_CO2(N_CL),                              # last liquid cell
       ω_CO2(1), ω_CO(1), ω_H2(1), p(1),                             # GDL cell 1 gas (4 DOFs)
       ...
       ω_CO2(N_GDL), ..., p(N_GDL),                                   # last GDL cell
       ω_CO2(1)_cl, ..., p(1)_cl,                                     # CL cell 1 gas
       ...
       ω_CO2(N_CL)_cl, ..., p(N_CL)_cl ]                             # last CL gas cell
```

### Global DOF vector (Stage 5 — full electrochemistry, adds φ_s and φ_L)

Stage 5 appends `N_GDL + N_CL` solid potential DOFs (φ_s on the entire GDL+CL domain) and `N_CL` liquid potential DOFs (φ_L on the CL only — no liquid in GDL):
```
u_stage5 = [ <Stage 4 layout above> ;
             φ_s_gdl(1), ..., φ_s_gdl(N_GDL),                          # solid potential GDL
             φ_s_cl(1), ..., φ_s_cl(N_CL),                              # solid potential CL
             φ_L_cl(1), ..., φ_L_cl(N_CL) ]                             # liquid potential CL
```
Total Stage 5 DOFs at the **production default** N_CL=80, N_GDL=120: `5×80 + 4×(120+80) + 120 + 80 + 80 = 400 + 800 + 120 + 80 + 80 = 1480`. (At the development-only 20/30 mesh: 370 DOFs.)

**Use log-concentrations for liquid DOFs.** Store `u[i] = ln(c_i)` and recover `c_i = exp(u[i])`. This ensures positivity automatically and improves conditioning — concentration spans ~10 orders of magnitude from H⁺ (≈10⁻⁵ mol/m³) to HCO₃⁻ (≈980 mol/m³); CO₂(aq) at the GDL boundary is ~12.2 mol/m³ from the combined T+Sechenov Henry's law at 50°C with p_CO₂ = 0.88 atm and 1 M CsHCO₃ ion composition (see §7.4 for the derivation; the van 't Hoff-only value of ~16.0 mol/m³ applies in Stages 1–2 where Sechenov is not yet enabled — Sechenov first turns on in Stage 3 along with the gas DOFs).

---

## 3. Governing Equations — Liquid Phase

### 3.1 Nernst-Planck Transport (diffusion + migration with electroneutrality closure)

For species i at steady state, the FV conservation law per cell is:

```
(J_i,right − J_i,left) / dx = S_i(c)
```

where `J_i` is the full Nernst-Planck flux [mol/m²/s]:
```
J_i = −D_eff,i × ∇c_i − (z_i F)/(RT) × D_eff,i × c_i × ∇φ_L
```

**The liquid potential φ_L is a key variable in the model.** It is NOT set to zero.

**Electroneutrality closure for φ_L:** The Poisson equation is replaced by the electroneutrality constraint
```
Σᵢ zᵢ cᵢ + c_Cs + C_fix × ε_I = 0
```
where the sum runs over the 5 transported species (H⁺, OH⁻, HCO₃⁻, CO₃²⁻, CO₂(aq); CO₂ has z=0), `c_Cs` is the (positive) free Cs⁺, and `C_fix × ε_I` is the (positive) immobile anion-exchange fixed charge. Solving for c_Cs:
```
c_Cs = c_OH + c_HCO3 + 2 c_CO3 − c_H − C_fix × ε_I
```
which matches §2 line 63 exactly. Cs⁺ is recovered from electroneutrality (not a transported DOF). The 5 transported species + electroneutrality define the ionic chemistry; `φ_L` is then a separate DOF whose residual is the **ionic charge balance** at each cell.

**φ_L is a DOF, not a derived/Picard quantity.** Treat φ_L as N_CL extra cell-centered unknowns and add N_CL extra residual rows. The residual at each cell is the local charge balance: the divergence of the ionic current density `i_L(x) = F × Σ_all zⱼ Jⱼ(x)` equals the volumetric faradaic source:
```
∂i_L/∂x = −n_e × F × (R_CO2R + R_HER)
```
Discretized as a cell-centered FV:
```
F_phi_L[ix] = (i_L_face[ix+1] − i_L_face[ix]) + n_e × F × (R_CO2R + R_HER)[ix] × dx[ix]
```
where the face current is computed from the NP flux at that face:
```
i_L_face[f] = F × Σᵢ∈{H⁺,OH⁻,HCO3⁻,CO3²⁻,CO2(aq)} zᵢ × J_i_face[f]
J_i_face[f]   = −Dᵢ × (cᵢ[ix_R] − cᵢ[ix_L])/dx_between − (zᵢ F/RT) × Dᵢ × c̄ᵢ × (φ_L[ix_R] − φ_L[ix_L])/dx_between
```
Cs⁺ is recovered algebraically from the electroneutrality closure (§2) — it is not a transported DOF and does not contribute its own per-face NP flux to the liquid current sum above. The five-species sum together with the φ_L gauge fix (§3.1, below) closes the ionic system.
**Boundary conditions for φ_L:**
- **Left (x = 0, GDL/CL interface)** — **natural Neumann** `i_L_face[1] = 0`. This is automatically satisfied by the ion no-flux BC in the species residuals (`J_H⁺ = J_OH⁻ = J_HCO3⁻ = J_CO3²⁻ = 0` at face 1, CO₂ has `z = 0`), so `i_L_face[1] = F × Σᵢ zᵢ J_i,face[1] = 0`. **Do not add a separate equation** — the BC lives inside cell 1's charge-balance residual as boundary data.
- **Right (x = L_CL, electrolyte)** — **Dirichlet override** `φ_L[N_CL] = 0`. This replaces the cell-N_CL charge-balance residual with a single row `F[phi_L_cl_idx(N_CL)] = phi_L_cell[N_CL]`. It is **necessary for well-posedness** because the per-cell φ_L residual is algebraically `F_φL[ix] = −F × Σᵢ zᵢ × F_species[ix]` at every cell (buffer reactions conserve charge pairwise, faradaic reactions contribute `−n_e × R` to Σ zᵢ × source_i), so the φ_L DOFs have exactly one null direction — a uniform shift of the whole φ_L field. The Dirichlet override kills that null direction and fixes the gauge. **Without the override the Jacobian is rank-deficient by 1 and Newton cannot converge.** Use an override, not an added equation.

The Newton solver handles the c↔φ_L coupling monolithically — **no Picard iteration is needed**. The flux `J_i_face[f]` includes both diffusion (depends on c) and migration (depends on φ_L), so the Jacobian column for each φ_L DOF picks up nonzero entries in both the species residual rows and the φ_L residual row. The five species residuals and the φ_L residual together close the system.

Alternative explicit form for `∇φ_L` (algebraic, **for diagnostic display only** — do NOT use as a warm-start; the §9.5.5 anchor-at-V_c=0 continuation supplies the Stage 5 initial guess and is much easier to implement), obtained by setting `Σᵢ zᵢ Jᵢ = i_L/F` and solving for ∇φ_L:
```
∇φ_L|_face = −(i_L_face/F + Σᵢ zᵢ Dᵢ ∇cᵢ|_face) / ((F/(RT)) × Σᵢ zᵢ² Dᵢ c̄ᵢ|_face)
```
Note the **leading minus** and the `+` between the two terms in the numerator (this is the standard Henderson diffusion potential plus the ohmic drop term — both have the same sign relative to the conductivity denominator).

**Liquid potential in the CL:** The liquid potential φ_L is also needed to compute the effective overpotential for electrochemical reactions. The electrode overpotential is:

```
η = (φ_s − φ_L) − U_eq
```

where `φ_s` is the solid (electron-conducting) phase potential and `φ_L` is the local liquid potential. Previously, with `φ_L = 0`, the overpotential was simply `η = φ_s − U_eq`. Now, `φ_L` must be subtracted from `φ_s` before computing the overpotential.

**No separate ionic conductivity parameter.** The model does NOT carry an independent `σ_CL` constant. Ionic conduction emerges automatically from the Nernst-Planck migration term — the per-species effective diffusivities `D_eff,i` and the local concentrations `cᵢ(x)` together determine all transport, including the ionic current. If you want to *report* an effective conductivity for diagnostics, it is the Nernst-Einstein expression evaluated at the local composition:
```
σ_eff_CL,local = (F²/(RT)) × Σᵢ∈{H⁺,OH⁻,HCO3⁻,CO3²⁻} zᵢ² × D_eff,i × cᵢ(x)        [diagnostic only]
```
But this never enters the residual — it is fully equivalent to what the NP fluxes already do, and using both would double-count.

> **Δφ_L sanity check — always run this after a Stage 5 solve.** The single most important post-solve diagnostic for the Stage 5 implementation is the magnitude of the liquid-phase potential drop across the CL: at any voltage past first-onset (V_RHE ≲ −1.0 V), `|φ_L_cell[1] − φ_L_cell[N_CL]|` should be in the **5–25 mV** range, growing with current. Print `(φ_L[1], φ_L[N_CL/2], φ_L[N_CL])` at each Stage 5 voltage as part of the standard sweep log.

**Domains where φ_L exists.** Liquid NP transport lives only on `x ∈ [0, L_CL]`. The GDL is a gas-phase domain (porous carbon fibers carrying humidified CO₂ via Stefan-Maxwell + Darcy, §6) — there is **no liquid electrolyte, no dissolved ions, and no φ_L in the GDL**. Do not introduce `σ_GDL` or any GDL liquid potential.

**Effective diffusivity in the CL:**
```
D_eff,i = ε_I^(3/2) × α_ion × D_w,i
```
where `ε_I = 0.43` is the combined ionomer+liquid volume fraction (Bruggeman correction) and `α_ion = 0.11` is the ion diffusivity correction factor that accounts for the reduced mobility of ions in the ionomer/liquid phase relative to free solution. The same `α_ion` is applied to neutral CO₂(aq) as well: even though CO₂ does not feel the ionomer fixed-charge field, it is still confined inside the same nanometer-scale ionomer film/pore network and experiences the same tortuosity and reduced free volume as the dissolved ions. Treating it with `α_ion` (rather than 1) gives a unified, simpler effective-medium expression in this bulk-scale model.

### 3.2 Source Terms

Each CL cell has volumetric source terms from:
1. **Buffer chemistry** `R_buf,i(c)` [mol/m³/s]
2. **Electrochemical reactions** (MHC CO₂R, BV HER)
3. **Gas-liquid phase transfer** `R_PT,CO2` [mol/m³/s] (Stage 3+)

The bulk residual for cell `ix` and species `i` (integrated FV form — see §6.4):
```
F[bulk_idx(ix,i)] = J_i,left − J_i,right + S_total,i × dx[ix]
```

where `S_total,i = R_buf,i + R_echem,i + δ(i=5) × R_PT,CO2`. The integrated form keeps residual entries O(1) rather than O(1/dx); the divided form `(J_right − J_left)/dx + S` is explicitly *not* used anywhere in this model (see §14 Common Pitfalls).

### 3.3 Electrochemical Source Terms

CO₂R consumes CO₂(aq) and generates CO (in gas) and OH⁻:
```
CO₂ + H₂O + 2e⁻ → CO + 2 OH⁻
```
- `R_CO2R(c_CO2, φ_s, φ_L, c_H)` [mol/m³/s] = volumetric CO₂R rate (MHC, details in §4)
- Source on CO₂(aq): `S_CO2_from_CO2R = −R_CO2R` (consumed)
- Source on OH⁻:      `S_OH_from_CO2R  = +2 × R_CO2R`
- CO production enters the gas phase (not a liquid DOF)

HER consumes H₂O (effectively) and generates H₂ (gas) and OH⁻:
```
2H₂O + 2e⁻ → H₂ + 2 OH⁻
```
- `R_HER(φ_s, φ_L, c_H)` [mol/m³/s] = volumetric HER rate (BV, details in §4)
- Source on OH⁻: `S_OH_from_HER = +2 × R_HER`
- H₂ enters gas phase

The faradaic current density [A/m²]:
```
i_faradaic = n_e × F × ∫₀^L_CL (R_CO2R + R_HER) dx
```

**Volume fraction scaling:** The kinetic rates `R_CO2R` and `R_HER` in [mol/m³_CL/s] must account for the catalyst surface area per unit CL volume:
```
R = (a_v / (n_e F)) × i_rxn
```
where `a_v` [m⁻¹] is the specific catalyst area per CL volume, corrected by the pore saturation `S`:
```
S = ε_I / (ε_I + ε_G) = 0.43 / (0.43 + 0.34) ≈ 0.558
a_v_CL = S × 3 × ε_S / r_p = 0.558 × 3 × 0.23 / 25e-9 ≈ 1.54×10⁷ m⁻¹
```
The saturation correction `S` accounts for the fraction of catalyst surface that is wetted by the ionomer/liquid phase and thus electrochemically accessible.

The MHC pre-factor for the bulk model uses the saturation-corrected `a_v_CL` directly:
```julia
pf = a_v_CL * i0 / (n_e * F * c_ref)
```
With the initial-guess `i0_CO2R = 540 A/m²` (used through Stage 4; the fitting procedure in §17, which runs after Stage 4, will adjust this), `a_v_CL = 1.54×10⁷ m⁻¹`, `n_e = 2`, `F = 96485 C/mol`, `c_ref = 1000 mol/m³`, the pre-factor evaluates to `pf = (1.54×10⁷ × 540) / (2 × 96485 × 1000) ≈ 43.1 s⁻¹` (units check: m⁻¹ × A/m² ÷ (C/mol × mol/m³) = (C/(s·m³)) ÷ (C/m³) = 1/s ✓). The fitted pf will differ — recompute once the fitted i0_CO2R is known. Use the same value of `a_v_CL` everywhere — do **not** use `3×ε_S/r_p` (the unsaturated value) in any other place.

### 3.4 Solid-Phase Ohm's Law (φ_s as a DOF on GDL + CL)

The solid (electron-conducting) phase spans **both** the GDL and the CL. In the GDL it is the porous carbon fiber network (Sigracet 39BC). In the CL it is the percolating Ag/C composite — the carbon support and the Ag catalyst nanoparticles are in metallic contact, so they share a Fermi level and behave as a single continuous electronic conductor with one combined effective conductivity. Electrons enter from the current collector at `x = −L_GDL`, travel through the GDL carbon fibers, then through the CL carbon-support + Ag-catalyst network, and drive faradaic reactions at every catalyst surface site inside the CL. The solid potential `φ_s(x)` is **NOT a constant equal to V_applied** at the catalyst surface — there is a real ohmic drop along the path, and this drop affects the local kinetic driving force.

**Domain.** φ_s lives on `x ∈ [−L_GDL, L_CL]` (one DOF per cell in both the GDL mesh and the CL mesh). The CL φ_s DOFs represent the local potential of the combined Ag/C electronic phase at each CL cell — this is what enters the kinetics as `phi_s_local` because it is the local Fermi level of the metal where electron transfer happens.

**Charge conservation in the solid (Newman porous-electrode convention):**
```
∇·i_s = +i_faradaic_volumetric(x)            (cathodic case, defined positive)
i_s   = −σ_s_eff × ∇φ_s
```
which gives
```
−∇·(σ_s_eff × ∇φ_s) = i_faradaic_vol(x)      ⇔   ∇·(σ_s_eff × ∇φ_s) = −i_faradaic_vol(x)
```
with the volumetric source defined positive for cathodic reactions:
```
i_faradaic_vol(x) = n_e × F × (R_CO2R(x) + R_HER(x))   [A/m³, ≥ 0]
```
The source is **zero in the GDL** (no faradaic reaction in the bare GDL fibers) and **nonzero only in the CL**.

**Sign convention check.** At a cathode the electrons are *consumed* by the faradaic reaction at every catalyst site, so the conventional solid current i_s decreases in magnitude as you move from the collector (x = −L_GDL) toward the electrolyte side (x = L_CL). With i_s pointing in −x (electrons in +x means conventional current in −x), "decreasing in magnitude" means i_s becomes less negative, i.e. d(i_s)/dx > 0, i.e. ∇·i_s > 0. By complementarity ∇·i_L < 0 in the same cell — see §3.1 for the φ_L residual sign, which has the *opposite* sign on the same `i_far_vol`. The two charge-conservation laws together satisfy `∇·i_s + ∇·i_L = 0` (no net charge accumulation in the porous medium).

**Effective electronic conductivities (Bruggeman on the solid volume fraction):**
```
σ_s_eff_GDL = σ_s_GDL × (1 − ε_GDL)^1.5      # solid fraction = 1 − porosity
σ_s_eff_CL  = σ_s_CL  × ε_S^1.5
```
with
```
σ_s_GDL = 220 S/m       # Sigracet 39BC raw electronic conductivity (Lees et al. 2024)
σ_s_CL  = 100 S/m       # CL carbon support raw electronic conductivity (Lees et al. 2024)
ε_GDL   = 0.52          # GDL porosity → solid fraction = 0.48
ε_S     = 0.23          # CL solid volume fraction (catalyst + carbon)
```
giving `σ_s_eff_GDL ≈ 73 S/m` and `σ_s_eff_CL ≈ 11 S/m` after Bruggeman.

**Boundary conditions for φ_s:**
- **Left (x = −L_GDL, current collector)** — **Dirichlet `φ_s = V_applied_SHE` via ghost cell, NOT a cell-1 residual override.** The leftmost GDL cell keeps its standard continuity equation `F[φ_s_gdl[1]] = i_s_face[2] − i_s_face[1]`, and the BC value enters by plugging it in as the "left neighbor" of the leftmost face flux:
  ```
  i_s_face[1] = −σ_s_eff_GDL × (φ_s_gdl[1] − V_applied_SHE) / dx_half
  ```
  At convergence the cell-1 DOF `φ_s_gdl[1]` is slightly less negative than `V_applied_SHE` by the half-cell ohmic drop `i × dx_half / σ_s_eff_GDL`, which is physically correct.
  **CRITICAL: Do not replace the cell-1 residual with `F = φ_s_gdl[1] − V_applied_SHE`.** Overriding cell 1 decouples the boundary face current from the rest of the φ_s system and breaks the consistency check `|i_s_face[1]| = i_far_total` — you will observe `|Δi|/i ≈ 2` (equal magnitudes, opposite signs) instead of the correct ~10⁻¹². The override is wrong for φ_s because φ_s represents a transported electronic-current quantity; use the override pattern only for the φ_L gauge fix in §3.1.
- **Right (x = L_CL, electrolyte side)** — **Neumann `∇φ_s = 0`** implemented as `i_s_face[N_CL+1] = 0` in the last CL cell's residual. Electrons cannot leave the solid phase into the liquid except via faradaic reaction inside the CL.
- **Continuity at the GDL/CL interface (`x = 0`)** — φ_s and the electronic current density `i_s = −σ_s_eff ∇φ_s` are continuous across the interface. Use a harmonic mean of `σ_s_eff_GDL` and `σ_s_eff_CL` for the interface face.

**Magnitude of the ohmic drop.** At i = 1 A/cm² = 10⁴ A/m², the GDL drop is `ΔV_GDL ≈ i × L_GDL / σ_s_eff_GDL = 10⁴ × 3.25×10⁻⁴ / 73 ≈ 45 mV`, and the CL averaged drop adds another ~5 mV. So **φ_s at the catalyst surface (the value used in kinetics) can differ from V_applied by 50–100 mV at typical operating currents** — comparable to or larger than the local φ_L drop. Skipping this is a worse error than skipping the φ_L migration term.

**FV residual for φ_s in cell ix:**
```
F_phi_s[ix] = (i_s_face[ix+1] − i_s_face[ix]) − i_faradaic_vol[ix] × dx[ix]
```
The `−` sign on the source term comes from `∇·i_s = +i_far_vol` ⇒ integrated form `i_s[right] − i_s[left] = +i_far_vol × dx` ⇒ residual = `(i_s[right] − i_s[left]) − i_far_vol × dx = 0`. With this sign and the BCs (Dirichlet `V_applied` at the collector, Neumann zero at the electrolyte side), φ_s rises monotonically from `V_applied` (e.g. −1 V) toward a less-negative value at the catalyst surface (e.g. −0.95 V at 1 A/cm²) — exactly the ohmic drop expected. Compare to the φ_L residual in §3.1 which has a `+` on the same `i_far_vol`; the opposite signs are required by global charge conservation.

The face current is computed with the local effective conductivity and harmonic-mean interpolation across material boundaries. For GDL cells, `i_faradaic_vol[ix] = 0`. For CL cells, `i_faradaic_vol[ix] = n_e × F × (R_CO2R(c[ix], φ_s[ix], φ_L[ix]) + R_HER(...))` — note that this couples the φ_s residual back to the kinetics, making the solve nonlinear.

**Kinetics now use φ_s_local from the DOF:**
```
η = (φ_s_local − φ_L_local) − U_eq(pH_local)
```
where `φ_s_local` is the local φ_s DOF value at the CL cell (NOT a global V_applied). See §4 for the function signatures.

**DOF count update.** Adding φ_s adds `N_GDL + N_CL` DOFs to the residual vector. At the production default mesh (N_GDL=120 + N_CL=80), that's 200 extra DOFs (Stage 5 total: 1480 DOFs).

---

## 4. Electrochemical Kinetics

All potential arguments use **V vs SHE**. The applied potential (set by the potentiostat at the current collector) is converted from the experimental reference:
```julia
V_applied_SHE = V_cathode_RHE_pH8 - V_RHE_to_SHE_offset
V_RHE_to_SHE_offset = 2.303 × R × T / F × pH_anode   # = 0.51305 V at T=323.15 K, pH=8
```
`V_applied_SHE` is **only** the Dirichlet boundary condition for `φ_s` at `x = −L_GDL` (see §3.4). It is **NOT** the value passed into the kinetics functions.

**Critical:** The kinetics functions require BOTH:
- `φ_s_local` — the **local** solid-phase potential at the CL cell where the rate is being computed, taken from the φ_s DOF (§3.4). This differs from `V_applied_SHE` by the GDL + CL ohmic drop (50–100 mV at typical currents).
- `φ_L_local` — the local liquid potential at the same CL cell, from the NP transport solver (§3.1).

The driving force for electron transfer is the metal-liquid potential difference at the catalyst surface: `(φ_s_local − φ_L_local)`. Both potentials are spatially varying DOFs and must be evaluated at each CL cell — passing a global `V_applied` is wrong because it skips the solid ohmic drop.

### 4.1 MHC Kinetics for CO₂R

Marcus-Hush-Chidsey (MHC) theory gives a pseudo-first-order rate constant. **Regime of validity:** non-adiabatic single-electron outer-sphere transfer, parabolic free-energy surfaces, weak electronic coupling, single adsorbate surface intermediate. CO₂/CO₂•⁻ on Ag is in practice an inner-sphere concerted proton-electron transfer in most pH ranges; using MHC here is a well-established engineering simplification (see Kopljar et al., *J. Electrochem. Soc.* 163, E342; Dunwell et al., *JACS* 139, 3774) that absorbs the adsorption and decoupled-PCET details into an effective `i₀_CO2R` and `λ_MHC`. Do not over-interpret the fitted λ_MHC as the true solvent reorganization energy.

```
k_MHC(η) = (a_v × i₀_CO2R) / (n_e × F × c_ref) × √(πλ_ND) × f(η) × erfc(arg)
```

where:
- `η = (φ_s − φ_L) − U_CO₂R_nernst` [V] — overpotential
- `U_CO₂R_nernst = U_CO₂R⁰ − (2.303 RT/F) × pH_local` — Nernst-corrected equilibrium at local CL pH
- `U_CO₂R⁰ = −0.11` V vs SHE (standard reduction potential CO₂/CO)
- `λ_ND = λ_MHC / (k_B_eV × T)` — dimensionless reorganization energy
- `λ_MHC = 1.8` eV (initial guess — the fitting procedure in §17 will adjust this)
- `η_ND = n_e × F × η / (RT)` — dimensionless overpotential
- `f(η) = 1 / (1 + exp(η_ND))` — Fermi function (clamp `η_ND` to [−700, 700])
- `arg = (λ_ND − √(1 + √λ_ND + η_ND²)) / (2 × √λ_ND)`
- `c_ref = 1000` mol/m³

```julia
# Argument convention: pass λ_MHC in eV; mhc_k_elec computes λ_ND internally.
# This matches the Ref-backed accessor lambda_MHC() from §13/§17.
function mhc_k_elec(eta, i0, a_v, lambda_eV, n_e)
    lambda_ND = lambda_eV / (k_B_eV * T)              # dimensionless reorganization energy
    eta_ND = n_e * F_const * eta / (R_gas * T)
    eta_ND_clamp = clamp(eta_ND, -700.0, 700.0)
    fermi = 1.0 / (1.0 + exp(eta_ND_clamp))
    arg = (lambda_ND - sqrt(1.0 + sqrt(lambda_ND) + eta_ND^2)) / (2.0 * sqrt(lambda_ND))
    pf = a_v * i0 / (n_e * F_const * c_ref)
    return max(pf * sqrt(π * lambda_ND) * fermi * erfc(arg), 0.0)
end

function R_CO2R(c_CO2, phi_s, phi_L, c_H, a_v, lambda_eV, i0, n_e)
    pH_local = -log10(max(c_H / 1000.0, 1e-20))     # c_H in mol/m³ → pH
    U_eq = U_CO_0 - (2.303 * R_gas * T / F_const) * pH_local
    eta = (phi_s - phi_L) - U_eq                     # MUST subtract phi_L
    k = mhc_k_elec(eta, i0, a_v, lambda_eV, n_e)
    return k * max(c_CO2, 0.0)
end
```

Volumetric rate:
```
R_CO2R(c_CO2, φ_s, φ_L, c_H) = k_MHC(η(φ_s, φ_L, pH(c_H))) × c_CO2    [mol/m³_CL/s]
```

### 4.2 Butler-Volmer Kinetics for HER

Tafel/Butler-Volmer with Ag on carbon support (pH-dependent exchange current):

```
R_HER(φ_s, φ_L, c_H) = (a_v / (n_e F)) × i₀_HER(pH) × exp(−α_c × F × η_HER / (RT))
```

where:
- `η_HER = (φ_s − φ_L) − U_HER_nernst` — HER overpotential
- `U_HER_nernst = 0 − (2.303 RT/F) × pH_local` — HER equilibrium at local CL pH
- `i₀_HER(pH) = HER_prefactor × exp(−(83000 + 1000 × pH) / (RT))` [mA/cm²] × 10 → [A/m²]
- `HER_prefactor = 2.409 × 10¹³` (initial guess — fitting will adjust)
- `α_c = 0.0872` (initial guess — fitting will adjust)
- Only cathodic branch (cathodic reduction dominant at these potentials)
- Clamp exponential argument to [−700, 700]

```julia
function R_HER(phi_s, phi_L, c_H, a_v, n_e)
    pH_local = -log10(max(c_H / 1000.0, 1e-20))
    U_eq_HER = -(2.303 * R_gas * T / F_const) * pH_local
    eta_HER = (phi_s - phi_L) - U_eq_HER             # MUST subtract phi_L
    i0_mA_cm2 = HER_prefactor * exp(-(83000.0 + 1000.0 * pH_local) / (R_gas * T))
    i0 = i0_mA_cm2 * 10.0                            # convert to A/m²
    exponent = clamp(-alpha_c * F_const * eta_HER / (R_gas * T), -700.0, 700.0)
    rate = (a_v / (n_e * F_const)) * i0 * exp(exponent)
    return max(rate, 0.0)
end
```

**Note:** Both kinetics functions take `phi_L` as an explicit argument. The caller (the residual assembly) must pass the local liquid potential at the CL cell where the rate is being computed. `phi_L` is a **monolithic DOF** solved from the per-cell ionic charge balance `∇·i_L = −i_far_vol` — see §3.1 (NOT a Picard closure computed from a Henderson-style diffusion-potential formula).

---

## 5. Buffer Chemistry

Five equilibrium reactions in aqueous CO₂/carbonate system:

| # | Reaction | Forward rate | Reverse rate |
|---|----------|-------------|-------------|
| 1 | H₂O ⇌ H⁺ + OH⁻ | k₁ = 2.861×10³ mol/m³/s | k₁ᵣ = k₁/Kw |
| 2 | CO₂ + H₂O ⇌ H⁺ + HCO₃⁻ | k₂ = 0.265 s⁻¹ | k₂ᵣ = k₂/K₂ |
| 3 | HCO₃⁻ ⇌ H⁺ + CO₃²⁻ | k₃ = 59.44 s⁻¹ | k₃ᵣ = k₃/K₃ |
| 4 | CO₂ + OH⁻ ⇌ HCO₃⁻ | k₄ = 12.4 m³/(mol·s) | k₄ᵣ = k₄/K₄ |
| 5 | HCO₃⁻ + OH⁻ ⇌ CO₃²⁻ + H₂O | k₅ = 6.0×10⁶ m³/(mol·s) | k₅ᵣ = k₅/K₅ |

All concentrations in **mol/m³** (NOT mol/L). Rate constants already converted from literature M units.

**Equilibrium constants at 50°C:**
```
Kw = 5.48×10⁻¹⁴ × 10⁶   = 5.48×10⁻⁸  (mol/m³)²
K₂ = 5.16×10⁻⁷  × 10³   = 5.16×10⁻⁴  mol/m³
K₃ = 6.74×10⁻¹¹ × 10³   = 6.74×10⁻⁸  mol/m³
K₄ = K₂/Kw,   K₅ = K₃/Kw   [ALWAYS derived, never independent]
```

**Net volumetric source for each species [mol/m³/s]:**
```
S_H    = +k₁ − k₁ᵣ×c_H×c_OH + k₂×c_CO2 − k₂ᵣ×c_H×c_HCO3 + k₃×c_HCO3 − k₃ᵣ×c_H×c_CO3
S_OH   = +k₁ − k₁ᵣ×c_H×c_OH − k₄×c_CO2×c_OH + k₄ᵣ×c_HCO3 − k₅×c_HCO3×c_OH + k₅ᵣ×c_CO3
S_HCO3 = +k₂×c_CO2 − k₂ᵣ×c_H×c_HCO3 + k₄×c_CO2×c_OH − k₄ᵣ×c_HCO3 − k₃×c_HCO3 + k₃ᵣ×c_H×c_CO3 − k₅×c_HCO3×c_OH + k₅ᵣ×c_CO3
S_CO3  = +k₃×c_HCO3 − k₃ᵣ×c_H×c_CO3 + k₅×c_HCO3×c_OH − k₅ᵣ×c_CO3
S_CO2  = −k₂×c_CO2 + k₂ᵣ×c_H×c_HCO3 − k₄×c_CO2×c_OH + k₄ᵣ×c_HCO3
```

In FV form, multiply by `ε_I × dx` (buffer sources are per unit ionomer+liquid volume, scaled by the ionomer+liquid volume fraction in the cell):
```julia
S_buf_i = eps_I * buffer_rate_i(c)
```

**Implementation tip:** Scale buffer sources by a continuation parameter `α_buf ∈ [0, 1]` during bootstrap. Ramp `α_buf: 0 → 1` at mild voltage before ramping electrochemistry.

---

## 6. Gas-Phase Transport — Stefan-Maxwell + Darcy

### 6.1 Stefan-Maxwell Mixture-Average Diffusion

The mixture-average approximation for species j flux [kg/(m²·s)]:
```
J_j = −ρ × D_j^MA × ∇ω_j − ρ × D_j^MA × ω_j × (∇M_n / M_n) + ρ_j × u_G
```

where:
- `ρ` = gas density [kg/m³] = `p × M_n / (RT)`
- `M_n` = mixture molar mass [kg/mol] = 1/Σ(ω_j/M_j)
- `u_G` = Darcy velocity [m/s]
- The `∇M_n/M_n` term is the mixture-molar-mass correction for composition gradients

**Mixture-average diffusivity:**
```
D_j^MA = (1 − x_j) / Σ_{k≠j} (x_k / D_jk_eff)
```
where `x_j` are mole fractions and `D_jk_eff` are effective binary diffusivities (Bruggeman-corrected Fuller-Schettler-Giddings values).

**H₂O is NOT a DOF:** Set `ω_H2O = max(1 − ω_CO2 − ω_CO − ω_H2, 0)` at every evaluation.

### 6.2 Darcy Flow

```
u_G = −(κ / μ_gas) × (dp/dx)
```

**Pressure is the 4th gas DOF per cell.** The continuity equation for total mass:
```
∂(ρu_G)/∂x = S_mass
```

In the GDL (no sources): `∂(ρu_G)/∂x = 0`
In the CL: `S_mass = −R_PT × M_CO2 + R_CO2R × M_CO + R_HER × M_H2` [kg/(m³·s)]

### 6.3 Fuller-Schettler-Giddings Binary Diffusivities

```
D_ij [cm²/s] = 1.00×10⁻³ × T^1.75 × √(1/M_i + 1/M_j) / (p_atm × (v_i^(1/3) + v_j^(1/3))²)
```
Convert to m²/s: multiply by 10⁻⁴.

**Note:** The prefactor is `1.00×10⁻³`, NOT `1.013×10⁻²`. Using the wrong value gives ~10× too high diffusivities. See Perry's Chemical Engineers' Handbook, 9th edition.

Diffusion volumes `v_i` [cm³/mol]:

| Species | v_i |
|---------|-----|
| CO₂ | 26.9 |
| CO | 18.9 |
| H₂ | 7.07 |
| H₂O | 12.7 |

### 6.4 FV Form — Integrated Residual (Critical)

**Use the integrated FV form to avoid 1/dx scaling issues on graded meshes:**

For species j at CL cell ix:
```
F[gas_cl_idx(ix,j)] = J_left[j] − J_right[j] + S_j × dx[ix]    [residual in kg/(m²·s), same units as J; S_j × dx = kg/(m³·s) × m = kg/(m²·s)]
```

NOT the divided form `(J_right − J_left)/dx + S`. The integrated form has entries O(1) rather than O(1/dx) ≈ O(1e9) for fine mesh cells.

**Pressure (continuity) equation per cell:**
```
F[gas_cl_idx(ix,4)] = ρu_L − ρu_R + S_mass × dx[ix]
```

where `ρu` is the Darcy mass flux at each face.

### 6.5 Gas Source Terms for CL Cells

```
S_CO2  = −R_PT_CO2[ix] × M_CO2   [kg/(m³·s)]
S_CO   = +R_CO2R[ix] × M_CO
S_H2   = +R_HER[ix] × M_H2
S_mass = S_CO2 + S_CO + S_H2   (net mass source for continuity)
```

where `R_PT_CO2[ix]` [mol/(m³·s)] is the gas→liquid CO₂ phase transfer rate (see §7.4).

**Reaction water is supplied from the liquid/ionomer phase, not from the gas H₂O.** Both CO₂R (CO₂ + H₂O + 2e⁻ → CO + 2OH⁻) and HER (2H₂O + 2e⁻ → H₂ + 2OH⁻) consume H₂O molecules at the catalyst surface — but the H₂O comes from the wetted ionomer film, not from the gas-phase H₂O in the macropores. The gas H₂O (mass fraction ω_H₂O) is therefore *closure-only* in the SM residual: it does not have its own source/sink and is computed from `ω_H₂O = 1 − ω_CO₂ − ω_CO − ω_H₂`. The faradaic reactions appear only in `S_CO`, `S_H2`, and (via R_PT coupling) `S_CO2`. This is the standard treatment for half-cell models where the ionomer is the water reservoir.

---

## 7. Boundary Conditions by Stage

### 7.1 Stage 1: Constant Gas Dirichlet + Dirichlet Electrolyte

**Gas side (x = 0, CL face):**
- CO₂(aq): Dirichlet = `H_eff(50°C) × p_CO2_inlet` [mol/m³]
  ```
  H_eff(T) = 34.0 × exp(2400 × (1/T − 1/298.15))  [mol/(m³·atm)]  (van't Hoff)
  At T = 323.15 K: H_eff(50°C) = 34.0 × exp(2400 × (1/323.15 − 1/298.15)) = 34.0 × 0.5365 ≈ 18.24 mol/(m³·atm)
  p_CO2_inlet = x_CO₂,inlet × p_total = 0.88 × 1.0 = 0.88 atm
    (Humidified CO₂ feed at 50°C: water saturation pressure is ~0.122 atm by Antoine,
     so x_H2O,inlet ≈ 0.12, x_CO2,inlet ≈ 0.88 in the inlet gas mole fractions.)
  c_CO2_bc = H_eff(50°C) × p_CO2_inlet ≈ 18.24 × 0.88 ≈ 16.0 mol/m³ at 50°C
  ```
- Ions (H⁺, OH⁻, HCO₃⁻, CO₃²⁻): zero flux at x = 0

**Electrolyte side (x = L_CL):**
- Dirichlet for all species = equilibrium concentrations `c_eq` (1M CsHCO₃ at 50°C):
  ```
  c_eq = solve_buffer_equilibrium(c_Cs=1000, C_total=1000)
  ```

### 7.2 Stage 2: Sherwood-Reynolds Neumann BC at Electrolyte Side

Replace the Dirichlet at x = L_CL with a convective mass transfer condition:

```
J_i(L_CL) = −k_MT,i × (c_bulk_elyte,i − c_i(L_CL))    [mol/m²/s]
         = +k_MT,i × (c_i(L_CL) − c_bulk_elyte,i)
```
The minus sign is required under the +x-positive flux convention used throughout this guide: inflow at the right boundary (species entering the CL from the electrolyte reservoir, e.g. HCO₃⁻) is a *negative* flux value in the +x frame. This formula applies directly in Stages 1–4 where φ_L = 0. **In Stage 5, the right-boundary flux must include the Nernst-Planck migration term** — use `np_flux(c[i,N], c_eq[i], φ_L[N], 0, D_eff[i], z[i], L_eff)` with `L_eff = D_eff[i] / k_MT[i]`, which reduces to the Sherwood formula when ∇φ_L = 0 but correctly carries the CO₃²⁻ migration current when φ_L ≠ 0. Without this, ionic charge conservation at the boundary is violated and the φ_L profile becomes unphysical. See the Stage 5 pseudocode in §11 for the exact implementation.

The mass transfer coefficient is computed from the Blasius/Pohlhausen flat-plate laminar correlation `Sh = 0.664 Re^(1/2) Sc^(1/3)`, consistent with Lees et al. 2024, Nat. Chem. Eng. **Regime of validity:** laminar flow over a flat plate, Re ≲ 5×10⁵, Sc ≳ 0.6, developing concentration boundary layer. At `Re ≈ 14600` and `Sc ≈ 10²–10³` this sits comfortably in the laminar regime; the real channel is rectangular, not a semi-infinite plate, so treat the result as a ±30% engineering estimate rather than a first-principles value. The definitions are given explicitly below.

**Reynolds number (`Re`):**
```
Re = u_elec × L_elec × ρ_elec / μ_elec
```

**Mass transfer coefficient per species (`k_MT,i` for species with diffusivity D_i):**
```
k_MT,i = (D_i / L_elec) × 0.664 × Re^(1/2) × (μ_elec / (ρ_elec × D_i))^(1/3)
```

where `i ≠ w` (water is excluded). Note that `μ_elec / (ρ_elec × D_i) = ν / D_i = Sc_i` (the Schmidt number).

**Electrolyte channel parameters (Lees et al. experimental setup):**
```
L_elec   = 0.0225 m           # characteristic length (electrode length)
Q_elyte  = 25 mL/min = 4.167×10⁻⁷ m³/s   # electrolyte flowrate
A_ch     = 0.8 mm × 0.8 mm = 6.4×10⁻⁷ m²  # channel cross-sectional area
d_h      = 0.8×10⁻³ m        # hydraulic diameter (square channel: d_h = side length)
ρ_elec   = 1000 kg/m³         # approximated as water
μ_elec   = 1.0×10⁻³ Pa·s     # approximated as water
```

**Numerical evaluation:**
```
u_elec = Q_elyte / A_ch = 4.167e-7 / 6.4e-7 ≈ 0.651 m/s
Re = 0.651 × 0.0225 × 1000 / 1.0e-3 ≈ 14648
```

For OH⁻ (D_OH = 8.19×10⁻⁹ m²/s):
```
Sc_OH = μ_elec / (ρ_elec × D_OH) = 1e-3 / (1000 × 8.19e-9) ≈ 122
k_MT_OH = (8.19e-9 / 0.0225) × 0.664 × 14648^0.5 × 122^(1/3)
        = 3.64e-7 × 0.664 × 121.0 × 4.96
        ≈ 1.45×10⁻⁴ m/s
```

**Recommended implementation:**
```julia
const L_elec    = 0.0225        # m, electrode/characteristic length
const Q_elyte   = 25.0 / 60e6   # m³/s, electrolyte flowrate (25 mL/min)
const A_ch      = 0.8e-3 * 0.8e-3  # m², channel cross-section
const rho_elec  = 1000.0        # kg/m³
const mu_elec   = 1.0e-3        # Pa·s

u_elec = Q_elyte / A_ch
Re = u_elec * L_elec * rho_elec / mu_elec   # Re = ρ u L / μ

function k_MT(D_i)
    # (D_i/L_elec)*0.664*Re^(0.5)*(mu_elec/(rho_elec*D_i))^(1/3)
    return (D_i / L_elec) * 0.664 * sqrt(Re) * (mu_elec / (rho_elec * D_i))^(1/3)
end

# Per-species k_MT — pass D_w[i] (bare-water D), NOT D_eff[i].
const k_MT = [k_MT(D_w[i]) for i in 1:5]
```

> **⚠️ Pass `D_w`, not `D_eff`.** The Sherwood/Blasius-Pohlhausen
> correlation describes the boundary layer in the *free electrolyte
> channel*, where there is no porous matrix — the Bruggeman-corrected
> `D_eff = ε_I^{1.5} × α_ion × D_w` does not apply on that side of the
> boundary. The worked example above uses `D_w` (`Sc = ν / 8.19e-9 = 122`
> for OH⁻); pass the bare-water `D_w[i]` to the `k_MT(D_i)` function.
>
> Mis-applying `D_eff[i]` makes `k_MT` ~10× too small (`D_eff/D_w ≈ 0.031`,
> and `k_MT ∝ D^{2/3}` ⇒ ratio ≈ 0.099). Symptom: NM has to push
> `i0_CO2R` ~25× higher than the true value to recover Lees currents
> through the depleted boundary, and the fitted kinetics will not
> transfer to a model with a correctly-scaled `k_MT` (e.g. an
> independent replicate of the same physics). This was the dominant fit-disagreement bug in earlier
> v6 builds — confirmed by re-running the fit at `k_MT(D_w)` and
> recovering ~100 A/m² (vs 2500 A/m² with the bug).

Neumann BC for cell N_CL (last cell at the electrolyte-side boundary):
```
J_i_right = −k_MT_i × (c_eq[i] − c_bulk[N_CL, i])
```

In the FV residual, use this as the right-face flux for the last cell:
```julia
N_bulk_flux[i, N_CL+1] = -k_MT[i] * (c_eq[i] - c_bulk[N_CL, i])
```

### 7.3 Stage 3: Stefan-Maxwell Gas (Dirichlet at flow-field)

**GDL left face (x = −L_GDL, flow-field side):**
- Pressure: Dirichlet `p_total = 101325` Pa
- **Mole fractions** (NOT mass fractions): Dirichlet `x_inlet = [0.88, 0.0, 0.0, 0.12]` (CO₂, CO, H₂, H₂O). At 50°C and 1 atm with x_H₂O = 0.12, the partial pressure of water is ~0.12 atm — close to the saturation vapor pressure of water at 50°C (0.122 atm), consistent with a humidified CO₂ feed.

```julia
x_inlet = [0.88, 0.0, 0.0, 0.12]       # MOLE fractions, humidified CO₂ at 50°C
omega_inlet = mole_to_mass(x_inlet)     # convert to mass fractions for SM residual
p_total = 101325.0                      # Pa
```

The Stefan-Maxwell residual (§6) operates on mass fractions, so convert mole → mass at module load time:
```julia
M_avg = sum(x_inlet[j] * M_gas[j] for j in 1:4)   # = 0.88×44.01 + 0.12×18.015 ≈ 40.89 g/mol
omega_inlet[j] = x_inlet[j] * M_gas[j] / M_avg     # → ≈ [0.947, 0, 0, 0.053]
```

This converts the inlet to mass fractions, which is what `sm_residual!` expects internally. **Do not** start from `[0.88, 0, 0, 0.12]` as mass fractions — that would give p_H₂O ≈ 0.25 atm, which exceeds saturation pressure at 50°C and is unphysical.

**CL right face (x = L_CL, electrolyte-channel boundary):**
- Zero flux for all gas species: `J_j = 0` (gas doesn't penetrate electrolyte)
- Pressure: dp/dx = 0 (no-flow)

**CO₂(aq) left BC (gas→liquid coupling):**
Replace fixed Dirichlet from Stage 1 with Henry's law using the local gas-phase CO₂:
```julia
x_CO2 = mole_fraction_CO2(omega_cl[1,:])
p_CO2_local = x_CO2 * p_cl[1] / 101325.0   # atm
c_CO2_bc = H_eff(c_ions) * p_CO2_local
```

### 7.4 Stage 3: Volumetric Phase Transfer in CL (added with the gas DOFs)

When the gas DOFs are introduced in Stage 3 (Stefan-Maxwell), the volumetric phase transfer source `R_PT` is added at the same time — gas-phase CO₂ has nowhere to couple to the liquid otherwise. (Stage 4 then finalizes the Darcy pressure equation, but R_PT itself is a Stage-3 addition.)

Add volumetric CO₂ phase transfer in every CL cell:
```
R_PT = a_v_GL × k_GL × (H_eff × p_CO2 − c_CO2_aq)    [mol/(m³·s)]
```

where:
- `a_v_GL = 3 × ε_G / R_pore ≈ 4.08×10⁷ m⁻¹` is the **ionomer/gas-pore contact area per unit CL volume**, not a bubble area. This is the surface where dissolved CO₂(aq) in the ionomer-wetted catalyst region exchanges with gas-phase CO₂ in the gas-filled pores. With ε_G = 0.34 and R_pore ≈ 25 nm (characteristic gas-pore radius), `a_v_GL = 3 × 0.34 / 25e-9 = 4.08×10⁷ m⁻¹`.
- `k_GL = 1×10⁻⁴ m/s` is the mass transfer coefficient across this ionomer/gas interface.
- `p_CO2` from local gas DOF, `c_CO2_aq` from local liquid DOF.
- Sechenov correction to Henry's constant (salting-out by Cs⁺):
  ```
  log₁₀(H/H₀) = −(h_CO2 + h_Cs)×[Cs⁺]/1000 − (h_CO2 + h_OH)×[OH⁻]/1000 − ...
  ```
  where `h_CO2 = −0.02599`, `h_Cs = 0.0759`, `h_OH = 0.0893`, `h_HCO3 = 0.0967`, `h_CO3 = 0.1423`

**Combined Henry constant (van't Hoff + Sechenov).** The H that appears here and in the gas-side BC of §7.3 is the *combined* T-corrected and ion-corrected effective Henry constant:
```
H_eff(T, c_ions) = H_CO2_0 × exp(2400 × (1/T − 1/298.15))
                              × 10^(−Σᵢ (h_CO2 + h_i) × cᵢ / 1000)
```
where the van't Hoff factor handles temperature (H_CO2_0 = 34.0 mol/(m³·atm) at 25°C), and the Sechenov factor handles salting-out by the local ions (h_CO2 = −0.02599; h_Cs, h_OH, h_HCO3, h_CO3 from the table below). At 1 M CsHCO₃, 50°C, the raw per-species coefficients are `(h_CO2 + h_Cs) = 0.0499`, `(h_CO2 + h_HCO3) = 0.0707`, `(h_CO2 + h_CO3) = 0.1163` (all in L/mol ≡ M⁻¹; the formula on the previous line converts cᵢ from mol/m³ → mol/L via the `/1000` factor so the h·c product is dimensionless). Multiplying each by the local concentration divided by 1000 and summing over the dominant species (c_Cs ≈ 1000, c_HCO3 ≈ 978, c_CO3 ≈ 11 mol/m³) gives per-species contributions 0.0499 + 0.0692 + 0.0013 ≈ 0.120 for the Sechenov exponent. So H_eff ≈ 18.24 × 10^(−0.120) ≈ 13.8 mol/(m³·atm), and at the GDL face c_CO2,eq ≈ 13.8 × 0.88 ≈ 12.2 mol/m³. This is the Henry's-law value you would see in a CL cell that is both fully wetted *and* already at the 1 M CsHCO₃ ion composition; the first CL cell near the GDL typically sits slightly below this because the ionomer has not fully equilibrated.

**Phase transfer source signs.** With `R_PT = a_v_GL × k_GL × (H_eff × p_CO2 − c_CO2_aq)`, the convention is: `R_PT > 0` when the gas is supersaturating the liquid (gas → liquid net dissolution). Then:
- Liquid CO₂(aq) source: `+R_PT` (gas dissolves into the liquid, raising c_CO2_aq).
- Gas CO₂ mass source: `−R_PT × M_CO2` (gas-phase CO₂ is lost to the liquid).

Both signs are consistent with mass conservation across the interface.

**Darcy pressure BCs:**
- Flow-field: `p = p_total` (Dirichlet, first GDL cell)
- Electrolyte side: `dp/dx = 0` (no gas flow)

---

## 8. Complete Parameter Tables

### 8.1 Physical Constants

| Symbol | Value | Units | Reference |
|--------|-------|-------|-----------|
| F | 96485.332 | C/mol | CODATA 2018 |
| R | 8.314462618 | J/(mol·K) | CODATA 2018 |
| T | 323.15 | K | 50°C |
| k_B | 8.617333×10⁻⁵ | eV/K | CODATA 2018 |

### 8.2 Geometry

| Symbol | Value | Units | Reference |
|--------|-------|-------|-----------|
| L_CL | 10×10⁻⁶ | m | Lees et al. 2024, Nat. Chem. Eng. |
| L_GDL | 325×10⁻⁶ | m | Sigracet 39BC spec sheet |
| r_p | 25×10⁻⁹ | m | catalyst particle radius, Cetinbas 2017 |

**Default mesh (production):** `N_CL = 80`, `N_GDL = 120` cells, both with geometric grading `q = 1.3` (CL refined toward `x = L_CL`; GDL refined toward `x = 0` at the GDL/CL interface). See §9.1 for the rationale and §9.5.10 for the mesh-independence study (production N=80/120 is within 0.2 % of the N=1000 asymptote). Use `MESH_N_CL=20 MESH_N_GDL=30` env-var overrides only for development/debugging — fit results at 20/30 are biased and should not be reported.

### 8.3 Volume Fractions (Lees-style bulk model)

| Symbol | Value | Description | Reference |
|--------|-------|-------------|-----------|
| ε_S | 0.23 | solid volume fraction (Ag/C + carbon) | Secanell 2007, Seteiz 2024 |
| ε_G | 0.34 | gas-filled macro-pores | Updated |
| ε_I | 0.43 | ionomer + liquid combined phase | Updated (ε_S + ε_G + ε_I = 1.0) |
| α_ion | 0.11 | ion diffusivity correction factor | Lees et al. 2024, Nat. Chem. Eng. |
| C_fix | 1000 | fixed charge concentration [mol/m³] (1 M) | Ionomer IEC |
| S_pore | ε_I/(ε_I+ε_G) ≈ 0.558 | pore saturation | Derived |

**Effective liquid diffusivity** in the CL (Bruggeman correction with ionomer+liquid volume fraction):
```
D_eff,i = ε_I^1.5 × α_ion × D_w,i
```
where `ε_I = 0.43` is the combined ionomer+liquid volume fraction and `α_ion = 0.11` is the ion diffusivity correction factor for transport through the ionomer/liquid phase. See §8.3 for details.

### 8.4 Liquid Diffusivities at 50°C

| Species | D_w [m²/s] | Method | Reference |
|---------|-----------|--------|-----------|
| H⁺ | 1.33×10⁻⁸ | Nernst-Einstein from λ° | Robinson & Stokes 1959 |
| OH⁻ | 8.19×10⁻⁹ | Nernst-Einstein from λ° | Robinson & Stokes 1959 |
| HCO₃⁻ | 2.09×10⁻⁹ | Stokes-Einstein ×T/T₀×η₀/η | Li & Gregory 1974 |
| CO₃²⁻ | 1.63×10⁻⁹ | Stokes-Einstein ×T/T₀×η₀/η | Robinson & Stokes 1959 |
| CO₂(aq) | 3.39×10⁻⁹ | Stokes-Einstein ×T/T₀×η₀/η | Tamimi et al. 1994 |
| Cs⁺ | 3.63×10⁻⁹ | Stokes-Einstein correction | Robinson & Stokes 1959 |

Temperature scaling from 25°C: multiply 25°C values by `(T/298.15) × (η(25°C)/η(50°C))` ≈ `1.764`.

### 8.5 Kinetic Parameters (CO₂R and HER)

| Symbol | Value | Units | Reference / Notes |
|--------|-------|-------|-------------------|
| U_CO₂R⁰ | −0.11 | V vs SHE | Standard CO₂/CO reduction, Bard & Faulkner |
| i₀_CO2R | 540 (initial guess) | A/m² | Initial guess taken from Seteiz 2024 table S2 for Ag/CNT in 1 M CsHCO₃. Use as the starting kinetics for Stages 1–4; the fitting procedure (§17, run after Stage 4) will determine the final value. |
| λ_MHC | 1.8 (initial guess) | eV | Reorganization energy (inner-sphere + outer-sphere reorganization for CO₂/CO₂•⁻ on Ag). The fit is weakly sensitive to λ in the 1.5–2.0 eV window. Marcus 1965; Chidsey 1991. |
| c_ref | 1000 | mol/m³ | Reference concentration for MHC normalization |
| HER_prefactor | 2.409×10¹³ (initial guess) | mA/cm² | Arrhenius pre-exponential (multiplied by exp(−Eₐ(pH)/RT) to give i₀_HER in mA/cm²; convert with ×10 to get A/m²). Fitted in log-space; use as the starting kinetics for Stages 1–4; the fitting procedure (§17, run after Stage 4) will determine the final value. |
| α_c | 0.0872 (initial guess) | — | Cathodic transfer coefficient HER (effective). Note: α_c_HER ≈ 0.09 implies a Tafel slope of ~735 mV/decade at 50°C (2.303 RT/(α_c F) = 0.06413/0.0872 ≈ 0.735 V/dec), much larger than the ~128 mV/dec ideal for a 1-electron rate-limiting step at 50°C (118 mV/dec at 25°C). This is an *empirical* fit parameter that absorbs (a) the pH dependence of the Frumkin double-layer correction, (b) Volmer-Heyrovsky/Tafel mechanism crossover, and (c) reaction-order non-idealities not captured by the simple BV form. Treat as fitting freedom, not a physical Frumkin coefficient. |
| E_act_HER | 83000 | J/mol | Activation energy HER (pH=0) |
| dE_act/dpH | 1000 | J/(mol·pH) | pH sensitivity of E_act, Quaino et al. |

### 8.6 Buffer Chemistry at 50°C

| Constant | Value | Units | Notes |
|----------|-------|-------|-------|
| K_w | 5.48×10⁻⁸ | (mol/m³)² | Water autoprotolysis at 50°C |
| K₂ | 5.16×10⁻⁴ | mol/m³ | CO₂ first dissociation at 50°C |
| K₃ | 6.74×10⁻⁸ | mol/m³ | Bicarbonate dissociation at 50°C |
| k₁ | 2.861×10³ | mol/m³/s | H₂O → H⁺ + OH⁻ forward, Pinsent et al. 1956 |
| k₂ | 0.265 | s⁻¹ | CO₂ + H₂O → H⁺ + HCO₃⁻, Pinsent et al. 1956 |
| k₃ | 59.44 | s⁻¹ | HCO₃⁻ → H⁺ + CO₃²⁻ |
| k₄ | 12.4 | m³/(mol·s) | CO₂ + OH⁻ → HCO₃⁻ (12400 M⁻¹s⁻¹ ÷ 1000) |
| k₅ | 6.0×10⁶ | m³/(mol·s) | HCO₃⁻ + OH⁻ → CO₃²⁻ (6×10⁹ M⁻¹s⁻¹ ÷ 1000) |

**Reverse rates:** `k₁ᵣ = k₁/K_w`, `k₂ᵣ = k₂/K₂`, `k₃ᵣ = k₃/K₃`, `k₄ᵣ = k₄/K₄`, `k₅ᵣ = k₅/K₅`. Never set these independently.

### 8.7 Gas Transport Parameters

| Symbol | Value | Units | Reference |
|--------|-------|-------|-----------|
| M_CO₂ | 44.01×10⁻³ | kg/mol | — |
| M_CO | 28.01×10⁻³ | kg/mol | — |
| M_H₂ | 2.016×10⁻³ | kg/mol | — |
| M_H₂O | 18.015×10⁻³ | kg/mol | — |
| v_CO₂ | 26.9 | cm³/mol | FSG diffusion volume, Perry's 9th ed. |
| v_CO | 18.9 | cm³/mol | FSG diffusion volume |
| v_H₂ | 7.07 | cm³/mol | FSG diffusion volume |
| v_H₂O | 12.7 | cm³/mol | FSG diffusion volume |
| x_CO₂,inlet | 0.88 | — | **MOLE fraction**, humidified CO₂ feed at 50°C |
| x_H₂O,inlet | 0.12 | — | **MOLE fraction**, saturated steam at 50°C (p_H₂O/p_total ≈ 0.122 atm/1 atm) |
| ω_CO₂,inlet | ≈ 0.947 | — | Mass fraction, derived from x via M_avg ≈ 40.89 g/mol |
| ω_H₂O,inlet | ≈ 0.053 | — | Mass fraction, derived from x |
| p_total | 101325 | Pa | 1 atm |
| ε_GDL | 0.52 | — | Sigracet 39BC porosity |
| ε_G (CL) | 0.34 | — | CL gas void fraction |
| κ_GDL | 8.4×10⁻¹³ | m² | Sigracet 39BC dry Darcy permeability — no saturation correction (GDL has no liquid in this model) |
| κ_CL | 8.0×10⁻¹⁶ × (1−S)^3 ≈ 7.0×10⁻¹⁷ | m² | Darcy permeability CL, Leverett-style saturation correction with S = ε_I/(ε_I+ε_G) ≈ 0.558 |
| μ_gas | 1.59×10⁻⁵ | Pa·s | Wilke mixing rule at 50°C, x_CO₂ = 0.88, x_H₂O = 0.12 |

**Bruggeman corrections (gas-phase diffusivities):**
```
D_eff_GDL[i,j] = ε_GDL^1.5 × D_binary[i,j]
D_eff_CL[i,j]  = ε_G^1.5   × D_binary[i,j]
```
For liquid diffusivities, the Bruggeman correction uses the ionomer+liquid volume fraction:
```
D_eff_liquid,i = ε_I^1.5 × α_ion × D_w,i
```
Each domain uses a single, consistent Bruggeman exponent of 1.5. **Regime of validity:** Bruggeman's ε^1.5 relation (tortuosity τ = ε^(−0.5)) assumes isotropic random packing of spherical grains, no percolation threshold effects, and scale separation between pore size and domain length. For Sigracet 39BC (fibrous GDL) the fiber anisotropy violates the sphere assumption and the "true" in-plane vs through-plane tortuosity differs by ~30–50%; we absorb that uncertainty into the raw `σ_s_GDL = 220 S/m`. For the CL (sub-50 nm pore network in an ionomer film) the ion-diffusivity hit is dominated by the separate `α_ion = 0.11` multiplier, not by Bruggeman, so the choice of exponent is a second-order effect there.

### 8.8 Solid-Phase Electronic Conductivity (GDL + CL)

These are real, independent parameters describing electron transport through the solid phase: the carbon fibers in the GDL, and the combined Ag-catalyst + carbon-support percolating network in the CL. They drive the φ_s ohmic drop solved in §3.4.

| Symbol | Value | Units | Notes |
|--------|-------|-------|-------|
| σ_s_GDL | 220 | S/m | Sigracet 39BC carbon-fiber raw electronic conductivity, Lees et al. 2024 |
| σ_s_CL | 100 | S/m | Combined Ag/C percolating phase raw electronic conductivity, Lees et al. 2024 |
| σ_s_eff_GDL | σ_s_GDL × (1−ε_GDL)^1.5 ≈ 73 | S/m | Bruggeman with solid fraction (1−0.52) = 0.48 |
| σ_s_eff_CL | σ_s_CL × ε_S^1.5 ≈ 11 | S/m | Bruggeman with solid fraction ε_S = 0.23 (CL solid = catalyst + support) |

**These are NOT ionic conductivities** — they describe electron transport through the carbon fibers / catalyst support, not ion transport through the liquid. The two are unrelated:
- Electronic conduction (this section): in the solid phase, on `[−L_GDL, L_CL]`, parameterized by `σ_s` constants.
- Ionic conduction: in the liquid phase, on `[0, L_CL]` only, emerges from the per-species `D_eff,i` and concentrations via the NP migration term — **no independent ionic σ parameter exists**, see §3.1.

At 1 A/cm² operating current, the solid ohmic drops are:
- GDL drop ≈ `i × L_GDL / σ_s_eff_GDL` = `10⁴ × 3.25e-4 / 73` ≈ **45 mV**
- CL averaged drop ≈ `i × L_CL / (2 × σ_s_eff_CL)` = `10⁴ × 1e-5 / 22` ≈ **5 mV** (factor of 2 because current decreases linearly through the CL as it is consumed by reaction)

So φ_s at the catalyst surface differs from `V_applied` (set at the current collector) by ~50 mV at 1 A/cm² and proportionally less at lower currents. This must be solved, not ignored.

### 8.9 Electrolyte Channel / Mass Transfer Parameters

| Symbol | Value | Units | Notes |
|--------|-------|-------|-------|
| L_elec | 0.0225 | m | Electrode/characteristic length, Lees et al. 2024 |
| Q_elyte | 25 | mL/min | Electrolyte flowrate |
| A_ch | 6.4×10⁻⁷ | m² | Channel cross-section (0.8 mm × 0.8 mm) |
| d_h | 8.0×10⁻⁴ | m | Hydraulic diameter (square channel) |
| ρ_elec | 1000 | kg/m³ | Approximated as water |
| μ_elec | 1.0×10⁻³ | Pa·s | Approximated as water |
| Re | u_elec×L_elec×ρ_elec/μ_elec ≈ 14648 | — | Blasius flat-plate Re |

### 8.10 Gas-Liquid Interface Parameters

| Symbol | Value | Units | Notes |
|--------|-------|-------|-------|
| H_CO2(T=25°C) | 34.0 | mol/(m³·atm) | Henry's constant CO₂ at 25°C |
| ΔH_sol | 2400 | K | van't Hoff temperature sensitivity |
| k_GL | 1×10⁻⁴ | m/s | GL mass transfer coefficient |
| a_v_GL | 3×ε_G/R_pore ≈ 4.08×10⁷ | m⁻¹ | Ionomer/gas-pore contact area per CL volume; R_pore = 25 nm |

Sechenov parameters [from Schumpe 1993, Weisenberger & Schumpe 1996]:
| Ion | h_ion |
|-----|-------|
| CO₂ (gas) | −0.02599 |
| Cs⁺ | +0.0759 |
| OH⁻ | +0.0893 |
| HCO₃⁻ | +0.0967 |
| CO₃²⁻ | +0.1423 |

---

## 9. Numerical Methods

### 9.1 Finite Volume Discretization

Use a cell-centered FV scheme on a graded mesh. For a 1D domain:

```
|-- dx[1] --|-- dx[2] --|-- ... --|-- dx[N_CL] --|
|    c₁    |     c₂    |         |     c_N      |
x_face[1]   x_face[2]             x_face[N_CL+1]
```

**Face flux (pure diffusion, Stages 1–4):**
```julia
function diffusion_flux(c_L, c_R, D_eff, dx)
    return -D_eff * (c_R - c_L) / dx
end
```
Note `dx` here is the distance between the two cell *centers*, not cell width.

**Nernst-Planck face flux (diffusion + migration, Stage 5):**
```julia
function np_flux(c_L::Real, c_R::Real,
                 phi_L_left::Real, phi_L_right::Real,
                 D_eff::Real, z_i::Real, dx_cc::Real)
    c_avg = 0.5 * (c_L + c_R)
    diff  = -D_eff * (c_R - c_L) / dx_cc
    migr  = -(z_i * F_const / (R_gas * T)) * D_eff * c_avg *
            (phi_L_right - phi_L_left) / dx_cc
    return diff + migr
end
```
Use `np_flux` for **all** Stage 5 face fluxes: interior faces AND the right boundary. For the right boundary, set `phi_L_right = 0` (gauge point) and `dx_cc = L_eff = D_eff[i] / k_MT[i]` (effective Sherwood length). This reduces to the Sherwood formula when `phi_L_left = 0` (Stages 1–4) and correctly carries migration current when `phi_L_left ≠ 0` (Stage 5). See §11 Stage 5 gotchas #12.

**Graded mesh for CL (recommended):**
```julia
function make_cl_mesh(N_CL, L_CL; q=1.2)
    # Geometric progression with refinement at x=L_CL (electrolyte side)
    # q > 1: cells get larger toward x=0 (GDL side)
    r = q ^ (1/(N_CL-1))
    ...
end
```
For the bulk-only model, the **production default** is a graded mesh with `N_CL = 80` and `N_GDL = 120`, both with `q ≈ 1.3` (CL refined near the electrolyte BC where mass-transport limitation creates the steepest gradients; GDL refined near the CL interface). This mesh is within 0.2 % of the asymptotic mesh-converged solution (verified to N_CL = 1000 in §9.5.10) and is the smallest mesh that's robustly mesh-independent at typical operating currents.

> **Why not N_CL = 20?** The 20/30 mesh is fast (300–400 DOFs per stage) but mesh-discretization error on i_CO and i_H₂ at high cathodic V is ~3–5 %, comparable to the model–experiment fit residual. At 20/30 the cell-centered FV stencil also under-resolves the Sechenov-driven c_CO2(x) collapse near the EL face, biasing fit step §17. **Use N_CL = 80, N_GDL = 120 for any production fit or polarization comparison.** N_CL = 20 is acceptable only for development/debugging where build-test cycle time matters more than 3 % accuracy.

**GDL mesh:** Graded with `N_GDL = 120`, `q = 1.3`, coarsening away from CL.

### 9.2 Log-Concentration DOFs

Store `u[i] = ln(c_i)` and recover `c_i = exp(clamp(u[i], -50, 50))`.

**Why:**
1. Ensures c > 0 always (no negative concentrations from Newton steps)
2. Improves conditioning: concentration spans O(10¹⁰) range in pH gradient
3. Newton step clamping in log space: `max_step = 5.0` (in log units) limits concentration changes to `exp(5) ≈ 150×` per step

**Residual scaling:** When computing the integrated FV residual `J_left − J_right + S × dx` (see §6.4) with log-DOFs, the chain rule gives an extra `c_i` factor:
```
∂F/∂u_i = ∂F/∂c_i × ∂c_i/∂u_i = ∂F/∂c_i × c_i
```
This is automatically handled by FD Jacobian.

### 9.3 Jacobians — FD for bring-up, ForwardDiff AD for production

> **🛑 One rule, stated once:** every production stage (Stages 1–5 + the Fit step) uses the **ForwardDiff AD Jacobian** of §9.3.2. Finite differences are acceptable only for a Stage-1/2 bring-up at N_CL ≤ 20. The reasons — probe cost and truncation-error corruption of the kinetic coupling — are given in §9.3.2; the failure modes of ignoring this rule are catalogued in §9.4.

#### 9.3.1 Dense FD Jacobian (small-mesh bring-up only)

For `N ≤ ~300` DOFs (i.e. N_CL ≤ 20) in **Stages 1–2**, where the residual scale spread is modest, a dense forward-difference Jacobian is acceptable for the initial bring-up:

```julia
function fd_jacobian!(J, res!, F0, u; eps=1e-7)
    N = length(u)
    u_p = copy(u)
    F_p = similar(F0)
    for j in 1:N
        h = eps * max(abs(u[j]), 1.0)
        u_p[j] = u[j] + h
        res!(F_p, u_p)
        @. J[:, j] = (F_p - F0) / h
        u_p[j] = u[j]
    end
end
```

Use `eps = 1e-7`. The `max(abs(u[j]), 1.0)` prevents `h → 0` for near-zero DOFs.

#### 9.3.2 ForwardDiff AD Jacobian (required for ALL production stages)

**Every production stage requires an exact Jacobian** because of the kinetic-coupling scale spread. The Jacobian scale spread grows with each stage — Stage 3 spans ~12 orders of magnitude (gas pressure ~10⁵, species log-conc ~10⁻⁷), Stage 5 spans ~17 (φ_s divergence ~4×10⁷, φ_L ~10⁵, gas mass-fractions ~10⁻⁷, species log-conc ~10⁻¹⁰). FD with `eps = 1e−7` produces systematically corrupted kinetic cross-terms: the truncation error on the dominant diagonal contribution overwhelms the small off-diagonal kinetic coupling (`∂R/∂φ_s × dx ≈ 40`). Newton's direction is corrupted as soon as kinetics turn on (V_RHE ≲ −0.6) and the iterate stalls at ‖F‖∞ ~ 1e-3 — at every stage from Stage 3 onwards, and at Stage 5 even at small meshes. FD is also slow: O(N) residual evaluations per Jacobian build is ~1.5–2.5 s per Newton iteration at the 1200-DOF Stage 3, vs ~3 ms for AD. This is a direct finding from implementing the spec; Brady et al., *J. Electrochem. Soc.* **168**, 113501 (2021) document the same failure mode for Newman-tradition porous electrode models and demonstrate the AD fix.

The cleanest exact-Jacobian path in Julia is `ForwardDiff.jacobian!` on a type-generic residual function. This means **the entire residual chain must accept `AbstractVector{<:Real}`** (not `AbstractVector{Float64}`) so that `ForwardDiff.Dual` numbers propagate through every operation.

**Type-generalization checklist** — every function called inside `residual_stage5!` must meet these conditions:

1. Signatures use `::Real` / `<:Real` for scalar arguments and `AbstractVector{<:Real}` or `AbstractVector{ElT} where ElT<:Real` for vector arguments — not `::Float64`.
2. Local work arrays are allocated with `zeros(ElT, ...)` (with the type parameter from the residual's outer `where {ElT<:Real}` signature) or equivalently `zeros(eltype(u), ...)`, never `zeros(...)` which defaults to Float64 and will error when ForwardDiff tries to write Dual values.
3. Literal-zero assignments use `zero(ElT)` (or `zero(eltype(u))`), not `0.0`.
4. Tuple/matrix return values carry the dual type through — e.g. `cl_omega(u, ix)` should return `NTuple{4,ElT}` (element type from `u`), not `NTuple{4,Float64}`.
5. Constant matrices/vectors like `D_bin_raw` can stay `Matrix{Float64}` — scalar operations promote automatically.
6. **Do NOT name a type parameter `T`** anywhere in the residual chain — `T` is the global temperature constant from §8.1 and will be shadowed with silent wrong-type errors. Use `ElT` (as shown in items 1–4 above) or similar.

**Usage:** build a `JacobianConfig` once with chunk size 12 and reuse it for all Newton iterations. **Use the same closure for both the config construction and `jacobian!` calls** — ForwardDiff checks tags and will error if you pass different anonymous functions.

```julia
using ForwardDiff

res_closure! = (F_, u_) -> residual_stage5!(F_, u_, V_cathode, cl_mesh, gdl_mesh,
                                             c_eq_vec, layout5)
cfg = ForwardDiff.JacobianConfig(res_closure!, zeros(N), u, ForwardDiff.Chunk{12}())
J   = zeros(N, N)
ForwardDiff.jacobian!(J, res_closure!, zeros(N), u, cfg)
```

Cost: ~N/12 forward passes per Jacobian, each pass a single residual evaluation with dual-number arithmetic (~2× the cost of Float64). At the production Stage-5 mesh (N = 1480) that is ~123 passes ≈ 250× the cost of one plain residual evaluation, vs ~1480× for dense FD — AD is **~6× faster per Jacobian** AND exact to machine precision. Brady et al. 2021 discusses applying AD specifically to Newman BAND for the same reason.

#### 9.3.3 Sparsity

The production kernel (§9.5.3) converts the dense AD Jacobian to sparse once per iteration (`sparse(J)`) and hands it to the persistent LU workspace (§9.5.2). At the production DOF counts (400–1480) the dense fill + conversion is a few milliseconds and not worth optimizing away. If you push far past this size (N ≳ 5000), switch to a colored sparse-AD fill (the FV Jacobian is banded, so the color count grows with the per-cell DOF count, not with N_CL) — but measure first; at every mesh used in this guide the linear solve, not the Jacobian, dominates the iteration cost (§9.5.10).

### 9.4 Why Naive Newton Fails on This System — Required Numerical Treatments

The Stage 5 system has `cond(J) ≈ 10²⁰` — a consequence of the coupled NP + Ohm's law formulation that is standard in Newman porous-electrode models but requires careful numerical treatment. A textbook damped Newton (FD Jacobian, row-max scaling, residual-norm line search) fails on this system in several independent ways. This section is the complete list of facts and required treatments; the production stack in §9.5 is designed around them. **Implement all of these from the start.**

**1. Monolithic Newton.** All DOFs (species, gas, φ_s, φ_L) in a single residual vector, single Jacobian, single linear solve. This captures the full c↔φ_L↔φ_s kinetic coupling in one step. Picard iteration (alternating subproblems) misses this coupling and requires 5–10 outer rounds. Use monolithic Newton exclusively.

**2. Exact AD Jacobian (not FD).** The kinetic feedback `∂R_CO2R/∂φ_s ≈ 40` is 6 orders of magnitude smaller than the φ_s divergence diagonal `σ/dx ≈ 4×10⁷`. FD with `eps=1e-7` produces truncation error ~4, comparable to the feedback term itself — the kinetic coupling is buried in FD noise. ForwardDiff AD computes these to machine precision (§9.3.2; Brady et al. 2021).

**3. The φ_L block has exactly one algebraic null direction — fix the gauge, then direct Newton works.** The per-cell φ_L residual is algebraically `−F × Σᵢ zᵢ × F_species` (buffer reactions conserve charge), so the φ_L rows are nearly linearly dependent on the species rows and the φ_L DOFs admit one null direction: a uniform shift. The Dirichlet override `φ_L[N_CL] = 0` (item 6 below) replaces cell N_CL's residual with the gauge equation and removes the singularity. **With the gauge fixed, an exact continuation predictor keeping the iterate in-basin (§9.5.4–9.5.5), and damping measured on Newton corrections (§9.5.3), a direct sparse-LU Newton solve `J \ (−F)` is robust at every stage and mesh** — this is the production path, validated across four independent builds. Normal-equations regularization (`(JᵀJ + λI) du = −JᵀF`, Levenberg-Marquardt) also works and was used by early builds, but it is not required, converges more slowly (tens of iterations per voltage from cold seeds), and, run cold-seeded on the multi-rooted Stage-5 system, can land on the wrong solution branch (§9.5.5). Do not use it as the production kernel.

**4. Log-step clamp (max 5 log-units).** From a poor starting point, early iterations can produce large species updates. Clamp `max |Δu_species| ≤ 5.0` log-units, limiting concentration changes to `exp(5) ≈ 150×` per step. This is a guard, not a crutch — with the continuation predictor of §9.5.4 the clamp essentially never fires in production sweeps.

**5. No row scaling.** Row scaling (dividing each Jacobian row by its max) crushes the kinetic feedback (`∂R/∂φ_s ≈ 40`) against the divergence diagonal (`~4×10⁷`), making the coupling numerically invisible (~10⁻⁶ relative). The Newton step then ignores how the kinetics respond to φ_s and diverges. The production kernel scales *the step norm* by per-DOF iterate-derived weights (§9.5.3) — it never rescales Jacobian rows and never takes a scaled residual norm.

**6. Ghost-cell φ_s BC (not override); φ_L gauge override (not ghost).** `φ_s = V_applied` at the current collector enters through the leftmost face flux `i_s[1] = −σ(φ_s[1] − V_applied)/dx_half`, keeping cell 1's continuity equation intact — overriding cell 1 decouples the boundary current and breaks charge conservation (symptom: `|Δi|/i ≈ 2`). `φ_L = 0` at `x = L_CL` is the exact opposite: it *replaces* cell N_CL's charge-balance residual, fixing the gauge (item 3). The rule: **override φ_L (gauge fix, no physical flux to absorb the BC), ghost φ_s (physical electronic current flux).** See §3.4 and §11 Stage 5 gotchas #6–7.

**7. NP migration at the right boundary.** The Sherwood BC from Stages 1–4 omits migration. In Stage 5, CO₃²⁻ carries the ionic current through the CL/electrolyte boundary via migration. Use `np_flux(c[N], c_eq, φ_L[N], 0, D_eff, z, D_eff/k_MT)`. Without this: ionic charge conservation violated by 2–20%, φ_L profile unphysical.

**8. Stage 5 runs with fitted kinetics from the start.** The fitted kinetic parameters from §17 (run after Stage 4) must be loaded *before* the Stage 5 sweep begins — the anchor solve and every continuation step use them. Do **not** seed Stage 5 from a Stage-4 or fit-step cache: the sweep is seeded once, at V_c = 0, from the constructed equilibrium guess of §9.5.5, and every subsequent voltage is reached by the continuation predictor from the previous converged state. Expect ~3–5 Newton iterations at the anchor and ~2 per continuation step.

### 9.5 Production Solver Stack — All Stages

> **🛑 If you only read one section of §9, read this one.** This is the production solver stack used for **every stage** (Stages 1–5 + the Fit step) at the production default mesh (N_CL=80, N_GDL=120). Per-stage DOF counts are 400 (Stages 1–2), 1200 (Stages 3–4 + Fit), and 1480 (Stage 5). The same kernel and the same driver pattern work at every stage; only the residual function and the anchor recipe differ.
>
> The stack is an **error-oriented affine-covariant damped Newton** (Deuflhard, *Newton Methods for Nonlinear Problems*, Springer 2011, §3.3 / Alg. 3.13 — the NLEQ-ERR family) wrapped in a **tangent-predictor continuation** over the voltage grid. Three properties distinguish it from the textbook kernels of §9.4:
>
> 1. **No norm of the residual is ever taken inside the kernel.** Damping, the step-acceptance test, and termination all measure Newton *corrections* in an iterate-derived norm. There is no residual weight vector to mis-set — the entire class of "one residual row is invisible to the line search" bugs (e.g. a ~1e-5 Darcy continuity residual divided by a ~1e5 Pa DOF scale) cannot occur, because no residual is ever divided by anything.
> 2. **Termination is an error estimate, not a step-size heuristic or an iteration count.** The kernel stops when its estimate of the *remaining solution error* drops below `rtol`. Accuracy is a dial, and the residual guarantee is enforced separately, after the solve (§9.5.6).
> 3. **The continuation predictor uses the true local sensitivity `du/dV`**, obtained by one extra back-substitution against the factorization that is already in memory — 20–500× more accurate than carrying the previous state forward, at ~5 % of the cost of one Newton iteration.

#### 9.5.1 The five components (in dependency order)

```
ForwardDiff AD Jacobian     →  §9.3.2   (dense jacobian! → sparse(J))
persistent sparse LU        →  newton_affine.jl   (§9.5.2)
affine-covariant Newton     →  newton_affine.jl   (§9.5.3)
tangent-predictor driver    →  run_stageN.jl      (§9.5.4 – 9.5.5)
post-solve residual check   →  run_stageN.jl      (§9.5.6)
```

Dependencies: `ForwardDiff.jl` plus the stdlib `SparseArrays` and `LinearAlgebra`. The linear solver is Julia's built-in sparse LU (UMFPACK, via `lu` / `lu!`) — **no external linear-algebra binary, no MPI**. The whole stack is ~300 lines in one file (`newton_affine.jl`) plus a ~100-line driver per stage.

#### 9.5.2 Persistent sparse LU workspace

`lu(J)` performs a symbolic analysis (pivot ordering from the sparsity pattern) followed by the numeric factorization. Across a continuation sweep the sparsity pattern is **fixed** — same `colptr`/`rowval` at every Newton iteration at every voltage — so the symbolic phase only needs to run once. `lu!(F, J)` refreshes the numeric factorization in an existing object, reusing the symbolic analysis. The workspace also keeps the factorization **live** after each solve so that the kernel (§9.5.3) and the tangent predictor (§9.5.4) can back-substitute against it (`plu.F \ b`) without refactorizing.

```julia
mutable struct AffineLU
    F::Union{Nothing,SparseArrays.UMFPACK.UmfpackLU{Float64,Int}}
    pattern_nnz::Int
end
AffineLU() = AffineLU(nothing, -1)

"""
    aff_factor_solve!(plu, J, b) -> (x, ok)

Factorize `J` (reusing the symbolic analysis when the sparsity pattern is
unchanged) and solve `J x = b`.  Leaves the factorization live in `plu` so
`aff_resolve` can back-substitute against it.
"""
function aff_factor_solve!(plu::AffineLU, J::SparseMatrixCSC{Float64,Int},
                           b::Vector{Float64})
    if plu.F === nothing || plu.pattern_nnz != nnz(J)
        try
            plu.F = lu(J); plu.pattern_nnz = nnz(J)
        catch
            return (zeros(length(b)), false)
        end
    else
        try
            lu!(plu.F, J)                     # refresh values, reuse symbolic
        catch
            try
                plu.F = lu(J); plu.pattern_nnz = nnz(J)
            catch
                return (zeros(length(b)), false)
            end
        end
    end
    x = plu.F \ b
    return (x, all(isfinite, x))
end

"Back-substitute against the live factorization (valid right after aff_factor_solve!)."
@inline aff_resolve(plu::AffineLU, b::Vector{Float64}) = plu.F \ b
```

The `catch → rebuild` fallback covers the rare case where `lu!` refuses a numerically difficult refresh; the `pattern_nnz` check covers a mesh/stage change mid-process.

#### 9.5.3 Affine-covariant damped Newton kernel

The heart of the stack. Per Newton iteration `k`, with iterate `u_k`:

1. **Weights.** Rebuild the per-DOF weight vector from the current iterate:
   `w_j = |u_j| + afrac · rms(|u| over j's DOF block)`, with `afrac = 1e-3` and the six natural DOF blocks of the layout (liquid log-c, gas GDL, gas CL, φ_s GDL, φ_s CL, φ_L). The block floor keeps a DOF passing through zero (the gauged φ_L, or φ_s at the anchor) from blowing up the norm. **If an entire block is *identically* zero the formula is 0/0.** That happens on the first iteration of the Stage-5 anchor: the §9.5.5 recipe sets `φ_s = V_applied_SHE = 0` on GDL+CL and `φ_L = 0` on the CL, so both potential blocks are exactly zero and `‖d‖_w` returns `NaN` on the very first Stage-5 solve. **Fall back to a unit weight for any block whose rms is exactly zero.** Both blocks are non-zero from the first Newton step onward, so this is the only place a weight is not iterate-derived. Apart from that fallback there are **no hand-set per-physics constants**: the weights come from the iterate itself, every iteration. The weighted norm used throughout is `‖d‖_w = sqrt(Σ (d_j/w_j)² / N)`.
2. **Ordinary Newton correction.** Build the AD Jacobian (§9.3.2), factorize + solve `J_k Δu = −F(u_k)` via `aff_factor_solve!`. Clamp the liquid log-DOF block of `Δu` at `max_log_step = 5.0` (§9.4 item 4).
3. **A-priori damping.** Predict the damping factor α from the quantities observed in the previous iteration (a Kantorovich-style estimate — see the code); clamp to `[α_min, 1]`. At the first iteration α = `alpha_init`.
4. **Trial + simplified correction.** Evaluate `F(u_k + α·Δu)`, then back-substitute **against the still-live factorization of `J_k`**: `J_k Δū = −F(u_k + α·Δu)`. `Δū` is the *simplified Newton correction* — the affine-covariant estimate of the error remaining at the trial point. Cost: one residual + one back-substitution (~5 % of a full iteration); **never** a refactorization.
5. **Monotonicity test.** Accept the trial iff `‖Δū‖_w ≤ (1 − α/2)·‖Δu‖_w`. On acceptance, an a-posteriori estimate may propose a substantially larger α for an immediate retry; on rejection, halve α and go to 4. This is a natural (affine-covariant) monotonicity test — note it compares two *correction* norms, never residual norms.
6. **Termination.** Stop when `‖Δū‖_w ≤ rtol`. The last computed `Δū` is then **added to the iterate for free** (it is already the best available correction), giving one extra order of convergence at zero cost.

```julia
"""
    newton_affine!(u, res!; ly, plu, rtol=1e-8, max_iter=40, alpha_init=1.0,
                   afrac=1e-3, max_log_step=5.0, n_log_dofs=0,
                   Jdense=nothing, cfg=nothing, final_correction=true)
        -> (converged, iterations, residual, err_est, theta, alpha_final,
            n_res, n_bsub, reason)

Affine-covariant damped Newton (error-oriented).  `rtol` is the accuracy
dial: stop when the estimated remaining solution error (weighted norm of the
simplified Newton correction) falls below it.  Mutates `u`.  `residual` in
the return is ‖F‖∞ at exit — reported for the §9.5.6 check, never used as a
termination or damping test.
```

The complete reference implementation (~180 lines) is `newton_affine.jl`; reproduce it as written. Key excerpts:

```julia
# a-priori damping from the previous iteration
if it > 1 && isfinite(du_prev_n) && isfinite(dubar_n)
    @inbounds for j in 1:N; scr[j] = dubar_prev[j] - du[j]; end
    den = wnorm(scr, w) * du_n
    den > 0 && (alpha = min(1.0, (du_prev_n * dubar_n) / den * alpha_prev))
end
alpha = clamp(alpha, alpha_min, 1.0)

# trial, simplified correction, monotonicity test
for _ in 1:max_damp
    @inbounds for j in 1:N; u_tr[j] = u[j] + alpha * du[j]; end
    res!(F_tr, u_tr)
    if all(isfinite, F_tr)
        dubar = aff_resolve(plu, -F_tr)            # back-substitute only
        if all(isfinite, dubar)
            dbn = wnorm(dubar, w)
            if dbn <= (1.0 - alpha/2) * du_n       # natural monotonicity test
                # a-posteriori α proposal (retry only if ≥ 4× larger) ...
                u .= u_tr; F .= F_tr
                accepted = true; break
            end
        end
    end
    alpha *= 0.5
    alpha < alpha_min && break
end

# termination on the error estimate; final correction applied free
if dubar_n <= rtol
    final_correction && (u .+= dubar)
    ...
end
```

**Why correction norms and not residual norms.** The residual components of this system span ~17 orders of magnitude and have physically incommensurate units (mol/m²/s, kg/m²/s, A/m²). Any residual-norm merit function therefore needs a hand-built weight vector — and a single wrong entry silently hides a residual row from the line search (the classic instance: weighting the ~1e-5 kg/m²/s Darcy continuity row by the ~1e5 Pa pressure DOF magnitude divides it to ~1e-10; α collapses and Stage 4 stalls at ~66/89 voltages). Corrections, by contrast, live in solution space, where the iterate itself provides the natural scale. The affine-covariant construction makes the algorithm's decisions invariant under row scalings of F — the whole weight-tuning problem disappears rather than being solved.

**Why an error estimate and not an iteration count.** With a good predictor the first Newton step is nearly converged, and any fixed per-step iteration count either wastes iterations (predictor good) or under-converges (predictor stressed near the kinetics turn-on). `‖Δū‖_w` measures the thing you actually care about — how far the iterate is from the solution — so the kernel takes 1 iteration where 1 suffices and 3 where 3 are needed, with a bound instead of a hope. Validated behaviour: 2.0 iterations/step average across all stages and meshes at `rtol = 1e-8`, with ‖F‖∞ ≤ ~1e-7 everywhere (tighter than any fixed-count scheme tested, at equal or lower cost).

#### 9.5.4 Tangent continuation predictor

At a converged state `(u, V)` with the factorization of the final Jacobian still live in the workspace, the branch sensitivity is one back-substitution away:

```julia
"""
    tangent_predictor!(dudV, plu, res_at_V!, u, V; h=1e-6) -> ok

Fill `dudV` with du/dV = -J⁻¹ (∂F/∂V), using the live factorization.
`res_at_V!(F, u, V)` evaluates the residual at an arbitrary V.
Cost: two residual evaluations (central difference for ∂F/∂V) + one
back-substitution.
"""
function tangent_predictor!(dudV, plu, res_at_V!, u, V; h=1e-6, Fp=..., Fm=...)
    res_at_V!(Fp, u, V + h);  res_at_V!(Fm, u, V - h)
    @. Fp = (Fp - Fm) / (2h)                     # ∂F/∂V
    d = aff_resolve(plu, -Fp)
    all(isfinite, d) || return false
    dudV .= d
    return true
end
```

The predictor step is then simply `u .+= dV .* dudV`. If the tangent is unavailable (factorization not live, or a non-finite solve), fall back to the closed-form φ_s shift: add `dV` to every φ_s DOF and leave everything else unchanged — the φ_s boundary value moves by exactly `dV`, so this component of the step is known exactly.

**Structure of ∂F/∂V.** In Stage 5, V enters the system only through the φ_s ghost-cell BC at the current collector — `∂F/∂V` has exactly **one nonzero entry**, and the tangent is essentially free. In Stages 1–4, V enters the simplified kinetics at every CL cell (`φ_s_local = V_applied`), so `∂F/∂V` is dense within the kinetically-active liquid rows (~160/400 entries at Stages 1–2, ~320–400/1200 at Stages 3–4); the cost is unchanged — two residuals and one back-substitution.

**Measured accuracy** (production mesh, full 89-V sweeps): the tangent predictor lands 20–50× closer to the converged solution than the carry-forward + φ_s-shift predictor at a typical step (median ~30×), and up to ~500× closer at the kinetics turn-on (V_RHE ≈ −1.15) — exactly where the carry-forward predictor is weakest. It was better at **every** continuation step in every build tested, and the branch check (§9.5.5) confirms it does not alter which solution branch the sweep lands on.

> **Do not confuse the tangent with a higher-order extrapolating predictor.** Extrapolation from *past states* (e.g. quadratic Lagrange through the previous three solutions) was tested and rejected: it extrapolates φ_L across the kinetics turn-on and can walk the iterate onto a spurious branch. The tangent is the *local* derivative of the current branch at the current point — it follows the bend instead of extrapolating across it, and is safe by construction and by measurement.

#### 9.5.5 Unified continuation driver — every stage

Every stage runs the same driver pattern; only the residual and the anchor recipe differ.

```
anchor:  build the stage's anchor guess  →  newton_affine! (alpha_init = 1e-2,
         max_iter = 200)  →  tangent_predictor!
sweep:   for each subsequent grid voltage:
             u .+= dV .* dudV                     (tangent predictor)
             newton_affine! (alpha_init = 1.0, max_iter = 40)
             post-solve residual check (§9.5.6)
             save to cache;  tangent_predictor!
recovery: on a failed step, roll back, halve dV, solve the two half-steps
          with alpha_init = 1e-2, then continue.
```

**Canonical voltage grid** (used for every stage's cache and every sweep):

```julia
# 89 evenly spaced points in V vs RHE@pH8, converted to V vs SHE for the solver
const V_RHE_grid = collect(range(-0.300, stop = -2.500, length = 89))   # step = 0.025 V
const V_SHE_grid = V_RHE_grid .- V_RHE_to_SHE_offset                     # offset = 0.51305 V at 50°C, pH=8
# Range chosen so the 5 Lees et al. experimental voltages (−0.816, −1.205, −1.644, −2.079, −2.413 V RHE)
# all sit inside the grid with at least ~3 grid points of margin on either side for warm-starts.
```

Every cache file (Stages 1–5) uses exactly these 89 voltages; "89/89 converged" in §14 and §17 refers to this grid.

**Anchor recipes per stage:**

| Stage | Anchor voltage | Anchor guess | Sweep |
|---|---|---|---|
| 1, 2 | −0.300 V RHE (mildest grid point) | every cell at the exact bulk equilibrium (§9.6): `u[liquid_idx(ix,i)] = log(c_eq[i])` | tangent continuation down the remaining 88 grid points |
| 3, 4 | −0.300 V RHE | equilibrium liquid + gas at inlet composition and `p_total` in every GDL and CL cell | same |
| Fit | — | per-voltage polish, warm-started from the Stage-4 cache (§17.3); same kernel, no continuation | 5 experimental voltages per NM trial |
| 5 | **V_c = 0** (zero overpotential) | equilibrium liquid + inlet gas + `φ_s = 0` on GDL+CL + `φ_L = 0` — **constructed from physics, never from a cache** | pre-anchor ramp V_c = 0 → first output point in −0.025 V steps, then the 89 output points |

Stages 1–4 are **self-anchoring**: each converges from the equilibrium-based guess in ~5 iterations to ‖F‖∞ ≈ 1e-13, so the previous-stage warm-start chain of §10.5 is an optional convenience, not a requirement.

**Why Stage 5 specifically anchors at V_c = 0.** Stage 5 introduces φ_s and φ_L DOFs and is **multi-rooted**: the same `F(u, V) = 0` admits two solution branches near each output voltage — a "high-CO" branch matching the experimental data of §16.1, and a "low-CO" branch ~20 % below it. Both produce small residuals and pass charge conservation to ≤1e-9 — only the polarization curve and the Δφ_L diagnostic distinguish them (low-CO: Δφ_L ≈ 3 mV instead of ~15 mV). Independent per-voltage solves seeded from a cache can land on the low-CO root. At V_c = 0 the overpotential is zero, the kinetics are negligible, and the solution is the trivial unique branch; continuation from there tracks the high-CO branch at every voltage without ever leaving its basin. **Stages 1–4 have no φ_L bistability and need no special anchor** — the mildest grid voltage from equilibrium is in-basin by construction.

The V_c chain: `V_c = V_RHE − V_RHE_to_SHE_offset`; pre-anchor ramp `0, −0.025, …` down to the first output point (~32 points), then the 89 output points in cathodic order (~121 total). Sub-step recovery (halve `dV_c`, `α_init = 1e-2`) is wired in but empirically untriggered in production runs.

**Branch check (run once per build):** the currents produced by the anchored continuation sweep must agree with a per-voltage reference (an earlier validated build, or the 5 fit points) to well under 1 % — a wrong-branch sweep is off by ~20 %, not 0.001 %.

#### 9.5.6 Post-solve residual assertion

`rtol` bounds the estimated *solution* error in the weighted correction norm; it does not directly bound ‖F‖∞. The residual guarantee is therefore enforced as an explicit **post-solve assertion**, outside the termination test:

```julia
res_at_V!(Fchk, u, V)
if maximum(abs, Fchk) > F_INF_TOL          # production: 1e-6
    # one extra Newton iteration from the converged state, then re-check
    newton_affine!(u, res!; ly, plu, rtol = 1e-10, max_iter = 3, alpha_init = 1.0)
    res_at_V!(Fchk, u, V)
    maximum(abs, Fchk) > F_INF_TOL && error("residual assertion failed at V=$V — halve dV and re-approach")
end
```

At `rtol = 1e-8` the assertion is pure insurance: measured max ‖F‖∞ across four builds × 89 voltages × all stages is ≤ ~1e-7, an order of magnitude inside the tolerance, and the recovery branch never fires. Keeping the check *outside* termination is deliberate: folding a hard ‖F‖∞ floor into the stopping test forces extra iterations at every voltage (measured +37–47 % Newton iterations in builds that gated termination on ‖F‖∞), while the assertion costs one residual evaluation per emitted voltage.

#### 9.5.7 Production settings (do not change without measuring)

| Parameter | Value | Why |
|---|---|---|
| `rtol` | **1e-8** | The accuracy dial. Measured trade at production mesh (Stage 5, per-step averages): 1e-3 → 1.03 iters/step, median ‖F‖∞ ~2e-6; 1e-6 → 1.28 iters/step, ~2e-7; **1e-8 → 2.0 iters/step, ~4e-8 with max ≤ ~1e-7**. Production uses 1e-8 so the §9.5.6 assertion never fires. |
| `F_INF_TOL` (assertion) | **1e-6** | Post-solve ‖F‖∞ guarantee at every emitted voltage (§9.5.6). Not a termination criterion. |
| Anchor `alpha_init` | **1e-2** | The anchor guess is approximate; start damped. The a-priori estimator raises α within 1–2 iterations. |
| Continuation `alpha_init` | **1.0** | The tangent predictor is in-basin; full steps from the start. Backtracking is available but does not fire in production sweeps. It fires on wild NM trial kinetics in the Fit step and recovers *many* of them, **but not all**: at α_c ≈ 0.13 (a routine 1.5× NM probe, implying 29–80 A/cm²) a warm-started polish reports `damping_stalled` at the two most cathodic voltages at both `alpha_init = 1.0` and `alpha_init = 1e-2`. When a direct polish fails, ramp the kinetic parameters from the last feasible vector to the trial vector in 2 → 4 → 8 sub-steps, re-solving at each — the same continuation idea applied along the parameter direction instead of V. Trials that still fail after the ramp are genuinely extreme and should be rejected. |
| `alpha_min` | **1e-8** | Below this the damping search has stalled; abort the step and sub-step. |
| `max_iter` | **40** (sweep) / **200** (anchor) | Generous caps; typical usage is 2 and ~5. |
| `max_log_step` | **5.0** | Limits a single step to `exp(5) ≈ 150×` per concentration (§9.4 item 4). |
| `afrac` | **1e-3** | Weight-floor fraction per DOF block (§9.5.3 step 1). Insensitive; do not tune per-physics. |
| `dV_c` step | **−0.025 V** | Matches the grid spacing; the pre-anchor ramp uses the same step. |
| Sub-step recovery | halve `dV`, `alpha_init = 1e-2` | Safety net; untriggered in production runs. |

#### 9.5.8 Critical reproducibility checklist

> Every box must be checked. The Stage 5 sweep does NOT silently warn when it lands on the wrong branch — only the polarization curve and the Δφ_L diagnostic distinguish the two roots.

- [ ] `Project.toml` includes `ForwardDiff` (plus stdlib `SparseArrays`, `LinearAlgebra`). No external linear-solver binary.
- [ ] Residual chain fully type-generic (§9.3.2 checklist) — AD validated by a finite-difference spot check on a mild-voltage state.
- [ ] Persistent LU implemented (§9.5.2) — first call factorizes symbolically + numerically; subsequent calls `lu!` only; factorization stays live for back-substitution.
- [ ] Affine-covariant kernel implemented (§9.5.3) — correction-norm damping and monotonicity, error-estimate termination at `rtol = 1e-8`, final correction applied, log-clamp on the liquid block. **No residual norm anywhere in the kernel.**
- [ ] Tangent predictor implemented (§9.5.4) with the φ_s-shift fallback.
- [ ] Driver anchors per the §9.5.5 table — Stage 5 at **V_c = 0** from the constructed physics guess, never from a cache.
- [ ] Post-solve residual assertion (§9.5.6) wired at every emitted voltage, `F_INF_TOL = 1e-6`.
- [ ] Sub-step recovery (halve `dV`, `alpha_init = 1e-2`) wired in.
- [ ] **Δφ_L diagnostic check.** At converged Stage 5 states with `i_total > 500 mA/cm²`, `|φ_L[1] − φ_L[N_CL]|` should be **5–25 mV** (typically ~15 mV at V_c = −2 with 1 M CsHCO₃). Values < 5 mV → landed on the low-CO branch. Values > 30 mV → numerical artefact, halve `dV_c` near that V and re-sweep.
- [ ] **Polarization smoothness check.** Plot i_CO vs −V_SHE. It should run smoothly 0 → peak → tail. Measure it as a **sign-change count on the first difference**, not as a percentage step: require **exactly 1 sign change in di_CO/dV** (one rise, one rollover) and **0 sign changes in di_H₂/dV** (strictly monotone). Do *not* use a "spikes larger than ±5 %" rule on the raw first difference — near the i_CO peak the differences pass through zero, so any ratio metric explodes there and reports tens of per cent on a demonstrably smooth curve. Extra sign changes away from the peak, or a φ_L that flips sign between neighbouring voltages, are the real branch-hop signatures; halve `dV_c` for the pre-anchor ramp and re-sweep.

> **Sign of Δφ_L.** Converged Δφ_L is negative (`φ_L[1] ≈ −15 mV`, gauge `φ_L[N_CL] = 0`). The diagnostic checks the magnitude. Wrong-sign-with-correct-magnitude is a sign bug in `i_L` assembly, not a basin issue.

#### 9.5.9 First-build sanity tests (run after each component is in place)

Execute these in order. If any step fails, fix it before continuing — failures cascade.

| Step | Test | Pass criterion |
|---|---|---|
| 1 | Persistent LU: solve the same Jacobian twice | Second call skips the symbolic phase (measurably faster); `plu.F \ b` back-substitutes without refactorizing |
| 2 | Stage-1 anchor at −0.300 V from the equilibrium guess, `newton_affine!` with `alpha_init = 1e-2` | Converges in ≤ ~6 iterations; ‖F‖∞ ≤ 1e-9 after the final correction |
| 3 | `∂F/∂V` structure probe at a converged state | Stage 5: exactly **1** nonzero entry. Stages 1–2: ~160/400. Stages 3–4: ~320–400/1200 |
| 4 | Predictor A/B over one sweep: record `‖u_pred − u*‖` for tangent vs carry-forward + φ_s shift | Tangent better at every step; median improvement > 10× |
| 5 | Full Stage-5 continuation: anchor at V_c = 0 (`alpha_init = 1e-2`), then the chain at `rtol = 1e-8` | All chain points converge at ~2 iterations/step; anchor ≤ ~5 iterations to ‖F‖∞ ≤ 1e-9 |
| 6 | Post-solve assertion audit | ‖F‖∞ ≤ 1e-6 at all 89 emitted voltages without the recovery branch firing (typical max ~1e-7) |
| 7 | Δφ_L + smoothness (§9.5.8) | `|Δφ_L|` in [10, 20] mV at high current; polarization monotone-smooth 0 → peak → tail |
| 8 | Compare polarization to the experimental data (§16.1) at V_RHE ∈ {−0.816, −1.205, −1.644, −2.079, −2.413} | Within ~30 % of i_CO_exp / i_H₂_exp (the fit uses 5 points; remaining sweep voltages are predictions) |

**If step 2 fails (anchor doesn't converge):** check the residual encoding — at the mild anchor the solution is near-equilibrium and Newton should walk in directly. Common causes: wrong sign on Sechenov H_eff at the GDL/CL boundary; Cs⁺ back-derive sign; wrong electroneutrality offset (`C_fix · ε_I`); a type-generalization gap that corrupts the AD Jacobian (§9.3.2).

**If step 5 produces peak i_CO ≪ expected:** you landed on the low-CO branch. Verify §9.5.8 — most commonly: the sweep was seeded from a cache instead of the V_c = 0 anchor, or the anchor builder wrote V in the wrong reference frame into φ_s (§11 Stage 5 gotcha #9a).

**If step 5 has a spike at a single V:** the step was too large for a bend in the branch at that V. The sub-step recovery should catch it automatically; if it doesn't, the recovery is mis-wired.

#### 9.5.10 Performance characterization

Measured on the reference build (Apple M2 Pro, Julia 1.11, single process, production mesh N_CL=80 / N_GDL=120, `rtol = 1e-8`, 89-V sweeps):

| Stage | DOFs | Sweep wall | Newton iters/step | max ‖F‖∞ |
|---|---|---|---|---|
| 1 | 400 | ~2 s | 2.0 | ~1e-12 |
| 2 | 400 | ~2 s | 2.0 | ~3e-12 |
| 3 | 1200 | ~5 s | 1.9 | ~1e-11 |
| 4 | 1200 | ~5 s | 1.9 | ~1e-11 |
| 5 (incl. anchor + pre-ramp, 121 steps) | 1480 | ~11 s | 2.0 | ~1e-7 |
| Fit (NM, ~180 evals × 5 polishes) | 1200 | ~80 s | ~2 per polish | — |

Per-iteration cost is dominated by the sparse LU numeric factorization; the AD Jacobian fill and the residual evaluations are minor at these sizes, and the tangent predictor plus simplified correction add ~5–10 % overhead while cutting the iteration count roughly in half relative to a carry-forward predictor with a fixed two-iteration schedule. Newton iteration counts are deterministic; wall times vary by machine.

### 9.6 Equilibrium Initial Guess — Solve It Exactly

**This is the most important implementation detail for avoiding large initial residuals.**

The buffer chemistry reactions are extremely stiff. The forward and reverse rates are both large (k₅ = 6×10⁶ m³/(mol·s)), but nearly cancel at equilibrium. If your initial guess is even slightly off equilibrium, the buffer source terms `S_buf,i` will be O(10³–10⁵) mol/m³/s — orders of magnitude larger than the transport fluxes. Newton will fail or take hundreds of iterations even at `α_buf = 0.001`.

**The fix: solve the charge balance exactly before you start.**

```julia
function solve_buffer_equilibrium(c_Cs_mol_m3=1000.0, C_total_mol_m3=1000.0)
    # All concentrations in mol/m³ internally

    function charge_balance(log_cH)
        cH    = 10.0^log_cH
        cOH   = K_w / cH
        denom = 1.0 + K_2/cH + K_2*K_3/cH^2
        cCO2  = C_total_mol_m3 / denom
        cHCO3 = K_2 * cCO2 / cH
        cCO3  = K_3 * cHCO3 / cH
        return cH + c_Cs_mol_m3 - cOH - cHCO3 - 2*cCO3  # charge balance residual
    end

    using Roots
    log_cH = find_zero(charge_balance, (-6.0, 0.0), Bisection())
    cH = 10.0^log_cH
    cOH = K_w / cH
    denom = 1.0 + K_2/cH + K_2*K_3/cH^2
    cCO2  = C_total_mol_m3 / denom
    cHCO3 = K_2 * cCO2 / cH
    cCO3  = K_3 * cHCO3 / cH
    return (H=cH, OH=cOH, HCO3=cHCO3, CO3=cCO3, CO2=cCO2,
            Cs=c_Cs_mol_m3, pH=-log10(cH/1000.0))
end
```

For 1M CsHCO₃ at 50°C, this gives approximately (bicarbonate equivalence point pH = (pKa1 + pKa2)/2 = (6.288 + 10.171)/2 = 8.23):
```
c_H    ≈ 5.9e-6  mol/m³   (pH ≈ 8.23)
c_OH   ≈ 9.3e-3  mol/m³
c_HCO3 ≈ 978     mol/m³
c_CO3  ≈ 11      mol/m³
c_CO2  ≈ 11      mol/m³
c_Cs   = 1000    mol/m³
```

**Verify by equilibrium product checks:**
- `c_H × c_OH = 5.9e-6 × 9.3e-3 = 5.49e-8` ≈ K_w = 5.48e-8 ✓
- `c_H × c_HCO3 / c_CO2 = 5.9e-6 × 978 / 11 = 5.25e-4` ≈ K_2 = 5.16e-4 ✓
- `c_H × c_CO3 / c_HCO3 = 5.9e-6 × 11 / 978 = 6.64e-8` ≈ K_3 = 6.74e-8 ✓

**Verify that your equilibrium solution satisfies all 5 buffer rates simultaneously:**
```julia
S = buffer_sources(c_eq_vec)
@assert maximum(abs.(S)) < 1e-6   # mol/m³/s — must be near zero
```

If `maximum(abs.(S)) > 1e-4`, your K values or forward rates are inconsistent — recheck the unit conversions (mol/L vs mol/m³).

**Initial guess for the full DOF vector:**
```julia
ln_c_eq = log.(c_eq_vec)   # 5-vector of log concentrations
u0 = zeros(5 * N_CL)
for ix in 1:N_CL
    u0[5*(ix-1)+1 : 5*ix] .= ln_c_eq
end
```

Every cell starts at exactly the bulk equilibrium. With `α_buf = 0` (buffer off) and `α_elec = 0` (kinetics off), the residual at this initial guess is **identically zero** (only transport remains, and there are no gradients). This is the ideal starting point — Newton converges in 0 iterations.

**Why log-concentrations help here:** Even if you perturb `u0` slightly (e.g., numerical noise), `exp(ln_c)` always returns the exact positive value, and the buffer Jacobian is smooth. With raw concentrations, a small negative step can give `c < 0` and `log(c)` throws a domain error.

### 9.7 Bootstrap Protocol

With the exact equilibrium guess (§9.6) and the continuation driver (§9.5.5), bootstrap is short — there are no ramps in the production path:

```
Step A: Solve the buffer equilibrium exactly (§9.6).  With the equilibrium
        guess, the transport-only residual is identically zero — this is a
        constructed starting point, not a solve.

Step B: Anchor solve for the stage (§9.5.5 table):
        Stages 1–4  →  −0.300 V RHE from the equilibrium(+inlet-gas) guess;
        Stage 5     →  V_c = 0 from the constructed physics guess, with
                       fitted kinetics loaded BEFORE the anchor.
        newton_affine! with alpha_init = 1e-2; ~5 iterations.

Step C: Continuation sweep down the grid with the tangent predictor
        (§9.5.4), post-solve assertion at every emitted voltage (§9.5.6).
```

**Debugging fallback (not part of the production path):** if an anchor refuses to converge even after the §9.5.9 step-2 checks, a continuation parameter on the source terms can isolate the culprit — scale the buffer sources by `α_buf ∈ [0, 1]` and/or the kinetic sources by `α_elec ∈ [0, 1]` and ramp each from 0 → 1 at the anchor voltage. In a correct implementation this is never needed; needing it is evidence of a residual-encoding bug, and the ramp is a bisection tool for finding it.

**Recovery option — polish high-residual voltages instead of restarting.**
If a completed sweep shows a small number of voltages where the §9.5.6 assertion had to fire (or a diagnostic pass finds ‖F‖∞ above tolerance) and the polarization curve is visibly jagged *only at those points*, you do NOT need to throw out the sweep. Re-solve just the bad voltages from their cached states:

```julia
for V in bad_Vs
    u = load_cache(s5_cache, V)                    # start from the cached solution
    newton_affine!(u, res!; ly, plu, rtol = 1e-10, max_iter = 20, alpha_init = 1.0)
    save_cache!(s5_cache, V, u)
end
```

This works because the cached solution is already in the correct basin — the polish tightens it in 1–3 iterations.

**When polishing is NOT the right fix:** if the entire curve is jagged, or adjacent voltages have opposite-sign φ_L, the solutions are on a *different branch* (continuation bug, wrong anchor builder, missing NP migration in the right BC). Polishing will just drive ‖F‖ down on the wrong branch — fix the root cause and re-sweep from the anchor. Polishing is a smoothing tool for isolated numerical outliers, not a recovery for structural bugs.

**What NOT to do (typical failure patterns that produce jagged or off-branch polarization curves):**
- Solving each Stage-5 voltage independently from a cache entry (`output_stage4/cache/` or `output_fit/cache/`) instead of continuing from the anchor — lands on the low-CO root at some or all voltages. That root passes small-residual *and* charge-conservation checks, and is identifiable only by Δφ_L ≈ 3 mV instead of ~15 mV and a peak i_CO ~20 % below experiment.
- Extrapolating the predictor from *past states* (e.g. quadratic Lagrange through the previous three solutions) — it extrapolates φ_L across the kinetics turn-on and walks the iterate onto the low-CO branch. The tangent predictor (§9.5.4) is the local derivative of the branch, not an extrapolation, and does not have this failure mode.
- Omitting the φ_s-shift fallback when the tangent is unavailable — the φ_s boundary value moves by exactly dV at every step; discarding the one component of the step that is known in closed form makes the corrector work harder for no reason.
- Anchoring Stage 5 at the first output voltage instead of ramping from V_c = 0 — the trivial-solution uniqueness argument only holds at zero overpotential.

**Canonical sweep drivers.** The reference implementation is the unified driver of §9.5.5 — per-stage anchor, tangent-predictor continuation, post-solve assertion — with the **fitted kinetic parameters** from §17 loaded before the Stage-5 anchor (the fit runs after Stage 4, before Stage 5). Copy that structure. Do not write a per-voltage independent-solve loop.

### 9.8 Storage Strategy

See §10 for the full caching system. **The short version: cache every single converged solution to a per-stage binary directory, immediately after Newton converges, before doing anything else.**

---

## 10. Solution Caching — Per-Stage Binary Cache

### 10.1 Why Caching Is Non-Negotiable

**Rule: Never compute a solution and discard it.** Every converged Newton solution must be saved to disk before moving on to the next step. This is critical because:

1. **Warm-starting:** Later stages need converged solutions from earlier stages as initial guesses. Without the cache, you must re-solve the entire earlier stage.
2. **Continuation:** If the process crashes or is interrupted mid-sweep, the cache lets you resume from the last converged point.
3. **Plotting and analysis:** The human operator must review every stage before you proceed. Plots are generated from cached files, not from in-memory arrays.
4. **Stale-cache detection:** Each cache encodes what kinetics (i0, λ) were used. If kinetics change, you delete the cache and re-solve — but you never lose the code.

### 10.2 Cache Directory Layout

Each stage gets its own cache directory. These are created by the stage's driver script at startup:

```
co2r_bulk/
├── output_stage1/
│   └── cache/
│       ├── s1_V-0.300000.bin
│       ├── s1_V-0.310000.bin
│       ├── ...
│       └── s1_V-2.500000.bin
├── output_stage2/
│   └── cache/
│       ├── s2_V-0.300000.bin
│       ├── ...
│       └── s2_V-2.500000.bin
├── output_stage3/
│   └── cache/
│       ├── s3_V-0.300000.bin
│       ├── ...
│       └── s3_V-2.500000.bin
├── output_stage4/
│   └── cache/
│       ├── s4_V-0.300000.bin
│       ├── ...
│       └── s4_V-2.500000.bin
├── output_stage5/
│   └── cache/
│       ├── s5_V-0.300000.bin
│       ├── ...
│       └── s5_V-2.500000.bin
```

### 10.3 Binary Format — Simple and Fast

**Do NOT use JLD2 or any serialization library.** Use raw binary I/O — it is simpler, faster, and has no dependency issues. The format is:

```julia
using Printf

const STAGE1_CACHE = joinpath(@__DIR__, "output_stage1", "cache")

function save_stage1(u::Vector{Float64}, V::Float64)
    mkpath(STAGE1_CACHE)
    fn = joinpath(STAGE1_CACHE, @sprintf("s1_V%.6f.bin", V))
    open(fn, "w") do f
        write(f, Int64(length(u)))   # 8 bytes: DOF count (integrity check)
        write(f, u)                   # 8 × length(u) bytes: solution vector
    end
    @printf("  [CACHED] %s (%d DOFs)\n", fn, length(u))
    flush(stdout)
end

function load_stage1(V::Float64)
    fn = joinpath(STAGE1_CACHE, @sprintf("s1_V%.6f.bin", V))
    isfile(fn) || return nothing
    open(fn, "r") do f
        n = read(f, Int64)
        read!(f, Vector{Float64}(undef, n))
    end
end
```

**Binary format details:**
- First 8 bytes: `Int64` DOF count (used to verify the file is for the right stage)
- Remaining bytes: `Float64` array of the full solution vector `u`
- File name encodes the voltage to 6 decimal places: `s1_V-0.300000.bin`
- The `@sprintf("%.6f", V)` format ensures exact filename matching for reload

Repeat this pattern for each stage (`save_stage2`, `save_stage3`, `save_stage4`, `save_stage5`) with their own cache directories and filename prefixes (`s2_`, `s3_`, `s4_`, `s5_`).

### 10.4 Cache on Every Sweep Step

Inside the continuation sweep loop, save **immediately** after the solve passes the §9.5.6 assertion — before computing any diagnostics, before printing summary stats, before anything else:

```julia
# Inside the sweep loop (§9.5.5 driver, §9.5.3 kernel):
info = newton_affine!(u_cur, res!; ly=layout, plu=plu,
                      rtol=1e-8, max_iter=40, alpha_init=1.0,
                      n_log_dofs=5*N_CL)

if info.converged
    save_stage1(u_cur, V_cur)   # ← FIRST thing after convergence
    push!(solutions, copy(u_cur))
    push!(voltages, V_cur)
    @printf("step %d: V=%.4f, ||F||=%.2e, iter=%d\n",
            step, V_cur, info.residual, info.iterations)
else
    @printf("step %d: FAILED at V=%.4f — sub-stepping (§9.5.5)\n", step, V_cur)
    # halve dV, solve the two half-steps at alpha_init = 1e-2
end
```

### 10.5 Loading Caches for Warm-Starting Later Stages

Each later stage warm-starts from the previous stage's cache. The pattern:

```julia
# Stage 2 warm-starts from Stage 1
const STAGE1_CACHE = joinpath(@__DIR__, "output_stage1", "cache")

function load_warmstart_stage1(V_target::Float64)
    files = filter(f -> startswith(f, "s1_V") && endswith(f, ".bin"), readdir(STAGE1_CACHE))
    isempty(files) && error("No Stage 1 cache found in $STAGE1_CACHE — run Stage 1 first!")
    Vs = [parse(Float64, f[5:end-4]) for f in files]  # strip "s1_V" prefix and ".bin" suffix
    idx = argmin(abs.(Vs .- V_target))
    fn = joinpath(STAGE1_CACHE, files[idx])
    u = open(fn, "r") do f; n = read(f, Int64); read!(f, Vector{Float64}(undef, n)) end
    @printf("Warm-start: loaded Stage 1 at V=%.4f (%d DOFs)\n", Vs[idx], length(u))
    return u, Vs[idx]
end
```

**Stage warm-start chain:**
```
Stage 1: self-anchoring — equilibrium guess at −0.300 V (§9.5.5)
Stage 2: self-anchoring (same anchor recipe); Stage-1 cache optional warm-start
Stage 3: self-anchoring — equilibrium liquid + inlet gas; Stage-2 cache optional
Stage 4: self-anchoring; Stage-3 cache optional warm-start
Fit:    operate on Stage 4 cache — fit kinetics against experimental data
        using the simplified (φ_L=0, φ_s=V_applied) residual
Stage 5: fitted kinetics, FOLLOW the §9.5.5 production driver — anchor at
         V_c = 0 from the constructed physics guess, then tangent-predictor
         continuation in V_c with the affine-covariant kernel (§9.5.3).
         DO NOT cold-seed every V from the Stage-4 cache — independent
         cache-seeded solves can land on the wrong (low-CO) solution branch.

Because every stage is self-anchoring (§9.5.5), the previous-stage warm-start
is an optional convenience for the first voltage, not a requirement.
```

### 10.6 Listing and Inspecting Caches

Add a helper to list all cached voltages for a given stage:

```julia
function list_cache(cache_dir, prefix)
    files = filter(f -> startswith(f, prefix) && endswith(f, ".bin"), readdir(cache_dir))
    Vs = sort([parse(Float64, f[length(prefix)+1:end-4]) for f in files])
    @printf("Cache: %d solutions, V ∈ [%.3f, %.3f]\n", length(Vs), Vs[1], Vs[end])
    return Vs
end
```

### 10.7 Stale Cache Detection

At the top of every stage driver, print the current kinetic parameters so the human can verify the cache was built with the right kinetics:

```julia
@printf("=== Stage %d ===\n", stage)
@printf("i0_CO2R = %.0f A/m², λ_MHC = %.2f eV\n", i0_CO2R(), lambda_MHC())
@printf("HER_pf = %.3e, α_c = %.4f\n", HER_prefactor(), alpha_c_HER())
@printf("Cache dir: %s\n", CACHE_DIR)
if isdir(CACHE_DIR) && !isempty(readdir(CACHE_DIR))
    @printf("WARNING: Existing cache found. Verify kinetics match before reusing!\n")
end
```

If kinetics change (e.g., re-fitting i0), **delete the entire cache directory** and re-run from that stage forward. Never mix solutions from different kinetics.

---

## 11. Implementation Stages

> **CRITICAL WORKFLOW RULE:** Each stage is a **hard stop gate.** After completing a stage:
> 1. **Cache** all converged solutions to the stage's binary cache directory
> 2. **Generate all required plots** for that stage (see §12)
> 3. **Print a summary** (voltage range, number of solutions, max residual)
> 4. **STOP and wait for human review.** Do NOT proceed to the next stage until the human confirms the plots look correct.
>
> This is not optional. The human operator must visually inspect the results before you move on. If you skip this, bugs compound across stages and are much harder to debug.

**Stage progression overview:**

| Stage | Adds | DOFs (N_CL=80, N_GDL=120 production / dev N_CL=20, N_GDL=30) | φ_L | φ_s | Diffusion loss through electrolyte? |
|-------|------|--------------------------|-----|-----|-------------------------------------|
| 1 | Liquid NP **diffusion only** + Dirichlet BC + kinetics with V_applied | 5×N_CL = 400 / 100 | =0 | =V_applied | no (Dirichlet at c_eq) |
| 2 | Sherwood-Reynolds Neumann at electrolyte | 400 / 100 | =0 | =V_applied | yes (k_MT) |
| 3 | Stefan-Maxwell gas transport (CO₂/CO/H₂/H₂O coupled) | 5×N_CL + 4×(N_GDL+N_CL) = 1200 / 300 | =0 | =V_applied | yes |
| 4 | Darcy pressure (already a DOF in Stage 3 — finalize the pressure equation) | 1200 / 300 | =0 | =V_applied | yes |
| Fit | Fit i₀_CO2R, λ_MHC, HER_pf, α_c against experimental currents using Stage 4 residual | 1200 / 300 | =0 | =V_applied | yes |
| 5 | **Potentials**: φ_L (NP migration, N_CL DOFs) + φ_s (solid Ohm's law on GDL+CL, N_GDL+N_CL DOFs) — **uses fitted kinetics** | 1200 + N_CL + (N_GDL+N_CL) = 1480 / 370 | **solved (DOF)** | **solved (DOF)** | yes |

**Stages 1–4 use the simplified kinetics:** `φ_L_local = 0` and `φ_s_local = V_applied_SHE` everywhere. The kinetics functions in §4 still take both arguments, but the caller passes constants. This means the fitted parameters from Stage 4 are *apparent* values that absorb the missing potential drops, but the framework (transport, BCs, gas, Darcy) gets validated stage by stage in isolation.

**The fit comes after Stage 4**, not after Stage 5 — this is a deliberate choice for computational speed. The Stage 5 system is extremely stiff (cond(J) ~ 10²⁰), making Nelder-Mead fitting needlessly expensive: each objective evaluation would need the full Stage-5 continuation machinery to stay branch-safe. By fitting against the Stage 4 residual (where the simplified kinetics leave the system single-rooted and each polish converges in ~2 Newton iterations), the fit converges in minutes. The trade-off is that the fitted kinetic parameters are *apparent* values that absorb the ~50–150 mV potential drops missing from Stages 1–4. These apparent parameters are then used in Stage 5 where the full potential physics corrects the driving force. The resulting Stage 5 currents may differ from the fit targets by 10–30% — this is expected and physically correct. If needed, a second-pass refit against Stage 5 can be done, but the Stage 4 fit provides an excellent starting point.

**Stage 5 turns on full electrochemistry with the fitted kinetics:** the migration term in the NP fluxes, φ_L as a monolithic DOF solved from the per-cell ionic charge balance, and the solid-phase Ohm's law for φ_s. After Stage 5 converges, the kinetic driving force changes from `V_applied − U_eq` to `(φ_s_local − φ_L_local) − U_eq`, which is typically 50–150 mV less cathodic at high current. Currents will drop ~10–30% from Stage 4 → Stage 5.

---

### Stage 1: Liquid NP — Diffusion Only, Dirichlet BCs

**Goal:** Get buffer chemistry + CO₂R + HER + electroneutrality + log-concentration solver working in the simplest possible setting. **No migration, no φ_L, no φ_s solve.** Kinetics use `V_applied_SHE` directly as the metal potential at every CL cell — this is wrong by ~50 mV at high current, but it's enough to get the framework converging.

**Cache directory:** `output_stage1/cache/` — prefix `s1_`

**DOFs:** `5 × N_CL` (all liquid log-concentrations). At the production default mesh N_CL=80: **400 DOFs**. (At the dev-only N_CL=20 mesh: 100 DOFs.)

**Solver: §9.5 production stack with the unified driver** (§9.5.5) — affine-covariant kernel + persistent sparse LU + ForwardDiff AD Jacobian, self-anchoring at −0.300 V. Do NOT use the naive dense FD-Newton of §9.4 at production mesh.

**Residual structure:**
```julia
function residual_stage1!(F, u, V_cathode, cl_mesh)
    N = cl_mesh.N
    c = unpack_log_conc(u, N)                        # 5 × N physical
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset

    # ── Pure diffusion fluxes (NO migration in Stage 1)
    J = zeros(5, N+1)
    J[5, 1] = diffusion_flux(c_CO2_henry, c[5,1], D_eff[5], dx_face_left)
    J[1:4, 1] .= 0.0                                 # ions: no-flux at GDL side
    for f in 2:N
        for i in 1:5
            J[i,f] = diffusion_flux(c[i,f-1], c[i,f], D_eff[i], dx_between[f-1])
        end
    end
    for i in 1:5
        J[i,N+1] = diffusion_flux(c[i,N], c_eq[i], D_eff[i], dx_face_right)
    end

    # ── Cell residuals: integrated FV form (J_left − J_right + S × dx)
    # Use the integrated form throughout the guide — see §6.4 — to avoid 1/dx
    # scaling on graded meshes and to keep the Jacobian rows well conditioned in
    # the monolithic gas+liquid solve at later stages.
    for ix in 1:N
        c_ix = c[:,ix]
        # Stage 1 ONLY: phi_s = V_applied, phi_L = 0 (both wrong; will be fixed in Stage 5)
        r_co2r = R_CO2R(c_ix[5], V_applied_SHE, 0.0, c_ix[1],
                        a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        r_her  = R_HER(V_applied_SHE, 0.0, c_ix[1], a_v_CL, n_e)

        buf = buffer_sources(c_ix)
        for i in 1:5
            src = eps_I * buf[i]
            if i == 2; src += 2*r_co2r + 2*r_her; end
            if i == 5; src -= r_co2r; end
            F[liquid_idx(ix,i)] = J[i,ix] - J[i,ix+1] + src * dx[ix]
        end
    end
end
```

**Physicality checks after Stage 1:**
1. pH across CL should be ~9-12 (alkaline from OH⁻ production)
2. CO₂(aq) depletes from inlet value (~16.0 mol/m³ from Henry's law at 50°C, p_CO₂ = 0.88 atm = x_CO₂ × 1 atm) toward electrolyte
3. Total current = `n_e × F × ∫(R_CO2R + R_HER) dx` should be smooth and monotonic in V
4. CO selectivity = `∫R_CO2R / ∫(R_CO2R + R_HER)` at 80-95% near onset
5. CO₂R currents will be **underpredicted** vs experimental — that's expected, gas-phase transfer is not yet online so CO₂ supply to the catalyst is limited. The full gas-phase coupling (Stefan-Maxwell + phase transfer) is added in **Stage 3**, and the remaining potential-drop physics (φ_L migration and φ_s solid Ohm's law) in **Stage 5**

> **🛑 STOP GATE:** Generate Stage 1 plots, commit, and STOP for human review.

---

### Stage 2: Sherwood-Reynolds Neumann BC at Electrolyte Side

**Goal:** Replace the perfect-stirred-tank Dirichlet at `x = L_CL` with a convective Neumann that captures the finite mass transfer through the boundary layer in the electrolyte channel.

**Cache directory:** `output_stage2/cache/` — prefix `s2_`

**Warm-start:** Load Stage 1 cache at each voltage as initial guess.

**Change:** Replace the last face flux only:
```julia
J[i, N+1] = -k_MT[i] * (c_eq[i] - c[i,N])   # convective Neumann, sign matches NP convention
```

**No structural change to DOF count, residual layout, or solver.** Still `5 × N_CL` DOFs (400 at production mesh / 100 at dev mesh), still pure diffusion + Dirichlet on the GDL side, still kinetics with `V_applied_SHE` directly.

**Solver: §9.5 stack with the Stages 1–4 driver** (§9.5.5).

**New physicality checks:**
- Compute `Bi_i = k_MT,i × L_CL / D_eff,i` (Biot number). Should be O(0.1–10).
- `c_CO₂(aq)(L_CL)` at high current should approach 0 (depletion-limited; CO₂ is consumed by the CL and only supplied from the GDL side). OH⁻ is *produced* in the CL, so `c_OH⁻(L_CL)` should sit noticeably *above* the bulk equilibrium `c_eq_OH⁻ ≈ 9.3×10⁻³ mol/m³` — not near zero.
- Check the species flux at the electrolyte face equals `k_MT × (c_i(L_CL) − c_eq_i)` (same `(c_local − c_eq)` form as §7.2). The sign falls out naturally: for OH⁻ (produced in CL, `c_OH(L_CL) > c_eq_OH`) it's positive (leaving the CL); for CO₂ (depleted at the electrolyte side, `c_CO2(L_CL) < c_eq_CO2`) it's negative (reservoir resupplying). No species-specific sign rule — just whatever falls out of the concentration difference.

> **🛑 STOP GATE:** Generate Stage 2 plots, including overlay vs Stage 1, commit, and STOP for human review.

---

### Stage 3: Stefan-Maxwell Gas Transport

**Goal:** Replace the constant gas-side BC with a full multicomponent gas-phase model (CO₂/CO/H₂/H₂O) coupled by Stefan-Maxwell diffusion. The pressure DOF is already present here as the 4th gas DOF per cell, but Darcy flow is finalized in Stage 4.

**Cache directory:** `output_stage3/cache/` — prefix `s3_`

**Warm-start:** Liquid DOFs from Stage 2 cache. Gas DOFs from a gas-only bootstrap (see below).

**New DOFs:** `4 × (N_GDL + N_CL)` gas DOFs (ω_CO₂, ω_CO, ω_H₂, p_gas at every gas cell). Total at production default mesh (N_CL=80, N_GDL=120): `400 + 800 = 1200` DOFs. (At dev mesh N_CL=20, N_GDL=30: `100 + 200 = 300` DOFs.)

**Solver: §9.5 production stack with the unified driver** (§9.5.5). At 1200 DOFs an FD-Jacobian Newton is not viable — it stalls at the kinetics turn-on (V_RHE ≲ −0.6) for hours per voltage (§9.3.2). Use the ForwardDiff AD Jacobian + persistent sparse LU + `newton_affine!`.

**Stage 3 still uses simplified kinetics:** `phi_L_local = 0` and `phi_s_local = V_applied_SHE` everywhere (no migration, no φ_s solve). The point of Stage 3 is to validate the SM gas transport in isolation.

**Stage 3 sets `u_G = 0`** (no Darcy convection). The species flux is purely diffusive: `J_j = −ρ × D_j^MA × ∇ω_j − ρ × D_j^MA × ω_j × (∇M_n / M_n)`. The convective term `ρ × ω_j × u_G` from §6.1 is omitted here and activated in Stage 4 when Darcy flow is finalized.

**Key changes to residual:**
1. Add `sm_residual!` block for gas DOFs (see §6), but with `u_G = 0` (no convective flux).
2. CO₂(aq) left BC switches from constant `c_CO2_henry` to `Henry × ω_CO₂(x=0)`.
3. Phase transfer source `R_PT_CO2 = a_v_GL × k_GL × (Henry × p_CO2 − c_CO2_aq)` is added (signed) with **`+R_PT` on the liquid CO₂(aq) residual** and **`−R_PT × M_CO2` on the gas CO₂ mass continuity**. Sign convention: R_PT > 0 ⇒ gas → liquid net dissolution, so liquid gains and gas loses.
4. Gas-side faradaic sources: CO and H₂ produced by the kinetics inside the CL enter the gas residual.

**Bootstrap protocol for Stage 3:**
```
1. Solve gas-only (set all liquid sources to zero temporarily):
   sm_residual!(F_gas, u_gas) with R_PT=0, R_CO=0, R_H2=0
   → All cells at inlet composition: trivially converged.

2. Monolithic Newton on full gas+liquid system.
   Warm-start: liquid from Stage 2 converged solution, gas from step 1.
```

**Why run gas-only first?** The `sm_residual!` function has many conditionals and helper calls that get JIT-compiled on first call. The gas-only solve triggers all this compilation cheaply (just 4 × N_GDL DOFs, no liquid coupling). Then the full monolithic Newton is fast.

**Physicality checks after Stage 3:**
- ω_CO₂ decreases from ~0.947 at the GDL inlet (the mass fraction corresponding to `x_CO₂ = 0.88` with M_avg ≈ 40.89 g/mol) to noticeably lower values at the CL exit at high current.
- ω_CO and ω_H₂ increase through the CL.
- `p_CO₂ = x_CO₂ × p` consistent with Henry's law equilibrium at the gas-liquid interface.

> **🛑 STOP GATE:** Generate Stage 3 plots (gas composition profiles, polarization overlay vs Stages 1-2), commit, and STOP for human review.

---

### Stage 4: Darcy Pressure (Finalize the Pressure Equation)

**Goal:** Finalize the pressure-DOF equation as a true Darcy momentum balance. The pressure DOF was already a placeholder in Stage 3; here we wire up the proper boundary conditions and the harmonic-mean permeability at the GDL/CL interface.

**Cache directory:** `output_stage4/cache/` — prefix `s4_`

**Warm-start:** Direct from Stage 3 cache (no new DOFs).

**Stage 4 still uses simplified kinetics:** `phi_L_local = 0`, `phi_s_local = V_applied_SHE`. Potentials still come in Stage 5.

**Solver: §9.5 stack with the Stages 1–4 driver** (§9.5.5). 1200 DOFs at production mesh.

**Pressure-equation finalization:**
1. **GDL pressure Dirichlet** at `x = -L_GDL`: `p = p_total = 1 atm`
2. **CL pressure no-flux** at `x = L_CL`: `dp/dx = 0`
3. **Harmonic mean permeability** at the GDL-CL interface:
   ```julia
   κ_interface = 2 * κ_GDL * κ_CL / (κ_GDL + κ_CL)
   ```
4. Net mass source from the gas residual (CO₂ consumed, CO + H₂ produced) drives the pressure gradient via Darcy continuity.

**Species flux now includes Darcy convection:** With the pressure field finalized, the Darcy velocity `u_G = −(κ/μ_gas) × (dp/dx)` is nonzero and the full species flux from §6.1 is activated:
```
J_j = −ρ × D_j^MA × ∇ω_j − ρ × D_j^MA × ω_j × (∇M_n / M_n) + ρ × ω_j × u_G
```
The convective term `ρ × ω_j × u_G` was zero in Stage 3 (where `u_G = 0`). Stage 4 is the first stage where the species transport includes bulk gas convection driven by the pressure gradient.

**Physicality checks after Stage 4:**
- Total Darcy mass flux at GDL inlet ≈ net gas production: `Σ(S_j × dx_j)` (mass conservation).
- Pressure gradient satisfies `dp/dx ≈ μ × u_G / κ` everywhere.
- At i ≈ 2 A/cm², total pressure drop across CL is ~1–100 Pa (small but nonzero).
- Stage 4 currents should be **almost identical to Stage 3** (Darcy is a tiny correction at these currents).

> **🛑 STOP GATE:** Generate Stage 4 plots (pressure profile, Darcy velocity, gas-side mass-balance check), commit, and STOP for human review.

---

### Fit Step (after Stage 4)

Once Stage 4 is converged and the physicality checks pass, run the kinetic parameter fit (§17). The fit uses **Stage 4 solutions** with the simplified kinetics (`φ_L = 0`, `φ_s = V_applied_SHE`). This is a deliberate choice for computational speed — the Stage 4 residual avoids the φ_L/φ_s coupling and is well-conditioned, making the ~1000 Newton solves required by Nelder-Mead tractable in minutes rather than hours.

**Solver: §9.5 production stack**, restricted to the 5 experimental voltages per NM trial (per-voltage polish warm-started from the Stage-4 cache — §9.5.5 table). Same `newton_affine!` kernel + persistent sparse LU + AD Jacobian as Stages 1–4.

**Fitting parameters (in order of sensitivity):**
1. `i0_CO2R` — scales the CO₂R partial current
2. `λ_MHC` — sets the CO₂R onset potential and Tafel curvature
3. `HER_prefactor` — scales the HER partial current
4. `α_c_HER` — sets the HER Tafel slope

After the fit, the fitted parameters are *apparent* values that absorb the ~50–150 mV potential drops missing from Stages 1–4. These apparent parameters are then used in Stage 5 where the full potential physics corrects the driving force. The resulting Stage 5 currents may differ from the fit targets by 10–30% — this is expected and physically correct.

> **🛑 STOP GATE:** Print fitted parameters, generate Stage 4 polarization plots with fitted kinetics overlaid on experimental data, and STOP for human review before proceeding to Stage 5.

---


---

### Stage 5: Potentials — φ_L (NP migration) + φ_s (solid Ohm's law)

**Goal:** Turn on the full coupled electrochemistry **using the fitted kinetic parameters from the Fit Step**. Add the Nernst-Planck migration term in the liquid fluxes, promote `φ_L(x)` to a monolithic DOF solved from the per-cell ionic charge balance, and add the solid-phase Ohm's law on `[−L_GDL, L_CL]`. Switch the kinetics from `V_applied_SHE` to local `(φ_s_local − φ_L_local)`. This is the **largest physics change** in the staging — currents will drop noticeably from Stage 4 because the missing potential drops are now subtracted from the driving force.

**Cache directory:** `output_stage5/cache/` — prefix `s5_`

**Seeding:** no cache. Stage 5 is a continuation (§9.5.5): solve once at V_c = 0 from a guess constructed from physics (liquid = bulk buffer equilibrium, gas = inlet composition at `p_total`, `φ_s = 0` on GDL+CL, `φ_L = 0` on the CL), then reach every output voltage by predictor–corrector in −0.025 V steps. Do **not** seed individual voltages from `output_stage4/cache/` or `output_fit/cache/` — see §11 Stage 5 gotcha #0. **Load fitted kinetic parameters** (i0_CO2R, λ_MHC, HER_prefactor, α_c) from the Fit Step before running the Stage 5 sweep.

---

> ### 🔴 CRITICAL STAGE 5 IMPLEMENTATION NOTES (read before writing any code)
>
> Stage 5 has numerical pitfalls that are not present in Stages 1–4. Every single one of these was learned by a previous implementation stumbling and having to backtrack. If you are implementing Stage 5 from scratch, **do all of these from the start** or you will waste hours debugging convergence.
>
> **🛑 Read §9.5 (Production Solver Stack) FIRST.** It is the recipe that lands on the experimentally-correct (high-CO) solution branch. The Stage 5 residual is multi-rooted — naive per-voltage cold-seeded solvers converge to a low-CO root that *passes every numerical check* (small residual, charge conservation < 1e-9) but produces a peak i_CO ~20 % below experiment and Δφ_L ~3 mV instead of ~15 mV. The §9.5 stack (anchor-at-V_c=0 continuation + affine-covariant Newton + tangent predictor) is required for branch-correct reproducibility.
>
> **0. Stage 5 is seeded once, at V_c = 0 — not from any cache.** The Stage 5 sweep is a continuation (§9.5.5). The only initial guess you construct by hand is the anchor at V_c = 0 — bulk buffer equilibrium in the liquid, inlet composition and `p_total` in the gas, `φ_s = φ_L = 0` — and every subsequent voltage is reached by the predictor–corrector from the previous converged state. Do **not** seed individual voltages from `output_stage4/cache/` or from `output_fit/cache/`. Two distinct failures follow from doing so. (i) **Wrong branch:** the Stage 5 residual is multi-rooted, and an independent solve seeded from a Stage-4-layout state can land on the low-CO root, which passes ‖F‖∞ < 1e-3 and charge conservation ≤ 1e-9 but gives Δφ_L ≈ 3 mV instead of ~15 mV and a peak i_CO ~20 % below experiment. (ii) **Wrong kinetic regime:** the `output_stage4/cache/` entries were built by `run_stage4.jl` with the **pre-fit** kinetics (`i0_CO2R = 540 A/m²`, `λ_MHC = 1.8 eV`) while Stage 5 solves with the post-fit values (`i0_CO2R ≈ 100–200 A/m²`, `λ_MHC ≈ 1.6 eV`), a ~5× mismatch that puts the concentration / gas / pH fields ~50–150 mV off the correct regime at every cell. The diagnostic signature of a cache-seeded sweep is: (a) jagged / non-monotone polarization with sub-cell current discontinuities, (b) ionic charge-balance error at the CL / electrolyte boundary of 10–80 % on tolerance-converged points while the solid charge balance stays ≤ 10⁻⁶, (c) φ_L changing sign between neighbouring voltages. The 89-V refit sweep that `run_fit.jl` writes to `output_fit/cache/fit_V*.bin` (§17.8) is still worth producing — it is the Stage-4 physics at the fitted kinetics, and it isolates the effect of the fit from the effect of adding the potential fields — but it is a diagnostic artifact, not the Stage 5 warm start.
>
> **1. The entire residual chain must be type-generic (`AbstractVector{<:Real}`), not `Float64`.**
> Stage 5 requires a ForwardDiff AD Jacobian (see §9.3.2), which only works if Dual numbers can flow through every function the residual calls: `buffer_sources!`, `R_CO2R`, `R_HER`, `mhc_k_elec`, `diffusion_flux`, `np_flux`, `sm_face_flux_full`, `omega_to_x`, `Mn_from_omega`, `D_MA_species`, `omega_H2O_closure`, `unpack_log_conc`, `cl_omega`, `gdl_omega`, `cl_p`, `gdl_p`. All scalar arguments annotated `::Real`, all vectors `AbstractVector{<:Real}`. Constant matrices like `D_bin_raw` can stay `Matrix{Float64}` — scalar promotion handles it.
>
> **2. Never name a type parameter `T` in the residual.** `T` is the global temperature constant from §8.1 and will be shadowed with silent wrong-type errors. Use `ElT`.
>
> **3. Allocate work arrays with `zeros(ElT, ...)` (or `zeros(eltype(u), ...)`), not `zeros(...)`.** Local buffers like `Jliq = zeros(5, N_CL+1)` will pin the type to Float64 and ForwardDiff will error at assignment.
>
> **4. Use the §9.5 production kernel, not a textbook damped Newton.** Stage 5's `cond(J) ~ 10²⁰` comes from the 17-order scale spread between the φ_s divergence block (O(10⁷)) and the species log-conc block (O(10⁻¹⁰)) plus one *algebraic* null direction in the φ_L block. The null direction is removed exactly by the φ_L gauge override (#7); the conditioning is then handled by the continuation predictor keeping the iterate in-basin and by damping measured on Newton corrections (§9.4 item 3, §9.5.3). With those three pieces in place, a direct sparse-LU Newton solve is robust — cold-started far from a solution *without* them, it produces O(30) log-unit species updates and diverges, which is why the anchor + continuation structure is not optional.
>
> **5. Disable row scaling.** Per-row max-absolute scaling crushes the kinetic feedback `∂i_far_vol/∂φ_s ≈ 40` against the divergence Jacobian `4×10⁷` in the same row, making the kinetic coupling numerically invisible (10⁻⁶ relative after scaling). Newton then takes a step that ignores how kinetics respond to φ_s changes and diverges. The production kernel (§9.5.3) never rescales Jacobian rows — and never takes a residual norm at all.
>
> **6. φ_s Dirichlet at the current collector is a GHOST-CELL BC, not a cell-1 residual override.**
> The leftmost GDL cell keeps its normal continuity residual `F[φ_s_gdl[1]] = i_s_face[2] − i_s_face[1]`, and `V_applied_SHE` enters through `i_s_face[1] = −σ × (φ_s_gdl[1] − V_applied_SHE) / dx_half`. If you override the cell-1 residual with `F = φ_s_gdl[1] − V_applied`, you will see the current-consistency check fail with `|Δi|/i ≈ 2` (equal magnitude, opposite sign) — because `i_s_face[1]` becomes mechanically zero while the rest of the system still carries the full faradaic current. See §3.4.
>
> **7. φ_L Dirichlet at the electrolyte is a residual override, NOT a ghost cell.** (The exact opposite of φ_s.) The per-cell φ_L residual is algebraically `F_φL[ix] = −F × Σᵢ zᵢ × F_species[ix]` at every cell (a pure identity, no time derivatives involved — see §3.1), so the φ_L DOFs have exactly one null direction: a uniform shift of all φ_L values. The Dirichlet `φ_L[N_CL] = 0` override replaces cell N_CL's residual with that one equation, fixing the gauge. Without the override, the Jacobian is rank-deficient by 1 and Newton cannot converge. With any attempt to ALSO add the cell N_CL charge-balance row, the system is overdetermined.
>
> The rule: **override φ_L (gauge fix, no physical flux to absorb the BC), ghost φ_s (physical electronic current flux).**
>
> **8. Terminate on the error estimate (`rtol = 1e-8`), assert the residual after the solve.** The kernel's stopping test is the estimated remaining *solution* error (§9.5.3); the residual guarantee `‖F‖∞ ≤ 1e-6` is a separate post-solve assertion (§9.5.6). In practice the continuation sweep reaches `‖F‖∞ ≈ 1e-7` everywhere at ~2 iterations/step — do not fold a `‖F‖∞` floor into the termination test (measured +37–47 % iterations for no accuracy gain). The global charge-conservation check `|i_s_at_collector| = i_far_total` passing to ≤ 10⁻⁹ remains the physics-level correctness check.
>
> **9. Continue across voltages from the V_c = 0 anchor; do NOT rebuild an independent warm start at each voltage.** The Stage 5 driver (§9.5.5) solves the anchor once and then walks the V_c chain. At each step it (a) advances the previous converged state by the tangent predictor `u .+= dV_c .* du/dV` (§9.5.4; fall back to the exact φ_s shift — add `dV_c` to every φ_s DOF, GDL and CL blocks both — if the tangent is unavailable), (b) calls `newton_affine!` with `alpha_init = 1`, and (c) runs the §9.5.6 post-solve assertion before caching. If a step fails, halve `dV_c`, solve the first half-step at `alpha_init = 1e-2`, then take the second half-step from there. **Do NOT solve each voltage independently from a cache entry** (`output_stage4/cache/` or `output_fit/cache/`): the Stage 5 residual is multi-rooted and cache-seeded independent solves can land on the low-CO branch, which passes every numerical check — see gotcha #0 for the full diagnostic signature. See §9.5.5 for the canonical driver pattern.
>
> **9a. `V_cathode` / `V_applied` unit convention, and the RHE→SHE conversion in the anchor builder.** Throughout Stage 5, the argument named `V_cathode` (passed into `residual_stage5!` and the anchor / initial-guess builder) is in **V vs RHE@pH 8** — the experimental convention. The anchor builder MUST convert internally via `V_applied_SHE = V_cathode - V_RHE_to_SHE_offset` (= `V_cathode - 0.51305` at 50°C, pH 8 — the anode reference pH of §15.2) and write the SHE value into the φ_s DOFs (both the GDL and CL φ_s blocks), with φ_L initialized to 0. `residual_stage5!` likewise converts `V_cathode → V_applied_SHE` and applies that SHE value at the Dirichlet ghost-cell. If the anchor builder (or the continuation predictor) writes `V_cathode` (RHE) directly into the φ_s DOFs instead of `V_applied_SHE`, the initial φ_s is off by the reference offset (513 mV at pH 8) and the solve either diverges or converges to a shifted (wrong) branch → jagged / shifted polarization curve. See §15.2 for the offset derivation.
>
> **10. Expected Stage 5 behaviour.** With the tangent predictor in-basin, iteration counts are flat across the whole sweep: ~2 per continuation step at `rtol = 1e-8` (anchor ~5), including the plateau regions and the deeply cathodic tail. A voltage that suddenly needs many more iterations — or trips the sub-step recovery — is a bug signal (predictor not applied, residual regression, branch bend under-resolved), not something to wait out.
>
> **11. A correct Stage 5 should converge all chain points with `|Δi|/i ≤ 10⁻⁶` everywhere, pass the §9.5.6 assertion at all 89 output voltages, and produce smooth monotone currents.** If the anchor at V_c = 0 doesn't reach `‖F‖∞ ≤ 1e-9` in ≤ ~6 iterations, something on this list is still wrong.
>
> **12. The right-boundary flux MUST include NP migration in Stage 5.** The Sherwood-only formula `J[i] = -k_MT[i] × (c_eq[i] - c[i,N])` from Stages 1–4 omits the migration term `-(zᵢ F/RT) D c̄ ∇φ_L` at the CL/electrolyte interface. In Stage 5, CO₃²⁻ carries the bulk of the ionic current through this boundary via migration. Without the migration term, the ionic current at the right boundary does NOT equal the integrated faradaic current (charge conservation violated by 2–20%) and the φ_L profile develops unphysical sign reversals. Use `np_flux(c[i,N], c_eq[i], φ_L[N], 0, D_eff[i], z[i], L_eff)` with `L_eff = D_eff[i] / k_MT[i]`. This reduces to the Sherwood formula when φ_L = 0 (Stages 1–4) and correctly carries migration when φ_L ≠ 0 (Stage 5).
>
> **13. Verify ionic current conservation at the right boundary.** After convergence, compute `i_ionic_R = F × Σᵢ zᵢ × J_i_right` from the boundary fluxes and compare to `i_far_total = n_e F ∫(R_CO2R + R_HER) dx`. These MUST agree to <1% (`rel_err = |abs(i_far) - abs(i_ionic_R)| / abs(i_far) < 0.01`). If they don't, either the right-boundary flux is missing migration (#12 above) or the solutions aren't converged tightly enough.
>
> **14. Stage 5 uses fitted kinetics from the Fit Step.** The kinetic parameters (i0_CO2R, λ_MHC, HER_prefactor, α_c) must be loaded from the Fit Step output before running the Stage 5 sweep. If you accidentally run Stage 5 with the default (pre-fit) kinetics, the results will not match the experimental data. Always verify by printing the kinetic parameters at the top of the Stage 5 driver and comparing to the fitted values.

---

**New DOFs:** add `N_GDL + N_CL` solid potential DOFs (φ_s on the full GDL+CL domain) **and** `N_CL` liquid potential DOFs (φ_L on the CL only — no liquid in GDL). φ_L is a real DOF, not a derived quantity, so the Newton solver couples everything monolithically with no Picard iteration. Total Stage-5 DOFs:
```
5×N_CL  (liquid concentrations)
+ 4×(N_GDL + N_CL)  (gas DOFs from Stages 3-4)
+ N_GDL + N_CL  (solid potential)
+ N_CL  (liquid potential)
= 5(80) + 4(200) + 200 + 80 = 400 + 800 + 200 + 80 = 1480 DOFs (production N_CL=80, N_GDL=120)
[At dev mesh N_CL=20, N_GDL=30: 100 + 200 + 50 + 20 = 370 DOFs]
```

```
DOF layout: u = [ ln(c)_1..5N_CL ;  gas_DOFs_1..4(N_GDL+N_CL) ;
                  φ_s_GDL_1..N_GDL ;  φ_s_CL_1..N_CL ;  φ_L_CL_1..N_CL ]
```

**Stage5Layout struct and index functions (required):**

```julia
struct Stage5Layout
    N_CL::Int; N_GDL::Int; N_TOT::Int
end
function Stage5Layout(N_CL, N_GDL)
    N_TOT = 5*N_CL + 4*(N_GDL + N_CL) + N_GDL + N_CL + N_CL
    Stage5Layout(N_CL, N_GDL, N_TOT)
end

# Liquid species: u[5*(ix-1)+i] for species i at CL cell ix
liquid_idx(ix, i) = 5*(ix - 1) + i

# Gas DOFs: first N_GDL cells (GDL), then N_CL cells (CL), each with 4 DOFs
const _gas_start(ly) = 5 * ly.N_CL
gas_gdl_idx5(ly, ig, j) = _gas_start(ly) + 4*(ig - 1) + j
gas_cl_idx5(ly, ix, j)  = _gas_start(ly) + 4*ly.N_GDL + 4*(ix - 1) + j

# φ_s GDL: after all gas DOFs
const _phi_s_gdl_start(ly) = _gas_start(ly) + 4*(ly.N_GDL + ly.N_CL)
phi_s_gdl_idx5(ly, ig) = _phi_s_gdl_start(ly) + ig

# φ_s CL: after φ_s GDL
phi_s_cl_idx5(ly, ix)  = _phi_s_gdl_start(ly) + ly.N_GDL + ix

# φ_L CL: after φ_s CL
phi_L_idx5(ly, ix)     = _phi_s_gdl_start(ly) + ly.N_GDL + ly.N_CL + ix
```

**Unpacking helpers** (must return the element type from `u` for ForwardDiff compatibility):
```julia
function unpack_log_conc5(u, ly)
    ElT = eltype(u)
    c = zeros(ElT, 5, ly.N_CL)
    @inbounds for ix in 1:ly.N_CL, i in 1:5
        c[i, ix] = exp(clamp(u[liquid_idx(ix, i)], -50.0, 50.0))
    end
    return c
end
```
Similarly for gas DOFs: `cl_omega5(u, ly, ix)` returns `(ω₁, ω₂, ω₃, 1-ω₁-ω₂-ω₃)` and `cl_p5(u, ly, ix)` returns the pressure, both from the gas block at CL cell `ix`. Implement `gdl_omega5`/`gdl_p5` analogously for GDL cells.

**Return value of the Newton kernel (`newton_affine!`):** a NamedTuple
`(converged, iterations, residual, err_est, theta, alpha_final, n_res, n_bsub, reason)`
— see §9.5.3. `residual` is ‖F‖∞ at exit, reported for the §9.5.6 assertion;
`err_est` is the terminating error estimate; `theta` the final contraction.

**`dx_half` for the ghost-cell φ_s BC:** This is the distance from the current collector (boundary face at x = −L_GDL) to the center of the first GDL cell: `gdl_mesh.dx_between[1]` (half of the first cell's width for a uniform mesh, or the face-to-center distance for a graded mesh).

**`n_log_dofs`:** Set to `5 * N_CL` (the liquid log-concentration block only). Gas, φ_s, and φ_L DOFs are NOT clamped — they have different physical scales and the log-clamp would be wrong for them.

**Residual structure (full Stage 5):**

> The pseudocode below shows the **physics structure**; when you actually write the Julia code, every `zeros(...)` below must become `zeros(ElT, ...)` and every `unpack_...` must return an eltype-matched container so ForwardDiff.Dual numbers flow through. See gotchas #1–#3 above for the type-generalization checklist.

```julia
function residual_stage5!(F::AbstractVector{ElT},
                          u::AbstractVector{ElT},
                          V_cathode::Real,
                          cl_mesh, gdl_mesh,
                          c_eq_vec, ly::Stage5Layout) where {ElT<:Real}
    N_CL = cl_mesh.N; N_GDL = gdl_mesh.N
    c          = unpack_log_conc5(u, ly)            # Matrix{ElT} (5×N_CL)
    gas        = u                                  # gas DOFs accessed via gdl_omega5/gdl_p5/cl_omega5/cl_p5(u, ly, ...)
    phi_s_gdl  = unpack_phi_s_gdl5(u, ly)           # Vector{ElT}
    phi_s_cl   = unpack_phi_s_cl5 (u, ly)           # Vector{ElT}
    phi_L_cell = unpack_phi_L_cl5 (u, ly)           # Vector{ElT}; ← real DOF, not closure
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset

    # ── Step 1: liquid NP fluxes (diffusion + migration), integrated FV form
    J = zeros(ElT, 5, N_CL+1)
    J[5, 1] = diffusion_flux(c_CO2_from_gas(gas), c[5,1], D_eff[5], dx_face_left)
    J[1:4, 1] .= 0.0
    for f in 2:N_CL
        for i in 1:5
            J[i,f] = np_flux(c[i,f-1], c[i,f], phi_L_cell[f-1], phi_L_cell[f],
                              D_eff[i], z[i], dx_between[f-1])
        end
    end
    # Right face: full Nernst-Planck flux to the equilibrium reservoir
    # (c_eq at φ_L = 0, the gauge point). The per-species effective length
    # L_eff = D_eff / k_MT matches the old Sherwood resistance when ∇φ_L = 0,
    # but correctly carries the CO₃²⁻ migration current at the right boundary.
    # WITHOUT the migration term here, the ionic current at the CL/electrolyte
    # interface does not match the integrated faradaic current — charge
    # conservation is violated and the φ_L profile becomes unphysical.
    for i in 1:5
        L_eff = D_eff[i] / k_MT[i]
        J[i,N_CL+1] = np_flux(c[i,N_CL], c_eq[i],
                               phi_L_cell[N_CL], zero(ElT),
                               D_eff[i], z[i], L_eff)
    end

    # ── Step 2: faradaic rates + phase transfer per CL cell — kinetics use LOCAL φ_s, φ_L
    R_CO2R_cell = zeros(ElT, N_CL); R_HER_cell = zeros(ElT, N_CL); R_PT_cell = zeros(ElT, N_CL)
    for ix in 1:N_CL
        c_ix        = c[:,ix]
        phi_L_local = phi_L_cell[ix]                        # ← from φ_L DOF
        phi_s_local = phi_s_cl[ix]                          # ← from φ_s DOF, NOT V_applied
        R_CO2R_cell[ix] = R_CO2R(c_ix[5], phi_s_local, phi_L_local, c_ix[1],
                                  a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        R_HER_cell[ix]  = R_HER(phi_s_local, phi_L_local, c_ix[1], a_v_CL, n_e)
        # Phase transfer (Stage 3+): R_PT > 0 ⇒ gas → liquid net dissolution
        p_CO2_local  = pCO2_from_gas(gas, ix)               # local gas-phase CO₂ partial pressure
        H_local      = sechenov_H_CO2(c_ix)                 # local Henry constant w/ Sechenov
        R_PT_cell[ix] = a_v_GL * k_GL * (H_local * p_CO2_local - c_ix[5])
    end

    # ── Step 3: liquid species residuals (integrated FV form: J_left − J_right + S × dx)
    for ix in 1:N_CL
        buf = buffer_sources(c[:,ix])
        for i in 1:5
            src = eps_I * buf[i]
            if i == 2; src += 2*R_CO2R_cell[ix] + 2*R_HER_cell[ix]; end       # OH⁻ source
            if i == 5; src -= R_CO2R_cell[ix]; src += R_PT_cell[ix]; end       # CO₂(aq): consumed by CO₂R, replenished by R_PT (+R_PT, signed)
            F[liquid_idx(ix,i)] = J[i,ix] - J[i,ix+1] + src * dx[ix]
        end
    end

    # ── Step 4: gas-phase residual (Stages 3-4 logic, with R_PT_cell as the GL coupling source)
    sm_residual5!(F, u, ly, R_PT_cell, R_CO2R_cell, R_HER_cell, ...)   # see §6

    # ── Step 5: φ_L residual — local ionic charge balance per CL cell (§3.1)
    # i_L_face = F × Σᵢ zᵢ × J_i_face (computed from the same J array used above)
    i_L_face = zeros(ElT, N_CL+1)
    for f in 1:N_CL+1
        i_L_face[f] = F_const * sum(z[i] * J[i,f] for i in 1:5)
    end
    # No ionic current crosses x=0 (GDL side has no liquid):
    @assert i_L_face[1] ≈ 0.0   # implicit since J[1:4,1] = 0 and J[5,1] is for neutral CO₂
    for ix in 1:N_CL
        i_far_vol = n_e * F_const * (R_CO2R_cell[ix] + R_HER_cell[ix])
        F[phi_L_idx5(ly, ix)] = (i_L_face[ix+1] - i_L_face[ix]) + i_far_vol * dx[ix]
    end
    # Anchor φ_L = 0 at the electrolyte-side cell (Dirichlet gauge)
    F[phi_L_idx5(ly, N_CL)] = phi_L_cell[N_CL]   # overrides the charge-balance row for the last cell

    # ── Step 6: φ_s residual on GDL + CL (§3.4) — integrated FV form, MINUS sign on i_far_vol
    i_s = solid_face_currents(phi_s_gdl, phi_s_cl, V_applied_SHE,
                               σ_s_eff_GDL, σ_s_eff_CL, gdl_mesh, cl_mesh)
    for ig in 1:N_GDL                          # GDL: no faradaic source
        F[phi_s_gdl_idx5(ly, ig)] = i_s[ig+1] - i_s[ig]
    end
    for ix in 1:N_CL                           # CL: source = volumetric faradaic current (cathodic, ∇·i_s = +i_far_vol)
        i_far_vol = n_e * F_const * (R_CO2R_cell[ix] + R_HER_cell[ix])
        F[phi_s_cl_idx5(ly, ix)] = (i_s[N_GDL+1+ix] - i_s[N_GDL+ix]) - i_far_vol * dx[ix]
    end
    # NOTE: Do NOT override cell 1's φ_s residual. The Dirichlet V_applied_SHE
    # enters through the ghost cell on the leftmost GDL face flux:
    #   i_s[1] = -σ_s_eff_GDL × (phi_s_gdl[1] - V_applied_SHE) / dx_half
    # which is used in cell 1's standard continuity residual
    #   F[phi_s_gdl_idx5(ly, 1)] = i_s[2] - i_s[1]
    # already computed in the loop above. Overriding cell 1 breaks the
    # global charge-conservation check — see §3.4 and the Stage 5 gotchas.
end
```

**No Picard iteration.** Both φ_L and φ_s are real DOFs in `u`, so the Newton solver picks up the c↔φ_L↔φ_s coupling through the Jacobian. The φ_L residual (Step 5) is the algebraic charge-balance constraint at every CL cell; the species residuals already include the migration term via the same `J_i_face` array, so the Jacobian columns for each `phi_L_cell[ix]` populate both the species rows AND the φ_L row at that cell. Newton converges in one solve per voltage step, no nested iteration.

**Bootstrap protocol for Stage 5** (simplified — see §9.5.5 for the production driver, §9.7 for the bootstrap protocol):
1. **Ensure fitted kinetic parameters are loaded from the Fit Step** before anything else — the anchor solve and every continuation step use them.
2. Build the V_c = 0 anchor guess from physics, not from a cache: liquid = bulk buffer equilibrium (`solve_buffer_equilibrium`), gas = inlet mass fractions at `p_total`, `φ_s = 0` on GDL+CL, `φ_L = 0` on the CL. Solve with `newton_affine!` at `alpha_init = 1e-2`. At V_c = 0 the overpotential is zero, the kinetics are negligible, the solution is the trivial one, and it converges in ~3–5 iterations to ‖F‖∞ ≈ 1e-10.
3. Walk V_c from 0 down through the output grid in −0.025 V steps. At each step: advance the previous converged state with the tangent predictor (§9.5.4; φ_s-shift fallback), call `newton_affine!(…; alpha_init = 1, max_iter = 40)`, and run the §9.5.6 assertion. Expect ~2 iterations per step. On a failed step, halve `dV_c` and solve the two half-steps at `alpha_init = 1e-2`. **Do not seed any voltage from `output_stage4/cache/` or `output_fit/cache/`** — independent cache-seeded solves can land on the low-CO branch, which passes every numerical check (§11 Stage 5 gotcha #0).

**Physicality checks after Stage 5:**
1. **Solid ohmic drop:** `φ_s_CL[N_CL] − V_applied` (the potential rise from the current collector at x = −L_GDL to the electrolyte-side catalyst at x = L_CL; positive because φ_s rises toward a less-negative value in the direction of conventional solid current) should be 5–80 mV depending on current. At i = 1 A/cm², expect ~50 mV.
2. **GDL vs CL split of the solid drop:** GDL accounts for ~80–90% of the total solid drop (because L_GDL is 30× thicker than L_CL, while σ_s_eff_GDL is only ~7× larger).
3. **φ_L profile:** monotonically varying across the CL (sign depends on which species dominate the migration). Magnitude 5–50 mV.
4. **Current consistency:** the integrated faradaic current `n_e F ∫(R_CO2R + R_HER) dx` should equal the magnitude of the solid current at the current-collector face: `|i_s_face[1]| = |−σ_s_eff_GDL × (φ_s_gdl[1] − V_applied_SHE) / dx_half|`. With the ghost-cell Dirichlet BC (§3.4), this should agree to `|Δi|/i ≤ 10⁻⁶` — typically ~10⁻⁹ to 10⁻¹² once Newton converges. **Compare magnitudes**, not signed values: `i_far_total > 0` (cathodic reaction rate) while `i_s_face[1] < 0` (conventional current flows in −x direction = electrons in +x direction into the cathode). They have opposite signs and equal magnitudes. This is the cleanest correctness check for the whole stage and will immediately flag any mistake in the faradaic source sign, the φ_s BC implementation, or the charge-weighted species source bookkeeping.
5. **CO/H₂ currents drop ~10–30% vs Stage 4** because the kinetic driving force is now `(φ_s − φ_L) − U_eq` instead of `V_applied − U_eq`. This is the physically correct value; Stages 1–4 were overshooting.
6. **Ionic current conservation at the right boundary:** Compute `i_ionic_R = F × Σᵢ zᵢ × J_i_right` from the NP boundary fluxes (including migration!) and compare to `i_far_total`. They must agree to `rel_err = |abs(i_far) − abs(i_ionic_R)| / abs(i_far) < 0.01` (1%). If they don't, the right-boundary flux is missing the migration term (see gotcha #12). With the corrected NP flux at the boundary, expect `rel_err < 0.001` at tight convergence (||F||≤1e-5).
7. **Partial currents and FE are smooth and monotone.** i_CO should rise smoothly, peak, then decrease. i_H₂ should rise monotonically. FE_CO should peak at ~70-80% then decline. Any zigzags, step discontinuities, or non-monotone behavior indicates a branch switch or convergence issue.

> **🛑 STOP GATE:** Generate Stage 5 plots — polarization (overlay all 5 stages), φ_s profile (GDL+CL), φ_L profile (CL), current-balance check, then commit and STOP for human review. **This is the final physics gate** — Stage 5 is the complete model with fitted kinetics and full potential physics.

---

## 12. Physicality Checks and Debug Plots

### 12.1 Equilibrium Sanity Check (Before Any Voltage)

At V = 0 (or mild V = −0.3 V), with `α_elec = 0`:
- All species should be at bulk equilibrium: `c ≈ c_eq` everywhere
- Buffer residuals should be ≈ 0: `|S_buf,i| < 1e-6 mol/m³/s`
- Check electroneutrality: `|Σ z_i c_i| < 1e-8 mol/m³` in each cell

```python
# Python check
c_eq = solve_buffer_equilibrium(1000, 1000)
print(f"pH = {c_eq.pH:.2f}")  # should be ~8.23 (bicarbonate equivalence point = (pKa1+pKa2)/2 at 50°C, from K_w, K_2, K_3 — NOT 8, that's the anode reference pH used only for RHE→SHE conversion)
print(f"c_CO2 = {c_eq.CO2:.3f} mol/m³")  # ~11 mol/m³ at the bicarbonate equivalence point
                                          # (this is the CL-bulk CO₂ from carbonate equilibrium with 1 M total carbonate;
                                          #  NOT the Henry's-law boundary value ~12.2 mol/m³ which only applies at the GDL face)
print(f"charge balance = {c_eq.H + 1000 - c_eq.OH - c_eq.HCO3 - 2*c_eq.CO3:.2e}")
```

### 12.2 Current-Voltage Polarization Curve

Primary diagnostic. Plot vs V (V vs SHE):
- Total current density `i_total` [A/m²] (should match Lees et al. data)
- CO partial current `i_CO = 2F × ∫R_CO2R dx`
- H₂ partial current `i_H2 = 2F × ∫R_HER dx`
- CO faradaic efficiency (FE_CO) = `i_CO / i_total`

Expected qualitative behavior (anchored to the Lees et al. 2024 data, §16.1):
- First measurable current (~35 mA/cm² CO + ~30 mA/cm² H₂) at ~−0.82 V vs RHE@pH8 ≈ −1.33 V vs SHE; physical CO onset a few tens of mV less cathodic than this.
- CO FE peaks ~80% near E_cath ≈ −1.2 V vs RHE@pH8 (≈ −1.72 V vs SHE).
- H₂ FE overtakes CO at high overpotential (E_cath < ~−2.0 V vs RHE@pH8 ≈ −2.5 V vs SHE).
- Total current is a few hundred mA/cm² at E_cath ≈ −1.6 V vs RHE@pH8 ≈ −2.1 V vs SHE.

Note: the conversion is `V_SHE = V_RHE@pH8 − 0.51305 V` (§15.2). The operating window in the Lees data spans E_cath ≈ −0.82 → −2.41 V vs RHE@pH8, i.e. −1.33 → −2.93 V vs SHE.

```python
# Compare to Lees et al. 2024, Nat. Chem. Eng. Fig. 2 (see §16.1)
# Experimental: first measurable CO2R current at ~-0.82 V vs RHE@pH8
# (~-1.33 V vs SHE), peak FE_CO ~79% near -1.2 V vs RHE@pH8 (-1.72 V vs SHE)
```

### 12.3 Concentration Profiles

For selected voltages spanning the full reporting window V ≈ −1.0 to −3.0 V vs SHE (e.g., V = −1.0, −1.5, −2.0, −2.5, −3.0 V vs SHE):

**Liquid profiles:**
- `c_CO2(aq)(x)`: should deplete from GDL side toward the electrolyte-channel boundary. At high current, may → 0 (mass transfer limited)
- `c_OH⁻(x)`: should increase from bulk value (OH⁻ produced by electrochemistry)
- `c_HCO3⁻(x)`, `c_CO3²⁻(x)`: carbonate species shift with pH
- `pH(x) = −log₁₀(c_H / 1000)`: should increase toward the electrolyte-channel boundary (more alkaline)

**Gas profiles (Stage 3+):**
- `x_CO₂(x)` across GDL + CL: decreases from 0.88 at inlet
- `x_CO(x)`, `x_H₂(x)`: increase through CL
- `p(x)`: slight pressure drop (O(Pa))

### 12.4 Residual Convergence Check

Monitor Newton convergence at every voltage:
```julia
println("V = $(round(V,digits=3)), ||F||_inf = $(maximum(abs.(F))), iter = $it")
```

Warning signs:
- `||F||_inf > 1e-3` after 100 iterations → solver stuck, reduce `ds` or restart
- `||F||_inf` oscillating without decreasing → check Jacobian scaling
- Negative concentrations → increase step clamp limit

### 12.6 Mass Balance Check

At every voltage, verify mass conservation:
```
Σ fluxes at boundaries + Σ sources in CL = 0
```

For CO₂(aq):
```
J_CO2_in(x=0) - J_CO2_out(x=L_CL) + ∫R_PT dx - ∫R_CO2R dx = 0
```

Tolerance: `< 1e-6 mol/(m²·s)` (should be numerically exact for FV).

### 12.7 Debug Plot Checklist

**EVERY stage must produce the following plots, no exceptions.** Add stage-specific plots on top of these for the new physics introduced.

#### A. Always (Stages 1, 2, 3, 4, 5, and Fit)

**A1. 3-panel polarization summary** (one figure, three subpanels):
- (a) **Linear polarization** vs −E_cath [V vs SHE]: partial CO₂R current, partial HER current, total current — model curves AND experimental scatter points (Lees et al. 2024 Fig. 2; see §16.1 data table). y-axis: mA/cm².
- (b) **Log polarization** — same data as (a) on log y-axis. y-axis: mA/cm² (log).
- (c) **Faradaic efficiency** vs −E_cath: FE_CO and FE_H₂ model curves AND experimental points. y-axis: % (0–100).

**A2. Liquid-phase 3×3 diagnostic figure** (one figure, nine subpanels arranged in a 3×3 grid). Each subpanel plots the quantity across the CL at **all sweep voltages** (e.g., −1.0, −1.5, −2.0, −2.5, −3.0 V vs SHE), with each voltage as a separate curve in a light-to-dark color ramp. x-axis for all: `x [μm]` (physical position, 0 = GDL side, L_CL = electrolyte side).

Layout:
```
(a) pH [−]              (b) CO₂(aq) [mol/m³]     (c) OH⁻ [mol/m³]
(d) HCO₃⁻ [mol/m³]     (e) CO₃²⁻ [mol/m³]      (f) Cs⁺ [mol/m³]
(g) H⁺ [mol/m³]         (h) R_CO2R [mol/m³/s]    (i) R_HER [mol/m³/s]
```

Notes:
- (a) pH = −log₁₀(c_H⁺/1000), typically 9–12 during operation
- (f) Cs⁺ recovered from electroneutrality
- (g) H⁺ y-axis uses scientific notation (values ~10⁻⁶ mol/m³)
- (h)–(i) are the volumetric reaction rate profiles that were formerly A3
- All subpanels share a common legend (voltage labels in each panel or a single shared legend)

#### B. Stage 3+ (gas DOFs added)

In addition to A1–A2, add:

**B1. Gas-phase 2×2 diagnostic figure** (one figure, four subpanels arranged in a 2×2 grid). Each subpanel plots the quantity across the GDL+CL at **all sweep voltages** (same voltage set and color ramp as A2). x-axis for all: `x [μm]` (physical position, x = 0 at the GDL inlet, GDL/CL interface marked with a vertical dashed line).

Layout:
```
(a) p_CO₂ [atm]         (b) p_CO [atm]
(c) p_H₂ [atm]          (d) p_total [Pa or atm]
```

Notes:
- (a)–(c) are partial pressures computed from mass fractions: `p_j = x_j × p_total` where x_j is the mole fraction
- (d) is the total Darcy pressure profile — in Stage 3 this is approximately uniform (placeholder); in Stage 4+ it shows the small pressure gradient driven by mass sources
- All subpanels share a common legend (voltage labels)

#### D. Stage 5 (potentials: φ_L and φ_s)

In addition to A1–A2 and B1, add:

**D1. φ_s profile** across the GDL+CL.
- x-axis: x [m] from x = -L_GDL (where φ_s = V_applied) to x = L_CL (Neumann zero-flux end)
- y-axis: solid potential [V vs SHE]
- Plot at multiple voltages (low, medium, high current) to show the ohmic drop scaling

**D2. φ_L profile** across the CL only (no liquid in GDL).
- x-axis: x/L_CL
- y-axis: liquid potential [V vs SHE], anchored to φ_L = 0 at x = L_CL
- Plot at multiple voltages

#### E. Fit step (after Stage 4)

Generate **A1–A2 plots** (and B1 since gas DOFs exist) using the fitted kinetic parameters with the Stage 4 residual (simplified kinetics), side-by-side with the default-kinetics Stage 4 plots, so the human can see how the fit affected each diagnostic. D1/D2 plots are not applicable at this stage (no φ_s/φ_L DOFs yet).

---

#### Required-plot summary table

| Plot category | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Fit | Stage 5 |
|---|---|---|---|---|---|---|
| **A1** 3-panel polarization summary (i, log-i, FE vs exp) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **A2** 3×3 liquid-phase (pH, 6 species, R_CO2R, R_HER vs x at all voltages) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **B1** 2×2 gas-phase (p_CO₂, p_CO, p_H₂, p_total vs x at all voltages) |  |  | ✓ | ✓ | ✓ | ✓ |
| **D1** φ_s profile |  |  |  |  |  | ✓ |
| **D2** φ_L profile |  |  |  |  |  | ✓ |

**All x-axes that show voltage use V vs SHE (positive = more cathodic on the right):**
```python
V_SHE = V_RHE_at_pH8 - 0.51305   # at T = 50°C = 323.15 K, pH = 8
# Then plot: ax.plot(-V_SHE, ...)  so that more cathodic is to the right
```

**Why these plots specifically:**
- **A1** is the headline result — it's what you compare to the experimental data and what fits depend on. Same panel layout in every stage so you can flip between cached plots and immediately see what changed.
- **A2** (3×3 liquid-phase) is the *physics diagnostic*: it tells you whether the buffer chemistry, transport, kinetics, and pH are doing the right thing inside the CL. Plotting at all sweep voltages shows how profiles evolve from mild to mass-transport-limited regimes. If A1 is wrong, A2 tells you why.
- **B1** (2×2 gas-phase) is needed once gas DOFs exist — verifies the gas transport is depleting CO₂ and producing CO/H₂ as expected, and that total pressure is physical. Plotting at all voltages shows how gas depletion scales with current.
- **D1/D2** are the most important Stage 5 diagnostics — they verify that the ohmic drops are physical (~50 mV) and that the full potential physics is working correctly with the fitted kinetics.

---

## 13. Module Structure (Recommended)

```
co2r_bulk/
├── params.jl          # All physical constants, geometry, kinetics (Ref-backed mutables)
├── mesh.jl            # CLMesh, GDLMesh structs with x_faces, x_centers, dx, dx_between
├── chemistry.jl       # buffer_sources! function
├── kinetics.jl        # R_CO2R (MHC), R_HER (BV)
├── transport.jl       # diffusion_flux (Stages 1–4), np_flux (Stage 5 NP with migration)
├── gas_transport.jl   # SM + Darcy: compute_sm_flux!, sm_residual!, DOF indexing
├── assembly.jl        # full_residual! — assembles liquid + gas residual;
│                      # compute_currents for i_CO/i_H2/i_tot from the DOF vector
├── newton_affine.jl   # AffineLU (persistent sparse LU), newton_affine! (affine-
│                      # covariant kernel), tangent_predictor! — used by ALL stages
│                      # (1-5 + Fit) at production mesh (§9.5)
├── driver.jl          # the §9.5.5 continuation driver (anchor → tangent-predictor
│                      # sweep → §9.5.6 assertion → cache), shared by ALL stages.
│                      # Write it ONCE — five per-stage copies diverge.
├── solver.jl          # legacy: fd_jacobian! — keep for small-mesh debugging only,
│                      # not used by production drivers
├── run_stage1.jl      # Stage 1 driver: diffusion only + Dirichlet + simplified kinetics → STOP
├── run_stage2.jl      # Stage 2 driver: + Sherwood Neumann electrolyte BC → STOP
├── run_stage3.jl      # Stage 3 driver: + Stefan-Maxwell gas + R_PT phase transfer → STOP
├── run_stage4.jl      # Stage 4 driver: + Darcy pressure finalized → STOP
├── run_fit.jl         # Fit step (after Stage 4): i0_CO2R, λ_MHC, HER_pf, α_c
│                      # uses Stage 4 residual (φ_L=0, φ_s=V_applied) for speed → STOP
├── run_stage5.jl      # Stage 5 driver: + φ_L migration + φ_s solid Ohm's law
│                      # uses FITTED kinetics from run_fit.jl → STOP
├── plot_results.py    # Python plotting (matplotlib, Lato fonts, V vs SHE)
├── output_stage1/
│   └── cache/         # s1_V*.bin — Stage 1 converged solutions (5×N_CL DOFs)
├── output_stage2/
│   └── cache/         # s2_V*.bin — Stage 2 converged solutions (same DOFs as Stage 1)
├── output_stage3/
│   └── cache/         # s3_V*.bin — Stage 3 converged solutions (liquid + gas)
├── output_stage4/
│   └── cache/         # s4_V*.bin — Stage 4 converged solutions (liquid + gas + pressure)
└── output_stage5/
    └── cache/         # s5_V*.bin — Stage 5 converged solutions (fitted kinetics, 1480 DOFs at production N=80/120)
```

**Canonical driver execution order (a fresh agent MUST run in this sequence):**
```
1. run_stage1.jl          → STOP, human review of plots
2. run_stage2.jl          → STOP
3. run_stage3.jl          → STOP
4. run_stage4.jl          → STOP
5. run_fit.jl             → Nelder-Mead on (i0_CO2R, λ_MHC, log₁₀ HER_pf, α_c);
                            uses Stage 4 residual (φ_L=0, φ_s=V_applied) for speed;
                            polishes at 5 experimental voltages at every NM eval → STOP
6. run_stage5.jl          → sweep all 89 voltages with FITTED kinetics + full potential
                            physics (φ_L, φ_s as DOFs); anchor solve at V_c = 0 built from
                            physics, then tangent-predictor continuation in −0.025 V steps
                            (§9.5.5) — no cache seeding; post-solve ‖F‖∞ ≤ 1e-6 assertion
                            at every output voltage (§9.5.6) → STOP
7. plot_results.py        → final experimental-comparison plots (§18)
```

**Each `run_stageN.jl` script must:**
1. Define `CACHE_DIR` and `save_stageN` / `load_stageN` at the top
2. Print current kinetic parameters (i0, λ, HER_pf, α_c) for stale-cache detection
3. Cache every converged solution immediately after Newton converges
4. Generate all required plots at the end
5. Print a summary and **exit** — do NOT automatically chain to the next stage

**Ref-backed mutable parameters (for in-process fitting):**
```julia
const _i0_CO2R_ref = Ref{Float64}(540.0)
i0_CO2R() = _i0_CO2R_ref[]
set_i0_CO2R!(v) = (_i0_CO2R_ref[] = v)
```

This allows sweeping parameters without recompiling or restarting Julia.

---

## 14. Common Pitfalls and Solutions

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Approximate equilibrium initial guess | Buffer residuals O(10³–10⁵) at start; Newton takes hundreds of iterations or diverges even at α_buf=0.001 | Solve charge balance exactly with `find_zero(charge_balance, Bisection())`. Verify `max(abs(S_buf)) < 1e-6` before running Newton |
| FSG prefactor 1.013e-2 vs 1.00e-3 | Gas diffusivities 10× too high, CO₂ doesn't deplete | Use Perry's 9th ed. value: `1.00e-3` |
| Phase transfer double-counting | CO₂(aq) goes negative or oscillates | Never use constant `p_CO2` AND SM simultaneously. Use SM ω_CO₂ → p_CO₂ → R_PT only |
| 1/dx scaling on graded mesh | Jacobian entries O(1e9), Newton diverges | Use integrated FV form: `F = J_left - J_right + S×dx`, not `(J_right - J_left)/dx + S` |
| Cold-start at a deeply cathodic voltage | Newton never converges | Always anchor at the mild end and continue down the grid (§9.5.5): equilibrium guess → anchor → tangent-predictor continuation |
| JIT compilation slow | First Jacobian takes >20 min | Run gas-only solve first (triggers JIT cheaply) before monolithic Newton |
| Parallel Julia processes | Computer crash | **Never** launch >2-3 Julia processes simultaneously. The continuation sweep is fast in a single process (§9.5.10) |
| Stale cache at wrong kinetics | Wrong polarization curves | Always check i0 in cache header; delete and rerun if mismatched |
| **Stage 5 seeded from a cache instead of continued from the V_c = 0 anchor** | Jagged / non-monotone polarization; ionic charge-balance error at CL/electrolyte boundary 10–80 % on tolerance-converged points; φ_L changes sign between neighbouring voltages; branch flips on single 25 mV steps. Or — worse — a perfectly smooth curve on the *low-CO* branch (Δφ_L ≈ 3 mV, peak i_CO ~20 % below experiment) that passes ‖F‖∞ and charge conservation. Solid charge balance stays clean either way, hiding the bug. | Use the §9.5.5 continuation driver: anchor at V_c = 0 from the constructed equilibrium guess, then predictor–corrector in −0.025 V steps. Do not seed voltages from `output_stage4/cache/` (pre-fit kinetics, i0_CO2R = 540 vs ~5× smaller post-fit — fields ~50–150 mV off the correct regime) or from `output_fit/cache/` (right kinetics, wrong basin). See §11 Stage 5 gotcha #0 and §17.8. |
| Negative c from Newton step | `c = exp(u)` → exp(−50) OK but large step blows up | Clamp step: `max_step = 5.0` log-units. Never clamp to zero |
| **Stage 5**: FD Jacobian at Stage 5 | Newton stalls 100–400+ iterations, fails past ~130 mA/cm² even with line search | Use the ForwardDiff AD Jacobian (§9.3.2). Type-generalize the entire residual chain to `AbstractVector{<:Real}` first |
| **Stage 5**: FD Jacobian has corrupted kinetic cross-terms | φ_s/c cross-terms (~40) buried in truncation error of diagonal (~4×10⁷) | AD as above. This is the Brady et al. 2021 finding for Newman porous electrodes. |
| **Stage 5**: Row scaling in Newton | Kinetic feedback `∂i_far_vol/∂φ_s` is crushed to ~10⁻⁶ relative and Newton diverges | Never rescale Jacobian rows. The §9.5.3 kernel damps on correction norms and takes no residual norm at all |
| **Stage 5**: Newton cold-started far from a solution | `du[liq] ~ 30` log units (`exp(30) ≈ 10¹³` concentration changes); residual blows up | Fix the φ_L gauge (§9.4 items 3, 6), keep the iterate in-basin via the anchor + tangent-predictor continuation (§9.5.5), and keep the 5-log-unit step clamp. Direct sparse-LU Newton is then robust |
| **Stage 5**: Cell-1 φ_s Dirichlet override | Consistency check `|Δi|/i` returns ~2 (equal magnitudes, opposite signs) instead of ~10⁻⁹ | Use ghost-cell BC: `i_s[1] = −σ × (φ_s_gdl[1] − V_applied) / dx_half`, keep cell 1's continuity residual. NEVER override (§3.4) |
| **Stage 5**: hard ‖F‖∞ floor folded into the termination test | +37–47 % Newton iterations across the sweep for no accuracy gain | Terminate on the error estimate (`rtol = 1e-8`, §9.5.3); enforce ‖F‖∞ ≤ 1e-6 as a post-solve assertion (§9.5.6) |
| **Stage 5**: Type parameter named `T` in residual | Silent shadowing of the global `T` (temperature) constant. Cryptic type errors | Use `ElT` or `El` as the type parameter name in any `function ...(u::AbstractVector{ElT}) where {ElT<:Real}` signature (§11 Stage 5 gotchas #2) |
| **Stage 5**: Right-boundary flux uses Sherwood-only (no migration) | Ionic current at CL/electrolyte interface ≠ integrated faradaic current (rel_err 2–20%). φ_L profile has wrong sign. | Use `np_flux(c[i,N], c_eq[i], φ_L[N], 0, D_eff[i], z[i], L_eff)` with `L_eff = D_eff[i]/k_MT[i]` at the right boundary. This reduces to Sherwood when φ_L=0 (Stages 1–4) and correctly carries CO₃²⁻ migration when φ_L≠0 (Stage 5). See §11 Stage 5 gotchas #12. |
| **Stage 5**: continuation step carries the previous state forward but applies no predictor at all | Predictor is a full 25 mV stale at every step; iteration counts climb, and near the kinetics turn-on the iterate can fall out of the high-CO basin. Polarization curve goes jagged / non-monotone, i_CO zigzags in the plateau, φ_L changes sign between neighbouring voltages, \|Δi\|/i only ~10⁻³ instead of ~10⁻⁹. | Apply the tangent predictor `u .+= dV_c .* du/dV` before each corrector call; if the tangent is unavailable, at minimum add `dV_c` to **every** φ_s DOF (GDL and CL) — the φ_s boundary value moves by exactly `dV_c`. See §9.5.4 and §11 Stage 5 gotcha #9. |
| **Stage 5**: Anchor builder writes V_cathode (RHE) into φ_s DOFs instead of V_applied_SHE | Initial φ_s off by `V_RHE_to_SHE_offset ≈ 0.513 V`; the solve either diverges or converges to a shifted branch → jagged / shifted polarization. | Compute `V_applied_SHE = V_cathode - V_RHE_to_SHE_offset` in the anchor builder and in the residual. See §11 Stage 5 gotcha #9a, §15.2. |
| **Stage 5**: Running with default (pre-fit) kinetics instead of fitted | Stage 5 currents don't match experimental data; fit step results wasted | Always load fitted kinetic parameters at top of `run_stage5.jl`. Print and verify `i0_CO2R`, `λ_MHC`, `HER_prefactor`, `α_c` before the sweep. |

---

## 15. Potential Referencing — Nernst Shift, RHE, SHE

This section is **critical**. Getting the potential reference wrong will silently shift all kinetics and make fitting impossible. Read carefully.

### 15.1 Reference Frames

There are three potential reference frames in play:

| Frame | Symbol | Definition |
|-------|--------|------------|
| Standard Hydrogen Electrode | SHE | Absolute thermodynamic reference. pH-independent. |
| Reversible Hydrogen Electrode | RHE | SHE shifted by local pH: `E_RHE = E_SHE + (2.303 RT/F) × pH` |
| RHE at pH 8 (anode) | RHE@pH8 | The experimental reference used by Lees et al. The electrolyte is pH 8, so the anode reference potential is fixed at this pH. |

**The model works internally in SHE.** All experimental data from Lees et al. is reported in **RHE@pH8** (the anode-side reference). You must convert.

### 15.2 The Nernst Shift at 50°C

The conversion factor depends on temperature:

```
nernst_fac = 2.303 × R × T / F
           = 2.303 × 8.314462618 × 323.15 / 96485.332
           = 0.06413 V per pH unit   (at 50°C)
```

**NOT 0.0592 V** — that is the 25°C value. At 50°C (323.15 K), the Nernst slope is 8% larger.

The total shift for pH = 8:
```
V_RHE_to_SHE_offset = nernst_fac × pH_anode
                     = 0.06413 × 8
                     = 0.51305 V   (at 50°C, pH = 8)

V_SHE = V_RHE@pH8 − 0.51305 V
V_RHE@pH8 = V_SHE + 0.51305 V
```

**Implementation in params.jl:**
```julia
const pH_anode = 8.0
const nernst_fac = 2.303 * R_gas * T / F_const     # 0.06413 V/pH at 50°C
const V_RHE_to_SHE_offset = nernst_fac * pH_anode   # 0.51305 V
```

### 15.3 How the Shift Enters the Kinetics

The model receives `V_cathode` in RHE@pH8 (matching the experimental convention). Convert to SHE:

```julia
V_applied_SHE = V_cathode - V_RHE_to_SHE_offset      # V vs SHE
```

`V_applied_SHE` is the **Dirichlet boundary condition** for the φ_s DOF at the current collector (`x = −L_GDL`). It is NOT what enters the kinetics directly. The actual kinetic driving force at each CL cell uses three local quantities:

```julia
# CO₂R: CO₂ + H₂O + 2e⁻ → CO + 2OH⁻
U_nernst_CO2R = U_CO_0 - nernst_fac × pH_local
              = -0.11 - 0.06413 × pH_local      # V vs SHE

η_CO2R = (phi_s_local − phi_L_local) − U_nernst_CO2R   # negative = cathodic

# HER: 2H₂O + 2e⁻ → H₂ + 2OH⁻
U_nernst_HER = 0.0 − nernst_fac × pH_local
             = -0.06413 × pH_local              # V vs SHE

η_HER = (phi_s_local − phi_L_local) − U_nernst_HER
```

where:
- `phi_s_local` = local solid potential at this CL cell (from the φ_s DOF, §3.4). Differs from `V_applied_SHE` by the GDL + CL ohmic drop (50–100 mV at typical currents).
- `phi_L_local` = local liquid potential at this CL cell (from the φ_L DOF, §3.1).
- `pH_local = −log₁₀(c_H⁺ / 1000)` from the local c_H (c_H in mol/m³, divide by 1000 for mol/L).

**Three common mistakes:**
1. **Using pH = 8 (the anode reference pH) for the Nernst correction.** The local CL pH is typically 9–12 due to OH⁻ production, which shifts U_eq by 60–250 mV. Use the local pH from c_H, not the bulk anolyte pH.
2. **Forgetting to subtract `phi_L_local`.** The driving force for electron transfer is the metal-liquid potential difference `(φ_s − φ_L)`, not just `φ_s`. φ_L differs from its anolyte reference by 10–50 mV due to migration inside the CL.
3. **Using V_applied as φ_s in the kinetics instead of the local φ_s DOF.** The solid phase has a real ohmic drop from the current collector (where V_applied is set) through the GDL fibers and CL carbon support to the catalyst surface. At 1 A/cm² this drop is ~50 mV in the GDL alone. Treating V_applied as if it equals the catalyst-surface potential biases the fitted i₀ by a factor of ~2–3. Use `phi_s_local` from the §3.4 DOF.

All three corrections must be applied together — the kinetics functions in §4 take both `phi_s_local` and `phi_L_local` as explicit arguments and compute the local pH internally from `c_H`.

### 15.4 Standard Reduction Potentials

| Reaction | E⁰ (V vs SHE) | Variable |
|----------|----------------|----------|
| CO₂ + H₂O + 2e⁻ → CO + 2OH⁻ | −0.11 | `U_CO_0` |
| 2H₂O + 2e⁻ → H₂ + 2OH⁻ | 0.00 | (implicit) |

### 15.5 Converting Experimental Data Points

When plotting, convert experimental cathode potentials to SHE:
```python
_V_RHE_to_SHE = 2.303 * 8.314462618 * 323.15 / 96485.332 * 8.0   # 0.51305 V at 50°C (CODATA R, F)
E_exp_SHE = E_exp_RHE_pH8 - _V_RHE_to_SHE
```

**Always plot as −E_cath vs SHE** (positive x-axis = more cathodic). The sign convention: the x-axis shows `−V_SHE`, so that more negative cathode potentials appear to the right.

---

## 16. Experimental Data Tables — Lees et al. 2024

Reference: Lees, E.W. et al., *Nat. Chem. Eng.* **1** (2024). System: Ag/C catalyst, Sustainion X37-50 AEM, 50°C, 1M CsHCO₃ electrolyte.

### 16.1 Primary Fitting Data: Partial Current Densities (5 points)

These are the data points you fit against. Source: Fig. 2a of Lees et al.

| E_cath (V vs RHE@pH8) | i_CO (mA/cm²) | i_H₂ (mA/cm²) | i_total (mA/cm²) | FE_CO (%) | FE_H₂ (%) |
|------------------------|----------------|----------------|-------------------|-----------|-----------|
| −0.816 | 35.1 | 29.9 | 65.0 | 54.0 | 46.0 |
| −1.205 | 189.6 | 49.3 | 238.9 | 79.4 | 20.6 |
| −1.644 | 299.1 | 119.6 | 418.7 | 71.4 | 28.6 |
| −2.079 | 204.3 | 449.9 | 654.2 | 31.2 | 68.8 |
| −2.413 | 64.6 | 850.5 | 915.1 | 7.1 | 92.9 |

**To convert to V vs SHE:** subtract 0.51305 V:

| E_cath (V vs SHE) | i_CO (mA/cm²) | i_H₂ (mA/cm²) |
|--------------------|----------------|----------------|
| −1.329 | 35.1 | 29.9 |
| −1.718 | 189.6 | 49.3 |
| −2.157 | 299.1 | 119.6 |
| −2.592 | 204.3 | 449.9 |
| −2.926 | 64.6 | 850.5 |

**Hardcode these arrays directly:**
```julia
# Experimental data — Lees et al. 2024, Nat. Chem. Eng.
const E_exp_RHE = [-0.816, -1.205, -1.644, -2.079, -2.413]   # V vs RHE@pH8
const i_CO_exp  = [35.1, 189.6, 299.1, 204.3, 64.6]          # mA/cm²
const i_H2_exp  = [29.9, 49.3, 119.6, 449.9, 850.5]          # mA/cm²

# Derived
const E_exp_SHE = E_exp_RHE .- V_RHE_to_SHE_offset            # V vs SHE
const FE_CO_exp = i_CO_exp ./ (i_CO_exp .+ i_H2_exp) .* 100   # %
const FE_H2_exp = i_H2_exp ./ (i_CO_exp .+ i_H2_exp) .* 100   # %
```

---

## 17. Fitting Strategy and Procedure

### 17.1 Fitting Targets

**Primary fit (4 parameters → 10 data points):**
- Fit to `i_CO(V)` and `i_H₂(V)` at the 5 experimental voltages above
- This is 10 data points (5 × 2 currents) vs 4 adjustable parameters
- The fit is overdetermined (good), so you minimize a least-squares objective

### 17.2 The 4 Fitting Parameters

| Parameter | Variable | Initial Guess | Physical Range | Sensitivity |
|-----------|----------|--------------|----------------|-------------|
| CO₂R exchange current density | `i0_CO2R` | 540 A/m² | 10–5000 A/m² | High — scales all CO partial currents linearly |
| MHC reorganization energy | `λ_MHC` | 1.8 eV | 0.5–3.0 eV | Medium — controls onset voltage and Tafel slope curvature |
| HER prefactor | `HER_prefactor` | 2.409×10¹³ | 10⁸–10¹⁸ | High — scales all H₂ partial currents; fit in log-space |
| HER transfer coefficient | `α_c_HER` | 0.0872 | 0.01–0.5 | Medium — controls HER Tafel slope. Empirical — absorbs Frumkin + mechanism-crossover effects, not a true single-step α |

**Initial guess / bounds for Nelder-Mead** (required for reproducibility):
```julia
# Starting simplex vertex — these are the default (initial-guess) kinetics
params0 = [540.0, 1.8, log10(2.409e13), 0.0872]
# Hard lower/upper bounds (enforced inside the objective by returning 1e6 if violated):
lower   = [ 10.0, 0.5,  8.0, 0.01]
upper   = [5000.0, 3.0, 18.0, 0.50]
# NM termination
Optim.Options(iterations = 200, g_tol = 1e-6, f_reltol = 1e-6, show_trace = true)
```
A typical run with these settings converges after ~120–180 function evaluations. The final fitted values are what you discover — do not expect them to match the initial guesses. See §17.6 for the acceptance criteria used to decide when the fit is "good enough" to stop.

**Use Ref-backed parameters so you can sweep without recompiling:**
```julia
const _i0_ref = Ref{Float64}(540.0)
const _lam_ref = Ref{Float64}(1.8)
const _her_pf_ref = Ref{Float64}(2.409e13)
const _ac_ref = Ref{Float64}(0.0872)

i0_CO2R() = _i0_ref[]
lambda_MHC() = _lam_ref[]
HER_prefactor() = _her_pf_ref[]
alpha_c_HER() = _ac_ref[]

set_i0_CO2R!(v) = (_i0_ref[] = v)
set_lambda_MHC!(v) = (_lam_ref[] = v)
set_HER_prefactor!(v) = (_her_pf_ref[] = v)
set_alpha_c_HER!(v) = (_ac_ref[] = v)
```

### 17.3 Pre-Fitting Voltage Polish (Required)

**Problem:** The Stage 4 voltage sweep uses a uniform 0.025 V grid (e.g., V_RHE = −0.300, −0.325, −0.350, …). The 5 experimental data points (−0.816, −1.205, −1.644, −2.079, −2.413 V RHE) do **not** fall on this grid. Evaluating the fitting objective by interpolating between cached grid solutions introduces interpolation error that corrupts the optimizer's gradient estimates — especially near the CO rollover where i_CO changes rapidly with voltage.

**Solution:** Before fitting, solve Stage 4 at the **exact** 5 experimental voltages and cache the results. Each solve warm-starts from the nearest grid-voltage cache entry.

```julia
# ── Pre-fitting polish: solve at exact experimental voltages ──
function polish_at_experimental_voltages!(exp_cache, E_exp_RHE, grid_cache, grid_voltages)
    for V_rhe in E_exp_RHE
        # Find nearest grid voltage for warm-start
        idx_nearest = argmin(abs.(grid_voltages .- V_rhe))
        u_warm = copy(grid_cache[idx_nearest])

        # Solve at exact experimental voltage
        V_cathode = V_rhe - V_RHE_to_SHE_offset
        res! = (F_, u_) -> residual_stage4!(F_, u_, V_cathode, cl_mesh, gdl_mesh,
                                             c_eq_vec)
        # Use the §9.5 production kernel: newton_affine! + persistent sparse LU + AD Jacobian
        info = newton_affine!(u_warm, res!; ly=layout4, plu=plu,
                              rtol=1e-8, max_iter=40, alpha_init=1.0,
                              n_log_dofs=5*N_CL)
        @assert info.converged "Polish failed at V_RHE = $V_rhe"
        exp_cache[V_rhe] = u_warm
        @printf("Polish: V_RHE = %.3f → converged in %d iters (‖F‖∞ = %.2e)\n",
                V_rhe, info.iterations, info.residual)
    end
end

# Run once before any fitting
exp_cache = Dict{Float64, Vector{Float64}}()
polish_at_experimental_voltages!(exp_cache, E_exp_RHE, s4_cache, s4_voltages)
```

**This function must be called at the start of every fitting iteration** (after updating kinetic parameters) because the converged solution at each experimental voltage depends on the current kinetic parameters. The warm-start from the previous iteration's polished solution is excellent — typically 2–5 Newton iterations per voltage per fitting iteration.

### 17.4 Fitting Objective Function

```julia
function fitting_objective(params::Vector{Float64})
    set_i0_CO2R!(params[1])
    set_lambda_MHC!(params[2])
    set_HER_prefactor!(10.0^params[3])   # fit in log-space
    set_alpha_c_HER!(params[4])

    # Re-polish at experimental voltages with updated parameters
    # Warm-start from previous polish (stored in exp_cache)
    for V_rhe in E_exp_RHE
        u_warm = copy(exp_cache[V_rhe])       # previous iteration's solution
        V_cathode = V_rhe - V_RHE_to_SHE_offset
        res! = (F_, u_) -> residual_stage4!(F_, u_, V_cathode, cl_mesh, gdl_mesh,
                                             c_eq_vec)
        info = newton_affine!(u_warm, res!; ly=layout4, plu=plu,
                              rtol=1e-8, max_iter=40, alpha_init=1.0,
                              n_log_dofs=5*N_CL)
        info.converged || return 1e6          # penalty for non-convergence
        exp_cache[V_rhe] = u_warm             # update cache for next iteration
    end

    sse = 0.0
    for (k, V_rhe) in enumerate(E_exp_RHE)
        u = exp_cache[V_rhe]

        # Extract model currents [mA/cm²]
        # Stage 4: phi_s = V_applied_SHE, phi_L = 0
        V_applied_SHE = V_rhe - V_RHE_to_SHE_offset
        i_CO_model = compute_i_CO(u, V_applied_SHE) / 10.0    # A/m² → mA/cm²
        i_H2_model = compute_i_H2(u, V_applied_SHE) / 10.0

        # Relative error (log-space for currents spanning decades)
        sse += (log10(max(i_CO_model, 0.1)) - log10(i_CO_exp[k]))^2
        sse += (log10(max(i_H2_model, 0.1)) - log10(i_H2_exp[k]))^2
    end
    return sse
end
```

**Why log-space?** The currents span 3 decades (29.9 to 850.5 mA/cm²). Linear least-squares would overweight high-current points. Log-space gives equal weight per decade.

**Current extractors (needed above — define once in `assembly.jl`):**
```julia
# Partial geometric current densities in A/m² (of CL geometric cross-section).
# Both reactions are 2-electron so n_e = 2; the integral is over x ∈ [0, L_CL].
# For Stages 1–4 (simplified kinetics): pass V_applied_SHE as phi_s, 0.0 as phi_L.
function compute_i_CO(u, V_applied_SHE, cl_mesh)
    c_all = unpack_log_conc(u, cl_mesh.N)           # 5 × N
    i = 0.0
    for ix in 1:cl_mesh.N
        r = R_CO2R(c_all[5,ix], V_applied_SHE, 0.0, c_all[1,ix],
                   a_v_CL, lambda_MHC(), i0_CO2R(), 2)      # mol/m³/s
        i += 2 * F_const * r * cl_mesh.dx[ix]                # A/m²
    end
    return i
end
# Analogous compute_i_H2 using R_HER(...); i_total = i_CO + i_H2.
```
For Stage 5 (full physics), replace `V_applied_SHE` with `phi_s_cl[ix]` and `0.0` with `phi_L[ix]` when calling these — same accumulator but reading from the φ_s and φ_L DOFs.

### 17.5 Fitting Procedure — Step by Step

**Step 1: Manual exploration.** Before any automated fitting, sweep each parameter individually while holding others fixed. Plot polarization curves overlaid on experimental data. Build intuition:
- `i0_CO2R` shifts CO curve up/down
- `λ_MHC` shifts CO onset voltage left/right
- `HER_prefactor` shifts H₂ curve up/down
- `α_c_HER` changes H₂ Tafel slope

**Step 2: Fit CO₂R parameters first.** Fix HER parameters at initial guesses. Sweep `i0_CO2R` and `λ_MHC` to match i_CO at the 5 data points. CO₂R is nearly independent of HER at moderate voltages.

**Step 3: Fit HER parameters.** With CO₂R parameters fixed from Step 2, sweep `HER_prefactor` and `α_c_HER` to match i_H₂. HER is sensitive to local pH (which depends on CO₂R rates), so iterate Steps 2-3 if needed.

**Step 4: Joint refinement.** Use Nelder-Mead or a derivative-free optimizer on all 4 parameters:
```julia
using Optim
result = optimize(fitting_objective,
                  [540.0, 1.8, log10(2.409e13), 0.0872],
                  NelderMead(),
                  Optim.Options(iterations=200, show_trace=true))
```

### 17.6 Fit Quality Acceptance Criteria — When to Stop Tweaking

These are the acceptance criteria for the 4-parameter Nelder-Mead fit against Lees et al. 2024 Fig. 2a. **A fresh agent must stop tweaking parameters once ALL of the criteria below are satisfied simultaneously** — iterating past this point is over-fitting.

**Per-point current error tolerances** (at the 5 Lees voltages, V_RHE = −0.816, −1.205, −1.644, −2.079, −2.413):

| Point | Tolerance on i_CO | Tolerance on i_H₂ | Notes |
|-------|-------------------|-------------------|-------|
| V_RHE = −0.816 | ±30% | ±30% | onset |
| V_RHE = −1.205 | ±30% | ±30% | FE_CO peak region |
| V_RHE = −1.644 | ±30% | ±30% | near CO rollover |
| V_RHE = −2.079 | ±30% | ±30% | H₂-dominated |
| V_RHE = −2.413 | **±80% on i_CO acceptable** (transport-limited plateau); ±30% on i_H₂ | ±30% | See note below |

The −2.413 V RHE point sits at the CO transport-limited plateau where small changes in local CO₂(aq) depletion produce disproportionately large fractional changes in i_CO. A ~80% error on i_CO at this single point is acceptable as long as all other metrics pass; tightening this point alone tends to break the fit elsewhere.

**Qualitative curve-shape criteria (hard requirements):**

1. **HER prediction near-perfect vs Lees experimental points.** All 5 i_H₂ model values should fall within ±30% of experimental; visually the model curve should pass through or within one marker-width of every H₂ experimental point on both linear and log panels.
2. **CO2R good agreement on FE (faradaic efficiency) scale.** FE_CO model curve passes through or very close to every experimental FE_CO point; peak FE_CO is 70–85% (experimental is 79.4% at V_RHE = −1.205).
3. **No zigzags or discontinuities in derivatives.** Both i_CO(V) and i_H₂(V) must be smooth curves with monotone (or smoothly varying) first derivatives. Any visible kink, step, or wiggle on either linear or log polarization panels is a **hard failure** — it indicates the solver jumped between solution branches and the fit is not on a single physical branch.
4. **One continuous branch across all 89 voltages.** The Stage 5 sweep (which uses the fitted kinetics) must converge at every one of the 89 grid voltages in §9.5.5's `V_RHE_grid` (−0.300 to −2.500 V RHE, 0.025 V step). If fewer than 89/89 converge, the sweep is incomplete and the fit cannot be accepted. All 89 converged solutions must lie on the *same* solution branch — verify by overlaying the 5 polished Lees-voltage points on the 89-point sweep curve; all 5 filled circles must sit exactly on the sweep line (no offset, no branch split).
5. **Linear AND log polarization plots both smooth.** The log panel often reveals wiggles that the linear panel hides (especially at low current); both must pass the smoothness test.

**Quantitative sanity checks (auto-verifiable):**

| Metric | Target |
|--------|--------|
| i_CO relative error at points 1–4 | ≤ 30% |
| i_CO relative error at point 5 (V_RHE = −2.413) | ≤ 80% (transport-limit plateau) |
| i_H₂ relative error at all 5 points | ≤ 30% |
| FE_CO peak location | Within 100 mV of experimental (near V_RHE = −1.205) |
| FE_CO peak value | 70–85% (experimental is 79.4%) |
| Sweep coverage | 89/89 voltages converged on one continuous branch |
| Monotonicity of i_CO(V) derivative sign changes | ≤ 1 (smooth rise, smooth rollover — no extra wiggles) |
| Monotonicity of i_H₂(V) | strictly monotone increasing in |V_RHE| |

**Decision rule: STOP fitting when all of the following are true:**
- All per-point current tolerances above are met
- All qualitative criteria 1–5 above are satisfied
- Stage 5 89-voltage sweep (with fitted kinetics) completed with 89/89 convergence on one continuous branch

If any of these fail, investigate the solver (warm-start source, max_iter, branch drift — see §11 Stage 5 gotchas) *before* tweaking parameters further. Parameter tweaks will not fix a broken sweep.

### 17.7 Diagnostics When Fit Is Poor

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| CO onset too early (< −0.7 V RHE) | λ_MHC too small | Increase λ_MHC (shifts onset cathodic) |
| CO onset too late (> −1.2 V RHE) | λ_MHC too large or i0 too small | Decrease λ_MHC or increase i0_CO2R |
| CO peak FE too low | i0_CO2R too low relative to HER | Increase i0_CO2R or decrease HER_prefactor |
| H₂ dominates at all potentials | HER_prefactor too high or α_c too large | Decrease HER_prefactor |
| CO doesn't roll over at high |η|| CO₂ transport not limiting enough | Check D_eff, k_MT, gas diffusivities |

### 17.8 Important: Cache Invalidation After Re-Fitting

When you change kinetic parameters, **all cached solutions are stale**. You must:
1. Delete the cache directory for the current stage and all later stages
2. Re-run the continuation sweep from scratch with the new parameters
3. Never mix solutions from different kinetics in the same cache

The stage driver prints `i0_CO2R = XXX A/m²` at startup — always verify this matches the intended kinetics before reusing a cache.

**The refit sweep is a diagnostic artifact — Stage 5 does NOT warm-start from it.** After Nelder-Mead converges, `run_fit.jl` applies the fitted kinetics via the Ref setters and performs one final 89-voltage Stage 4 sweep at the fitted parameters, writing the result to `output_fit/cache/fit_V*.bin`. Produce it: it is the Stage-4 physics at the fitted kinetics, and plotting it against both the pre-fit Stage 4 sweep and the Stage 5 sweep is what separates the effect of the parameter fit from the effect of adding the potential fields. But it is **not** the Stage 5 initial guess. Stage 5 is seeded once, at V_c = 0, from the constructed equilibrium guess of §9.5.5, and every subsequent voltage is reached by continuation. Seeding individual voltages from a cache — this one, or the pre-fit `output_stage4/cache/s4_V*.bin` — risks landing on the low-CO branch, which passes ‖F‖∞ and charge conservation but gives Δφ_L ≈ 3 mV and a peak i_CO ~20 % below experiment; the pre-fit cache additionally puts every cell in the wrong kinetic regime (~5× larger `i0_CO2R`). See §11 Stage 5 gotcha #0 and the §14 pitfalls-table row "Stage 5 seeded from a cache instead of continued from the V_c = 0 anchor".

`run_stage5.jl` should therefore build its own anchor and read no warm-start cache:
```julia
u_state = make_u0_stage5_anchor(c_eq_vec, cl_mesh, gdl_mesh)   # V_c = 0, constructed from physics
```

### 17.9 Second-Tier Fitting: Letting k_GL Float

**Only attempt this after exhausting the 4 kinetic parameters.** If the best kinetics-only fit still shows systematic errors — particularly if the CO rollover at high overpotential is too sharp or too gradual — then the gas-liquid mass transfer coefficient `k_GL` becomes the 5th adjustable parameter.

**Why k_GL and not other transport parameters?**
- `k_GL` controls how fast CO₂ dissolves from the gas phase into the liquid electrolyte in the CL pores
- Default value: `k_GL = 1e-4 m/s` (order-of-magnitude estimate, not well-constrained experimentally)
- Physical range: `1e-5 to 1e-3 m/s`
- It directly controls the CO₂(aq) concentration in the CL, which determines where CO₂R becomes mass-transport-limited and CO production rolls over
- If `k_GL` is too high, CO₂(aq) is always at Henry's law equilibrium → no transport limitation → CO never rolls over (FE_CO stays high too long)
- If `k_GL` is too low, CO₂(aq) is severely depleted → CO rolls over too early → FE_CO peak shifts to milder potentials

**Procedure:**

**Step 6 (only if Steps 1-5 fail):** Hold the 4 kinetic parameters at their best-fit values from Step 4. Sweep `k_GL` over `[1e-5, 3e-5, 1e-4, 3e-4, 1e-3]` and plot polarization curves + FE_CO for each. Look for which `k_GL` best matches the shape of the CO rollover region (points 3-5 in the experimental data).

**Step 7: Joint 5-parameter refinement.** If a better `k_GL` is found, re-optimize all 5 parameters jointly:

```julia
# Add k_GL as 5th parameter
const _kGL_ref = Ref{Float64}(1e-4)
k_GL() = _kGL_ref[]
set_k_GL!(v) = (_kGL_ref[] = v)

function fitting_objective_5(params::Vector{Float64})
    set_i0_CO2R!(params[1])
    set_lambda_MHC!(params[2])
    set_HER_prefactor!(10.0^params[3])
    set_alpha_c_HER!(params[4])
    set_k_GL!(10.0^params[5])              # fit in log-space

    # ... same SSE computation as before ...
end

result = optimize(fitting_objective_5,
                  [540.0, 1.8, log10(2.409e13), 0.0872, log10(1e-4)],
                  NelderMead(),
                  Optim.Options(iterations=300, show_trace=true))
```

**Diagnostic table for k_GL effects:**

| Symptom | k_GL issue | Fix |
|---------|-----------|-----|
| CO rolls over too late (FE_CO still high at −2.0 V RHE) | k_GL too high — CO₂(aq) never depletes | Decrease k_GL |
| CO rolls over too early (FE_CO drops before −1.5 V RHE) | k_GL too low — CO₂(aq) depleted prematurely | Increase k_GL |
| Good FE_CO match but CO rollover shape wrong | k_GL is the likely culprit | Sweep k_GL with kinetics fixed |

**Important constraints:**
- `k_GL` must stay within `[1e-5, 1e-3]` — outside this range is unphysical
- If the optimizer pushes `k_GL` to a boundary, the model structure (not k_GL) is the problem
- After adjusting `k_GL`, re-check that all buffer and transport physicality checks still pass (§12)
- `k_GL` changes invalidate all caches, same as kinetics changes (§17.7)

---

## 18. Experimental Comparison Plots (Required)

### 18.1 4-Panel Summary Figure

Generate after each fitting iteration:

| Panel | Content | Axes |
|-------|---------|------|
| (a) Linear polarization | i_CO, i_H₂, i_total vs −E_cath | x: −V (V vs SHE), y: mA/cm² |
| (b) Log polarization | Same data, log y-axis | x: −V (V vs SHE), y: log mA/cm² |
| (c) CO faradaic efficiency | FE_CO and FE_H₂ model + exp points | x: −V (V vs SHE), y: % |

Overlay experimental scatter points on all panels. Model curves from the 25 mV resolution voltage sweep.

**Plot formatting:**
- All x-axes: `−E_cath [V vs SHE]` (positive = more cathodic)
- x range: 0.7 to 3.1
- Experimental: open markers (□ for CO, △ for H₂)
- Model: solid lines

### 18.2 FE_CO and FE_H₂ Overlay

Dedicated figure with both FE_CO and FE_H₂ (= 100 − FE_CO) on the same axes. This shows the crossover point where H₂ begins to dominate. Experimental FE points:

```python
FE_CO_exp  = [54.0, 79.4, 71.4, 31.2, 7.1]    # %
FE_H2_exp  = [46.0, 20.6, 28.6, 68.8, 92.9]    # %
```
