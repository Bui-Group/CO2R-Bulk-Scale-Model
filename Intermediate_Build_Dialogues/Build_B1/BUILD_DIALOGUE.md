# CO₂R Bulk Model — Build Dialogue (Session 2)

**Purpose:** Complete record of the second collaborative build session between a human researcher and Claude for the 1D CO₂R catalyst layer model. This session rebuilt the model from the implementation guide, discovering and fixing several bugs in the process.

**Date:** 2026-03-25 to 2026-03-26
**Model:** CO₂R Bulk-Scale Julia Model (no agglomerate, no membrane, no anode)
**Reference system:** Ag/C catalyst, 1M CsHCO₃, 50°C (Lees et al. 2024, Nat. Chem. Eng.)
**AI Model used:** Claude Opus 4.6 (primary)

---

## 1. Task Initiation

**Human:** Provided the same 1,498-line implementation guide (`CO2R_BULK_MODEL_IMPLEMENTATION_GUIDE.md`) used in Session 1. Instruction: *"Follow the implementation guide provided in the folder [guide folder] exactly. Take note of all the stop gates allowing for human operator review at each stage."*

**Claude:** Read the full specification. Identified the 4-stage build plan with mandatory stop gates. Began implementation at Stage 1.

---

## 2. Stage 1: Completed Successfully

Stage 1 (Liquid Nernst-Planck + Dirichlet BCs) was implemented and converged without major issues.

**Key results:**
- 897 converged solutions from V = −0.3 to −2.5 V (RHE@pH8)
- PAC continuation used for the voltage sweep
- Bootstrap: buffer ramp → electrochemistry ramp → PAC sweep
- All physicality checks passed

**Human approved Stage 1:** *"The results look good. Please proceed."*

---

## 3. Stage 2: The Neumann BC Sign Error (Again)

### 3.1 The Problem

The task was to replace the Dirichlet BC at x = L_CL with a convective Neumann BC:

```
J_i(L_CL) = k_MT,i × (c_eq,i − c_i(L_CL))    [from implementation guide, §7.2]
```

Multiple convergence strategies were attempted:

| Attempt | Strategy | Result |
|---------|----------|--------|
| 1 | Direct Dirichlet → Neumann switch | Newton diverges |
| 2 | α_neumann blending (0→1) | Bifurcation at α ≈ 0.883 |
| 3 | Echem-off → BC switch → echem-on | Echem ramp fails at α_elec ≈ 0.23 |
| 4 | V=0 bootstrap with Neumann | HER rate still significant, fails |
| 5 | Physics-informed initial guess | Newton oscillates at \|\|F\|\| ≈ 2000 |
| 6 | PTC (Pseudo-Transient Continuation) | Stalls at \|\|F\|\| ≈ 2×10⁴ |
| 7 | Sh ramp: Sh=10000→100 (Newton) | Works! Converges to Sh=100 |
| 8 | Sh ramp: 100→30 (Newton+PTC) | Fails at Sh=98 |
| 9 | Sh ramp with PTC-SER dt adaptation | Fails at every Sh step below 100 |

### 3.2 Root Cause Discovery

After attempt 9 failed even with proper u-backup and SER (Switched Evolution Relaxation) dt adaptation, the initial residual at Sh=99 was only ~28 (a tiny perturbation from Sh=100). Yet PTC diverged to ||F|| ≈ 10⁶ after 8000 iterations. This proved the issue was not numerical — it was a sign error in the physics.

**The sign analysis:**

The Dirichlet BC (Stage 1) computes the right-face flux as:
```julia
J_dirichlet = diffusion_flux(c[i,N], c_eq[i], D_eff[i], dx_face_right)
            = D_eff * (c[i,N] - c_eq[i]) / dx
```
This is **positive** when c_CL > c_bulk (species leaves the CL). Correct.

The Neumann BC (Stage 2) as written in the guide and implemented:
```julia
J_neumann = k_MT[i] * (c_eq[i] - c[i,N])
```
This is **negative** when c_CL > c_bulk (species flows INTO the CL). **Wrong direction.**

The sign is reversed relative to the Dirichlet convention. When OH⁻ accumulates in the CL (c_OH > c_eq_OH), the wrong sign pumps MORE OH⁻ into the CL instead of allowing it to leave. This creates an unstable positive feedback: OH⁻ builds up → more OH⁻ enters → more buildup → divergence.

### 3.3 The Fix

One line change in both `assembly.jl` and `run_stage2.jl`:

```julia
# Before (wrong — from implementation guide §7.2):
J_neumann = k_MT[i] * (c_eq[i] - c[i,N])

# After (correct — consistent with diffusion flux convention):
J_neumann = k_MT[i] * (c[i,N] - c_eq[i])
```

### 3.4 Result After Fix

Everything converged immediately:

- **Phase 1 (Sh ramp ∞→100):** All steps converged in 4–5 Newton iterations
- **Phase 2 (PTC-SER ramp 100→30):** 8 steps, all converged in 11–12 iterations
- **PAC voltage sweep (−0.3 → −2.5):** 911 solutions cached, all converged

### 3.5 Stage 2 Results

| V_RHE [V] | i_CO [mA/cm²] | i_H₂ [mA/cm²] | FE_CO [%] |
|-----------|---------------|----------------|-----------|
| −0.50 | 0.27 | 24.0 | 1.1 |
| −0.80 | 18.5 | 44.2 | 29.5 |
| −1.00 | 39.4 | 61.8 | 38.9 |
| −1.20 | 66.5 | 82.7 | 44.6 |
| −1.50 | 98.6 | 141.5 | 41.1 |
| −2.00 | 93.0 | 415.2 | 18.3 |
| −2.50 | 67.7 | 1335.9 | 4.8 |

**Observations:**
- CO₂R peaks at ~99 mA/cm² near V = −1.5 (mass-transfer limited by liquid diffusion)
- HER dominates at high overpotential (expected)
- Boundary pH at Sh=30: ~8.8 at x=L_CL (elevated from equilibrium 8.2, OH⁻ removal limited)

### 3.6 Lessons Learned

This is the **same sign error** discovered in Session 1 (§7.1 of the original BUILD_DIALOGUE). The implementation guide's formula `J = k_MT × (c_eq − c)` uses a mass-transfer convention where positive J means flux from bulk to surface. But the finite-volume code uses positive J = flux in +x direction (from CL to bulk). The conventions are opposite.

**For the paper:** The guide's formula is ambiguous. A future revision should explicitly state the sign convention relative to the FV flux direction.

**Human approved Stage 2:** *"Results look good. Please continue."*

---

## 4. Stage 3: Stefan-Maxwell Gas Transport

### 4.1 Implementation

Created `gas_transport.jl` with:
- Fuller-Schettler-Giddings (FSG) binary diffusivities
- Stefan-Maxwell mixture-average diffusion
- Darcy flow with permeability-weighted pressure gradient
- Gas-liquid phase transfer: R_PT = a_v_GL × k_GL × (H_eff × p_CO₂ − c_CO₂(aq))
- Henry's law coupling at x=0

**New DOFs:** 4 × (30 + 50) = 320 gas DOFs → **570 total DOFs**

Modified `assembly.jl` with `residual_stage3!` combining liquid NP + gas SM + Darcy + phase transfer.

### 4.2 Bug #1: FSG Molar Mass Units

The FSG formula (Perry's 9th ed.) expects molar mass in **g/mol**:

```
D_ij [cm²/s] = 1.00×10⁻³ × T^1.75 × √(1/M_i + 1/M_j) / (p × (v_i^{1/3} + v_j^{1/3})²)
```

But `params.jl` stores `M_gas` in **kg/mol** (e.g., M_CO₂ = 0.04401). The `sqrt(1/M)` term is off by √1000 ≈ 31.6:

| Quantity | With kg/mol (wrong) | With g/mol (correct) | Ratio |
|----------|-------------------|---------------------|-------|
| D_CO₂-CO | 5.876×10⁻⁴ m²/s | 1.858×10⁻⁵ m²/s | 31.6× |
| D_CO₂-H₂ | 2.322×10⁻³ m²/s | 7.342×10⁻⁵ m²/s | 31.6× |

**Impact:** With 31.6× too-high gas diffusivities, the GDL provided essentially unlimited CO₂ supply. Gas x_CO₂ barely depleted (0.750 → 0.745) even at extreme overpotential. This gave CO currents of ~2800 mA/cm² (vs experimental ~300 mA/cm²).

**Fix:**
```julia
Mi_g = M_gas[i] * 1000.0   # kg/mol → g/mol
Mj_g = M_gas[j] * 1000.0
```

**Verification:** D_CO₂-N₂ at 25°C = 0.164 cm²/s (matches literature value of 0.165 cm²/s).

### 4.3 Bug #2: PAC Tangent Dominated by Pressure DOF

The PAC solver computes a unit tangent (ψ, σ) in the augmented (u, V) space. With 570 DOFs including pressure (~101325 Pa), the pressure components dominated ψ, pushing σ → 0. Result: each PAC step advanced only ~0.00005 V in voltage. After 2158 steps, V had only reached −1.016 (target: −2.5).

**Fix:** Replaced PAC with direct voltage stepping for Stage 3 (no turning points expected):
- Step V in fixed increments (dV = −0.005 to −0.01 V)
- Newton-solve at each step, warm-starting from previous solution
- Adaptive dV: grow on success, halve on failure

**Result:** 223 solutions from V = −0.3 to −2.5, all converged in 3–5 Newton iterations. Runtime: ~60 seconds.

### 4.4 Stage 3 Results (After FSG Fix)

| V_RHE [V] | i_CO [mA/cm²] | i_H₂ [mA/cm²] | FE_CO [%] | x_CO₂(CL exit) |
|-----------|---------------|----------------|-----------|-----------------|
| −0.50 | 0.3 | 24.0 | 1.3 | 0.750 |
| −1.00 | 320 | 35.5 | 90.0 | 0.733 |
| −1.50 | 2154 | 67.2 | 97.0 | 0.615 |
| −2.00 | 2308 | 404 | 85.1 | 0.603 |
| −2.50 | 2165 | 1307 | 62.4 | 0.608 |

**CO current still ~7× too high.** Root cause: phase transfer (a_v_GL × k_GL = 3600 s⁻¹) keeps liquid CO₂ at ~5–6 mol/m³ throughout the CL. In Stage 2, CO₂ entered only at x=0 and depleted across the CL. In Stage 3, every cell gets CO₂ from the gas phase.

**Human:** *"The CO partial current density looks much too high. Please try again and look to see why the CO overshot the data so much."*

**Claude:** Investigated and explained the physics — the high currents are physically correct for these parameters. The kinetic parameters (fitted for the diffusion-limited Stage 2 regime) need refitting for the gas-coupled Stage 3 regime.

**Human approved Stage 3:** *"Okay, results look good. Please continue."*

---

## 5. Stage 4: Sechenov Correction

### 5.1 Implementation

Added Sechenov salting-out correction to Henry's law:

```
log₁₀(H/H₀) = −Σ (h_CO₂ + h_ion) × [ion]/1000
```

With Sechenov parameters from Schumpe 1993:
- h_CO₂ = −0.02599, h_Cs⁺ = +0.0759, h_OH⁻ = +0.0893
- h_HCO₃⁻ = +0.0967, h_CO₃²⁻ = +0.1423

**Effect at equilibrium (1M CsHCO₃):**
- H_eff(pure) = 18.24 mol/(m³·atm)
- H_eff(Sechenov) = 13.83 mol/(m³·atm)
- **24.2% reduction** in CO₂ solubility

At elevated pH (CL interior), the reduction is larger due to higher OH⁻ and CO₃²⁻.

### 5.2 Stage 4 Results

| V_RHE [V] | i_CO (S3) | i_CO (S4) | i_H₂ (S4) | FE_CO |
|-----------|-----------|-----------|-----------|-------|
| −0.80 | 56 | **41** | 40 | 50.6% |
| −1.00 | 320 | **192** | 40 | 82.8% |
| −1.20 | 1474 | **435** | 45 | 90.6% |
| −1.50 | 2154 | **702** | 82 | 89.6% |
| −2.00 | 2308 | **645** | 346 | 65.1% |
| −2.50 | 2165 | **389** | 1324 | 22.7% |

**The Sechenov correction reduced peak CO current from ~2300 to ~700 mA/cm²** — a 3.3× reduction. This brought the model much closer to experimental data (~300 mA/cm² peak CO).

**Key physics:** The Sechenov correction reduces CO₂ equilibrium concentration in the liquid, which directly reduces the phase transfer driving force and therefore CO₂R rates. The effect is amplified at high pH (CL interior) where ionic strength is highest.

**Human approved Stage 4:** *"Looks good. Please proceed with the parameter fitting."*

---

## 6. Parameter Fitting

### 6.1 Setup

**4 adjustable parameters:**
1. `i0_CO2R` — CO₂R exchange current density [A/m²]
2. `λ_MHC` — MHC reorganization energy [eV]
3. `HER_prefactor` — HER pre-exponential (fitted in log₁₀ space)
4. `α_c_HER` — HER cathodic transfer coefficient

**Objective:** Log-space SSE of i_CO and i_H₂ at 5 experimental voltages (10 data points).

**Method:** Sequential manual sweeps followed by Nelder-Mead refinement via Optim.jl.

### 6.2 Manual Parameter Sweeps

**Step 1 — Sweep i0_CO2R (λ=1.80, HER params fixed):**

| i0_CO2R [A/m²] | SSE |
|-----------------|-----|
| 50 | 0.880 |
| 100 | 0.393 |
| **150** | **0.383** |
| 200 | 0.477 |
| 300 | 0.724 |
| 540 (initial) | 1.255 |

Best: i0 = 150 A/m² (SSE reduced 69%)

**Step 2 — Sweep λ_MHC (i0=150, HER fixed):**

| λ_MHC [eV] | SSE |
|-------------|-----|
| 1.00 | 0.855 |
| 1.40 | 0.421 |
| **1.60** | **0.288** |
| 1.80 | 0.383 |
| 2.00 | 0.952 |

Best: λ = 1.60 eV (SSE reduced further to 0.288)

**Step 3 — Sweep HER parameters:**

HER_prefactor and α_c were already near-optimal. Best values remained at initial: HER_pf = 2.409×10¹³, α_c = 0.0872.

### 6.3 Final Fitted Parameters

| Parameter | Initial | Fitted | Change |
|-----------|---------|--------|--------|
| i0_CO2R [A/m²] | 540 | **150** | −72% |
| λ_MHC [eV] | 1.80 | **1.60** | −11% |
| HER_prefactor | 2.409×10¹³ | 2.409×10¹³ | unchanged |
| α_c_HER | 0.0872 | 0.0872 | unchanged |

**SSE: 1.255 → 0.288** (77% improvement)

### 6.4 Model vs Experiment Comparison

| E_cath (V RHE) | i_CO model | i_CO exp | Ratio | i_H₂ model | i_H₂ exp | Ratio |
|----------------|-----------|---------|-------|-----------|---------|-------|
| −0.816 | 44.2 | 35.1 | 1.26 | 40.8 | 29.9 | 1.36 |
| −1.205 | 294.4 | 189.6 | 1.55 | 56.1 | 49.3 | 1.14 |
| −1.644 | 356.8 | 299.1 | 1.19 | 167.9 | 119.6 | 1.40 |
| −2.079 | 263.0 | 204.3 | 1.29 | 513.2 | 449.9 | 1.14 |
| −2.413 | 163.4 | 64.6 | 2.53 | 1127.3 | 850.5 | 1.33 |

**Assessment against guide targets (§17.5):**
- i_CO within 30% at 3 of 5 points ✓ (points 1, 3, 4)
- i_CO within 55% at point 2 (moderate overshoot)
- i_CO 2.5× high at point 5 (model doesn't roll over sharply enough)
- i_H₂ within factor of 1.4 at all points ✓
- FE_CO peak value: 84% (target: 70–90%) ✓
- FE_CO peak location: ~V = −1.2 RHE (target: near −1.2 V) ✓

### 6.5 Interpretation

The **i0_CO2R reduction from 540 → 150 A/m²** is physically significant. The original value (540 A/m²) was fitted for a model WITHOUT gas-phase coupling — CO₂ was supply-limited by liquid diffusion. With phase transfer providing CO₂ to every CL cell, the effective CO₂ availability is much higher, so a lower i0 is needed to match the same experimental currents.

The **HER parameters are unchanged** because HER kinetics are independent of CO₂ transport. The H₂ current depends only on local pH and overpotential, which are similar between Stage 2 and Stage 4.

The **CO rollover at V = −2.4** is too gradual in the model (163 mA/cm² vs 65 mA/cm² experimental). This suggests the gas-phase CO₂ depletion mechanism is not aggressive enough at extreme currents. Possible improvements:
1. Lower k_GL to reduce phase transfer rate
2. Add Sechenov correction dependence on local pH (already implemented but may need refinement)
3. Consider that the experimental CO rollover may involve additional mechanisms (catalyst degradation, flooding) not in the model

---

## 7. Bugs Found in This Session

| # | Bug | Location | Impact | Fix |
|---|-----|----------|--------|-----|
| 1 | **Neumann BC sign error** | assembly.jl, run_stage2.jl | Newton diverges at every attempt to switch to Neumann BC | Change `c_eq - c` to `c - c_eq` |
| 2 | **FSG molar mass units** | gas_transport.jl | Gas diffusivities 31.6× too high, CO₂ supply unlimited | Convert M from kg/mol to g/mol |
| 3 | **PAC pressure scaling** | solver.jl (conceptual) | PAC tangent dominated by p~10⁵, voltage barely advances | Use direct voltage stepping instead |
| 4 | **Newton corrupts u on failure** | run_stage2.jl (Sh ramp) | Failed Newton modifies u in-place, PTC starts from garbage | Backup u before Newton, restore on failure |

### 7.1 Comparison with Session 1 Bugs

The Neumann BC sign error (Bug #1) is **identical** to the bug found in Session 1. This confirms that the implementation guide's formula is genuinely ambiguous — two independent implementations (Session 1 and Session 2, separated by 5 days) both got it wrong in the same way.

The FSG units bug (Bug #2) is **new** to Session 2. Session 1 also implemented FSG but may have used a different formula or caught the units issue earlier.

---

## 8. Build Process Summary

**Timeline:**
- 2026-03-25: Stages 1–2 (including Neumann BC debugging)
- 2026-03-26: Stages 3–4 + parameter fitting

**Files created/modified in this session:**

| File | Status | Purpose |
|------|--------|---------|
| `gas_transport.jl` | New | SM + Darcy + FSG + phase transfer + Sechenov |
| `assembly.jl` | Modified | Added `residual_stage3!` with Sechenov flag |
| `run_stage2.jl` | Modified | Sh ramp + PTC-SER + Neumann sign fix |
| `run_stage3.jl` | New | Stage 3 driver with direct voltage stepping |
| `run_stage4.jl` | New | Stage 4 driver (Sechenov correction) |
| `fit_model.jl` | New | Parameter fitting with Optim.jl |
| `plot_results.py` | Modified | Added `plot_stage3()` and `plot_stage4()` |

**Cached solutions:**

| Stage | Solutions | Voltage range | DOFs |
|-------|-----------|---------------|------|
| 1 | 897 | −0.3 to −2.5 V | 250 |
| 2 | 911 | −0.3 to −2.5 V | 250 |
| 3 | 223 | −0.3 to −2.5 V | 570 |
| 4 | 221 | −0.3 to −2.5 V | 570 |

**What worked well:**
- PTC with SER dt adaptation for the Sh ramp (robust convergence)
- Direct voltage stepping (simple, fast, avoids PAC scaling issues)
- Phase transfer ramp (α_PT: 0→1) for smooth gas-liquid coupling activation
- Manual parameter sweeps before automated optimization (builds intuition)

**What required iteration:**
- Neumann BC sign error (9 attempts before identifying root cause)
- FSG units (caught by comparing CO current to experimental data)
- PAC for 570-DOF system (abandoned in favor of direct stepping)

---

## 9. Key Numbers

| Metric | Value |
|--------|-------|
| Total DOFs | 570 (250 liquid + 320 gas) |
| Peak FE_CO (unfitted) | 90% at V ≈ −1.2 RHE |
| Fitted i0_CO2R | 150 A/m² (was 540) |
| Fitted λ_MHC | 1.60 eV (was 1.80) |
| Fitting SSE | 0.288 (77% reduction from initial) |
| H₂ fit quality | All points within 40% |
| CO fit quality | 4 of 5 points within 55%, 1 point 2.5× high |
| Sechenov H_eff reduction | 24.2% at equilibrium |
| Pressure drop at peak current | ~120 Pa (0.12% of p_total) |
| Gas CO₂ depletion at peak current | x_CO₂: 0.75 → 0.71 |

---

## 10. Post-Session Remediation: D_eff Bruggeman Error

### 10.1 Discovery

**Date:** 2026-03-27

After Session 2 was complete and parameters were fitted, the human researcher compared the Build B1 model (Session 2) against an independent implementation of the same model built by a separate Claude instance (the Build A model, Session 1). Both models were built from the same implementation guide (`CO2R_BULK_MODEL_IMPLEMENTATION_GUIDE.md`) and fitted to the same experimental data (Lees et al. 2024).

**Observation:** The Build B1 model consistently overestimated CO partial current density, while the Build A model slightly underestimated it. Despite both models using identical physical constants, kinetics formulations, buffer chemistry, and gas transport equations, the fitted kinetic parameters diverged significantly:

| Parameter | Build A (Session 1) | Build B1 (Session 2) |
|-----------|--------------------|-----------------------|
| i0_CO2R [A/m²] | 42.26 | 72.44 |
| λ_MHC [eV] | 1.538 | 1.564 |
| HER_prefactor | 9.46×10¹² | 1.26×10¹³ |
| α_c_HER | 0.0924 | 0.0928 |
| SSE (log-space) | 0.130 | 0.070 |

### 10.2 Systematic Comparison

A file-by-file comparison of all model components was conducted using Claude (Opus 4.6, 1M context). The following files were compared side-by-side:

| File | Result |
|------|--------|
| `params.jl` | **One difference found** (see below) |
| `chemistry.jl` | Identical physics (buffer sources, equilibrium solver) |
| `kinetics.jl` | Identical physics (MHC CO₂R, BV HER); different function names only |
| `transport.jl` | Identical (Fick diffusion flux) |
| `gas_transport.jl` | Identical physics (FSG, SM mixture-average, Darcy, Sechenov, phase transfer); different code structure |
| `assembly.jl` | Identical physics; Build B1 uses inline gas assembly, Build A uses separate `gas_residual!` function |
| `mesh.jl` | **One difference found** (see below) |
| `solver.jl` | Same Newton solver with damping and row-scaling |

### 10.3 Root Cause: Effective Liquid Diffusivity

The critical difference was in the computation of effective liquid-phase diffusivities in `params.jl`:

**Build A (correct):**
```julia
# params.jl, line 37
const D_eff = eps_L .* D_w   # D_eff = ε_L × D_w
```

**Build B1 (incorrect):**
```julia
# params.jl, line 33
const D_eff = eps_L^1.5 .* D_w   # D_eff = ε_L^1.5 × D_w  (Bruggeman)
```

With `eps_L = 0.08`:
- Build A: D_eff = 0.08 × D_w = **0.0800 × D_w**
- Build B1: D_eff = 0.08^1.5 × D_w = **0.02263 × D_w**

**Build B1's effective diffusivities were 3.54× too low.**

### 10.4 Origin of the Ambiguity

The implementation guide contains two conflicting specifications for D_eff:

**§4.1 (Liquid Transport), lines 124–127:**
> For the Lees-style bulk model:
> `D_eff,i = eps_CL × D_w,i`
> where `eps_CL = 0.08` is the liquid-filled macro-pore fraction. **Use this value for the simplified bulk model.**

**§14 (Quick-Reference Parameter Table), lines 534–537:**
> For the bulk-only model (no agglomerate), use:
> `D_eff,i = eps_L^1.5 × D_w,i` (Bruggeman for macro-pore liquid)
> with `eps_L = 0.08`.

Session 2 (Build B1) followed the §14 formula (Bruggeman with ε_L = 0.08). Session 1 (Build A) followed the §4.1 formula (linear with eps_CL = 0.08), which is the recommended form explicitly marked as the one to use for the simplified bulk model.

**For the paper:** The implementation guide should be revised to remove this ambiguity. The §14 formula applies only when ε_L represents the total liquid-accessible void fraction (ε_L ≈ 0.38), not the macro-pore fraction (0.08). When using ε_L = 0.08 (macro-pores only), the simple linear scaling `D_eff = ε_L × D_w` is correct because the Bruggeman tortuosity correction is already implicitly absorbed into the choice of ε_L = 0.08 rather than the full liquid volume fraction.

### 10.5 Physical Impact of the Error

The 3.54× reduction in liquid diffusivities had cascading effects:

1. **CO₂(aq) transport:** Lower D_eff created steeper CO₂ depletion gradients within the CL. CO₂ was consumed near the gas-liquid interface (x ≈ 0) faster than it could diffuse to the membrane-side (x = L_CL).

2. **OH⁻ removal:** Lower D_eff for OH⁻ impeded removal of electrochemically produced OH⁻, causing higher local pH and stronger Sechenov suppression of CO₂ solubility.

3. **Parameter compensation:** During fitting, the optimizer compensated for the transport limitation by increasing i0_CO2R (72 vs 42 A/m²) to match experimental CO currents. This created an artificially high intrinsic rate that overdrove CO₂R at moderate overpotentials where transport was not yet limiting, causing CO overestimation in the mid-voltage range.

4. **HER coupling:** The higher local pH (from impeded OH⁻ transport) also shifted HER kinetics, requiring a higher HER_prefactor (1.26×10¹³ vs 9.46×10¹²) to compensate.

### 10.6 The Fix

One line changed in `params.jl`:

```julia
# Before (Bruggeman, incorrect for eps_L = 0.08):
const D_eff = eps_L^1.5 .* D_w

# After (linear, correct for Lees-style bulk model):
const D_eff = eps_L .* D_w
```

### 10.7 Refitting Results

After correcting D_eff, the full parameter fitting was re-run (Nelder-Mead, 100 iterations, ~6 minutes runtime). The refitted parameters now closely match the Build A model:

| Parameter | Build B1 (refitted) | Build A | Build B1 (original) |
|-----------|---------------------|--------|---------------------|
| i0_CO2R [A/m²] | **42.85** | 42.26 | 72.44 |
| λ_MHC [eV] | **1.546** | 1.538 | 1.564 |
| HER_prefactor | **8.17×10¹²** | 9.46×10¹² | 1.26×10¹³ |
| α_c_HER | **0.0955** | 0.0924 | 0.0928 |
| SSE | **0.128** | 0.130 | 0.070 |

**Model vs experiment after refit:**

| E_cath (V RHE) | i_CO model [mA/cm²] | i_CO exp | Ratio | i_H₂ model | i_H₂ exp | Ratio |
|----------------|---------------------|----------|-------|------------|----------|-------|
| −0.816 | 34.4 | 35.1 | 0.98 | 26.1 | 29.9 | 0.87 |
| −1.205 | 188.8 | 189.6 | 1.00 | 49.3 | 49.3 | 1.00 |
| −1.644 | 204.9 | 299.1 | 0.69 | 155.1 | 119.6 | 1.30 |
| −2.079 | 168.0 | 204.3 | 0.82 | 448.4 | 449.9 | 1.00 |
| −2.413 | 120.8 | 64.6 | 1.87 | 978.6 | 850.5 | 1.15 |

The remaining small differences between the Build B1 (refitted) and Build A models are attributable to:
1. **GDL mesh grading:** Build A uses a geometrically graded GDL mesh (q = 1.3, finer near CL), while Build B1 uses a uniform GDL mesh. This affects accuracy of gas-phase gradients at the GDL–CL interface.
2. **Optimizer convergence:** Different starting trajectories in the Nelder-Mead optimization lead to slightly different local minima.

### 10.8 Secondary Difference: GDL Mesh

| Model | GDL Mesh | Implementation |
|-------|----------|----------------|
| Build A | Graded (q = 1.3) | `make_gdl_mesh(N_GDL, L_GDL; q=1.3)` with geometric ratio |
| Build B1 | Uniform | `make_gdl_mesh(N, L)` with equal cell widths |

Both use N_GDL = 30, N_CL = 50. The graded mesh provides better resolution near the CL interface where gas-phase concentration gradients are steepest. This is a minor effect compared to the D_eff error but contributes to the residual parameter differences.

### 10.9 Lessons Learned

1. **Implementation guide ambiguity:** This is the third bug traceable to ambiguous specification in the guide (after the Neumann BC sign error in Sessions 1 and 2, and the FSG molar mass units in Session 2). A guide revision should:
   - Remove the conflicting D_eff formula in §14
   - Explicitly state which formula applies to the simplified bulk model vs the flooded-agglomerate model
   - Add a dimensional-analysis check: for eps_L = 0.08, D_eff(CO₂) should be ~2.7×10⁻¹⁰ m²/s

2. **Cross-validation between independent builds:** The error was only discovered by comparing two independent implementations. A single build would not have detected it, because the optimizer compensates for the transport error by adjusting kinetic parameters — the model still fits the data (SSE = 0.070), just with physically inconsistent parameters.

3. **Parameter reasonableness checks:** The original Build B1 i0_CO2R = 72 A/m² was not obviously wrong in isolation. But comparing it against the Build A value (42 A/m²) — fitted with the same guide, same data, same kinetic model — immediately flagged the discrepancy.

---

## 11. Updated Key Numbers (After Remediation)

| Metric | Original (Session 2) | Refitted |
|--------|---------------------|----------|
| D_eff formula | ε_L^1.5 × D_w | ε_L × D_w |
| D_eff(CO₂) [m²/s] | 7.67×10⁻¹¹ | 2.71×10⁻¹⁰ |
| Fitted i0_CO2R [A/m²] | 72.44 → 150 (manual) | **42.85** |
| Fitted λ_MHC [eV] | 1.564 | **1.546** |
| Fitted HER_prefactor | 1.26×10¹³ | **8.17×10¹²** |
| Fitted α_c_HER | 0.0928 | **0.0955** |
| Fitting SSE | 0.070 | **0.128** |
| CO fit at V = −1.2 | 1.55× exp (overestimate) | **1.00× exp** |
| CO fit at V = −1.6 | 1.19× exp | **0.69× exp** |
| H₂ fit quality | All within 40% | **All within 30%** |

**Note on SSE:** The original model's lower SSE (0.070 vs 0.128) does not indicate a better model — it reflects the optimizer having more freedom to compensate for incorrect transport physics by overfitting the kinetic parameters. The refitted model has physically consistent parameters that agree with the independent Build A implementation.

---

*Session 2 remediation complete. 2026-03-27. Cross-validated against independent Session 1 (Build A) build.*
