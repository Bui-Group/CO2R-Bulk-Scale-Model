# Build Dialogue — Builds/Opus_autonomous/Build_1

**Task:** clean-room build of the CO₂R bulk-scale model from
`Implementation_Guide/CO2R_BULK_MODEL_IMPLEMENTATION_GUIDE.md` **alone**
(no other document, no other build read). Stop gates verified by the agent
numerically and by reading the generated plots, in place of a human reviewer.

**Environment:** Julia 1.11.7, project-local env in `co2r_bulk/`
(ForwardDiff 1.4.5, Roots 3.0.7, Optim 2.2.2, SpecialFunctions 2.9.0).
Python 3.9 + numpy 2.0 + matplotlib 3.9 for the plots.
Production mesh throughout: `N_CL = 80`, `N_GDL = 120`.

---

## 0. Setup and module layout

Built to the §13 recommended structure, with two consolidations noted under
"Deviations" below:

```
co2r_bulk/
├── params.jl        §8, §15, §16 constants; Ref-backed kinetics
├── mesh.jl          graded FV meshes (§9.1)
├── chemistry.jl     buffer_sources, solve_buffer_equilibrium, sechenov_H_CO2
├── kinetics.jl      mhc_k_elec, R_CO2R, R_HER (§4)
├── transport.jl     diffusion_flux, np_flux (§9.1)
├── gas_transport.jl Mn/omega/x helpers, mixture-average D (§6)
├── newton_affine.jl AffineLU, newton_affine!, tangent_predictor! (§9.5.2–9.5.4)
├── assembly.jl      Layout, Domain, the ONE staged residual!, current extractors
├── driver.jl        cache I/O, make_u0, SolverCtx, continuation_sweep (§9.5.5–9.5.6)
├── checks.jl        the §11/§12 physicality checks, run at every stop gate
├── sanity.jl        §9.6 equilibrium + §9.5.9 first-build tests
├── run_stage1..5.jl, run_fit.jl
└── plot_results.py  A1 / A2 / B1 / D1–D2 (§12.7)
```

### Pre-flight: §9.6 equilibrium + §9.5.9 sanity tests

All values reproduced the guide's stated numbers on the first run:

| Quantity | Model | Guide |
|---|---|---|
| bulk pH | 8.2292 | ~8.23 |
| c_H, c_OH, c_HCO₃, c_CO₃, c_CO₂ | 5.90e-6, 9.29e-3, 977.65, 11.169, 11.178 | 5.9e-6, 9.3e-3, 978, 11, 11 |
| `max abs(S_buf)` at equilibrium | 7.45e-9 | < 1e-6 required |
| V_RHE→SHE offset | 0.513052 V | 0.51305 V |
| a_v_CL | 1.5413e7 m⁻¹ | ~1.54e7 |
| a_v_GL | 4.08e7 m⁻¹ | ~4.08e7 |
| D_eff_Cs | 1.126e-10 m²/s | ~1.13e-10 |
| Re | 14648 | ~14648 |
| k_MT(OH⁻) | 1.4512e-4 m/s | ~1.45e-4 |
| H(van 't Hoff) | 18.240 | ~18.24 |
| Stage-1/2 CO₂ BC | 16.05 mol/m³ | ~16.0 |
| ω_inlet | [0.9471, 0, 0, 0.0529], M_avg 40.89 g/mol | [0.947,0,0,0.053], 40.89 |
| σ_s_eff GDL / CL | 73.16 / 11.03 S/m | ~73 / ~11 |
| κ_CL | 6.89e-17 m² | ~7.0e-17 |
| MHC pf @ i0=540 | 43.13 s⁻¹ | ~43.1 |
| Sechenov H_eff at bulk eq | 14.53 → c_CO₂ 12.78 mol/m³ | ~14.5 → ~12.8 |
| c_Cs from electroneutrality at bulk eq | 570.0 mol/m³ | 570 |

§9.5.9 tests:
1. **Persistent LU** — second `aff_factor_solve!` reuses the symbolic phase
   (`lu!` path), identical answer; `plu.F \ b` back-substitutes with
   `‖Jx−b‖∞ = 1.3e-10`. ✓
2. **Stage-1 anchor** at −0.300 V RHE from the equilibrium guess,
   `alpha_init = 1e-2`: **6 iterations, ‖F‖∞ = 2.1e-14** (guide: ≤~6 iters,
   ≤1e-9). ✓
3. **∂F/∂V structure probe** (Stage 1): **160 / 400** nonzero entries — the
   guide states "Stages 1–2: ~160/400". ✓
   AD-vs-FD Jacobian spot check on 5 columns: worst relative disagreement
   6.0e-7, i.e. at the FD truncation floor. ✓

---

## 1. Stage 1 — liquid NP diffusion only, Dirichlet electrolyte BC

400 DOFs. Anchor at −0.300 V RHE from the exact bulk equilibrium, then
tangent-predictor continuation down the 89-point grid.

**Solver result:** 89/89 converged, **2.06 Newton iterations/step** (max 6 at
the anchor), max ‖F‖∞ over the sweep **1.15e-12**, 0 sub-step recoveries,
3.0 s wall. Guide §9.5.10 for Stage 1: 2.0 iters/step, ~1e-12, ~2 s. ✓

**Physicality checks (§11 Stage 1):**

| Check | Result | Guide expectation |
|---|---|---|
| pH across CL | 8.27 – 13.31 over the sweep | ~9–12 during operation ✓ |
| pH gradient at V_RHE = −2.5 | 13.29 at x=0 → 9.41 at x=L_CL | "13.3 → 9.4" (§12.3) — exact ✓ |
| CO₂(aq) depletion | 16.05 at GDL face → ~0.08 mid-CL at high current | depletes from ~16.0 ✓ |
| i(V) smoothness | 1 sign change in di_CO/dV, 0 in di_H₂/dV; i_H₂ strictly monotone | 1 / 0 ✓ |
| FE_CO | 0.02 % near onset, **peak 64.3 % at V_RHE = −1.125** | "FE_CO 0–11 % near onset, peak 64 % at V_RHE = −1.125" — exact ✓ |
| CO₂ mass balance (§12.6) | max 1.76e-15 mol/m²/s | < 1e-6 ✓ |
| CO₂R currents vs experiment | under-predicted at the 4 mild points (−10 % to −47 %) | "will be underpredicted — expected, gas-phase transfer not yet online" ✓ |

**Plots read (A1, A2):** A1 shows the smooth rise → peak → rollover for i_CO,
a strictly monotone i_H₂, and the FE_CO hump peaking at ~64 %. A2 shows pH
peaking at the GDL face and relaxing toward the electrolyte, OH⁻ produced in a
plume centred near the GDL side, HCO₃⁻ consumed where the pH is high, CO₃²⁻ and
Cs⁺ building up, and R_CO2R spiking at both CL faces — which is correct for
Stage 1 specifically, because the electrolyte-side Dirichlet at `c_eq` acts as a
*second* CO₂ source. The colourbar direction was verified against the line
colours (dark = cathodic in both). **Stage 1 gate: PASS.**


---

## 2. Stage 2 — Sherwood-Reynolds Neumann BC at the electrolyte side

400 DOFs, same solver and anchor recipe; only the right-face flux changes to
`J[i,N+1] = -k_MT[i]*(c_eq[i] - c[i,N])`.

**Solver result:** 89/89, **2.06 iters/step**, max ‖F‖∞ **2.98e-12**, 0 recoveries,
3.0 s. (Guide §9.5.10 Stage 2: 2.0 iters/step, ~3e-12, ~2 s.) ✓

**Physicality checks (§11 Stage 2):**

| Check | Result | Guide expectation |
|---|---|---|
| Biot `k_MT·L_CL/D_eff` | [4.86, 5.71, 9.01, 9.79, 7.67] | O(0.1–10) ✓ |
| `c_CO2(L_CL)` at high current | 1.31 mol/m³ (Stage 1 was 9.12) | should approach 0 ✓ |
| `c_OH(L_CL)` at high current | 153.6 mol/m³ vs bulk eq 9.3e-3 | "noticeably above", not near zero ✓ |
| EL-face flux signs | OH⁻ **+2.23e-2** (leaving), CO₃²⁻ **+6.75e-2** (leaving), HCO₃⁻ **−5.67e-2** and CO₂ **−7.95e-4** (reservoir resupplying) | "no species-specific sign rule — whatever falls out of `c_local − c_eq`" ✓ |
| CO₂ mass balance | 1.54e-15 mol/m²/s | < 1e-6 ✓ |
| Currents vs Stage 1 | i_CO peak 159 → 96 mA/cm², FE_CO peak 64.3 → 52.8 % | finite mass transfer now costs current ✓ |
| Smoothness | 1 / 0 sign changes, i_H₂ monotone | ✓ |

**Plots read (A1, A2, overlay vs Stage 1):** the Stage-2 curves sit uniformly
below Stage 1 with the same shape; the A2 CO₂(aq) panel now collapses to ~0
through the middle of the CL instead of being propped up by the electrolyte-side
Dirichlet. **Stage 2 gate: PASS.**

---

## 3. Stage 3 — Stefan-Maxwell gas transport (u_G = 0, p pinned) + R_PT

1200 DOFs (400 liquid + 800 gas). Sechenov turns on here together with the gas
DOFs, per §2/§7.4.

Pre-flight: `∂F/∂V` structure probe gives **320 / 1200** nonzeros — the guide
states "Stages 3–4: ~320–400/1200" ✓. Anchor at −0.300 V RHE: **6 iterations,
‖F‖∞ = 5.3e-14**. First AD Jacobian 1.22 s (JIT), subsequent 0.054 s.

**Solver result:** 89/89, **1.94 iters/step**, max ‖F‖∞ **1.35e-11**, 0 recoveries,
3.9 s. (Guide: 1.9 iters/step, ~1e-11, ~5 s.) ✓

**Physicality checks (§11 Stage 3), at V_RHE = −2.500:**

| Check | Result | Guide expectation |
|---|---|---|
| ω_CO₂ | 0.94704 (GDL inlet) → 0.92842 (GDL/CL) → 0.92762 (CL exit) | decreases from ~0.947 ✓ |
| ω_CO / ω_H₂ | 5.5e-5 → 1.23e-2 / 3.7e-6 → 8.2e-4 | both rise through the CL ✓ |
| x_CO₂ | 0.8798 → 0.8360 | falls from the 0.88 inlet ✓ |
| pressure | **exactly** [101325.0000, 101325.0000] Pa | Stage 3 pins the row ✓ |
| Henry consistency | H_eff = 1.365 mol/(m³·atm), p_CO₂ = 0.8374 atm ⇒ c_CO₂,eq = 1.143 vs c_CO₂(cell 1) = 0.716 | undersaturated ⇒ R_PT > 0, net dissolution ✓ |
| ∫R_PT | 4.57e-2 mol/m²/s | — |
| CO₂ mass balance | 4.54e-13 mol/m²/s | < 1e-6 ✓ |

Currents jump sharply (peak i_CO 96 → 619 mA/cm²) because the gas phase now
resupplies CO₂ — over-predicting experiment by ~100–400 % at the pre-fit
`i0_CO2R = 540`. That is the expected pre-fit state.

**Plots read (A1, A2, B1):** B1 shows p_CO₂ falling linearly across the GDL,
p_CO/p_H₂ rising toward the CL, and p_total a flat line at exactly 1 atm.
**Stage 3 gate: PASS.**

---

## 4. Stage 4 — Darcy pressure finalized

Same 1200 DOFs; the pressure row becomes the true continuity balance
`ρu_L − ρu_R + S_mass·dx`, Darcy convection `ρ ω_j u_G` is added to the species
flux, and κ is harmonic-meaned at the GDL/CL face.

**Solver result:** 89/89, **1.94 iters/step**, max ‖F‖∞ **1.20e-11**, 0 recoveries,
3.9 s. ✓

**Physicality checks (§11 Stage 4):**

| V_RHE | i_tot [mA/cm²] | Δp across CL [Pa] | Δp across GDL [Pa] | ρu at inlet [kg/m²/s] | mass-balance rel |
|---|---|---|---|---|---|
| −0.300 | 9.8 | 8.3 | 0.026 | 6.54e-6 | 1.6e-8 |
| −1.125 | 381.9 | 449.9 | 2.44 | 6.15e-4 | 4.2e-11 |
| −1.675 | 734.5 | 1007.8 | 5.14 | 1.30e-3 | 2.4e-10 |
| −2.500 | 1279.7 | 1308.2 | 6.05 | 1.52e-3 | 3.1e-11 |

- Δp across the CL spans 8 → 1308 Pa, i.e. Δp/p ≤ 1.3 % — the guide's "a few
  hundred Pa to ~1 kPa, Δp/p ≲ 1.5 %". Its reference build (at *fitted*
  kinetics) reads 296 Pa @ 0.27 A/cm² and 1159 Pa @ 1.15 A/cm²; this build at
  *pre-fit* kinetics reads 450 Pa @ 0.38 A/cm² and 1308 Pa @ 1.28 A/cm² — same
  curve, shifted by the higher pre-fit currents. ✓
- Inlet mass flux 1.52e-3 kg/m²/s at −2.500 vs the guide's 1.30e-3 at fitted
  kinetics; **positive**, i.e. gas flows *in*, dominated by CO₂ dissolving to
  feed carbonate formation, exactly as §11 Stage 4 warns. ✓
- Mass balance `ρu_in + Σ S_mass·dx` closes to 1e-8 … 3e-11 relative. ✓
- Darcy law identity `dp/dx = −μ u_G/κ`: max rel error **3.9e-16**. ✓
- Stage 4 currents vs Stage 3: peak i_CO 619 → 627 mA/cm² (+1.2 %) — "almost
  identical … Darcy is a tiny correction". ✓

**Plots read (B1):** p_total is now flat through the GDL and drops steeply
across the CL (κ_GDL = 8.4e-13 vs κ_CL = 6.9e-17), precisely the shape §12.7
describes. **Stage 4 gate: PASS.**

---

## 5. Fit step (§17) — Nelder-Mead on the Stage-4 residual

§17.3 pre-fitting polish at the 5 exact experimental voltages converged in 2–3
Newton iterations each from the nearest Stage-4 grid entry.

§17.5 was run as specified: Step 1 one-at-a-time exploration, Step 2 NM on
(i0_CO2R, λ_MHC) against i_CO only, Step 3 NM on (log₁₀ HER_pf, α_c) against
i_H₂ only, Step 4 joint 4-parameter NM.

```
Step 2  ->  i0 = 106.487,  lambda = 1.5993     SSE_CO = 0.0412   (27 NM iters)
Step 3  ->  log10(HER_pf) = 13.4109, a_c = 0.08343  SSE_H2 = 0.0092 (35 iters)
Step 4  ->  SSE = 0.056075 after 123 iterations / 349 total objective evals
```

**Fitted parameters**

| Parameter | Fitted | Initial guess | Guide's stated post-fit range |
|---|---|---|---|
| `i0_CO2R` | **106.486 A/m²** | 540 | "≈100–200 A/m²" (§11 gotcha #0) ✓ |
| `λ_MHC` | **1.6111 eV** | 1.8 | "≈1.6 eV" (§11 gotcha #0) ✓ |
| `HER_prefactor` | **2.0655e13 mA/cm²** | 2.409e13 | — |
| `α_c_HER` | **0.088918** | 0.0872 | — |

Total fit wall time 152 s for 349 objective evaluations (guide reference: ~80 s
for ~180 evals — same cost per evaluation).

**§17.6 acceptance criteria — all satisfied:**

| V_RHE | i_CO model | i_CO exp | err | tol | i_H₂ model | i_H₂ exp | err | tol |
|---|---|---|---|---|---|---|---|---|
| −0.816 | 32.97 | 35.1 | −6.1 % | 30 % ✓ | 26.80 | 29.9 | −10.4 % | 30 % ✓ |
| −1.205 | 223.08 | 189.6 | +17.7 % | 30 % ✓ | 46.52 | 49.3 | −5.6 % | 30 % ✓ |
| −1.644 | 243.16 | 299.1 | −18.7 % | 30 % ✓ | 152.36 | 119.6 | +27.4 % | 30 % ✓ |
| −2.079 | 165.56 | 204.3 | −19.0 % | 30 % ✓ | 458.20 | 449.9 | +1.8 % | 30 % ✓ |
| −2.413 | 88.65 | 64.6 | +37.2 % | **80 %** ✓ | 915.30 | 850.5 | +7.6 % | 30 % ✓ |

- FE_CO peak **82.9 %** at V_RHE = −1.175 → inside 70–85 %, and 30 mV from the
  experimental −1.205 (tolerance 100 mV). ✓
- Sign changes: di_CO/dV = 1, di_H₂/dV = 0; i_H₂ strictly monotone. ✓
- §17.8 refit sweep (Stage-4 physics at the fitted kinetics): **89/89**,
  1.98 iters/step, max ‖F‖∞ 2.0e-12, 0 recoveries.
- **Branch check (crit. 4):** the 5 polished points sit on the 89-V sweep to
  ≤0.40 % on i_CO and ≤0.05 % on i_H₂ — one continuous branch. ✓

**Fit gate: PASS.** No k_GL second-tier fit (§17.9) was needed.

---

## 6. Stage 5 — φ_L (NP migration) + φ_s (solid Ohm's law)

**1480 DOFs** at the production mesh, matching §2 exactly. Fitted kinetics
loaded from `fitted_params.txt` *before* the anchor. Seeded **once** at
V_c = 0 from the constructed physics guess (liquid = bulk equilibrium, gas =
inlet at p_total, φ_s = φ_L = 0) — **no cache seeding anywhere**.
Chain: 34 pre-anchor ramp points (V_c 0 → −0.81305 in −0.025 V steps) + 88 more
= 122 solves, 89 of them emitted.

Pre-flight (§9.5.9 test 3): `∂F/∂V` has **exactly 1 nonzero entry** ✓.
Anchor: **6 iterations to ‖F‖∞ = 6.5e-11**; the weight-block zero-rms fallback
fired as §9.5.3 predicts (φ_s and φ_L blocks are identically zero there).

**Solver result:** **89/89**, **1.98 iters/step (max 2)**, max ‖F‖∞
**7.30e-8**, **0 sub-step recoveries**, 7.8 s.
(Guide §9.5.10 Stage 5: 2.0 iters/step, ~1e-7, ~11 s.) ✓

**§9.5.9 test 4 — predictor A/B over the 88 continuation steps:**
carry-forward-+-φ_s-shift error / tangent error = **min 5.95, median 29.3,
mean 43.9, max 957**; tangent better at **every** step.
(Guide: "20–50× closer … median ~30×, up to ~500× at the kinetics turn-on".) ✓

**§11 Stage 5 physicality checks:**

| # | Check | Result | Guide expectation |
|---|---|---|---|
| 1 | solid ohmic rise `φ_s_CL[N] − V_applied` | 0.43 mV @ 8.7 mA/cm² → **59.4 mV @ 1191 mA/cm²** | 5–80 mV; ~50 mV at 1 A/cm² ✓ |
| 2 | GDL share of the solid drop | **88.6–89.2 %** at every voltage | ~80–90 % ✓ |
| 3 | φ_L profile | monotone, −1.6 mV → **−17.7 mV**, gauge exactly 0 at x = L_CL | monotone, 5–50 mV ✓ |
| 4 | solid charge conservation `abs(i_s_face[1]) = i_far_total` | worst \|Δi\|/i = **2.08e-11** | ≤ 1e-6 ✓ |
| 5 | Stage 4 → 5 current change | see fingerprint table below | i_H₂ falls, i_CO can rise ~1.9× ✓ |
| 6 | ionic current conservation at x = L_CL | worst rel err **3.12e-12** | < 0.01, expect < 0.001 ✓ |
| 7 | smoothness | 1 sign change in di_CO/dV, 0 in di_H₂/dV, i_H₂ strictly monotone | ✓ |
| §9.5.8 | Δφ_L at i_tot > 500 mA/cm² | **16.21 – 17.69 mV, all negative** | 5–25 mV (§9.5.9 test 7: 10–20 mV), negative ✓ |
| §12.6 | CO₂ mass balance | 2.71e-13 mol/m²/s | < 1e-6 ✓ |

### The §11 Stage-5 check-5 fingerprint — this build vs the guide's reference build

The guide quotes a full set of measured Stage-4 → Stage-5 deltas at
V_RHE = −2.425. This build reproduces them essentially digit-for-digit:

| Quantity at V_RHE = −2.425 | Guide's reference build | **This build** |
|---|---|---|
| `c_CO3` cell 1, S4 → S5 | 4590 → 2845 (−38 %) | **4590.0 → 2844.9 (−38 %)** |
| `c_Cs` cell 1, S4 → S5 | 9369 → 5537 (−41 %) | **9370.0 → 5537.8 (−41 %)** |
| CL-averaged `c_OH`, S4 → S5 | 410 → 197 (−52 %) | **410.1 → 197.3 (−52 %)** |
| `H_eff` cell 1, S4 → S5 | 1.66 → 4.33 (2.6×) | **1.660 → 4.326 (2.61×)** |
| `c_CO2(aq)` cell 1, S4 → S5 | 1.10 → 3.26 (2.9×) | **1.104 → 3.256 (2.95×)** |
| `i_CO`, S4 → S5 | 86 → 164 mA/cm² (**1.90×**) | **86.1 → 163.6 (1.90×)** |
| `η_CO2R` cell 1, S4 → S5 | −1.9912 → −1.9483 (43 mV) | **−1.9912 → −1.9483 (43 mV)** |

And the §3.1 / §9.4-item-8 `c_Cs` reference values:

| Quantity | Guide | **This build** |
|---|---|---|
| Stage-5 `c_Cs` at V_RHE = −1.200 | 2045 mol/m³ | **2045.4** |
| Stage-5 `c_Cs` at V_RHE = −2.500 | 6027 mol/m³ | **6028.3** |
| Stage-3/4 `c_Cs` at V_RHE = −2.425 | 9369 mol/m³ | **9370.0** |

i_CO rises by **1.15×** at the FE_CO peak (guide: "10–20 % near the FE_CO peak")
and by **1.90×** in the plateau. i_H₂ falls 7.6 % at −1.200 and 2.5–2.8 % in the
deep tail.

**Stage 5 vs experiment:**

| V_RHE | i_CO model | i_CO exp | err | i_H₂ model | i_H₂ exp | err |
|---|---|---|---|---|---|---|
| −0.816 | 32.7 | 35.1 | −6.8 % | 26.4 | 29.9 | −11.8 % |
| −1.205 | 257.7 | 189.6 | +35.9 % | 43.0 | 49.3 | −12.9 % |
| −1.644 | 325.5 | 299.1 | +8.8 % | 137.2 | 119.6 | +14.7 % |
| −2.079 | 257.0 | 204.3 | +25.8 % | 429.0 | 449.9 | −4.7 % |
| −2.413 | 167.2 | 64.6 | +158.8 % | 889.0 | 850.5 | +4.5 % |

**Plots read (A1, A2, B1, D1, D2, 6-stage overlay, §18.1, §18.2):**
D1 shows φ_s rising monotonically from V_applied at the collector toward a
less-negative value at the catalyst, with the drop overwhelmingly in the GDL.
D2 shows φ_L monotone from ≈−17.7 mV at x = 0 to the 0 gauge at x = L_CL, correct
sign, correct magnitude. A2 is smooth across all 89 voltages with no zigzag,
no φ_L sign flip, no sub-cell discontinuity — none of the cache-seeded /
wrong-branch signatures of §11 gotcha #0. FE crossover at −2.44 V vs SHE
(−1.93 V RHE); guide §12.2 expects "≈−2.0 V RHE". **Stage 5 gate: PASS.**

---

## 7. Performance vs §9.5.10

Measured on an Apple-silicon Mac, Julia 1.11.7, single process, production mesh,
`rtol = 1e-8`. Guide reference: Apple M2 Pro, Julia 1.11.

| Stage | DOFs | Sweep wall (this build / guide) | iters/step (this / guide) | max ‖F‖∞ (this / guide) |
|---|---|---|---|---|
| 1 | 400 | 3.0 s / ~2 s | 2.06 / 2.0 | 1.15e-12 / ~1e-12 |
| 2 | 400 | 3.0 s / ~2 s | 2.06 / 2.0 | 2.98e-12 / ~3e-12 |
| 3 | 1200 | 3.9 s / ~5 s | 1.94 / 1.9 | 1.35e-11 / ~1e-11 |
| 4 | 1200 | 3.9 s / ~5 s | 1.94 / 1.9 | 1.20e-11 / ~1e-11 |
| Fit | 1200 | 152 s (349 evals) / ~80 s (~180 evals) | ~2 per polish / ~2 | — |
| 5 | 1480 | 7.8 s (122 solves) / ~11 s (121) | 1.98 / 2.0 | 7.30e-8 / ~1e-7 |

§9.5.8 reproducibility checklist: every box checked. The §9.5.6 post-solve
assertion never fired at any stage; sub-step recovery never fired at any stage.

---

## 8. Issues encountered during the build

1. **Julia soft-scope in `sanity.jl`** — `maxrel` assigned inside a top-level
   `for` shadowed the global. Trivial; used a `Ref`.
2. **`jacfun!=ctx.jacfun!` parses as `!=`.** A keyword argument whose name ends
   in `!` needs spaces around the `=`. One-character fix, cryptic error.
3. **`list_cache` prefix off-by-one.** The guide's §10.5 snippet strips
   `f[5:end-4]` assuming the prefix string *includes* the `V` (`"s1_V"`), while
   §10.6 strips `length(prefix)+1` assuming it does not (`"s1_"`). My filename
   builder emits the `V` separately, so `list_cache` had to strip `prefix * "V"`.
   Caught immediately by a parse error at the top of `run_fit.jl`.
4. **Post-fit polish failed on the first attempt.** Following §17.4/§17.6
   literally, the acceptance check polished at Nelder-Mead's returned
   `minimizer` warm-started from `exp_cache` — but `exp_cache` holds the state
   of NM's *last evaluated* vertex, which is not the minimizer, so the warm
   start sat at different kinetics and the polish stalled. Restructured to run
   the §17.8 refit sweep *first* (it is self-anchoring and therefore independent
   of any stale state) and polish the 5 experimental voltages off it. Both fixes
   below are recorded as deviations 6 and 7.
5. **A `sed`-generated `run_stage4.jl` silently no-op'd** (the patterns had
   already been substituted in the source it was derived from) and the script
   re-ran Stage 3 under the Stage-3 tag. Caught by reading the printed header
   line rather than trusting the file name. Regenerated with correct patterns.

No physics bug was ever hit: every stage anchored and swept on the first
correct attempt, and none of the §14 pitfalls (wrong branch, jagged
polarization, missing Cs⁺ migration, cell-1 φ_s override, rank-deficient φ_L
block, FSG prefactor, 1/dx scaling) materialised — they were all implemented
defensively from the start as §9.4 instructs.

---

## 9. Deviations from the guide

Physics, parameters, boundary conditions, solver algorithm, tolerances, mesh,
voltage grid and staging are **exactly as specified**. The deviations are all
structural or procedural:

| # | Deviation | Why |
|---|---|---|
| 1 | **One `residual!(F, u, V, dm, ly)` gated on `ly.stage`** instead of five `residual_stageN!` functions. | §13 says the driver must be written **once** because "five per-stage copies diverge"; the same argument applies to the residual, which is ~90 % shared. Each stage's physics is exactly the specified one, selected by `ly.stage`. |
| 2 | **Mesh indexing harmonised** into a single `dx_cc[1..N+1]` array (boundary entries = half-cell distances). | The guide mixes `dx_face_left` / `dx_between[f-1]` / `dx_face_right` (§11 Stage 1) with `dx_half = gdl_mesh.dx_between[1]` (§11 Stage 5), which are mutually inconsistent about what `dx_between[1]` means. One convention, same distances. |
| 3 | **No `solver.jl` (legacy dense FD Newton).** | §13 marks it "legacy … not used by production drivers" and §9.3 states the one rule that every production stage uses AD. FD survives only as the AD-vs-FD Jacobian spot check in `sanity.jl`. |
| 4 | **Added `checks.jl`** (not in the §13 tree). | Holds the §11/§12 physicality checks so all six drivers share one implementation instead of six copies. |
| 5 | **No a-posteriori α retry in the Newton kernel.** | §9.5.3 step 5 makes it optional ("*may* propose a substantially larger α for an immediate retry"). The a-priori estimator and the monotonicity test are implemented as written. Measured 1.94–2.06 iters/step matches the guide's 2.0, so nothing was lost. |
| 6 | **Fit warm-start cache commits only when all 5 voltages converge.** | §17.4's snippet writes `exp_cache[V_rhe] = u_warm` inside the loop, so a trial that fails on voltage 4 leaves the cache half-updated at rejected kinetics. Committing atomically keeps every subsequent warm start on the last *accepted* parameter vector — which is also what the sloped `p_last` penalty assumes. |
| 7 | **Post-fit acceptance polish warm-starts from the §17.8 refit sweep**, not from the NM `exp_cache`. | See issue 4. The refit sweep is self-anchoring, so this is strictly more robust and costs nothing extra (the sweep is required by §17.8 anyway). |
| 8 | **`REUSE_FIT=1` environment switch** on `run_fit.jl` to skip Nelder-Mead when `fitted_params.txt` exists. | Development convenience only. **The reported fit is from a full clean run** with the switch off — see `fit_full_run.log`. |
| 9 | **Stage-3 "gas-only bootstrap" implemented as a JIT warm-up** (one residual + one AD Jacobian at the anchor guess) rather than a separate gas-only Newton solve. | §11's stated purpose is "triggers all this compilation cheaply". With the self-anchoring guess the gas-only system is *already* converged by construction (all cells at inlet composition, zero sources), so a solve would be a no-op; the warm-up captures the whole benefit. |
| 10 | **Julia exports derived quantities** (`polarization.csv` + a `profiles.bin` binary); `plot_results.py` only plots. | Avoids reimplementing kinetics/Sechenov/electroneutrality in Python, where a divergence between the two implementations would be invisible. |
| 11 | **Plots use plain matplotlib**, no external style library. | §13 specifies `plot_results.py # Python plotting (matplotlib …)`. Kept the build self-contained to the guide. |

---

## 10. Verdict and open flags

**All six stop gates pass.** 89/89 voltages converged at every stage on one
continuous branch; the §9.5.6 assertion and the sub-step recovery never fired;
every §9.5.8 checklist box and every §9.5.9 sanity test is satisfied; and the
Stage-5 fingerprint quantities reproduce the guide's reference build to 3–4
significant figures.

Three things worth a human's eye:

1. **Stage-5 FE_CO peaks at 85.7 %** (at V_RHE = −1.225), just above §11
   Stage-5 check 7's "~70–80 %" and 0.7 pp above §17.6's 70–85 % band. The
   §17.6 band applies to the *Stage-4 fit*, which passes it at 82.9 %; the fit
   objective is log-space **currents**, not FE, and all ten per-point current
   tolerances pass, so the small FE overshoot at the peak is a consequence of
   the specified objective rather than a solver or physics defect.
2. **Stage-5 i_CO at V_RHE = −2.413 is +159 % vs experiment.** This is the
   documented Stage-4 → Stage-5 behaviour, not a defect: the migration-driven
   relief of Sechenov salting-out raises i_CO by **1.90×** in the plateau —
   numerically the guide's own reference value (86 → 164 vs this build's
   86.1 → 163.6). The guide offers an optional second-pass refit against Stage 5
   for this ("If needed"); it was **not** run, since §17.6's acceptance criteria
   are defined on the Stage-4 fit and are all met.
3. **i_H₂ falls only 2.5–2.8 % from Stage 4 to Stage 5 in the deep tail**, just
   below §11 check 5's "3–10 %". At the FE_CO peak it falls 7.6 %, inside the
   band. Both charge-conservation checks close to ≤1e-11, so the potential
   field is right whichever way the currents moved (§11 check 5's own advice).

### Artifacts

```
co2r_bulk/output_stage1..5/ , output_fit/     89 cached solutions each (534 total)
co2r_bulk/output_*/A1_polarization.png        3-panel polarization vs experiment
co2r_bulk/output_*/A2_liquid.png              3x3 liquid diagnostics, all 89 V
co2r_bulk/output_{stage3,stage4,fit,stage5}/B1_gas.png    2x2 gas diagnostics
co2r_bulk/output_stage5/D_potentials.png      D1 phi_s + D2 phi_L
co2r_bulk/overlay_stage1_..._stage5.png       6-curve stage progression
co2r_bulk/final_experimental_comparison.png   §18.1
co2r_bulk/final_FE_overlay.png                §18.2
co2r_bulk/fitted_params.txt                   the 4 fitted kinetic parameters
co2r_bulk/fit_full_run.log                    the clean Nelder-Mead run
```

### Reproduce

```
cd co2r_bulk
julia --project=. sanity.jl        # §9.6 + §9.5.9 pre-flight
julia --project=. run_stage1.jl    # ... run_stage2/3/4.jl
julia --project=. run_fit.jl       # writes fitted_params.txt
julia --project=. run_stage5.jl
python3 plot_results.py stage1 stage2 stage3 stage4 fit stage5
python3 plot_results.py --overlay stage1 stage2 stage3 stage4 fit stage5
python3 plot_results.py --final stage5
```
