# Build Dialogue — Opus_autonomous / Build_2

**Task.** Clean-room implementation of the CO₂R bulk-scale 1-D model from
`Implementation_Guide/CO2R_BULK_MODEL_IMPLEMENTATION_GUIDE.md` (v7.1)
and *nothing else*. No other document in the repository was read. At each of the
guide's stop gates the human review was replaced by self-verification: numeric
checks against every quantitative claim the guide makes, plus reading the
generated figures back.

**Environment.** macOS, Julia 1.11.7, Python 3.9.6 / matplotlib 3.9.4.
Dependencies: `ForwardDiff`, `Roots`, `Optim`, `SpecialFunctions` + stdlib
`SparseArrays`, `LinearAlgebra`, `Printf`. No external linear-algebra binary,
no MPI, as §9.5.1 requires.

**Layout.** `Builds/Opus_autonomous/Build_2/co2r_bulk/` follows the §13 module structure:

```
params.jl  mesh.jl  chemistry.jl  kinetics.jl  transport.jl  gas_transport.jl
assembly.jl  newton_affine.jl  driver.jl  checks.jl  model.jl
run_stage1.jl … run_stage5.jl  run_fit.jl  plot_results.py
output_stage1..5/  output_fit/  figures/  tests/sanity.jl
```

---

## 0. Setup

Only friction: the Julia package registry on this machine could not resolve the
UUIDs I first wrote into `Project.toml` (I had guessed them). Read the real
UUIDs out of `~/.julia/environments/v1.11/Project.toml` and instantiated
against the local depot. Not a model issue.

Two structural choices made up front, both aimed at the guide's own instruction
in §13 ("write the driver ONCE — five per-stage copies diverge"):

* **One residual, five stages.** `full_residual!(F, u, V, …, ::Val{S})` carries
  all five stages; `S` is a compile-time parameter, so each stage specialises to
  exactly the code the guide specifies for it and the dead branches vanish.
  `residual_stage1!` … `residual_stage5!` are thin wrappers with the guide's
  names.
* **One continuation driver.** `run_sweep!` in `driver.jl` implements the §9.5.5
  pattern (anchor → tangent predictor → corrector → §9.5.6 assertion → cache)
  and is called unchanged by every stage; only the anchor recipe and the chain
  differ.

`tests/sanity.jl` reproduces every hand-checkable number in §8/§9.6 before any
solve. All matched:

| quantity | model | guide |
|---|---|---|
| bulk pH | 8.2292 | ~8.23 |
| c_H, c_OH, c_HCO₃, c_CO₃, c_CO₂ | 5.90e-6, 9.29e-3, 977.7, 11.17, 11.18 | 5.9e-6, 9.3e-3, 978, 11, 11 |
| max\|S_buf\| at equilibrium | 7.5e-9 | < 1e-6 |
| V_RHE→SHE offset | 0.51305 V | 0.51305 V |
| a_v_CL / a_v_GL | 1.5413e7 / 4.08e7 m⁻¹ | 1.54e7 / 4.08e7 |
| κ_CL | 6.887e-17 m² | ~7.0e-17 |
| σ_s_eff GDL / CL | 73.16 / 11.03 S/m | ~73 / ~11 |
| D_eff_Cs | 1.126e-10 m²/s | 1.13e-10 |
| Re | 14648 | 14648 |
| k_MT(OH⁻) | 1.4512e-4 m/s | 1.45e-4 |
| H (van 't Hoff) / c_CO₂ BC (Stages 1–2) | 18.240 / 16.051 | 18.24 / 16.0 |
| c_Cs from electroneutrality | 570.0 mol/m³ | 570 |
| H_eff (Sechenov) / c_CO₂ BC (Stage 3+) | 14.526 / 12.783 | ~14.5 / ~12.8 |
| ω_inlet | [0.9471, 0, 0, 0.0529] | [0.947, 0, 0, 0.053] |
| M_avg inlet | 40.891 g/mol | 40.89 |

Hand-evaluating §4.2 at the bulk pH and V_RHE = −0.300 gives i_H₂ ≈ 16 mA/cm²
against i_CO ≈ 0.007 mA/cm², matching the guide's "~10 mA/cm² of H₂ against
~0.002 mA/cm² of CO" to the order of magnitude it claims.

---

## 1. Stage 1 — liquid NP diffusion, Dirichlet electrolyte BC

Anchor at −0.300 V RHE from the exact equilibrium guess; ‖F‖∞ at that guess was
1.5e-3, right at the "≈ 2×10⁻³ at the production mesh" of §9.6.

**Result (production mesh 80/120, 400 DOFs):** 89/89 converged, 2.04 Newton
iterations per step, max ‖F‖∞ = 8.5e-13, 0.8 s. §9.5.10 quotes 2.0 iters/step,
~1e-12, ~2 s — reproduced.

**Self-verification against §11's Stage-1 checks:**

| check | guide | measured |
|---|---|---|
| pH across CL | 9–12 during operation | 8.27 – 13.31 over the sweep |
| pH direction at V_RHE = −2.5 | 13.3 at x=0 → 9.4 at x=L_CL | **13.29 → 9.41** |
| CO₂(aq) depletes from ~16.0 | yes | 16.05 at the GDL face → ~0 in the reaction zone |
| FE_CO peak (pre-fit) | 64 % at V_RHE = −1.125 | **64.3 % at −1.125** |
| FE_CO near onset | 0–11 % | 0–11 % |
| i_CO vs experiment | under-predicted | −33 % to −47 % at points 2–4 |
| sign changes di_CO/dV, di_H₂/dV | 1, 0 | 1, 0 |
| global CO₂/OH⁻ balance (§12.6) | < 1e-6 mol/m²/s | 6e-17 / 2e-12 |

The three-digit agreement on the pH gradient and on the FE_CO peak location and
height is a strong signal that the buffer chemistry, kinetics, electroneutrality
and BCs are all encoded as intended.

**Plots read back.** A1: smooth rise → CO peak → rollover, H₂ monotone. A2: pH
maximal at the GDL face and relaxing toward the electrolyte; CO₂(aq) pinned at
16.05 (Henry) on the left and 11.18 (Dirichlet c_eq) on the right with a
depleted interior; OH⁻ driven to the Dirichlet value at x = L_CL; R_CO2R peaked
where CO₂ is available, R_HER peaked at the low-pH electrolyte face. All as
described in §12.3.

Gate passed.

---

## 2. Stage 2 — Sherwood-Reynolds Neumann BC

Self-anchored at −0.300 V (§9.5.5 table); the Stage-1 cached anchor is loaded as
an optional warm start for the anchor solve only. **Deviation note:** the Stage-2
section says "warm-start: load Stage 1 cache at each voltage", but §9.5.5,
§9.7 and the §14 pitfalls table all forbid per-voltage independent solves and
state that stages 1–4 are self-anchoring. I followed the §9.5.5 driver (single
anchor + continuation) and used the Stage-1 cache only for the anchor.

**Result:** 89/89, 2.03 iters/step, max ‖F‖∞ = 3.0e-12, 0.8 s.

| §11 Stage-2 check | target | measured |
|---|---|---|
| Biot number k_MT·L_CL/D_eff | O(0.1–10) | 4.86, 5.71, 9.01, 9.78, 7.67 |
| c_CO₂(L_CL) at high current | → 0 | 1.31 mol/m³ |
| c_OH⁻(L_CL) | ≫ bulk 9.3e-3 | 153.6 mol/m³ |
| J_OH⁻ at the electrolyte face | positive (leaving) | +2.23e-2 |
| J_CO₂ at the electrolyte face | negative (resupplying) | −7.95e-4 |

Currents drop relative to Stage 1 (finite mass transfer now resists product
removal / reactant supply) — the expected direction. Overlay plot confirms the
two curves are smooth and Stage 2 sits below Stage 1 everywhere.

Gate passed.

---

## 3. Stage 3 — Stefan-Maxwell gas + phase transfer

Gas domain implemented as the concatenation `[GDL 1..N_GDL ; CL 1..N_CL]`, so
combined cell index k and face index f are single ranges; the GDL/CL interface
face is the only one using harmonic-mean D_eff and κ, per §8.7. `u_G = 0` and
the pressure row is pinned to `F = p − p_total` (§11 Stage 3).

Gas-only bootstrap: with every cell at the inlet composition the gas rows are
1.6e-4 (only the CL faradaic/R_PT sources are nonzero), confirming the
"trivially converged" claim, and it JIT-compiles the SM chain cheaply.

**Result (1200 DOFs):** 89/89, 1.93 iters/step, max ‖F‖∞ = 1.4e-11, 3.2 s.
§9.5.10 quotes 1.9 iters/step, ~1e-11, ~5 s — reproduced.

| §11 Stage-3 check | measured |
|---|---|
| ω_CO₂ from 0.947 at the GDL inlet → lower at the CL exit | 0.9471 → 0.9276 at V_RHE = −2.5 |
| ω_CO, ω_H₂ increase through the CL | 1.2e-2 and 8.2e-4 at the CL exit |
| p_CO₂ consistent with Henry equilibrium at the interface | H_eff·p_CO₂ = 6.38 vs c_CO₂(aq) = 4.77 mol/m³ (net dissolution, R_PT > 0) |
| pressure exactly uniform | max\|p − p_total\| = **0.000e+00 Pa** |

Currents now over-predict experiment (i_CO +105 % to +400 %) — expected, since
the pre-fit `i0_CO2R = 540 A/m²` was never calibrated and the gas phase has just
removed the CO₂-supply limitation. That is exactly what the Fit step exists for.

One physics observation worth recording: c_CO₂(aq) is *lowest* at the GDL face
even though the gas-phase p_CO₂ is highest there. The cause is the Sechenov
salting-out of §7.4 — carbonate and Cs⁺ peak at the GDL face, so H_eff is
smallest there. This is the same mechanism the guide predicts will reverse in
Stage 5 when migration pulls anions out of the CL.

**B1 read back:** p_CO₂ falls monotonically from 0.88 atm at the flow field,
p_CO/p_H₂ rise, p_total a dead-flat line at 101325 Pa. Gate passed.

---

## 4. Stage 4 — Darcy pressure finalised

Pressure BCs: flow-field Dirichlet applied through the *ghost cell* on face 1
(`u_G[1] = −(κ_GDL/μ)(p[1] − p_total)/dxc[1]`), keeping cell 1's continuity row
intact — the same pattern the guide mandates for φ_s; electrolyte side
`ρu = 0`. Harmonic-mean κ at the GDL/CL interface. Convective term
`ρ ω_j u_G` switched on in the species flux.

**Result:** 89/89, 1.89 iters/step, max ‖F‖∞ = 1.2e-11, 3.4 s.

| §11 Stage-4 check | guide | measured |
|---|---|---|
| gas mass balance ρu_inlet = −ΣS_mass·dx | closes to ~1e-10 | **1.3e-13 – 2.9e-13** |
| Δp across the CL | few hundred Pa – ~1 kPa, Δp/p ≲ 1.5 % | 9 Pa at 0.018 A/cm² → 1308 Pa at 1.28 A/cm² (Δp/p = 1.3 %) |
| Δp at ~0.27 A/cm² | 296 Pa (at fitted kinetics) | 263 Pa at 0.257 A/cm² (at pre-fit kinetics) |
| share of dissolving CO₂ going to CO₂R | 71 % → 11 % across the sweep | 74.8 % → 31.5 % |
| Stage 4 vs Stage 3 currents | "almost identical" | max \|Δi\|/i = **1.0 %** |

The Δp and CO₂-share numbers are quoted in the guide *at the fitted kinetics*;
this sweep is still at the pre-fit kinetics, which is why the share stays higher
at the cathodic end (pre-fit i_CO is ~5× too large, so more of the dissolving
CO₂ is consumed faradaically). The trend and the magnitude both match. The
`dp/dx ≈ μ u_G/κ` check is satisfied identically rather than approximately,
because `u_G` is *defined* by that relation in the discretisation.

**B1 read back:** p_total is flat across the GDL and drops steeply across the CL
— exactly the shape §12.7 predicts from κ_GDL / κ_CL ≈ 1.2×10⁴. Gate passed.

---

## 5. Fit step (after Stage 4)

Four parameters (`i0_CO2R`, `λ_MHC`, `log₁₀ HER_prefactor`, `α_c_HER`) against the
10 numbers of §16.1, on the Stage-4 residual, with the §17.3 exact-voltage
polish re-run at every objective evaluation. Both §17.2 warnings implemented as
**sloped** penalties (bound violation and polish failure), and the §9.5.7
parameter-direction ramp (2 → 4 → 8 sub-steps between the last feasible vector
and the trial) as the fallback when a direct polish fails.

Steps §17.5-1 through §17.5-4 all executed. The one-at-a-time exploration
reproduced every sensitivity the guide lists (i₀ scales i_CO, λ moves the CO
onset, HER_pf scales i_H₂, α_c sets the H₂ Tafel slope) and, usefully, showed
`α_c ≥ 0.15` blowing i_H₂ up by 30× — the regime §9.5.7 warns about, which the
sloped penalty and the parameter ramp handle without stalling NM.

**Fitted values** (322 objective evaluations, 98 NM iterations, SSE = 0.0561,
167 s):

```
i0_CO2R       = 106.604 A/m²          (initial guess 540)
lambda_MHC    = 1.6114 eV             (initial guess 1.8)
HER_prefactor = 2.0632e13             (initial guess 2.409e13)
alpha_c_HER   = 0.088942              (initial guess 0.0872)
```

These land inside the window the guide states for the post-fit kinetics
(§11 gotcha #0: "`i0_CO2R ≈ 100–200 A/m²`, `λ_MHC ≈ 1.6 eV`").

**§17.6 acceptance criteria — all satisfied:**

| criterion | result |
|---|---|
| per-point i_CO tolerance (30/30/30/30/80 %) | 6.1, 17.7, 18.6, 18.9, 37.3 % — **ALL PASS** |
| per-point i_H₂ tolerance (30 %) | 10.4, 5.7, 27.4, 1.9, 7.7 % — **ALL PASS** |
| HER passes through every experimental point | yes (visually on both linear and log panels) |
| FE_CO peak value 70–85 % | **82.9 %** (experiment 79.4 %) |
| FE_CO peak within 100 mV of −1.205 V | at **−1.175 V**, 30 mV away |
| sign changes in di_CO/dV = 1, di_H₂/dV = 0 | 1, 0 |
| no kinks on linear *or* log panels | none |
| 89/89 on one continuous branch | 89/89, verified in §7 below |

The §17.8 diagnostic 89-V Stage-4 sweep at the fitted kinetics was produced
(`output_fit/cache/fit_V*.bin`): 89/89, 1.97 iters/step, max ‖F‖∞ = 1.9e-12.
It is used only for the Stage-4 → Stage-5 comparison, never as a Stage-5 seed.

Gate passed.

---

## 6. Stage 5 — φ_L and φ_s as monolithic DOFs

1480 DOFs, exactly as §11 states for the 80/120 mesh. All fourteen Stage-5
gotchas implemented from the start rather than discovered: type-generic residual
chain with `ElT` (never `T`), `zeros(ElT, …)` throughout, ghost-cell φ_s BC at
the collector with cell 1's continuity row intact, φ_L gauge *override* at
`x = L_CL`, Cs⁺ migration inside `i_L`, NP migration at the right boundary with
`L_eff = D_eff/k_MT`, no row scaling and no residual norm anywhere in the kernel,
V_c = 0 anchor built from physics with `V_applied_SHE` (not `V_cathode`) written
into the φ_s DOFs, and the tangent-predictor continuation.

**Result:** anchor converged in 4 iterations to ‖F‖∞ = 8.5e-11 (guide: ~3–5
iterations to ~1e-10). 121 chain points (33-step pre-anchor ramp from V_c = 0
plus the 89 grid voltages), 89/89 emitted, **1.99 iterations per step**, max
‖F‖∞ = 7.4e-8, **0 sub-step recoveries**, 6.5 s. §9.5.10 quotes 2.0 iters/step,
~1e-7, ~11 s.

**§9.5.8 / §11 checks:**

| check | target | measured |
|---|---|---|
| ∂F/∂V nonzero entries | exactly 1 | **1** |
| \|Δφ_L\| at i_tot > 500 mA/cm² | 5–25 mV, ~15 mV typical | **16.2 – 17.7 mV** |
| sign of Δφ_L | negative | −17.69 mV at the deepest V |
| φ_s(L_CL) − V_applied | 5–80 mV; ~50 mV at 1 A/cm² | 0.4 → 59.4 mV; **50 mV at ≈1.0 A/cm²** |
| GDL share of the solid drop | 80–90 % | **88.6 – 89.2 %** across the whole sweep |
| \|Δi\|/i at the collector | ≤ 1e-6 | **max 2.2e-11** |
| ionic-current rel. err at x = L_CL | < 0.01, expect < 0.001 | **max 1.3e-12** |
| φ_L profile monotone across the CL | yes | monotone at every voltage |
| sign changes di_CO/dV, di_H₂/dV | 1, 0 | 1, 0 |
| post-solve assertion firings | none expected | **none** |

**§11 Stage-5 check 5 — the migration → Sechenov-relief mechanism.** The guide
gives specific reference numbers for the Stage-4 → Stage-5 change at
V_RHE = −2.425, CL cell 1. This build reproduces them to three or four
significant figures:

| quantity | this build | guide's reference build |
|---|---|---|
| c_CO₃²⁻ | 4590.6 → 2845.6 (−38 %) | 4590 → 2845 (−38 %) |
| c_Cs⁺ | 9371.9 → 5539.4 (−41 %) | 9369 → 5537 (−41 %) |
| c_OH⁻ (CL average) | 410.5 → 197.5 (−52 %) | 410 → 197 (−52 %) |
| H_eff | 1.660 → 4.324 (2.61×) | 1.66 → 4.33 (2.6×) |
| c_CO₂(aq) | 1.103 → 3.255 (2.95×) | 1.10 → 3.26 (2.9×) |
| i_CO | 86.2 → 163.7 mA/cm² (**1.90×**) | 86 → 164 (**1.90×**) |
| η_CO2R | −1.9912 → −1.9482 V (43 mV) | −1.9912 → −1.9483 V (43 mV) |
| i_H₂ | −2.8 % | falls 3–10 % |

Across the sweep i_CO/i_CO(Stage 4) runs 1.15× at the FE peak to 1.99× in the
plateau, and i_H₂ falls 2.5–10 % — the guide's stated behaviour, including the
warning that "a rising i_CO is not a bug".

**Plots read back.** D1: φ_s rises monotonically from V_applied at the collector
toward the electrolyte side, the tilt growing with current. D2: φ_L monotone
from ≈−17.7 mV at x = 0 to the 0 gauge at x = L_CL. D3: solid charge balance
1e-11–1e-14 and ionic 1e-15 across all 89 voltages, five and eleven orders below
their tolerances. A1: smooth rise → peak → tail on both linear and log panels.
A2 shows the migration signature directly — Cs⁺ peaks at ~6100 instead of
~9400 mol/m³ and OH⁻ at ~350 instead of ~850, with c_CO₂(aq) never falling
below 2.2 mol/m³.

Gate passed.

---

## 7. Solver-stack verification (`tests/solver_tests.jl`)

The §9.5.9 tests that the stage drivers do not exercise:

* **Step 1 — persistent LU.** Second `aff_factor_solve!` on the same sparsity
  pattern takes the `lu!` path; back-substitution is **20–270× cheaper** than a
  refactorization and returns bitwise the same vector. (Whether the *second*
  factorization times faster than the first is JIT-warm-up noise at this size —
  the meaningful number is the back-substitution ratio, which the kernel and the
  tangent predictor both depend on.)
* **§9.5.8 AD validation.** Central-difference spot check at a converged
  V_RHE = −1.200 Stage-5 state: directional derivatives agree to **4e-11**
  relative, and per-column checks covering every physical DOF type (ln c for
  four species, ω_CO₂/ω_CO/ω_H₂, p in both domains, φ_s in both domains, φ_L)
  agree to 1e-11 – 1e-7, with the worst case on the small-norm `ln c_H` column.
  The AD Jacobian is exact.
* **Step 3 — ∂F/∂V structure.** Exactly 1 nonzero entry at Stage 5.
* **Step 4 — predictor A/B.** Tangent beats carry-forward + φ_s shift at
  **14/14** steps tested, median **29.8×**, best 116× at the kinetics turn-on
  (V_RHE ≈ −1.15). The guide states 20–50×, median ~30×, up to ~500× at turn-on,
  better at every step.
* **§9.5.5 / §17.6-4 branch check.** The five experimental voltages, reached by
  continuation from the neighbouring cached grid state, sit on the 89-point
  sweep line to **≤0.42 %** (≤0.04 % away from the curve's sharpest bend, where
  the residual gap is linear-interpolation error, not a branch difference). A
  wrong-branch sweep would be off by ~20 %. One continuous high-CO branch.

---

## 8. Issues encountered

Four, none of them physics:

1. **Package UUIDs.** I invented the `ForwardDiff`/`Optim` UUIDs when writing
   `Project.toml`; resolution failed with "expected package … to be registered".
   Read the real ones from the machine's default environment.
2. **Warm-start DOF mismatch.** A dev-mesh (300 DOF) run loaded the
   production-mesh (1200 DOF) Stage-3 cache and ForwardDiff threw a
   `DimensionMismatch`. Added a length guard so a mismatched cache is ignored
   rather than silently or noisily misapplied.
3. **Julia soft-scope.** `allpass &= …` inside a top-level `for` in
   `run_fit.jl` created a new local and threw `UndefVarError` *after* the fit had
   finished. Replaced with an accumulator vector. Cost one full re-run of the
   fit; the fitted parameters were identical (the fit is deterministic).
4. **Duplicate voltage at the Stage-5 ramp/grid junction.** The pre-anchor ramp's
   last point lands exactly on `V_RHE_grid[1] = −0.300`, so with
   `chain = [V_ramp; V_RHE_grid]` that voltage was solved and emitted twice —
   90 rows instead of 89, and the zero-length first difference registered as a
   spurious sign change in di_H₂/dV (reported 1, target 0). Dropping
   `V_ramp[end]` fixed both; every other diagnostic was unchanged.

One test-design error worth recording, because it looks like a model bug and is
not: my first AD-vs-FD check scaled the probe direction by each DOF *block's*
rms. The gas blocks mix ω (~1e-3 … 0.95) with p (~1e5), so the block rms is
~5e4 and an rms-scaled step perturbed the mass fractions by ±0.05, driving
ω_CO and ω_H₂ negative and the mixture-average diffusivities out of their
domain — the "discrepancy" was 1e78. With per-DOF relative steps the same
Jacobian checks out at 1e-11. Same root cause as the caveat below.

---

## 9. Deviations from the guide

Only three, all small:

1. **Stage-2 warm start.** §11's Stage-2 section says "load Stage 1 cache at each
   voltage as initial guess", but §9.5.5, §9.7, §10.5 and the §14 pitfalls table
   all state Stages 1–4 are self-anchoring and forbid per-voltage independent
   solves. I followed the §9.5.5 driver — single anchor at −0.300 V plus tangent
   continuation — and used the Stage-1 cache only as an optional warm start for
   the *anchor solve*. Same for Stage 4 from Stage 3. Both anchors converge from
   the plain equilibrium guess as well, so this changes nothing.
2. **One residual and one driver instead of five copies.** `full_residual!` is
   parameterised by `Val{S}`, and `residual_stage1!` … `residual_stage5!` are
   thin wrappers with the guide's names. `run_sweep!` is the single §9.5.5
   driver. This follows §13's explicit instruction ("write it ONCE") rather than
   the literal per-file structure.
3. **Stage-5 pre-anchor ramp step size.** §9.5.5 specifies −0.025 V steps from
   V_c = 0 to the first output point. Since 0.81305 is not a multiple of 0.025, I
   used 33 equal steps of −0.024638 V so the ramp lands exactly on the first grid
   voltage. §9.5.5 itself estimates "~32 points", so this matches the intent.

Everything else — every parameter, BC, sign convention, residual form, solver
setting in §9.5.7, cache format, plot specification — is as written.

### Caveat found in the specification (not a deviation, but worth flagging)

**§9.5.3's weight formula is coarse for the gas blocks.** The weights are
`w_j = |u_j| + afrac·rms(|u| over j's DOF block)` over "the six natural DOF
blocks … (liquid log-c, gas GDL, gas CL, φ_s GDL, φ_s CL, φ_L)". Each gas block
contains three mass fractions (0.001 – 0.95) *and* one pressure (~1e5) per cell,
so the block rms is ~5e4 and, at `afrac = 1e-3`, the ω weights become ~50
instead of ~1. The error estimate is therefore ~50× looser on ω than on p, and
`rtol = 1e-8` effectively bounds the ω error at ~5e-7 rather than ~1e-8.

I implemented it exactly as specified and it is **not** a problem in practice:
max ‖F‖∞ was 1.4e-11 at Stages 3–4 and 7.4e-8 at Stage 5, the §9.5.6 assertion
never fired at any voltage in any stage, and iteration counts matched the
guide's. But if a future build sees loose gas residuals, splitting the gas
blocks into separate ω and p blocks is the first thing to try. The affine
covariance argument of §9.5.3 is unaffected — this is about how tight the
stopping test is, not about which direction Newton takes.

---

## 10. Summary

Every stage gate passed on first physics attempt. Full pipeline, production mesh
(N_CL = 80, N_GDL = 120), single Julia process:

| stage | DOFs | converged | iters/step | max ‖F‖∞ | wall |
|---|---|---|---|---|---|
| 1 | 400 | 89/89 | 2.04 | 8.5e-13 | 0.8 s |
| 2 | 400 | 89/89 | 2.03 | 3.0e-12 | 0.8 s |
| 3 | 1200 | 89/89 | 1.93 | 1.4e-11 | 3.2 s |
| 4 | 1200 | 89/89 | 1.89 | 1.2e-11 | 3.4 s |
| Fit | 1200 | 322 evals + 89/89 sweep | 1.97 | 1.9e-12 | 167 s |
| 5 | 1480 | 89/89 (121 chain pts) | 1.99 | 7.4e-8 | 6.5 s |

Against §9.5.10's reference table (2.0 / 2.0 / 1.9 / 1.9 / ~2.0 iters per step;
1e-12 / 3e-12 / 1e-11 / 1e-11 / 1e-7) every column matches. The Stage-4 → Stage-5
migration numbers of §11 check 5 reproduce the guide's validated build to three
or four significant figures.

**Conclusion for the study.** The guide was sufficient to build the model
without human intervention. Nothing in it had to be guessed at, worked around,
or resolved by looking at another document. The one internal inconsistency
(Stage-2 per-voltage warm start vs. the §9.5.5 self-anchoring driver) is
resolvable from the guide's own text, and the one soft spot in the specification
(the gas-block weight scale) is a tolerance-tightness issue that never became
visible in the results. All four issues I hit were Julia/tooling mechanics, not
specification gaps.

### Reproducing

```bash
cd co2r_bulk
julia --project=. tests/sanity.jl        # §8/§9.6 constants and equilibrium
julia --project=. run_stage1.jl          # → STOP gate
julia --project=. run_stage2.jl
julia --project=. run_stage3.jl
julia --project=. run_stage4.jl
julia --project=. run_fit.jl             # writes output_fit/fitted_params.txt
julia --project=. run_stage5.jl
julia --project=. tests/solver_tests.jl  # §9.5.9 steps 1/3/4 + branch check
julia --project=. tests/stage45_compare.jl
python3 plot_results.py output_stage1 output_stage2 output_stage3 \
                        output_stage4 output_fit output_stage5
python3 plot_results.py --overlay output_stage1 output_stage2 output_stage3 \
                        output_stage4 output_fit output_stage5
```

Set `MESH_N_CL=20 MESH_N_GDL=30` for a fast development run (biased results —
never for a reported fit, per §8.2).
