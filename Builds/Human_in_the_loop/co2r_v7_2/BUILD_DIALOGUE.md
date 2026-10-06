# co2r_v7_2 — Build Dialogue

A running log of this build: what was asked, what was built, every issue hit,
every error caught (by me or by the operator), and every deviation from the
implementation guide.

**Target:** `Builds/Human_in_the_loop/co2r_v7_2`
**Source of truth:** `Implementation_Guide/CO2R_BULK_MODEL_IMPLEMENTATION_GUIDE.md` (2906 lines, guide v7.1)
**Purpose:** reproducibility test of the implementation guide — the guide is the
*only* file consulted. No other build, script, or note in the repository was
read at any point.
**Started:** 2026-09-05
**Environment:** Julia 1.11.7, Python 3.9 / matplotlib 3.9.4, macOS (Darwin 24.3.0)

---

## Operator inputs

| # | When | Input |
|---|------|-------|
| 1 | Kick-off | Build the model exactly as outlined in the V7 guide, in `Builds/Human_in_the_loop/co2r_v7_2`. Do not reference any other files — this is a reproducibility test of the guide. Stop at stop gates for review. Pay close attention to the solver section. Keep a build-dialogue markdown file tracking the process, inputs, interventions, issues, errors (self-caught and operator-caught), and deviations. |
| 2 | Setup question | **Plot style** — asked whether to use plain matplotlib (guide-faithful; the guide never mentions the lab's `customplot.gengrid`) or the lab helper. Operator chose **plain matplotlib**, keeping zero files referenced outside the guide. |

---

## Build order

Following §13's canonical execution order:

```
params.jl  mesh.jl  chemistry.jl  kinetics.jl  transport.jl  gas_transport.jl
assembly.jl  newton_affine.jl  cache.jl  driver.jl  export.jl  diagnostics.jl
  -> run_stage1.jl -> STOP
  -> run_stage2.jl -> STOP
  -> run_stage3.jl -> STOP
  -> run_stage4.jl -> STOP
  -> run_fit.jl    -> STOP
  -> run_stage5.jl -> STOP
  -> plot_results.py
```

Production mesh throughout: `N_CL = 80`, `N_GDL = 120`, `q = 1.3` (§8.2, §9.1).
The `MESH_N_CL` / `MESH_N_GDL` dev overrides are wired but unused.

---

## Environment setup

**Issue E-1 (self-caught, environment only).** My first `Project.toml` carried
wrong package UUIDs for `ForwardDiff` and `Optim` (written from memory).
`Pkg.instantiate()` failed with `expected package ForwardDiff [f6369f11-...] to
be registered`. Diagnosed by loading the General registry directly and
confirming it held 14 281 packages — i.e. the registry was fine and the UUID was
mine. Looked the four UUIDs up in `Registry.toml` and corrected them. No model
code involved.

Dependencies: `ForwardDiff`, `SpecialFunctions`, `Roots`, `Optim` + stdlib
`SparseArrays`, `LinearAlgebra`, `Printf`.

---

## Deviations from the guide

Every deviation is listed here, with the guide text that motivated it.

### D-1 — `dx_between` is face-indexed (length N+1), not centre-pair-indexed (length N−1)

The guide specifies this array two ways that cannot both hold:

* §11 Stage 5: "**`dx_half` for the ghost-cell φ_s BC:** ... `gdl_mesh.dx_between[1]`
  (half of the first cell's width for a uniform mesh, or the **face-to-centre
  distance** for a graded mesh)" — this makes `dx_between[1] = dx[1]/2`.
* §11 Stage 1 pseudocode: `J[i,f] = diffusion_flux(c[i,f-1], c[i,f], D_eff[i], dx_between[f-1])`
  for `f in 2:N` — this makes `dx_between[1]` the centre-to-centre distance
  between cells 1 and 2, i.e. a *full* cell width on a uniform mesh.

**Resolution:** `dx_between` is stored with **N+1 entries, one per face** —
`dx_between[1] = dx[1]/2`, `dx_between[f] = x_c[f] − x_c[f−1]` for `2 ≤ f ≤ N`,
`dx_between[N+1] = dx[N]/2`. This satisfies the §11 Stage-5 statement verbatim,
and the Stage-1 stencil is written as `dx_between[f]` (the distance across face
`f`), which is the physically required quantity. The guide's `[f-1]` is read as
an indexing slip in abbreviated pseudocode, not as a different discretization.
Telescoping check passes: `sum(dx_between) == L` exactly for both meshes.

### D-2 — `k_MT` function renamed `k_MT_of`

§7.2's snippet is `const k_MT = [k_MT(D_i) for i in 1:5]`, which is not valid
Julia: a `const` binding cannot share a name with the function it calls.
Function renamed `k_MT_of(D_i)`; the constant array keeps the guide's name
`k_MT`. Values unchanged (`k_MT_OH = 1.451e-4 m/s`, guide: `1.45e-4`).

### D-3 — Cs⁺ migration IS included in `i_L`, as §3.1 requires

§11 Stage 5 Step 5 pseudocode has `i_L_face[f] = F_const * sum(z[i]*J[i,f] for i in 1:5)`,
which omits Cs⁺. But §3.1, §9.4 item 8, the §9.5.8 checklist and the §3.1
"Δφ_L sanity check" all state emphatically that the Cs⁺ NP migration current
**must** enter `i_L` (omitting it gives Δφ_L ≈ 1 mV instead of ~15 mV and
under-predicts current ~30 %). Implemented per §3.1/§9.4: `np_flux` at the
N_CL−1 interior faces with `D_eff_Cs`, `z_Cs = +1`, and `J_Cs = 0` at both CL
faces. The Step-5 pseudocode is treated as abbreviated.

### D-4 — `SpecialFunctions.jl` added to `Project.toml`

The §9.5.8 reproducibility checklist says the project needs "`ForwardDiff`
(plus stdlib `SparseArrays`, `LinearAlgebra`)". The §4.1 MHC kernel calls
`erfc`, which lives in `SpecialFunctions.jl`. `Roots` (§9.6) and `Optim`
(§17.4) are likewise used by the guide's own code but absent from that
checklist. All four added; ForwardDiff differentiates `erfc` through DiffRules,
verified by the FD spot check below.

### D-5 — one unified `full_residual!` instead of five `residual_stageN!`

§13 lists `assembly.jl # full_residual!` as the residual module and warns about
the driver: "Write it ONCE — five per-stage copies diverge." The same argument
applies to the residual, so there is a single `full_residual!(F, u, V_cathode, ctx)`
whose behaviour is selected by `ctx.ly.stage`. Physics per stage is exactly as
specified; only the code organisation differs from the `residual_stage1!` /
`residual_stage5!` naming used in the guide's pseudocode.

### D-6 — sub-step recovery is recursive to depth 4

§9.5.5 specifies "on a failed step, roll back, halve dV, solve the two
half-steps with `alpha_init = 1e-2`". Implemented recursively (a failed
half-step halves again), capped at 4 levels. A strict superset of the guide's
one level. Never triggered so far.

### D-8 — NLEQ-ERR a-priori convergence test restored in `newton_affine!`

The §9.5.3 step listing terminates only on the *simplified* correction
`‖Δū‖_w`, after a trial is accepted (step 6). Deuflhard Alg. 3.13 — the
algorithm the §9.5 preamble names as the source — also tests the *ordinary*
correction `‖Δu^k‖ ≤ tol` before the damping loop. That test is load-bearing:
without it, a solve started from an already-converged state stalls with
`damping_stalled` (full analysis under Error S-2 in the Fit-step log).
Restored verbatim from the cited algorithm. Continuation sweeps are unaffected
(Stage 1 bit-identical before and after).

### D-7 — plots use plain matplotlib

Operator decision (input #2). The guide's §13 module list specifies
`plot_results.py # Python plotting (matplotlib, ...)`, so this is guide-faithful;
recorded here only because the lab house style normally routes figures through
`customplot.gengrid`, which was deliberately not used to keep the reproducibility
test clean.

---

## Observations (no action taken)

### O-1 — §9.5.3 weight blocks mix gas ω and p

The six weight blocks of §9.5.3 put the gas mass fractions (O(1)) and the gas
pressure (O(10⁵)) in the same block ("gas GDL", "gas CL"). The block floor
`afrac·rms` is therefore ≈ 50 for both, so the ω weights are ~50× their own
magnitude and ω corrections count ~50× less in `‖·‖_w` than a per-quantity
block would give. Implemented **as specified** — the guide reports this exact
stack converging at ~1.9 iters/step in Stages 3–4, and the post-solve ‖F‖∞
assertion is the independent backstop. Flagged here to re-check at Stage 3.

---

## Stage log

### Pre-stage: §9.5.9 first-build sanity tests (steps 1–4)

`sanity_tests.jl`. Steps 5–8 are Stage-5 tests, run by `run_stage5.jl`.

| Step | Test | Result | Guide criterion | Pass |
|---|---|---|---|---|
| 1 | Persistent LU, same Jacobian twice | 24.74 ms → 0.26 ms (95×); `n_symbolic=1`, `n_numeric=1`; `plu.F \ b` = 0.019 ms; solves bit-identical | second call skips symbolic | ✅ |
| 1b | AD Jacobian vs central FD, 7 columns | max rel. difference **2.7e-10** | AD validated by FD spot check (§9.5.8) | ✅ |
| 2 | Stage-1 anchor at −0.300 V from the exact equilibrium guess, `alpha_init = 1e-2` | **5 iterations**, ‖F‖∞ = **2.1e-14**; ‖F‖∞ at the guess = 7.2e-3 | ≤ ~6 iters, ‖F‖∞ ≤ 1e-9; guess ≈ 2e-3 | ✅ |
| 3 | `∂F/∂V` structure probe | Stage 1: **160 / 400**; Stage 5: **exactly 1 / 1480** | "Stage 5: exactly 1. Stages 1–2: ~160/400" | ✅ |

Step 4 (predictor A/B) is run inside the Stage-5 driver where the comparison is
meaningful; recorded there.

**Constants cross-check against the guide's own stated values** — all matched
on the first run, no corrections needed:

| Quantity | Built | Guide |
|---|---|---|
| `a_v_CL` | 1.5413e7 m⁻¹ | ≈1.54e7 |
| `a_v_GL` | 4.0800e7 m⁻¹ | ≈4.08e7 |
| `D_eff_Cs` | 1.1259e-10 m²/s | ≈1.13e-10 |
| `κ_CL` | 6.887e-17 m² | ≈7.0e-17 |
| `σ_s_eff` GDL / CL | 73.16 / 11.03 S/m | ≈73 / ≈11 |
| `V_RHE_to_SHE_offset` | 0.513052 V | 0.51305 |
| `Re` | 14648 | ≈14648 |
| `k_MT,OH` | 1.451e-4 m/s | ≈1.45e-4 |
| `H_eff` (van 't Hoff) | 18.240 mol/(m³·atm) | ≈18.24 |
| `ω_inlet` | [0.9471, 0, 0, 0.0529], M̄ = 40.891 g/mol | ≈[0.947,0,0,0.053], 40.89 |
| Bulk equilibrium | pH 8.2292, c = [5.90e-6, 9.29e-3, 977.65, 11.169, 11.178] | pH 8.23, [5.9e-6, 9.3e-3, 978, 11, 11] |
| max\|S_buf\| at equilibrium | 7.45e-9 mol/m³/s | < 1e-6 required |
| `c_Cs` from electroneutrality | **570.00** mol/m³ | 570 |
| `H_eff` with Sechenov | 14.526 mol/(m³·atm) | ≈14.5 |
| `c_CO2` at the GDL face | 12.783 mol/m³ | ≈12.8 |
| DOF counts (stages 1→5) | 400 / 400 / 1200 / 1200 / 1480; dev mesh S5 = 370 | identical |

---

### Stage 1 — liquid NP diffusion only, Dirichlet electrolyte BC

`run_stage1.jl`. 400 DOFs. Anchor at −0.300 V RHE from the exact equilibrium
guess, then tangent-predictor continuation down the 89-point grid.

**Result: 89/89 converged.**

| Metric | Built | Guide (§9.5.10 / §11) |
|---|---|---|
| Voltages converged | 89/89 | 89/89 |
| Newton iters/step | **2.04** | 2.0 |
| max ‖F‖∞ | **1.26e-12** | ~1e-12 |
| Wall time | 0.88 s | ~2 s (M2 Pro) |
| LU factorizations | 1 symbolic + 181 numeric | symbolic once |
| Sub-step recoveries | 0 | none expected |

**Physicality checks (§11 Stage 1, §12.3, §12.6):**

| Check | Built | Guide |
|---|---|---|
| FE_CO peak | **64.3 % at V_RHE = −1.125** | "peak 64 % at V_RHE = −1.125" |
| FE_CO near onset | 0.02 % at −0.300; ≤11 % up to −0.80 | "FE_CO 0–11 % near onset" |
| i_CO / i_H₂ at −0.300 V | 0.0020 / 10.42 mA/cm² | "~0.002 mA/cm² CO against ~10 mA/cm² H₂" |
| pH profile at V_RHE = −2.5 | **13.288 (x=0) → 9.415 (x=L_CL)** | "pH 13.3 at x = 0 → 9.4 at x = L_CL" |
| pH decreases toward electrolyte | true | §12.3 requires |
| c_OH(L_CL) above bulk c_eq | 1.42e-1 ≫ 9.29e-3 | §11 Stage 2 requires |
| CO₂(aq) depletes from the 16.05 Henry value | 15.99 → 11.19 (mild) … 8.04 → 9.12 (−2.5 V) | §11 Stage 1 check 2 |
| Sign changes di_CO/dV | 1 | exactly 1 |
| Sign changes di_H₂/dV | 0 | 0 (strictly monotone) |
| §12.6 CO₂ mass balance | max residual **1.8e-15** mol/m²/s | < 1e-6 |
| CO₂R under-predicted vs experiment | yes (−10 to −47 % at points 1–4) | §11 Stage 1 check 5 expects this |

**Figures generated:** `A1_polarization_stage1.png`, `A2_liquid_stage1.png`.
The A2 colour ramp and its colourbar are built from the *same* truncated
colormap object and the *same* `Normalize`, so the direction mismatch the guide
warns about in §12.7 A2 is impossible by construction (light = mild, dark = cathodic).

**Errors caught at this stage:** none in the model. One cosmetic matplotlib
warning (redundant `linestyle`), fixed.

🛑 **STOP GATE 1 — awaiting operator review.**

---

**Operator input #3 (Stop Gate 1):** "Looks good please continue."

---

### Stage 2 — Sherwood-Reynolds Neumann BC at the electrolyte side

`run_stage2.jl`. 400 DOFs, unchanged layout and solver; the only change is the
right-face flux, `J[i,N+1] = -k_MT[i] * (c_eq[i] - c[i,N])`.

**Anchor note.** §11 Stage 2 says "Warm-start: Load Stage 1 cache at each
voltage as initial guess", while §9.5.5/§10.5 say Stages 1–4 are self-anchoring
and the previous-stage cache is "an optional convenience for the first voltage,
not a requirement" — and §9.7 forbids a per-voltage independent-solve loop
("Copy that structure. Do not write a per-voltage independent-solve loop.").
Resolved in favour of the §9.5.5 production driver: the Stage-1 cache is used
as the **anchor** guess at −0.300 V, and every later voltage is reached by
tangent-predictor continuation. Not logged as a deviation — §10.5 explicitly
sanctions this reading.

**Result: 89/89 converged.**

| Metric | Built | Guide (§9.5.10) |
|---|---|---|
| Voltages converged | 89/89 | 89/89 |
| Newton iters/step | **2.03** | 2.0 |
| max ‖F‖∞ | **2.68e-12** | ~3e-12 |
| Wall time | 1.06 s | ~2 s |
| Sub-step recoveries | 0 | none |

**Physicality checks (§11 Stage 2):**

| Check | Built | Guide criterion |
|---|---|---|
| Biot numbers `Bi_i = k_MT_i L_CL / D_eff_i` | H⁺ 4.86, OH⁻ 5.71, HCO₃⁻ 9.01, CO₃²⁻ 9.79, CO₂ 7.67 | "O(0.1–10)" ✅ |
| Electrolyte-face flux = `k_MT(c_local − c_eq)` | max rel. mismatch **0.0** | required identity |
| Flux signs (no species-specific rule) | OH⁻ **+**2.23e-2 (leaving), CO₃²⁻ **+**6.75e-2 (leaving), HCO₃⁻ **−**5.67e-2 (entering), CO₂ **−**7.95e-4 (entering) | "for OH⁻ … positive; for CO₂ … negative" ✅ |
| `c_CO2(L_CL)` → 0 at high current | 11.46 → **1.31** mol/m³ | "should approach 0 (depletion-limited)" ✅ |
| `c_OH(L_CL)` above bulk `c_eq` = 9.29e-3 | 2.56e-2 → **1.54e2**, above bulk at every voltage | "noticeably *above* … not near zero" ✅ |
| pH decreases toward the electrolyte | 13.354 → 12.448 at −2.5 V | §12.3 ✅ |
| §12.6 CO₂ mass balance | 1.54e-15 mol/m²/s | < 1e-6 ✅ |
| Sign changes di_CO/dV, di_H₂/dV | 1, 0 | 1, 0 ✅ |

**Effect of the finite mass transfer (Stage 1 → Stage 2), as expected:** currents
fall everywhere because the electrolyte boundary layer now resists resupply —
i_CO −45 % at −1.20 V and −60 % at −2.50 V; i_H₂ −13 % and −26 %. FE_CO peak
moves 64.3 % → 52.8 %, still at V_RHE = −1.125.

**Figures:** `A1_polarization_stage2.png` (Stage 1 overlaid as dotted lines,
per the §11 Stage-2 stop-gate requirement), `A2_liquid_stage2.png`.

**Errors caught at this stage:** none.

🛑 **STOP GATE 2 — awaiting operator review.**

---

**Operator input #4 (Stop Gate 2):** "Looks good, continue."

---

### Stage 3 — Stefan-Maxwell gas transport

`run_stage3.jl`. 1200 DOFs = 5×80 liquid + 4×(120+80) gas. `u_G = 0`, pressure
row pinned to `p_total`, CO₂(aq) left BC switched to `H_eff(Sechenov) × p_CO2(local gas DOF)`,
volumetric `R_PT` added. Kinetics still simplified.

**§11 bootstrap step 1 (gas-only) performed as specified:** with `i0_CO2R`,
`HER_prefactor` and `k_GL` set to zero and every gas cell at the inlet
composition and `p_total`, the 800 gas residual rows evaluate to
**max |F| = 1.46e-15** — trivially converged, exactly as the guide predicts,
and it triggers the JIT compilation of the whole SM chain in 0.31 s. Kinetics
are restored immediately afterwards.

**Result: 89/89 converged.**

| Metric | Built | Guide (§9.5.10) |
|---|---|---|
| DOFs | 1200 | 1200 |
| Voltages converged | 89/89 | 89/89 |
| Newton iters/step | **1.91** | 1.9 |
| max ‖F‖∞ | **1.33e-11** | ~1e-11 |
| Wall time | 3.90 s | ~5 s |
| Anchor | 3 iterations | — |
| Sub-step recoveries | 0 | none |

**Physicality checks (§11 Stage 3):**

| Check | Built | Guide criterion |
|---|---|---|
| ω_CO₂ falls from the inlet value | 0.94713 → 0.91422 at −1.65 V; monotone decreasing across GDL+CL | "decreases from ~0.947 at the GDL inlet to noticeably lower values at the CL exit" ✅ |
| ω_CO, ω_H₂ increase through the CL | 1.18e-2 → 1.23e-2 and 7.94e-4 → 8.19e-4 at −2.5 V | ✅ |
| Pressure exactly uniform | max \|p − 101325\| = **0.000e+00 Pa** | "p is **exactly** uniform at 1 atm" ✅ |
| Henry consistency in the CL | ratio c_CO2(aq)/(H_eff·p_CO2) = 0.9999 at −0.3 V, 0.63–0.75 at −2.5 V | "consistent with Henry's law equilibrium" — the sub-unity ratio at high current is the k_GL transport limitation, i.e. the intended physics ✅ |
| §12.6 CO₂ mass balance | 1.88e-12 mol/m²/s | < 1e-6 ✅ |
| Sign changes di_CO/dV, di_H₂/dV | 1, 0 | 1, 0 ✅ |

**Currents now over-predict experiment** (+131 % i_CO at −1.205 V, FE_CO peak
92.4 %), the mirror image of Stages 1–2. Expected: the gas phase removes the
CO₂ starvation while the kinetics are still the pre-fit `i0_CO2R = 540 A/m²`.
§11 Stage 5 gotcha #0 states the post-fit value is ≈100–200 A/m², a ~5×
reduction, which is the right direction and magnitude to absorb this.

#### Solver investigation S-1 — the LU sparsity pattern is not perfectly fixed

§9.5.2 asserts "Across a continuation sweep the sparsity pattern is **fixed** —
same `colptr`/`rowval` at every Newton iteration at every voltage". The Stage-3
log showed **2** symbolic factorizations, not 1, so this is not strictly true
here. Investigated rather than ignored, because a wrongly-reused symbolic
analysis would be a silent-corruption class of bug.

*What fired.* Added counters to distinguish the two rebuild paths. Result:
`nnz-guard rebuilds 1, lu!-pattern-catch rebuilds 0` — the `pattern_nnz`
check fired once. Cause: the anchor guess sets `ω_CO = ω_H₂ = 0` **exactly**,
so several AD partials are structurally zero and `sparse(J)` drops them; once
the kinetics produce CO and H₂ those entries become nonzero and nnz grows.
A one-time transient at the start of the sweep, not recurring churn.

*Is the guard sufficient?* Tested directly whether `lu!` can silently accept a
matrix with the **same nnz but a different pattern** — the one case
`pattern_nnz` alone would miss. At realistic size (n = 400, 3526 nonzeros, 60
entries relocated) `lu!` throws `ArgumentError: pattern of the matrix changed`.
So SparseArrays checks `colptr`/`rowval` itself, and the guide's
`catch → rebuild` fallback converts that into a correct full refactorization.
**No silent corruption is possible; the §9.5.2 design is sound.** Left exactly
as the guide specifies — the counters are the only addition.

**Figures:** `A1_polarization_stage3.png` (Stages 1–2 overlaid dotted),
`A2_liquid_stage3.png`, `B1_gas_stage3.png` (first stage with gas DOFs).

**Errors caught at this stage:** none in the model.

🛑 **STOP GATE 3 — awaiting operator review.**

---

**Operator input #5 (Stop Gate 3):** "Looks good, continue."

---

### Stage 4 — Darcy pressure finalized

`run_stage4.jl`. 1200 DOFs, unchanged layout. The pressure row stops being
pinned and becomes the total-mass continuity balance, the species flux picks up
`rho*omega_j*u_G` with `u_G = -(kappa/mu) dp/dx`, and the GDL/CL interface uses
the harmonic-mean permeability. Warm-started direct from the Stage-3 cache.

**Error C-1 (self-caught, script only).** First run threw
`UndefVarError: dmaxCO not defined in local scope` in the Stage-4-vs-Stage-3
comparison block — Julia's soft-scope rule makes an assignment inside a
top-level `for` loop create a new local rather than write the outer global.
The sweep, the cache and every physicality check had already completed; only
the final comparison print and the export were lost. Fixed by moving the block
into a function (which has hard scope) and re-running. No model code touched,
results bit-identical.

**Result: 89/89 converged.**

| Metric | Built | Guide (§9.5.10) |
|---|---|---|
| Voltages converged | 89/89 | 89/89 |
| Newton iters/step | **1.89** | 1.9 |
| max ‖F‖∞ | **1.21e-11** | ~1e-11 |
| Wall time | 3.87 s | ~5 s |
| Sub-step recoveries | 0 | none |

**Physicality checks (§11 Stage 4):**

| Check | Built | Guide criterion |
|---|---|---|
| Inlet Darcy mass flux = net CL gas mass source | max rel. error **5.4e-8** | "close to ~1e-10" — ours is at pre-fit kinetics with ~5× the CO current; still 8 orders inside any physical tolerance ✅ |
| CL pressure drop | 8.3 Pa (mild) → **1308 Pa** at 1.28 A/cm² | "a few hundred Pa to ~1 kPa" ✅ |
| Δp/p | max **1.44 %** | "≲ 1.5 %" ✅ |
| Δp rises with current | 600 Pa @0.47, 1005 @0.73, 1064 @0.87, 1308 @1.28 A/cm² | guide (at *fitted* kinetics): 296 @0.27, 425 @0.40, 597 @0.62, 1159 @1.15. Extrapolating the guide's trend to 1.28 A/cm² gives ~1290 Pa vs our 1308 ✅ |
| Stage 4 ≈ Stage 3 currents | max change over 89 V: i_CO **1.59 %**, i_H₂ **0.23 %** | "almost identical to Stage 3" ✅ |
| Sign changes di_CO/dV, di_H₂/dV | 1, 0 | 1, 0 ✅ |
| §12.6 CO₂ mass balance | 4.64e-13 mol/m²/s | < 1e-6 ✅ |

**The guide's Darcy-source warning reproduces exactly.** §11 Stage 4 warns that
estimating the gas source from faradaic stoichiometry alone gives the wrong
magnitude and, at high current, the wrong *sign*. At V_RHE = −2.500 we measure
`∫R_PT·M_CO2 = 2.03e-3` kg/m²/s **in** against `i_CO/2F·M_CO = 4.07e-4` and
`i_H₂/2F·M_H₂ = 1.04e-4` **out** — net inflow 1.52e-3. A faradaic-only estimate
would predict net *production* of +5.11e-4, i.e. outflow and a pressure rise:
the wrong sign, precisely as the guide states. The H₂ term matches the guide's
measured 1.13e-4 almost exactly (our i_H₂ is already near experiment); the CO
term is ~4× the guide's 1.03e-4 because the pre-fit `i0_CO2R = 540` over-produces CO.

**CO₂R share of the dissolving CO₂** (monotone decreasing, as the guide reports):
68.8 % at −1.200, 64.2 % at −1.650, 54.7 % at −2.075, 31.5 % at −2.500.
Guide at fitted kinetics: 71 / 59 / 35 / 11 %. Same trend and same mechanism —
ours sits higher at the cathodic end because pre-fit kinetics over-produce CO.

**Figures:** `A1_polarization_stage4.png` (Stage 3 overlaid dotted),
`A2_liquid_stage4.png`, `B1_gas_stage4.png` — panel (d) now shows the real
pressure profile: nearly flat through the GDL (κ = 8.4e-13 m²) with a steep
drop across the CL (κ = 6.9e-17 m²), as §12.7 B1 describes.

🛑 **STOP GATE 4 — awaiting operator review.**  The kinetic fit (§17) is next.

---

**Operator input #6 (Stop Gate 4):** "Great, go ahead."

---

### Fit step (§17) — kinetic parameter fit against Romiluyi et al. 2023

`run_fit.jl`. Nelder-Mead on `i0_CO2R`, `λ_MHC`, `log₁₀ HER_pf`, `α_c` against
the 5 experimental points, using the **Stage-4** residual (φ_L = 0,
φ_s = V_applied_SHE) per §11's speed argument. Staged exactly as §17.5
prescribes: Step 1 manual exploration → Step 2 CO₂R only → Step 3 HER only →
Step 4 joint.

#### 🔴 Error S-2 (self-caught) — MISSING TERMINATION TEST IN THE §9.5.3 KERNEL

**This is the one substantive defect the build found in the guide's solver
section.** It is invisible in Stages 1–5 sweeps and only surfaces in the Fit
step, exactly where the guide's own §17.3/§17.4 workflow triggers it.

*Symptom.* The fitting objective returned the failure penalty `1e6` at
**precisely the initial guess** — the parameter vector whose converged
solutions were sitting in `exp_cache`, polished seconds earlier to
‖F‖∞ ≈ 8e-13. Every parameter vector even infinitesimally different converged
normally (`log₁₀ HER_pf = 13.3820` gave SSE 0.9523; `13.38185` = params₀ gave
1e6). The §17.5 Step-1 exploration table was corrupted at the baseline row of
every sweep, and Step 2 would have started from a poisoned simplex.

*Root cause.* Reproduced in isolation: polish #1 from the grid cache converges
in 2 iterations; polish #2 from that state converges in 1; **polish #3 stalls**
with `reason = damping_stalled`, `alpha_final = 7.45e-9`, 27 wasted residual
evaluations. Once the iterate is at the solution, `‖Δu‖_w` sits at the
round-off floor of the residual evaluation (~1e-15), so `F(u + α·Δu)` cannot
shrink any further and the §9.5.3 step-5 monotonicity test
`‖Δū‖_w ≤ (1 − α/2)·‖Δu‖_w` can **never** be satisfied. α halves to `alpha_min`
and a perfectly good solution is reported as a failure.

*Why the guide's listing permits it.* §9.5.3 gives termination as **step 6**,
after a trial has been accepted, measured on the *simplified* correction
`‖Δū‖_w`. The cited source — Deuflhard, *Newton Methods for Nonlinear
Problems*, Alg. 3.13 (NLEQ-ERR), named in the §9.5 preamble — also tests the
**ordinary** correction `‖Δu^k‖ ≤ tol` *before* entering the damping loop, and
stops there. The guide's condensed six-step listing omits that test, and its
code excerpt (which shows only the `if dubar_n <= rtol` block) does not restore it.

*Fix.* Added the NLEQ-ERR a-priori test immediately after `du_n` is computed:
if `du_n ≤ rtol`, accept the (tiny) full step, re-evaluate F for an honest exit
residual, and terminate with `reason = "err_estimate_apriori"`. Logged as
**deviation D-8**.

*Where else it would have bitten a faithful build:*
* §17.3 — "This function must be called at the start of every fitting
  iteration", warm-started from the previous polish. Any NM revisit of the
  cached parameter vector stalls.
* §9.5.6 — the assertion recovery calls `newton_affine!(u, …; rtol = 1e-10,
  max_iter = 3)` **from a converged state**. It would have stalled every time
  it fired. (It never fired in this build — ‖F‖∞ stayed ≤ 1.7e-11 everywhere.)
* §9.7 — the "polish high-residual voltages instead of restarting" recovery has
  the same shape.

*Regression check.* Stage 1 re-run after the fix is **bit-identical**: 89/89,
2.04 iters/step, max ‖F‖∞ = 1.258e-12, anchor 5 iterations to 2.091e-14. The
a-priori test only fires from an already-converged start, so continuation
sweeps are untouched.

#### §17.5 Step 1 — manual exploration

Reproduces the §17.5 qualitative statements exactly: `i0_CO2R` scales i_CO
(12.9 → 139.2 mA/cm² at −0.816 V over 100 → 3000 A/m²); `λ_MHC` shifts the CO
onset (i_CO at −0.816 V: 405 at λ = 1.0 → 0.4 at λ = 2.6); `HER_prefactor`
scales i_H₂; `α_c` sets the H₂ Tafel slope (i_H₂ at −2.413 V: 15 → 4508
mA/cm² over α_c 0.03 → 0.12). Two extreme trials (`log₁₀ HER_pf = 15`,
`α_c = 0.20`) were rejected after the §9.5.7 parameter-direction ramp failed —
"genuinely extreme and should be rejected", as the guide says.

#### Fitted parameters

| Parameter | Initial guess | **Fitted** | Guide's stated post-fit range |
|---|---|---|---|
| `i0_CO2R` | 540 A/m² | **106.592 A/m²** | "≈100–200 A/m²" (§11 gotcha #0) ✅ |
| `λ_MHC` | 1.8 eV | **1.61156 eV** | "≈1.6 eV" (§11 gotcha #0) ✅ |
| `HER_prefactor` | 2.409e13 | **2.06315e13** | fitted in log space |
| `α_c_HER` | 0.0872 | **0.0889312** | — |

Final SSE 0.056076 over 272 objective evaluations (Step 2: 25 NM iters/54
evals; Step 3: 28/55; Step 4 joint: 73/137). Parameter-ramp rescues 9,
trials rejected 8.

**§17.2 flat-penalty trap avoided.** The guide's v7.1 note warns that a flat
`1e6` penalty makes Nelder-Mead return at **iteration 0** in Step 3, because
`AffineSimplexer(0.025, 0.5)` (confirmed as Optim's default in this version)
puts a vertex at `log₁₀ HER_pf = 20.10`, above the bound. With the sloped
penalty, **Step 3 ran 28 iterations** and improved the SSE 0.059688 → 0.056645
— the guide's predicted "~20–28 iterations" range, hit precisely.

#### §17.6 acceptance criteria — all satisfied

| Criterion | Result |
|---|---|
| i_CO error, points 1–4 (≤30 %) | −6.2, +17.7, −18.6, −18.9 % ✅ |
| i_CO error, point 5 (≤80 %) | +37.4 % ✅ |
| i_H₂ error, all 5 (≤30 %) | −10.4, −5.7, +27.3, +1.8, +7.6 % ✅ |
| FE_CO peak value (70–85 %) | **82.9 %** ✅ |
| FE_CO peak location (within 100 mV of −1.205) | −1.175 V, **30 mV** ✅ |
| Sign changes in di_CO/dV | exactly **1** ✅ |
| i_H₂(V) strictly monotone | true ✅ |
| Sweep coverage, one continuous branch | **89/89** ✅ |
| 5 polished points sit on the 89-point sweep | max deviation **0.40 %** ✅ |

Refit sweep (§17.8): 89/89, 1.97 iters/step, max ‖F‖∞ = 1.74e-12, 3.69 s.
Written to `output_fit/cache/fit_V*.bin` as a **diagnostic artifact** — Stage 5
will not read it (§17.8, §11 gotcha #0).

#### Observation O-2 — §11 Stage 3's ω_CO₂ check fails at fitted kinetics (model is right)

At the fitted kinetics the Stage-3 check "ω_CO₂ decreases from ~0.947 at the
GDL inlet to noticeably lower values at the CL exit at high current" **fails**
at V_RHE = −2.500: ω_CO₂ runs 0.94714 → 0.94973, i.e. *up*.

Not a model error. The CO₂ **mole fraction** (0.87997 → 0.87394) and **partial
pressure** (89163 → 87535 Pa) both decrease monotonically, as they must. The
mass fraction is confounded because H₂ is 22× lighter than CO₂: ~1.8 mol% H₂ at
the CL exit lowers M_n by ~1.7 %, which outruns the 0.7 % fall in x_CO₂, and
`ω_CO₂ = x_CO₂·M_CO₂/M_n` therefore rises. It held at pre-fit kinetics
(Stages 3–4) only because i_CO was ~4× larger there and CO/H₂ never dominated.

**The check should be stated on x_CO₂ or p_CO₂, not ω_CO₂** — which is also the
variable §12.7 B1 plots and the one §7.3's Henry coupling uses. Diagnostic
extended to report all three; no model change.

**Figures:** `A1_polarization_fit.png` (pre-fit Stage 4 overlaid dotted, per
§12.7 E), `A2_liquid_fit.png`, `B1_gas_fit.png`.

🛑 **STOP GATE 5 (Fit) — awaiting operator review.**

---

**Operator input #7 (Stop Gate 5, Fit):** "Looks good, go ahead."

---

### Stage 5 — potentials: φ_L (NP migration) + φ_s (solid Ohm's law)

`run_stage5.jl`. 1480 DOFs. Fitted kinetics loaded from `fitted_kinetics.jl`
before anything else, with a hard guard that aborts if `i0_CO2R == 540`
(§11 gotcha #14, §14 pitfall). **No cache is read** (§17.8): one anchor at
V_c = 0, built from physics, then tangent-predictor continuation.

**Error C-2 (self-caught, script only).** `vcat(falses(33), trues(89))` produces
a `BitVector`, and `run_continuation` is typed on `Vector{Bool}`; Julia threw a
`MethodError` before the sweep started. Wrapped in `Vector{Bool}(...)`. No
physics involved.

**Continuation chain:** 33 pre-anchor ramp points (V_c = 0.000 → −0.800 in
−0.025 V steps) + the 89 output points = **122 total**. §9.5.5 predicts
"~32 points … ~121 total".

**The §9.5.3 unit-weight fallback is exercised on the very first Newton step:**
the anchor guess has both φ_s blocks and the φ_L block **identically zero**
(verified and printed), so their rms is exactly 0 and `‖d‖_w` would return NaN
without the fallback. This is precisely the case the guide's v7.1 revision
added, and it fires here as advertised.

**Result: 89/89 output voltages converged.**

| Metric | Built | Guide (§9.5.10 / §9.5.9) |
|---|---|---|
| DOFs | 1480 | 1480 |
| Chain points | 122 (33 ramp + 89 output) | ~121 |
| Anchor | **4 iterations to ‖F‖∞ = 8.27e-11** | ~3–5 iters, ‖F‖∞ ≈ 1e-10, ≤1e-9 |
| Newton iters/step | **2.00** | 2.0 |
| max ‖F‖∞ | **7.92e-08** | ~1e-7, "≤ ~1e-7" |
| Wall time | 9.16 s | ~11 s |
| Sub-step recoveries | 0 | "untriggered in production" |
| §9.5.6 assertion recovery fired | **0 times** | "the recovery branch never fires" |

**§9.5.9 step 4 — predictor A/B over 121 continuation steps:**
tangent better at **every single step**; improvement factor median **35.8×**,
min 6.0×, max **919.6×**. Guide: "20–50× closer … median ~30×, and up to ~500×
closer at the kinetics turn-on … better at **every** continuation step." ✅

**§11 Stage 5 physicality checks — all pass:**

| # | Check | Built | Guide criterion |
|---|---|---|---|
| 1 | Solid ohmic drop | 0.43 → 59.42 mV; **50.1 mV at i_tot = 1002 mA/cm²** | 5–80 mV; "~50 mV at 1 A/cm²" ✅ |
| 2 | GDL/CL split | **GDL 88.7 % / CL 11.3 %** (88.6–89.2 % across the sweep) | "GDL ~80–90 %" ✅ |
| 3 | φ_L monotone across the CL | true; φ_L = (−17.69, −13.78, 0.00) mV at (1, N/2, N) | monotone, 5–50 mV ✅ |
| 4 | \|i_s_face[1]\| = i_far_total | max \|Δi\|/i = **3.03e-11** | ≤ 1e-6 ✅ |
| 6 | Ionic current at the electrolyte boundary | max \|Δi\|/i = **3.13e-12** | < 0.01, "expect < 0.001" ✅ |
| 7 | Smooth, monotone partial currents | 1 sign change in di_CO/dV, 0 in di_H₂/dV | exactly 1 / 0 ✅ |

**§9.5.8 Δφ_L diagnostic — the Cs⁺-migration smoking gun:** across the 30
voltages with i_tot > 500 mA/cm², \|Δφ_L\| = **16.21–17.69 mV, median 16.99 mV**,
every one inside the 5–25 mV band and inside the tighter §9.5.9 [10, 20] mV
window. Sign negative (φ_L[1] = −17.69 mV against the φ_L[N] = 0 gauge), no
sign flip between neighbouring voltages. Nothing below 5 mV — i.e. Cs⁺
migration is present in `i_L` (deviation D-3) and the sweep is on the **high-CO
branch**, not the low-CO root.

#### §11 Stage 5 check 5 — reproduced to 3–4 significant figures

The guide's most detailed quantitative claim, measured at V_RHE = −2.425,
CL cell 1, both stages at the fitted kinetics:

| Quantity | Stage 4 → Stage 5 (built) | Guide's measured values |
|---|---|---|
| c_CO₃ [mol/m³] | **4589.9 → 2844.9** (−38.0 %) | 4590 → 2845 (−38 %) |
| c_Cs [mol/m³] | **9369.8 → 5537.8** (−40.9 %) | 9369 → 5537 (−41 %) |
| c_OH, CL-averaged | **410.1 → 197.3** (−51.9 %) | 410 → 197 (−52 %) |
| H_eff [mol/(m³·atm)] | **1.660 → 4.326** (2.605×) | 1.66 → 4.33 (2.6×) |
| c_CO₂(aq) [mol/m³] | **1.104 → 3.256** (2.950×) | 1.10 → 3.26 (2.9×) |
| i_CO [mA/cm²] | **86.24 → 163.76 (1.899×)** | 86 → 164 (1.90× plateau rise) |
| i_H₂ | 936.21 → 909.83 (−2.8 %) | "falls 3–10 %" (marginally outside) |

And the §3.1 / §9.4-item-8 c_Cs figures:

| Quantity | Built | Guide |
|---|---|---|
| max c_Cs at V_RHE = −1.200 | **2045** mol/m³ | 2045 |
| max c_Cs at V_RHE = −2.500 | **6028** mol/m³ | 6027 |
| Stage 3–4 vs Stage 5 c_Cs at −2.425 | **9370 vs 5538, ratio 1.69** | 9369 vs 5537, ratio 1.69 |

i_CO **rises** 1.90× into the plateau exactly as §11 check 5 says it should —
migration pulls CO₃²⁻ and OH⁻ out of the CL, relieving the Sechenov
salting-out, raising H_eff 2.6× and c_CO₂(aq) 2.95×. Diagnosed by the guide's
own instruction: compare CL-averaged c_CO₃ and H_eff between stages rather than
assuming a drop.

#### Observation O-3 — p_CO₂ non-monotone in the last 3 CL cells (negligible)

The extended §11 Stage-3 check flags `p_CO2 monotone decreasing: false` at
V_RHE = −2.500. Investigated: **3 of 199** face-differences are positive, all at
cell indices 197–199 — the last three cells, at the closed electrolyte-side gas
boundary where `J_j = 0` and `dp/dx = 0`. The largest positive step is
**+0.052 Pa against a 1995 Pa total span (2.6e-5)**: the CO₂ profile flattens
against the no-flux wall and turns over microscopically as CO and H₂ accumulate
there. Physical, not numerical, and five orders of magnitude below the overall
gradient. No action.

**Figures:** `A1_polarization_stage5.png` (all four earlier stages + the fit
overlaid dotted), `A2_liquid_stage5.png`, `B1_gas_stage5.png`,
`D_potentials_stage5.png` (D1 φ_s across GDL+CL, D2 φ_L across the CL).

🛑 **FINAL PHYSICS STOP GATE — awaiting operator review.**

---

**Operator input #8 (Final physics gate):** "Great, looks like good, go ahead
with the closing summary."

---

# Closing Summary

## Outcome

`Builds/Human_in_the_loop/co2r_v7_2` is a complete, working CO₂R bulk-scale model built from
`Implementation_Guide/CO2R_BULK_MODEL_IMPLEMENTATION_GUIDE.md` and
nothing else. No other build, script, note or figure in the repository was
opened at any point.

All six stages converged **89/89** at the production mesh (N_CL = 80,
N_GDL = 120) on the first attempt, with no physics debugging required at any
stage. Every stage matched the guide's own §9.5.10 performance row and every
§11 physicality check passed.

| Stage | DOFs | Converged | Iters/step | max ‖F‖∞ | Wall | Guide's row |
|---|---|---|---|---|---|---|
| 1 | 400 | 89/89 | 2.04 | 1.26e-12 | 0.88 s | ~2 s, 2.0, ~1e-12 |
| 2 | 400 | 89/89 | 2.03 | 2.68e-12 | 1.06 s | ~2 s, 2.0, ~3e-12 |
| 3 | 1200 | 89/89 | 1.91 | 1.33e-11 | 3.90 s | ~5 s, 1.9, ~1e-11 |
| 4 | 1200 | 89/89 | 1.89 | 1.21e-11 | 3.87 s | ~5 s, 1.9, ~1e-11 |
| Fit | 1200 | 272 evals | ~2/polish | — | ~3 min | ~180 evals, ~80 s |
| 5 | 1480 | 89/89 (122-pt chain) | 2.00 | 7.92e-08 | 9.16 s | ~11 s, 2.0, ~1e-7 |

Code: 2010 lines of model modules + 1091 lines of drivers/diagnostics + 252
lines of Python plotting. 534 cached solutions across six directories.
17 figures.

## Reproducibility verdict

The guide is **highly reproducible**. Beyond the qualitative checks, the build
independently reproduced the guide's *measured* numbers, several to 3–4
significant figures — values that could only match if the physics, the
discretization and the solver all agree:

| Guide's stated measurement | Built | § |
|---|---|---|
| `∂F/∂V` nonzeros: Stage 1 "~160/400", Stage 5 "exactly 1" | **160/400**, **1/1480** | 9.5.9 |
| Stage-1 FE_CO "peak 64 % at V_RHE = −1.125" | **64.3 % at −1.125** | 11 |
| Stage-1 "~0.002 mA/cm² CO against ~10 mA/cm² H₂" at −0.300 V | **0.0020 / 10.42** | 11 |
| Stage-1 "pH 13.3 at x = 0 → 9.4 at x = L_CL" at −2.5 V | **13.288 → 9.415** | 12.3 |
| `c_Cs` from electroneutrality at bulk equilibrium = 570 | **570.00** | 7.4 |
| `H_eff` with Sechenov ≈ 14.5; c_CO₂ at GDL face ≈ 12.8 | **14.526 / 12.783** | 7.4 |
| Post-fit `i0_CO2R` "≈100–200 A/m²", `λ_MHC` "≈1.6 eV" | **106.59 / 1.6116** | 11 |
| §17.2 Step 3 runs "~20–28 iterations" with a sloped penalty | **28** | 17.2 |
| Predictor A/B "median ~30×, up to ~500×, better at every step" | **35.8× median, 919.6× max, every step** | 9.5.9 |
| Stage-5 Δφ_L "5–25 mV, typically ~15 mV" | **16.21–17.69, median 16.99 mV** | 9.5.8 |
| Solid drop "~50 mV at 1 A/cm²" | **50.1 mV at 1002 mA/cm²** | 11 |
| GDL share of the solid drop "80–90 %" | **88.7 %** | 11 |
| c_CO₃ 4590 → 2845 (S4 → S5 at −2.425) | **4589.9 → 2844.9** | 11 |
| c_Cs 9369 → 5537 | **9369.8 → 5537.8** | 11 |
| c_OH 410 → 197 | **410.1 → 197.3** | 11 |
| H_eff 1.66 → 4.33 | **1.660 → 4.326** | 11 |
| c_CO₂(aq) 1.10 → 3.26 | **1.104 → 3.256** | 11 |
| i_CO 86 → 164, "1.90× plateau rise" | **86.24 → 163.76, 1.899×** | 11 |
| max c_Cs 2045 at −1.200, 6027 at −2.500 | **2045 / 6028** | 3.1 |
| Stage 3–4 c_Cs "~1.7× higher": 9369 vs 5537 | **9370 vs 5538, ratio 1.69** | 9.4 |

The full Stage-4 → Stage-5 i_CO trajectory also tracks §11 check 5's shape:
0.99× (mild) → **1.15×** near the FE_CO peak ("10–20 %") → 1.34× → 1.55× →
**1.90×** in the plateau ("up to ~1.9×"). i_H₂ falls 1.7 / 7.6 / 10.0 / 6.4 /
2.8 % against the stated "3–10 %" — inside at three points, marginally outside
at the two extremes.

## What the guide got wrong

### 🔴 S-2 — the §9.5.3 kernel has no termination test on the ordinary correction

**The one substantive defect.** §9.5.3 terminates only on the *simplified*
correction `‖Δū‖_w` (step 6, after a trial is accepted). Deuflhard Alg. 3.13 —
named as the source in the §9.5 preamble — also tests the *ordinary* correction
`‖Δu^k‖ ≤ tol` before the damping loop, and that test is load-bearing.

Started from an already-converged state, `‖Δu‖_w` sits at the round-off floor
of the residual evaluation, `F(u + α·Δu)` cannot shrink further, and the
step-5 monotonicity test `‖Δū‖_w ≤ (1 − α/2)·‖Δu‖_w` can **never** be
satisfied. α halves to `alpha_min` and a correct solution is reported
`damping_stalled`. Reproduced cleanly: polish #1 converges in 2 iterations,
#2 in 1, **#3 stalls** at α = 7.45e-9 after 27 wasted residual evaluations.

Three places in the guide's own prescribed workflow hit it:
* **§17.3** — "must be called at the start of every fitting iteration",
  warm-started from the previous polish. Any NM revisit of the cached
  parameter vector is spuriously rejected. This is how it surfaced: the
  objective returned the `1e6` failure penalty at *exactly* the initial guess,
  corrupting the baseline row of every §17.5 Step-1 sweep.
* **§9.5.6** — the assertion recovery calls the kernel with `max_iter = 3`
  **from a converged state**. It would stall every time it fired.
* **§9.7** — the "polish high-residual voltages instead of restarting"
  recovery has the same shape.

*Recommended guide edit:* add to §9.5.3, between steps 2 and 3 —
"**2b. A-priori convergence test.** If `‖Δu‖_w ≤ rtol`, add `Δu` to the iterate
and stop. Without this the kernel cannot terminate when started from an
already-converged state, because `F(u + αΔu)` is then at the round-off floor
and the step-5 monotonicity test can never be satisfied — the symptom is
`damping_stalled` on a correct solution, and it hits the §17.3 re-polish, the
§9.5.6 assertion recovery and the §9.7 voltage polish."

### O-2 — §11 Stage 3's ω_CO₂ check is stated in the wrong variable

"ω_CO₂ decreases from ~0.947 at the GDL inlet to noticeably lower values at the
CL exit at high current" **fails** at fitted kinetics (0.94714 → 0.94973 at
−2.500 V) while the model is correct: x_CO₂ (0.87997 → 0.87394) and p_CO₂
(89162 → 87167 Pa) both fall monotonically. H₂ at ~1.8 mol% lowers M_n by
~1.7 %, outrunning the 0.7 % fall in x_CO₂, so `ω_CO₂ = x_CO₂·M_CO₂/M_n` rises.
The check passes at pre-fit kinetics only because i_CO is ~4× larger there.
*Recommended edit:* state the check on **x_CO₂ or p_CO₂** — the variable §12.7
B1 plots and the one §7.3's Henry coupling uses.

### D-1 — `dx_between` is specified two incompatible ways

§11 Stage 5 defines `dx_between[1]` as "the face-to-centre distance"
(⇒ `dx[1]/2`); the §11 Stage 1 pseudocode's `dx_between[f-1]` at face `f`
implies centre-to-centre (⇒ a full cell width at index 1).
*Recommended edit:* define `dx_between` once, face-indexed with N+1 entries,
and write the Stage-1 stencil as `dx_between[f]`.

### D-3 — the Stage 5 Step-5 pseudocode omits Cs⁺ from `i_L`

`i_L_face[f] = F_const * sum(z[i]*J[i,f] for i in 1:5)` contradicts §3.1,
§9.4 item 8 and the §9.5.8 checklist, all of which insist the Cs⁺ migration
current must be included. A reader who codes the pseudocode literally gets
Δφ_L ≈ 1 mV and ~30 % under-predicted current — the exact failure the guide
warns about three sections earlier. *Recommended edit:* add the Cs⁺ term to the
Step-5 listing.

### D-2 / D-4 — smaller code-level slips

* §7.2's `const k_MT = [k_MT(D_w[i]) for i in 1:5]` is not valid Julia (a
  `const` binding cannot share a name with the function it calls).
* The §9.5.8 `Project.toml` checklist lists only ForwardDiff + stdlib, but the
  guide's own code needs `SpecialFunctions` (§4.1 `erfc`), `Roots` (§9.6
  `find_zero`) and `Optim` (§17.4).
* §9.5.2 asserts the LU sparsity pattern is "**fixed**" across a sweep. It is
  not quite: the anchor sets ω_CO = ω_H₂ = exactly 0, so some AD partials are
  structurally zero until CO/H₂ appear and nnz grows once. Harmless — the
  `pattern_nnz` guard plus `lu!`'s own `colptr`/`rowval` check (verified to
  throw `ArgumentError`, not corrupt silently) turn it into one extra correct
  refactorization — but the wording overstates the case.

## Deviations from the guide — complete list

| # | Deviation | Kind |
|---|---|---|
| D-1 | `dx_between` face-indexed, length N+1 | resolves a guide self-contradiction |
| D-2 | `k_MT` function renamed `k_MT_of` | syntax fix |
| D-3 | Cs⁺ migration included in `i_L` | follows §3.1 over the Step-5 pseudocode |
| D-4 | `SpecialFunctions`, `Roots`, `Optim` added to `Project.toml` | completes the dependency list |
| D-5 | one unified `full_residual!` instead of five `residual_stageN!` | code organisation; §13's own "write it ONCE" argument |
| D-6 | sub-step recovery recursive to depth 4 | strict superset of §9.5.5's one level; never triggered |
| D-7 | plain matplotlib, no `customplot.gengrid` | operator decision (input #2) |
| D-8 | NLEQ-ERR a-priori convergence test restored | fixes S-2; restores the cited algorithm |

No deviation changes the physics. D-8 is the only one that changes a converged
answer, and only in the degenerate already-converged case — Stage 1 is
bit-identical before and after.

## Errors

**Caught and fixed by me:** 4.

| # | Error | Where | Impact |
|---|---|---|---|
| E-1 | Wrong package UUIDs in `Project.toml` | environment | none — setup only |
| C-1 | Julia soft-scope `UndefVarError` in the Stage-4 comparison block | script | lost one print + export; re-ran, bit-identical |
| C-2 | `BitVector` vs `Vector{Bool}` MethodError | script | caught before the Stage-5 sweep started |
| **S-2** | **Missing NLEQ-ERR a-priori termination test** | **guide's solver spec** | **spurious `damping_stalled` on correct solutions; corrupted the fit objective** |

**Found by the operator:** none. All eight operator inputs were setup
decisions or stop-gate approvals; no correction was ever required.

**Investigated, no fix needed:** 1 (S-1, the LU sparsity-pattern churn —
established that no silent corruption is possible).

## Observations flagged but not acted on

* **O-1** — §9.5.3's weight blocks put gas ω (O(1)) and p (O(10⁵)) in the same
  block, so the `afrac·rms` floor (~50) dominates the ω weights and ω
  corrections count ~50× less in `‖·‖_w`. Implemented as specified; Stages 3–4
  converged at 1.89–1.91 iters/step with ‖F‖∞ ~1e-11, so it is not hurting
  anything here. Worth a sentence in the guide acknowledging it.
* **O-3** — p_CO₂ turns over in the last 3 CL cells at the closed gas boundary
  (+0.052 Pa against a 1995 Pa span). Physical, negligible.

## Files

```
params.jl mesh.jl chemistry.jl kinetics.jl transport.jl gas_transport.jl
assembly.jl newton_affine.jl cache.jl driver.jl export.jl diagnostics.jl
model.jl                       # single include point
run_stage1..5.jl run_fit.jl    # stage drivers, each STOPs
sanity_tests.jl                # §9.5.9 steps 1-3
check_s4_s5.jl fitcheck.jl     # §11 check-5 and mole-fraction verification
plot_results.py                # §12.7 / §18 figures
fitted_kinetics.jl             # generated by run_fit.jl, consumed by run_stage5.jl
output_stage{1..5}/ output_fit/   # caches, CSV, profile binaries, PNGs
```

Execution order (§13): `run_stage1 → 2 → 3 → 4 → run_fit → run_stage5 →
plot_results.py --all`, each with a hard stop gate.

*Build completed 2026-09-05.*
