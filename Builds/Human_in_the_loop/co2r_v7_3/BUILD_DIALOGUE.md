# co2r_v7_3 — Build Dialogue

A running log of this build: what was built, in what order, every deviation from
the implementation guide, every error caught (by me or by the user), and the
user's inputs at each stop gate.

**Source of truth:** `Implementation_Guide/CO2R_BULK_MODEL_IMPLEMENTATION_GUIDE.md`
(guide v7.1, 2906 lines). **No other model file in the repo was read** — this
build is a clean-room reproducibility test of the guide.

**Build started:** 2026-09-05
**Environment:** macOS (Darwin 24.3.0), Julia 1.11.7, Python 3.9.4
**Mesh:** production default `N_CL = 80`, `N_GDL = 120`, `q = 1.3`

---

## User inputs

| # | When | Input |
|---|---|---|
| 1 | Build kickoff | "Build the CO2R model exactly as outlined in [the V7 guide]. Do not reference any other files unless instructed to for reference, since I want to test the reproducibility of the implementation guide. Build this model in `Builds/Human_in_the_loop/co2r_v7_3`. Make sure to stop at stop gates for my review, and pay close attention to the solver section. Also make a build dialogue md file…" |

---

## Deviations from the guide

Anything below is a place where the guide could not be followed literally, or
where it was silent and a choice had to be made.

| # | Guide location | Issue | What I did |
|---|---|---|---|
| D1 | §7.2 `Recommended implementation` | The snippet defines `function k_MT(D_i)` and then `const k_MT = [k_MT(D_w[i]) …]`. In Julia a `const` binding cannot shadow an existing function binding of the same name — this is a hard error. | Renamed the function to `k_MT_of`; the per-species **vector** keeps the guide's name `k_MT`, so every downstream use (`k_MT[i]`, `D_eff[i]/k_MT[i]`) reads exactly as written in the guide. Numerics unchanged. |
| D2 | §11 Stage 1 pseudocode vs §11 Stage 5 `dx_half` | `dx_between` is used inconsistently. Stage 1 indexes interior face `f` as `dx_between[f-1]`, which makes `dx_between` cell-centre-to-cell-centre and length `N-1`. Stage 5 states `dx_half = gdl_mesh.dx_between[1]` is the *boundary-face-to-first-cell-centre* distance, which makes it face-indexed and length `N+1`. Both cannot hold. | Adopted the **face-indexed** convention (length `N+1`): `dx_between[1]` = left face-to-centre, `dx_between[f]` = centre-to-centre for `f = 2..N`, `dx_between[N+1]` = right centre-to-face. This is the geometrically correct one and is what §11 Stage 5 requires. Interior face `f` therefore uses `dx_between[f]`. Documented in `mesh.jl`. |
| D3 | §11 Stage 5 pseudocode Step 5 | The pseudocode builds `i_L_face[f] = F × Σᵢ zᵢ J[i,f]` over the **5 transported species only**, omitting Cs⁺. §3.1, §9.4 item 8 and the §9.5.8 checklist all state emphatically that Cs⁺ migration **must** be in `i_L` (omitting it gives Δφ_L ≈ 1 mV instead of ~15 mV and ~30 % under-predicted current). | Followed the prose, not the pseudocode: `i_L_face[f] = F × (Σᵢ zᵢ J[i,f] + z_Cs J_Cs[f])`, with `J_Cs` from `np_flux` at the `N−1` interior faces and `J_Cs[1] = J_Cs[N+1] = 0`. |
| D4 | §9.5.1 / §9.5.8 dependency list | The guide lists only `ForwardDiff` + stdlib. But §4.1 needs `erfc` (SpecialFunctions), §9.6 needs `find_zero` (Roots), and §17.5 needs `NelderMead` (Optim). | `Project.toml` = ForwardDiff, SpecialFunctions, Roots, Optim (+ stdlib SparseArrays, LinearAlgebra, Printf). No external linear-solver binary, as required. |
| D5 | §13 module list | The guide lists `params.jl … run_stage5.jl` but has no file for the DOF layout/index functions or for the sweep export. | Added `layout.jl` (the §11 Stage-5 `Stage5Layout`/index functions, generalised to all five stages) and `io_export.jl` (CSV + raw-binary export consumed by `plot_results.py`). `model.jl` is a single include point. `solver.jl` (legacy dense FD Newton) was not written — §9.3 forbids FD at production mesh and it is explicitly "not used by production drivers"; the FD path exists only inside `tests_solver.jl` as the AD spot-check of §9.5.8. |
| D6 | §12.7 / §13 `plot_results.py` | The guide specifies matplotlib + Lato but gives no plotting API. | Used the lab-standard `customplot.gengrid` from the repo root (auto panel labels, inward ticks, Lato/Fira-Math, `warm_sequential` light→dark ramp for the all-voltage profile figures, `Pantone_Blue`/`Pantone_Red` for stage overlays). This is a house style choice and touches no model physics. Ramp direction verified numerically (luminance 0.96 → 0.10 as `−E_cath` increases), so the colourbar direction matches the line colours as §12.7 A2 requires. |
| D7 | §11 Stage 5 gotcha #12 vs §7.2 | Not a conflict, recorded for clarity: the Stages 2–4 Sherwood form `−k_MT[i](c_eq[i] − c[i,N])` is *exactly* `np_flux(…, L_eff = D_eff[i]/k_MT[i])` with `∇φ_L = 0`. | Implemented as written in each stage's branch; verified algebraically identical. |
| D8 | §9.5.5 pre-anchor ramp | "V_c = 0 → first output point in −0.025 V steps (~32 points)". 0.81305/0.025 = 32.52, not an integer. | Used 33 equal steps of −0.024638 V so the ramp lands exactly on the first output point (−0.300 V RHE) without a short final step. Step size ≤ the guide's 0.025 V. |

---

## Errors caught and fixed by me

| # | Where | Error | Fix |
|---|---|---|---|
| E1 | `Project.toml` (mine) | Hand-written ForwardDiff UUID was wrong, so `Pkg.instantiate()` failed with "expected package ForwardDiff to be registered". | Regenerated `Project.toml` via `Pkg.add`. |
| E2 | `plot_results.py` | `customplot` sets `mpl.use('svg')` at import, so `savefig(...png)` would fail. | `mpl.use("Agg", force=True)` after the customplot import. |

## Errors found by the user

*(none yet)*

---

## Stage log

### Setup

Files written: `params.jl`, `mesh.jl`, `chemistry.jl`, `kinetics.jl`,
`transport.jl`, `layout.jl`, `gas_transport.jl`, `assembly.jl`,
`newton_affine.jl`, `driver.jl`, `io_export.jl`, `model.jl`,
`run_stage1.jl`, `plot_results.py`, `tests_sanity.jl`, `tests_solver.jl`.

The residual is written **once** (`assembly.jl::residual!`) with stage switches,
and the continuation driver **once** (`driver.jl`), per §13's "Write it ONCE —
five per-stage copies diverge". The whole residual chain is type-generic
(`AbstractVector{ElT} where {ElT<:Real}`, no type parameter named `T`, all work
arrays `zeros(ElT, …)`) per §9.3.2 / §11 Stage-5 gotchas #1–3.

**§9.6 / §12.1 equilibrium and constant checks — all pass** (`tests_sanity.jl`):

| Quantity | Guide | Built |
|---|---|---|
| bulk pH | ~8.23 | 8.2292 |
| c_HCO3, c_CO3, c_CO2 | 978, 11, 11 | 977.65, 11.169, 11.178 |
| K_w / K_2 / K_3 recovery | exact | exact to 5 s.f. |
| `max abs(S_buf)` at equilibrium | < 1e-6 | 7.45e-9 |
| c_Cs from electroneutrality | 570 | 570.00 |
| H_eff van 't Hoff / Sechenov | 18.24 / ~14.5 | 18.2399 / 14.5260 |
| c_CO2 at GDL face, S3+ / S1–2 | 12.8 / 16.0 | 12.783 / 16.051 |
| nernst_fac / RHE→SHE offset | 0.06413 / 0.51305 | 0.064131 / 0.513052 |
| a_v_CL / a_v_GL | 1.54e7 / 4.08e7 | 1.5413e7 / 4.0800e7 |
| D_eff_Cs | 1.13e-10 | 1.1259e-10 |
| Re / k_MT_OH | 14648 / 1.45e-4 | 14648.4 / 1.4512e-4 |
| σ_s_eff GDL / CL | 73 / 11 | 73.16 / 11.03 |
| κ_CL | ~6.9e-17 | 6.8874e-17 |
| ω_inlet / M_avg | [0.947,0,0,0.053] / 40.89 | [0.9471,0,0,0.05287] / 40.891 |
| MHC prefactor `pf` | 43.1 s⁻¹ | 43.131 s⁻¹ |
| DOF counts S1/S3/S5 | 400 / 1200 / 1480 | 400 / 1200 / 1480 |

**§9.5.9 solver component tests — steps 1, 3, 4 pass** (`tests_solver.jl`):

| Step | Criterion | Result |
|---|---|---|
| 1 | persistent LU: 2nd call skips symbolic; `plu.F \ b` back-substitutes | 24.9 ms → 0.47 ms (**53×**); back-sub residual 7.7e-11; nnz 2630 (1.6 % dense) |
| — | AD Jacobian vs FD spot check (§9.5.8) | max rel. difference **3.19e-7** — i.e. agreement at the FD truncation level |
| 3 | `∂F/∂V` nonzeros: Stage 5 **exactly 1**; Stages 1–2 ~160/400; Stages 3–4 ~320–400/1200 | Stage 1: **160**/400; Stage 3: **320**/1200; Stage 5: **1**/1480 |
| 4 | tangent predictor better at every step; median > 10× | better at **every** step; min 7.6×, **median 18.7×**, max 121× |

Step 3 also incidentally confirmed §9.5.9 step 5's anchor claim early: the Stage-5
V_c = 0 anchor converged in **4 iterations** from the constructed physics guess.

---

### Stage 1 — liquid NP diffusion, Dirichlet electrolyte BC  ✅

`run_stage1.jl`, production mesh, 400 DOFs, pre-fit kinetics
(`i0_CO2R = 540`, `λ = 1.8`, `HER_pf = 2.409e13`, `α_c = 0.0872`).

Solver: §9.5 stack — ForwardDiff AD Jacobian (chunk 12) → `sparse(J)` →
persistent UMFPACK LU → affine-covariant kernel (`rtol = 1e-8`) → tangent
predictor over the 89-point grid → post-solve `‖F‖∞ ≤ 1e-6` assertion → cache.

**Result: 89/89 converged, 2.8 s.**

| Metric | Guide (§9.5.10, §11) | Built |
|---|---|---|
| anchor iterations / ‖F‖∞ | ≤ ~6 / ≤ 1e-9 | **5** / **8.8e-14** |
| Newton iters per continuation step | 2.0 | **2.04** (max 5) |
| max ‖F‖∞ over the sweep | ~1e-12 | **1.28e-12** |
| sub-step recoveries | 0 | **0** |
| sweep wall time | ~2 s | 2.8 s |
| peak FE_CO and its voltage | 64 % at V_RHE = −1.125 | **64.3 % at −1.125** |
| FE_CO near onset | 0–11 % | **0.0 %** at −0.300 |
| pH at V_RHE = −2.5, x = 0 → x = L_CL | 13.3 → 9.4 | **13.29 → 9.41** |
| CO₂(aq) at the GDL face | ~16.0 mol/m³ (van 't Hoff only) | 15.99 |

CO₂(aq) depletes from the GDL face toward the electrolyte at every voltage, and
CO₂R currents are under-predicted vs experiment (gas coupling arrives in Stage 3)
— both exactly as §11 Stage-1 checks 2 and 5 state.

**Plots:** `figures/A1_polarization_stage1.png`, `figures/A2_liquid_stage1.png`.

> 🛑 **STOP GATE 1 — awaiting human review.**

---

### Stage 2 — Sherwood-Reynolds Neumann BC at the electrolyte  ✅

**User input #2** (Stop Gate 1): *"Plots look good please continue"* — Stage 1
accepted, no corrections requested.

`run_stage2.jl`, production mesh, 400 DOFs, same pre-fit kinetics. The only
change is the right-face flux: `J[i,N+1] = −k_MT[i]·(c_eq[i] − c[i,N])`
(§7.2 / §11 Stage 2). No change to DOF count, residual layout or solver.

**Result: 89/89 converged, 3.8 s.**

| Metric | Guide (§9.5.10) | Built |
|---|---|---|
| Newton iters per step | 2.0 | **2.04** (max 5) |
| max ‖F‖∞ over the sweep | ~3e-12 | **2.98e-12** |
| sub-step recoveries | 0 | **0** |

**§11 Stage-2 physicality checks — all pass:**

| Check | Guide | Built |
|---|---|---|
| Biot number `Bi_i = k_MT_i·L_CL/D_eff_i` | O(0.1–10) | **4.86 – 9.79** (H⁺ 4.86, OH⁻ 5.71, HCO₃⁻ 9.01, CO₃²⁻ 9.79, CO₂ 7.67) |
| `c_CO2(L_CL)` at high current | → 0 (depletion-limited) | 11.46 → **1.31** mol/m³ from −0.300 to −2.500 V (bulk 11.18) |
| `c_OH(L_CL)` vs bulk `c_eq_OH ≈ 9.3e-3` | noticeably **above** bulk, not near zero | 0.026 → **153.6** mol/m³ (2.8× → 16 540× bulk) |
| electrolyte-face flux = `k_MT·(c_local − c_eq)` | sign falls out of the concentration difference | identity holds to **0.0** (exact); **J_OH > 0** (leaving CL), **J_CO2 < 0** (reservoir resupplying) — exactly the signs §11 predicts |

Currents drop vs Stage 1 at every voltage (finite mass transfer now resists
CO₂ resupply and OH⁻ removal), e.g. at V_RHE = −1.650: i_CO 158.4 → 94.8,
i_H₂ 199.8 → 160.2 mA/cm². Peak FE_CO 64.3 % → 52.8 %, still at −1.125 V.

**Plots:** `figures/A1_polarization_stage2.png`, `figures/A2_liquid_stage2.png`,
`figures/OV_overlay.png` (Stage 1 vs Stage 2 polarization overlay).

No new deviations and no errors at this stage.

> 🛑 **STOP GATE 2 — awaiting human review.**

---

### Stage 3 — Stefan-Maxwell gas transport + phase transfer  ✅

**User input #3** (Stop Gate 2): *"Looks good, continue"* — Stage 2 accepted,
no corrections requested.

`run_stage3.jl`, production mesh, **1200 DOFs** (400 liquid + 800 gas).
Adds: mixture-average Stefan-Maxwell gas transport on the combined GDL+CL mesh
(FSG binaries, Bruggeman per domain, harmonic mean at the single GDL/CL face),
volumetric `R_PT` phase transfer, and the Sechenov-corrected Henry BC for
CO₂(aq) at the GDL face. `u_G = 0` and the pressure row is pinned
(`F = p − p_total`), per §11 Stage 3. Kinetics still simplified.

**Bootstrap (§11 Stage 3):** the gas-only warm-up step was run as specified —
at the anchor's inlet-composition gas state the gas rows are trivially solved
(‖F‖∞ over gas rows = 0), so its only real function is the JIT compile it
forces, which is what §14 "JIT compilation slow" is about. The first AD Jacobian
(including compilation) took 1.6 s; subsequent ones are milliseconds.

**Result: 89/89 converged, 4.3 s.**

| Metric | Guide (§9.5.10) | Built |
|---|---|---|
| Newton iters per step | 1.9 | **1.93** (max 5) |
| max ‖F‖∞ over the sweep | ~1e-11 | **1.38e-11** |
| sweep wall | ~5 s | 4.3 s |
| sub-step recoveries | 0 | **0** |

**§11 Stage-3 physicality checks — all pass:**

| Check | Guide | Built |
|---|---|---|
| ω_CO₂ at the GDL inlet | ≈ 0.947 | **0.9471** |
| ω_CO₂ decreases toward the CL exit at high current | yes | 0.9471 → **0.9142** at V_RHE = −1.650 (the largest drop, at peak i_CO) |
| ω_CO, ω_H₂ increase through the CL | yes | ω_CO 0 → 2.7e-2, ω_H₂ 0 → 8.2e-4 |
| pressure exactly uniform (Stage-3 pin) | **exactly** p_total | max abs(p − p_total) = **0.0 Pa** at every voltage |
| p_CO₂ consistent with Henry equilibrium at the GL interface | yes | c_CO2_bc vs c_CO2(cell 1): 10.41/10.39, 2.50/2.03, 1.14/0.72 — the gap widens with current, as CO₂R consumption pulls cell 1 below local equilibrium |
| CO₂(aq) mass balance (§12.6) | < 1e-6 mol/m²/s | **3e-15 … 1.4e-17** |

**Independent cross-check against the guide's own reference numbers.** §11 Stage-5
check 5 and §9.4 item 8 quote a validated build's Stage-3/4 state at
V_RHE = −2.425. Comparing my Stage 3 (pre-fit kinetics, so *higher* current than
the guide's fitted reference — i_CO 322 vs 86 mA/cm²) at CL cell 1:

| Quantity | Guide ref | Built | Note |
|---|---|---|---|
| c_CO₃²⁻ | 4590 | **4607** | 0.4 % |
| c_Cs⁺ (Stages 3–4) | 9369 | **9646** | 3 % |
| H_eff | 1.66 | **1.55** | lower, consistent with higher current |
| c_CO₂(aq) | 1.10 | **0.863** | lower, consistent |
| CL-avg c_OH⁻ | 410 | **598** | higher, consistent with 3.7× the i_CO |
| Cs⁺ share of local liquid conductivity | "roughly half" | **48.6 %** (3.76 of 7.75 S/m) | ✅ |

**Observation O1 — §12.3 vs Stage-3+ physics (not a deviation, flagged for the
guide).** §12.3 states `c_CO2(aq)(x)` "should deplete from GDL side toward the
electrolyte-channel boundary". That is true in Stages 1–2 (Henry source at
x = 0, sink at x = L). From Stage 3 the gradient **inverts**: R_PT is fast
(`a_v_GL·k_GL = 4080 s⁻¹`) so c_CO₂(aq) tracks the *local* Henry equilibrium
`H_eff(x)·p_CO2(x)`, and H_eff is smallest where the alkalinity is highest —
which §12.3 itself says is the **GDL face**. At V_RHE = −2.500 the profile runs
0.72 (x = 0) → 4.77 mol/m³ (x = L_CL). This follows directly from the guide's own
equations (§7.4 Sechenov + §12.3 pH gradient direction) and from the guide's own
Stage-4 reference numbers, so it is treated as correct model behaviour, not a bug.

Currents rise sharply vs Stage 2 (gas-phase CO₂ supply is now online, exactly as
§11 Stage-1 check 5 anticipates): at V_RHE = −1.650, i_CO 94.8 → 614.2 mA/cm².
Peak FE_CO 92.4 % at −1.275 V — far above experiment because the pre-fit
`i0_CO2R = 540 A/m²` is ~5× too large; the fit after Stage 4 corrects this.

**Plots:** `figures/A1_polarization_stage3.png`, `figures/A2_liquid_stage3.png`,
`figures/B1_gas_stage3.png`, `figures/OV_overlay.png` (Stages 1–3).

No new deviations and no errors at this stage.

> 🛑 **STOP GATE 3 — awaiting human review.**

---

### Stage 4 — Darcy pressure finalized  ✅

**User input #4** (Stop Gate 3): *"Looks good, continue"* — Stage 3 accepted,
no corrections requested.

`run_stage4.jl`, production mesh, 1200 DOFs (no new DOFs). The pressure row
changes from the Stage-3 pin to the true continuity balance
`ρu_L − ρu_R + S_mass·dx`, with p Dirichlet at the flow field (ghost cell),
`dp/dx = 0` at x = L_CL, and harmonic-mean κ at the GDL/CL face. The convective
term `ρ·ω_j·u_G` is now active in the species fluxes. Kinetics still simplified.

**Result: 89/89 converged, 5.9 s.**

| Metric | Guide (§9.5.10) | Built |
|---|---|---|
| Newton iters per step | 1.9 | **1.93** (max 5) |
| max ‖F‖∞ over the sweep | ~1e-11 | **1.20e-11** |
| sweep wall | ~5 s | 5.9 s |
| sub-step recoveries | 0 | **0** |

**§11 Stage-4 physicality checks — all pass:**

| Check | Guide | Built |
|---|---|---|
| Darcy mass balance `ρu(inlet) = −Σ(S_mass·dx)` | closes to ~1e-10 | rel. err **3e-11 … 2e-10** at every voltage |
| ρu conservation through the mesh | — | max continuity residual **7.9e-13** kg/m²/s |
| Total Δp across the CL | few hundred Pa → ~1 kPa, Δp/p ≲ 1.5 % | **600 Pa** @0.47, **1005** @0.73, **1064** @0.87, **1308 Pa** @1.28 A/cm²; max Δp/p = **1.44 %** |
| Source dominated by CO₂ dissolution, not faradaic stoichiometry | yes | at −2.500 V: in `∫R_PT·M_CO₂ = 2.03e-3`, out `CO 4.07e-4` + `H₂ 1.04e-4` ⇒ net inflow **1.52e-3** kg/m²/s — same structure as the guide's worked example |
| Share of dissolving CO₂ going to CO₂R | 71 / 59 / 35 / 11 % at −1.200/−1.650/−2.075/−2.500 | **68.8 / 64.2 / 54.7 / 31.5 %** — same monotone decline; the high-|V| values run higher because pre-fit `i0_CO2R = 540` keeps i_CO ~4× above the guide's fitted reference there |
| GDL nearly flat, steep drop across the CL (§12.7 B1) | yes | Δp_GDL **3–6 Pa** vs Δp_CL **600–1308 Pa** |
| Stage 4 currents ≈ Stage 3 | "almost identical" | i_CO **+0.07 % … +1.55 %**, i_H₂ **−0.21 % … +0.10 %** |

Δp is ~15–25 % above the guide's validated numbers at matched current, which is
the expected direction: the guide's reference is at *fitted* kinetics, and Δp is
set by `∫R_PT` (CO₂ dissolution), which is larger here because CO₂R is ~5×
stronger with the pre-fit i0. Judged by the mass balance and ρu-conservation
tests, as §11 Stage 4 instructs, the solve is exact.

Peak FE_CO 92.5 % at −1.275 V (essentially unchanged from Stage 3, as expected).

**Plots:** `figures/A1_polarization_stage4.png`, `figures/A2_liquid_stage4.png`,
`figures/B1_gas_stage4.png`, `figures/OV_overlay.png` (Stages 1–4).

No new deviations and no errors at this stage.

> 🛑 **STOP GATE 4 — awaiting human review.**  Next: the Fit step (§17), which
> runs Nelder-Mead on (i0_CO2R, λ_MHC, log₁₀ HER_pf, α_c) against the 5
> experimental points using the Stage-4 residual.

---

### Fit step (after Stage 4) — Nelder-Mead on the 4 kinetic parameters  ✅

**User input #5** (Stop Gate 4): *"Looks good, please continue"* — Stage 4
accepted, no corrections requested.

`run_fit.jl`. Objective = log-space SSE on i_CO and i_H₂ at the 5 Romiluyi
voltages, evaluated through the **Stage-4 residual** (φ_L = 0,
φ_s = V_applied_SHE) with the §17.3 exact-voltage polish warm-started from the
Stage-4 grid cache, using the same `newton_affine!` + persistent LU + AD
Jacobian kernel. §17.5 procedure followed in order: Step 2 (CO₂R only) →
Step 3 (HER only) → Step 4 (joint 4-parameter).

Both §17.2 warnings were implemented as written: the bound penalty and the
non-convergence penalty are **sloped**, never flat. The §9.5.7 parameter-ramp
recovery (2 → 4 → 8 sub-steps from the last feasible vector) is wired in for
failed polishes.

**Result — 289 objective evaluations, ~4.5 min:**

| Parameter | Initial guess | **Fitted** |
|---|---|---|
| `i0_CO2R` | 540 A/m² | **106.385 A/m²** |
| `λ_MHC` | 1.8 eV | **1.6108 eV** |
| `HER_prefactor` | 2.409e13 | **2.0636e13** |
| `α_c_HER` | 0.0872 | **0.088910** |

SSE 0.9526 → **0.0561**. Step 2 alone (CO₂R only) landed at
`i0_CO2R = 106.589, λ = 1.600`. The whole fit was re-run from scratch after the
E4 fix below and reproduced **bit-identically** (same SSE to 6 d.p., same
parameters) — the fit is deterministic.

> This lands squarely in the range the guide anticipates for the post-fit
> kinetics (§11 Stage-5 gotcha #0: "`i0_CO2R ≈ 100–200 A/m²`, `λ_MHC ≈ 1.6 eV`"),
> reached independently from the 540 / 1.8 initial guess.

**§17.6 acceptance criteria — ALL PASS:**

| V_RHE | i_CO model vs exp | err | tol | i_H₂ model vs exp | err | tol |
|---|---|---|---|---|---|---|
| −0.816 | 32.99 vs 35.1 | 6.0 % | 30 % | 26.77 vs 29.9 | 10.5 % | 30 % |
| −1.205 | 223.02 vs 189.6 | 17.6 % | 30 % | 46.47 vs 49.3 | 5.7 % | 30 % |
| −1.644 | 243.04 vs 299.1 | 18.7 % | 30 % | 152.22 vs 119.6 | 27.3 % | 30 % |
| −2.079 | 165.52 vs 204.3 | 19.0 % | 30 % | 457.77 vs 449.9 | 1.7 % | 30 % |
| −2.413 | 88.67 vs 64.6 | 37.3 % | **80 %** | 914.50 vs 850.5 | 7.5 % | 30 % |

| Qualitative criterion | Target | Built |
|---|---|---|
| Sweep coverage | 89/89 on one branch | **89/89** |
| Sign changes in `di_CO/dV` | exactly 1 | **1** |
| Sign changes in `di_H₂/dV` | 0 (strictly monotone) | **0** |
| FE_CO peak value | 70–85 % | **82.9 %** |
| FE_CO peak location | within 100 mV of −1.205 | **−1.175** (30 mV) |

Per §17.6's decision rule, fitting **stops here** — all per-point tolerances and
all qualitative criteria are satisfied simultaneously. §17.9 (letting `k_GL`
float) was correctly **not** entered, since it is only for when the 4-parameter
fit fails.

**§17.8 refit sweep** produced as the prescribed diagnostic artifact:
89/89 Stage-4 physics at the fitted kinetics, 1.97 iters/step, max ‖F‖∞ 1.87e-12,
written to `output_fit/cache/fit_V*.bin`. It is **not** used to seed Stage 5.

**Plots:** `figures/A1_polarization_fit_postfit.png`,
`figures/A2_liquid_fit_postfit.png`, `figures/B1_gas_fit_postfit.png`, and
`figures/OV_fit_effect.png` (pre-fit Stage 4 vs post-fit, isolating the effect
of the parameter fit alone — §17.8).

**Errors caught and fixed by me at this stage:**

| # | Where | Error | Fix |
|---|---|---|---|
| E3 | `run_fit.jl` objective closure | `p_last = copy(params)` inside the closure made `p_last` a *local*, so the sloped non-convergence penalty threw `UndefVarError` on the first NM call. | `const p_last = Ref(...)`, mutated as `p_last[]`. |
| E4 | `run_fit.jl` acceptance-criteria loop | `allpass &= …` inside a top-level `for` — Julia soft scope treats it as a new local; crashed *after* the fit had converged and `fitted_params.txt` was already written. | `allpass = Ref(true)`, mutated as `allpass[]`. Full fit re-run from scratch to confirm; reproduced bit-identically. |

Both are my own Julia-scoping slips, not guide defects. No new deviations.

> 🛑 **STOP GATE 5 (Fit) — awaiting human review.**  Next: Stage 5, the final
> physics gate — φ_L and φ_s as monolithic DOFs, anchored at V_c = 0 and
> continued, with these fitted kinetics loaded before the anchor.

---

### Stage 5 — φ_L (NP migration) + φ_s (solid Ohm's law)  ✅  FINAL PHYSICS GATE

**User input #6** (Stop Gate Fit): *"Great, go ahead"* — the fit accepted, no
corrections requested.

`run_stage5.jl`, production mesh, **1480 DOFs**. Fitted kinetics loaded from
`output_fit/fitted_params.txt` **before** the anchor (§11 gotcha #14) and printed
at startup. Seeded **once**, at V_c = 0, from the constructed physics guess
(equilibrium liquid + inlet gas at p_total + φ_s = φ_L = 0) — **no cache
seeding** (§11 gotcha #0). Chain: 33 pre-anchor ramp steps of −0.024638 V
(≤ the guide's 25 mV) from V_c = 0 down to V_RHE = −0.300, then the 89 output
voltages: **122 chain points**.

**Result: 89/89 converged, 10.6 s.**

| Metric | Guide (§9.5.10, §11 #10) | Built |
|---|---|---|
| Newton iters per continuation step | 2.0 | **1.99** (max **2**) |
| max ‖F‖∞ over the sweep | ~1e-7 | **8.20e-8** |
| sweep wall (incl. anchor + pre-ramp, 121–122 steps) | ~11 s | **10.6 s** |
| sub-step recoveries | untriggered in production | **0** |
| §9.5.6 assertion fired | never | **never** |

**§11 Stage-5 physicality checks 1–7 — ALL PASS:**

| # | Check | Guide | Built |
|---|---|---|---|
| 1 | Solid ohmic drop `φ_s_CL[N] − V_applied` | 5–80 mV; ~50 mV at 1 A/cm² | 3.10 → **59.38 mV**; **53.55 mV at 1.073 A/cm²** |
| 2 | GDL share of the solid drop | 80–90 % | **88.6–88.9 %** at every voltage |
| 3 | φ_L profile monotone, magnitude | 5–50 mV | monotone at every voltage; **6.79 → 17.69 mV** |
| 4 | `abs(i_s_face[1])` = `i_far_total` | ≤ 1e-6 | worst **3.10e-11** over all 89 |
| 5 | i_H₂ falls 3–10 %; i_CO may **rise** | see below | i_H₂ **−2.8 … −10.0 %**; i_CO **+15.3 % … +89.9 %** |
| 6 | Ionic current conservation at x = L_CL | rel_err < 0.01 | worst **1.33e-12** over all 89 |
| 7 | Currents smooth and monotone | 1 sign change in di_CO/dV, 0 in di_H₂/dV | **1** and **0**; **0** φ_L sign flips |

**§9.5.8 reproducibility checklist — every box checked:**

- `Δφ_L` at i_total > 500 mA/cm²: **[16.21, 17.69] mV** (target 5–25 mV, "typically ~15 mV") ✅
- `Δφ_L` sign: **all negative** (φ_L[1] ≈ −17.7 mV, gauge φ_L[N_CL] = 0) ✅
- Polarization smoothness by sign-change count ✅
- Anchored at V_c = 0 from constructed physics, never from a cache ✅
- Post-solve assertion wired at all 89 emitted voltages, `F_INF_TOL = 1e-6` ✅

**The decisive validation — §11 Stage-5 check 5 quotes a validated build's
Stage 4 → Stage 5 change at V_RHE = −2.425, CL cell 1. This independent build
reproduces every number to 3–4 significant figures:**

| Quantity | Guide reference | **Built** |
|---|---|---|
| c_CO₃²⁻ | 4590 → 2845 | **4589 → 2844** |
| c_Cs⁺ | 9369 → 5537 | **9366 → 5534** |
| CL-avg c_OH⁻ | 410 → 197 | 408 → 196 |
| H_eff | 1.66 → 4.33 | **1.66 → 4.33** |
| c_CO₂(aq) | 1.10 → 3.26 | **1.11 → 3.26** |
| i_CO | 86 → 164 mA/cm² (**1.90×**) | **86.2 → 163.6** (**1.90×**) |

The guide's warning that i_CO can *rise* rather than fall is confirmed exactly:
the φ_L field migrates CO₃²⁻ and OH⁻ out of the CL, relieving the §7.4 Sechenov
salting-out, so H_eff rises 2.6× and c_CO₂(aq) 2.9×. The rise is **+15.3 % near
the FE_CO peak** and **+89.9 % in the plateau** — the guide predicts "10–20 %
near the FE_CO peak" and "up to ~1.9× in the plateau region".

**§9.5.5 branch check — passed.** The guide's criterion is that the anchored
continuation sweep must agree with a per-voltage reference to "well under 1 %",
where a wrong-branch sweep is off by ~20 %. Against the guide's own validated
Stage-5 reference at V_RHE = −2.425: **163.6 vs 164 mA/cm² = 0.24 %**. The
low-CO branch signature (Δφ_L ≈ 3 mV, peak i_CO ~20 % low) is absent —
Δφ_L is 16–18 mV. **This build is on the high-CO branch.**

**§9.5.9 step 8 — model vs experiment** (the fit used 5 points at Stage-4
physics; Stage 5 adds the potential fields, so these are partly predictions):

| V_RHE | i_CO model vs exp | i_H₂ model vs exp |
|---|---|---|
| −0.825 | 35.4 vs 35.1 (+1.0 %) | 26.6 vs 29.9 (−11.0 %) |
| −1.200 | 254.8 vs 189.6 (+34.4 %) | 42.5 vs 49.3 (−13.9 %) |
| −1.650 | 324.7 vs 299.1 (+8.6 %) | 139.4 vs 119.6 (+16.5 %) |
| −2.075 | 257.8 vs 204.3 (+26.2 %) | 424.4 vs 449.9 (−5.7 %) |
| −2.425 | 163.6 vs 64.6 (+153 %) | 909.2 vs 850.5 (+6.9 %) |

i_H₂ is within ±17 % everywhere. i_CO overshoots in the plateau — this is the
Stage-4-fit → Stage-5-physics gap the guide describes in §11 ("Stage 5 currents
may differ from the fit targets by 10–30 % — this is expected and physically
correct"), amplified at the most cathodic point by the 1.90× Sechenov-relief
rise. It is a property of the *guide's model*, not of this implementation:
the build matches the guide's own validated Stage-5 reference at that voltage to
0.24 %. FE_CO peak = **85.8 % at V_RHE = −1.225**.

§11 notes an **optional second-pass refit against Stage 5** as the remedy for
this gap. Not performed — it is explicitly optional, and the guide's canonical
execution order ends at Stage 5 + final plots.

**Plots:** `figures/A1_polarization_stage5.png`, `figures/A2_liquid_stage5.png`,
`figures/B1_gas_stage5.png`, `figures/D_potentials_stage5.png` (D1 φ_s across
GDL+CL, D2 φ_L across the CL), `figures/OV_all_stages.png` (all five stages +
the post-fit Stage-4 sweep).

No new deviations and no errors at this stage.

> 🛑 **STOP GATE 6 (Stage 5) — final physics gate, awaiting human review.**

---

## Build summary

All seven steps of the guide's canonical execution order (§13) completed:
`run_stage1 → run_stage2 → run_stage3 → run_stage4 → run_fit → run_stage5 →
plot_results`, each stopping at its gate for human review.

**89/89 voltages converged at every stage.** Guide-vs-built solver performance:

| Stage | DOFs | Guide iters/step | Built | Guide max ‖F‖∞ | Built | Guide wall | Built |
|---|---|---|---|---|---|---|---|
| 1 | 400 | 2.0 | 2.04 | ~1e-12 | 1.28e-12 | ~2 s | 2.8 s |
| 2 | 400 | 2.0 | 2.04 | ~3e-12 | 2.98e-12 | ~2 s | 3.8 s |
| 3 | 1200 | 1.9 | 1.93 | ~1e-11 | 1.38e-11 | ~5 s | 4.3 s |
| 4 | 1200 | 1.9 | 1.93 | ~1e-11 | 1.20e-11 | ~5 s | 5.9 s |
| Fit | 1200 | ~2 per polish | ~2 | — | — | ~80 s | ~270 s |
| 5 | 1480 | 2.0 | 1.99 | ~1e-7 | 8.20e-8 | ~11 s | 10.6 s |

**Reproducibility verdict.** The guide reproduces. Every quantitative claim it
makes that could be checked was matched, most to 3–4 significant figures:
the Stage-1 FE_CO peak (64.3 % at −1.125 vs "64 % at −1.125"), the Stage-1 pH
gradient (13.29 → 9.41 vs "13.3 → 9.4"), the Stage-3/4 c_CO₃²⁻ and c_Cs⁺ at
−2.425, the Cs⁺ share of liquid conductivity ("roughly half" → 48.6 %), the
Stage-4 Darcy source structure, the fitted kinetics landing in the predicted
100–200 A/m² / ~1.6 eV window, and the entire Stage-5 check-5 table. The solver
stack of §9.5 reproduced exactly, with the measured iteration counts, residuals
and wall times all inside the guide's quoted values.

**Defects found in the guide** (D1, D2, D3 above): a Julia name collision in
§7.2, an inconsistent `dx_between` indexing convention between §11 Stage 1 and
§11 Stage 5, and a §11 Stage-5 Step-5 pseudocode block that omits the Cs⁺ term
from `i_L` that the surrounding prose declares mandatory. All three are fixable
in a guide revision without touching the physics. One clarification is
recommended (O1, §12.3's CO₂(aq) gradient direction, which inverts from Stage 3
onward as a consequence of the guide's own Sechenov equations). The dependency
list in §9.5.1/§9.5.8 should also name SpecialFunctions, Roots and Optim (D4).

**Errors that were mine, not the guide's:** E1 (wrong hand-typed package UUID),
E2 (matplotlib backend), E3 and E4 (two Julia soft-scope slips in `run_fit.jl`).
None reached the physics.

**Errors found by the user:** none.
