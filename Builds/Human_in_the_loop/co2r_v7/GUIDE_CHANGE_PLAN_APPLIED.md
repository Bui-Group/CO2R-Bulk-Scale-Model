# Guide change plan — **APPLIED IN PLACE TO V7** (now v7.1)

> **Status: all items below were applied directly to
> `Implementation_Guide/CO2R_BULK_MODEL_IMPLEMENTATION_GUIDE.md`** on
> 2026-08-28 at the user's direction ("you can just edit v7, no need to make a
> v8"). 26 targeted edits, each applied by exact-match assertion; a pre-edit
> backup is in the session scratchpad. Verified afterwards that all 14 stale
> strings are gone and all 14 new anchors are present. The guide footer now
> reads *v7.1*.
>
> **Two claims were corrected during implementation**, after re-measuring rather
> than trusting the draft plan:
>
> 1. **P1-5 (c_Cs range).** The draft attributed the elevated `c_Cs` to the
>    pre-fit kinetics. Re-measurement shows that is wrong: pre-fit vs post-fit
>    Stage-4 differ by only ~4 % (9714 vs 9369 mol/m³ at V_RHE = −2.425). The
>    real contrast is **Stage 4 vs Stage 5** — 9369 → 5537 at the same voltage —
>    i.e. the same migration effect as P1-1. The guide text now says that.
> 2. **P1-3 (CL pressure drop).** The draft mixed pre-fit and post-fit numbers
>    and claimed "~1/3 of the dissolving CO₂ goes to CO₂R". Re-measured at the
>    fitted kinetics, that share runs 71 % → 11 % from V_RHE = −1.200 to −2.500,
>    and the faradaic-only estimate is not merely ~11× low but **24× low and of
>    the opposite sign** at −2.500 (HER makes H₂ mass without consuming CO₂).
>    The guide text now carries the measured numbers.
>
> Everything else went in as drafted.

---

## Original plan (V7 → V8)

Derived entirely from building `Builds/Human_in_the_loop/co2r_v7` from the V7 guide with no
other reference. Every item below is backed by something observed in that build;
line numbers refer to the V7 file.

**Headline.** The guide's *numerics* reproduced essentially perfectly — solver
metrics, DOF counts, derived constants, equilibrium values and branch
diagnostics all matched, most to the digit. The defects cluster in two places:
**four code snippets that don't compile or use the wrong voltage frame**, and **a
set of physicality-check ranges carried over from older builds that a correct
implementation now fails.** The second group is the more insidious: a fresh agent
hits a passing model, reads a failing check, and goes hunting for a bug that
isn't there.

Priorities: **P0** blocks or silently corrupts a build · **P1** sends a correct
build chasing a phantom bug · **P2** genuine ambiguity, two defensible readings ·
**P3** nits.

---

## P0 — Must fix: breaks or silently corrupts a literal implementation

### P0-1 · §17.3 and §17.4 double-convert the voltage frame (lines 2562, 2596)

Both fit snippets do

```julia
V_cathode = V_rhe - V_RHE_to_SHE_offset
res! = (F_, u_) -> residual_stage4!(F_, u_, V_cathode, cl_mesh, gdl_mesh, c_eq_vec)
```

but every `residual_stageN!` in the guide converts internally
(`V_applied_SHE = V_cathode - V_RHE_to_SHE_offset`, lines 1665 and 1977), and
§11 gotcha #9a states explicitly that the argument is in V vs RHE@pH 8. Passing
an already-converted value subtracts the offset **twice** — a 513 mV error at
every fit point. This is precisely the failure mode gotcha #9a warns about,
present in the guide's own code.

*Fix:* delete the conversion line in both snippets and pass `V_rhe` directly.
Keep the separate `V_applied_SHE = V_rhe - V_RHE_to_SHE_offset` used for the
current extractors — that one is correct. Add one line to §15.3 stating the
convention once, unambiguously: **residual and anchor-builder arguments are RHE;
current extractors and φ_s DOF values are SHE.**

### P0-2 · §9.6 `solve_buffer_equilibrium` does not parse (line 1348)

`using Roots` sits inside the function body. Julia rejects this:
`syntax: "using" expression not at top level` (verified). The guide's single
most important initialisation routine will not compile as printed.

*Fix:* move `using Roots` to the top of the §13 `chemistry.jl` listing.

### P0-3 · §9.5.3 weight formula is 0/0 at the Stage-5 anchor

`w_j = |u_j| + afrac · rms(|u| over j's block)` divides by zero whenever an
entire DOF block is identically zero. The guide's own Stage-5 anchor recipe
(§9.5.5) sets `φ_s = 0` on GDL+CL and `φ_L = 0` on the CL at V_c = 0, so **both
potential blocks are exactly zero on the first iteration** and `‖d‖_w` returns
`Inf`/`NaN` immediately. The prose acknowledges "a DOF passing through zero" but
only handles the single-DOF case, not a whole zero block.

*Fix:* add to §9.5.3 step 1: *"If a block's rms is exactly zero (the φ_s and φ_L
blocks at the Stage-5 anchor guess), fall back to a unit weight for that block.
The block is non-zero from the first Newton step onward, so this affects only
iteration 1."* Worth also stating that this is the sole case in which a weight is
not iterate-derived.

### P0-4 · §17.2's flat `1e6` bound penalty stalls Nelder-Mead

§17.2 prescribes "enforced inside the objective by returning 1e6 if violated".
Implemented literally, **§17.5 Step 3 does nothing and reports no error.** From
the Step-3 start `[log10 HER_pf, α_c] = [13.3818, 0.0872]`, Optim's
`AffineSimplexer(a=0.025, b=0.5)` places the two non-base vertices at
`[20.098, 0.0872]` (above the `log10 HER_pf ≤ 18` bound) and `[13.3818, 0.1558]`
(α_c large enough that the polish cannot converge). Both return **exactly**
`1e6`; with two vertices tied, Optim returns at **iteration 0** with
`g_converged = true` and `minimum = 1e6`, while `minimizer` is the good vertex.

Reproducer in this build: `test_nm_penalty.jl`. The fix is not about magnitude —
sloping the penalty works at every magnitude tested:

| flat `1e6` | sloped `PEN·(1 + Σ_j (violation_j / box_j)²)` |
|---|---|
| **0 iters, min = 1e6** | 1e6 → 28 iters, min = 0.05665 |
| | 1e3 / 1e1 / 1e0 → 25 / 21 / 20 iters, same minimum |

*Fix:* replace the §17.2 sentence with a sloped-penalty formula, and slope the
§17.4 non-convergence return the same way (e.g. scaled by distance from the last
feasible parameter vector) so two failing vertices never tie. Add a one-line
warning: *"A constant penalty plateau can stall Nelder-Mead at iteration 0 —
always make the penalty vary with the violation."*

---

## P1 — Physicality-check ranges inherited from older builds

These are the ones flagged in review as stale. Each one a correct build now
fails. Recommend re-deriving every numeric range in §11 and §12 against a
current reference run and, where a range depends on the fit, saying so.

### P1-1 · §11 Stage 5 check 5 — "CO/H₂ currents drop ~10–30 % vs Stage 4"

**Measured (vs the Stage-4 sweep at the *fitted* kinetics):**

| V_RHE | i_CO | i_H₂ |
|---|---|---|
| −1.200 | 221.1 → 254.8 (**+15 %**) | 46.0 → 42.5 (−7.6 %) |
| −1.650 | 242.4 → 324.9 (**+34 %**) | 154.9 → 139.5 (−10.0 %) |
| −2.075 | 166.4 → 257.9 (**+55 %**) | 453.9 → 424.7 (−6.4 %) |
| −2.425 | 86.1 → 163.6 (**+90 %**) | 936.1 → 909.8 (−2.8 %) |

The driving-force argument in the guide is correct and confirmed — η_CO2R at
−2.425 goes from −1.9912 to −1.9483, 43 mV less cathodic — and **i_H₂ does drop**.
But a second, larger effect is missing from the guide's account: NP migration
sweeps anions (CO₃²⁻ especially, z = −2) out of the CL, relieving the Sechenov
salting-out:

| V_RHE = −2.425, CL cell 1 | Stage 4 | Stage 5 |
|---|---|---|
| c_CO₃²⁻ | 4590 | 2845 mol/m³ (−38 %) |
| c_Cs⁺ | 9369 | 5537 mol/m³ (−41 %) |
| ⟨c_OH⁻⟩ over the CL | 410 | 197 mol/m³ (−52 %) |
| H_eff | 1.66 | 4.33 mol/(m³·atm) (**2.6×**) |
| c_CO₂(aq) | 1.10 | 3.26 mol/m³ (**2.9×**) |

HER has no CO₂ dependence, so it sees only the lost driving force. CO₂R sees 2.9×
the CO₂ supply against a 43 mV loss, and rises. Migration direction verified
independently (φ_L runs −17 mV → 0 across the CL) and both charge balances close
to 1e-11.

*Proposed replacement:* **"5. i_H₂ falls 3–10 % vs Stage 4 as the ~50–70 mV of
combined φ_s + φ_L drop is subtracted from the driving force. i_CO may fall or
**rise**: the same φ_L field that costs driving force also migrates CO₃²⁻ and
OH⁻ out of the CL, which relieves the Sechenov salting-out and can raise
c_CO₂(aq) by 2–3×. Where the carbonate/Sechenov limitation is active, expect a
plateau-region i_CO rise of up to ~1.9×. Diagnose by comparing ⟨c_CO₃²⁻⟩ and
H_eff between the two stages, not by assuming a drop."** Note the guide already
anticipates the magnitude elsewhere ("may differ from the fit targets by
10–30 % … a second-pass refit against Stage 5 can be done") — the two statements
should be reconciled.

### P1-2 · §11 Stage 1 check 4 — "CO selectivity 80–95 % near onset"

Stage 1 runs with the *initial-guess* kinetics, where `HER_prefactor = 2.409e13`
and `α_c = 0.0872` put 10.4 mA/cm² of H₂ against 0.002 mA/cm² of CO at
V_RHE = −0.300. Measured: **0–11 % near onset, FE_CO peak 64.3 %**. Hand-evaluating
§4.2 at pH 8.23 reproduces the model number exactly, so this is the parameter
set, not the implementation. §12.2's 80 % figure is explicitly for the fitted
model.

*Proposed replacement:* **"4. CO selectivity with the pre-fit kinetics is low —
expect FE_CO ≲ 15 % near onset and a peak around 60–70 %. The initial-guess HER
parameters over-predict H₂ at mild potentials by roughly an order of magnitude;
80 %+ selectivity only appears after the §17 fit. Do not treat a low Stage-1
FE_CO as a bug."** Same caveat belongs on the Stage 2/3/4 checks that quote
post-fit numbers.

### P1-3 · §11 Stage 4 check — "total pressure drop across CL ~1–100 Pa"

Measured **296–1159 Pa** post-fit (677–1453 Pa pre-fit), while the Darcy solve
itself is exact (mass balance 1.9e-10, ρu conserved to 1.2e-5 through the
source-free GDL, min p = 0.986 atm). The discrepancy is fully accounted for: at
V_RHE = −2.5 the inlet mass flux is 1.52e-3 kg/m²/s, of which `∫R_PT·M_CO₂` =
2.0e-3 in against `i_CO/2F·M_CO` = 4.1e-4 and `i_H₂/2F·M_H₂` = 1.0e-4 out. Only
~1/3 of the dissolving CO₂ goes to CO₂R; the rest feeds `CO₂ + OH⁻ → HCO₃⁻`.
Recomputing the net source from faradaic stoichiometry **alone** gives
Δp_CL ≈ 123 Pa — i.e. the guide's range appears to have been derived without the
carbonation sink.

*Proposed replacement:* **"At i ≈ 1 A/cm², Δp across the CL is a few hundred Pa
to ~1 kPa (Δp/p ≲ 1.5 %). The Darcy flux is set by the *net* gas mass source,
which is dominated by CO₂ dissolving to feed carbonate formation, not by faradaic
stoichiometry — typically 3× larger. If you compute only i_CO/2F·M_CO₂ you will
predict ~100 Pa and wrongly conclude the solve is wrong."**

### P1-4 · §12.3 states the pH gradient backwards

"pH(x) … should increase toward the electrolyte-channel boundary (more
alkaline)." With the guide's own BCs — ions no-flux at x = 0, Sherwood drain at
x = L_CL — OH⁻ is produced throughout the CL and can only leave at the right, so
pH must be **highest at the GDL side**. Measured Stage 1: pH 13.29 at x = 0 →
9.41 at x = L_CL. §11's own Stage-2 check agrees ("for OH⁻ … it's positive
(leaving the CL)").

*Fix:* invert the sentence in §12.3 and cross-reference the Stage-2 flux check.

### P1-5 · §3.1's "c_Cs reaches 2000–4500 mol/m³ in the reaction zone"

Measured at the GDL-side CL cell: **5537** mol/m³ at Stage 5 / V_RHE = −2.425
(post-fit), and **10 134** at Stage 3/4 with the pre-fit kinetics.

*Fix:* widen to ~2000–6000 at the fitted kinetics and state the voltage it
applies at; add that with pre-fit kinetics it can roughly double, which is
expected rather than a sign of error.

### P1-6 · §9.5.7 overstates what backtracking recovers

"Backtracking … does fire, and recovers, on wild NM trial kinetics in the Fit
step." It did not here: at α_c = 0.1308 (a routine 1.5× NM probe) the corrector
reported `damping_stalled` at the two most cathodic voltages at **both**
`alpha_init = 1.0` and `alpha_init = 1e-2`.

*Fix:* soften to *"backtracking recovers many wild NM trials but not all. When a
direct warm-started polish fails, ramp the kinetic parameters from the last
feasible vector to the trial vector in 2 → 4 → 8 sub-steps, re-solving at each —
the §9.5 continuation idea applied along the parameter direction. Trials that
still fail are genuinely extreme (α_c ≈ 0.13 implies 29–80 A/cm²) and should be
rejected."*

---

## P2 — Ambiguities worth closing

### P2-1 · Stage 3 pressure equation

§11 Stage 4 calls Stage 3's pressure DOF a "placeholder"; §12.7 B1 calls it
"approximately uniform (placeholder)" — which reads as a solved-but-flat Darcy
field. Two defensible implementations. Resolved in review: **pin it**
(`F[p] = p − p_total`, exactly uniform) and add the Darcy balance in Stage 4.

*Fix:* say so explicitly in §11 Stage 3, and change §12.7 B1's wording from
"approximately uniform" to "exactly uniform".

### P2-2 · Sechenov `c_Cs` — which value?

§7.4's worked example uses `c_Cs ≈ 1000` and lands on H_eff ≈ 13.8, but the code
path §11 specifies (`sechenov_H_CO2(c_ix)`, taking only the 5-species vector) can
only use the electroneutrality value, `c_Cs = c_OH + c_HCO3 + 2c_CO3 − c_H −
C_fix·ε_I` = **570** at bulk. That gives H_eff = 14.53 and a GDL-face Henry value
of 12.8 rather than the 12.2 quoted in §2 and §7.4.

*Fix:* state which `c_Cs` the function uses, and rerun the worked example and the
§2 "≈12.2 mol/m³" figure with it. (Recommend keeping the electroneutrality value
— it is the only one the model actually has.)

### P2-3 · Gas diffusivity at the GDL/CL interface is unspecified

§8.7 gives `D_eff_GDL = ε_GDL^1.5 D_bin` and `D_eff_CL = ε_G^1.5 D_bin`; §11
Stage 4 specifies a harmonic mean for κ at that face but says nothing about D.

*Fix:* specify the harmonic mean for D too (what this build used), for symmetry
with κ.

### P2-4 · Mesh grading parameter `q` is never defined

§9.1's sketch gives `r = q^(1/(N_CL-1))` and says "q > 1: cells get larger toward
x=0", but never states what `q` *is*. It is only unambiguous if you notice that
`r^(N−1) = q`, making q the **ratio of the largest to the smallest cell** (1.3 ⇒
a mild 30 % spread, not a strong grading).

*Fix:* one sentence: *"`q` is the ratio of largest to smallest cell width; q = 1.3
means the coarsest cell is 30 % wider than the finest."*

---

## P3 — Nits

* **§11 gotcha #9a says "pH 8.68" twice** while `V_RHE_to_SHE_offset = 0.51305`
  is `0.06413 × 8.0`. At pH 8.68 the offset would be 0.5566 V. §15.2 correctly
  uses pH = 8. Fix the two occurrences to 8.
* **§17.4 calls `compute_i_CO(u, V_applied_SHE)` (2 args, line 2613) but defines
  it with 3 (line 2631).** Add `cl_mesh` at the call site.
* **§9.6 "the residual at this initial guess is identically zero"** holds only at
  `α_buf = α_elec = 0`; in the production path (kinetics on at −0.300 V) the
  anchor residual is ~2e-3 and Newton takes ~5 iterations, which §9.5.9 step 2
  already says. Reword to avoid the contradiction.
* **§12.7 A2/B1 say "at all sweep voltages (e.g. −1.0, −1.5, −2.0, −2.5, −3.0)"** —
  89 curves or 5? Pick one. (This build plots all 89 with a light→dark ramp plus
  a colourbar, which works well.)
* **§9.5.8's polarization-smoothness criterion ("sharp spikes ±5 % at single
  voltages")** is not directly measurable on a raw first-difference: near the
  i_CO peak the differences cross zero and any ratio metric explodes (32.8 % here
  on a demonstrably smooth curve; the known-good Stage-4 sweep shows 22.0 % at
  its own peak). Recommend restating it as **"the count of sign changes in
  di_CO/dV must be exactly 1, and di_H₂/dV must not change sign at all"** — both
  robust and both used successfully in this build.
* **§13 module list** could add `driver.jl` (the shared §9.5.5 continuation
  driver). The guide currently implies each `run_stageN.jl` carries its own copy,
  which invites five diverging drivers.

---

## Suggested order of work

1. **P0-1 → P0-4** first — these four are the difference between a fresh build
   working and a fresh build silently producing wrong numbers. P0-1 and P0-4 in
   particular fail *without any error message*.
2. **Re-derive every numeric range in §11 and §12 against a current reference
   run** (P1-1 … P1-5), and tag each range with the kinetics it assumes
   (pre-fit vs post-fit). Most of the P1 group exists because ranges from an
   older build were never re-measured; a standing rule that check ranges are
   regenerated whenever the reference build changes would prevent recurrence.
3. **P2** — four short clarifying sentences, each removing a real fork in the
   road.
4. **P3** — copy-editing pass.

## What did *not* need changing

Worth recording, since it is most of the guide: §§1–6 (domain, DOF layout,
governing equations, kinetics, buffer chemistry, gas transport), §8 (every
parameter table — all 13 derived constants I audited matched to the digit),
§9.1–9.4, **all of §9.5 (the solver stack — reproduced exactly, all eight §9.5.9
tests passing on the first attempt)**, §10 (caching), §15 (potential
referencing), §16 (experimental data). The §9.5 stack in particular did what it
claims: 89/89 at every stage, ~2 Newton iterations per step, zero sub-step
recoveries, and the Stage-5 anchor-at-V_c=0 continuation landed on the high-CO
branch on the first try.
