# LLM-Assisted Generation of Transparent, Open-Source Multiphysics Models of Electrochemical Devices

Code and specification release accompanying the manuscript of the same title
(Castro, Schuchert, McCluskey, Lees, Bui). This repository contains the
machine-readable implementation guide used as the agent harness, the Julia
models built from it, and the build dialogues recorded during each build.

> Three substitutions were applied throughout, including inside quoted prompts
> in the build dialogues: researcher names were replaced with the researcher
> letters and build labels used in this repository; working-tree file paths and
> internal folder names with the location each item occupies here (the code of
> the intermediate builds is not included, so their code paths name folders
> that are not present); and references to the agents' project-memory files
> with a bracketed description. The planted-error study records were also
> edited after a re-run; see *Planted-error study* below. Stage 5 of the nine
> builds in `Builds/` was re-run after one correction to the guide; see
> *Stage-5 correction* below.

## Layout

| Folder | Manuscript section | Contents |
|---|---|---|
| `Implementation_Guide/` | Methods: *The implementation guide*; Supplementary Table 12 | Guide v7.1 as Markdown and PDF, the version the builds in `Builds/` were made from (`co2r_v7` from its predecessor, v7.0), with one correction to its Stage-5 section made on 2026-09-27, after the builds were complete (see *Stage-5 correction* below). The guide's closing revision notes were removed for this release, so the file is shorter than the 2,906 lines the build dialogues report; the same applies to the guide copies in `Planted_Error_Study/`. |
| `Builds/Human_in_the_loop/` | Results: *Validation against COMSOL and experiment*, Fig. 2 (`co2r_v7_2`) and Fig. 3; Supplementary Note 8 (builds 1–3) | Three independent builds (`co2r_v7`, `co2r_v7_2`, `co2r_v7_3`) with a human reviewer at each stage gate. `co2r_v7` was built from guide v7.0 and its findings became v7.1. v7.0 is not included; `co2r_v7/GUIDE_CHANGE_PLAN_APPLIED.md` lists the changes, with line numbers that refer to v7.0. |
| `Builds/Opus_autonomous/` | Results: *Fully autonomous builds*; Supplementary Note 8 (builds 4–6 = `Build_1` to `Build_3`) | Three Claude Opus 5 builds with stop gates bypassed and no human input. |
| `Builds/Sonnet_autonomous/` | Results: *Fully autonomous builds*; Supplementary Note 8 (builds 7–9 = `Build_1` to `Build_3`) | Three Claude Sonnet 5 builds, same protocol. |
| `Planted_Error_Study/` | Results: *Error handling with multiphysics agents*, Fig. 5; Supplementary Note 6 | `BaseCase` is the unmodified control. `Case_1` to `Case_4` each carry one planted defect. Each folder holds its guide (`BaseCase.md`, `Case_N_guide.md`) beside the build. These builds were re-run before release; see *Planted-error study* below. |
| `COMSOL_Reference/` | Fig. 3; Methods: *Solve-time benchmarking*; Supplementary Note 3 | The COMSOL reference model (`.mph`) and plain-text exports from it (current densities, fluxes, species and rate profiles over the 61-potential sweep). |
| `Intermediate_Build_Dialogues/` | Results: *Diagnostic transparency*; Supplementary Note 7 | Build dialogues of the six March 2026 builds against the first-generation guide (Builds A, B1, B2, C1, C2, D), dialogues only. |

## What is in each build folder

Every build follows the module structure prescribed by the guide (section 13):

- `params.jl`, `mesh.jl`, `chemistry.jl`, `kinetics.jl`, `transport.jl`,
  `gas_transport.jl`, `assembly.jl`, solver files, and `run_stage{1..5}.jl`
  plus `run_fit.jl` drivers. Exact file names vary slightly between builds
  because the agent chose them.
- `Project.toml` / `Manifest.toml` pinning the Julia environment, in most
  builds. Builds without them (`Opus_autonomous/Build_3` and the planted-error
  cases other than `BaseCase`) ran in a shared Julia 1.11 environment; the
  packages they use are named in their `BUILD_DIALOGUE.md`, but their versions
  were not recorded. `Opus_autonomous/Build_3` reproduces its Stage-5 outputs
  byte for byte in the environment pinned by `Human_in_the_loop/co2r_v7_2`.
- `output_stage{1..5}/` and `output_fit/`: converged sweeps, diagnostic plots
  and per-voltage binary solution caches (`*.bin`) written by the drivers.
  Some builds write their CSVs and plots to other folders (for example
  `data/`, `figures/` or `plots/`).
- `BUILD_DIALOGUE.md`: the living build log. It records the operator
  prompts, the errors encountered, the errors the agent fixed on its own,
  and the discrepancies it noticed in the guide.

The autonomous builds keep the model under `Build_N/co2r_bulk/` with the
dialogue beside it. `Planted_Error_Study/Case_3` has the same nested layout.

All six autonomous builds were started with the same prompt, which named the
build folder (shown here for `Opus_autonomous/Build_2`):

> I am doing a study on how well agents can build a model from a well-developed specification alone, without human intervention. The guide Implementation_Guide/CO2R_BULK_MODEL_IMPLEMENTATION_GUIDE.md is self-contained, and you should not read any other document besides this one. Please follow the specification carefully, building in Builds/Opus_autonomous/Build_2. At the stop gates, instead of a human verifying, you will verify your own output numerically and by reading the plots. Do your best, and if you run into any uncertainty or trouble use your best judgement. Track your progress in a build dialogue md file that summarizes your build process, including issues you run into when building the model. Also track any deviations from the guide you take, if any.

## Running a build

From a build folder, with Julia 1.11 or later:

```bash
julia --project=. run_stage1.jl
julia --project=. run_stage2.jl
julia --project=. run_stage3.jl
julia --project=. run_stage4.jl
julia --project=. run_fit.jl
julia --project=. run_stage5.jl
```

Each driver exits after its stage so that the stop gate is a mechanism, not a
policy. The fit warm-starts from the cached Stage-4 solutions; Stage 5 takes
the fitted parameters but starts its sweep from a constructed state, not a
cache. The caches are included, so a stage can be re-run without rebuilding
the earlier ones.

Plotting entry points differ between builds (a Python `plot_results.py` in
most, with build-specific arguments; Julia plotting in some); each build's
`BUILD_DIALOGUE.md` shows how its figures were made. The plotting scripts of
`co2r_v7` and `co2r_v7_3` import a lab plotting-style helper (`customplot.py`)
that is not included, so they do not run as shipped; their figures are
included.

## Stage-5 correction (2026-09-27)

After all nine builds were complete, one defect was found in the Stage-5
specification of guide v7.1, while comparing CO₂(aq) profiles with the COMSOL
reference. The guide carried the Henry's-law CO₂(aq) flux at the GDL/CL face,
the only CO₂ supply of the gas-free Stages 1–2, into Stage 5. There it added a
second gas–liquid interface beside the volumetric phase transfer, and one that
was not conservative, because the CO₂ it supplied was withdrawn from no gas
cell. The COMSOL model has no flux at that face.

The guide's Stage-5 section was corrected in place (Stage 5 note 15; the
version label is unchanged). In every build the face flux was set to zero in
Stage 5 only, in the residual and in any diagnostic that recomputes it; each
such line is marked `# 2026-09-27 post-build change`. Before the change, a
control re-run of each build's unchanged Stage 5 reproduced every published
Stage-5 data file byte for byte. Stage 5 of every build was then re-run with
its own code and fitted kinetics, so the Stage-5 outputs, caches and logs, and
the plots that contain Stage 5, are from the re-run. Builds that had kept no
Stage-5 log gained one, `output_stage5/stage5_rerun_log.txt`. Stages 1–4, the
fits and all other outputs are as built.

The `BUILD_DIALOGUE.md` files were not changed and describe the original run.
The correction lowers each build's Stage-5 i_CO by 0 to 0.34 % (most at the
most cathodic potentials) and changes i_H₂ by at most 0.06 % and FE_CO by at
most 0.04 percentage points, so a Stage-5 number quoted in a dialogue can differ
from the files by that much.

## Planted-error study

The five builds in `Planted_Error_Study/` were first made with the solver of
the earlier guide version their guide copies derive from. Before release each
was ported to the solver of the released guide (v7.1) and every stage was
re-run. The solver sections of the five guide copies were replaced with the
released text, and the build logs were edited to report the re-run's solver
behaviour and numbers; passages and operator prompts that concerned only the
earlier solver were rewritten or removed. The physics specification of each
guide copy, the planted defects, and what each agent flagged, at which stage,
and how it diagnosed it are as originally recorded. Each dialogue opens with an
editor's note listing what was changed in it, and Supplementary Note 6
describes the procedure.

## License

Copyright © 2026 New York University and The University of British Columbia.

The Julia source code in this repository (the `.jl` files) is released under
the MIT License; see `LICENSE`. Other files are not covered by that license.
