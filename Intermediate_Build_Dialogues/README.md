# Build dialogues from intermediate guide versions

Provenance record for Supplementary Note 7 (Supplementary Table 8). These six
builds were made against the first-generation implementation guide (four
stages, no electrostatic coupling; about 1,500 lines at Build A and about 1,900,
including the fitting sections §16 to §18, by the later builds). They are the builds that
surfaced the errors, ambiguities and omissions later written back into the
guide, and three of the four planted defects of the planted-error study
(Cases 1, 3 and 4) first occurred in them. They are a qualitative record, not
a controlled comparison; no quantitative claim in the main text rests on them.

Labels follow the SI: researchers are A to D, builds are ordered by start date.

| Folder | Dates (2026) | Agent | Stages reached | Files |
|---|---|---|---|---|
| `Build_A/` | 20 Mar | Claude Opus 4.6 | 1 to 4 + fit | `BUILD_DIALOGUE.md`, `.pdf` |
| `Build_B1/` | 25 to 26 Mar | Claude Opus 4.6 | 1 to 4 + fit | `BUILD_DIALOGUE.md`, `.pdf` |
| `Build_C1/` | 25 to 28 Mar | Claude Opus 4.6 | 1 to 4 + fit | `BUILD_DIALOGUE.md`, `.pdf` |
| `Build_D/` | 26 to 29 Mar | Claude Sonnet 4.6 | 1 to 3 + fit | `BUILD_DIALOGUE.md`, `.pdf` |
| `Build_C2/` | 27 to 28 Mar | Claude Opus 4.6 | 1 to 3 + fit | `BUILD_DIALOGUE.md`, `.pdf` |
| `Build_B2/` | 30 to 31 Mar | Claude Sonnet 4.6 | 1 to 4 | `BUILD_DIALOGUE.md`, `.pdf` |

Each dialogue was written by the agent and records the operator's prompts, the
errors encountered at each stage, how they were diagnosed, and any deviation
from the guide. Most were kept up to date during the build. `Build_C1` was
assembled at the end from the agent's project memory, a recovered partial
transcript and the final live session; its opening section says which parts
are reconstructed. `Build_B1` ends with a post-build comparison against
`Build_A`.

File names were normalized to `BUILD_DIALOGUE` here. Researcher names were
replaced with the researcher letters and build labels used above, and
working-tree file paths and folder names with the location each item occupies
in this repository; both substitutions were applied inside quoted prompts as
well. Because the code of these builds is not included, their code paths
(for example `Build_A/co2r_bulk/`) name folders that are not present. Apart from
these substitutions and one deleted parenthetical in `Build_B2`, the documents
are as written; pipe characters inside table cells were escaped so that the
tables render.

`Build_B1` titles itself "Session 2" and refers to `Build_A` as "Session 1".
`Build_D` is the curated dialogue distilled from a multi-session raw
transcript; the raw transcript is not included.

The dialogues credit the experimental polarization data used for the fit to
Lees et al. 2024, as the first-generation guide did. The manuscript and the
released guide cite the original measurement, Romiluyi et al. 2023; the values
are the same.

Only the dialogues are provided for these builds. Their code was superseded by
the guide revisions that they motivated and is not part of the release.
