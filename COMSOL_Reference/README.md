# COMSOL reference model exports

Plain-text exports from the COMSOL Multiphysics reference model
`JuliaCheck_07_08_26.mph` (COMSOL 6.4, Electrochemistry and CFD modules),
exported 8 July 2026. These are the data used for every Julia-versus-COMSOL
comparison in the manuscript (Fig. 3, the RMSD figures in the text, and
the nine-build statistics), so those comparisons can be reproduced without a
COMSOL license.

The model file itself is included here as `AgenticPaper_PublishedModel.mph`
(saved 21 September 2026); `JuliaCheck_07_08_26.mph` above is the file name it
carried when these exports were written, which is what the export headers
record.

## Files (`JuliaCheck_07_08_26_exports/`)

All files share the COMSOL export header (model, version, date, table) as `%`
comment lines. The sweep is 61 applied potentials from 0 to -3 V in 50 mV steps.

| File | Contents |
|---|---|
| `CDs.txt` | Per-potential line integrals: `V_c`, CO partial current density, H2 partial current density, total current density (A/m², cathodic negative), and CO molar flux. **This is the table plotted in Fig. 3.** |
| `fluxes.txt` | Per-potential HCO3- and CO32- boundary fluxes at the electrolyte face (CO2 utilisation). |
| `pH.txt`, `cH.txt`, `cOH.txt`, `cHCO3.txt`, `cCO3.txt`, `cCO2aq.txt`, `cCs.txt` | Liquid-phase profiles along x (3001 nodes), one column per potential. |
| `cCO2g.txt`, `cCO.txt`, `cH2.txt`, `P.txt` | Gas-phase concentration and pressure profiles, same layout. |
| `RCOER.txt`, `RHER.txt` | Local CO2 reduction and hydrogen evolution reaction-rate profiles. |

`V_c` is the applied potential in the SHE convention used inside the model;
the Julia builds report both SHE and RHE columns, and the figure scripts
convert before overlaying.

## Verification

On 21 September 2026 the `CDs.txt` and `fluxes.txt` tables were re-exported
from the same `.mph` file under COMSOL 6.4.0.429 and compared against these
July exports: every numerical value was identical. Only the header date,
build number and table index differed.
