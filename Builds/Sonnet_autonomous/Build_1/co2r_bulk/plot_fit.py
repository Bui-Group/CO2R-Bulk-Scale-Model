import os
import pandas as pd
from plot_common import plot_A1, plot_A2, plot_B1

base = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(base, "output_fit")
s4dir = os.path.join(base, "output_stage4")
plotdir = os.path.join(base, "plots")
os.makedirs(plotdir, exist_ok=True)

plot_A1(os.path.join(outdir, "polarization.csv"), "Fit step -- Polarization (Stage-4 physics, FITTED kinetics)",
        os.path.join(plotdir, "fit_A1_polarization.png"))
plot_A2(os.path.join(outdir, "liquid_profiles.csv"), "Fit step -- Liquid-phase diagnostics (all 89 voltages, fitted kinetics)",
        os.path.join(plotdir, "fit_A2_liquid.png"))
plot_B1(os.path.join(outdir, "gas_profiles.csv"), "Fit step -- Gas-phase diagnostics (all 89 voltages, fitted kinetics)",
        os.path.join(plotdir, "fit_B1_gas.png"))

# side-by-side pre-fit (Stage 4) vs post-fit (Fit step) polarization for direct comparison
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plot_common import E_exp_SHE, i_CO_exp, i_H2_exp, V_RHE_TO_SHE_OFFSET

d4 = pd.read_csv(os.path.join(s4dir, "polarization.csv"))
dfit = pd.read_csv(os.path.join(outdir, "polarization.csv"))
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(-d4["V_SHE"], d4["i_CO_mA_cm2"], color="#1f77b4", lw=1.2, ls="--", label="$i_{CO}$ pre-fit (Stage 4)")
ax.plot(-d4["V_SHE"], d4["i_H2_mA_cm2"], color="#d62728", lw=1.2, ls="--", label="$i_{H_2}$ pre-fit (Stage 4)")
ax.plot(-dfit["V_SHE"], dfit["i_CO_mA_cm2"], color="#1f77b4", lw=2.0, label="$i_{CO}$ fitted")
ax.plot(-dfit["V_SHE"], dfit["i_H2_mA_cm2"], color="#d62728", lw=2.0, label="$i_{H_2}$ fitted")
ax.scatter(-E_exp_SHE, i_CO_exp, marker="s", facecolor="none", edgecolor="#1f77b4", s=60, label="$i_{CO}$ (exp)")
ax.scatter(-E_exp_SHE, i_H2_exp, marker="^", facecolor="none", edgecolor="#d62728", s=60, label="$i_{H_2}$ (exp)")
ax.set_yscale("log")
ax.set_xlim(0.7,3.1)
ax.set_xlabel("$-E_{cath}$ [V vs SHE]"); ax.set_ylabel("i [mA/cm$^2$] (log)")
ax.set_title("Pre-fit vs fitted kinetics (Stage-4 physics) vs experiment")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(plotdir, "fit_prefit_vs_postfit.png"), dpi=150)
plt.close(fig)

print("Fit plots written to", plotdir)
