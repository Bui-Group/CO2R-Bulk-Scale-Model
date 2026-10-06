import os
from plot_common import plot_A1, plot_A2, plot_B1, plot_D1, plot_D2

base = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(base, "output_stage5")
plotdir = os.path.join(base, "plots")
os.makedirs(plotdir, exist_ok=True)

plot_A1(os.path.join(outdir, "polarization.csv"), "Stage 5 -- Polarization (FULL physics: phi_L + phi_s, fitted kinetics)",
        os.path.join(plotdir, "stage5_A1_polarization.png"))
plot_A2(os.path.join(outdir, "liquid_profiles.csv"), "Stage 5 -- Liquid-phase diagnostics (all 89 voltages)",
        os.path.join(plotdir, "stage5_A2_liquid.png"))
plot_B1(os.path.join(outdir, "gas_profiles.csv"), "Stage 5 -- Gas-phase diagnostics (all 89 voltages)",
        os.path.join(plotdir, "stage5_B1_gas.png"))
plot_D1(os.path.join(outdir, "phi_s_profiles.csv"), "Stage 5 -- phi_s profile (GDL+CL) at selected voltages",
        os.path.join(plotdir, "stage5_D1_phi_s.png"), x_interface_m=0.0)
plot_D2(os.path.join(outdir, "phi_L_profiles.csv"), "Stage 5 -- phi_L profile (CL only) at selected voltages",
        os.path.join(plotdir, "stage5_D2_phi_L.png"))

# overlay all 5 stages' polarization for the final comparison figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from plot_common import E_exp_SHE, i_CO_exp, i_H2_exp

fig, ax = plt.subplots(figsize=(9,7))
stages = [("output_stage1","Stage 1",0.6,":"), ("output_stage2","Stage 2",0.6,":"),
          ("output_stage3","Stage 3",0.8,"--"), ("output_stage4","Stage 4",0.8,"--"),
          ("output_fit","Fit (fitted kinetics, Stage-4 physics)",1.0,"-."),
          ("output_stage5","Stage 5 (full physics)",2.2,"-")]
for folder, label, lw, ls in stages:
    p = os.path.join(base, folder, "polarization.csv")
    if not os.path.exists(p):
        continue
    d = pd.read_csv(p)
    ax.plot(-d["V_SHE"], d["i_CO_mA_cm2"], color="#1f77b4", lw=lw, ls=ls, alpha=0.85, label=f"$i_{{CO}}$ {label}")
ax.scatter(-E_exp_SHE, i_CO_exp, marker="s", facecolor="none", edgecolor="black", s=70, label="$i_{CO}$ (exp)", zorder=10)
ax.set_yscale("log"); ax.set_xlim(0.7,3.1)
ax.set_xlabel("$-E_{cath}$ [V vs SHE]"); ax.set_ylabel("$i_{CO}$ [mA/cm$^2$] (log)")
ax.set_title("i_CO across all stages")
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(plotdir, "all_stages_iCO_overlay.png"), dpi=150)
plt.close(fig)

print("Stage 5 plots written to", plotdir)
