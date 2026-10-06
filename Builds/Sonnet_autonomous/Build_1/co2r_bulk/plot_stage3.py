import os
from plot_common import plot_A1, plot_A2, plot_B1

base = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(base, "output_stage3")
plotdir = os.path.join(base, "plots")
os.makedirs(plotdir, exist_ok=True)

plot_A1(os.path.join(outdir, "polarization.csv"), "Stage 3 -- Polarization (+ Stefan-Maxwell gas, u_G=0, pre-fit kinetics)",
        os.path.join(plotdir, "stage3_A1_polarization.png"))
plot_A2(os.path.join(outdir, "liquid_profiles.csv"), "Stage 3 -- Liquid-phase diagnostics (all 89 voltages)",
        os.path.join(plotdir, "stage3_A2_liquid.png"))
plot_B1(os.path.join(outdir, "gas_profiles.csv"), "Stage 3 -- Gas-phase diagnostics (all 89 voltages)",
        os.path.join(plotdir, "stage3_B1_gas.png"))
print("Stage 3 plots written to", plotdir)
