import os
from plot_common import plot_A1, plot_A2, plot_B1

base = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(base, "output_stage4")
plotdir = os.path.join(base, "plots")
os.makedirs(plotdir, exist_ok=True)

plot_A1(os.path.join(outdir, "polarization.csv"), "Stage 4 -- Polarization (+ Darcy pressure/convection, pre-fit kinetics)",
        os.path.join(plotdir, "stage4_A1_polarization.png"))
plot_A2(os.path.join(outdir, "liquid_profiles.csv"), "Stage 4 -- Liquid-phase diagnostics (all 89 voltages)",
        os.path.join(plotdir, "stage4_A2_liquid.png"))
plot_B1(os.path.join(outdir, "gas_profiles.csv"), "Stage 4 -- Gas-phase diagnostics (all 89 voltages)",
        os.path.join(plotdir, "stage4_B1_gas.png"))
print("Stage 4 plots written to", plotdir)
