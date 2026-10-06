import os
from plot_common import plot_A1, plot_A2

base = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(base, "output_stage2")
plotdir = os.path.join(base, "plots")
os.makedirs(plotdir, exist_ok=True)

plot_A1(os.path.join(outdir, "polarization.csv"), "Stage 2 -- Polarization (+ Sherwood-Reynolds Neumann BC, pre-fit kinetics)",
        os.path.join(plotdir, "stage2_A1_polarization.png"))
plot_A2(os.path.join(outdir, "liquid_profiles.csv"), "Stage 2 -- Liquid-phase diagnostics (all 89 voltages)",
        os.path.join(plotdir, "stage2_A2_liquid.png"))
print("Stage 2 plots written to", plotdir)
