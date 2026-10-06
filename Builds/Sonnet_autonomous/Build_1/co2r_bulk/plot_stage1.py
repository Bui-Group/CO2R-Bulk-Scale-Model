import os
from plot_common import plot_A1, plot_A2

base = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(base, "output_stage1")
plotdir = os.path.join(base, "plots")
os.makedirs(plotdir, exist_ok=True)

plot_A1(os.path.join(outdir, "polarization.csv"), "Stage 1 -- Polarization (diffusion-only, Dirichlet BC, pre-fit kinetics)",
        os.path.join(plotdir, "stage1_A1_polarization.png"))
plot_A2(os.path.join(outdir, "liquid_profiles.csv"), "Stage 1 -- Liquid-phase diagnostics (all 89 voltages)",
        os.path.join(plotdir, "stage1_A2_liquid.png"))
print("Stage 1 plots written to", plotdir)
