#!/usr/bin/env python3
"""
plot_results.py — diagnostic plots for the CO2R bulk-scale model (BaseCase.md §12, §18).

Usage:  python3 plot_results.py <stage>      # stage in {1,2,3,4,5} or 'fit'

Reads output_stage<N>/polarization.csv and profiles.csv (written by run_stage<N>.jl) and
renders the required figures:
  A1  3-panel polarization summary (linear i, log i, FE) vs -E_cath [V vs SHE], with Lees data.
  A2  3x3 liquid-phase diagnostic (pH, CO2, OH, HCO3, CO3, Cs, H, R_CO2R, R_HER) vs x at all voltages.
  B1  2x2 gas-phase partial pressures vs x          (Stage 3+; if gas_profiles.csv exists)
  D1/D2 phi_s / phi_L profiles                       (Stage 5; if potentials.csv exists)

All voltage x-axes use V vs SHE (positive = more cathodic), per the guide.
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# ---- try Lato; fall back gracefully ----
try:
    from matplotlib import font_manager
    if any("Lato" in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = "Lato"
except Exception:
    pass
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 120, "savefig.dpi": 150})

# ---- experimental data, Lees et al. 2024 (BaseCase.md §16.1) ----
OFFSET = 2.303 * 8.314462618 * 323.15 / 96485.332 * 8.0          # 0.51305 V
E_exp_RHE = np.array([-0.816, -1.205, -1.644, -2.079, -2.413])
i_CO_exp  = np.array([35.1, 189.6, 299.1, 204.3, 64.6])
i_H2_exp  = np.array([29.9, 49.3, 119.6, 449.9, 850.5])
E_exp_SHE = E_exp_RHE - OFFSET
FE_CO_exp = 100 * i_CO_exp / (i_CO_exp + i_H2_exp)
FE_H2_exp = 100 * i_H2_exp / (i_CO_exp + i_H2_exp)

C_CO, C_H2, C_TOT = "#1f77b4", "#d62728", "#2ca02c"


def stage_dir(stage):
    here = os.path.dirname(os.path.abspath(__file__))
    if str(stage) == "fit":
        return os.path.join(here, "output_fit")
    return os.path.join(here, f"output_stage{stage}")


def plot_A1(stage, sd):
    d = np.genfromtxt(os.path.join(sd, "polarization.csv"), delimiter=",", names=True)
    order = np.argsort(-d["V_RHE"])                  # mild -> cathodic
    mV = -d["V_SHE"][order]                           # -E_cath (positive = cathodic)
    iCO, iH2, itot = d["i_CO_mAcm2"][order], d["i_H2_mAcm2"][order], d["i_tot_mAcm2"][order]
    feco = d["FE_CO"][order]; feh2 = d["FE_H2"][order]
    mVexp = -E_exp_SHE

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    # (a) linear polarization
    ax[0].plot(mV, iCO, "-", color=C_CO, label="CO (model)")
    ax[0].plot(mV, iH2, "-", color=C_H2, label="H$_2$ (model)")
    ax[0].plot(mV, itot, "-", color=C_TOT, label="total (model)")
    ax[0].plot(mVexp, i_CO_exp, "s", mfc="none", mec=C_CO, ms=8, label="CO (Lees)")
    ax[0].plot(mVexp, i_H2_exp, "^", mfc="none", mec=C_H2, ms=8, label="H$_2$ (Lees)")
    ax[0].set_title("(a) Linear polarization"); ax[0].set_ylabel("i [mA/cm$^2$]")
    ax[0].legend(fontsize=8, ncol=2)
    # (b) log polarization
    ax[1].semilogy(mV, np.maximum(iCO, 1e-3), "-", color=C_CO)
    ax[1].semilogy(mV, np.maximum(iH2, 1e-3), "-", color=C_H2)
    ax[1].semilogy(mV, np.maximum(itot, 1e-3), "-", color=C_TOT)
    ax[1].semilogy(mVexp, i_CO_exp, "s", mfc="none", mec=C_CO, ms=8)
    ax[1].semilogy(mVexp, i_H2_exp, "^", mfc="none", mec=C_H2, ms=8)
    ax[1].set_title("(b) Log polarization"); ax[1].set_ylabel("i [mA/cm$^2$] (log)")
    ax[1].set_ylim(1e-1, 3e3)
    # (c) faradaic efficiency
    ax[2].plot(mV, feco, "-", color=C_CO, label="FE$_{CO}$ (model)")
    ax[2].plot(mV, feh2, "-", color=C_H2, label="FE$_{H_2}$ (model)")
    ax[2].plot(mVexp, FE_CO_exp, "s", mfc="none", mec=C_CO, ms=8, label="FE$_{CO}$ (Lees)")
    ax[2].plot(mVexp, FE_H2_exp, "^", mfc="none", mec=C_H2, ms=8, label="FE$_{H_2}$ (Lees)")
    ax[2].set_title("(c) Faradaic efficiency"); ax[2].set_ylabel("FE [%]"); ax[2].set_ylim(0, 100)
    ax[2].legend(fontsize=8)
    for a in ax:
        a.set_xlabel("$-E_{cath}$ [V vs SHE]"); a.set_xlim(0.7, 3.1)
    fig.suptitle(f"Stage {stage}: A1 polarization summary (model vs Lees et al. 2024)", y=1.02)
    fig.tight_layout()
    out = os.path.join(sd, f"stage{stage}_A1_polarization.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def load_profiles(sd):
    d = np.genfromtxt(os.path.join(sd, "profiles.csv"), delimiter=",", names=True)
    Vs = np.unique(d["V_RHE"])
    Vs = Vs[np.argsort(-Vs)]                          # mild -> cathodic
    return d, Vs


def plot_A2(stage, sd):
    d, Vs = load_profiles(sd)
    norm = Normalize(vmin=-Vs.max(), vmax=-Vs.min())   # color by -V_RHE
    cmap = plt.cm.viridis
    panels = [("pH", "pH [-]"), ("c_CO2", r"CO$_2$(aq) [mol/m$^3$]"), ("c_OH", r"OH$^-$ [mol/m$^3$]"),
              ("c_HCO3", r"HCO$_3^-$ [mol/m$^3$]"), ("c_CO3", r"CO$_3^{2-}$ [mol/m$^3$]"),
              ("c_Cs", r"Cs$^+$ [mol/m$^3$]"), ("c_H", r"H$^+$ [mol/m$^3$]"),
              ("R_CO2R", r"R$_{CO2R}$ [mol/m$^3$/s]"), ("R_HER", r"R$_{HER}$ [mol/m$^3$/s]")]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for ax, (key, lab) in zip(axes.flat, panels):
        for V in Vs:
            m = d["V_RHE"] == V
            x = d["x_um"][m]; y = d[key][m]
            idx = np.argsort(x)
            ax.plot(x[idx], y[idx], color=cmap(norm(-V)), lw=0.8)
        ax.set_xlabel(r"x [$\mu$m]  (0=GDL, $L_{CL}$=electrolyte)"); ax.set_ylabel(lab)
        if key in ("c_H",):
            ax.set_yscale("log")
        if key in ("R_CO2R", "R_HER"):
            ax.set_yscale("symlog", linthresh=1e-2)
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.01)
    cb.set_label(r"$-E_{cath}$ [V vs RHE@pH8]")
    fig.suptitle(f"Stage {stage}: A2 liquid-phase diagnostics (curves = all sweep voltages)", y=0.995)
    out = os.path.join(sd, f"stage{stage}_A2_liquid.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def plot_overlay(stage, sd):
    """Overlay polarization of stages 1..stage that have a polarization.csv (§11 Stage 2+)."""
    prev = [s for s in range(1, int(stage) + 1) if os.path.isfile(os.path.join(stage_dir(s), "polarization.csv"))]
    if len(prev) < 2:
        return
    cmap = plt.cm.plasma
    colors = [cmap(t) for t in np.linspace(0.0, 0.85, len(prev))]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for s, col in zip(prev, colors):
        d = np.genfromtxt(os.path.join(stage_dir(s), "polarization.csv"), delimiter=",", names=True)
        o = np.argsort(-d["V_RHE"]); mV = -d["V_SHE"][o]
        ax[0].plot(mV, d["i_CO_mAcm2"][o], "-", color=col, label=f"Stage {s}")
        ax[1].plot(mV, d["i_H2_mAcm2"][o], "-", color=col, label=f"Stage {s}")
        ax[2].plot(mV, d["FE_CO"][o], "-", color=col, label=f"Stage {s}")
    mVexp = -E_exp_SHE
    ax[0].plot(mVexp, i_CO_exp, "ks", mfc="none", ms=8, label="Lees")
    ax[1].plot(mVexp, i_H2_exp, "k^", mfc="none", ms=8, label="Lees")
    ax[2].plot(mVexp, FE_CO_exp, "ks", mfc="none", ms=8, label="Lees")
    ax[0].set_title("(a) i$_{CO}$ by stage"); ax[0].set_ylabel("i$_{CO}$ [mA/cm$^2$]")
    ax[1].set_title("(b) i$_{H_2}$ by stage"); ax[1].set_ylabel("i$_{H_2}$ [mA/cm$^2$]")
    ax[2].set_title("(c) FE$_{CO}$ by stage"); ax[2].set_ylabel("FE$_{CO}$ [%]"); ax[2].set_ylim(0, 100)
    for a in ax:
        a.set_xlabel("$-E_{cath}$ [V vs SHE]"); a.set_xlim(0.7, 3.1); a.legend(fontsize=8)
    fig.suptitle(f"Stage overlay (1..{stage}) — model evolution vs Lees et al. 2024", y=1.02)
    fig.tight_layout()
    out = os.path.join(sd, f"stage{stage}_overlay.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def plot_B1(stage, sd):
    """2x2 gas-phase partial pressures vs x over GDL+CL (BaseCase.md §12.7 B1)."""
    d = np.genfromtxt(os.path.join(sd, "gas_profiles.csv"), delimiter=",", names=True)
    Vs = np.unique(d["V_RHE"]); Vs = Vs[np.argsort(-Vs)]
    norm = Normalize(vmin=-Vs.max(), vmax=-Vs.min()); cmap = plt.cm.viridis
    panels = [("p_CO2_atm", r"p$_{CO_2}$ [atm]"), ("p_CO_atm", r"p$_{CO}$ [atm]"),
              ("p_H2_atm", r"p$_{H_2}$ [atm]"), ("p_total_atm", r"p$_{total}$ [atm]")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (key, lab) in zip(axes.flat, panels):
        for V in Vs:
            m = d["V_RHE"] == V
            x = d["x_um"][m]; y = d[key][m]; idx = np.argsort(x)
            ax.plot(x[idx], y[idx], color=cmap(norm(-V)), lw=0.8)
        ax.axvline(0.0, ls="--", color="k", lw=0.8, alpha=0.6)   # GDL/CL interface
        ax.set_xlabel(r"x [$\mu$m]  (GDL: x<0, CL: x>0)"); ax.set_ylabel(lab)
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.02); cb.set_label(r"$-E_{cath}$ [V vs RHE@pH8]")
    fig.suptitle(f"Stage {stage}: B1 gas-phase partial pressures (all sweep voltages)", y=0.98)
    out = os.path.join(sd, f"stage{stage}_B1_gas.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def plot_D(stage, sd):
    """D1 φ_s(x) over GDL+CL and D2 φ_L(x) over CL (BaseCase.md §12.7 D)."""
    d = np.genfromtxt(os.path.join(sd, "potentials.csv"), delimiter=",", names=True,
                      dtype=None, encoding=None)
    Vs = np.unique(d["V_RHE"]); Vs = Vs[np.argsort(-Vs)]
    norm = Normalize(vmin=-Vs.max(), vmax=-Vs.min()); cmap = plt.cm.viridis
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for V in Vs:
        m = d["V_RHE"] == V
        x = d["x_um"][m]; phis = d["phi_s_V"][m]; phil = d["phi_L_V"][m]
        idx = np.argsort(x)
        ax[0].plot(x[idx], phis[idx], color=cmap(norm(-V)), lw=0.8)
        clmask = np.array([str(r) == "CL" for r in d["region"][m]])
        xs = x[clmask]; pl = phil[clmask]; j = np.argsort(xs)
        ax[1].plot(xs[j], pl[j] * 1000, color=cmap(norm(-V)), lw=0.8)
    ax[0].axvline(0.0, ls="--", color="k", lw=0.8, alpha=0.6)
    ax[0].set_xlabel(r"x [$\mu$m] (collector −325 → electrolyte +10)"); ax[0].set_ylabel(r"$\phi_s$ [V vs SHE]")
    ax[0].set_title("(D1) solid potential $\\phi_s$ (GDL+CL)")
    ax[1].set_xlabel(r"x [$\mu$m] (CL: 0 → $L_{CL}$)"); ax[1].set_ylabel(r"$\phi_L$ [mV]")
    ax[1].set_title("(D2) liquid potential $\\phi_L$ (CL, gauge $\\phi_L$=0 at electrolyte)")
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02); cb.set_label(r"$-E_{cath}$ [V vs RHE@pH8]")
    fig.suptitle(f"Stage {stage}: D1/D2 potential profiles", y=1.0)
    out = os.path.join(sd, f"stage{stage}_D_potentials.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def plot_fit_vs_default(sd):
    """Fitted (output_fit) vs default-kinetics Stage 4 polarization (BaseCase.md §12.7 E)."""
    s4 = os.path.join(stage_dir(4), "polarization.csv")
    if not os.path.isfile(s4):
        return
    df = np.genfromtxt(os.path.join(sd, "polarization.csv"), delimiter=",", names=True)
    dd = np.genfromtxt(s4, delimiter=",", names=True)
    of = np.argsort(-df["V_RHE"]); od = np.argsort(-dd["V_RHE"])
    mVexp = -E_exp_SHE
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for a, (key, exp, lab) in zip(ax, [("i_CO_mAcm2", i_CO_exp, "i$_{CO}$"),
                                       ("i_H2_mAcm2", i_H2_exp, "i$_{H_2}$"),
                                       ("FE_CO", FE_CO_exp, "FE$_{CO}$")]):
        a.plot(-dd["V_SHE"][od], dd[key][od], "--", color="gray", label="Stage 4 (default kinetics)")
        a.plot(-df["V_SHE"][of], df[key][of], "-", color=C_CO, label="fitted kinetics")
        a.plot(mVexp, exp, "ks", mfc="none", ms=8, label="Lees")
        a.set_xlabel("$-E_{cath}$ [V vs SHE]"); a.set_title(lab); a.set_xlim(0.7, 3.1); a.legend(fontsize=8)
    ax[2].set_ylim(0, 100)
    fig.suptitle("Fit step: fitted vs default kinetics (Stage 4 residual) vs Lees", y=1.02)
    fig.tight_layout()
    out = os.path.join(sd, "fit_vs_default.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "1"
    sd = stage_dir(stage)
    assert os.path.isdir(sd), f"missing {sd}"
    plot_A1(stage, sd)
    plot_A2(stage, sd)
    if str(stage).isdigit() and int(stage) >= 2:
        plot_overlay(stage, sd)
    if os.path.isfile(os.path.join(sd, "gas_profiles.csv")):     # B1 (Stage 3+)
        plot_B1(stage, sd)
    if os.path.isfile(os.path.join(sd, "potentials.csv")):       # D1/D2 (Stage 5)
        plot_D(stage, sd)
    if str(stage) == "fit":
        plot_fit_vs_default(sd)
    print("done.")


if __name__ == "__main__":
    main()
