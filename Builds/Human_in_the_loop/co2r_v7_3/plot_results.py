#!/usr/bin/env python3
"""
plot_results.py — figures for the CO2R bulk-scale model (§12.7, §18).

Reads output_stage<N>/summary_<tag>.csv and profiles_<tag>.bin written by the
Julia drivers.  Figures follow the lab convention: customplot.gengrid from the
repo root (auto panel labels, inward ticks, Lato/Fira-Math fonts).

Usage:  python3 plot_results.py <tag> [<tag> ...]
        tags: stage1 stage2 stage3 stage4 fit_prefit fit_postfit stage5
"""
import os, sys, json
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
_cwd = os.getcwd()
os.chdir(REPO_ROOT)                 # customplot reads the swatch xlsx relatively
sys.path.insert(0, REPO_ROOT)
from customplot import (gengrid, tableau20, Pantone_Blue, Pantone_Red,
                        warm_sequential, cool_sequential)
import matplotlib as mpl
mpl.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
os.chdir(_cwd)

FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

V_OFF = 2.303 * 8.314462618 * 323.15 / 96485.332 * 8.0     # 0.51305 V
E_exp_RHE = np.array([-0.816, -1.205, -1.644, -2.079, -2.413])
i_CO_exp  = np.array([35.1, 189.6, 299.1, 204.3, 64.6])
i_H2_exp  = np.array([29.9, 49.3, 119.6, 449.9, 850.5])
E_exp_SHE = E_exp_RHE - V_OFF
FE_CO_exp = i_CO_exp / (i_CO_exp + i_H2_exp) * 100
FE_H2_exp = i_H2_exp / (i_CO_exp + i_H2_exp) * 100

RAMP = LinearSegmentedColormap.from_list("warm_seq", list(warm_sequential))
C_CO, C_H2, C_TOT = tableau20[0], tableau20[6], tableau20[14]
FS = 8


def stage_dir(tag):
    if tag.startswith("stage"):
        return os.path.join(HERE, "output_" + tag)
    return os.path.join(HERE, "output_fit")


def load(tag):
    d = stage_dir(tag)
    csv = np.genfromtxt(os.path.join(d, f"summary_{tag}.csv"), delimiter=",",
                        names=True)
    with open(os.path.join(d, f"profiles_{tag}.bin"), "rb") as f:
        nV, N_CL, N_G, stage = np.fromfile(f, dtype=np.int64, count=4)
        x_cl = np.fromfile(f, dtype=np.float64, count=N_CL)
        x_gas = np.fromfile(f, dtype=np.float64, count=N_G)
        V, liq, gas, phis, phiL = [], [], [], [], []
        for _ in range(nV):
            V.append(np.fromfile(f, dtype=np.float64, count=1)[0])
            liq.append(np.fromfile(f, dtype=np.float64,
                                   count=9 * N_CL).reshape(9, N_CL))
            if stage >= 3:
                gas.append(np.fromfile(f, dtype=np.float64,
                                       count=4 * N_G).reshape(4, N_G))
            if stage >= 5:
                phis.append(np.fromfile(f, dtype=np.float64, count=N_G))
                phiL.append(np.fromfile(f, dtype=np.float64, count=N_CL))
    return dict(tag=tag, stage=int(stage), csv=csv, x_cl=x_cl, x_gas=x_gas,
                V=np.array(V), liq=np.array(liq),
                gas=np.array(gas) if gas else None,
                phis=np.array(phis) if phis else None,
                phiL=np.array(phiL) if phiL else None)


def _ramp_colors(V):
    """light = mild, dark = cathodic; returns colors + the matching norm."""
    mv = -np.asarray(V) - V_OFF                  # -E_cath vs SHE, increasing
    norm = Normalize(vmin=mv.min(), vmax=mv.max())
    return RAMP(norm(mv)), norm


def _cbar(fig, axs, norm, label):
    sm = plt.cm.ScalarMappable(cmap=RAMP, norm=norm)
    cb = fig.colorbar(sm, ax=list(np.ravel(axs)), fraction=0.02, pad=0.02)
    cb.set_label(label, fontsize=FS)
    cb.ax.tick_params(labelsize=FS - 1)


def fig_A1(D, extra=None):
    """3-panel polarization summary (§12.7 A1)."""
    c = D["csv"]
    mv = -c["V_SHE"]
    fig, axs, _ = gengrid(3, 1, size_inches=(9.75, 2.9), ticklabel_size=7)
    for k, ax in enumerate(axs):
        if k < 2:
            ax.plot(mv, c["i_CO_mAcm2"], "-", color=C_CO, lw=1.4, label=r"$i_{CO}$ model")
            ax.plot(mv, c["i_H2_mAcm2"], "-", color=C_H2, lw=1.4, label=r"$i_{H_2}$ model")
            ax.plot(mv, c["i_tot_mAcm2"], color=C_TOT, lw=1.0, ls="--",
                    label=r"$i_{tot}$ model")
            if extra is not None:
                ec = extra["csv"]
                ax.plot(-ec["V_SHE"], ec["i_CO_mAcm2"], ":", color=C_CO, lw=1.0)
                ax.plot(-ec["V_SHE"], ec["i_H2_mAcm2"], ":", color=C_H2, lw=1.0)
            ax.plot(-E_exp_SHE, i_CO_exp, "s", mfc="none", mec=C_CO, ms=4.5,
                    mew=1.1, ls="none", label=r"$i_{CO}$ exp")
            ax.plot(-E_exp_SHE, i_H2_exp, "^", mfc="none", mec=C_H2, ms=4.5,
                    mew=1.1, ls="none", label=r"$i_{H_2}$ exp")
            ax.set_ylabel(r"$i$ [mA cm$^{-2}$]", fontsize=FS)
            if k == 1:
                ax.set_yscale("log"); ax.set_ylim(1e-3, 5e3)
        else:
            fe_co = c["FE_CO_pct"]
            ax.plot(mv, fe_co, "-", color=C_CO, lw=1.4, label=r"FE$_{CO}$ model")
            ax.plot(mv, 100 - fe_co, "-", color=C_H2, lw=1.4, label=r"FE$_{H_2}$ model")
            ax.plot(-E_exp_SHE, FE_CO_exp, "s", mfc="none", mec=C_CO, ms=4.5,
                    mew=1.1, ls="none", label=r"FE$_{CO}$ exp")
            ax.plot(-E_exp_SHE, FE_H2_exp, "^", mfc="none", mec=C_H2, ms=4.5,
                    mew=1.1, ls="none", label=r"FE$_{H_2}$ exp")
            ax.set_ylabel("FE [%]", fontsize=FS); ax.set_ylim(0, 100)
        ax.set_xlabel(r"$-E_{cath}$ [V vs SHE]", fontsize=FS)
        ax.set_xlim(0.7, 3.1)
        ax.legend(fontsize=5.5, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, f"A1_polarization_{D['tag']}.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_A2(D):
    """3x3 liquid-phase diagnostic (§12.7 A2)."""
    names = [("pH", "pH [--]"), ("CO2", r"CO$_2$(aq) [mol m$^{-3}$]"),
             ("OH", r"OH$^-$ [mol m$^{-3}$]"), ("HCO3", r"HCO$_3^-$ [mol m$^{-3}$]"),
             ("CO3", r"CO$_3^{2-}$ [mol m$^{-3}$]"), ("Cs", r"Cs$^+$ [mol m$^{-3}$]"),
             ("H", r"H$^+$ [mol m$^{-3}$]"),
             ("RCO2R", r"$R_{CO_2R}$ [mol m$^{-3}$ s$^{-1}$]"),
             ("RHER", r"$R_{HER}$ [mol m$^{-3}$ s$^{-1}$]")]
    cols, norm = _ramp_colors(D["V"])
    fig, axs, _ = gengrid(3, 3, size_inches=(9.0, 8.0), ticklabel_size=7)
    for p in range(9):
        ax = axs[p // 3][p % 3]
        for k in range(len(D["V"])):
            ax.plot(D["x_cl"], D["liq"][k, p, :], color=cols[k], lw=0.55)
        ax.set_xlabel(r"$x$ [$\mu$m]", fontsize=FS)
        ax.set_ylabel(names[p][1], fontsize=FS)
        if p == 6:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.yaxis.get_offset_text().set_fontsize(6)
    _cbar(fig, axs, norm, r"$-E_{cath}$ [V vs SHE]")
    fig.savefig(os.path.join(FIGDIR, f"A2_liquid_{D['tag']}.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_B1(D):
    """2x2 gas-phase diagnostic (§12.7 B1)."""
    if D["gas"] is None:
        return
    labs = [r"$p_{CO_2}$ [atm]", r"$p_{CO}$ [atm]", r"$p_{H_2}$ [atm]",
            r"$p_{total}$ [Pa]"]
    cols, norm = _ramp_colors(D["V"])
    x_if = 0.0
    fig, axs, _ = gengrid(2, 2, size_inches=(7.0, 6.0), ticklabel_size=7)
    for p in range(4):
        ax = axs[p // 2][p % 2]
        for k in range(len(D["V"])):
            ax.plot(D["x_gas"], D["gas"][k, p, :], color=cols[k], lw=0.55)
        ax.axvline(x_if, color="0.4", ls="--", lw=0.8)
        ax.set_xlabel(r"$x$ [$\mu$m]", fontsize=FS)
        ax.set_ylabel(labs[p], fontsize=FS)
    _cbar(fig, axs, norm, r"$-E_{cath}$ [V vs SHE]")
    fig.savefig(os.path.join(FIGDIR, f"B1_gas_{D['tag']}.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_D(D):
    """phi_s and phi_L profiles (§12.7 D1/D2)."""
    if D["phis"] is None:
        return
    cols, norm = _ramp_colors(D["V"])
    fig, axs, _ = gengrid(2, 1, size_inches=(7.0, 2.9), ticklabel_size=7)
    for k in range(len(D["V"])):
        axs[0].plot(D["x_gas"], D["phis"][k], color=cols[k], lw=0.55)
        axs[1].plot(D["x_cl"] / D["x_cl"][-1], D["phiL"][k] * 1e3,
                    color=cols[k], lw=0.55)
    axs[0].axvline(0.0, color="0.4", ls="--", lw=0.8)
    axs[0].set_xlabel(r"$x$ [$\mu$m]  (GDL + CL)", fontsize=FS)
    axs[0].set_ylabel(r"$\phi_s$ [V vs SHE]", fontsize=FS)
    axs[1].set_xlabel(r"$x/L_{CL}$", fontsize=FS)
    axs[1].set_ylabel(r"$\phi_L$ [mV vs SHE]", fontsize=FS)
    _cbar(fig, axs, norm, r"$-E_{cath}$ [V vs SHE]")
    fig.savefig(os.path.join(FIGDIR, f"D_potentials_{D['tag']}.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_overlay(tags, name="overlay"):
    """Polarization overlay across stages."""
    fig, axs, _ = gengrid(2, 1, size_inches=(7.0, 2.9), ticklabel_size=7)
    cs = [Pantone_Blue[6], Pantone_Blue[4], Pantone_Blue[2],
          Pantone_Red[5], Pantone_Red[3], tableau20[4]]
    for i, t in enumerate(tags):
        try:
            c = load(t)["csv"]
        except OSError:
            continue
        axs[0].plot(-c["V_SHE"], c["i_CO_mAcm2"], "-", color=cs[i % len(cs)],
                    lw=1.2, label=t)
        axs[1].plot(-c["V_SHE"], c["i_H2_mAcm2"], "-", color=cs[i % len(cs)],
                    lw=1.2, label=t)
    axs[0].plot(-E_exp_SHE, i_CO_exp, "s", mfc="none", mec="k", ms=4.5, mew=1.1, ls="none")
    axs[1].plot(-E_exp_SHE, i_H2_exp, "^", mfc="none", mec="k", ms=4.5, mew=1.1, ls="none")
    for ax, lab in zip(axs, [r"$i_{CO}$ [mA cm$^{-2}$]", r"$i_{H_2}$ [mA cm$^{-2}$]"]):
        ax.set_xlabel(r"$-E_{cath}$ [V vs SHE]", fontsize=FS)
        ax.set_ylabel(lab, fontsize=FS)
        ax.set_xlim(0.7, 3.1)
        ax.legend(fontsize=5.5, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, f"OV_{name}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    tags = sys.argv[1:] or ["stage1"]
    for t in tags:
        if t.startswith("--"):
            continue
        D = load(t)
        fig_A1(D); fig_A2(D); fig_B1(D); fig_D(D)
        print(f"  figures written for {t}")
    if "--overlay" in sys.argv:
        real = [t for t in tags if not t.startswith("--")]
        name = "overlay"
        for a in sys.argv[1:]:
            if a.startswith("--name="):
                name = a.split("=", 1)[1]
        fig_overlay(real, name)
        print(f"  overlay OV_{name} written")
