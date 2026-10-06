#!/usr/bin/env python3
"""
plot_results.py — §12.7 debug plots and §18 experimental comparison figures.

Usage:  python plot_results.py <stage>        stage in {1,2,3,4,fit,5}
        python plot_results.py all

Figures use the project-standard grid generator `customplot.gengrid`
(repo root); the CWD is switched to the repo root for the import because
customplot reads its colour swatch workbook by relative path.
"""
import os, sys, glob
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_CWD0 = os.getcwd()
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)
from customplot import gengrid, Pantone_Blue, warm_sequential, tableau20   # noqa: E402
import matplotlib as mpl                                                   # noqa: E402
import matplotlib.pyplot as plt                                            # noqa: E402
from matplotlib.colors import LinearSegmentedColormap                      # noqa: E402
os.chdir(_CWD0)

# ── experimental data — Romiluyi et al. 2023 (§16.1) ─────────────────
V_OFF      = 2.303 * 8.314462618 * 323.15 / 96485.332 * 8.0     # 0.51305 V
E_exp_RHE  = np.array([-0.816, -1.205, -1.644, -2.079, -2.413])
i_CO_exp   = np.array([35.1, 189.6, 299.1, 204.3, 64.6])
i_H2_exp   = np.array([29.9, 49.3, 119.6, 449.9, 850.5])
E_exp_SHE  = E_exp_RHE - V_OFF
FE_CO_exp  = i_CO_exp / (i_CO_exp + i_H2_exp) * 100
FE_H2_exp  = i_H2_exp / (i_CO_exp + i_H2_exp) * 100

C_CO, C_H2, C_TOT = tableau20[0], tableau20[6], tableau20[4]
FS = 8
PNG_DPI = int(os.environ.get("PNG_DPI", "200"))   # raster diagnostics; SVG stays 600
RAMP = LinearSegmentedColormap.from_list("vramp", list(warm_sequential))
_RLO, _RHI = 0.12, 1.0          # span of RAMP actually used by _vcolors
RAMP_BAR = LinearSegmentedColormap.from_list(
    "vramp_bar", RAMP(np.linspace(_RLO, _RHI, 256)))

STAGE_DIR = {1: "output_stage1", 2: "output_stage2", 3: "output_stage3",
             4: "output_stage4", "fit": "output_fit", 5: "output_stage5"}
STAGE_LBL = {1: "Stage 1", 2: "Stage 2", 3: "Stage 3", 4: "Stage 4",
             "fit": "Fit (Stage-4 physics)", 5: "Stage 5"}


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), format=ext,
                    dpi=(600 if ext == "svg" else PNG_DPI), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.join(outdir, name)}.png/.svg")


def _vcolors(vals):
    v = np.asarray(vals, float)
    if len(v) == 1:
        return [RAMP(0.5)]
    t = (v - v.min()) / (v.max() - v.min())
    return [RAMP(_RLO + (_RHI - _RLO) * (1.0 - x)) for x in t]   # light = mild, dark = cathodic


def _add_vbar(fig, axes, Vs):
    """Colourbar keyed to -E_cath.  Uses the SAME truncated ramp, unreversed,
    so bar position and line colour agree (mild = light, cathodic = dark)."""
    v = np.asarray(Vs, float)
    sm = plt.cm.ScalarMappable(cmap=RAMP_BAR,
                               norm=mpl.colors.Normalize(-v.max(), -v.min()))
    cb = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label(r"$-E_{\rm cath}$ [V vs SHE]", fontsize=FS)
    cb.ax.tick_params(labelsize=6)
    return cb


# ── A1: 3-panel polarization summary ─────────────────────────────────
def plot_A1(stage, overlays=()):
    d = os.path.join(_HERE, STAGE_DIR[stage])
    s = pd.read_csv(os.path.join(d, "summary.csv")).sort_values("V_she")
    x = -s.V_she.values
    fig, axs, _ = gengrid(3, 1, size_inches=(9.75, 2.75), ticklabel_size=7)

    for ax, logy in ((axs[0], False), (axs[1], True)):
        for lbl, so in overlays:
            ax.plot(-so.V_she.values, so.i_tot_mAcm2.values, lw=0.8, ls=":",
                    color="0.6", zorder=1)
            ax.plot(-so.V_she.values, so.i_CO_mAcm2.values, lw=0.8, ls=":",
                    color=C_CO, alpha=0.45, zorder=1)
            ax.plot(-so.V_she.values, so.i_H2_mAcm2.values, lw=0.8, ls=":",
                    color=C_H2, alpha=0.45, zorder=1)
        ax.plot(x, s.i_CO_mAcm2.values,  color=C_CO,  lw=1.4, label=r"$i_{\rm CO}$")
        ax.plot(x, s.i_H2_mAcm2.values,  color=C_H2,  lw=1.4, label=r"$i_{\rm H_2}$")
        ax.plot(x, s.i_tot_mAcm2.values, color=C_TOT, lw=1.4, label=r"$i_{\rm tot}$")
        ax.plot(-E_exp_SHE, i_CO_exp, "s", mfc="none", mec=C_CO, ms=5, mew=1.1)
        ax.plot(-E_exp_SHE, i_H2_exp, "^", mfc="none", mec=C_H2, ms=5, mew=1.1)
        ax.set_xlabel(r"$-E_{\rm cath}$ [V vs SHE]", fontsize=FS)
        ax.set_ylabel(r"$i$ [mA cm$^{-2}$]", fontsize=FS)
        ax.set_xlim(0.7, 3.1)
        if logy:
            ax.set_yscale("log"); ax.set_ylim(1e-2, 5e3)
        else:
            ax.set_ylim(bottom=0)
        ax.legend(fontsize=6, frameon=False, loc="upper left")

    ax = axs[2]
    for lbl, so in overlays:
        ax.plot(-so.V_she.values, so.FE_CO.values, lw=0.8, ls=":", color=C_CO, alpha=0.45)
        ax.plot(-so.V_she.values, so.FE_H2.values, lw=0.8, ls=":", color=C_H2, alpha=0.45)
    ax.plot(x, s.FE_CO.values, color=C_CO, lw=1.4, label=r"FE$_{\rm CO}$")
    ax.plot(x, s.FE_H2.values, color=C_H2, lw=1.4, label=r"FE$_{\rm H_2}$")
    ax.plot(-E_exp_SHE, FE_CO_exp, "s", mfc="none", mec=C_CO, ms=5, mew=1.1)
    ax.plot(-E_exp_SHE, FE_H2_exp, "^", mfc="none", mec=C_H2, ms=5, mew=1.1)
    ax.set_xlabel(r"$-E_{\rm cath}$ [V vs SHE]", fontsize=FS)
    ax.set_ylabel("Faradaic efficiency [%]", fontsize=FS)
    ax.set_xlim(0.7, 3.1); ax.set_ylim(0, 100)
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    _save(fig, os.path.join(d, "figs"), f"A1_polarization_{stage}")


# ── A2: 3×3 liquid-phase diagnostics ─────────────────────────────────
def plot_A2(stage):
    d = os.path.join(_HERE, STAGE_DIR[stage])
    p = pd.read_csv(os.path.join(d, "profiles_liquid.csv"))
    Vs = np.sort(p.V_she.unique())[::-1]
    cols = _vcolors(Vs)
    panels = [("pH", "pH [–]", False), ("c_CO2", r"CO$_2$(aq) [mol m$^{-3}$]", False),
              ("c_OH", r"OH$^-$ [mol m$^{-3}$]", True),
              ("c_HCO3", r"HCO$_3^-$ [mol m$^{-3}$]", False),
              ("c_CO3", r"CO$_3^{2-}$ [mol m$^{-3}$]", False),
              ("c_Cs", r"Cs$^+$ [mol m$^{-3}$]", False),
              ("c_H", r"H$^+$ [mol m$^{-3}$]", True),
              ("R_CO2R", r"$R_{\rm CO_2R}$ [mol m$^{-3}$ s$^{-1}$]", True),
              ("R_HER", r"$R_{\rm HER}$ [mol m$^{-3}$ s$^{-1}$]", True)]
    fig, axs, _ = gengrid(3, 3, size_inches=(9.75, 8.0), ticklabel_size=7)
    for k, (col, ylab, logy) in enumerate(panels):
        ax = axs[k // 3][k % 3]
        for c, V in zip(cols, Vs):
            g = p[np.isclose(p.V_she, V)]
            ax.plot(g.x_um.values, g[col].values, color=c, lw=0.7)
        ax.set_xlabel(r"$x$ [$\mu$m]", fontsize=FS)
        ax.set_ylabel(ylab, fontsize=FS)
        if logy:
            ax.set_yscale("log")
    _add_vbar(fig, [a for row in axs for a in row], Vs)
    os.makedirs(os.path.join(d, "figs"), exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(d, "figs", f"A2_liquid_{stage}.{ext}"),
                    format=ext, dpi=(600 if ext == "svg" else PNG_DPI), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.join(d,'figs')}/A2_liquid_{stage}.png/.svg")


# ── B1: 2×2 gas-phase diagnostics ────────────────────────────────────
def plot_B1(stage):
    d = os.path.join(_HERE, STAGE_DIR[stage])
    f = os.path.join(d, "profiles_gas.csv")
    if not os.path.isfile(f):
        return
    p = pd.read_csv(f)
    Vs = np.sort(p.V_she.unique())[::-1]
    cols = _vcolors(Vs)
    panels = [("p_CO2_atm", r"$p_{\rm CO_2}$ [atm]"), ("p_CO_atm", r"$p_{\rm CO}$ [atm]"),
              ("p_H2_atm", r"$p_{\rm H_2}$ [atm]"), ("p_total_Pa", r"$p_{\rm total}$ [Pa]")]
    fig, axs, _ = gengrid(2, 2, size_inches=(6.5, 6.25), ticklabel_size=7)
    for k, (col, ylab) in enumerate(panels):
        ax = axs[k // 2][k % 2]
        for c, V in zip(cols, Vs):
            g = p[np.isclose(p.V_she, V)]
            ax.plot(g.x_um.values, g[col].values, color=c, lw=0.7)
        ax.axvline(0.0, color="0.35", ls="--", lw=0.8)
        ax.set_xlabel(r"$x$ [$\mu$m]  (0 = GDL/CL interface)", fontsize=FS)
        ax.set_ylabel(ylab, fontsize=FS)
    _add_vbar(fig, [a for row in axs for a in row], Vs)
    os.makedirs(os.path.join(d, "figs"), exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(d, "figs", f"B1_gas_{stage}.{ext}"),
                    format=ext, dpi=(600 if ext == "svg" else PNG_DPI), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.join(d,'figs')}/B1_gas_{stage}.png/.svg")


# ── D1/D2: Stage-5 potential profiles ────────────────────────────────
def plot_D(stage):
    d = os.path.join(_HERE, STAGE_DIR[stage])
    f = os.path.join(d, "profiles_phi.csv")
    if not os.path.isfile(f):
        return
    p = pd.read_csv(f)
    Vs = np.sort(p.V_she.unique())[::-1]
    cols = _vcolors(Vs)
    fig, axs, _ = gengrid(2, 1, size_inches=(6.5, 3.0), ticklabel_size=7)
    for c, V in zip(cols, Vs):
        g = p[np.isclose(p.V_she, V)].sort_values("x_um")
        axs[0].plot(g.x_um.values, g.phi_s.values, color=c, lw=0.7)
        gl = g[g.region == "CL"]
        axs[1].plot(gl.x_um.values / (10.0), gl.phi_L.values * 1e3, color=c, lw=0.7)
    axs[0].axvline(0.0, color="0.35", ls="--", lw=0.8)
    axs[0].set_xlabel(r"$x$ [$\mu$m]  (0 = GDL/CL interface)", fontsize=FS)
    axs[0].set_ylabel(r"$\phi_s$ [V vs SHE]", fontsize=FS)
    axs[1].set_xlabel(r"$x/L_{\rm CL}$ [–]", fontsize=FS)
    axs[1].set_ylabel(r"$\phi_L$ [mV vs SHE]", fontsize=FS)
    _add_vbar(fig, list(axs), Vs)
    os.makedirs(os.path.join(d, "figs"), exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(d, "figs", f"D_potentials_{stage}.{ext}"),
                    format=ext, dpi=(600 if ext == "svg" else PNG_DPI), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.join(d,'figs')}/D_potentials_{stage}.png/.svg")


def _load_overlays(stage):
    order = [1, 2, 3, 4, "fit", 5]
    out = []
    for s in order:
        if s == stage:
            break
        f = os.path.join(_HERE, STAGE_DIR[s], "summary.csv")
        if os.path.isfile(f):
            out.append((STAGE_LBL[s], pd.read_csv(f).sort_values("V_she")))
    return out


def run(stage):
    print(f"[plot] {STAGE_LBL[stage]}")
    plot_A1(stage, overlays=_load_overlays(stage))
    plot_A2(stage)
    plot_B1(stage)
    plot_D(stage)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "1"
    if arg == "all":
        for s in [1, 2, 3, 4, "fit", 5]:
            if os.path.isfile(os.path.join(_HERE, STAGE_DIR[s], "summary.csv")):
                run(s)
    else:
        run(int(arg) if arg.isdigit() else arg)
