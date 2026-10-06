#!/usr/bin/env python3
"""plot_results.py — all required debug plots (§12.7, §18).

Usage:  python3 plot_results.py <tag> [<tag> ...]        e.g. stage1 stage5
        python3 plot_results.py --final                   (§18 experimental figs)

All voltage x-axes are  -E_cath [V vs SHE]  (positive = more cathodic).
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)

V_RHE_TO_SHE = 2.303 * 8.314462618 * 323.15 / 96485.332 * 8.0    # 0.51305 V

# §16.1 Romiluyi et al. 2023
E_exp_RHE = np.array([-0.816, -1.205, -1.644, -2.079, -2.413])
i_CO_exp = np.array([35.1, 189.6, 299.1, 204.3, 64.6])
i_H2_exp = np.array([29.9, 49.3, 119.6, 449.9, 850.5])
E_exp_SHE = E_exp_RHE - V_RHE_TO_SHE
FE_CO_exp = 100 * i_CO_exp / (i_CO_exp + i_H2_exp)
FE_H2_exp = 100 * i_H2_exp / (i_CO_exp + i_H2_exp)

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
                     "legend.fontsize": 8, "figure.dpi": 130,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})

C_CO, C_H2, C_TOT = "#1f6fb4", "#c8442c", "#2f2f2f"


def load_csv(tag, kind):
    fn = os.path.join(DATA, f"{tag}_{kind}.csv")
    if not os.path.exists(fn):
        return None
    return np.genfromtxt(fn, delimiter=",", names=True)


def _ramp():
    """Light -> dark blue ramp; the colourbar is built from THIS object and
    THIS norm, so bar direction and line colours cannot disagree."""
    return LinearSegmentedColormap.from_list(
        "mild2cathodic", ["#cfe3f5", "#7fb2de", "#2f6fae", "#123a63", "#05172b"])


def _add_ramp_bar(fig, axes, vmin, vmax, cmap, norm):
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.015)
    cb.set_label(r"$-E_{\rm cath}$  [V vs RHE@pH8]")
    return cb


# ── A1: 3-panel polarization summary ──────────────────────────────────────
def plot_A1(tag, extra=None, title=None):
    d = load_csv(tag, "polarization")
    if d is None:
        print(f"  (no polarization data for {tag})"); return
    x = -d["V_SHE"]
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.5))

    for a, logy in ((ax[0], False), (ax[1], True)):
        a.plot(x, d["i_CO_mAcm2"], "-", color=C_CO, lw=1.6, label=r"$i_{\rm CO}$ model")
        a.plot(x, d["i_H2_mAcm2"], "-", color=C_H2, lw=1.6, label=r"$i_{\rm H_2}$ model")
        a.plot(x, d["i_tot_mAcm2"], color=C_TOT, lw=1.2, ls="--", label=r"$i_{\rm tot}$ model")
        if extra is not None:
            a.plot(-extra["V_SHE"], extra["i_CO_mAcm2"], ":", color=C_CO, lw=1.2, alpha=.8)
            a.plot(-extra["V_SHE"], extra["i_H2_mAcm2"], ":", color=C_H2, lw=1.2, alpha=.8)
        a.plot(-E_exp_SHE, i_CO_exp, "s", mfc="none", mec=C_CO, ms=6, mew=1.3, label=r"$i_{\rm CO}$ exp")
        a.plot(-E_exp_SHE, i_H2_exp, "^", mfc="none", mec=C_H2, ms=6, mew=1.3, label=r"$i_{\rm H_2}$ exp")
        a.set_xlabel(r"$-E_{\rm cath}$  [V vs SHE]")
        a.set_ylabel(r"$i$  [mA cm$^{-2}$]")
        a.set_xlim(0.7, 3.1)
        if logy:
            a.set_yscale("log"); a.set_ylim(1e-3, 5e3); a.set_title("(b) log polarization")
        else:
            a.set_title("(a) linear polarization")
    ax[0].legend(loc="upper left", frameon=False)

    ax[2].plot(x, d["FE_CO"], "-", color=C_CO, lw=1.6, label=r"FE$_{\rm CO}$ model")
    ax[2].plot(x, d["FE_H2"], "-", color=C_H2, lw=1.6, label=r"FE$_{\rm H_2}$ model")
    if extra is not None:
        ax[2].plot(-extra["V_SHE"], extra["FE_CO"], ":", color=C_CO, lw=1.2, alpha=.8)
        ax[2].plot(-extra["V_SHE"], extra["FE_H2"], ":", color=C_H2, lw=1.2, alpha=.8)
    ax[2].plot(-E_exp_SHE, FE_CO_exp, "s", mfc="none", mec=C_CO, ms=6, mew=1.3)
    ax[2].plot(-E_exp_SHE, FE_H2_exp, "^", mfc="none", mec=C_H2, ms=6, mew=1.3)
    ax[2].set_xlabel(r"$-E_{\rm cath}$  [V vs SHE]"); ax[2].set_ylabel("FE  [%]")
    ax[2].set_xlim(0.7, 3.1); ax[2].set_ylim(0, 100); ax[2].set_title("(c) faradaic efficiency")
    ax[2].legend(loc="upper right", frameon=False)

    fig.suptitle(title or f"A1 — polarization summary ({tag})", y=1.02)
    fig.tight_layout()
    fn = os.path.join(FIGS, f"{tag}_A1_polarization.png")
    fig.savefig(fn, bbox_inches="tight"); plt.close(fig)
    print("  wrote", os.path.relpath(fn, HERE))


# ── A2: 3x3 liquid-phase diagnostic ───────────────────────────────────────
def plot_A2(tag):
    d = load_csv(tag, "liquid")
    if d is None:
        print(f"  (no liquid data for {tag})"); return
    Vs = np.unique(d["V_RHE"])[::-1]                 # mild -> cathodic
    cmap = _ramp()
    norm = Normalize(vmin=np.min(-Vs), vmax=np.max(-Vs))
    panels = [("pH", "pH  [-]"), ("cCO2", r"CO$_2$(aq)  [mol m$^{-3}$]"),
              ("cOH", r"OH$^-$  [mol m$^{-3}$]"), ("cHCO3", r"HCO$_3^-$  [mol m$^{-3}$]"),
              ("cCO3", r"CO$_3^{2-}$  [mol m$^{-3}$]"), ("cCs", r"Cs$^+$  [mol m$^{-3}$]"),
              ("cH", r"H$^+$  [mol m$^{-3}$]"), ("R_CO2R", r"$R_{\rm CO2R}$  [mol m$^{-3}$ s$^{-1}$]"),
              ("R_HER", r"$R_{\rm HER}$  [mol m$^{-3}$ s$^{-1}$]")]
    fig, axes = plt.subplots(3, 3, figsize=(12, 9.0), constrained_layout=True)
    tags = "abcdefghi"
    for V in Vs:
        m = d["V_RHE"] == V
        x = d["x_um"][m]
        col = cmap(norm(-V))
        for k, (key, _) in enumerate(panels):
            axes.flat[k].plot(x, d[key][m], "-", color=col, lw=0.7)
    for k, (key, lab) in enumerate(panels):
        a = axes.flat[k]
        a.set_xlabel(r"$x$  [$\mu$m]"); a.set_ylabel(lab)
        a.set_title(f"({tags[k]})", loc="left", fontsize=9)
        if key == "cH":
            a.set_yscale("log")
    _add_ramp_bar(fig, axes.ravel().tolist(), np.min(-Vs), np.max(-Vs), cmap, norm)
    fig.suptitle(f"A2 — liquid-phase diagnostics across the CL, all {len(Vs)} voltages ({tag})")
    fn = os.path.join(FIGS, f"{tag}_A2_liquid.png")
    fig.savefig(fn, bbox_inches="tight"); plt.close(fig)
    print("  wrote", os.path.relpath(fn, HERE))


# ── B1: 2x2 gas-phase diagnostic ──────────────────────────────────────────
def plot_B1(tag):
    d = load_csv(tag, "gas")
    if d is None:
        print(f"  (no gas data for {tag})"); return
    Vs = np.unique(d["V_RHE"])[::-1]
    cmap = _ramp(); norm = Normalize(vmin=np.min(-Vs), vmax=np.max(-Vs))
    panels = [("p_CO2_atm", r"$p_{\rm CO_2}$  [atm]"), ("p_CO_atm", r"$p_{\rm CO}$  [atm]"),
              ("p_H2_atm", r"$p_{\rm H_2}$  [atm]"), ("p_tot_Pa", r"$p_{\rm total}$  [Pa]")]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), constrained_layout=True)
    tags = "abcd"
    for V in Vs:
        m = d["V_RHE"] == V
        x = d["x_um"][m]; col = cmap(norm(-V))
        for k, (key, _) in enumerate(panels):
            axes.flat[k].plot(x, d[key][m], "-", color=col, lw=0.7)
    for k, (key, lab) in enumerate(panels):
        a = axes.flat[k]
        a.axvline(0.0, color="0.35", ls="--", lw=0.9)
        a.set_xlabel(r"$x$  [$\mu$m]   (0 = GDL/CL interface)"); a.set_ylabel(lab)
        a.set_title(f"({tags[k]})", loc="left", fontsize=9)
    _add_ramp_bar(fig, axes.ravel().tolist(), np.min(-Vs), np.max(-Vs), cmap, norm)
    fig.suptitle(f"B1 — gas-phase diagnostics across GDL+CL, all {len(Vs)} voltages ({tag})")
    fn = os.path.join(FIGS, f"{tag}_B1_gas.png")
    fig.savefig(fn, bbox_inches="tight"); plt.close(fig)
    print("  wrote", os.path.relpath(fn, HERE))


# ── D1 / D2: potential profiles ───────────────────────────────────────────
def plot_D(tag):
    ds = load_csv(tag, "phis"); dl = load_csv(tag, "phiL")
    if ds is None or dl is None:
        return
    sel = [-0.300, -0.825, -1.200, -1.650, -2.075, -2.500]
    cmap = _ramp(); norm = Normalize(vmin=0.3, vmax=2.5)
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.6))
    for V in sel:
        Vs = np.unique(ds["V_RHE"]); Vn = Vs[np.argmin(abs(Vs - V))]
        m = ds["V_RHE"] == Vn
        ax[0].plot(ds["x_um"][m], ds["phi_s"][m], "-", color=cmap(norm(-Vn)), lw=1.4,
                   label=f"{Vn:.3f}")
        ax[2].plot(ds["x_um"][m], 1e3 * (ds["phi_s"][m] - (Vn - V_RHE_TO_SHE)),
                   "-", color=cmap(norm(-Vn)), lw=1.4, label=f"{Vn:.3f}")
        m = dl["V_RHE"] == Vn
        ax[1].plot(dl["x_norm"][m], 1e3 * dl["phi_L"][m], "-", color=cmap(norm(-Vn)),
                   lw=1.4, label=f"{Vn:.3f}")
    ax[0].axvline(0.0, color="0.35", ls="--", lw=0.9)
    ax[0].set_xlabel(r"$x$  [$\mu$m]   ($-L_{\rm GDL}\to L_{\rm CL}$)")
    ax[0].set_ylabel(r"$\phi_s$  [V vs SHE]"); ax[0].set_title("D1 — solid potential (GDL+CL)")
    ax[0].legend(title=r"$E_{\rm cath}$ [V vs RHE]", frameon=False, fontsize=7, ncol=2)
    ax[1].set_xlabel(r"$x/L_{\rm CL}$"); ax[1].set_ylabel(r"$\phi_L$  [mV vs SHE]")
    ax[1].set_title(r"D2 — liquid potential (CL), gauge $\phi_L(L_{\rm CL})=0$")
    ax[1].legend(title=r"$E_{\rm cath}$ [V vs RHE]", frameon=False, fontsize=7, ncol=2)
    ax[2].axvline(0.0, color="0.35", ls="--", lw=0.9)
    ax[2].set_xlabel(r"$x$  [$\mu$m]"); ax[2].set_ylabel(r"$\phi_s - V_{\rm applied}$  [mV]")
    ax[2].set_title("solid ohmic rise from the collector")
    ax[2].legend(title=r"$E_{\rm cath}$ [V vs RHE]", frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fn = os.path.join(FIGS, f"{tag}_D_potentials.png")
    fig.savefig(fn, bbox_inches="tight"); plt.close(fig)
    print("  wrote", os.path.relpath(fn, HERE))


# ── overlay of several stages ─────────────────────────────────────────────
def plot_overlay(tags, name="overlay", labels=None):
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.5))
    cols = plt.cm.viridis(np.linspace(0, 0.88, len(tags)))
    for t, c in zip(tags, cols):
        d = load_csv(t, "polarization")
        if d is None: continue
        lbl = (labels or {}).get(t, t)
        x = -d["V_SHE"]
        for a, logy in ((ax[0], False), (ax[1], True)):
            a.plot(x, d["i_CO_mAcm2"], "-", color=c, lw=1.5, label=f"{lbl} CO")
            a.plot(x, d["i_H2_mAcm2"], "--", color=c, lw=1.2, label=f"{lbl} H$_2$")
        ax[2].plot(x, d["FE_CO"], "-", color=c, lw=1.5, label=lbl)
    for a in ax[:2]:
        a.plot(-E_exp_SHE, i_CO_exp, "s", mfc="none", mec="k", ms=6)
        a.plot(-E_exp_SHE, i_H2_exp, "^", mfc="none", mec="k", ms=6)
        a.set_xlabel(r"$-E_{\rm cath}$  [V vs SHE]"); a.set_ylabel(r"$i$  [mA cm$^{-2}$]")
        a.set_xlim(0.7, 3.1)
    ax[1].set_yscale("log"); ax[1].set_ylim(1e-3, 5e3)
    ax[0].set_title("(a) linear"); ax[1].set_title("(b) log")
    ax[0].legend(frameon=False, fontsize=6.5, ncol=2, loc="upper left")
    ax[2].plot(-E_exp_SHE, FE_CO_exp, "s", mfc="none", mec="k", ms=6)
    ax[2].set_xlabel(r"$-E_{\rm cath}$  [V vs SHE]"); ax[2].set_ylabel(r"FE$_{\rm CO}$  [%]")
    ax[2].set_xlim(0.7, 3.1); ax[2].set_ylim(0, 100); ax[2].set_title("(c) FE$_{CO}$")
    ax[2].legend(frameon=False, fontsize=7)
    fig.suptitle("stage overlay — partial currents and FE", y=1.02)
    fig.tight_layout()
    fn = os.path.join(FIGS, f"{name}_polarization.png")
    fig.savefig(fn, bbox_inches="tight"); plt.close(fig)
    print("  wrote", os.path.relpath(fn, HERE))


def plot_FE_overlay(tag):
    """§18.2 dedicated FE_CO / FE_H2 figure."""
    d = load_csv(tag, "polarization")
    if d is None: return
    x = -d["V_SHE"]
    fig, a = plt.subplots(figsize=(5.2, 3.8))
    a.plot(x, d["FE_CO"], "-", color=C_CO, lw=1.8, label=r"FE$_{\rm CO}$ model")
    a.plot(x, d["FE_H2"], "-", color=C_H2, lw=1.8, label=r"FE$_{\rm H_2}$ model")
    a.plot(-E_exp_SHE, FE_CO_exp, "s", mfc="none", mec=C_CO, ms=7, mew=1.4, label=r"FE$_{\rm CO}$ exp")
    a.plot(-E_exp_SHE, FE_H2_exp, "^", mfc="none", mec=C_H2, ms=7, mew=1.4, label=r"FE$_{\rm H_2}$ exp")
    a.set_xlabel(r"$-E_{\rm cath}$  [V vs SHE]"); a.set_ylabel("FE  [%]")
    a.set_xlim(0.7, 3.1); a.set_ylim(0, 100); a.legend(frameon=False)
    a.set_title(f"FE crossover ({tag})")
    fig.tight_layout()
    fn = os.path.join(FIGS, f"{tag}_FE_overlay.png")
    fig.savefig(fn, bbox_inches="tight"); plt.close(fig)
    print("  wrote", os.path.relpath(fn, HERE))


def do_tag(tag, extra_tag=None):
    print(f"[{tag}]")
    extra = load_csv(extra_tag, "polarization") if extra_tag else None
    plot_A1(tag, extra=extra)
    plot_A2(tag)
    plot_B1(tag)
    plot_D(tag)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__); sys.exit(0)
    if args[0] == "--final":
        plot_overlay(["stage1", "stage2", "stage3", "stage4", "fit", "stage5"],
                     name="all_stages",
                     labels={"stage1": "S1", "stage2": "S2", "stage3": "S3",
                             "stage4": "S4", "fit": "S4-fitted", "stage5": "S5"})
        plot_A1("stage5", title="§18.1 — final model vs Romiluyi et al. 2023 (Stage 5)")
        plot_FE_overlay("stage5")
    else:
        for t in args:
            do_tag(t)
