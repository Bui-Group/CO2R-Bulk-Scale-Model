#!/usr/bin/env python3
"""
plot_results.py -- figures for every stage (§12.7, §18).

Required-plot summary table (§12.7):
    A1  3-panel polarization summary (i, log-i, FE vs experiment)  stages 1-5 + fit
    A2  3x3 liquid-phase diagnostic at all sweep voltages          stages 1-5 + fit
    B1  2x2 gas-phase diagnostic at all sweep voltages             stages 3-5 + fit
    D1  phi_s profile across GDL+CL                                stage 5
    D2  phi_L profile across the CL                                stage 5

All voltage axes are -E_cath [V vs SHE], positive = more cathodic to the right
(§12.7, §18.1).  Experimental data: open markers (square CO, triangle H2).

Usage:  python3 plot_results.py [stage_tag ...]        e.g. stage1 stage4 fit stage5
        python3 plot_results.py --all
"""
import os, sys, struct
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- §15.5 / §16.1 experimental data, Romiluyi et al. 2023 ------------------
V_RHE_TO_SHE = 2.303 * 8.314462618 * 323.15 / 96485.332 * 8.0      # 0.51305 V
E_EXP_RHE = np.array([-0.816, -1.205, -1.644, -2.079, -2.413])
E_EXP_SHE = E_EXP_RHE - V_RHE_TO_SHE
I_CO_EXP  = np.array([35.1, 189.6, 299.1, 204.3, 64.6])
I_H2_EXP  = np.array([29.9, 49.3, 119.6, 449.9, 850.5])
I_TOT_EXP = I_CO_EXP + I_H2_EXP
FE_CO_EXP = 100 * I_CO_EXP / I_TOT_EXP
FE_H2_EXP = 100 - FE_CO_EXP

C_CO, C_H2, C_TOT = "#1f5fa9", "#c0392b", "#2f2f2f"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.linewidth": 0.8, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.5, "legend.frameon": False, "legend.fontsize": 8,
    "axes.titlesize": 10, "axes.labelsize": 9,
})

# One truncated colormap object is used for BOTH the lines and the colourbar,
# so the bar direction cannot disagree with the line colours (§12.7 A2 warning).
_base = plt.get_cmap("Blues")
RAMP = LinearSegmentedColormap.from_list("ramp", _base(np.linspace(0.25, 1.0, 256)))


def _read_csv(path):
    return np.genfromtxt(path, delimiter=",", names=True)


def _read_liquid(path):
    with open(path, "rb") as f:
        nV, Nc = struct.unpack("<qq", f.read(16))
        x = np.frombuffer(f.read(8 * Nc), dtype="<f8")
        V = np.frombuffer(f.read(8 * nV), dtype="<f8")
        d = np.frombuffer(f.read(8 * nV * 9 * Nc), dtype="<f8").reshape(nV, 9, Nc)
    return x, V, d


def _read_gas(path):
    with open(path, "rb") as f:
        nV, Nt, Ng = struct.unpack("<qqq", f.read(24))
        x = np.frombuffer(f.read(8 * Nt), dtype="<f8")
        V = np.frombuffer(f.read(8 * nV), dtype="<f8")
        d = np.frombuffer(f.read(8 * nV * 4 * Nt), dtype="<f8").reshape(nV, 4, Nt)
    return x, V, d, Ng


def _read_pot(path):
    with open(path, "rb") as f:
        nV, Ng, Nc = struct.unpack("<qqq", f.read(24))
        xs = np.frombuffer(f.read(8 * (Ng + Nc)), dtype="<f8")
        xl = np.frombuffer(f.read(8 * Nc), dtype="<f8")
        V = np.frombuffer(f.read(8 * nV), dtype="<f8")
        raw = np.frombuffer(f.read(8 * nV * (Ng + 2 * Nc)), dtype="<f8")
    raw = raw.reshape(nV, Ng + 2 * Nc)
    return xs, xl, V, raw[:, :Ng + Nc], raw[:, Ng + Nc:], Ng


def _ramp_setup(V_rhe):
    """Colour every sweep voltage: light = mild, dark = cathodic."""
    mV = -np.asarray(V_rhe)                    # 0.300 (mild) .. 2.500 (cathodic)
    norm = Normalize(vmin=mV.min(), vmax=mV.max())
    return norm, [RAMP(norm(v)) for v in mV]


def _add_cbar(fig, axes, norm, label=r"$-E_{\rm cath}$ [V vs RHE@pH8]"):
    sm = ScalarMappable(norm=norm, cmap=RAMP); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.015)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    return cb


# ---------------------------------------------------------------- A1 --------
def fig_A1(tag, od, cmp_tags=()):
    d = _read_csv(os.path.join(od, "polarization.csv"))
    mV = -d["V_SHE"]
    fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.9))

    for a in ax[:2]:
        a.plot(mV, d["i_CO_mAcm2"],  "-", color=C_CO,  lw=1.7, label=r"model $i_{\rm CO}$")
        a.plot(mV, d["i_H2_mAcm2"],  "-", color=C_H2,  lw=1.7, label=r"model $i_{\rm H_2}$")
        a.plot(mV, d["i_tot_mAcm2"], color=C_TOT, lw=1.2, ls="--", label=r"model $i_{\rm tot}$")
        a.plot(-E_EXP_SHE, I_CO_EXP, "s", mfc="none", mec=C_CO, ms=7, mew=1.4, ls="none", label=r"exp $i_{\rm CO}$")
        a.plot(-E_EXP_SHE, I_H2_EXP, "^", mfc="none", mec=C_H2, ms=7, mew=1.4, ls="none", label=r"exp $i_{\rm H_2}$")
        a.set_xlabel(r"$-E_{\rm cath}$ [V vs SHE]")
        a.set_ylabel(r"$i$ [mA cm$^{-2}$]")
        a.set_xlim(0.7, 3.1)

    # overlay other stages for context
    for ct in cmp_tags:
        p = os.path.join(HERE, "output_" + ct, "polarization.csv")
        if not os.path.exists(p):
            continue
        o = _read_csv(p)
        for a in ax[:2]:
            a.plot(-o["V_SHE"], o["i_CO_mAcm2"], color=C_CO, lw=0.9, alpha=0.42, ls=":")
            a.plot(-o["V_SHE"], o["i_H2_mAcm2"], color=C_H2, lw=0.9, alpha=0.42, ls=":")
        ax[2].plot(-o["V_SHE"], o["FE_CO_pct"], color=C_CO, lw=0.9, alpha=0.42, ls=":")

    ax[0].set_title("(a) linear polarization")
    ax[0].legend(loc="upper left", ncol=1)
    ax[1].set_yscale("log"); ax[1].set_title("(b) log polarization")
    ax[1].set_ylim(1e-3, 3e3)

    ax[2].plot(mV, d["FE_CO_pct"], "-", color=C_CO, lw=1.7, label=r"model FE$_{\rm CO}$")
    ax[2].plot(mV, d["FE_H2_pct"], "-", color=C_H2, lw=1.7, label=r"model FE$_{\rm H_2}$")
    ax[2].plot(-E_EXP_SHE, FE_CO_EXP, "s", mfc="none", mec=C_CO, ms=7, mew=1.4, ls="none", label=r"exp FE$_{\rm CO}$")
    ax[2].plot(-E_EXP_SHE, FE_H2_EXP, "^", mfc="none", mec=C_H2, ms=7, mew=1.4, ls="none", label=r"exp FE$_{\rm H_2}$")
    ax[2].set_xlabel(r"$-E_{\rm cath}$ [V vs SHE]"); ax[2].set_ylabel("FE [%]")
    ax[2].set_xlim(0.7, 3.1); ax[2].set_ylim(0, 100)
    ax[2].set_title("(c) faradaic efficiency"); ax[2].legend(loc="upper right")

    ttl = tag + (" (dotted: " + ", ".join(cmp_tags) + ")" if cmp_tags else "")
    fig.suptitle("A1  polarization summary — " + ttl, y=1.005, fontsize=11)
    fig.tight_layout()
    out = os.path.join(od, f"A1_polarization_{tag}.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


# ---------------------------------------------------------------- A2 --------
A2_TITLES = [r"(a) pH  [$-$]", r"(b) CO$_2$(aq) [mol m$^{-3}$]", r"(c) OH$^-$ [mol m$^{-3}$]",
             r"(d) HCO$_3^-$ [mol m$^{-3}$]", r"(e) CO$_3^{2-}$ [mol m$^{-3}$]", r"(f) Cs$^+$ [mol m$^{-3}$]",
             r"(g) H$^+$ [mol m$^{-3}$]", r"(h) $R_{\rm CO_2R}$ [mol m$^{-3}$ s$^{-1}$]",
             r"(i) $R_{\rm HER}$ [mol m$^{-3}$ s$^{-1}$]"]


def fig_A2(tag, od):
    x, V, d = _read_liquid(os.path.join(od, "profiles_liquid.bin"))
    xum = x * 1e6
    norm, cols = _ramp_setup(V)
    fig, axes = plt.subplots(3, 3, figsize=(11.6, 8.6), sharex=True)
    for p in range(9):
        a = axes[p // 3][p % 3]
        for k in range(len(V)):
            a.plot(xum, d[k, p, :], color=cols[k], lw=0.6)
        a.set_title(A2_TITLES[p])
        if p == 6:
            a.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        if p >= 6:
            a.set_xlabel(r"$x$ [$\mu$m]   (0 = GDL side)")
    _add_cbar(fig, axes.ravel().tolist(), norm)
    fig.suptitle(f"A2  liquid-phase diagnostics across the CL, all {len(V)} sweep voltages — {tag}",
                 y=0.995, fontsize=11)
    out = os.path.join(od, f"A2_liquid_{tag}.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


# ---------------------------------------------------------------- B1 --------
def fig_B1(tag, od):
    p = os.path.join(od, "profiles_gas.bin")
    if not os.path.exists(p):
        return None
    x, V, d, Ng = _read_gas(p)
    xum = x * 1e6
    norm, cols = _ramp_setup(V)
    titles = [r"(a) $p_{\rm CO_2}$ [atm]", r"(b) $p_{\rm CO}$ [atm]",
              r"(c) $p_{\rm H_2}$ [atm]", r"(d) $p_{\rm total}$ [Pa]"]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.0), sharex=True)
    for q in range(4):
        a = axes[q // 2][q % 2]
        for k in range(len(V)):
            a.plot(xum, d[k, q, :], color=cols[k], lw=0.6)
        a.axvline(0.0, color="0.35", ls="--", lw=0.9)
        a.set_title(titles[q])
        if q >= 2:
            a.set_xlabel(r"$x$ [$\mu$m]   (GDL inlet $\to$ CL; dashed = GDL/CL interface)")
    _add_cbar(fig, axes.ravel().tolist(), norm)
    fig.suptitle(f"B1  gas-phase diagnostics across GDL+CL, all {len(V)} sweep voltages — {tag}",
                 y=0.995, fontsize=11)
    out = os.path.join(od, f"B1_gas_{tag}.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


# ------------------------------------------------------------- D1 / D2 ------
def fig_D(tag, od):
    p = os.path.join(od, "profiles_potential.bin")
    if not os.path.exists(p):
        return None
    xs, xl, V, phis, phil, Ng = _read_pot(p)
    norm, cols = _ramp_setup(V)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))
    for k in range(len(V)):
        axes[0].plot(xs * 1e6, phis[k], color=cols[k], lw=0.6)
        axes[1].plot(xl / xl[-1], phil[k] * 1e3, color=cols[k], lw=0.6)
    axes[0].axvline(0.0, color="0.35", ls="--", lw=0.9)
    axes[0].set_xlabel(r"$x$ [$\mu$m]   ($-L_{\rm GDL}$ = collector; dashed = GDL/CL interface)")
    axes[0].set_ylabel(r"$\varphi_s$ [V vs SHE]")
    axes[0].set_title(r"D1  solid potential $\varphi_s$ across GDL+CL")
    axes[1].set_xlabel(r"$x/L_{\rm CL}$")
    axes[1].set_ylabel(r"$\varphi_L$ [mV vs SHE]")
    axes[1].set_title(r"D2  liquid potential $\varphi_L$ across the CL ($\varphi_L=0$ gauge at $x=L_{\rm CL}$)")
    _add_cbar(fig, axes.tolist(), norm)
    fig.suptitle(f"D1/D2  potential profiles, all {len(V)} sweep voltages — {tag}", y=1.0, fontsize=11)
    out = os.path.join(od, f"D_potentials_{tag}.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


CMP = {"stage1": (), "stage2": ("stage1",), "stage3": ("stage1", "stage2"),
       "stage4": ("stage3",), "fit": ("stage4",),
       "stage5": ("stage1", "stage2", "stage3", "stage4", "fit")}


def run(tag):
    od = os.path.join(HERE, "output_" + tag)
    if not os.path.isdir(od):
        print(f"  ! no output dir for {tag}"); return
    made = [fig_A1(tag, od, CMP.get(tag, ())), fig_A2(tag, od), fig_B1(tag, od), fig_D(tag, od)]
    for m in made:
        if m:
            print("  wrote", os.path.relpath(m, HERE))


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "--all":
        args = [t for t in ("stage1", "stage2", "stage3", "stage4", "fit", "stage5")
                if os.path.isdir(os.path.join(HERE, "output_" + t))]
    for t in args:
        print("plotting", t)
        run(t)
