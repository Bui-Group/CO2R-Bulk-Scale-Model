#!/usr/bin/env python3
"""plot_results.py — required debug plots (§12.7) and experimental comparison (§18).

Usage:  python3 plot_results.py <stage_dir> [<stage_dir> ...]
        python3 plot_results.py --overlay <dir1> <dir2> ...
All voltage x-axes are -E_cath [V vs SHE] (positive = more cathodic).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

plt.rcParams.update({
    "font.size": 8, "axes.linewidth": 0.8, "figure.dpi": 150,
    "savefig.dpi": 150, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.5, "legend.frameon": False,
})

V_RHE_TO_SHE = 2.303 * 8.314462618 * 323.15 / 96485.332 * 8.0   # 0.51305 V

E_exp_RHE = np.array([-0.816, -1.205, -1.644, -2.079, -2.413])
i_CO_exp  = np.array([35.1, 189.6, 299.1, 204.3, 64.6])
i_H2_exp  = np.array([29.9, 49.3, 119.6, 449.9, 850.5])
E_exp_SHE = E_exp_RHE - V_RHE_TO_SHE
FE_CO_exp = 100 * i_CO_exp / (i_CO_exp + i_H2_exp)
FE_H2_exp = 100 * i_H2_exp / (i_CO_exp + i_H2_exp)

RAMP = LinearSegmentedColormap.from_list(
    "mild_to_cathodic", ["#d3e5f7", "#8fb8dd", "#3f7fb5", "#1b4f81", "#08203f"])


def read_mat(path):
    with open(path, "rb") as f:
        nr, nc = np.fromfile(f, dtype=np.int64, count=2)
        d = np.fromfile(f, dtype=np.float64, count=int(nr * nc))
    return d.reshape((int(nr), int(nc)), order="F")


def read_vec(path):
    return read_mat(path)[:, 0]


def read_summary(d):
    import csv
    rows = {}
    with open(os.path.join(d, "summary.csv")) as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames
        for c in cols:
            rows[c] = []
        for r in rd:
            for c in cols:
                rows[c].append(float(r[c]))
    return {k: np.array(v) for k, v in rows.items()}


def ramp_colors(V_RHE):
    """light = mild, dark = cathodic. Returns (colors, norm) over -E_cath[V vs SHE]."""
    x = -(V_RHE - V_RHE_TO_SHE)                 # positive, increases cathodically
    norm = Normalize(vmin=x.min(), vmax=x.max())
    return RAMP(norm(x)), norm


def add_ramp_cbar(fig, axes, norm):
    sm = ScalarMappable(cmap=RAMP, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label(r"$-E_{cath}$ [V vs SHE]")
    return cb


# ── A1: 3-panel polarization summary ──────────────────────────────────────
def plot_A1(d, label, out):
    s = read_summary(d)
    x = -s["V_SHE"]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
    for a in ax:
        a.set_xlabel(r"$-E_{cath}$ [V vs SHE]")
        a.set_xlim(0.7, 3.1)
    ax[0].plot(x, s["i_CO_mAcm2"], "-", color="#1f6fb4", lw=1.5, label=r"$i_{CO}$ model")
    ax[0].plot(x, s["i_H2_mAcm2"], "-", color="#c1483a", lw=1.5, label=r"$i_{H_2}$ model")
    ax[0].plot(x, s["i_tot_mAcm2"], "-", color="#444444", lw=1.2, label=r"$i_{tot}$ model")
    ax[0].plot(-E_exp_SHE, i_CO_exp, "s", mfc="none", mec="#1f6fb4", ms=6, label="CO exp")
    ax[0].plot(-E_exp_SHE, i_H2_exp, "^", mfc="none", mec="#c1483a", ms=6, label=r"H$_2$ exp")
    ax[0].set_ylabel(r"$i$ [mA cm$^{-2}$]")
    ax[0].set_title("(a) linear polarization")
    ax[0].legend(fontsize=6.5, loc="upper left")

    ax[1].semilogy(x, np.maximum(s["i_CO_mAcm2"], 1e-4), "-", color="#1f6fb4", lw=1.5)
    ax[1].semilogy(x, np.maximum(s["i_H2_mAcm2"], 1e-4), "-", color="#c1483a", lw=1.5)
    ax[1].semilogy(x, np.maximum(s["i_tot_mAcm2"], 1e-4), "-", color="#444444", lw=1.2)
    ax[1].semilogy(-E_exp_SHE, i_CO_exp, "s", mfc="none", mec="#1f6fb4", ms=6)
    ax[1].semilogy(-E_exp_SHE, i_H2_exp, "^", mfc="none", mec="#c1483a", ms=6)
    ax[1].set_ylabel(r"$i$ [mA cm$^{-2}$]")
    ax[1].set_title("(b) log polarization")

    ax[2].plot(x, s["FE_CO_pct"], "-", color="#1f6fb4", lw=1.5, label=r"FE$_{CO}$ model")
    ax[2].plot(x, s["FE_H2_pct"], "-", color="#c1483a", lw=1.5, label=r"FE$_{H_2}$ model")
    ax[2].plot(-E_exp_SHE, FE_CO_exp, "s", mfc="none", mec="#1f6fb4", ms=6)
    ax[2].plot(-E_exp_SHE, FE_H2_exp, "^", mfc="none", mec="#c1483a", ms=6)
    ax[2].set_ylim(0, 100)
    ax[2].set_ylabel("faradaic efficiency [%]")
    ax[2].set_title("(c) faradaic efficiency")
    ax[2].legend(fontsize=6.5, loc="center left")
    fig.suptitle(f"A1 — polarization summary — {label}", y=1.02, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"A1_polarization_{label}.png"), bbox_inches="tight")
    plt.close(fig)


# ── A2: 3x3 liquid-phase diagnostics ──────────────────────────────────────
def plot_A2(d, label, out):
    V = read_vec(os.path.join(d, "V_RHE.bin"))
    x = read_vec(os.path.join(d, "x_cl_um.bin"))
    cols, norm = ramp_colors(V)
    names = [("pH", "pH [-]"), ("cCO2", r"CO$_2$(aq) [mol m$^{-3}$]"),
             ("cOH", r"OH$^-$ [mol m$^{-3}$]"), ("cHCO3", r"HCO$_3^-$ [mol m$^{-3}$]"),
             ("cCO3", r"CO$_3^{2-}$ [mol m$^{-3}$]"), ("cCs", r"Cs$^+$ [mol m$^{-3}$]"),
             ("cH", r"H$^+$ [mol m$^{-3}$]"), ("Rco", r"$R_{CO2R}$ [mol m$^{-3}$ s$^{-1}$]"),
             ("Rher", r"$R_{HER}$ [mol m$^{-3}$ s$^{-1}$]")]
    fig, axs = plt.subplots(3, 3, figsize=(11.5, 8.5), constrained_layout=True)
    tags = "abcdefghi"
    for k, (nm, lab) in enumerate(names):
        a = axs.flat[k]
        M = read_mat(os.path.join(d, f"prof_{nm}.bin"))
        for j in range(M.shape[0]):
            a.plot(x, M[j, :], lw=0.6, color=cols[j])
        a.set_xlabel(r"$x$ [$\mu$m]")
        a.set_ylabel(lab)
        a.set_title(f"({tags[k]})", loc="left", fontsize=8)
        if nm in ("cH",):
            a.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    add_ramp_cbar(fig, axs.ravel().tolist(), norm)
    fig.suptitle(f"A2 — liquid-phase diagnostics across the CL — {label}", fontsize=9)
    fig.savefig(os.path.join(out, f"A2_liquid_{label}.png"), bbox_inches="tight")
    plt.close(fig)


# ── B1: 2x2 gas-phase diagnostics ─────────────────────────────────────────
def plot_B1(d, label, out):
    p = os.path.join(d, "prof_pCO2.bin")
    if not os.path.exists(p):
        return
    V = read_vec(os.path.join(d, "V_RHE.bin"))
    x = read_vec(os.path.join(d, "x_gas_um.bin"))
    cols, norm = ramp_colors(V)
    names = [("pCO2", r"$p_{CO_2}$ [atm]"), ("pCO", r"$p_{CO}$ [atm]"),
             ("pH2", r"$p_{H_2}$ [atm]"), ("ptot", r"$p_{total}$ [Pa]")]
    fig, axs = plt.subplots(2, 2, figsize=(9.5, 6.5), constrained_layout=True)
    tags = "abcd"
    for k, (nm, lab) in enumerate(names):
        a = axs.flat[k]
        M = read_mat(os.path.join(d, f"prof_{nm}.bin"))
        for j in range(M.shape[0]):
            a.plot(x, M[j, :], lw=0.6, color=cols[j])
        a.axvline(0.0, ls="--", lw=0.8, color="0.4")
        a.set_xlabel(r"$x$ [$\mu$m]   (GDL $<0<$ CL)")
        a.set_ylabel(lab)
        a.set_title(f"({tags[k]})", loc="left", fontsize=8)
    add_ramp_cbar(fig, axs.ravel().tolist(), norm)
    fig.suptitle(f"B1 — gas-phase diagnostics across GDL+CL — {label}", fontsize=9)
    fig.savefig(os.path.join(out, f"B1_gas_{label}.png"), bbox_inches="tight")
    plt.close(fig)


# ── D1/D2: potentials ─────────────────────────────────────────────────────
def plot_D(d, label, out):
    p = os.path.join(d, "prof_phis.bin")
    if not os.path.exists(p):
        return
    V = read_vec(os.path.join(d, "V_RHE.bin"))
    xg = read_vec(os.path.join(d, "x_gas_um.bin"))
    xc = read_vec(os.path.join(d, "x_cl_um.bin"))
    PS = read_mat(p)
    PL = read_mat(os.path.join(d, "prof_phiL.bin"))
    cols, norm = ramp_colors(V)
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)
    for j in range(PS.shape[0]):
        axs[0].plot(xg, PS[j, :], lw=0.6, color=cols[j])
        axs[1].plot(xc / xc[-1], PL[j, :] * 1000, lw=0.6, color=cols[j])
    axs[0].axvline(0.0, ls="--", lw=0.8, color="0.4")
    axs[0].set_xlabel(r"$x$ [$\mu$m]   (GDL $<0<$ CL)")
    axs[0].set_ylabel(r"$\phi_s$ [V vs SHE]")
    axs[0].set_title("D1 — solid potential", loc="left", fontsize=8)
    axs[1].set_xlabel(r"$x/L_{CL}$")
    axs[1].set_ylabel(r"$\phi_L$ [mV vs SHE]")
    axs[1].set_title(r"D2 — liquid potential ($\phi_L=0$ gauge at $x=L_{CL}$)",
                     loc="left", fontsize=8)
    add_ramp_cbar(fig, axs.ravel().tolist(), norm)
    fig.suptitle(f"D — potentials — {label}", fontsize=9)
    fig.savefig(os.path.join(out, f"D_potentials_{label}.png"), bbox_inches="tight")
    plt.close(fig)

    # current-balance check figure
    s = read_summary(d)
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
    x = -s["V_SHE"]
    axs[0].semilogy(x, np.maximum(s["chg_solid"], 1e-16), "-o", ms=2.5, color="#1f6fb4")
    axs[0].axhline(1e-6, ls="--", color="r", lw=0.8)
    axs[0].set_ylabel(r"$|\Delta i|/i$ solid (collector)")
    axs[1].semilogy(x, np.maximum(s["chg_ionic"], 1e-16), "-o", ms=2.5, color="#c1483a")
    axs[1].axhline(1e-2, ls="--", color="r", lw=0.8)
    axs[1].set_ylabel(r"rel. err ionic current at $x=L_{CL}$")
    axs[2].plot(x, s["dphiL_mV"], "-o", ms=2.5, color="#1f6fb4", label=r"$\Delta\phi_L$")
    axs[2].plot(x, s["dphis_mV"], "-o", ms=2.5, color="#c1483a", label=r"$\phi_s(L_{CL})-V_{app}$")
    axs[2].axhline(0, color="0.6", lw=0.6)
    axs[2].set_ylabel("[mV]")
    axs[2].legend(fontsize=7)
    for a in axs:
        a.set_xlabel(r"$-E_{cath}$ [V vs SHE]")
        a.set_xlim(0.7, 3.1)
    fig.suptitle(f"D3 — charge-conservation and potential-drop diagnostics — {label}",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"D3_charge_{label}.png"), bbox_inches="tight")
    plt.close(fig)


# ── multi-stage polarization overlay ──────────────────────────────────────
def plot_overlay(dirs, out, name="overlay"):
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
    palette = ["#9ecae1", "#4292c6", "#2171b5", "#08519c", "#6a51a3", "#c1483a"]
    for k, d in enumerate(dirs):
        lab = os.path.basename(d.rstrip("/")).replace("output_", "")
        s = read_summary(d)
        x = -s["V_SHE"]
        c = palette[k % len(palette)]
        ax[0].plot(x, s["i_CO_mAcm2"], "-", color=c, lw=1.3, label=lab)
        ax[1].plot(x, s["i_H2_mAcm2"], "-", color=c, lw=1.3, label=lab)
        ax[2].plot(x, s["FE_CO_pct"], "-", color=c, lw=1.3, label=lab)
    ax[0].plot(-E_exp_SHE, i_CO_exp, "ks", mfc="none", ms=6, label="exp")
    ax[1].plot(-E_exp_SHE, i_H2_exp, "k^", mfc="none", ms=6, label="exp")
    ax[2].plot(-E_exp_SHE, FE_CO_exp, "ks", mfc="none", ms=6, label="exp")
    ax[0].set_ylabel(r"$i_{CO}$ [mA cm$^{-2}$]"); ax[0].set_title("(a) CO partial current")
    ax[1].set_ylabel(r"$i_{H_2}$ [mA cm$^{-2}$]"); ax[1].set_title("(b) H$_2$ partial current")
    ax[2].set_ylabel(r"FE$_{CO}$ [%]"); ax[2].set_title("(c) CO faradaic efficiency")
    ax[2].set_ylim(0, 100)
    for a in ax:
        a.set_xlabel(r"$-E_{cath}$ [V vs SHE]"); a.set_xlim(0.7, 3.1)
        a.legend(fontsize=6.5)
    fig.suptitle("stage overlay", y=1.02, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{name}.png"), bbox_inches="tight")
    plt.close(fig)


def plot_FE_overlay(d, label, out):
    """§18.2 — dedicated FE_CO / FE_H2 overlay showing the crossover."""
    s = read_summary(d)
    x = -s["V_SHE"]
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.plot(x, s["FE_CO_pct"], "-", color="#1f6fb4", lw=1.8, label=r"FE$_{CO}$ model")
    ax.plot(x, s["FE_H2_pct"], "-", color="#c1483a", lw=1.8, label=r"FE$_{H_2}$ model")
    ax.plot(-E_exp_SHE, FE_CO_exp, "s", mfc="none", mec="#1f6fb4", ms=7, label=r"FE$_{CO}$ exp")
    ax.plot(-E_exp_SHE, FE_H2_exp, "^", mfc="none", mec="#c1483a", ms=7, label=r"FE$_{H_2}$ exp")
    xc = x[np.argmin(np.abs(s["FE_CO_pct"] - s["FE_H2_pct"]))]
    ax.axvline(xc, ls=":", lw=0.9, color="0.5")
    ax.annotate(f"crossover\n{xc:.2f} V", xy=(xc, 50), xytext=(xc + 0.08, 55),
                fontsize=7, color="0.35")
    ax.set_xlabel(r"$-E_{cath}$ [V vs SHE]")
    ax.set_ylabel("faradaic efficiency [%]")
    ax.set_xlim(0.7, 3.1); ax.set_ylim(0, 100)
    ax.legend(fontsize=7)
    ax.set_title(f"FE crossover — {label}", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"FE_overlay_{label}.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    args = sys.argv[1:]
    if args and args[0] == "--overlay":
        dirs = args[1:]
        out = os.path.dirname(os.path.abspath(dirs[-1]))
        out = os.path.join(out, "figures")
        os.makedirs(out, exist_ok=True)
        plot_overlay(dirs, out)
        print("wrote overlay ->", out)
        return
    for d in args:
        d = os.path.abspath(d)
        label = os.path.basename(d).replace("output_", "")
        out = os.path.join(os.path.dirname(d), "figures")
        os.makedirs(out, exist_ok=True)
        plot_A1(d, label, out)
        plot_FE_overlay(d, label, out)
        plot_A2(d, label, out)
        plot_B1(d, label, out)
        plot_D(d, label, out)
        print("wrote figures for", label, "->", out)


if __name__ == "__main__":
    main()
