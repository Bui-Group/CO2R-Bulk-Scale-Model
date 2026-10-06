#!/usr/bin/env python3
"""plot_results.py — §12.7 debug plots for every stage.

Usage:  python3 plot_results.py <stage_tag> [<stage_tag> ...]
        python3 plot_results.py --overlay tag1 tag2 ...   (polarization overlay)

Reads output_<tag>/polarization.csv and output_<tag>/profiles.bin written by
the Julia drivers.  All voltage axes are -E_cath [V vs SHE] (§12.7).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

HERE = os.path.dirname(os.path.abspath(__file__))
V_RHE_TO_SHE = 2.303 * 8.314462618 * 323.15 / 96485.332 * 8.0   # 0.51305 V

E_exp_RHE = np.array([-0.816, -1.205, -1.644, -2.079, -2.413])
i_CO_exp  = np.array([35.1, 189.6, 299.1, 204.3, 64.6])
i_H2_exp  = np.array([29.9, 49.3, 119.6, 449.9, 850.5])
E_exp_SHE = E_exp_RHE - V_RHE_TO_SHE
FE_CO_exp = i_CO_exp / (i_CO_exp + i_H2_exp) * 100
FE_H2_exp = i_H2_exp / (i_CO_exp + i_H2_exp) * 100

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 130, "savefig.dpi": 150, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "lines.linewidth": 1.4,
})

C_CO, C_H2, C_TOT = "#1f5fa9", "#c0392b", "#2c3e50"


def outdir(tag):
    return os.path.join(HERE, "output_%s" % tag)


def load_pol(tag):
    fn = os.path.join(outdir(tag), "polarization.csv")
    d = np.genfromtxt(fn, delimiter=",", names=True)
    return d


def load_profiles(tag):
    fn = os.path.join(outdir(tag), "profiles.bin")
    raw = open(fn, "rb").read()
    hdr = np.frombuffer(raw[:40], dtype=np.int64)
    nV, N_CL, N_G, has_gas, has_pot = [int(v) for v in hdr]
    off = 40
    def take(n):
        nonlocal off
        a = np.frombuffer(raw[off:off + 8 * n], dtype=np.float64)
        off += 8 * n
        return a.copy()
    x_cl = take(N_CL); x_g = take(N_G); V = take(nV)
    liq = np.zeros((nV, 9, N_CL))
    gas = np.zeros((nV, 4, N_G)) if has_gas else None
    pot_s = np.zeros((nV, N_G)) if has_pot else None
    pot_L = np.zeros((nV, N_CL)) if has_pot else None
    for k in range(nV):
        for j in range(9):
            liq[k, j, :] = take(N_CL)
        if has_gas:
            for j in range(4):
                gas[k, j, :] = take(N_G)
        if has_pot:
            pot_s[k, :] = take(N_G)
            pot_L[k, :] = take(N_CL)
    return dict(nV=nV, N_CL=N_CL, N_G=N_G, x_cl=x_cl, x_g=x_g, V=V,
                liq=liq, gas=gas, pot_s=pot_s, pot_L=pot_L,
                has_gas=bool(has_gas), has_pot=bool(has_pot))


# ─────────────────────────────────────────────────────────────────────────────
def plot_A1(tag, title=None):
    d = load_pol(tag)
    mx = -d["V_SHE"]
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.4))
    for a in ax:
        a.set_xlabel(r"$-E_\mathrm{cath}$ [V vs SHE]")
        a.set_xlim(0.7, 3.1)
    ax[0].plot(mx, d["i_CO_mAcm2"], color=C_CO, label=r"$i_\mathrm{CO}$ model")
    ax[0].plot(mx, d["i_H2_mAcm2"], color=C_H2, label=r"$i_\mathrm{H_2}$ model")
    ax[0].plot(mx, d["i_tot_mAcm2"], color=C_TOT, ls="--", label=r"$i_\mathrm{tot}$ model")
    ax[0].plot(-E_exp_SHE, i_CO_exp, "s", mfc="none", mec=C_CO, ms=6, label=r"$i_\mathrm{CO}$ exp")
    ax[0].plot(-E_exp_SHE, i_H2_exp, "^", mfc="none", mec=C_H2, ms=6, label=r"$i_\mathrm{H_2}$ exp")
    ax[0].set_ylabel(r"$i$ [mA cm$^{-2}$]"); ax[0].set_title("(a) linear polarization")
    ax[0].legend(frameon=False, ncol=1)

    ax[1].semilogy(mx, np.maximum(d["i_CO_mAcm2"], 1e-4), color=C_CO)
    ax[1].semilogy(mx, np.maximum(d["i_H2_mAcm2"], 1e-4), color=C_H2)
    ax[1].semilogy(mx, np.maximum(d["i_tot_mAcm2"], 1e-4), color=C_TOT, ls="--")
    ax[1].semilogy(-E_exp_SHE, i_CO_exp, "s", mfc="none", mec=C_CO, ms=6)
    ax[1].semilogy(-E_exp_SHE, i_H2_exp, "^", mfc="none", mec=C_H2, ms=6)
    ax[1].set_ylabel(r"$i$ [mA cm$^{-2}$]"); ax[1].set_title("(b) log polarization")
    ax[1].set_ylim(1e-3, 1e4)

    itot = np.maximum(d["i_tot_mAcm2"], 1e-30)
    feco = 100 * d["i_CO_mAcm2"] / itot
    ax[2].plot(mx, feco, color=C_CO, label=r"FE$_\mathrm{CO}$ model")
    ax[2].plot(mx, 100 - feco, color=C_H2, label=r"FE$_\mathrm{H_2}$ model")
    ax[2].plot(-E_exp_SHE, FE_CO_exp, "s", mfc="none", mec=C_CO, ms=6, label=r"FE$_\mathrm{CO}$ exp")
    ax[2].plot(-E_exp_SHE, FE_H2_exp, "^", mfc="none", mec=C_H2, ms=6, label=r"FE$_\mathrm{H_2}$ exp")
    ax[2].set_ylabel("FE [%]"); ax[2].set_ylim(0, 100); ax[2].set_title("(c) faradaic efficiency")
    ax[2].legend(frameon=False)

    fig.suptitle(title or ("A1  polarization summary — %s" % tag), y=1.02)
    fig.tight_layout()
    fn = os.path.join(outdir(tag), "A1_polarization.png")
    fig.savefig(fn, bbox_inches="tight"); plt.close(fig)
    return fn


def _ramp(ax_list, fig, V, cmap="viridis_r"):
    """Shared colourbar; light = mild V, dark = cathodic (cmap reversed once)."""
    norm = Normalize(vmin=np.min(-V), vmax=np.max(-V))
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax_list, fraction=0.02, pad=0.015)
    cb.set_label(r"$-E_\mathrm{cath}$ [V vs RHE@pH8]")
    return sm, norm


def plot_A2(tag):
    d = load_profiles(tag)
    labels = ["(a) pH [–]", r"(b) CO$_2$(aq) [mol m$^{-3}$]", r"(c) OH$^-$ [mol m$^{-3}$]",
              r"(d) HCO$_3^-$ [mol m$^{-3}$]", r"(e) CO$_3^{2-}$ [mol m$^{-3}$]",
              r"(f) Cs$^+$ [mol m$^{-3}$]", r"(g) H$^+$ [mol m$^{-3}$]",
              r"(h) $R_\mathrm{CO_2R}$ [mol m$^{-3}$s$^{-1}$]",
              r"(i) $R_\mathrm{HER}$ [mol m$^{-3}$s$^{-1}$]"]
    fig, axs = plt.subplots(3, 3, figsize=(11.5, 9.2))
    fig.subplots_adjust(hspace=0.45, wspace=0.28)
    cmap = plt.get_cmap("viridis_r")
    V = d["V"]
    vmin, vmax = np.min(-V), np.max(-V)
    for j in range(9):
        a = axs[j // 3][j % 3]
        for k in range(d["nV"]):
            col = cmap((-V[k] - vmin) / max(vmax - vmin, 1e-12))
            a.plot(d["x_cl"], d["liq"][k, j, :], color=col, lw=0.7)
        a.set_xlabel(r"$x$ [$\mu$m]"); a.set_title(labels[j], fontsize=9)
        if j in (6,):
            a.set_yscale("log")
    sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    cb = fig.colorbar(sm, ax=axs.ravel().tolist(), fraction=0.018, pad=0.015)
    cb.set_label(r"$-E_\mathrm{cath}$ [V vs RHE@pH8]  (dark = more cathodic)")
    fig.suptitle("A2  liquid-phase diagnostics — %s (all %d sweep voltages)"
                 % (tag, d["nV"]), y=0.995)
    fn = os.path.join(outdir(tag), "A2_liquid.png")
    fig.savefig(fn, bbox_inches="tight"); plt.close(fig)
    return fn


def plot_B1(tag):
    d = load_profiles(tag)
    if not d["has_gas"]:
        return None
    labels = [r"(a) $p_\mathrm{CO_2}$ [atm]", r"(b) $p_\mathrm{CO}$ [atm]",
              r"(c) $p_\mathrm{H_2}$ [atm]", r"(d) $p_\mathrm{total}$ [Pa]"]
    fig, axs = plt.subplots(2, 2, figsize=(9.5, 6.8))
    fig.subplots_adjust(hspace=0.38, wspace=0.26)
    cmap = plt.get_cmap("viridis_r")
    V = d["V"]; vmin, vmax = np.min(-V), np.max(-V)
    for j in range(4):
        a = axs[j // 2][j % 2]
        for k in range(d["nV"]):
            col = cmap((-V[k] - vmin) / max(vmax - vmin, 1e-12))
            a.plot(d["x_g"], d["gas"][k, j, :], color=col, lw=0.7)
        a.axvline(0.0, color="k", ls="--", lw=0.8)
        a.set_xlabel(r"$x$ [$\mu$m]"); a.set_title(labels[j], fontsize=9)
    sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    cb = fig.colorbar(sm, ax=axs.ravel().tolist(), fraction=0.018, pad=0.015)
    cb.set_label(r"$-E_\mathrm{cath}$ [V vs RHE@pH8]  (dark = more cathodic)")
    fig.suptitle("B1  gas-phase diagnostics — %s (GDL/CL interface dashed)" % tag, y=0.99)
    fn = os.path.join(outdir(tag), "B1_gas.png")
    fig.savefig(fn, bbox_inches="tight"); plt.close(fig)
    return fn


def plot_D(tag):
    d = load_profiles(tag)
    if not d["has_pot"]:
        return None
    V = d["V"]
    pick = [int(round(f * (d["nV"] - 1))) for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6))
    cmap = plt.get_cmap("viridis_r")
    vmin, vmax = np.min(-V), np.max(-V)
    for k in pick:
        col = cmap((-V[k] - vmin) / max(vmax - vmin, 1e-12))
        axs[0].plot(d["x_g"], d["pot_s"][k, :], color=col,
                    label=r"$E_\mathrm{cath}$=%.2f V$_\mathrm{RHE}$" % V[k])
        axs[1].plot(d["x_cl"] / d["x_cl"][-1], d["pot_L"][k, :] * 1e3, color=col,
                    label=r"$E_\mathrm{cath}$=%.2f V$_\mathrm{RHE}$" % V[k])
    axs[0].axvline(0.0, color="k", ls="--", lw=0.8)
    axs[0].set_xlabel(r"$x$ [$\mu$m]  (GDL/CL interface dashed)")
    axs[0].set_ylabel(r"$\varphi_s$ [V vs SHE]"); axs[0].set_title(r"D1  solid potential")
    axs[0].legend(frameon=False, fontsize=7)
    axs[1].set_xlabel(r"$x/L_\mathrm{CL}$")
    axs[1].set_ylabel(r"$\varphi_L$ [mV vs SHE]"); axs[1].set_title(r"D2  liquid potential")
    axs[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fn = os.path.join(outdir(tag), "D_potentials.png")
    fig.savefig(fn, bbox_inches="tight"); plt.close(fig)
    return fn


def plot_overlay(tags, name="overlay"):
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.4))
    cols = plt.get_cmap("plasma")(np.linspace(0.05, 0.8, len(tags)))
    for t, c in zip(tags, cols):
        try:
            d = load_pol(t)
        except OSError:
            continue
        mx = -d["V_SHE"]
        ax[0].plot(mx, d["i_CO_mAcm2"], color=c, label=t)
        ax[1].plot(mx, d["i_H2_mAcm2"], color=c, label=t)
        itot = np.maximum(d["i_tot_mAcm2"], 1e-30)
        ax[2].plot(mx, 100 * d["i_CO_mAcm2"] / itot, color=c, label=t)
    ax[0].plot(-E_exp_SHE, i_CO_exp, "ks", mfc="none", ms=6, label="exp")
    ax[1].plot(-E_exp_SHE, i_H2_exp, "k^", mfc="none", ms=6, label="exp")
    ax[2].plot(-E_exp_SHE, FE_CO_exp, "ks", mfc="none", ms=6, label="exp")
    for a, ttl, yl in zip(ax, [r"$i_\mathrm{CO}$", r"$i_\mathrm{H_2}$", r"FE$_\mathrm{CO}$"],
                          [r"mA cm$^{-2}$", r"mA cm$^{-2}$", "%"]):
        a.set_xlabel(r"$-E_\mathrm{cath}$ [V vs SHE]"); a.set_ylabel(yl)
        a.set_title(ttl); a.set_xlim(0.7, 3.1); a.legend(frameon=False, fontsize=7)
    ax[2].set_ylim(0, 100)
    fig.tight_layout()
    fn = os.path.join(HERE, "%s.png" % name)
    fig.savefig(fn, bbox_inches="tight"); plt.close(fig)
    return fn


def plot_final(tag="stage5"):
    """§18.1 4-panel summary figure + §18.2 FE overlay, from the final stage."""
    d = load_pol(tag)
    mx = -d["V_SHE"]
    itot = np.maximum(d["i_tot_mAcm2"], 1e-30)
    feco = 100 * d["i_CO_mAcm2"] / itot

    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.5))
    for a in ax:
        a.set_xlabel(r"$-E_\mathrm{cath}$ [V vs SHE]"); a.set_xlim(0.7, 3.1)
    ax[0].plot(mx, d["i_CO_mAcm2"], color=C_CO, label=r"$i_\mathrm{CO}$")
    ax[0].plot(mx, d["i_H2_mAcm2"], color=C_H2, label=r"$i_\mathrm{H_2}$")
    ax[0].plot(mx, d["i_tot_mAcm2"], color=C_TOT, ls="--", label=r"$i_\mathrm{tot}$")
    ax[0].plot(-E_exp_SHE, i_CO_exp, "s", mfc="none", mec=C_CO, ms=7)
    ax[0].plot(-E_exp_SHE, i_H2_exp, "^", mfc="none", mec=C_H2, ms=7)
    ax[0].set_ylabel(r"$i$ [mA cm$^{-2}$]"); ax[0].set_title("(a) linear polarization")
    ax[0].legend(frameon=False)
    ax[1].semilogy(mx, np.maximum(d["i_CO_mAcm2"], 1e-4), color=C_CO)
    ax[1].semilogy(mx, np.maximum(d["i_H2_mAcm2"], 1e-4), color=C_H2)
    ax[1].semilogy(mx, np.maximum(d["i_tot_mAcm2"], 1e-4), color=C_TOT, ls="--")
    ax[1].semilogy(-E_exp_SHE, i_CO_exp, "s", mfc="none", mec=C_CO, ms=7)
    ax[1].semilogy(-E_exp_SHE, i_H2_exp, "^", mfc="none", mec=C_H2, ms=7)
    ax[1].set_ylim(1e-2, 3e3); ax[1].set_ylabel(r"$i$ [mA cm$^{-2}$]")
    ax[1].set_title("(b) log polarization")
    ax[2].plot(mx, feco, color=C_CO, label=r"FE$_\mathrm{CO}$")
    ax[2].plot(mx, 100 - feco, color=C_H2, label=r"FE$_\mathrm{H_2}$")
    ax[2].plot(-E_exp_SHE, FE_CO_exp, "s", mfc="none", mec=C_CO, ms=7)
    ax[2].plot(-E_exp_SHE, FE_H2_exp, "^", mfc="none", mec=C_H2, ms=7)
    ax[2].set_ylim(0, 100); ax[2].set_ylabel("FE [%]")
    ax[2].set_title("(c) faradaic efficiency"); ax[2].legend(frameon=False)
    fig.suptitle("§18.1  model (%s, fitted kinetics) vs Romiluyi et al. 2023" % tag, y=1.03)
    fig.tight_layout()
    f1 = os.path.join(HERE, "final_experimental_comparison.png")
    fig.savefig(f1, bbox_inches="tight"); plt.close(fig)

    fig, a = plt.subplots(figsize=(5.2, 3.8))
    a.plot(mx, feco, color=C_CO, lw=1.8, label=r"FE$_\mathrm{CO}$ model")
    a.plot(mx, 100 - feco, color=C_H2, lw=1.8, label=r"FE$_\mathrm{H_2}$ model")
    a.plot(-E_exp_SHE, FE_CO_exp, "s", mfc="none", mec=C_CO, ms=8, label=r"FE$_\mathrm{CO}$ exp")
    a.plot(-E_exp_SHE, FE_H2_exp, "^", mfc="none", mec=C_H2, ms=8, label=r"FE$_\mathrm{H_2}$ exp")
    xc = mx[np.argmin(np.abs(feco - 50))]
    a.axvline(xc, color="0.5", ls=":", lw=1)
    a.annotate("crossover\n%.2f V" % xc, (xc, 52), fontsize=7, color="0.35",
               ha="left", va="bottom")
    a.set_xlabel(r"$-E_\mathrm{cath}$ [V vs SHE]"); a.set_ylabel("FE [%]")
    a.set_xlim(0.7, 3.1); a.set_ylim(0, 100); a.legend(frameon=False, fontsize=8)
    a.set_title(r"§18.2  FE$_\mathrm{CO}$ / FE$_\mathrm{H_2}$ overlay")
    fig.tight_layout()
    f2 = os.path.join(HERE, "final_FE_overlay.png")
    fig.savefig(f2, bbox_inches="tight"); plt.close(fig)
    return f1, f2


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__); sys.exit(1)
    if args[0] == "--final":
        for f in plot_final(args[1] if len(args) > 1 else "stage5"):
            print(f)
    elif args[0] == "--overlay":
        print(plot_overlay(args[1:], name="overlay_" + "_".join(args[1:])))
    else:
        for tag in args:
            for f in (plot_A1(tag), plot_A2(tag), plot_B1(tag), plot_D(tag)):
                if f:
                    print(f)
