#!/usr/bin/env python3
"""plot_results.py — stage diagnostic plots (§12.7).

Usage: python3 plot_results.py <stage_dir> <label> [--gas] [--pot]

Reads CSVs written by the Julia drivers from <stage_dir>:
  polarization.csv, liquid_profiles.csv, [gas_profiles.csv], [potentials.csv]
Writes PNGs into <stage_dir>: A1_polarization.png, A2_liquid.png,
  [B1_gas.png], [D1_phis.png, D2_phiL.png].

All voltage axes are -E_cath [V vs SHE] (positive = more cathodic).
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OFFSET = 2.303 * 8.314462618 * 323.15 / 96485.332 * 8.0  # 0.51305 V (RHE@pH8 → SHE)

# Lees et al. 2024 experimental (§16.1)
E_exp_RHE = np.array([-0.816, -1.205, -1.644, -2.079, -2.413])
i_CO_exp  = np.array([35.1, 189.6, 299.1, 204.3, 64.6])
i_H2_exp  = np.array([29.9, 49.3, 119.6, 449.9, 850.5])
E_exp_SHE = E_exp_RHE - OFFSET
i_tot_exp = i_CO_exp + i_H2_exp
FE_CO_exp = 100 * i_CO_exp / i_tot_exp
FE_H2_exp = 100 * i_H2_exp / i_tot_exp


def _read_csv(path):
    if not os.path.isfile(path):
        return None
    return np.genfromtxt(path, delimiter=",", names=True)


def plot_A1(stage_dir, label):
    d = _read_csv(os.path.join(stage_dir, "polarization.csv"))
    if d is None:
        print("  [A1] no polarization.csv — skipping")
        return
    x = -np.atleast_1d(d["V_she"])
    order = np.argsort(x)
    x = x[order]
    iCO = np.atleast_1d(d["i_CO"])[order]
    iH2 = np.atleast_1d(d["i_H2"])[order]
    itot = np.atleast_1d(d["i_tot"])[order]
    feco = np.atleast_1d(d["FE_CO"])[order]
    feh2 = np.atleast_1d(d["FE_H2"])[order]
    xe = -E_exp_SHE

    # optional previous-stage overlay (thin dashed)
    prev = _read_csv(os.path.join(stage_dir, "prev_polarization.csv"))
    pv = plabel = None
    if prev is not None:
        po = np.argsort(-np.atleast_1d(prev["V_she"]))
        pv = (-np.atleast_1d(prev["V_she"])[po], np.atleast_1d(prev["i_CO"])[po],
              np.atleast_1d(prev["i_H2"])[po], np.atleast_1d(prev["i_tot"])[po],
              np.atleast_1d(prev["FE_CO"])[po], np.atleast_1d(prev["FE_H2"])[po])
        plabel = os.environ.get("PREV_LABEL", "prev stage")

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    # (a) linear
    ax[0].plot(x, iCO, "-", color="C0", label="i$_{CO}$ model")
    ax[0].plot(x, iH2, "-", color="C3", label="i$_{H_2}$ model")
    ax[0].plot(x, itot, "-", color="k", label="i$_{tot}$ model")
    ax[0].plot(xe, i_CO_exp, "s", mfc="none", color="C0", ms=8, label="i$_{CO}$ exp")
    ax[0].plot(xe, i_H2_exp, "^", mfc="none", color="C3", ms=8, label="i$_{H_2}$ exp")
    ax[0].plot(xe, i_tot_exp, "o", mfc="none", color="k", ms=8, label="i$_{tot}$ exp")
    if pv is not None:
        ax[0].plot(pv[0], pv[1], "--", color="C0", lw=1, alpha=0.5)
        ax[0].plot(pv[0], pv[2], "--", color="C3", lw=1, alpha=0.5)
        ax[0].plot(pv[0], pv[3], "--", color="k", lw=1, alpha=0.5, label=f"i$_{{tot}}$ {plabel}")
    ax[0].set_ylabel("current density [mA/cm²]")
    ax[0].set_title("(a) linear polarization")
    ax[0].legend(fontsize=8)
    # (b) log
    ax[1].semilogy(x, np.clip(iCO, 1e-3, None), "-", color="C0")
    ax[1].semilogy(x, np.clip(iH2, 1e-3, None), "-", color="C3")
    ax[1].semilogy(x, np.clip(itot, 1e-3, None), "-", color="k")
    ax[1].semilogy(xe, i_CO_exp, "s", mfc="none", color="C0")
    ax[1].semilogy(xe, i_H2_exp, "^", mfc="none", color="C3")
    ax[1].semilogy(xe, i_tot_exp, "o", mfc="none", color="k")
    ax[1].set_ylabel("current density [mA/cm²] (log)")
    ax[1].set_title("(b) log polarization")
    # (c) FE
    ax[2].plot(x, feco, "-", color="C0", label="FE$_{CO}$ model")
    ax[2].plot(x, feh2, "-", color="C3", label="FE$_{H_2}$ model")
    ax[2].plot(xe, FE_CO_exp, "s", mfc="none", color="C0", ms=8, label="FE$_{CO}$ exp")
    ax[2].plot(xe, FE_H2_exp, "^", mfc="none", color="C3", ms=8, label="FE$_{H_2}$ exp")
    if pv is not None:
        ax[2].plot(pv[0], pv[4], "--", color="C0", lw=1, alpha=0.5)
        ax[2].plot(pv[0], pv[5], "--", color="C3", lw=1, alpha=0.5)
    ax[2].set_ylabel("faradaic efficiency [%]")
    ax[2].set_ylim(0, 100)
    ax[2].set_title("(c) faradaic efficiency")
    ax[2].legend(fontsize=8)
    for a in ax:
        a.set_xlabel("−E$_{cath}$ [V vs SHE]")
        a.set_xlim(0.7, 3.1)
        a.grid(alpha=0.3)
    fig.suptitle(f"{label} — A1 polarization summary (Lees et al. 2024 overlaid)")
    fig.tight_layout()
    fig.savefig(os.path.join(stage_dir, "A1_polarization.png"), dpi=130)
    plt.close(fig)
    print("  [A1] written")


def _grouped(d, key="V_rhe"):
    Vs = np.unique(d[key])
    # sort by |V| ascending so color ramp goes mild→cathodic
    Vs = Vs[np.argsort(np.abs(Vs))]
    return Vs


def plot_A2(stage_dir, label):
    d = _read_csv(os.path.join(stage_dir, "liquid_profiles.csv"))
    if d is None:
        print("  [A2] no liquid_profiles.csv — skipping")
        return
    Vs = _grouped(d)
    cmap = plt.cm.viridis(np.linspace(0, 1, len(Vs)))
    fields = [("pH", "pH [−]"), ("cCO2", "CO₂(aq) [mol/m³]"), ("cOH", "OH⁻ [mol/m³]"),
              ("cHCO3", "HCO₃⁻ [mol/m³]"), ("cCO3", "CO₃²⁻ [mol/m³]"), ("cCs", "Cs⁺ [mol/m³]"),
              ("cH", "H⁺ [mol/m³]"), ("R_CO2R", "R_CO2R [mol/m³/s]"), ("R_HER", "R_HER [mol/m³/s]")]
    fig, axs = plt.subplots(3, 3, figsize=(15, 12))
    for k, (fld, ylab) in enumerate(fields):
        ax = axs[k // 3][k % 3]
        for j, V in enumerate(Vs):
            m = d["V_rhe"] == V
            xx = d["x_um"][m]; yy = d[fld][m]
            o = np.argsort(xx)
            ax.plot(xx[o], yy[o], color=cmap[j], lw=1.2,
                    label=f"{V:.2f}" if (k == 0) else None)
        ax.set_xlabel("x [μm]")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
        if fld in ("cH",):
            ax.set_yscale("log")
    axs[0][0].legend(title="V$_{RHE}$", fontsize=7, ncol=2)
    fig.suptitle(f"{label} — A2 liquid-phase diagnostics (GDL side x=0 → electrolyte x=L_CL)")
    fig.tight_layout()
    fig.savefig(os.path.join(stage_dir, "A2_liquid.png"), dpi=120)
    plt.close(fig)
    print("  [A2] written")


def plot_B1(stage_dir, label):
    d = _read_csv(os.path.join(stage_dir, "gas_profiles.csv"))
    if d is None:
        print("  [B1] no gas_profiles.csv — skipping")
        return
    Vs = _grouped(d)
    cmap = plt.cm.viridis(np.linspace(0, 1, len(Vs)))
    fields = [("pCO2", "p_CO₂ [atm]"), ("pCO", "p_CO [atm]"),
              ("pH2", "p_H₂ [atm]"), ("ptot", "p_total [atm]")]
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    for k, (fld, ylab) in enumerate(fields):
        ax = axs[k // 2][k % 2]
        for j, V in enumerate(Vs):
            m = d["V_rhe"] == V
            xx = d["x_um"][m]; yy = d[fld][m]
            o = np.argsort(xx)
            ax.plot(xx[o], yy[o], color=cmap[j], lw=1.2, label=f"{V:.2f}" if k == 0 else None)
        ax.axvline(0.0, ls="--", color="gray", lw=1)  # GDL/CL interface at x=0
        ax.set_xlabel("x [μm]  (GDL x<0, CL x>0)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
    axs[0][0].legend(title="V$_{RHE}$", fontsize=7, ncol=2)
    fig.suptitle(f"{label} — B1 gas-phase diagnostics")
    fig.tight_layout()
    fig.savefig(os.path.join(stage_dir, "B1_gas.png"), dpi=120)
    plt.close(fig)
    print("  [B1] written")


def plot_D(stage_dir, label):
    d = _read_csv(os.path.join(stage_dir, "potentials.csv"))
    if d is None:
        print("  [D] no potentials.csv — skipping")
        return
    Vs = _grouped(d)
    cmap = plt.cm.viridis(np.linspace(0, 1, len(Vs)))
    # D1 φ_s across GDL+CL
    fig, ax = plt.subplots(figsize=(8, 6))
    for j, V in enumerate(Vs):
        m = (d["V_rhe"] == V) & np.isfinite(d["phi_s"])
        xx = d["x_um"][m]; yy = d["phi_s"][m]
        o = np.argsort(xx)
        ax.plot(xx[o], yy[o], color=cmap[j], lw=1.2, label=f"{V:.2f}")
    ax.set_xlabel("x [μm]  (−L_GDL → L_CL)")
    ax.set_ylabel("φ_s [V vs SHE]")
    ax.set_title(f"{label} — D1 solid potential φ_s")
    ax.legend(title="V$_{RHE}$", fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(stage_dir, "D1_phis.png"), dpi=120); plt.close(fig)
    # D2 φ_L across CL
    fig, ax = plt.subplots(figsize=(8, 6))
    for j, V in enumerate(Vs):
        m = (d["V_rhe"] == V) & np.isfinite(d["phi_L"])
        xx = d["x_um"][m]; yy = d["phi_L"][m] * 1000.0
        o = np.argsort(xx)
        ax.plot(xx[o], yy[o], color=cmap[j], lw=1.2, label=f"{V:.2f}")
    ax.set_xlabel("x [μm]  (CL only)")
    ax.set_ylabel("φ_L [mV vs SHE]")
    ax.set_title(f"{label} — D2 liquid potential φ_L (gauge φ_L=0 at electrolyte)")
    ax.legend(title="V$_{RHE}$", fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(stage_dir, "D2_phiL.png"), dpi=120); plt.close(fig)
    print("  [D1/D2] written")


if __name__ == "__main__":
    stage_dir = sys.argv[1]
    label = sys.argv[2]
    plot_A1(stage_dir, label)
    plot_A2(stage_dir, label)
    if "--gas" in sys.argv:
        plot_B1(stage_dir, label)
    if "--pot" in sys.argv:
        plot_D(stage_dir, label)
    print(f"Plots done for {label}.")
