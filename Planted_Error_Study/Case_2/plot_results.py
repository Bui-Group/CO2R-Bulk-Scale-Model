#!/usr/bin/env python3
# ============================================================================
# plot_results.py — Stage diagnostic plots (§12.7).
#   Usage: python3 plot_results.py <stage>     e.g.  python3 plot_results.py 1
#   Reads output_stage<N>/polarization.csv and profiles.csv (written by Julia).
#   Produces A1 (3-panel polarization) and A2 (3x3 liquid-phase) figures.
#   All voltage x-axes are -E_cath [V vs SHE] (positive = more cathodic).
# ============================================================================
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

HERE = os.path.dirname(os.path.abspath(__file__))

# Experimental data — Lees et al. 2024, Nat. Chem. Eng. (§16.1)
V_RHE_to_SHE = 2.303 * 8.314462618 * 323.15 / 96485.332 * 8.0   # 0.51305 V
E_exp_RHE = np.array([-0.816, -1.205, -1.644, -2.079, -2.413])
i_CO_exp  = np.array([35.1, 189.6, 299.1, 204.3, 64.6])
i_H2_exp  = np.array([29.9, 49.3, 119.6, 449.9, 850.5])
E_exp_SHE = E_exp_RHE - V_RHE_to_SHE
i_tot_exp = i_CO_exp + i_H2_exp
FE_CO_exp = 100 * i_CO_exp / i_tot_exp
FE_H2_exp = 100 * i_H2_exp / i_tot_exp


def load_csv(path):
    return np.genfromtxt(path, delimiter=",", names=True)


def plot_A1(stage, outdir, pol):
    nE = -pol["V_SHE"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    # (a) linear
    ax[0].plot(nE, pol["iCO_mAcm2"], "-", color="C0", label="i$_{CO}$ model")
    ax[0].plot(nE, pol["iH2_mAcm2"], "-", color="C3", label="i$_{H2}$ model")
    ax[0].plot(nE, pol["itot_mAcm2"], "-", color="k", label="i$_{tot}$ model")
    ax[0].plot(-E_exp_SHE, i_CO_exp, "s", mfc="none", color="C0", label="i$_{CO}$ exp")
    ax[0].plot(-E_exp_SHE, i_H2_exp, "^", mfc="none", color="C3", label="i$_{H2}$ exp")
    ax[0].set_ylabel("current density [mA/cm²]")
    ax[0].set_title("(a) Linear polarization")
    # (b) log
    ax[1].semilogy(nE, np.clip(pol["iCO_mAcm2"], 1e-3, None), "-", color="C0")
    ax[1].semilogy(nE, np.clip(pol["iH2_mAcm2"], 1e-3, None), "-", color="C3")
    ax[1].semilogy(nE, np.clip(pol["itot_mAcm2"], 1e-3, None), "-", color="k")
    ax[1].semilogy(-E_exp_SHE, i_CO_exp, "s", mfc="none", color="C0")
    ax[1].semilogy(-E_exp_SHE, i_H2_exp, "^", mfc="none", color="C3")
    ax[1].set_ylabel("current density [mA/cm²] (log)")
    ax[1].set_title("(b) Log polarization")
    # (c) FE
    ax[2].plot(nE, pol["FE_CO"], "-", color="C0", label="FE$_{CO}$ model")
    ax[2].plot(nE, pol["FE_H2"], "-", color="C3", label="FE$_{H2}$ model")
    ax[2].plot(-E_exp_SHE, FE_CO_exp, "s", mfc="none", color="C0", label="FE$_{CO}$ exp")
    ax[2].plot(-E_exp_SHE, FE_H2_exp, "^", mfc="none", color="C3", label="FE$_{H2}$ exp")
    ax[2].set_ylabel("Faradaic efficiency [%]")
    ax[2].set_ylim(0, 100)
    ax[2].set_title("(c) Faradaic efficiency")
    for a in ax:
        a.set_xlabel("−E$_{cath}$ [V vs SHE]")
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.suptitle(f"Stage {stage} — A1: polarization summary (model vs Lees et al. 2024)")
    fig.tight_layout()
    fn = os.path.join(outdir, f"stage{stage}_A1_polarization.png")
    fig.savefig(fn, dpi=130)
    plt.close(fig)
    print("wrote", fn)


def plot_A2(stage, outdir, prof):
    Vs = np.unique(prof["V_RHE"])    # ascending: most cathodic → mild
    cmap = cm.viridis
    # Color each curve by its actual V value through the SAME norm as the
    # colorbar, so line colors and colorbar are guaranteed consistent.
    norm = plt.Normalize(vmin=Vs.min(), vmax=Vs.max())
    panels = [
        ("pH", "pH [−]", False),
        ("c_CO2", "CO₂(aq) [mol/m³]", False),
        ("c_OH", "OH⁻ [mol/m³]", False),
        ("c_HCO3", "HCO₃⁻ [mol/m³]", False),
        ("c_CO3", "CO₃²⁻ [mol/m³]", False),
        ("c_Cs", "Cs⁺ [mol/m³]", False),
        ("c_H", "H⁺ [mol/m³]", True),
        ("R_CO2R", "R_CO2R [mol/m³/s]", False),
        ("R_HER", "R_HER [mol/m³/s]", False),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for k, (col, ylabel, sci) in enumerate(panels):
        ax = axes[k // 3][k % 3]
        for V in Vs:
            m = prof["V_RHE"] == V
            x = prof["x_um"][m]
            y = prof[col][m]
            order = np.argsort(x)
            ax.plot(x[order], y[order], color=cmap(norm(V)), lw=0.8)
        ax.set_xlabel("x [μm]  (0 = GDL, L_CL = electrolyte)")
        ax.set_ylabel(ylabel)
        if sci:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("V$_{RHE}$ [V]")
    fig.suptitle(f"Stage {stage} — A2: liquid-phase 3×3 diagnostic (all sweep voltages)")
    fn = os.path.join(outdir, f"stage{stage}_A2_liquid.png")
    fig.savefig(fn, dpi=120)
    plt.close(fig)
    print("wrote", fn)


def plot_B1(stage, outdir, gas):
    """2x2 gas-phase partial-pressure profiles across GDL+CL, all voltages (§12.7 B1)."""
    Vs = np.unique(gas["V_RHE"])
    cmap = cm.viridis
    norm = plt.Normalize(vmin=Vs.min(), vmax=Vs.max())
    panels = [("pCO2_atm", "p$_{CO2}$ [atm]"), ("pCO_atm", "p$_{CO}$ [atm]"),
              ("pH2_atm", "p$_{H2}$ [atm]"), ("ptot_atm", "p$_{total}$ [atm]")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for k, (col, ylabel) in enumerate(panels):
        ax = axes[k // 2][k % 2]
        for V in Vs:
            m = gas["V_RHE"] == V
            x = gas["x_um"][m]; y = gas[col][m]
            o = np.argsort(x)
            ax.plot(x[o], y[o], color=cmap(norm(V)), lw=0.8)
        ax.axvline(0.0, ls="--", color="0.4", lw=1)   # GDL/CL interface at x=0
        ax.set_xlabel("x [μm]  (GDL inlet → CL/electrolyte; dashed = GDL/CL)")
        ax.set_ylabel(ylabel)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02); cb.set_label("V$_{RHE}$ [V]")
    fig.suptitle(f"Stage {stage} — B1: gas-phase partial pressures (all sweep voltages)")
    fn = os.path.join(outdir, f"stage{stage}_B1_gas.png")
    fig.savefig(fn, dpi=120); plt.close(fig)
    print("wrote", fn)


def plot_D1_D2(stage, outdir):
    """Stage 5: φ_s across GDL+CL (D1) and φ_L across CL (D2), at all voltages."""
    cmap = cm.viridis
    # D1: φ_s from gasprofiles.csv (column phi_s), x spans GDL+CL
    gp = os.path.join(outdir, "gasprofiles.csv")
    pl = os.path.join(outdir, "phiL.csv")
    if not (os.path.exists(gp) and os.path.exists(pl)):
        return
    gas = load_csv(gp); phiL = load_csv(pl)
    Vs = np.unique(gas["V_RHE"]); norm = plt.Normalize(vmin=Vs.min(), vmax=Vs.max())
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for V in Vs:
        m = gas["V_RHE"] == V; x = gas["x_um"][m]; y = gas["phi_s"][m]
        o = np.argsort(x); ax[0].plot(x[o], y[o], color=cmap(norm(V)), lw=0.8)
    ax[0].axvline(0.0, ls="--", color="0.4", lw=1)
    ax[0].set_xlabel("x [μm]  (−L_GDL → L_CL; dashed = GDL/CL)")
    ax[0].set_ylabel("φ_s [V vs SHE]"); ax[0].set_title("(D1) Solid potential φ_s (GDL+CL)")
    for V in Vs:
        m = phiL["V_RHE"] == V; x = phiL["x_um"][m]; y = phiL["phi_L"][m] * 1000
        o = np.argsort(x); ax[1].plot(x[o], y[o], color=cmap(norm(V)), lw=0.8)
    ax[1].set_xlabel("x [μm]  (0 = GDL side, L_CL = electrolyte)")
    ax[1].set_ylabel("φ_L [mV vs SHE]"); ax[1].set_title("(D2) Liquid potential φ_L (CL)")
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02); cb.set_label("V$_{RHE}$ [V]")
    fig.suptitle(f"Stage {stage} — D1/D2: potential profiles")
    fn = os.path.join(outdir, f"stage{stage}_D_potentials.png")
    fig.savefig(fn, dpi=130); plt.close(fig)
    print("wrote", fn)


def plot_overlay(stage, outdir):
    """Overlay polarization of stages 1..stage to show the effect of each addition."""
    sn = int(stage)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    for s in range(1, sn + 1):
        p = os.path.join(HERE, f"output_stage{s}", "polarization.csv")
        if not os.path.exists(p):
            continue
        pol = load_csv(p)
        nE = -pol["V_SHE"]
        ls = styles[(s - 1) % len(styles)]
        ax[0].plot(nE, pol["iCO_mAcm2"], linestyle=ls, color="C0", label=f"i$_{{CO}}$ S{s}")
        ax[0].plot(nE, pol["iH2_mAcm2"], linestyle=ls, color="C3", label=f"i$_{{H2}}$ S{s}")
        ax[1].semilogy(nE, np.clip(pol["itot_mAcm2"], 1e-3, None), linestyle=ls, color="k", label=f"i$_{{tot}}$ S{s}")
        ax[2].plot(nE, pol["FE_CO"], linestyle=ls, color="C0", label=f"FE$_{{CO}}$ S{s}")
    ax[0].plot(-E_exp_SHE, i_CO_exp, "s", mfc="none", color="C0")
    ax[0].plot(-E_exp_SHE, i_H2_exp, "^", mfc="none", color="C3")
    ax[2].plot(-E_exp_SHE, FE_CO_exp, "s", mfc="none", color="C0")
    ax[0].set_ylabel("current [mA/cm²]"); ax[0].set_title("(a) i_CO, i_H2 (linear)")
    ax[1].set_ylabel("i_total [mA/cm²] (log)"); ax[1].set_title("(b) i_total (log)")
    ax[2].set_ylabel("FE_CO [%]"); ax[2].set_ylim(0, 100); ax[2].set_title("(c) FE_CO")
    for a in ax:
        a.set_xlabel("−E$_{cath}$ [V vs SHE]"); a.legend(fontsize=7); a.grid(alpha=0.3)
    fig.suptitle(f"Stage {stage} — polarization overlay across stages 1–{stage}")
    fig.tight_layout()
    fn = os.path.join(outdir, f"stage{stage}_overlay.png")
    fig.savefig(fn, dpi=130); plt.close(fig)
    print("wrote", fn)


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "1"
    outdir = os.path.join(HERE, "output_fit" if stage == "fit" else f"output_stage{stage}")
    pol = load_csv(os.path.join(outdir, "polarization.csv"))
    plot_A1(stage, outdir, pol)
    prof_path = os.path.join(outdir, "profiles.csv")
    if os.path.exists(prof_path):
        prof = load_csv(prof_path)
        plot_A2(stage, outdir, prof)
    gas_path = os.path.join(outdir, "gasprofiles.csv")
    if os.path.exists(gas_path):
        gas = load_csv(gas_path)
        plot_B1(stage, outdir, gas)
    if stage == "5":
        plot_D1_D2(stage, outdir)
    # Cross-stage overlay only for numeric stages >= 2 (skip "fit").
    if stage.isdigit() and int(stage) >= 2:
        plot_overlay(stage, outdir)


if __name__ == "__main__":
    main()
