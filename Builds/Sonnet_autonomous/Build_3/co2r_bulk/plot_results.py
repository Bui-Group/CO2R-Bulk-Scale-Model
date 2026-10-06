#!/usr/bin/env python3
"""
plot_results.py — matplotlib plotting for the CO2R bulk-scale model.
Built from CO2R_BULK_MODEL_IMPLEMENTATION_GUIDE.md §12.7 (debug plot
checklist) and §18 (experimental comparison plots). Reads CSVs exported by
the Julia run_stageN.jl / run_fit.jl drivers (Julia does all physics;
this script only plots).

Usage:
    python3 plot_results.py --stage 1        # A1 + A2
    python3 plot_results.py --stage 3        # A1 + A2 + B1
    python3 plot_results.py --stage 5        # A1 + A2 + B1 + D1 + D2
    python3 plot_results.py --final          # §18 4-panel + FE overlay, all stages
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ── §15.5 / §16.1 experimental data (Romiluyi et al. 2023) ──────────────
V_RHE_TO_SHE_OFFSET = 2.303 * 8.314462618 * 323.15 / 96485.332 * 8.0   # 0.51305 V

E_exp_RHE = np.array([-0.816, -1.205, -1.644, -2.079, -2.413])
i_CO_exp = np.array([35.1, 189.6, 299.1, 204.3, 64.6])
i_H2_exp = np.array([29.9, 49.3, 119.6, 449.9, 850.5])
E_exp_SHE = E_exp_RHE - V_RHE_TO_SHE_OFFSET
FE_CO_exp = i_CO_exp / (i_CO_exp + i_H2_exp) * 100
FE_H2_exp = i_H2_exp / (i_CO_exp + i_H2_exp) * 100

HERE = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(HERE, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Prefer Lato if present; otherwise fall back cleanly (guide §13 says "Lato
# fonts" for plot_results.py but does not bundle a font — no other document
# may be consulted for a font convention per the study's ground rules, so
# fall back to matplotlib defaults if Lato is unavailable on this machine).
_lato = [f for f in fm.fontManager.ttflist if "Lato" in f.name]
if _lato:
    plt.rcParams["font.family"] = "Lato"
else:
    plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 140
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_f(row, key):
    return float(row[key])


# ── A1: 3-panel polarization summary ─────────────────────────────────────
def plot_A1(stage_dir, stage_label, csv_name="polarization.csv", out_suffix=""):
    rows = read_csv(os.path.join(stage_dir, csv_name))
    V_SHE = np.array([to_f(r, "V_SHE") for r in rows])
    iCO = np.array([to_f(r, "i_CO_mA_cm2") for r in rows])
    iH2 = np.array([to_f(r, "i_H2_mA_cm2") for r in rows])
    itot = np.array([to_f(r, "i_total_mA_cm2") for r in rows])
    FE_CO = np.array([to_f(r, "FE_CO") for r in rows])
    FE_H2 = np.array([to_f(r, "FE_H2") for r in rows])
    order = np.argsort(-V_SHE)  # ascending -V_SHE (more cathodic to the right)
    V_SHE, iCO, iH2, itot, FE_CO, FE_H2 = (a[order] for a in (V_SHE, iCO, iH2, itot, FE_CO, FE_H2))
    negV = -V_SHE

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))

    ax = axs[0]
    ax.plot(negV, iCO, color="#1f6f8b", lw=2, label="i$_{CO}$ (model)")
    ax.plot(negV, iH2, color="#c1440e", lw=2, label="i$_{H2}$ (model)")
    ax.plot(negV, itot, color="#333333", lw=1.6, ls="--", label="i$_{tot}$ (model)")
    ax.plot(-E_exp_SHE, i_CO_exp, "s", mfc="none", mec="#1f6f8b", ms=8, mew=1.8, label="i$_{CO}$ (exp)")
    ax.plot(-E_exp_SHE, i_H2_exp, "^", mfc="none", mec="#c1440e", ms=8, mew=1.8, label="i$_{H2}$ (exp)")
    ax.set_xlabel("$-E_{cath}$ [V vs SHE]"); ax.set_ylabel("i [mA/cm$^2$]")
    ax.set_title("(a) Linear polarization"); ax.legend(fontsize=8)

    ax = axs[1]
    ax.plot(negV, np.clip(iCO, 1e-6, None), color="#1f6f8b", lw=2)
    ax.plot(negV, np.clip(iH2, 1e-6, None), color="#c1440e", lw=2)
    ax.plot(negV, np.clip(itot, 1e-6, None), color="#333333", lw=1.6, ls="--")
    ax.plot(-E_exp_SHE, i_CO_exp, "s", mfc="none", mec="#1f6f8b", ms=8, mew=1.8)
    ax.plot(-E_exp_SHE, i_H2_exp, "^", mfc="none", mec="#c1440e", ms=8, mew=1.8)
    ax.set_yscale("log")
    ax.set_xlabel("$-E_{cath}$ [V vs SHE]"); ax.set_ylabel("i [mA/cm$^2$] (log)")
    ax.set_title("(b) Log polarization")

    ax = axs[2]
    ax.plot(negV, FE_CO, color="#1f6f8b", lw=2, label="FE$_{CO}$")
    ax.plot(negV, FE_H2, color="#c1440e", lw=2, label="FE$_{H2}$")
    ax.plot(-E_exp_SHE, FE_CO_exp, "s", mfc="none", mec="#1f6f8b", ms=8, mew=1.8)
    ax.plot(-E_exp_SHE, FE_H2_exp, "^", mfc="none", mec="#c1440e", ms=8, mew=1.8)
    ax.set_ylim(0, 100)
    ax.set_xlabel("$-E_{cath}$ [V vs SHE]"); ax.set_ylabel("FE [%]")
    ax.set_title("(c) Faradaic efficiency"); ax.legend(fontsize=8)

    fig.suptitle(f"{stage_label} — A1 polarization summary")
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, f"{stage_label}_A1{out_suffix}.png")
    fig.savefig(out); plt.close(fig)
    print("wrote", out)


# ── A2: 3x3 liquid-phase diagnostic ──────────────────────────────────────
def plot_A2(stage_dir, stage_label, csv_name="profiles.csv"):
    rows = read_csv(os.path.join(stage_dir, csv_name))
    byV = defaultdict(list)
    for r in rows:
        byV[to_f(r, "V_RHE")].append(r)
    Vs = sorted(byV.keys(), reverse=True)  # mild -> cathodic
    # plasma: cmap(0)=dark purple, cmap(1)=bright yellow. We want dark=cathodic,
    # light(yellow)=mild, so map t=0 at the most cathodic V (min(Vs)) and t=1 at
    # the mildest V (max(Vs)) -- i.e. t is just the *unreversed* normalized V.
    # The colourbar below uses the SAME cmap + norm (no .reversed()), so the
    # line colours and the bar direction are guaranteed consistent (§12.7 warns
    # explicitly about exactly this class of bug).
    cmap = plt.get_cmap("plasma")
    norm = plt.Normalize(vmin=min(Vs), vmax=max(Vs))

    def color_for(V):
        return cmap(norm(V))
    fig, axs = plt.subplots(3, 3, figsize=(13, 11))
    panels = [
        ("pH", "(a) pH [-]"),
        ("c_CO2", "(b) CO$_2$(aq) [mol/m$^3$]"),
        ("c_OH", "(c) OH$^-$ [mol/m$^3$]"),
        ("c_HCO3", "(d) HCO$_3^-$ [mol/m$^3$]"),
        ("c_CO3", "(e) CO$_3^{2-}$ [mol/m$^3$]"),
        ("c_Cs", "(f) Cs$^+$ [mol/m$^3$]"),
        ("c_H", "(g) H$^+$ [mol/m$^3$]"),
        ("R_CO2R", "(h) R$_{CO2R}$ [mol/m$^3$/s]"),
        ("R_HER", "(i) R$_{HER}$ [mol/m$^3$/s]"),
    ]
    for k, (key, title) in enumerate(panels):
        ax = axs[k // 3, k % 3]
        for V in Vs:
            rs = byV[V]
            x = np.array([to_f(r, "x_um") for r in rs])
            y = np.array([to_f(r, key) for r in rs])
            ax.plot(x, y, color=color_for(V), lw=0.7, alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x [$\\mu$m]", fontsize=8)
        if key == "c_H":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs, orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label("V$_{RHE}$ [V] (light = mild, dark = cathodic)")
    fig.suptitle(f"{stage_label} — A2 liquid-phase diagnostics (all sweep voltages)")
    out = os.path.join(PLOT_DIR, f"{stage_label}_A2.png")
    fig.savefig(out); plt.close(fig)
    print("wrote", out)


# ── B1: 2x2 gas-phase diagnostic ──────────────────────────────────────────
def plot_B1(stage_dir, stage_label, csv_name="gas_profiles.csv"):
    path = os.path.join(stage_dir, csv_name)
    if not os.path.exists(path):
        print("skip B1 (no gas_profiles.csv found):", path)
        return
    rows = read_csv(path)
    byV = defaultdict(list)
    for r in rows:
        byV[to_f(r, "V_RHE")].append(r)
    Vs = sorted(byV.keys(), reverse=True)
    cmap = plt.get_cmap("plasma")
    norm = plt.Normalize(vmin=min(Vs), vmax=max(Vs))

    def color_for(V):
        return cmap(norm(V))

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    panels = [("p_CO2_atm", "(a) p$_{CO2}$ [atm]"), ("p_CO_atm", "(b) p$_{CO}$ [atm]"),
              ("p_H2_atm", "(c) p$_{H2}$ [atm]"), ("p_total_Pa", "(d) p$_{total}$ [Pa]")]
    for k, (key, title) in enumerate(panels):
        ax = axs[k // 2, k % 2]
        for V in Vs:
            rs = byV[V]
            x = np.array([to_f(r, "x_um") for r in rs])
            y = np.array([to_f(r, key) for r in rs])
            ax.plot(x, y, color=color_for(V), lw=0.7, alpha=0.85)
        ax.axvline(0.0, color="k", ls=":", lw=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x [$\\mu$m]", fontsize=8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs, orientation="vertical", fraction=0.03, pad=0.03)
    cbar.set_label("V$_{RHE}$ [V] (light = mild, dark = cathodic)")
    fig.suptitle(f"{stage_label} — B1 gas-phase diagnostics")
    out = os.path.join(PLOT_DIR, f"{stage_label}_B1.png")
    fig.savefig(out); plt.close(fig)
    print("wrote", out)


# ── D1/D2: phi_s / phi_L profiles (Stage 5) ───────────────────────────────
def plot_D1_D2(stage_dir, stage_label):
    path = os.path.join(stage_dir, "phi_profiles.csv")
    if not os.path.exists(path):
        print("skip D1/D2 (no phi_profiles.csv):", path)
        return
    rows = read_csv(path)
    byV = defaultdict(list)
    for r in rows:
        byV[to_f(r, "V_RHE")].append(r)
    Vs = sorted(byV.keys(), reverse=True)
    pick = [Vs[0], Vs[len(Vs)//2], Vs[-1]]
    colors = ["#2b8cbe", "#f16913", "#54278f"]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for V, col in zip(pick, colors):
        rs = sorted(byV[V], key=lambda r: to_f(r, "x_m"))
        x = np.array([to_f(r, "x_m") for r in rs])
        phis = np.array([to_f(r, "phi_s") for r in rs])
        ax.plot(x, phis, color=col, lw=2, label=f"V_RHE={V:.3f}")
    ax.set_xlabel("x [m] (from $-L_{GDL}$ to $L_{CL}$)")
    ax.set_ylabel("$\\phi_s$ [V vs SHE]")
    ax.set_title(f"{stage_label} — D1 solid potential profile")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, f"{stage_label}_D1.png")
    fig.savefig(out); plt.close(fig)
    print("wrote", out)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for V, col in zip(pick, colors):
        rs = sorted([r for r in byV[V] if r["phi_L"] != ""], key=lambda r: to_f(r, "x_m"))
        if not rs:
            continue
        x = np.array([to_f(r, "x_m") for r in rs])
        xL = (x - x.min()) / (x.max() - x.min() + 1e-30)
        phiL = np.array([to_f(r, "phi_L") for r in rs])
        ax.plot(xL, phiL, color=col, lw=2, label=f"V_RHE={V:.3f}")
    ax.set_xlabel("x / L$_{CL}$")
    ax.set_ylabel("$\\phi_L$ [V vs SHE] (anchored 0 at x=L$_{CL}$)")
    ax.set_title(f"{stage_label} — D2 liquid potential profile")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, f"{stage_label}_D2.png")
    fig.savefig(out); plt.close(fig)
    print("wrote", out)


def plot_final_overlay(root):
    """§18: final 4-panel experimental comparison overlaying all stages."""
    stage_dirs = {
        "Stage1": "output_stage1", "Stage2": "output_stage2",
        "Stage3": "output_stage3", "Stage4": "output_stage4",
        "Fit": "output_fit", "Stage5": "output_stage5",
    }
    colors = {"Stage1": "#bbbbbb", "Stage2": "#999999", "Stage3": "#7fb3d5",
              "Stage4": "#5499c7", "Fit": "#e67e22", "Stage5": "#c0392b"}
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    for label, d in stage_dirs.items():
        path = os.path.join(root, d, "polarization.csv")
        if not os.path.exists(path):
            continue
        rows = read_csv(path)
        V_SHE = np.array([to_f(r, "V_SHE") for r in rows])
        iCO = np.array([to_f(r, "i_CO_mA_cm2") for r in rows])
        iH2 = np.array([to_f(r, "i_H2_mA_cm2") for r in rows])
        FE_CO = np.array([to_f(r, "FE_CO") for r in rows])
        order = np.argsort(V_SHE)
        negV = -V_SHE[order]
        axs[0].plot(negV, iCO[order], color=colors[label], lw=1.6, label=f"{label} i_CO")
        axs[1].plot(negV, np.clip(iCO[order], 1e-6, None), color=colors[label], lw=1.6)
        axs[2].plot(negV, FE_CO[order], color=colors[label], lw=1.6)
    axs[0].plot(-E_exp_SHE, i_CO_exp, "s", mfc="none", mec="k", ms=8, mew=1.8, label="exp i_CO")
    axs[0].plot(-E_exp_SHE, i_H2_exp, "^", mfc="none", mec="gray", ms=8, mew=1.8, label="exp i_H2")
    axs[1].plot(-E_exp_SHE, i_CO_exp, "s", mfc="none", mec="k", ms=8, mew=1.8)
    axs[2].plot(-E_exp_SHE, FE_CO_exp, "s", mfc="none", mec="k", ms=8, mew=1.8)
    axs[1].set_yscale("log")
    for ax, t in zip(axs, ["(a) Linear i_CO, all stages", "(b) Log i_CO, all stages", "(c) FE_CO, all stages"]):
        ax.set_xlabel("$-E_{cath}$ [V vs SHE]"); ax.set_xlim(0.7, 3.1); ax.set_title(t)
    axs[0].set_ylabel("i [mA/cm$^2$]"); axs[2].set_ylabel("FE_CO [%]"); axs[2].set_ylim(0, 100)
    axs[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "final_stage_overlay.png")
    fig.savefig(out); plt.close(fig)
    print("wrote", out)


def plot_FE_crossover(root):
    """§18.2: dedicated FE_CO / FE_H2 overlay showing the crossover point,
    using the final Stage 5 (fitted kinetics + full potential physics) sweep."""
    path = os.path.join(root, "output_stage5", "polarization.csv")
    if not os.path.exists(path):
        print("skip FE crossover (no Stage 5 polarization.csv)")
        return
    rows = read_csv(path)
    V_SHE = np.array([to_f(r, "V_SHE") for r in rows])
    FE_CO = np.array([to_f(r, "FE_CO") for r in rows])
    FE_H2 = np.array([to_f(r, "FE_H2") for r in rows])
    order = np.argsort(V_SHE)
    negV = -V_SHE[order]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(negV, FE_CO[order], color="#1f6f8b", lw=2.2, label="FE$_{CO}$ (model, Stage 5)")
    ax.plot(negV, FE_H2[order], color="#c1440e", lw=2.2, label="FE$_{H2}$ (model, Stage 5)")
    ax.plot(-E_exp_SHE, FE_CO_exp, "s", mfc="none", mec="#1f6f8b", ms=9, mew=2, label="FE$_{CO}$ (exp)")
    ax.plot(-E_exp_SHE, FE_H2_exp, "^", mfc="none", mec="#c1440e", ms=9, mew=2, label="FE$_{H2}$ (exp)")
    ax.axhline(50, color="gray", ls=":", lw=1)
    ax.set_xlim(0.7, 3.1); ax.set_ylim(0, 100)
    ax.set_xlabel("$-E_{cath}$ [V vs SHE]"); ax.set_ylabel("Faradaic efficiency [%]")
    ax.set_title("FE$_{CO}$ / FE$_{H2}$ crossover")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "FE_crossover.png")
    fig.savefig(out); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=str, default=None, help="1,2,3,4,fit,5")
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()

    stage_map = {
        "1": ("output_stage1", "Stage1", False, False),
        "2": ("output_stage2", "Stage2", False, False),
        "3": ("output_stage3", "Stage3", True, False),
        "4": ("output_stage4", "Stage4", True, False),
        "fit": ("output_fit", "Fit", True, False),
        "5": ("output_stage5", "Stage5", True, True),
    }
    if args.stage:
        d, label, has_gas, has_phi = stage_map[args.stage]
        stage_dir = os.path.join(HERE, d)
        plot_A1(stage_dir, label)
        plot_A2(stage_dir, label)
        if has_gas:
            plot_B1(stage_dir, label)
        if has_phi:
            plot_D1_D2(stage_dir, label)
    if args.final:
        plot_final_overlay(HERE)
        plot_FE_crossover(HERE)
