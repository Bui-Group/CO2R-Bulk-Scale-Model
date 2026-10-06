"""
plot_common.py -- shared plotting helpers for the CO2R bulk model (guide §12, §18).

Design-judgment note (documented in BUILD_DIALOGUE.md): the guide only specifies
plot CONTENT (panel layout, axes, required curves), not a house visual style, and
the build's hard constraint is guide-only (no other repository files consulted for
methodology/specification -- including any repo-root plotting-style helper). This
module is therefore a self-contained, plain-matplotlib implementation styled for
clarity, not any particular lab template.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
})

V_RHE_TO_SHE_OFFSET = 0.5130515153785598   # must match params.jl exactly

# Romiluyi et al. 2023 experimental data (guide §16.1)
E_exp_RHE = np.array([-0.816, -1.205, -1.644, -2.079, -2.413])
i_CO_exp  = np.array([35.1, 189.6, 299.1, 204.3, 64.6])
i_H2_exp  = np.array([29.9, 49.3, 119.6, 449.9, 850.5])
E_exp_SHE = E_exp_RHE - V_RHE_TO_SHE_OFFSET
FE_CO_exp = i_CO_exp/(i_CO_exp+i_H2_exp)*100
FE_H2_exp = i_H2_exp/(i_CO_exp+i_H2_exp)*100


def plot_A1(polar_csv, title, outpath, show_exp=True):
    """3-panel polarization summary: linear i, log i, FE (§12.7 A1)."""
    df = pd.read_csv(polar_csv)
    nEcath = -df["V_SHE"].values
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.plot(nEcath, df["i_CO_mA_cm2"], color="#1f77b4", lw=1.8, label="$i_{CO}$ (model)")
    ax.plot(nEcath, df["i_H2_mA_cm2"], color="#d62728", lw=1.8, label="$i_{H_2}$ (model)")
    ax.plot(nEcath, df["i_total_mA_cm2"], color="black", lw=1.2, ls="--", label="$i_{total}$ (model)")
    if show_exp:
        ax.scatter(-E_exp_SHE, i_CO_exp, marker="s", facecolor="none", edgecolor="#1f77b4", s=50, label="$i_{CO}$ (exp)")
        ax.scatter(-E_exp_SHE, i_H2_exp, marker="^", facecolor="none", edgecolor="#d62728", s=50, label="$i_{H_2}$ (exp)")
    ax.set_xlabel("$-E_{cath}$ [V vs SHE]"); ax.set_ylabel("i [mA/cm$^2$]")
    ax.set_xlim(0.7, 3.1)
    ax.set_title("(a) Linear polarization"); ax.legend()

    ax = axes[1]
    ax.plot(nEcath, np.clip(df["i_CO_mA_cm2"], 1e-3, None), color="#1f77b4", lw=1.8, label="$i_{CO}$ (model)")
    ax.plot(nEcath, np.clip(df["i_H2_mA_cm2"], 1e-3, None), color="#d62728", lw=1.8, label="$i_{H_2}$ (model)")
    if show_exp:
        ax.scatter(-E_exp_SHE, i_CO_exp, marker="s", facecolor="none", edgecolor="#1f77b4", s=50)
        ax.scatter(-E_exp_SHE, i_H2_exp, marker="^", facecolor="none", edgecolor="#d62728", s=50)
    ax.set_yscale("log")
    ax.set_xlabel("$-E_{cath}$ [V vs SHE]"); ax.set_ylabel("i [mA/cm$^2$] (log)")
    ax.set_xlim(0.7, 3.1)
    ax.set_title("(b) Log polarization"); ax.legend()

    ax = axes[2]
    ax.plot(nEcath, df["FE_CO_pct"], color="#1f77b4", lw=1.8, label="FE$_{CO}$ (model)")
    ax.plot(nEcath, df["FE_H2_pct"], color="#d62728", lw=1.8, label="FE$_{H_2}$ (model)")
    if show_exp:
        ax.scatter(-E_exp_SHE, FE_CO_exp, marker="s", facecolor="none", edgecolor="#1f77b4", s=50)
        ax.scatter(-E_exp_SHE, FE_H2_exp, marker="^", facecolor="none", edgecolor="#d62728", s=50)
    ax.set_ylim(0, 100)
    ax.set_xlim(0.7, 3.1)
    ax.set_xlabel("$-E_{cath}$ [V vs SHE]"); ax.set_ylabel("FE [%]")
    ax.set_title("(c) Faradaic efficiency"); ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _color_ramp(vs):
    """light (mild, near -0.3) -> dark (cathodic, near -2.5)."""
    norm = Normalize(vmin=min(vs), vmax=max(vs))
    cmap = plt.get_cmap("viridis_r")   # mild(high V, less negative)=light... use plain sequential
    return cmap, norm


def plot_A2(liquid_csv, title, outpath):
    """3x3 liquid-phase diagnostic grid at all sweep voltages (§12.7 A2)."""
    df = pd.read_csv(liquid_csv)
    Vs = np.sort(df["V_RHE"].unique())
    cmap = plt.get_cmap("plasma")
    norm = Normalize(vmin=Vs.min(), vmax=Vs.max())

    fig, axes = plt.subplots(3, 3, figsize=(14, 13))
    fig.subplots_adjust(hspace=0.45, wspace=0.35, top=0.92, bottom=0.05, left=0.06, right=0.90)
    panels = [
        ("pH", "(a) pH [-]", False),
        ("c_CO2", "(b) CO$_2$(aq) [mol/m$^3$]", False),
        ("c_OH", "(c) OH$^-$ [mol/m$^3$]", False),
        ("c_HCO3", "(d) HCO$_3^-$ [mol/m$^3$]", False),
        ("c_CO3", "(e) CO$_3^{2-}$ [mol/m$^3$]", False),
        ("c_Cs", "(f) Cs$^+$ [mol/m$^3$]", False),
        ("c_H", "(g) H$^+$ [mol/m$^3$]", True),
        ("R_CO2R", "(h) R$_{CO2R}$ [mol/m$^3$/s]", False),
        ("R_HER", "(i) R$_{HER}$ [mol/m$^3$/s]", False),
    ]
    for ax, (col, label, sci) in zip(axes.flat, panels):
        for V in Vs:
            sub = df[df["V_RHE"] == V]
            ax.plot(sub["x_um"], sub[col], color=cmap(norm(V)), lw=0.6, alpha=0.85)
        ax.set_xlabel("x [$\\mu$m]"); ax.set_title(label)
        if sci:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label("V vs RHE@pH8 (light=mild, dark=cathodic)")
    fig.suptitle(title, y=0.985)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_B1(gas_csv, title, outpath, x_interface_um=0.0):
    """2x2 gas-phase diagnostic grid at all sweep voltages (§12.7 B1)."""
    df = pd.read_csv(gas_csv)
    Vs = np.sort(df["V_RHE"].unique())
    cmap = plt.get_cmap("plasma")
    norm = Normalize(vmin=Vs.min(), vmax=Vs.max())

    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
    fig.subplots_adjust(hspace=0.35, wspace=0.3, top=0.90, bottom=0.07, left=0.08, right=0.88)
    panels = [("p_CO2_atm", "(a) p$_{CO2}$ [atm]"), ("p_CO_atm", "(b) p$_{CO}$ [atm]"),
              ("p_H2_atm", "(c) p$_{H2}$ [atm]"), ("p_total_Pa", "(d) p$_{total}$ [Pa]")]
    for ax, (col, label) in zip(axes.flat, panels):
        for V in Vs:
            sub = df[df["V_RHE"] == V]
            ax.plot(sub["x_um"], sub[col], color=cmap(norm(V)), lw=0.6, alpha=0.85)
        ax.axvline(x_interface_um, color="gray", ls="--", lw=1)
        ax.set_xlabel("x [$\\mu$m]"); ax.set_title(label)
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.03, pad=0.02)
    cbar.set_label("V vs RHE@pH8 (light=mild, dark=cathodic)")
    fig.suptitle(title, y=0.98)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_D1(phi_s_csv, title, outpath, x_interface_m=0.0, n_curves=6):
    df = pd.read_csv(phi_s_csv)
    Vs = np.sort(df["V_RHE"].unique())
    pick = Vs[np.linspace(0, len(Vs)-1, min(n_curves, len(Vs))).astype(int)]
    cmap = plt.get_cmap("plasma"); norm = Normalize(vmin=Vs.min(), vmax=Vs.max())
    fig, ax = plt.subplots(figsize=(7,5.5))
    for V in pick:
        sub = df[df["V_RHE"] == V]
        ax.plot(sub["x_m"]*1e6, sub["phi_s_V_SHE"], color=cmap(norm(V)), lw=1.5, label=f"{V:.2f} V_RHE")
    ax.axvline(x_interface_m*1e6, color="gray", ls="--", lw=1)
    ax.set_xlabel("x [$\\mu$m] (0 = GDL/CL interface)"); ax.set_ylabel("$\\phi_s$ [V vs SHE]")
    ax.set_title(title); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(outpath, dpi=150); plt.close(fig)


def plot_D2(phi_L_csv, title, outpath, n_curves=6):
    df = pd.read_csv(phi_L_csv)
    Vs = np.sort(df["V_RHE"].unique())
    pick = Vs[np.linspace(0, len(Vs)-1, min(n_curves, len(Vs))).astype(int)]
    cmap = plt.get_cmap("plasma"); norm = Normalize(vmin=Vs.min(), vmax=Vs.max())
    fig, ax = plt.subplots(figsize=(7,5.5))
    for V in pick:
        sub = df[df["V_RHE"] == V]
        ax.plot(sub["x_frac"], sub["phi_L_V_SHE"]*1000, color=cmap(norm(V)), lw=1.5, label=f"{V:.2f} V_RHE")
    ax.set_xlabel("x / L_CL"); ax.set_ylabel("$\\phi_L$ [mV vs SHE, gauge at x=L_CL]")
    ax.set_title(title); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(outpath, dpi=150); plt.close(fig)
