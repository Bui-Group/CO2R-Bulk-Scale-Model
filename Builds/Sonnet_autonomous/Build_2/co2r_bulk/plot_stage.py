#!/usr/bin/env python3
"""plot_stage.py — required debug plots per guide §12.7 (A1/A2/B1/D1/D2). Reads the CSVs written
by postprocess.jl and does no physics of its own (avoids any risk of the Julia/Python physics
implementations silently drifting apart). Usage: python3 plot_stage.py <stage_dir_name> [title]
"""
import sys, os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Axis-conversion constants only (NOT model physics) — §15.2, §16.1 ──
R_gas, T, F_const, pH_anode = 8.314462618, 323.15, 96485.332, 8.0
nernst_fac = 2.303 * R_gas * T / F_const
V_RHE_to_SHE_offset = nernst_fac * pH_anode

E_exp_RHE = np.array([-0.816, -1.205, -1.644, -2.079, -2.413])
i_CO_exp  = np.array([35.1, 189.6, 299.1, 204.3, 64.6])
i_H2_exp  = np.array([29.9, 49.3, 119.6, 449.9, 850.5])
E_exp_SHE = E_exp_RHE - V_RHE_to_SHE_offset
FE_CO_exp = i_CO_exp / (i_CO_exp + i_H2_exp) * 100
FE_H2_exp = 100 - FE_CO_exp


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    out = {}
    for k in rows[0].keys():
        try:
            out[k] = np.array([float(row[k]) for row in rows])
        except ValueError:
            out[k] = np.array([row[k] for row in rows])
    return out


def plot_A1(stage_dir, out_dir, title):
    pol = read_csv(os.path.join(stage_dir, "polarization.csv"))
    negV = -pol["V_SHE"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, logy in zip(axs[:2], [False, True]):
        ax.plot(negV, pol["i_CO_mA_cm2"], color="#1f77b4", label="CO (model)")
        ax.plot(negV, pol["i_H2_mA_cm2"], color="#d62728", label="H2 (model)")
        ax.plot(negV, pol["i_tot_mA_cm2"], color="black", lw=1, label="total (model)")
        ax.scatter(-E_exp_SHE, i_CO_exp, marker="s", facecolor="none", edgecolor="#1f77b4", s=40, label="CO (exp)", zorder=5)
        ax.scatter(-E_exp_SHE, i_H2_exp, marker="^", facecolor="none", edgecolor="#d62728", s=40, label="H2 (exp)", zorder=5)
        ax.set_xlabel("-E_cath [V vs SHE]"); ax.set_ylabel("i [mA/cm^2]" + (" (log)" if logy else ""))
        ax.set_xlim(0.7, 3.1)
        if logy:
            ax.set_yscale("log"); ax.set_title("Log polarization")
        else:
            ax.set_title("Linear polarization"); ax.legend(fontsize=8)

    ax = axs[2]
    ax.plot(negV, pol["FE_CO"], color="#1f77b4", label="FE_CO (model)")
    ax.plot(negV, pol["FE_H2"], color="#d62728", label="FE_H2 (model)")
    ax.scatter(-E_exp_SHE, FE_CO_exp, marker="s", facecolor="none", edgecolor="#1f77b4", s=40, label="FE_CO (exp)", zorder=5)
    ax.scatter(-E_exp_SHE, FE_H2_exp, marker="^", facecolor="none", edgecolor="#d62728", s=40, label="FE_H2 (exp)", zorder=5)
    ax.set_xlabel("-E_cath [V vs SHE]"); ax.set_ylabel("FE [%]"); ax.set_ylim(0, 100); ax.set_xlim(0.7, 3.1)
    ax.set_title("Faradaic efficiency"); ax.legend(fontsize=8)

    fig.suptitle(f"{title} — A1 polarization summary")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "A1_polarization.png"), dpi=150)
    plt.close(fig)


def _voltage_ramp_plot(ax, prof, Vs, xkey, ykey, cmap, norm):
    for V in Vs:
        mask = prof["V_RHE"] == V
        x = prof[xkey][mask]; y = prof[ykey][mask]
        order = np.argsort(x)
        ax.plot(x[order], y[order], color=cmap(norm(V)), lw=0.8)


def plot_A2(stage_dir, out_dir, title):
    prof = read_csv(os.path.join(stage_dir, "liquid_profiles.csv"))
    Vs = np.unique(prof["V_RHE"])
    cmap = matplotlib.colormaps["viridis"]
    norm = mcolors.Normalize(vmin=Vs.min(), vmax=Vs.max())   # dark=cathodic(min), light=mild(max)

    fields = [("pH", "pH [-]"), ("cCO2", "CO2(aq) [mol/m^3]"), ("cOH", "OH- [mol/m^3]"),
              ("cHCO3", "HCO3- [mol/m^3]"), ("cCO3", "CO3^2- [mol/m^3]"), ("cCs", "Cs+ [mol/m^3]"),
              ("cH", "H+ [mol/m^3]"), ("R_CO2R", "R_CO2R [mol/m^3/s]"), ("R_HER", "R_HER [mol/m^3/s]")]
    fig, axs = plt.subplots(3, 3, figsize=(14, 11))
    for idx, (key, label) in enumerate(fields):
        ax = axs[idx // 3, idx % 3]
        _voltage_ramp_plot(ax, prof, Vs, "x_um", key, cmap, norm)
        ax.set_xlabel("x [μm]"); ax.set_ylabel(label)
        if key == "cH":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs, orientation="vertical", fraction=0.025, pad=0.02)
    cbar.set_label("V_RHE [V]  (light=mild, dark=cathodic)")
    fig.suptitle(f"{title} — A2 liquid-phase diagnostics (all {len(Vs)} voltages)")
    fig.savefig(os.path.join(out_dir, "A2_liquid_3x3.png"), dpi=150)
    plt.close(fig)


def plot_B1(stage_dir, out_dir, title):
    path = os.path.join(stage_dir, "gas_profiles.csv")
    if not os.path.exists(path):
        return
    prof = read_csv(path)
    Vs = np.unique(prof["V_RHE"])
    cmap = matplotlib.colormaps["viridis"]
    norm = mcolors.Normalize(vmin=Vs.min(), vmax=Vs.max())

    fields = [("p_CO2_atm", "p_CO2 [atm]"), ("p_CO_atm", "p_CO [atm]"),
              ("p_H2_atm", "p_H2 [atm]"), ("p_total_Pa", "p_total [Pa]")]
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    interface_um = 0.0
    for idx, (key, label) in enumerate(fields):
        ax = axs[idx // 2, idx % 2]
        _voltage_ramp_plot(ax, prof, Vs, "x_um", key, cmap, norm)
        ax.axvline(interface_um, color="gray", ls="--", lw=1)
        ax.set_xlabel("x [μm] (GDL/CL interface dashed)"); ax.set_ylabel(label)

    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs, orientation="vertical", fraction=0.03, pad=0.02)
    cbar.set_label("V_RHE [V]  (light=mild, dark=cathodic)")
    fig.suptitle(f"{title} — B1 gas-phase diagnostics (all {len(Vs)} voltages)")
    fig.savefig(os.path.join(out_dir, "B1_gas_2x2.png"), dpi=150)
    plt.close(fig)


def plot_D1(stage_dir, out_dir, title):
    path = os.path.join(stage_dir, "phi_s_profile.csv")
    if not os.path.exists(path):
        return
    prof = read_csv(path)
    Vs_all = np.unique(prof["V_RHE"])
    # low / medium / high current -> pick mild, middle, and most-cathodic voltages
    picks = [Vs_all.max(), Vs_all[len(Vs_all)//2], Vs_all.min()]
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    for V, c in zip(picks, colors):
        mask = prof["V_RHE"] == V
        x = prof["x_m"][mask] * 1e6
        y = prof["phi_s"][mask]
        order = np.argsort(x)
        ax.plot(x[order], y[order], color=c, label=f"V_RHE={V:.3f}")
    ax.axvline(0.0, color="gray", ls="--", lw=1)
    ax.set_xlabel("x [μm] (GDL/CL interface dashed; GDL<0<CL)")
    ax.set_ylabel("phi_s [V vs SHE]")
    ax.set_title(f"{title} — D1 solid potential profile")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "D1_phi_s_profile.png"), dpi=150)
    plt.close(fig)


def plot_D2(stage_dir, out_dir, title):
    path = os.path.join(stage_dir, "phi_L_profile.csv")
    if not os.path.exists(path):
        return
    prof = read_csv(path)
    Vs_all = np.unique(prof["V_RHE"])
    picks = [Vs_all.max(), Vs_all[len(Vs_all)//2], Vs_all.min()]
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    for V, c in zip(picks, colors):
        mask = prof["V_RHE"] == V
        x = prof["x_over_L"][mask]
        y = prof["phi_L"][mask] * 1000   # V -> mV
        order = np.argsort(x)
        ax.plot(x[order], y[order], color=c, label=f"V_RHE={V:.3f}")
    ax.set_xlabel("x / L_CL"); ax.set_ylabel("phi_L [mV vs SHE, gauge phi_L(L_CL)=0]")
    ax.set_title(f"{title} — D2 liquid potential profile")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "D2_phi_L_profile.png"), dpi=150)
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print("usage: plot_stage.py <output_dir_name e.g. output_stage1> [title]")
        sys.exit(1)
    dirname = sys.argv[1]
    title = sys.argv[2] if len(sys.argv) > 2 else dirname
    here = os.path.dirname(os.path.abspath(__file__))
    stage_dir = os.path.join(here, dirname)
    out_dir = stage_dir
    os.makedirs(out_dir, exist_ok=True)

    plot_A1(stage_dir, out_dir, title)
    plot_A2(stage_dir, out_dir, title)
    plot_B1(stage_dir, out_dir, title)
    plot_D1(stage_dir, out_dir, title)
    plot_D2(stage_dir, out_dir, title)
    print(f"Plots written to {out_dir}")


if __name__ == "__main__":
    main()
