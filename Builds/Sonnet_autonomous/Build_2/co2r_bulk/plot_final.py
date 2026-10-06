#!/usr/bin/env python3
"""plot_final.py — the two remaining required figures:
  1. All-5-stages polarization overlay (explicitly required at the Stage 5 stop gate, §11).
  2. §18.2 dedicated FE_CO/FE_H2 overlay figure.
No physics; reads the polarization.csv files already written by postprocess.jl.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_stage import read_csv, E_exp_SHE, i_CO_exp, i_H2_exp, FE_CO_exp, FE_H2_exp

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = [("output_stage1", "Stage 1"), ("output_stage2", "Stage 2"), ("output_stage3", "Stage 3"),
          ("output_stage4", "Stage 4"), ("output_stage5", "Stage 5")]
COLORS = ["#9edae5", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]


def plot_all_stages_overlay():
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for (dirname, label), color in zip(STAGES, COLORS):
        pol = read_csv(os.path.join(HERE, dirname, "polarization.csv"))
        negV = -pol["V_SHE"]
        axs[0].plot(negV, pol["i_tot_mA_cm2"], color=color, label=label)
        axs[1].plot(negV, pol["i_tot_mA_cm2"], color=color, label=label)
    for ax in axs:
        ax.scatter(-E_exp_SHE, i_CO_exp + i_H2_exp, marker="o", facecolor="none",
                   edgecolor="black", s=40, label="i_total (exp)", zorder=5)
        ax.set_xlabel("-E_cath [V vs SHE]"); ax.set_ylabel("i_total [mA/cm^2]"); ax.set_xlim(0.7, 3.1)
    axs[1].set_yscale("log")
    axs[0].set_title("Linear"); axs[1].set_title("Log")
    axs[0].legend(fontsize=8)
    fig.suptitle("Total current density — overlay of all 5 stages (pre-fit kinetics in Stages 1-4,\n"
                 "fitted kinetics in Stage 5) vs. Romiluyi et al. 2023")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "output_stage5", "all_stages_polarization_overlay.png"), dpi=150)
    plt.close(fig)


def plot_FE_overlay():
    pol = read_csv(os.path.join(HERE, "output_stage5", "polarization.csv"))
    negV = -pol["V_SHE"]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(negV, pol["FE_CO"], color="#1f77b4", label="FE_CO (Stage 5 model)")
    ax.plot(negV, pol["FE_H2"], color="#d62728", label="FE_H2 (Stage 5 model)")
    ax.scatter(-E_exp_SHE, FE_CO_exp, marker="s", facecolor="none", edgecolor="#1f77b4", s=50, label="FE_CO (exp)")
    ax.scatter(-E_exp_SHE, FE_H2_exp, marker="^", facecolor="none", edgecolor="#d62728", s=50, label="FE_H2 (exp)")
    ax.axhline(50, color="gray", ls=":", lw=1)
    ax.set_xlabel("-E_cath [V vs SHE]"); ax.set_ylabel("FE [%]"); ax.set_ylim(0, 100); ax.set_xlim(0.7, 3.1)
    ax.set_title("§18.2 FE_CO / FE_H2 crossover overlay (Stage 5, fitted kinetics)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "output_stage5", "FE_overlay.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_all_stages_overlay()
    plot_FE_overlay()
    print("Final plots written to output_stage5/")
