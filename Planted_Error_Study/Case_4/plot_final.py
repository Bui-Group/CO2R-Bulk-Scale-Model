#!/usr/bin/env python3
"""plot_final.py — §18 experimental-comparison figures for the final model.

Usage: python3 plot_final.py <polarization.csv> <out_dir> [label]

Produces:
  Case4_final_polarization.png  — §18.1 3-panel (linear / log / FE) vs Lees, open markers.
  Case4_final_FE.png            — §18.2 FE_CO & FE_H2 crossover overlay.
All voltage axes: −E_cath [V vs SHE], range 0.7–3.1.
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OFFSET = 2.303 * 8.314462618 * 323.15 / 96485.332 * 8.0
E_exp_RHE = np.array([-0.816, -1.205, -1.644, -2.079, -2.413])
i_CO_exp = np.array([35.1, 189.6, 299.1, 204.3, 64.6])
i_H2_exp = np.array([29.9, 49.3, 119.6, 449.9, 850.5])
E_exp_SHE = E_exp_RHE - OFFSET
i_tot_exp = i_CO_exp + i_H2_exp
FE_CO_exp = 100 * i_CO_exp / i_tot_exp
FE_H2_exp = 100 * i_H2_exp / i_tot_exp
xe = -E_exp_SHE

csv = sys.argv[1]; out_dir = sys.argv[2]
label = sys.argv[3] if len(sys.argv) > 3 else "Case 4 model"
d = np.genfromtxt(csv, delimiter=",", names=True)
o = np.argsort(-np.atleast_1d(d["V_she"]))
x = -np.atleast_1d(d["V_she"])[o]
iCO = np.atleast_1d(d["i_CO"])[o]; iH2 = np.atleast_1d(d["i_H2"])[o]
itot = np.atleast_1d(d["i_tot"])[o]; feco = np.atleast_1d(d["FE_CO"])[o]; feh2 = np.atleast_1d(d["FE_H2"])[o]

# ── §18.1 3-panel summary ──
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
for a in ax:
    a.set_xlabel("−E$_{cath}$ [V vs SHE]"); a.set_xlim(0.7, 3.1); a.grid(alpha=0.3)
ax[0].plot(x, iCO, "-", color="C0", label="i$_{CO}$"); ax[0].plot(x, iH2, "-", color="C3", label="i$_{H_2}$")
ax[0].plot(x, itot, "-", color="k", label="i$_{tot}$")
ax[0].plot(xe, i_CO_exp, "s", mfc="none", color="C0", ms=9, label="i$_{CO}$ exp")
ax[0].plot(xe, i_H2_exp, "^", mfc="none", color="C3", ms=9, label="i$_{H_2}$ exp")
ax[0].plot(xe, i_tot_exp, "o", mfc="none", color="k", ms=9, label="i$_{tot}$ exp")
ax[0].set_ylabel("current density [mA/cm²]"); ax[0].set_title("(a) linear polarization"); ax[0].legend(fontsize=8)
ax[1].semilogy(x, np.clip(iCO, 1e-2, None), "-", color="C0"); ax[1].semilogy(x, np.clip(iH2, 1e-2, None), "-", color="C3")
ax[1].semilogy(x, np.clip(itot, 1e-2, None), "-", color="k")
ax[1].semilogy(xe, i_CO_exp, "s", mfc="none", color="C0", ms=9); ax[1].semilogy(xe, i_H2_exp, "^", mfc="none", color="C3", ms=9)
ax[1].semilogy(xe, i_tot_exp, "o", mfc="none", color="k", ms=9)
ax[1].set_ylabel("current density [mA/cm²] (log)"); ax[1].set_title("(b) log polarization")
ax[2].plot(x, feco, "-", color="C0", label="FE$_{CO}$"); ax[2].plot(x, feh2, "-", color="C3", label="FE$_{H_2}$")
ax[2].plot(xe, FE_CO_exp, "s", mfc="none", color="C0", ms=9, label="FE$_{CO}$ exp")
ax[2].plot(xe, FE_H2_exp, "^", mfc="none", color="C3", ms=9, label="FE$_{H_2}$ exp")
ax[2].set_ylabel("faradaic efficiency [%]"); ax[2].set_ylim(0, 100); ax[2].set_title("(c) faradaic efficiency"); ax[2].legend(fontsize=8)
fig.suptitle(f"{label} — experimental comparison (Lees et al. 2024)")
fig.tight_layout(); fig.savefig(os.path.join(out_dir, "Case4_final_polarization.png"), dpi=140); plt.close(fig)

# ── §18.2 FE crossover ──
fig, a = plt.subplots(figsize=(8, 6))
a.plot(x, feco, "-", color="C0", lw=2, label="FE$_{CO}$ model")
a.plot(x, feh2, "-", color="C3", lw=2, label="FE$_{H_2}$ model")
a.plot(xe, FE_CO_exp, "s", mfc="none", color="C0", ms=10, label="FE$_{CO}$ exp")
a.plot(xe, FE_H2_exp, "^", mfc="none", color="C3", ms=10, label="FE$_{H_2}$ exp")
a.axhline(50, ls=":", color="gray", lw=1)
a.set_xlabel("−E$_{cath}$ [V vs SHE]"); a.set_ylabel("faradaic efficiency [%]")
a.set_xlim(0.7, 3.1); a.set_ylim(0, 100); a.grid(alpha=0.3)
a.set_title(f"{label} — FE$_{{CO}}$/FE$_{{H_2}}$ crossover"); a.legend(fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(out_dir, "Case4_final_FE.png"), dpi=140); plt.close(fig)
print("§18 final plots written to", out_dir)
