"""Render figures/*.png from results/*.npz."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
R = lambda f: np.load(os.path.join(ROOT, "results", f))  # noqa: E731
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 1.8, "legend.frameon": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def title(ax, t):
    ax.set_title(t, fontsize=10, color=INK)


# 1. Spectra for four initial states
d = R("spectra.npz")
E, ev = d["energies"], d["evals"]
fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
for ax, s in zip(axes.flat, ["|++>", "|+->", "|00>", "|01>"]):
    w = d[f"weights {s}"]
    keep = w > 1e-3
    ax.vlines(ev[keep], 0, w[keep], color=GRID, lw=6, zorder=0)
    ax.plot(E, d[f"exact {s}"], color=C[0], label="exact rotations")
    ax.plot(E, d[f"gridsynth_1e-3 {s}"], color=C[2], ls=(0, (4, 2)), label="Clifford+T, gridsynth ε = 1e-3")
    ax.plot(E, d[f"nearest_pi4_field {s}"], color=C[1], lw=1.2, label="Clifford+T, field rounded to π/4")
    title(ax, f"initial state {s.replace('>', '⟩')}")
for ax in axes[1]:
    ax.set_xlabel("energy E")
for ax in axes[:, 0]:
    ax.set_ylabel("success probability")
axes[0, 0].legend(fontsize=8, loc="upper left")
fig.text(0.5, -0.01, "grey bars: exact eigenvalues of H = −Z₁Z₂ + X₁ + X₂, height = overlap with the initial state",
         ha="center", fontsize=8, color=MUTED)
fig.savefig(os.path.join(FIG, "spectra.png"))
plt.close(fig)

# 2. Error budget
b = R("budget.npz")
dts, epss = b["dts"], b["epss"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, n in zip(axes, (2, 4)):
    err = b[f"err_n{n}"]
    ax.plot(dts, err[0], color=INK, lw=2, label="exact rotations (Trotter only)")
    for i, eps in enumerate(epss):
        ax.plot(dts, err[i + 1], color=C[i], lw=1.4, marker="o", ms=2.5, label=f"ε = {eps:.0e}")
    ax.set(xscale="log", yscale="log", xlabel="Trotter step dt", ylim=(1e-7, 10))
    title(ax, f"n = {n} spins")
axes[0].set_ylabel("max eigenvalue error of H_eff")
axes[0].text(0.035, 3e-7, "left of each minimum:\nsynthesis error ∝ ε / dt", fontsize=8, color=MUTED)
axes[0].text(0.08, 2e-5, "Trotter error\n∝ dt²", fontsize=8, color=MUTED)
axes[1].legend(fontsize=7.5, loc="lower right", ncol=2)
fig.savefig(os.path.join(FIG, "error_budget.png"))
plt.close(fig)

# 3. Cost vs accuracy (n = 2)
fig, ax = plt.subplots(figsize=(7, 4.2))
err, tc = b["err_n2"], b["tcount_n2"]
for i, eps in enumerate(epss):
    ax.plot(err[i + 1], tc[i] / dts, "o", ms=3.5, color=C[i], label=f"ε = {eps:.0e}")
j = np.argmin(np.abs(dts - 0.005))
ax.plot(err[1:, j], tc[:, j] / dts[j], "s", ms=9, mfc="none", mec=INK, mew=1.2, label="dt ≈ 0.005 (many small steps)")
ax.set(xscale="log", yscale="log", xlabel="max eigenvalue error", ylabel="T gates per unit of simulated time")
title(ax, "Cost of accuracy, n = 2: each point is one (ε, dt) choice")
ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
fig.savefig(os.path.join(FIG, "cost_vs_accuracy.png"))
plt.close(fig)

# 4. Pitfall: small steps with a loose eps
p = R("pitfall.npz")
w = R("spectra.npz")["weights |00>"]
fig, ax = plt.subplots(figsize=(8, 3.4))
keep = w > 1e-3
ax.vlines(ev[keep], 0, w[keep], color=GRID, lw=6, zorder=0)
for (lab, c, ls) in (("dt=0.005, exact", INK, "-"), ("dt=0.1, eps=1e-3", C[2], "-"), ("dt=0.005, eps=1e-3", C[1], "-")):
    ax.plot(p["energies"], p[lab], color=c, ls=ls, lw=1.6, label=lab.replace("eps", "ε"))
ax.set(xlabel="energy E", ylabel="success probability")
title(ax, "Initial state |00⟩: the same ε = 1e-3 is accurate with dt = 0.1 and biased with dt = 0.005")
ax.legend(fontsize=8)
fig.savefig(os.path.join(FIG, "step_size_pitfall.png"))
plt.close(fig)
print("figures written")
