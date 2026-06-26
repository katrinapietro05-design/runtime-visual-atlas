from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path("charts/generated")
OUT.mkdir(parents=True, exist_ok=True)

# Deterministic synthetic runtime series for visualization only.
t = np.linspace(0, 10, 240)
omega_root = np.ones_like(t) * 0.82
omega_obs = 0.82 + 0.22 * np.sin(0.8 * t) * np.exp(-0.11 * t) + 0.05 * np.sin(3.1 * t)
delta_r = np.abs(omega_obs - omega_root)

weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
components = np.vstack([
    0.78 + 0.08 * np.sin(0.55 * t),
    0.72 + 0.10 * np.cos(0.45 * t + 0.4),
    0.82 - 0.16 * np.exp(-0.18 * t),
    0.68 + 0.18 * (1 - np.exp(-0.25 * t)),
    0.75 + 0.06 * np.sin(1.1 * t + 0.7),
])
ii = weights @ components

love_signal = 0.72 + 0.18 * np.exp(-0.18 * t) + 0.06 * np.sin(0.7 * t)
dt = t[1] - t[0]
pe = np.cumsum(love_signal) * dt
re = -np.gradient(pe, t)
vc = np.gradient(1 - delta_r, t)
collapse = np.maximum(0, np.gradient(delta_r, t))
phi = np.gradient(1 - ii, t)


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=220)
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)


fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(t, omega_obs, label=r"$\Omega_{obs}$")
ax.plot(t, omega_root, label=r"$\Omega_R$")
ax.plot(t, delta_r, label=r"$\Delta R(t)=\|\Omega_{obs}-\Omega_R\|$")
ax.set(title="SOVRINT Root–Observation Drift", xlabel="Time", ylabel="Normalized state")
ax.legend(); ax.grid(True, alpha=0.25)
save(fig, "01_root_observation_drift")

fig, ax = plt.subplots(figsize=(12, 7))
for idx, row in enumerate(components, start=1):
    ax.plot(t, row, alpha=0.55, label=fr"$I_{idx}(t)$")
ax.plot(t, ii, linewidth=3, label=r"$II(t)=\sum \omega_i I_i(t)$")
ax.set(title="SOVRINT Weighted Integrity Index", xlabel="Time", ylabel="Integrity", ylim=(0, 1.05))
ax.legend(ncol=3); ax.grid(True, alpha=0.25)
save(fig, "02_integrity_index")

fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(t, pe, label=r"$Pe(t)=\int L(t)\,dt$")
ax2 = ax.twinx(); ax2.plot(t, re, label=r"$Re(t)=-dPe/dt$")
ax.set(title="SOVRINT Presence Integral and Resonance Derivative", xlabel="Time", ylabel="Accumulated presence")
ax2.set_ylabel("Resonance derivative")
lines = ax.get_lines() + ax2.get_lines()
ax.legend(lines, [line.get_label() for line in lines], loc="upper left")
ax.grid(True, alpha=0.25)
save(fig, "03_presence_resonance")

fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(t, vc, label=r"$Vc(t)=d/dt(1-\Delta R)$")
ax.plot(t, collapse, label="Collapse velocity proxy")
ax.axhline(0, linewidth=1)
ax.set(title="SOVRINT Correction Velocity vs Collapse", xlabel="Time", ylabel="Velocity")
ax.legend(); ax.grid(True, alpha=0.25)
save(fig, "04_correction_vs_collapse")

fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(t, phi, label=r"$\Phi=\nabla(1-II)$")
ax.fill_between(t, 0, phi, alpha=0.25); ax.axhline(0, linewidth=1)
ax.set(title="SOVRINT Coherence Vector Metric", xlabel="Time", ylabel="Gradient")
ax.legend(); ax.grid(True, alpha=0.25)
save(fig, "05_coherence_vector")

rows = ["Physics", "Laws", "Temporal", "Emotional", "Omics", "Governance", "Correction", "Cartography"]
cols = ["Ingress", "Temporal", "Metrics", "Correlation", "Correction", "Governance", "Ledger", "Security", "Viz", "Closure", "Compliance"]
rng = np.random.default_rng(505)
coverage = np.clip(rng.normal(0.72, 0.16, (len(rows), len(cols))), 0.25, 1.0)
fig, ax = plt.subplots(figsize=(14, 8))
im = ax.imshow(coverage, aspect="auto")
ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
ax.set_yticks(range(len(rows)), rows)
ax.set_title("Coverage Heat Map — Canon (SOV) ↔ Runtime (ONX)")
fig.colorbar(im, ax=ax, label="Coverage intensity")
save(fig, "06_canon_runtime_coverage")

print(f"Generated 12 files in {OUT}")
