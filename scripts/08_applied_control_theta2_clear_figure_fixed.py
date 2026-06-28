"""
08_applied_control_theta2_clear_figure_fixed.py

Generates the applied greenhouse biological-control figure for the model
with nonlinear prey growth f(n) = 1 - n^2.

The system is dimensional:
    N' = eps N [1 - (N/K)^2] - alpha N P,
    P' = -gamma P + beta P Q,
    Q' = a(R-Q),
    R' = a(N-R).

The figure shows:
    1. A projected orbit in the (N,P)-plane with arrows.
    2. One period of the prey time series.
    3. One period of the predator time series.

Outputs:
    figures/applied_control_theta2_orbit_clear_en.pdf
    figures/applied_control_theta2_orbit_clear_en.png
    data/applied_theta2_long_trajectory.csv
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Output paths are relative to the location of this script.
# ---------------------------------------------------------------------
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

ROOT_DIR = BASE_DIR.parent if BASE_DIR.name == "scripts" else BASE_DIR
DATA_DIR = ROOT_DIR / "data"
FIG_DIR = ROOT_DIR / "figures"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Applied parameters
# ---------------------------------------------------------------------
K = 88.206925
EPS = 0.154743
GAMMA = 0.02
BETA = 0.001
ALPHA = 0.103094
A = 0.18                    # day^{-1}
A_H = 0.353291              # critical value in day^{-1}

N_STAR = GAMMA / BETA
P_STAR = EPS * (1.0 - (N_STAR / K) ** 2) / ALPHA
Q_STAR = N_STAR
R_STAR = N_STAR
SIGMA = np.array([N_STAR, P_STAR, Q_STAR, R_STAR], dtype=float)


# ---------------------------------------------------------------------
# ODE system and RK4 integrator
# ---------------------------------------------------------------------
def rhs(y):
    N, P, Q, R = y
    dN = EPS * N * (1.0 - (N / K) ** 2) - ALPHA * N * P
    dP = -GAMMA * P + BETA * P * Q
    dQ = A * (R - Q)
    dR = A * (N - R)
    return np.array([dN, dP, dQ, dR], dtype=float)


def rk4_integrate(y0, t0, tf, dt):
    n_steps = int(round((tf - t0) / dt)) + 1
    t = np.linspace(t0, tf, n_steps)
    y = np.zeros((n_steps, len(y0)), dtype=float)
    y[0] = np.asarray(y0, dtype=float)

    for i in range(n_steps - 1):
        h = t[i + 1] - t[i]
        k1 = rhs(y[i])
        k2 = rhs(y[i] + 0.5 * h * k1)
        k3 = rhs(y[i] + 0.5 * h * k2)
        k4 = rhs(y[i] + h * k3)
        y[i + 1] = y[i] + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        y[i + 1] = np.maximum(y[i + 1], 1.0e-12)
    return t, y


def save_csv(path, t, y):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "N", "P", "Q", "R"])
        for ti, yi in zip(t, y):
            writer.writerow([ti, yi[0], yi[1], yi[2], yi[3]])


def local_maxima_indices(x):
    return np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0] + 1


def add_arrows(ax, x, y, indices, color="black", lw=1.0, ms=12):
    for idx in indices:
        if 0 <= idx < len(x) - 1:
            ax.annotate(
                "",
                xy=(x[idx + 1], y[idx + 1]),
                xytext=(x[idx], y[idx]),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw, mutation_scale=ms),
            )


# ---------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------
Y0 = np.array([25.0, 1.0, 20.0, 20.0], dtype=float)
T0 = 0.0
TF = 1300.0
DT = 0.02

t, y = rk4_integrate(Y0, T0, TF, DT)
N = y[:, 0]
P = y[:, 1]

csv_path = DATA_DIR / "applied_theta2_long_trajectory.csv"
save_csv(csv_path, t, y)

# Use the last two maxima of N(t) to identify one period after the transient.
peak_idx = local_maxima_indices(N)
if len(peak_idx) < 3:
    raise RuntimeError("Not enough peaks were detected. Increase TF.")
start_idx = peak_idx[-2]
end_idx = peak_idx[-1]
period_mask = (np.arange(len(t)) >= start_idx) & (np.arange(len(t)) <= end_idx)
transient_mask = np.arange(len(t)) < start_idx

T_period = t[end_idx] - t[start_idx]


# ---------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------
fig = plt.figure(figsize=(11.0, 6.2))
gs = fig.add_gridspec(2, 2, width_ratios=[1.20, 1.0], height_ratios=[1.0, 1.0])

# Phase portrait
ax_phase = fig.add_subplot(gs[:, 0])
ax_phase.plot(N[transient_mask], P[transient_mask], linewidth=1.0, alpha=0.45, label="Transient trajectory")
ax_phase.plot(N[period_mask], P[period_mask], linewidth=2.5, label="Periodic orbit")
ax_phase.scatter([N_STAR], [P_STAR], s=45, zorder=5, label="Coexistence equilibrium")

# Arrows on the periodic orbit
N_per = N[period_mask]
P_per = P[period_mask]
arrow_positions = [int(frac * (len(N_per) - 2)) for frac in (0.10, 0.30, 0.50, 0.70, 0.88)]
add_arrows(ax_phase, N_per, P_per, arrow_positions, color="black", lw=1.0, ms=11)

ax_phase.set_xlabel(r"Prey density $N$")
ax_phase.set_ylabel(r"Predator density $P$")
ax_phase.set_title(r"Projected orbit, $a=0.18<a_H$")
ax_phase.legend(fontsize=8, loc="upper right")

# Prey time series over one period
ax_N = fig.add_subplot(gs[0, 1])
ax_N.plot(t[period_mask], N[period_mask], linewidth=2.0)
ax_N.set_ylabel(r"Prey density $N(t)$")
ax_N.set_title("One period after the transient")

# Predator time series over one period
ax_P = fig.add_subplot(gs[1, 1], sharex=ax_N)
ax_P.plot(t[period_mask], P[period_mask], linewidth=2.0)
ax_P.set_xlabel("Time (days)")
ax_P.set_ylabel(r"Predator density $P(t)$")

fig.tight_layout()

pdf_path = FIG_DIR / "applied_control_theta2_orbit_clear_en.pdf"
png_path = FIG_DIR / "applied_control_theta2_orbit_clear_en.png"
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Generated files:")
print(f"  {pdf_path}")
print(f"  {png_path}")
print(f"  {csv_path}")
print("Summary:")
print(f"  Equilibrium = ({N_STAR:.6f}, {P_STAR:.6f}, {Q_STAR:.6f}, {R_STAR:.6f})")
print(f"  a_H = {A_H:.6f} day^(-1), a = {A:.6f} day^(-1)")
print(f"  Period shown = {T_period:.6f} days")
