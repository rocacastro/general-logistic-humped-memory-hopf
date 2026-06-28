"""
02_hyperlogistic_bifurcation_diagram_en.py

Numerical bifurcation diagram for the hyperlogistic example.

The figure labels are in English.
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


K = 2.0
alpha = 1.0
beta = 1.0
gamma = 1.0

b = gamma / (K * beta)
sigma = np.array([b, 1.0 / 8.0, b, b], dtype=float)

a_H = 0.595743941976559
omega_0 = 0.323899435630521


def rhs(t, x, a):
    n, p, q, r = x
    return np.array([
        n * (1.0 - n) ** 2 - 2.0 * n * p,
        -p + 2.0 * p * q,
        a * (r - q),
        a * (n - r),
    ], dtype=float)


def integrate_for_a(a, x0, t_final=4000.0, dt=0.25):
    t_eval = np.arange(0.0, t_final + dt, dt)
    sol = solve_ivp(
        fun=lambda t, x: rhs(t, x, a),
        t_span=(0.0, t_final),
        y0=x0,
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed for a={a}: {sol.message}")
    return sol.t, sol.y.T


def extrema_after_transient(t, y, transient_fraction=0.65):
    start = int(transient_fraction * len(t))
    n_values = y[start:, 0]
    return float(np.min(n_values)), float(np.max(n_values))


def main():
    figures_dir = Path("figures_en")
    data_dir = Path("data_en")
    figures_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)

    a_cycle_values = np.linspace(a_H - 0.025, a_H - 0.002, 28)
    x0 = sigma + np.array([0.01, 0.002, 0.0, 0.0])

    rows = []
    for a in a_cycle_values:
        t, y = integrate_for_a(a, x0)
        n_min, n_max = extrema_after_transient(t, y)
        rows.append({"a": float(a), "n_min": n_min, "n_max": n_max})

    csv_path = data_dir / "bifurcation_extrema_hyperlogistic_en.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["a", "n_min", "n_max"])
        writer.writeheader()
        writer.writerows(rows)

    a_vals = np.array([row["a"] for row in rows])
    n_min_vals = np.array([row["n_min"] for row in rows])
    n_max_vals = np.array([row["n_max"] for row in rows])

    a_left = np.linspace(a_H - 0.03, a_H, 200)
    a_right = np.linspace(a_H, a_H + 0.03, 200)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.plot(
        a_left,
        np.full_like(a_left, b),
        linestyle="--",
        linewidth=2.0,
        label="Unstable equilibrium",
    )
    ax.plot(
        a_right,
        np.full_like(a_right, b),
        linestyle="-",
        linewidth=2.0,
        label="Stable equilibrium",
    )

    ax.plot(
        a_vals,
        n_max_vals,
        marker="o",
        markersize=3.5,
        linewidth=1.5,
        label="Maxima of n(t)",
    )
    ax.plot(
        a_vals,
        n_min_vals,
        marker="o",
        markersize=3.5,
        linewidth=1.5,
        label="Minima of n(t)",
    )

    ax.axvline(a_H, linestyle=":", linewidth=1.8)
    ax.text(a_H + 0.001, 0.53, r"$a=a_{\mathrm{H}}$", fontsize=11)

    ax.set_xlabel(r"$a$")
    ax.set_ylabel(r"$n$")
    ax.set_title("Local bifurcation diagram: hyperlogistic case")
    ax.set_xlim(a_H - 0.03, a_H + 0.03)

    y_min = min(float(np.min(n_min_vals)), b)
    y_max = max(float(np.max(n_max_vals)), b)
    margin = 0.08 * (y_max - y_min)
    ax.set_ylim(y_min - margin, y_max + margin)

    ax.legend(frameon=True)
    fig.tight_layout()

    pdf_path = figures_dir / "bifurcation_diagram_a_hyperlogistic_en.pdf"
    png_path = figures_dir / "bifurcation_diagram_a_hyperlogistic_en.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Generated files:")
    print(f"  {pdf_path}")
    print(f"  {png_path}")
    print(f"  {csv_path}")


if __name__ == "__main__":
    main()
