"""
06_floquet_multipliers_hyperlogistic_en.py

Floquet multipliers for the stable periodic orbit in the hyperlogistic example.

System:
    n' = n(1-n)^2 - 2 n p
    p' = -p + 2 p q
    q' = a(r-q)
    r' = a(n-r)

Parameters:
    a = 0.56 < a_H,
    K = 2, alpha = beta = gamma = 1.

The code:
1. Integrates a trajectory until it approaches the stable periodic orbit.
2. Detects local maxima of n(t).
3. Uses the last detected peak as an initial guess.
4. Refines the periodic orbit by a shooting method.
5. Integrates the variational equation over one period.
6. Computes the Floquet multipliers.

Requires:
    numpy, scipy, matplotlib

Usage:
    python 06_floquet_multipliers_hyperlogistic_en.py
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root


# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------
a_H = 0.595743941976559
a_value = 0.56
sigma = np.array([0.5, 0.125, 0.5, 0.5], dtype=float)


# ---------------------------------------------------------------------
# Vector field and Jacobian
# ---------------------------------------------------------------------
def rhs(t, x, a=a_value):
    n, p, q, r = x
    return np.array([
        n * (1.0 - n) ** 2 - 2.0 * n * p,
        -p + 2.0 * p * q,
        a * (r - q),
        a * (n - r),
    ], dtype=float)


def jacobian(x, a=a_value):
    n, p, q, r = x
    return np.array([
        [1.0 - 4.0 * n + 3.0 * n**2 - 2.0 * p, -2.0 * n, 0.0, 0.0],
        [0.0, -1.0 + 2.0 * q, 2.0 * p, 0.0],
        [0.0, 0.0, -a, a],
        [a, 0.0, 0.0, -a],
    ], dtype=float)


# ---------------------------------------------------------------------
# Peak event: local maxima of n(t)
# ---------------------------------------------------------------------
def peak_event(t, x):
    # Avoid detecting the initial time as an event.
    if t < 1.0e-6:
        return 1.0
    return rhs(t, x)[0]

peak_event.direction = -1.0
peak_event.terminal = False


def find_peak_guess():
    """Find a good initial guess on the periodic orbit using peak events."""
    x0 = sigma + np.array([0.01, 0.002, 0.0, 0.0], dtype=float)

    sol = solve_ivp(
        fun=lambda t, x: rhs(t, x),
        t_span=(0.0, 5000.0),
        y0=x0,
        method="DOP853",
        events=peak_event,
        rtol=1e-10,
        atol=1e-12,
        max_step=0.5,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    peak_times = sol.t_events[0]
    peak_states = sol.y_events[0]

    if len(peak_times) < 3:
        raise RuntimeError("Not enough peaks were detected.")

    T_guess = peak_times[-1] - peak_times[-2]
    x_guess = peak_states[-1]

    return x_guess, T_guess, peak_times, peak_states


# ---------------------------------------------------------------------
# Shooting method
# ---------------------------------------------------------------------
def flow(x0, T):
    sol = solve_ivp(
        fun=lambda t, x: rhs(t, x),
        t_span=(0.0, T),
        y0=x0,
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
        max_step=0.2,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[:, -1]


def shooting_equations(y):
    x0 = y[:4]
    T = y[4]

    if T <= 0:
        return np.ones(5) * 1.0e6

    xT = flow(x0, T)
    phase_condition = rhs(0.0, x0)[0]  # n'(0)=0 at the chosen section

    return np.concatenate([xT - x0, [phase_condition]])


def refine_periodic_orbit(x_guess, T_guess):
    y_guess = np.concatenate([x_guess, [T_guess]])

    sol = root(
        shooting_equations,
        y_guess,
        method="hybr",
        options={"xtol": 1e-10, "maxfev": 200},
    )

    if not sol.success:
        raise RuntimeError(f"Shooting failed: {sol.message}")

    x_periodic = sol.x[:4]
    T_periodic = sol.x[4]

    residual = shooting_equations(sol.x)

    return x_periodic, T_periodic, residual


# ---------------------------------------------------------------------
# Variational equation and Floquet multipliers
# ---------------------------------------------------------------------
def augmented_rhs(t, y, a=a_value):
    x = y[:4]
    Z = y[4:].reshape((4, 4))

    dx = rhs(t, x, a)
    dZ = jacobian(x, a) @ Z

    return np.concatenate([dx, dZ.reshape(16)])


def compute_monodromy(x0, T):
    Z0 = np.eye(4)
    y0 = np.concatenate([x0, Z0.reshape(16)])

    sol = solve_ivp(
        fun=lambda t, y: augmented_rhs(t, y),
        t_span=(0.0, T),
        y0=y0,
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
        max_step=0.05,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    yT = sol.y[:, -1]
    xT = yT[:4]
    M = yT[4:].reshape((4, 4))

    return xT, M


def plot_multipliers(multipliers, outdir):
    theta = np.linspace(0.0, 2.0 * np.pi, 600)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot(circle_x, circle_y, linestyle="--", linewidth=1.4, label="Unit circle")
    ax.axhline(0.0, linewidth=0.8)
    ax.axvline(0.0, linewidth=0.8)

    ax.scatter(
        np.real(multipliers),
        np.imag(multipliers),
        s=55,
        label="Floquet multipliers",
    )

    for j, lam in enumerate(multipliers, start=1):
        ax.text(np.real(lam) + 0.015, np.imag(lam) + 0.015, rf"$\mu_{j}$", fontsize=10)

    ax.set_xlabel("Real part")
    ax.set_ylabel("Imaginary part")
    ax.set_title("Floquet multipliers")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=True, loc="lower left")
    fig.tight_layout()

    pdf_path = outdir / "floquet_multipliers_hyperlogistic_en.pdf"
    png_path = outdir / "floquet_multipliers_hyperlogistic_en.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return pdf_path, png_path


def main():
    outdir = Path("figures_floquet_en")
    data_dir = Path("data_floquet_en")
    outdir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)

    x_guess, T_guess, peak_times, peak_states = find_peak_guess()
    x_periodic, T_periodic, residual = refine_periodic_orbit(x_guess, T_guess)

    xT, monodromy = compute_monodromy(x_periodic, T_periodic)
    multipliers = np.linalg.eigvals(monodromy)

    # Sort: multiplier closest to 1 first, then by modulus.
    idx_one = np.argmin(np.abs(multipliers - 1.0))
    remaining = [j for j in range(len(multipliers)) if j != idx_one]
    remaining = sorted(remaining, key=lambda j: abs(multipliers[j]), reverse=True)
    order = [idx_one] + remaining
    multipliers = multipliers[order]

    # Save table
    csv_path = data_dir / "floquet_multipliers_hyperlogistic_en.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "real", "imag", "modulus"])
        for j, lam in enumerate(multipliers, start=1):
            writer.writerow([j, np.real(lam), np.imag(lam), abs(lam)])

    # Save periodic orbit data over one period
    t_eval = np.linspace(0.0, T_periodic, 1000)
    sol_orbit = solve_ivp(
        fun=lambda t, x: rhs(t, x),
        t_span=(0.0, T_periodic),
        y0=x_periodic,
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-11,
        atol=1e-13,
        max_step=0.05,
    )

    orbit_path = data_dir / "periodic_orbit_hyperlogistic_en.csv"
    with orbit_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "n", "p", "q", "r"])
        for ti, xi in zip(sol_orbit.t, sol_orbit.y.T):
            writer.writerow([ti, xi[0], xi[1], xi[2], xi[3]])

    pdf_path, png_path = plot_multipliers(multipliers, outdir)

    print("Periodic orbit:")
    print(f"  a = {a_value:.12f}")
    print(f"  a_H = {a_H:.12f}")
    print(f"  period T = {T_periodic:.12f}")
    print("  initial point on the cycle:")
    for name, value in zip(["n0", "p0", "q0", "r0"], x_periodic):
        print(f"    {name} = {value:.12f}")

    print("Shooting residual:")
    print(f"  max |residual| = {np.max(np.abs(residual)):.3e}")
    print(f"  max |x(T)-x(0)| = {np.max(np.abs(xT - x_periodic)):.3e}")

    print("Floquet multipliers:")
    for j, lam in enumerate(multipliers, start=1):
        print(
            f"  mu_{j} = {np.real(lam): .12e}"
            f" {np.imag(lam):+.12e} i,"
            f" |mu_{j}| = {abs(lam):.12e}"
        )

    stable_transverse = all(abs(multipliers[j]) < 1.0 for j in range(1, len(multipliers)))
    print(f"Transversely stable: {stable_transverse}")

    print("Generated files:")
    print(f"  {csv_path}")
    print(f"  {orbit_path}")
    print(f"  {pdf_path}")
    print(f"  {png_path}")


if __name__ == "__main__":
    main()
