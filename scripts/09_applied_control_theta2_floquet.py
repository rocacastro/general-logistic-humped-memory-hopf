"""
09_applied_control_theta2_floquet.py

Computes Floquet multipliers for the periodic orbit in the applied greenhouse
biological-control scenario with nonlinear prey growth f(n) = 1 - n^2.

System:
    N' = eps N [1 - (N/K)^2] - alpha N P,
    P' = -gamma P + beta P Q,
    Q' = a(R-Q),
    R' = a(N-R).

The script:
    1. Integrates until the trajectory approaches the periodic orbit.
    2. Detects local maxima of N(t).
    3. Refines one periodic orbit by a shooting method.
    4. Integrates the variational equation over one period.
    5. Computes the Floquet multipliers.

Outputs:
    data/generated/applied_theta2_periodic_orbit.csv
    data/generated/applied_theta2_floquet_multipliers.csv
    figures/floquet_multipliers_applied_theta2_en.pdf
    figures/floquet_multipliers_applied_theta2_en.png
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root


# ---------------------------------------------------------------------
# Output paths are relative to the location of this script.
# ---------------------------------------------------------------------
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

ROOT_DIR = BASE_DIR.parent if BASE_DIR.name == "scripts" else BASE_DIR
DATA_DIR = ROOT_DIR / "data" / "generated"
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
A = 0.18
A_H = 0.353291

N_STAR = GAMMA / BETA
P_STAR = EPS * (1.0 - (N_STAR / K) ** 2) / ALPHA
SIGMA = np.array([N_STAR, P_STAR, N_STAR, N_STAR], dtype=float)


# ---------------------------------------------------------------------
# Vector field and Jacobian
# ---------------------------------------------------------------------
def rhs(t, x):
    N, P, Q, R = x
    return np.array(
        [
            EPS * N * (1.0 - (N / K) ** 2) - ALPHA * N * P,
            -GAMMA * P + BETA * P * Q,
            A * (R - Q),
            A * (N - R),
        ],
        dtype=float,
    )


def jacobian(x):
    N, P, Q, R = x
    return np.array(
        [
            [EPS * (1.0 - 3.0 * (N / K) ** 2) - ALPHA * P, -ALPHA * N, 0.0, 0.0],
            [0.0, -GAMMA + BETA * Q, BETA * P, 0.0],
            [0.0, 0.0, -A, A],
            [A, 0.0, 0.0, -A],
        ],
        dtype=float,
    )


# ---------------------------------------------------------------------
# Peak event: local maxima of N(t)
# ---------------------------------------------------------------------
def peak_event(t, x):
    if t < 1.0e-6:
        return 1.0
    return rhs(t, x)[0]


peak_event.direction = -1.0
peak_event.terminal = False


# ---------------------------------------------------------------------
# Periodic orbit by shooting
# ---------------------------------------------------------------------
def find_peak_guess():
    x0 = np.array([25.0, 1.0, 20.0, 20.0], dtype=float)
    sol = solve_ivp(
        rhs,
        (0.0, 2000.0),
        x0,
        method="DOP853",
        events=peak_event,
        rtol=1e-10,
        atol=1e-12,
        max_step=1.0,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    times = sol.t_events[0]
    states = sol.y_events[0]
    if len(times) < 3:
        raise RuntimeError("Not enough peaks detected. Increase integration time.")

    T_guess = times[-1] - times[-2]
    x_guess = states[-1]
    return x_guess, T_guess


def flow(x0, T):
    sol = solve_ivp(
        rhs,
        (0.0, T),
        x0,
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
        max_step=0.5,
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
    phase_condition = rhs(0.0, x0)[0]  # choose N'(0)=0 at a maximum of N
    return np.concatenate([xT - x0, [phase_condition]])


def refine_orbit(x_guess, T_guess):
    y_guess = np.concatenate([x_guess, [T_guess]])
    sol = root(shooting_equations, y_guess, method="hybr", options={"xtol": 1e-10, "maxfev": 250})
    if not sol.success:
        raise RuntimeError(f"Shooting failed: {sol.message}")
    x0 = sol.x[:4]
    T = sol.x[4]
    residual = shooting_equations(sol.x)
    return x0, T, residual


# ---------------------------------------------------------------------
# Variational equation and monodromy matrix
# ---------------------------------------------------------------------
def augmented_rhs(t, y):
    x = y[:4]
    Z = y[4:].reshape((4, 4))
    dx = rhs(t, x)
    dZ = jacobian(x) @ Z
    return np.concatenate([dx, dZ.reshape(16)])


def compute_monodromy(x0, T):
    y0 = np.concatenate([x0, np.eye(4).reshape(16)])
    sol = solve_ivp(
        augmented_rhs,
        (0.0, T),
        y0,
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
        max_step=0.2,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    xT = sol.y[:4, -1]
    M = sol.y[4:, -1].reshape((4, 4))
    return xT, M


# ---------------------------------------------------------------------
# Save and plot
# ---------------------------------------------------------------------
def save_periodic_orbit(x0, T):
    t_eval = np.linspace(0.0, T, 1200)
    sol = solve_ivp(
        rhs,
        (0.0, T),
        x0,
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-11,
        atol=1e-13,
        max_step=0.2,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    path = DATA_DIR / "applied_theta2_periodic_orbit.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "N", "P", "Q", "R"])
        for ti, xi in zip(sol.t, sol.y.T):
            writer.writerow([ti, xi[0], xi[1], xi[2], xi[3]])
    return path


def save_multipliers(multipliers):
    path = DATA_DIR / "applied_theta2_floquet_multipliers.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "real", "imag", "modulus"])
        for j, mu in enumerate(multipliers, start=1):
            writer.writerow([j, np.real(mu), np.imag(mu), abs(mu)])
    return path


def plot_multipliers(multipliers):
    theta = np.linspace(0.0, 2.0 * np.pi, 600)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot(np.cos(theta), np.sin(theta), linestyle="--", linewidth=1.3, label="Unit circle")
    ax.axhline(0.0, linewidth=0.8)
    ax.axvline(0.0, linewidth=0.8)
    ax.scatter(np.real(multipliers), np.imag(multipliers), s=55, label="Floquet multipliers")

    for j, mu in enumerate(multipliers, start=1):
        ax.text(np.real(mu) + 0.015, np.imag(mu) + 0.015, rf"$\mu_{j}$", fontsize=10)

    ax.set_xlabel("Real part")
    ax.set_ylabel("Imaginary part")
    ax.set_title("Floquet multipliers: applied scenario")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=True, loc="lower left")
    fig.tight_layout()

    pdf_path = FIG_DIR / "floquet_multipliers_applied_theta2_en.pdf"
    png_path = FIG_DIR / "floquet_multipliers_applied_theta2_en.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def main():
    x_guess, T_guess = find_peak_guess()
    x0, T, residual = refine_orbit(x_guess, T_guess)
    xT, M = compute_monodromy(x0, T)
    multipliers = np.linalg.eigvals(M)

    # Put the trivial multiplier closest to 1 first.
    idx_one = int(np.argmin(np.abs(multipliers - 1.0)))
    remaining = [j for j in range(len(multipliers)) if j != idx_one]
    remaining = sorted(remaining, key=lambda j: abs(multipliers[j]), reverse=True)
    order = [idx_one] + remaining
    multipliers = multipliers[order]

    orbit_path = save_periodic_orbit(x0, T)
    mult_path = save_multipliers(multipliers)
    pdf_path, png_path = plot_multipliers(multipliers)

    print("Applied scenario Floquet computation")
    print(f"  a = {A:.12f} day^(-1)")
    print(f"  a_H = {A_H:.12f} day^(-1)")
    print(f"  period T = {T:.12f} days")
    print("  initial point on the periodic orbit:")
    for name, value in zip(["N0", "P0", "Q0", "R0"], x0):
        print(f"    {name} = {value:.12f}")
    print(f"  max shooting residual = {np.max(np.abs(residual)):.3e}")
    print(f"  max |x(T)-x(0)| = {np.max(np.abs(xT - x0)):.3e}")
    print("  Floquet multipliers:")
    for j, mu in enumerate(multipliers, start=1):
        print(f"    mu_{j} = {np.real(mu): .12e} {np.imag(mu):+.12e} i, |mu_{j}| = {abs(mu):.12e}")

    print("Generated files:")
    print(f"  {orbit_path}")
    print(f"  {mult_path}")
    print(f"  {pdf_path}")
    print(f"  {png_path}")


if __name__ == "__main__":
    main()
