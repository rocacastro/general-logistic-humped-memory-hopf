"""
12_exponential_vs_humped_memory_en.py

Quantitative comparison between exponential fading memory and the order-two
Gamma (humped) memory in the predator-prey model used in the manuscript.

The script compares the two kernels at three levels:

1. Critical Hopf quantities for the data-informed greenhouse parameter set.
2. Critical mean memory times across representative prey-growth laws and the
   integer theta-logistic family.
3. Stable periodic orbits at the same mean memory time in the applied model.

Exponential kernel:
    G_E(s) = a_E exp(-a_E s),
    mean = 1/a_E, mode = 0.

Humped kernel:
    G_H(s) = a_H^2 s exp(-a_H s),
    mean = 2/a_H, mode = 1/a_H.

Applied dimensional parameters:
    K = 88.206925
    eps = 0.154743 day^{-1}
    alpha = 0.103094
    beta = 0.001
    gamma = 0.02 day^{-1}
    f(n) = 1 - n^2

The nonlinear comparison uses the already studied humped-memory value
    a_H = 0.18 day^{-1}.
To match the mean memory time, the exponential-memory rate is set to
    a_E = a_H / 2 = 0.09 day^{-1}.
Both kernels then have mean memory time 11.111... days.

Outputs are written to:
    data/generated/
    figures/

Required packages:
    numpy, scipy, pandas, matplotlib
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, root


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
if SCRIPT_PATH.parent.name == "scripts":
    ROOT_DIR = SCRIPT_PATH.parent.parent
else:
    ROOT_DIR = SCRIPT_PATH.parent

DATA_DIR = ROOT_DIR / "data" / "generated"
FIGURES_DIR = ROOT_DIR / "figures"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Applied parameters
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Parameters:
    K: float = 88.206925
    eps: float = 0.154743
    alpha: float = 0.103094
    beta: float = 0.001
    gamma: float = 0.02
    theta: float = 2.0


P = Parameters()
COMMON_MEAN_DAYS = 2.0 / 0.18  # Retains the humped-memory scenario in the paper.
A_HUMPED_COMPARE = 2.0 / COMMON_MEAN_DAYS
A_EXPONENTIAL_COMPARE = 1.0 / COMMON_MEAN_DAYS
THETA_VALUES = np.arange(1, 7, dtype=float)

# High-accuracy reference points from the verified version-2 computations.
# They are used only as deterministic initial guesses for the shooting method;
# the orbit and its period are recomputed and checked by the nonlinear solver.
REFERENCE_CYCLE_GUESSES = {
    "Exponential": (
        np.array([60.557036, 0.793530, 47.727199], dtype=float),
        158.243845,
    ),
    "Humped": (
        np.array([65.363027, 0.676782, 51.244452, 60.213754], dtype=float),
        163.699720,
    ),
}


# -----------------------------------------------------------------------------
# Growth law and critical quantities
# -----------------------------------------------------------------------------
def theta_growth_values(b: float, theta: float) -> Tuple[float, float, float, float]:
    """Return F=f(b), G=f'(b), H=f''(b), S=f'''(b) for f(n)=1-n^theta."""
    F = 1.0 - b**theta
    G = -theta * b ** (theta - 1.0)
    H = -theta * (theta - 1.0) * b ** (theta - 2.0)
    S = -theta * (theta - 1.0) * (theta - 2.0) * b ** (theta - 3.0)
    return F, G, H, S


def hyperlogistic_values(b: float) -> Tuple[float, float, float, float]:
    """Return F,G,H,S for f(n)=(1-n)^2."""
    return (1.0 - b) ** 2, -2.0 * (1.0 - b), 2.0, 0.0


def humped_phi(a_hat: float, K: float, beta_hat: float, b: float, F: float, G: float) -> float:
    return (
        a_hat**3
        + (2.0 * K * beta_hat * F / G - 2.0 * b * G) * a_hat**2
        + b * (b * G**2 - 2.0 * K * beta_hat * F) * a_hat
        + 0.5 * K * beta_hat * b**2 * F * G
    )


def positive_humped_root(K: float, beta_hat: float, b: float, F: float, G: float) -> float:
    upper = 1.0
    while humped_phi(upper, K, beta_hat, b, F, G) <= 0.0:
        upper *= 2.0
        if upper > 1.0e12:
            raise RuntimeError("Could not bracket the positive humped-memory Hopf root.")
    return float(brentq(lambda x: humped_phi(x, K, beta_hat, b, F, G), 0.0, upper))


def multilinear_forms(
    dimension: int,
    kappa: float,
    rho: float,
    b: float,
    G: float,
    H: float,
    S: float,
):
    """Return B and C for either reduced model; only components 1 and 2 are nonlinear."""

    def B(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        out = np.zeros(dimension, dtype=complex)
        out[0] = (2.0 * G + b * H) * x[0] * y[0] - kappa * (
            x[0] * y[1] + x[1] * y[0]
        )
        out[1] = rho * (x[1] * y[2] + x[2] * y[1])
        return out

    def C(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        out = np.zeros(dimension, dtype=complex)
        out[0] = (3.0 * H + b * S) * x[0] * y[0] * z[0]
        return out

    return B, C


def first_lyapunov_coefficient(
    A: np.ndarray,
    B: Callable[[np.ndarray, np.ndarray], np.ndarray],
    C: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    omega: float,
) -> float:
    """
    Standard Hopf normal-form coefficient with q_1=1 and <p,q>=1.

    This is the same normalization used in the humped-memory calculation in
    the manuscript, allowing a direct comparison of the two coefficients.
    """
    eigvals, eigvecs = np.linalg.eig(A)
    q = eigvecs[:, int(np.argmin(np.abs(eigvals - 1j * omega)))]
    q = q / q[0]

    eigvals_t, eigvecs_t = np.linalg.eig(A.T)
    pvec = eigvecs_t[:, int(np.argmin(np.abs(eigvals_t + 1j * omega)))]
    inner = np.vdot(pvec, q)
    pvec = pvec / np.conj(inner)

    term = (
        C(q, q, np.conj(q))
        - 2.0 * B(q, np.linalg.solve(A, B(q, np.conj(q))))
        + B(
            np.conj(q),
            np.linalg.solve(2j * omega * np.eye(A.shape[0]) - A, B(q, q)),
        )
    )
    return float(np.real(np.vdot(pvec, term)) / (2.0 * omega))


def applied_critical_metrics(p: Parameters = P) -> pd.DataFrame:
    b = p.gamma / (p.K * p.beta)
    F, G, H, S = theta_growth_values(b, p.theta)

    alpha_hat = p.alpha / p.eps
    beta_hat = p.beta / p.eps
    gamma_hat = p.gamma / p.eps
    kappa = p.K * alpha_hat
    rho = p.K * beta_hat

    # Exponential fading memory: source formula in the reparametrized system.
    a_exp_hat = b * G - p.K * beta_hat * F / G
    if a_exp_hat <= 0.0:
        raise RuntimeError("The applied exponential-memory threshold is not positive.")
    omega_exp_hat = float(np.sqrt(-a_exp_hat * b * G))
    a_exp_dim = p.eps * a_exp_hat
    omega_exp_dim = p.eps * omega_exp_hat

    A_exp = np.array(
        [
            [b * G, -kappa * b, 0.0],
            [0.0, 0.0, rho * F / kappa],
            [a_exp_hat, 0.0, -a_exp_hat],
        ],
        dtype=float,
    )
    B_exp, C_exp = multilinear_forms(3, kappa, rho, b, G, H, S)
    ell_exp = first_lyapunov_coefficient(A_exp, B_exp, C_exp, omega_exp_hat)

    # Humped memory.
    a_hump_hat = positive_humped_root(p.K, beta_hat, b, F, G)
    omega_hump_hat = float(np.sqrt(-a_hump_hat**2 * b * G / (2.0 * a_hump_hat - b * G)))
    a_hump_dim = p.eps * a_hump_hat
    omega_hump_dim = p.eps * omega_hump_hat

    A_hump = np.array(
        [
            [b * G, -kappa * b, 0.0, 0.0],
            [0.0, 0.0, rho * F / kappa, 0.0],
            [0.0, 0.0, -a_hump_hat, a_hump_hat],
            [a_hump_hat, 0.0, 0.0, -a_hump_hat],
        ],
        dtype=float,
    )
    B_hump, C_hump = multilinear_forms(4, kappa, rho, b, G, H, S)
    ell_hump = first_lyapunov_coefficient(A_hump, B_hump, C_hump, omega_hump_hat)

    rows = [
        {
            "memory_model": "Exponential",
            "reduced_dimension": 3,
            "critical_rate_day_inv": a_exp_dim,
            "critical_mode_days": 0.0,
            "critical_mean_days": 1.0 / a_exp_dim,
            "critical_frequency_day_inv": omega_exp_dim,
            "critical_period_days": 2.0 * np.pi / omega_exp_dim,
            "ell_1": ell_exp,
        },
        {
            "memory_model": "Humped",
            "reduced_dimension": 4,
            "critical_rate_day_inv": a_hump_dim,
            "critical_mode_days": 1.0 / a_hump_dim,
            "critical_mean_days": 2.0 / a_hump_dim,
            "critical_frequency_day_inv": omega_hump_dim,
            "critical_period_days": 2.0 * np.pi / omega_hump_dim,
            "ell_1": ell_hump,
        },
    ]
    return pd.DataFrame(rows)


def representative_growth_law_metrics() -> pd.DataFrame:
    """Comparison for K=2, beta=gamma=1 and b=1/2."""
    K = 2.0
    beta = 1.0
    gamma = 1.0
    b = gamma / (K * beta)

    laws = [
        ("1-n", *theta_growth_values(b, 1.0)),
        ("1-n^2", *theta_growth_values(b, 2.0)),
        ("1-n^3", *theta_growth_values(b, 3.0)),
        ("(1-n)^2", *hyperlogistic_values(b)),
    ]

    rows = []
    for label, F, G, H, S in laws:
        a_exp = b * G - K * beta * F / G
        a_hump = positive_humped_root(K, beta, b, F, G)
        rows.append(
            {
                "growth_law": label,
                "a_exp": a_exp if a_exp > 0.0 else np.nan,
                "critical_mean_exp": 1.0 / a_exp if a_exp > 0.0 else np.nan,
                "a_humped": a_hump,
                "critical_mode_humped": 1.0 / a_hump,
                "critical_mean_humped": 2.0 / a_hump,
                "exp_positive_threshold": bool(a_exp > 0.0),
            }
        )
    return pd.DataFrame(rows)


def theta_family_memory_metrics() -> pd.DataFrame:
    K = 2.0
    beta = 1.0
    gamma = 1.0
    b = gamma / (K * beta)
    rows = []
    for theta in THETA_VALUES:
        F, G, H, S = theta_growth_values(b, theta)
        a_exp = b * G - K * beta * F / G
        a_hump = positive_humped_root(K, beta, b, F, G)
        rows.append(
            {
                "theta": theta,
                "a_exp": a_exp,
                "critical_mean_exp": 1.0 / a_exp,
                "a_humped": a_hump,
                "critical_mean_humped": 2.0 / a_hump,
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Dimensional vector fields for the equal-mean nonlinear comparison
# -----------------------------------------------------------------------------
def rhs_exponential(t: float, x: np.ndarray) -> np.ndarray:
    N, predator, Q = x
    return np.array(
        [
            P.eps * N * (1.0 - (N / P.K) ** 2) - P.alpha * N * predator,
            -P.gamma * predator + P.beta * predator * Q,
            A_EXPONENTIAL_COMPARE * (N - Q),
        ],
        dtype=float,
    )


def jacobian_exponential(x: np.ndarray) -> np.ndarray:
    N, predator, Q = x
    a = A_EXPONENTIAL_COMPARE
    return np.array(
        [
            [P.eps * (1.0 - 3.0 * (N / P.K) ** 2) - P.alpha * predator, -P.alpha * N, 0.0],
            [0.0, -P.gamma + P.beta * Q, P.beta * predator],
            [a, 0.0, -a],
        ],
        dtype=float,
    )


def rhs_humped(t: float, x: np.ndarray) -> np.ndarray:
    N, predator, Q, R = x
    a = A_HUMPED_COMPARE
    return np.array(
        [
            P.eps * N * (1.0 - (N / P.K) ** 2) - P.alpha * N * predator,
            -P.gamma * predator + P.beta * predator * Q,
            a * (R - Q),
            a * (N - R),
        ],
        dtype=float,
    )


def jacobian_humped(x: np.ndarray) -> np.ndarray:
    N, predator, Q, R = x
    a = A_HUMPED_COMPARE
    return np.array(
        [
            [P.eps * (1.0 - 3.0 * (N / P.K) ** 2) - P.alpha * predator, -P.alpha * N, 0.0, 0.0],
            [0.0, -P.gamma + P.beta * Q, P.beta * predator, 0.0],
            [0.0, 0.0, -a, a],
            [a, 0.0, 0.0, -a],
        ],
        dtype=float,
    )


def make_peak_event(rhs: Callable[[float, np.ndarray], np.ndarray]):
    def event(t: float, x: np.ndarray) -> float:
        if t < 1.0e-6:
            return 1.0
        return float(rhs(t, x)[0])

    event.direction = -1.0
    event.terminal = False
    return event


def peak_guess(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    initial: np.ndarray,
    final_time: float = 4000.0,
) -> Tuple[np.ndarray, float]:
    sol = solve_ivp(
        rhs,
        (0.0, final_time),
        initial,
        method="DOP853",
        events=make_peak_event(rhs),
        rtol=1.0e-10,
        atol=1.0e-12,
        max_step=1.0,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    times = sol.t_events[0]
    states = sol.y_events[0]
    if len(times) < 4:
        raise RuntimeError("Not enough prey maxima were detected.")
    return states[-1], float(times[-1] - times[-2])


def refine_periodic_orbit(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    x_guess: np.ndarray,
    period_guess: float,
) -> Tuple[np.ndarray, float, float]:
    dimension = len(x_guess)

    def flow(x0: np.ndarray, period: float) -> np.ndarray:
        sol = solve_ivp(
            rhs,
            (0.0, period),
            x0,
            method="DOP853",
            rtol=1.0e-11,
            atol=1.0e-13,
            max_step=0.3,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol.y[:, -1]

    def equations(y: np.ndarray) -> np.ndarray:
        x0 = y[:dimension]
        period = y[dimension]
        if period <= 0.0:
            return np.full(dimension + 1, 1.0e6)
        residual = flow(x0, period) - x0
        phase = rhs(0.0, x0)[0]  # N'(0)=0 at a prey maximum.
        return np.concatenate([residual, [phase]])

    solution = root(
        equations,
        np.concatenate([x_guess, [period_guess]]),
        method="hybr",
        options={"xtol": 1.0e-11, "maxfev": 500},
    )
    if not solution.success:
        raise RuntimeError(f"Shooting method failed: {solution.message}")
    residual = float(np.max(np.abs(equations(solution.x))))
    return solution.x[:dimension], float(solution.x[dimension]), residual


def monodromy_and_orbit(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    jacobian: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    period: float,
    samples: int = 3000,
):
    dimension = len(x0)

    def augmented_rhs(t: float, y: np.ndarray) -> np.ndarray:
        x = y[:dimension]
        Z = y[dimension:].reshape((dimension, dimension))
        return np.concatenate([rhs(t, x), (jacobian(x) @ Z).reshape(-1)])

    y0 = np.concatenate([x0, np.eye(dimension).reshape(-1)])
    sol_aug = solve_ivp(
        augmented_rhs,
        (0.0, period),
        y0,
        method="DOP853",
        rtol=1.0e-11,
        atol=1.0e-13,
        max_step=0.15,
    )
    if not sol_aug.success:
        raise RuntimeError(sol_aug.message)
    monodromy = sol_aug.y[dimension:, -1].reshape((dimension, dimension))
    multipliers = np.linalg.eigvals(monodromy)

    t_eval = np.linspace(0.0, period, samples)
    sol_orbit = solve_ivp(
        rhs,
        (0.0, period),
        x0,
        method="DOP853",
        t_eval=t_eval,
        rtol=1.0e-11,
        atol=1.0e-13,
        max_step=0.15,
    )
    if not sol_orbit.success:
        raise RuntimeError(sol_orbit.message)

    return multipliers, sol_orbit.t, sol_orbit.y.T


def order_multipliers(multipliers: np.ndarray) -> np.ndarray:
    idx_one = int(np.argmin(np.abs(multipliers - 1.0)))
    remaining = [j for j in range(len(multipliers)) if j != idx_one]
    remaining = sorted(remaining, key=lambda j: abs(multipliers[j]), reverse=True)
    return multipliers[[idx_one] + remaining]


def cycle_metrics(label: str, rate: float, period: float, orbit: np.ndarray, multipliers: np.ndarray, residual: float) -> Dict[str, float | str]:
    nontrivial = order_multipliers(multipliers)[1:]
    N = orbit[:, 0]
    predator = orbit[:, 1]
    return {
        "memory_model": label,
        "rate_day_inv": rate,
        "mean_memory_days": COMMON_MEAN_DAYS,
        "period_days": period,
        "N_min": float(np.min(N)),
        "N_max": float(np.max(N)),
        "N_amplitude": float((np.max(N) - np.min(N)) / 2.0),
        "P_min": float(np.min(predator)),
        "P_max": float(np.max(predator)),
        "P_amplitude": float((np.max(predator) - np.min(predator)) / 2.0),
        "dominant_nontrivial_multiplier": float(np.max(np.abs(nontrivial))),
        "dominant_floquet_exponent_day_inv": float(np.log(np.max(np.abs(nontrivial))) / period),
        "shooting_residual": residual,
    }


def compute_equal_mean_cycles():
    configurations = [
        (
            "Exponential",
            A_EXPONENTIAL_COMPARE,
            rhs_exponential,
            jacobian_exponential,
            np.array([25.0, 1.0, 20.0]),
        ),
        (
            "Humped",
            A_HUMPED_COMPARE,
            rhs_humped,
            jacobian_humped,
            np.array([25.0, 1.0, 20.0, 20.0]),
        ),
    ]

    metrics_rows = []
    multiplier_rows = []
    orbit_data = {}

    for label, rate, rhs, jac, initial in configurations:
        # Start from a verified reference point to avoid a very long transient.
        # If a future parameter change invalidates this guess, fall back to the
        # transient-based peak detector.
        x_guess, period_guess = REFERENCE_CYCLE_GUESSES[label]
        try:
            x0, period, residual = refine_periodic_orbit(rhs, x_guess, period_guess)
        except RuntimeError:
            x_guess, period_guess = peak_guess(rhs, initial)
            x0, period, residual = refine_periodic_orbit(rhs, x_guess, period_guess)
        multipliers, times, orbit = monodromy_and_orbit(rhs, jac, x0, period)
        multipliers = order_multipliers(multipliers)

        metrics_rows.append(cycle_metrics(label, rate, period, orbit, multipliers, residual))
        for index, multiplier in enumerate(multipliers, start=1):
            multiplier_rows.append(
                {
                    "memory_model": label,
                    "index": index,
                    "real": float(np.real(multiplier)),
                    "imag": float(np.imag(multiplier)),
                    "modulus": float(abs(multiplier)),
                }
            )

        phase = times / period
        columns = ["N", "P", "Q"] if label == "Exponential" else ["N", "P", "Q", "R"]
        frame = pd.DataFrame(orbit, columns=columns)
        frame.insert(0, "time_days", times)
        frame.insert(1, "phase", phase)
        orbit_data[label] = frame

    return pd.DataFrame(metrics_rows), pd.DataFrame(multiplier_rows), orbit_data


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_equal_mean_kernels() -> None:
    s = np.linspace(0.0, 4.0 * COMMON_MEAN_DAYS, 1200)
    kernel_exp = A_EXPONENTIAL_COMPARE * np.exp(-A_EXPONENTIAL_COMPARE * s)
    kernel_hump = A_HUMPED_COMPARE**2 * s * np.exp(-A_HUMPED_COMPARE * s)

    pd.DataFrame(
        {
            "time_days": s,
            "exponential_kernel": kernel_exp,
            "humped_kernel": kernel_hump,
        }
    ).to_csv(DATA_DIR / "memory_kernels_equal_mean.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(s, kernel_exp, linewidth=2.0, label="Exponential memory")
    ax.plot(s, kernel_hump, linewidth=2.0, linestyle="--", label="Humped memory")
    ax.axvline(COMMON_MEAN_DAYS, linewidth=1.0, linestyle=":", label="Common mean")
    ax.axvline(1.0 / A_HUMPED_COMPARE, linewidth=1.0, linestyle="-.", label="Humped mode")
    ax.set_xlabel("Past time $s$ (days)")
    ax.set_ylabel("Memory weight")
    ax.set_title("Memory kernels with the same mean response time")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "memory_kernels_equal_mean_en.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "memory_kernels_equal_mean_en.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_theta_critical_means(theta_data: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    ax.plot(theta_data["theta"], theta_data["critical_mean_exp"], marker="o", label="Exponential memory")
    ax.plot(theta_data["theta"], theta_data["critical_mean_humped"], marker="s", linestyle="--", label="Humped memory")
    ax.set_xlabel(r"Exponent $\theta$ in $f_\theta(n)=1-n^\theta$")
    ax.set_ylabel("Critical mean memory time")
    ax.set_title("Kernel-dependent critical mean memory time")
    ax.set_xticks(theta_data["theta"])
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "memory_critical_mean_theta_en.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "memory_critical_mean_theta_en.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_phase_comparison(orbits: Dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for label, frame in orbits.items():
        ax.plot(frame["N"], frame["P"], linewidth=2.0, label=label)
    ax.scatter([P.gamma / P.beta], [P.eps * (1.0 - (P.gamma / (P.K * P.beta)) ** 2) / P.alpha], s=35, label="Coexistence equilibrium")
    ax.set_xlabel("Prey density $N$")
    ax.set_ylabel("Predator density $P$")
    ax.set_title("Periodic orbits at equal mean memory time")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "memory_cycle_phase_comparison_en.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "memory_cycle_phase_comparison_en.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_component_comparison(orbits: Dict[str, pd.DataFrame], component: str, ylabel: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for label, frame in orbits.items():
        ax.plot(frame["phase"], frame[component], linewidth=2.0, label=label)
    ax.set_xlabel("Normalized time $t/T$")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} over one periodic cycle")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{filename}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    critical = applied_critical_metrics(P)
    representative = representative_growth_law_metrics()
    theta_data = theta_family_memory_metrics()
    cycle_metrics_df, multipliers_df, orbits = compute_equal_mean_cycles()

    critical.to_csv(DATA_DIR / "memory_model_applied_critical_metrics.csv", index=False)
    representative.to_csv(DATA_DIR / "memory_model_representative_growth_laws.csv", index=False)
    theta_data.to_csv(DATA_DIR / "memory_model_theta_family.csv", index=False)
    cycle_metrics_df.to_csv(DATA_DIR / "memory_model_equal_mean_cycle_metrics.csv", index=False)
    multipliers_df.to_csv(DATA_DIR / "memory_model_equal_mean_floquet.csv", index=False)
    for label, frame in orbits.items():
        stem = label.lower().replace(" ", "_")
        frame.to_csv(DATA_DIR / f"memory_model_{stem}_periodic_orbit.csv", index=False)

    plot_equal_mean_kernels()
    plot_theta_critical_means(theta_data)
    plot_phase_comparison(orbits)
    plot_component_comparison(orbits, "N", "Prey density $N$", "memory_cycle_prey_comparison_en")
    plot_component_comparison(orbits, "P", "Predator density $P$", "memory_cycle_predator_comparison_en")

    print("Applied critical metrics")
    print(critical.to_string(index=False))
    print("\nRepresentative growth laws")
    print(representative.to_string(index=False))
    print("\nEqual-mean periodic comparison")
    print(cycle_metrics_df.to_string(index=False))
    print("\nFloquet multipliers")
    print(multipliers_df.to_string(index=False))
    print(f"\nCommon mean memory time: {COMMON_MEAN_DAYS:.12f} days")
    print(f"Exponential rate: {A_EXPONENTIAL_COMPARE:.12f} day^(-1)")
    print(f"Humped rate: {A_HUMPED_COMPARE:.12f} day^(-1)")
    print(f"Generated data in: {DATA_DIR}")
    print(f"Generated figures in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
