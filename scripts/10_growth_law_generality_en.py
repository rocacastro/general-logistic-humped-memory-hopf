"""
10_growth_law_generality_en.py

Numerical generality analysis for the humped-memory predator-prey model.

The script performs three complementary computations:

1. It evaluates the analytical Hopf quantities for four representative
   prey-growth laws, including one law outside the theta-logistic family.
2. It evaluates the integer theta-logistic family f_theta(n)=1-n^theta
   for theta=1,...,6. Integer values are used so that every function
   satisfies the standing assumption f in C^4([0,1]).
3. It computes stable periodic-orbit metrics for four representative laws
   at the same relative distance from their own Hopf threshold,
   a = 0.94 a_H.

Outputs are written to:
    data/generated/
    figures/

Required packages:
    numpy, scipy, pandas, matplotlib
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


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
# Model parameters
# -----------------------------------------------------------------------------
K = 2.0
ALPHA = 1.0
BETA = 1.0
GAMMA = 1.0
B = GAMMA / (K * BETA)
RELATIVE_A = 0.94
T_FINAL = 5000.0
TRANSIENT_START = 4000.0


@dataclass(frozen=True)
class GrowthLaw:
    key: str
    label: str
    f: Callable[[float], float]
    fp: Callable[[float], float]
    fpp: Callable[[float], float]
    fppp: Callable[[float], float]

    def local_values(self, b: float) -> tuple[float, float, float, float]:
        return self.f(b), self.fp(b), self.fpp(b), self.fppp(b)


def theta_law(theta: int) -> GrowthLaw:
    """Integer theta-logistic law f(n)=1-n^theta."""
    return GrowthLaw(
        key=f"theta_{theta}",
        label=rf"$1-n^{theta}$" if theta != 1 else r"$1-n$",
        f=lambda n, th=theta: 1.0 - n**th,
        fp=lambda n, th=theta: -th * n ** (th - 1),
        fpp=lambda n, th=theta: (
            -th * (th - 1) * n ** (th - 2) if th >= 2 else 0.0
        ),
        fppp=lambda n, th=theta: (
            -th * (th - 1) * (th - 2) * n ** (th - 3) if th >= 3 else 0.0
        ),
    )


def hyperlogistic_law() -> GrowthLaw:
    return GrowthLaw(
        key="hyperlogistic",
        label=r"$(1-n)^2$",
        f=lambda n: (1.0 - n) ** 2,
        fp=lambda n: -2.0 * (1.0 - n),
        fpp=lambda n: 2.0,
        fppp=lambda n: 0.0,
    )


# -----------------------------------------------------------------------------
# Analytical Hopf quantities
# -----------------------------------------------------------------------------
def phi(a: float, F: float, G: float) -> float:
    return (
        a**3
        + (2.0 * K * BETA * F / G - 2.0 * B * G) * a**2
        + B * (B * G**2 - 2.0 * K * BETA * F) * a
        + 0.5 * K * BETA * B**2 * F * G
    )


def find_a_h(F: float, G: float) -> float:
    if F <= 0.0:
        raise ValueError("The calculation requires F=f(b)>0.")
    if G >= 0.0:
        raise ValueError("The calculation requires G=f'(b)<0.")

    upper = 1.0
    while phi(upper, F, G) <= 0.0:
        upper *= 2.0
        if upper > 1.0e10:
            raise RuntimeError("Unable to bracket the positive root of Phi.")
    return float(brentq(lambda value: phi(value, F, G), 0.0, upper))


def omega_0(a_h: float, G: float) -> float:
    value = -a_h**2 * B * G / (2.0 * a_h - B * G)
    if value <= 0.0:
        raise ValueError("omega_0^2 must be positive.")
    return float(np.sqrt(value))


def g4_and_l1(
    F: float,
    G: float,
    H: float,
    S: float,
    a_h: float,
) -> tuple[float, float]:
    """Compact formula used in the manuscript."""
    a = a_h
    omega = omega_0(a_h, G)
    kappa = K * ALPHA
    rho = K * BETA
    imaginary = 1j

    m = (
        a * B * G
        + 3.0 * imaginary * B * G * omega
        - 2.0 * imaginary * a * omega
        + 4.0 * omega**2
    ) / (omega * (omega - imaginary * a))

    u1 = B * H + 2.0 * imaginary * omega / B
    u2 = (
        2.0
        * imaginary
        * omega
        * (B * G - imaginary * omega) ** 2
        / (F * kappa * B**2)
    )

    denominator_20 = (
        2.0 * imaginary * omega
        - B * G
        + (B * rho * F * a**2)
        / (2.0 * imaginary * omega * (a + 2.0 * imaginary * omega) ** 2)
    )

    h1 = (u1 - (kappa * B) / (2.0 * imaginary * omega) * u2) / denominator_20
    h3 = a**2 * h1 / (a + 2.0 * imaginary * omega) ** 2
    h2 = (u2 + (rho * F / kappa) * h3) / (2.0 * imaginary * omega)

    q3 = imaginary * omega * (B * G - imaginary * omega) / (rho * F * B)
    qbar2 = (B * G + imaginary * omega) / (kappa * B)
    qbar3 = -imaginary * omega * (B * G + imaginary * omega) / (rho * F * B)

    d1 = H + B * S + (G + B * H - imaginary * omega / B) * h1 - kappa * h2
    d2 = (2.0 * rho * H / kappa) * q3 + rho * (qbar2 * h3 + qbar3 * h2)

    g4 = float(np.real((d1 + imaginary * kappa * B * d2 / omega) / m))
    ell1 = g4 / (2.0 * omega)
    return g4, ell1


def analytical_row(law: GrowthLaw) -> dict[str, float | str]:
    F, G, H, S = law.local_values(B)
    a_h = find_a_h(F, G)
    omega = omega_0(a_h, G)
    g4, ell1 = g4_and_l1(F, G, H, S, a_h)
    return {
        "key": law.key,
        "label": law.label,
        "b": B,
        "F": F,
        "G": G,
        "H": H,
        "S": S,
        "a_H": a_h,
        "tau_H": 1.0 / a_h,
        "omega_0": omega,
        "T_0": 2.0 * np.pi / omega,
        "G_4": g4,
        "ell_1": ell1,
        "criticality": "supercritical" if ell1 < 0.0 else "subcritical",
    }


# -----------------------------------------------------------------------------
# Periodic-orbit metrics at a=0.94 a_H
# -----------------------------------------------------------------------------
def model_rhs(law: GrowthLaw, a_value: float) -> Callable[[float, np.ndarray], np.ndarray]:
    def rhs(_t: float, state: np.ndarray) -> np.ndarray:
        n, p, q, r = state
        return np.array(
            [
                n * law.f(n) - K * ALPHA * n * p,
                -GAMMA * p + K * BETA * p * q,
                a_value * (r - q),
                a_value * (n - r),
            ],
            dtype=float,
        )

    return rhs


def periodic_metrics(law: GrowthLaw, analytical: dict[str, float | str]) -> tuple[dict, pd.DataFrame]:
    F = float(analytical["F"])
    a_h = float(analytical["a_H"])
    a_value = RELATIVE_A * a_h
    equilibrium = np.array([B, F / (K * ALPHA), B, B], dtype=float)
    initial_state = equilibrium + np.array([0.02, 0.005, 0.0, 0.0], dtype=float)
    rhs = model_rhs(law, a_value)

    def maximum_event(t: float, state: np.ndarray) -> float:
        return float(rhs(t, state)[0])

    maximum_event.direction = -1
    maximum_event.terminal = False

    def minimum_event(t: float, state: np.ndarray) -> float:
        return float(rhs(t, state)[0])

    minimum_event.direction = 1
    minimum_event.terminal = False

    solution = solve_ivp(
        rhs,
        (0.0, T_FINAL),
        initial_state,
        method="DOP853",
        rtol=1.0e-10,
        atol=1.0e-12,
        max_step=0.25,
        dense_output=True,
        events=(maximum_event, minimum_event),
    )
    if not solution.success:
        raise RuntimeError(f"Integration failed for {law.key}: {solution.message}")

    maximum_times = solution.t_events[0]
    maximum_states = solution.y_events[0]
    minimum_times = solution.t_events[1]
    minimum_states = solution.y_events[1]

    maximum_mask = maximum_times >= TRANSIENT_START
    minimum_mask = minimum_times >= TRANSIENT_START
    maximum_times = maximum_times[maximum_mask]
    maximum_states = maximum_states[maximum_mask]
    minimum_times = minimum_times[minimum_mask]
    minimum_states = minimum_states[minimum_mask]

    if len(maximum_times) < 22 or len(minimum_times) < 20:
        raise RuntimeError(f"Too few asymptotic extrema for {law.key}.")

    last_maxima = maximum_states[-20:, 0]
    last_minima = minimum_states[-20:, 0]
    last_periods = np.diff(maximum_times[-21:])

    n_max = float(np.mean(last_maxima))
    n_min = float(np.mean(last_minima))
    numerical_period = float(np.mean(last_periods))
    amplitude = 0.5 * (n_max - n_min)

    # Two complete asymptotic cycles for a common normalized-time comparison.
    t_start = float(maximum_times[-3])
    t_end = float(maximum_times[-1])
    two_cycle_period = 0.5 * (t_end - t_start)
    sample_times = np.linspace(t_start, t_end, 1201)
    sample_states = solution.sol(sample_times).T
    normalized_time = (sample_times - t_start) / two_cycle_period

    cycle_frame = pd.DataFrame(
        {
            "key": law.key,
            "label": law.label,
            "normalized_time": normalized_time,
            "n": sample_states[:, 0],
            "p": sample_states[:, 1],
            "q": sample_states[:, 2],
            "r": sample_states[:, 3],
        }
    )

    metrics = {
        "key": law.key,
        "label": law.label,
        "a_over_a_H": RELATIVE_A,
        "a": a_value,
        "n_min": n_min,
        "n_max": n_max,
        "prey_amplitude": amplitude,
        "T_num": numerical_period,
        "T_num_over_T_0": numerical_period / float(analytical["T_0"]),
    }
    return metrics, cycle_frame


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------
def write_analytical_latex_table(frame: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"$f(n)$ & $a_{\mathrm{H}}$ & $1/a_{\mathrm{H}}$ & $\omega_0$ & $T_0$ & $G_4$ & $\ell_1(0)$ \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"{row['label']} & {row['a_H']:.6f} & {row['tau_H']:.6f} & "
            f"{row['omega_0']:.6f} & {row['T_0']:.6f} & "
            f"{row['G_4']:.6f} & {row['ell_1']:.6f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_periodic_latex_table(frame: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"$f(n)$ & $n_{\min}$ & $n_{\max}$ & $A_n$ & $T_{\mathrm{num}}$ \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"{row['label']} & {row['n_min']:.6f} & {row['n_max']:.6f} & "
            f"{row['prey_amplitude']:.6f} & {row['T_num']:.6f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_threshold_figure(theta_frame: pd.DataFrame) -> None:
    fig, left_axis = plt.subplots(figsize=(6.8, 4.6))
    right_axis = left_axis.twinx()

    line_1 = left_axis.plot(
        theta_frame["theta"],
        theta_frame["a_H"],
        marker="o",
        linewidth=1.8,
        label=r"Hopf threshold $a_{\mathrm{H}}$",
    )
    line_2 = right_axis.plot(
        theta_frame["theta"],
        theta_frame["tau_H"],
        marker="s",
        linestyle="--",
        linewidth=1.8,
        label=r"Critical memory time $1/a_{\mathrm{H}}$",
    )

    left_axis.set_xlabel(r"Integer exponent $\theta$")
    left_axis.set_ylabel(r"Hopf threshold $a_{\mathrm{H}}$")
    right_axis.set_ylabel(r"Critical memory time $1/a_{\mathrm{H}}$")
    left_axis.set_xticks(theta_frame["theta"])
    left_axis.set_title(r"Threshold variation in $f_\theta(n)=1-n^\theta$")

    handles = line_1 + line_2
    left_axis.legend(handles, [item.get_label() for item in handles], loc="best")
    fig.tight_layout()
    save_figure(fig, "theta_logistic_thresholds_en")


def make_hopf_metrics_figure(theta_frame: pd.DataFrame) -> None:
    fig, left_axis = plt.subplots(figsize=(6.8, 4.6))
    right_axis = left_axis.twinx()

    line_1 = left_axis.plot(
        theta_frame["theta"],
        theta_frame["omega_0"],
        marker="o",
        linewidth=1.8,
        label=r"Critical frequency $\omega_0$",
    )
    line_2 = right_axis.plot(
        theta_frame["theta"],
        theta_frame["ell_1"],
        marker="s",
        linestyle="--",
        linewidth=1.8,
        label=r"First Lyapunov coefficient $\ell_1(0)$",
    )

    right_axis.axhline(0.0, linewidth=0.8, linestyle=":")
    left_axis.set_xlabel(r"Integer exponent $\theta$")
    left_axis.set_ylabel(r"Critical frequency $\omega_0$")
    right_axis.set_ylabel(r"First Lyapunov coefficient $\ell_1(0)$")
    left_axis.set_xticks(theta_frame["theta"])
    left_axis.set_title(r"Local Hopf quantities in $f_\theta(n)=1-n^\theta$")

    handles = line_1 + line_2
    left_axis.legend(handles, [item.get_label() for item in handles], loc="best")
    fig.tight_layout()
    save_figure(fig, "theta_logistic_hopf_metrics_en")


def make_periodic_comparison_figure(cycles: pd.DataFrame, laws: Iterable[GrowthLaw]) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for law in laws:
        subset = cycles[cycles["key"] == law.key]
        axis.plot(
            subset["normalized_time"],
            subset["n"],
            linewidth=1.7,
            label=law.label,
        )

    axis.axhline(B, linestyle=":", linewidth=1.0, label=r"Coexistence density $b$")
    axis.set_xlabel(r"Normalized time $t/T_{\mathrm{num}}$")
    axis.set_ylabel(r"Prey density $n(t)$")
    axis.set_title(r"Stable prey oscillations at $a=0.94a_{\mathrm{H}}$")
    axis.set_xlim(0.0, 2.0)
    axis.legend(ncol=2, frameon=True)
    fig.tight_layout()
    save_figure(fig, "growth_law_periodic_comparison_en")


def main() -> None:
    representative_laws = [
        theta_law(1),
        theta_law(2),
        theta_law(3),
        hyperlogistic_law(),
    ]
    theta_laws = [theta_law(theta) for theta in range(1, 7)]

    analytical_representative = pd.DataFrame(
        [analytical_row(law) for law in representative_laws]
    )
    analytical_representative.to_csv(
        DATA_DIR / "growth_law_analytical_metrics.csv", index=False
    )
    write_analytical_latex_table(
        analytical_representative,
        DATA_DIR / "growth_law_analytical_table.tex",
    )

    theta_rows = []
    for theta, law in enumerate(theta_laws, start=1):
        row = analytical_row(law)
        row["theta"] = theta
        theta_rows.append(row)
    theta_frame = pd.DataFrame(theta_rows)
    theta_frame.to_csv(DATA_DIR / "theta_logistic_sweep.csv", index=False)

    periodic_rows: list[dict] = []
    cycle_frames: list[pd.DataFrame] = []
    analytical_by_key = {
        row["key"]: row for row in analytical_representative.to_dict(orient="records")
    }
    for law in representative_laws:
        metrics, cycle_frame = periodic_metrics(law, analytical_by_key[law.key])
        periodic_rows.append(metrics)
        cycle_frames.append(cycle_frame)

    periodic_frame = pd.DataFrame(periodic_rows)
    cycles_frame = pd.concat(cycle_frames, ignore_index=True)
    periodic_frame.to_csv(DATA_DIR / "growth_law_periodic_metrics.csv", index=False)
    cycles_frame.to_csv(DATA_DIR / "growth_law_normalized_cycles.csv", index=False)
    write_periodic_latex_table(
        periodic_frame,
        DATA_DIR / "growth_law_periodic_table.tex",
    )

    make_threshold_figure(theta_frame)
    make_hopf_metrics_figure(theta_frame)
    make_periodic_comparison_figure(cycles_frame, representative_laws)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.precision", 8)
    print("\nRepresentative analytical comparison")
    print(
        analytical_representative[
            ["label", "a_H", "tau_H", "omega_0", "T_0", "G_4", "ell_1"]
        ].to_string(index=False)
    )
    print("\nPeriodic-orbit comparison at a=0.94 a_H")
    print(
        periodic_frame[
            ["label", "a", "n_min", "n_max", "prey_amplitude", "T_num"]
        ].to_string(index=False)
    )

    print("\nGenerated data files:")
    for name in [
        "growth_law_analytical_metrics.csv",
        "growth_law_analytical_table.tex",
        "theta_logistic_sweep.csv",
        "growth_law_periodic_metrics.csv",
        "growth_law_periodic_table.tex",
        "growth_law_normalized_cycles.csv",
    ]:
        print(f"  {DATA_DIR / name}")

    print("\nGenerated figures:")
    for name in [
        "theta_logistic_thresholds_en.pdf",
        "theta_logistic_hopf_metrics_en.pdf",
        "growth_law_periodic_comparison_en.pdf",
    ]:
        print(f"  {FIGURES_DIR / name}")


if __name__ == "__main__":
    main()
