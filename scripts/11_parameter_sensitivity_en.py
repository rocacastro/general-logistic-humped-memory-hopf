"""
11_parameter_sensitivity_en.py

Local and one-at-a-time sensitivity analysis for the data-informed
humped-memory predator-prey scenario.

The script uses the nominal parameter set reported in the manuscript:
    K      = 88.206925
    eps    = 0.154743 day^{-1}
    alpha  = 0.103094
    beta   = 0.001
    gamma  = 0.02 day^{-1}
    theta  = 2, so f(n)=1-n^2

The following outputs are analyzed:
    a_H^dim      dimensional Hopf threshold
    tau_H        critical modal memory time, 1/a_H^dim
    omega_0^dim  dimensional Hopf frequency
    P_star       predator component of the coexistence equilibrium

Two complementary calculations are performed:
1. Normalized local sensitivities
       S_p^Y = (p/Y) dY/dp,
   evaluated by centered relative finite differences.
2. One-at-a-time parameter sweeps over [0.8 p_0, 1.2 p_0].

The script also verifies that the coexistence condition, the existence of a
positive Hopf threshold, and the supercritical sign ell_1(0)<0 are preserved
throughout the tested ranges.

Outputs are written to:
    data/generated/
    figures/

Required packages:
    numpy, scipy, pandas, matplotlib
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
# Nominal applied scenario
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Parameters:
    K: float = 88.206925
    eps: float = 0.154743
    alpha: float = 0.103094
    beta: float = 0.001
    gamma: float = 0.02
    theta: float = 2.0


NOMINAL = Parameters()
RELATIVE_STEP = 1.0e-5
SWEEP_FACTORS = np.linspace(0.8, 1.2, 101)
PARAMETER_ORDER = ("K", "eps", "beta", "gamma", "alpha")
PARAMETER_LABELS = {
    "K": r"$K$",
    "eps": r"$\varepsilon$",
    "beta": r"$\beta$",
    "gamma": r"$\gamma$",
    "alpha": r"$\alpha$",
}


# -----------------------------------------------------------------------------
# Analytical quantities
# -----------------------------------------------------------------------------
def growth_values(b: float, theta: float) -> tuple[float, float, float, float]:
    """Return F=f(b), G=f'(b), H=f''(b), S=f'''(b) for f(n)=1-n^theta."""
    F = 1.0 - b**theta
    G = -theta * b ** (theta - 1.0)
    H = -theta * (theta - 1.0) * b ** (theta - 2.0)
    S = -theta * (theta - 1.0) * (theta - 2.0) * b ** (theta - 3.0)
    return F, G, H, S


def phi_hat(a_hat: float, p: Parameters, b: float, F: float, G: float) -> float:
    beta_hat = p.beta / p.eps
    return (
        a_hat**3
        + (2.0 * p.K * beta_hat * F / G - 2.0 * b * G) * a_hat**2
        + b * (b * G**2 - 2.0 * p.K * beta_hat * F) * a_hat
        + 0.5 * p.K * beta_hat * b**2 * F * G
    )


def positive_root(p: Parameters, b: float, F: float, G: float) -> float:
    upper = 1.0
    while phi_hat(upper, p, b, F, G) <= 0.0:
        upper *= 2.0
        if upper > 1.0e12:
            raise RuntimeError("Could not bracket the positive root of Phi.")
    return float(brentq(lambda a: phi_hat(a, p, b, F, G), 0.0, upper))


def g4_and_l1(
    p: Parameters,
    b: float,
    F: float,
    G: float,
    H: float,
    S: float,
    a_hat: float,
    omega_hat: float,
) -> tuple[float, float]:
    """Compact first-Lyapunov-coefficient formula used in the manuscript."""
    alpha_hat = p.alpha / p.eps
    beta_hat = p.beta / p.eps
    kappa = p.K * alpha_hat
    rho = p.K * beta_hat
    imaginary = 1j

    m = (
        a_hat * b * G
        + 3.0 * imaginary * b * G * omega_hat
        - 2.0 * imaginary * a_hat * omega_hat
        + 4.0 * omega_hat**2
    ) / (omega_hat * (omega_hat - imaginary * a_hat))

    u1 = b * H + 2.0 * imaginary * omega_hat / b
    u2 = (
        2.0
        * imaginary
        * omega_hat
        * (b * G - imaginary * omega_hat) ** 2
        / (F * kappa * b**2)
    )

    denominator = (
        2.0 * imaginary * omega_hat
        - b * G
        + (b * rho * F * a_hat**2)
        / (2.0 * imaginary * omega_hat * (a_hat + 2.0 * imaginary * omega_hat) ** 2)
    )

    h1 = (u1 - (kappa * b) / (2.0 * imaginary * omega_hat) * u2) / denominator
    h3 = a_hat**2 * h1 / (a_hat + 2.0 * imaginary * omega_hat) ** 2
    h2 = (u2 + (rho * F / kappa) * h3) / (2.0 * imaginary * omega_hat)

    q3 = imaginary * omega_hat * (b * G - imaginary * omega_hat) / (rho * F * b)
    qbar2 = (b * G + imaginary * omega_hat) / (kappa * b)
    qbar3 = -imaginary * omega_hat * (b * G + imaginary * omega_hat) / (rho * F * b)

    d1 = H + b * S + (G + b * H - imaginary * omega_hat / b) * h1 - kappa * h2
    d2 = (2.0 * rho * H / kappa) * q3 + rho * (qbar2 * h3 + qbar3 * h2)

    G4 = float(np.real((d1 + imaginary * kappa * b * d2 / omega_hat) / m))
    ell1 = G4 / (2.0 * omega_hat)
    return G4, ell1


def evaluate(p: Parameters) -> Dict[str, float]:
    if min(p.K, p.eps, p.alpha, p.beta, p.gamma, p.theta) <= 0.0:
        raise ValueError("All parameters must be positive.")

    b = p.gamma / (p.K * p.beta)
    if not 0.0 < b < 1.0:
        raise ValueError(f"The coexistence condition 0<b<1 fails: b={b}.")

    F, G, H, S = growth_values(b, p.theta)
    if F <= 0.0 or G >= 0.0:
        raise ValueError("The Hopf calculation requires F>0 and G<0.")

    a_hat = positive_root(p, b, F, G)
    omega_hat_sq = -a_hat**2 * b * G / (2.0 * a_hat - b * G)
    if omega_hat_sq <= 0.0:
        raise ValueError("The computed Hopf frequency is not positive.")
    omega_hat = float(np.sqrt(omega_hat_sq))

    a_dim = p.eps * a_hat
    omega_dim = p.eps * omega_hat
    tau_mode = 1.0 / a_dim
    tau_mean = 2.0 / a_dim
    period = 2.0 * np.pi / omega_dim
    N_star = p.gamma / p.beta
    P_star = p.eps * F / p.alpha
    G4, ell1 = g4_and_l1(p, b, F, G, H, S, a_hat, omega_hat)

    return {
        "b": b,
        "F": F,
        "G": G,
        "H": H,
        "S": S,
        "a_hat": a_hat,
        "a_H_dim": a_dim,
        "tau_mode_H": tau_mode,
        "tau_mean_H": tau_mean,
        "omega_0_dim": omega_dim,
        "T_0": period,
        "N_star": N_star,
        "P_star": P_star,
        "G_4": G4,
        "ell_1": ell1,
    }


def perturb(p: Parameters, name: str, factor: float) -> Parameters:
    return replace(p, **{name: getattr(p, name) * factor})


def normalized_sensitivity(
    p: Parameters,
    parameter: str,
    metric: str,
    relative_step: float = RELATIVE_STEP,
) -> float:
    plus = evaluate(perturb(p, parameter, 1.0 + relative_step))[metric]
    minus = evaluate(perturb(p, parameter, 1.0 - relative_step))[metric]
    nominal = evaluate(p)[metric]
    # Because p_+-p_-=2 h p_0, the normalized sensitivity reduces to this form.
    return float((plus - minus) / (2.0 * relative_step * nominal))


# -----------------------------------------------------------------------------
# Tables
# -----------------------------------------------------------------------------
def make_local_sensitivity_table() -> pd.DataFrame:
    rows = []
    for name in PARAMETER_ORDER:
        rows.append(
            {
                "parameter": name,
                "label": PARAMETER_LABELS[name],
                "nominal_value": getattr(NOMINAL, name),
                "S_a_H_dim": normalized_sensitivity(NOMINAL, name, "a_H_dim"),
                "S_tau_mode_H": normalized_sensitivity(NOMINAL, name, "tau_mode_H"),
                "S_omega_0_dim": normalized_sensitivity(NOMINAL, name, "omega_0_dim"),
                "S_P_star": normalized_sensitivity(NOMINAL, name, "P_star"),
            }
        )
    return pd.DataFrame(rows)


def make_oat_data() -> pd.DataFrame:
    nominal_metrics = evaluate(NOMINAL)
    rows = []
    for name in PARAMETER_ORDER:
        for factor in SWEEP_FACTORS:
            metrics = evaluate(perturb(NOMINAL, name, float(factor)))
            rows.append(
                {
                    "parameter": name,
                    "label": PARAMETER_LABELS[name],
                    "factor": factor,
                    **metrics,
                    "a_H_ratio": metrics["a_H_dim"] / nominal_metrics["a_H_dim"],
                    "tau_ratio": metrics["tau_mode_H"] / nominal_metrics["tau_mode_H"],
                    "omega_ratio": metrics["omega_0_dim"] / nominal_metrics["omega_0_dim"],
                    "P_star_ratio": metrics["P_star"] / nominal_metrics["P_star"],
                }
            )
    return pd.DataFrame(rows)


def make_endpoint_table(oat: pd.DataFrame) -> pd.DataFrame:
    nominal_metrics = evaluate(NOMINAL)
    rows = []
    for name in PARAMETER_ORDER:
        subset = oat[oat["parameter"] == name].set_index("factor")
        lower = subset.loc[0.8]
        upper = subset.loc[1.2]
        rows.append(
            {
                "parameter": name,
                "label": PARAMETER_LABELS[name],
                "delta_a_H_minus20_pct": 100.0 * (lower["a_H_dim"] / nominal_metrics["a_H_dim"] - 1.0),
                "delta_a_H_plus20_pct": 100.0 * (upper["a_H_dim"] / nominal_metrics["a_H_dim"] - 1.0),
                "delta_tau_minus20_pct": 100.0 * (lower["tau_mode_H"] / nominal_metrics["tau_mode_H"] - 1.0),
                "delta_tau_plus20_pct": 100.0 * (upper["tau_mode_H"] / nominal_metrics["tau_mode_H"] - 1.0),
                "delta_omega_minus20_pct": 100.0 * (lower["omega_0_dim"] / nominal_metrics["omega_0_dim"] - 1.0),
                "delta_omega_plus20_pct": 100.0 * (upper["omega_0_dim"] / nominal_metrics["omega_0_dim"] - 1.0),
                "ell_1_min": min(float(lower["ell_1"]), float(upper["ell_1"])),
                "ell_1_max": max(float(lower["ell_1"]), float(upper["ell_1"])),
            }
        )
    return pd.DataFrame(rows)


def save_latex_tables(local: pd.DataFrame, endpoints: pd.DataFrame) -> None:
    local_lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Parameter & $S_p^{a_{\mathrm{H}}^{\mathrm{dim}}}$ & $S_p^{\tau_{\mathrm{H}}}$ & $S_p^{\omega_0^{\mathrm{dim}}}$ & $S_p^{P^*}$ \\",
        r"\midrule",
    ]
    for _, row in local.iterrows():
        local_lines.append(
            f"{row['label']} & {row['S_a_H_dim']:.6f} & {row['S_tau_mode_H']:.6f} & "
            f"{row['S_omega_0_dim']:.6f} & {row['S_P_star']:.6f} \\\\" 
        )
    local_lines += [r"\bottomrule", r"\end{tabular}"]
    (DATA_DIR / "parameter_local_sensitivity_table.tex").write_text(
        "\n".join(local_lines) + "\n", encoding="utf-8"
    )

    endpoint_lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Parameter & $\Delta a_{\mathrm{H}}^{-}$ & $\Delta a_{\mathrm{H}}^{+}$ & $\Delta\tau_{\mathrm{H}}^{-}$ & $\Delta\tau_{\mathrm{H}}^{+}$ & $\Delta\omega_0^{-}$ & $\Delta\omega_0^{+}$ \\",
        r"\midrule",
    ]
    for _, row in endpoints.iterrows():
        endpoint_lines.append(
            f"{row['label']} & {row['delta_a_H_minus20_pct']:.2f}\\% & {row['delta_a_H_plus20_pct']:.2f}\\% & "
            f"{row['delta_tau_minus20_pct']:.2f}\\% & {row['delta_tau_plus20_pct']:.2f}\\% & "
            f"{row['delta_omega_minus20_pct']:.2f}\\% & {row['delta_omega_plus20_pct']:.2f}\\% \\\\" 
        )
    endpoint_lines += [r"\bottomrule", r"\end{tabular}"]
    (DATA_DIR / "parameter_oat_endpoint_table.tex").write_text(
        "\n".join(endpoint_lines) + "\n", encoding="utf-8"
    )


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_local_sensitivities(local: pd.DataFrame) -> None:
    labels = [PARAMETER_LABELS[name] for name in PARAMETER_ORDER]
    y = np.arange(len(labels), dtype=float)
    width = 0.24

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.barh(y - width, local["S_a_H_dim"], height=width, label=r"$a_{\mathrm{H}}^{\mathrm{dim}}$")
    ax.barh(y, local["S_omega_0_dim"], height=width, label=r"$\omega_0^{\mathrm{dim}}$")
    ax.barh(y + width, local["S_P_star"], height=width, label=r"$P^*$")
    ax.axvline(0.0, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"Normalized local sensitivity $S_p^Y$")
    ax.set_title("Local sensitivity at the nominal greenhouse scenario")
    ax.legend()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, "parameter_local_sensitivity_en")


def plot_oat(oat: pd.DataFrame, metric: str, ylabel: str, title: str, stem: str) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.8))

    # K and beta produce the same curve because the threshold formulas depend
    # on them through K beta. Plot it once to avoid hiding one line behind the other.
    combined = oat[oat["parameter"] == "K"]
    ax.plot(combined["factor"], combined[metric], linewidth=2.0, label=r"$K$ or $\beta$")

    for name in ("gamma", "eps", "alpha"):
        subset = oat[oat["parameter"] == name]
        ax.plot(subset["factor"], subset[metric], linewidth=2.0, label=PARAMETER_LABELS[name])

    ax.axvline(1.0, linewidth=0.8, linestyle="--")
    ax.axhline(1.0, linewidth=0.8, linestyle="--")
    ax.set_xlabel(r"Parameter factor $p/p_0$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(0.8, 1.2)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, stem)


def main() -> None:
    nominal_metrics = evaluate(NOMINAL)
    local = make_local_sensitivity_table()
    oat = make_oat_data()
    endpoints = make_endpoint_table(oat)

    local.to_csv(DATA_DIR / "parameter_local_sensitivity.csv", index=False)
    oat.to_csv(DATA_DIR / "parameter_oat_sweep.csv", index=False)
    endpoints.to_csv(DATA_DIR / "parameter_oat_endpoints.csv", index=False)
    pd.DataFrame([nominal_metrics]).to_csv(
        DATA_DIR / "parameter_sensitivity_nominal_metrics.csv", index=False
    )
    save_latex_tables(local, endpoints)

    plot_local_sensitivities(local)
    plot_oat(
        oat,
        "a_H_ratio",
        r"$a_{\mathrm{H}}^{\mathrm{dim}}/a_{\mathrm{H},0}^{\mathrm{dim}}$",
        "One-at-a-time sensitivity of the dimensional Hopf threshold",
        "parameter_oat_threshold_en",
    )
    plot_oat(
        oat,
        "omega_ratio",
        r"$\omega_0^{\mathrm{dim}}/\omega_{0,0}^{\mathrm{dim}}$",
        "One-at-a-time sensitivity of the dimensional Hopf frequency",
        "parameter_oat_frequency_en",
    )

    # Numerical consistency check for the centered derivative.
    convergence = []
    for name in PARAMETER_ORDER:
        for metric in ("a_H_dim", "omega_0_dim", "P_star"):
            s1 = normalized_sensitivity(NOMINAL, name, metric, 1.0e-5)
            s2 = normalized_sensitivity(NOMINAL, name, metric, 5.0e-6)
            convergence.append(
                {
                    "parameter": name,
                    "metric": metric,
                    "S_h_1e-5": s1,
                    "S_h_5e-6": s2,
                    "absolute_difference": abs(s1 - s2),
                }
            )
    pd.DataFrame(convergence).to_csv(
        DATA_DIR / "parameter_sensitivity_step_check.csv", index=False
    )

    all_ell = oat["ell_1"].to_numpy()
    all_b = oat["b"].to_numpy()

    print("Nominal applied scenario")
    for key in (
        "a_H_dim",
        "tau_mode_H",
        "omega_0_dim",
        "T_0",
        "P_star",
        "ell_1",
    ):
        print(f"  {key:>14s} = {nominal_metrics[key]:.12g}")

    print("\nNormalized local sensitivities")
    print(
        local[
            [
                "parameter",
                "S_a_H_dim",
                "S_tau_mode_H",
                "S_omega_0_dim",
                "S_P_star",
            ]
        ].to_string(index=False)
    )

    print("\nEndpoint changes for +/-20% perturbations")
    print(endpoints.to_string(index=False))

    print("\nRobustness checks across all one-at-a-time sweeps")
    print(f"  b range          = [{all_b.min():.9f}, {all_b.max():.9f}]")
    print(f"  ell_1 range      = [{all_ell.min():.9f}, {all_ell.max():.9f}]")
    print(f"  all ell_1 < 0    = {bool(np.all(all_ell < 0.0))}")

    print("\nGenerated files")
    for path in sorted(DATA_DIR.glob("parameter_*")):
        print(f"  {path}")
    for path in sorted(FIGURES_DIR.glob("parameter_*")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
