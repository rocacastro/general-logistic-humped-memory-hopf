"""
13_application_uncertainty_en.py

Scenario-based uncertainty propagation for the data-informed greenhouse
application in the humped-memory predator-prey model.

Important interpretation:
- The greenhouse experiment provides only three control counts and three
  predator-treatment counts, with no replicated measurement-error model.
- Therefore, this script does NOT calculate confidence intervals or a
  posterior distribution.
- Instead, it evaluates a transparent scenario ensemble in which the five
  dimensional parameters K, eps, alpha, beta and gamma vary independently
  and uniformly between 80% and 120% of their nominal values.
- The prey-growth exponent is fixed at theta=2; its effect is studied
  separately in the growth-law generality analysis.

A Latin hypercube design with a fixed seed is used for reproducibility.
For every scenario, the script computes:
    b, a_H^dim, 1/a_H^dim, 2/a_H^dim, omega_0^dim, T_0, P*, ell_1(0).

Outputs:
    data/generated/application_uncertainty_samples.csv
    data/generated/application_uncertainty_summary.csv
    data/generated/application_uncertainty_classification.csv
    data/generated/application_uncertainty_summary_table.tex
    figures/application_uncertainty_distributions_en.pdf
    figures/application_uncertainty_distributions_en.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import qmc


SCRIPT_PATH = Path(__file__).resolve()
if SCRIPT_PATH.parent.name == "scripts":
    ROOT_DIR = SCRIPT_PATH.parent.parent
else:
    ROOT_DIR = SCRIPT_PATH.parent

DATA_DIR = ROOT_DIR / "data" / "generated"
FIGURES_DIR = ROOT_DIR / "figures"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Parameters:
    K: float = 88.206925
    eps: float = 0.154743
    alpha: float = 0.103094
    beta: float = 0.001
    gamma: float = 0.02
    theta: float = 2.0


NOMINAL = Parameters()
SAMPLE_SIZE = 20_000
SEED = 20_260_829
LOW_FACTOR = 0.8
HIGH_FACTOR = 1.2
APPLIED_MEMORY_RATE = 0.18  # day^{-1}; value used in the applied simulation
PARAMETER_NAMES = ("K", "eps", "alpha", "beta", "gamma")


def growth_values(b: float, theta: float) -> tuple[float, float, float, float]:
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
    b = p.gamma / (p.K * p.beta)
    if not 0.0 < b < 1.0:
        raise ValueError("The coexistence condition 0<b<1 is not satisfied.")

    F, G, H, S = growth_values(b, p.theta)
    if F <= 0.0 or G >= 0.0:
        raise ValueError("The Hopf calculation requires F>0 and G<0.")

    a_hat = positive_root(p, b, F, G)
    omega_hat_sq = -a_hat**2 * b * G / (2.0 * a_hat - b * G)
    omega_hat = float(np.sqrt(omega_hat_sq))

    a_dim = p.eps * a_hat
    omega_dim = p.eps * omega_hat
    G4, ell1 = g4_and_l1(p, b, F, G, H, S, a_hat, omega_hat)

    return {
        "b": b,
        "a_H_dim": a_dim,
        "tau_mode_H": 1.0 / a_dim,
        "tau_mean_H": 2.0 / a_dim,
        "omega_0_dim": omega_dim,
        "T_0": 2.0 * np.pi / omega_dim,
        "P_star": p.eps * F / p.alpha,
        "G_4": G4,
        "ell_1": ell1,
    }


def latin_hypercube_samples() -> pd.DataFrame:
    sampler = qmc.LatinHypercube(d=len(PARAMETER_NAMES), seed=SEED)
    unit = sampler.random(SAMPLE_SIZE)
    factors = LOW_FACTOR + (HIGH_FACTOR - LOW_FACTOR) * unit

    rows = []
    for factors_i in factors:
        values = {
            name: getattr(NOMINAL, name) * factor
            for name, factor in zip(PARAMETER_NAMES, factors_i)
        }
        p = Parameters(theta=NOMINAL.theta, **values)
        metrics = evaluate(p)
        row = {
            **values,
            "theta": NOMINAL.theta,
            **metrics,
            "applied_a_below_threshold": APPLIED_MEMORY_RATE < metrics["a_H_dim"],
            "supercritical": metrics["ell_1"] < 0.0,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def summary_table(samples: pd.DataFrame) -> pd.DataFrame:
    nominal = evaluate(NOMINAL)
    metrics = [
        ("a_H_dim", r"$a_{\mathrm H}^{\mathrm{dim}}$", r"day$^{-1}$"),
        ("tau_mode_H", r"$\tau_{\mathrm{mode,H}}$", "days"),
        ("tau_mean_H", r"$\tau_{\mathrm{mean,H}}$", "days"),
        ("omega_0_dim", r"$\omega_0^{\mathrm{dim}}$", r"day$^{-1}$"),
        ("T_0", r"$T_0$", "days"),
        ("P_star", r"$P^*$", "count units"),
        ("ell_1", r"$\ell_1(0)$", "--"),
    ]
    rows = []
    for key, label, unit in metrics:
        q025, q25, q50, q75, q975 = samples[key].quantile([0.025, 0.25, 0.5, 0.75, 0.975])
        rows.append(
            {
                "metric": key,
                "label": label,
                "unit": unit,
                "nominal": nominal[key],
                "p2_5": q025,
                "p25": q25,
                "median": q50,
                "p75": q75,
                "p97_5": q975,
            }
        )
    return pd.DataFrame(rows)


def write_latex_table(summary: pd.DataFrame) -> None:
    rows = []
    for _, r in summary.iterrows():
        rows.append(
            f"{r['label']} & {r['nominal']:.6f} & {r['median']:.6f} & "
            f"[{r['p2_5']:.6f}, {r['p97_5']:.6f}] " + r"\\"
        )
    content = "\n".join(
        [
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"Quantity & Nominal & Scenario median & Scenario 2.5--97.5\% range \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    (DATA_DIR / "application_uncertainty_summary_table.tex").write_text(content, encoding="utf-8")


def make_figure(samples: pd.DataFrame, summary: pd.DataFrame) -> None:
    nominal = evaluate(NOMINAL)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    items = [
        ("a_H_dim", r"Hopf threshold $a_H^{dim}$ (day$^{-1}$)", nominal["a_H_dim"]),
        ("tau_mode_H", r"Critical modal memory time (days)", nominal["tau_mode_H"]),
        ("omega_0_dim", r"Hopf frequency $\omega_0^{dim}$ (day$^{-1}$)", nominal["omega_0_dim"]),
    ]

    for ax, (key, xlabel, nominal_value) in zip(axes, items):
        values = samples[key].to_numpy()
        q025, q975 = np.quantile(values, [0.025, 0.975])
        ax.hist(values, bins=55, density=True, alpha=0.75)
        ax.axvline(nominal_value, linewidth=1.8, label="Nominal")
        ax.axvline(q025, linestyle="--", linewidth=1.2, label="2.5% and 97.5%")
        ax.axvline(q975, linestyle="--", linewidth=1.2)
        if key == "a_H_dim":
            ax.axvline(APPLIED_MEMORY_RATE, linestyle=":", linewidth=1.8, label=r"Applied $a=0.18$")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Scenario density")
        ax.legend(fontsize=8)

    fig.suptitle("Scenario-based propagation of parameter uncertainty")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "application_uncertainty_distributions_en.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "application_uncertainty_distributions_en.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    samples = latin_hypercube_samples()
    summary = summary_table(samples)

    samples.to_csv(DATA_DIR / "application_uncertainty_samples.csv", index=False)
    summary.to_csv(DATA_DIR / "application_uncertainty_summary.csv", index=False)

    classification = pd.DataFrame(
        {
            "quantity": [
                "sample_size",
                "fraction_with_a_0.18_below_threshold",
                "fraction_supercritical",
                "minimum_b",
                "maximum_b",
            ],
            "value": [
                len(samples),
                samples["applied_a_below_threshold"].mean(),
                samples["supercritical"].mean(),
                samples["b"].min(),
                samples["b"].max(),
            ],
        }
    )
    classification.to_csv(DATA_DIR / "application_uncertainty_classification.csv", index=False)

    write_latex_table(summary)
    make_figure(samples, summary)

    print("Scenario ensemble complete")
    print(f"Samples: {len(samples)}")
    print(
        "Fraction with a=0.18 day^-1 below the sampled threshold: "
        f"{samples['applied_a_below_threshold'].mean():.4f}"
    )
    print(f"Fraction with ell_1(0)<0: {samples['supercritical'].mean():.4f}")
    print(summary[["metric", "nominal", "median", "p2_5", "p97_5"]].to_string(index=False))


if __name__ == "__main__":
    main()
