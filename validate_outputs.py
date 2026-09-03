#!/usr/bin/env python3
"""Check that the files required by the revised manuscript are present."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED = [
    "data/raw/greenhouse_counts_alpysbayeva2024.csv",
    "data/generated/application_uncertainty_summary.csv",
    "data/generated/application_uncertainty_summary_table.tex",
    "data/generated/growth_law_analytical_metrics.csv",
    "data/generated/growth_law_periodic_metrics.csv",
    "data/generated/memory_model_applied_critical_metrics.csv",
    "data/generated/memory_model_equal_mean_cycle_metrics.csv",
    "data/generated/memory_model_equal_mean_floquet.csv",
    "data/generated/parameter_local_sensitivity.csv",
    "data/generated/parameter_oat_sweep.csv",
    "data/generated/theta_logistic_sweep.csv",
    "figures/application_uncertainty_distributions_en.pdf",
    "figures/applied_control_theta2_orbit_clear_en.pdf",
    "figures/bifurcation_diagram_a_hyperlogistic_en.pdf",
    "figures/memory_cycle_phase_comparison_en.pdf",
    "figures/memory_kernels_comparison_english.pdf",
    "figures/memory_kernels_equal_mean_en.pdf",
    "figures/parameter_oat_frequency_en.pdf",
    "figures/parameter_oat_threshold_en.pdf",
    "figures/theta_logistic_hopf_metrics_en.pdf",
    "figures/theta_logistic_thresholds_en.pdf",
]

missing = [relative for relative in REQUIRED if not (ROOT / relative).is_file()]
if missing:
    print("Missing required outputs:")
    for relative in missing:
        print(f"  - {relative}")
    raise SystemExit(1)

print(f"Validation passed: {len(REQUIRED)} required files are present.")
