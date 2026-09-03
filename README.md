# General logistic growth in humped-memory predator-prey models

<!-- Add the Zenodo v2.0.0 DOI badge here after the release is archived. -->

## Reproducibility archive — version 2.0.0

This repository contains the Python scripts, input data, generated numerical
results, and figures supporting a study of Hopf bifurcation in predator-prey
models with **general logistic prey growth** and humped memory.

The primary mathematical result is that the local Hopf spectral mechanism is
preserved for a broad class of admissible logistic functions, while the
critical threshold, frequency, direction, and resulting periodic behavior
retain the local signature of the selected growth law. The computational
archive also supports parameter-sensitivity, uncertainty, and matched-mean
memory-kernel comparisons.

The manuscript source files are intentionally **not** included.

> **Zenodo status.** This package is prepared for the GitHub release `v2.0.0`.
> Zenodo will assign a new version-specific DOI after the release is published.
> The previous archived version is available at
> `https://doi.org/10.5281/zenodo.20985929`, but that DOI does not identify the
> expanded version-2 files.

## Repository structure

```text
general-logistic-humped-memory-hopf/
├── README.md
├── CHANGELOG.md
├── CITATION.cff
├── .zenodo.json
├── LICENSE
├── VERSION
├── requirements.txt
├── run_all.py
├── validate_outputs.py
├── scripts/
├── data/
│   ├── README_data.md
│   ├── raw/
│   └── generated/
└── figures/
    └── README_figures.md
```

## Scripts

| Script | Purpose |
|---|---|
| `00_memory_kernels_comparison_en.py` | Introductory exponential/humped kernel comparison. |
| `02_hyperlogistic_bifurcation_diagram_en.py` | Local bifurcation diagram for the quadratic benchmark. |
| `06_floquet_multipliers_hyperlogistic_en.py` | Periodic orbit and Floquet multipliers for the quadratic benchmark. |
| `08_applied_control_theta2_clear_figure_fixed.py` | Applied greenhouse phase portrait and time series. |
| `09_applied_control_theta2_floquet.py` | Shooting and Floquet computation for the applied orbit. |
| `10_growth_law_generality_en.py` | Generality analysis across growth laws and theta-logistic sweep. |
| `11_parameter_sensitivity_en.py` | Local sensitivities and one-at-a-time parameter sweeps. |
| `12_exponential_vs_humped_memory_en.py` | Matched-mean comparison of exponential and humped memory. |
| `13_application_uncertainty_en.py` | Latin-hypercube scenario analysis with a fixed random seed. |

The former script `01_hopf_constants_table_en.py` is not part of version 2.0.0;
its role is fully superseded by script 10 and it remains available in the
version 1.0.1 archive.

## Requirements

Python 3.10 or later is recommended. Install the dependencies from the
repository root:

```bash
python -m pip install -r requirements.txt
```

Required packages:

```text
numpy
scipy
pandas
matplotlib
```

## Reproducing the outputs

To run the complete workflow:

```bash
python run_all.py
python validate_outputs.py
```

The full workflow includes long integrations, shooting computations, Floquet
multipliers, and 20,000 Latin-hypercube scenarios. A faster partial check is:

```bash
python run_all.py --quick
```

Individual scripts can also be executed directly, for example:

```bash
python scripts/10_growth_law_generality_en.py
python scripts/11_parameter_sensitivity_en.py
python scripts/12_exponential_vs_humped_memory_en.py
python scripts/13_application_uncertainty_en.py
```

Each script determines the repository root from its own location and writes
outputs to `data/generated/` and `figures/`.

## Data provenance

`data/raw/greenhouse_counts_alpysbayeva2024.csv` contains greenhouse counts of
*Tetranychus urticae* transcribed from the published experiment cited in the
manuscript. The repository contains no newly collected experimental data.
All other data files are generated computationally from the model equations,
parameter values, and declared scenario ranges.

The Latin-hypercube percentiles are scenario-based uncertainty summaries; they
are not statistical confidence or credible intervals inferred from the six
published observations.

## Main analyses supported

- Hopf thresholds for general logistic prey-growth laws.
- Integer theta-logistic sweeps and periodic-orbit metrics.
- Local and finite-variation parameter sensitivity.
- Biological interpretation of modal and mean memory times.
- Scenario-based uncertainty propagation.
- Equal-mean exponential versus humped-memory comparison.
- Bifurcation diagrams, shooting residuals, and Floquet multipliers.

## License

The Python code is released under the MIT License. The raw greenhouse counts
remain attributable to their published source. Generated outputs should be
cited together with the archived software release and the associated article.

## Citation

After Zenodo archives release `v2.0.0`, use its version-specific DOI for exact
reproducibility. The repository metadata files are prepared for that release.
The DOI can then be added to this section and to `CITATION.cff` without changing
the computational contents.
