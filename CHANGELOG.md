# Changelog

## Version 2.0.0 — expanded reproducibility archive

### Added

- Numerical generality analysis for several admissible prey-growth laws.
- Integer theta-logistic sweep and periodic-orbit comparison.
- Local normalized sensitivities and one-at-a-time parameter sweeps.
- Matched-mean quantitative comparison of exponential and humped memory.
- Latin-hypercube scenario analysis with 20,000 parameter sets.
- New generated CSV and LaTeX tables.
- New analysis and diagnostic figures in PDF and PNG formats.
- `run_all.py`, `validate_outputs.py`, and Zenodo metadata.

### Changed

- Updated repository metadata to version 2.0.0.
- Renamed the introductory kernel figure to
  `memory_kernels_comparison_english.*` to match the revised manuscript.
- Expanded `requirements.txt` and documentation.

### Removed

- The obsolete script `01_hopf_constants_table_en.py`, superseded by
  `10_growth_law_generality_en.py`.
- The obsolete generated files `hopf_constants_table_en.csv` and
  `hopf_constants_table_en.tex`.
- The incorrectly named empty placeholder `data/generated/gitkeep`.

The previous state remains permanently available in the Zenodo/GitHub
version 1.0.1 archive.
