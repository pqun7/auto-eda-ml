# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog.

## [0.1.4] - 2026-07-11

### Added

- Added `fit` and `transform` lifecycle methods to `PreprocessingBuilder`.
- Added adaptive config method `MLToolkitConfig.adapt_to_dataset(df)`.
- Added config field `n_jobs` for parallel model evaluation controls.
- Added CI workflow for multi-version Python test validation.
- Added contributing and governance documents.

### Changed

- Improved defaults for missingness, correlation, and skewness thresholds.
- Improved cross-validation splitter behavior to avoid integer-regression misclassification.
- Improved user-facing validation and error guidance in key modules.
- Updated dependency version bounds for compatibility clarity.
- Synchronized runtime package metadata with project metadata.

### Fixed

- Fixed recommendation input validation message key mismatch (`outliers` vs `outlier_reports`).
- Fixed visualization docstring contamination in `explain_visualizations`.
- Corrected repository URL inconsistency in documentation.
