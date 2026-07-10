"""Facts extraction layer — computes statistical profiles without
making any decisions or recommendations.

All functions and methods in this module are purely computational.
They accept a DataFrame and configuration, and return dataclass
instances from `ml_toolkit.schema`. No side effects, no global state.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ml_toolkit.config import MLToolkitConfig, default_config
from ml_toolkit.exceptions import DataValidationError, StatisticsError
from ml_toolkit.schema import (
    CategoricalProfile,
    CorrelationPair,
    DatasetMetadata,
    DuplicateReport,
    FeatureProfile,
    InfiniteReport,
    MissingColumnReport,
    MissingReport,
    NumericDistributionProfile,
    OutlierReport,
    TargetProfile,
)


class StatisticsEngine:
    """Computes factual statistics about a tabular dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    target : str, optional
        Name of the target column. If provided, the engine will compute
        a `TargetProfile`.
    config : MLToolkitConfig, optional
        Configuration for thresholds and methods. If not supplied, the
        global `default_config` is used.

    Raises
    ------
    DataValidationError
        If `df` is not a pandas DataFrame, or if `target` is not a column
        in `df`.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
        config: Optional[MLToolkitConfig] = None,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError(
                "Input must be a pandas DataFrame.",
                details=type(df),
            )
        if target is not None and target not in df.columns:
            raise DataValidationError(
                f"Target column '{target}' not found in DataFrame.",
                details=list(df.columns[:10]),
            )

        self.df = df.copy()
        self.target = target
        self.config = config or default_config

        # Precompute numeric and categorical column lists for reuse
        self._numeric_cols = self.df.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        self._categorical_cols = self.df.select_dtypes(
            exclude=[np.number]
        ).columns.tolist()

    # ------------------------------------------------------------------
    # Dataset‑level checks
    # ------------------------------------------------------------------
    def compute_dataset_metadata(self) -> DatasetMetadata:
        """Return basic metadata about the dataset.

        Returns
        -------
        DatasetMetadata
        """
        n_rows, n_columns = self.df.shape
        memory_mb = self.df.memory_usage(deep=True).sum() / (1024 ** 2)
        col_types = self.df.dtypes.value_counts().to_dict()
        # Convert dtype objects to string for serialisability
        col_types = {str(k): v for k, v in col_types.items()}
        return DatasetMetadata(
            n_rows=n_rows,
            n_columns=n_columns,
            memory_mb=round(memory_mb, 2),
            column_types=col_types,
        )

    def compute_duplicate_report(self) -> DuplicateReport:
        """Analyse duplicate rows.

        Returns
        -------
        DuplicateReport
        """
        n_duplicates = self.df.duplicated().sum()
        dup_percent = 100.0 * n_duplicates / len(self.df) if len(self.df) > 0 else 0.0
        sample_idx = (
            self.df[self.df.duplicated(keep=False)].index[:100].tolist()
            if n_duplicates > 0
            else []
        )
        return DuplicateReport(
            total_duplicates=n_duplicates,
            duplicate_percent=round(dup_percent, 2),
            sample_indices=sample_idx,
        )

    def compute_infinite_report(self) -> InfiniteReport:
        """Detect columns containing positive or negative infinity.

        Returns
        -------
        InfiniteReport
        """
        inf_mask = self.df.replace([np.inf, -np.inf], np.nan).isna() & ~self.df.isna()
        col_counts = inf_mask.sum()
        cols_with_inf = col_counts[col_counts > 0]
        return InfiniteReport(
            columns_with_inf=cols_with_inf.index.tolist(),
            counts=cols_with_inf.to_dict(),
        )

    # ------------------------------------------------------------------
    # Missing values
    # ------------------------------------------------------------------
    def compute_missing_report(self) -> MissingReport:
        """Compute missing values per column and overall.

        Returns
        -------
        MissingReport
        """
        total_missing = self.df.isna().sum().sum()
        missing_by_col = self.df.isna().sum()
        missing_pct = (100.0 * missing_by_col / len(self.df)).round(2)

        cols_with_missing = missing_by_col[missing_by_col > 0].index.tolist()

        reports = [
            MissingColumnReport(
                column=col,
                missing_count=int(missing_by_col[col]),
                missing_percent=float(missing_pct[col]),
            )
            for col in cols_with_missing
        ]

        return MissingReport(
            total_missing=int(total_missing),
            columns_with_missing=cols_with_missing,
            column_reports=reports,
        )

    # ------------------------------------------------------------------
    # Outlier detection
    # ------------------------------------------------------------------
    def _iqr_outliers(
        self, series: pd.Series
    ) -> Tuple[float, float, pd.Series]:
        """IQR‑based outlier detection.

        Parameters
        ----------
        series : pd.Series
            Clean numeric series (no NaN).

        Returns
        -------
        lower_bound : float
        upper_bound : float
        outlier_mask : pd.Series (boolean)
        """
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - self.config.iqr_multiplier * iqr
        upper = q3 + self.config.iqr_multiplier * iqr
        mask = (series < lower) | (series > upper)
        return lower, upper, mask

    def _zscore_outliers(
        self, series: pd.Series
    ) -> Tuple[Optional[float], Optional[float], pd.Series]:
        """Z‑score based outlier detection.

        Parameters
        ----------
        series : pd.Series
            Clean numeric series (no NaN).

        Returns
        -------
        lower_bound : None (not applicable)
        upper_bound : None (not applicable)
        outlier_mask : pd.Series (boolean)
        """
        z = np.abs(sp_stats.zscore(series, nan_policy="omit"))
        mask = z > self.config.zscore_threshold
        return None, None, mask

    def compute_outlier_report(self) -> List[OutlierReport]:
        """Compute outlier statistics for every numeric column.

        Returns
        -------
        List[OutlierReport]
        """
        reports: List[OutlierReport] = []
        method = self.config.outlier_method

        for col in self._numeric_cols:
            series = self.df[col].dropna()
            if len(series) == 0:
                continue

            if method == "iqr":
                lower, upper, mask = self._iqr_outliers(series)
            elif method == "zscore":
                lower, upper, mask = self._zscore_outliers(series)
            else:
                raise DataValidationError(
                    f"Unsupported outlier method: {method}",
                    details="Use 'iqr' or 'zscore'.",
                )

            n_outliers = int(mask.sum())
            pct = (100.0 * n_outliers / len(series)) if len(series) > 0 else 0.0

            reports.append(
                OutlierReport(
                    column=col,
                    method=method,
                    outlier_count=n_outliers,
                    outlier_percent=round(pct, 2),
                    lower_bound=lower if lower is not None else None,
                    upper_bound=upper if upper is not None else None,
                )
            )

        return reports

    # ------------------------------------------------------------------
    # Distribution profiles (numeric / categorical)
    # ------------------------------------------------------------------
    def _compute_numeric_profile(self, col: str) -> NumericDistributionProfile:
        """Create a NumericDistributionProfile for a single column."""
        series = self.df[col].dropna()
        if len(series) == 0:
            # Return a profile with defaults; the caller will handle empty
            return NumericDistributionProfile(
                column=col,
                count=0,
                mean=np.nan,
                median=np.nan,
                std=np.nan,
                cv=np.nan,
                min=np.nan,
                max=np.nan,
            )

        # Percentiles
        percentiles = {
            "1%": round(series.quantile(0.01), 4),
            "5%": round(series.quantile(0.05), 4),
            "25%": round(series.quantile(0.25), 4),
            "50%": round(series.quantile(0.50), 4),
            "75%": round(series.quantile(0.75), 4),
            "95%": round(series.quantile(0.95), 4),
            "99%": round(series.quantile(0.99), 4),
        }

        mean_val = series.mean()
        median_val = series.median()
        std_val = series.std()
        cv_val = (std_val / mean_val) if mean_val != 0 else np.nan

        # Skewness & kurtosis (safe fallback for constant columns)
        try:
            sk = series.skew()
        except Exception:
            sk = np.nan
        try:
            ku = series.kurtosis()
        except Exception:
            ku = np.nan

        zero_pct = (series == 0).mean() * 100
        neg_pct = (series < 0).mean() * 100

        unique_count = series.nunique()
        is_categorical_like = unique_count <= self.config.max_unique_for_categorical_like

        return NumericDistributionProfile(
            column=col,
            count=len(series),
            mean=round(mean_val, 4),
            median=round(median_val, 4),
            std=round(std_val, 4),
            cv=round(cv_val, 4) if not np.isnan(cv_val) else np.nan,
            min=round(series.min(), 4),
            max=round(series.max(), 4),
            percentiles=percentiles,
            skewness=round(sk, 4) if not np.isnan(sk) else np.nan,
            kurtosis=round(ku, 4) if not np.isnan(ku) else np.nan,
            zero_percent=round(zero_pct, 2),
            negative_percent=round(neg_pct, 2),
            is_categorical_like=is_categorical_like,
            unique_count=unique_count,
        )

    def _compute_categorical_profile(self, col: str) -> CategoricalProfile:
        """Create a CategoricalProfile for a single column."""
        series = self.df[col].dropna()
        missing = self.df[col].isna().sum()
        missing_pct = (100.0 * missing / len(self.df)) if len(self.df) > 0 else 0.0

        if len(series) == 0:
            return CategoricalProfile(
                column=col,
                unique_count=0,
                missing_count=missing,
                missing_percent=round(missing_pct, 2),
            )

        unique_count = series.nunique()
        top_categories = (
            series.value_counts()
            .head(5)
            .reset_index()
            .rename(columns={"index": "category", col: "count"})
            .to_dict(orient="records")
        )
        mode = series.mode().iloc[0] if not series.mode().empty else None

        return CategoricalProfile(
            column=col,
            unique_count=unique_count,
            top_categories=top_categories,
            missing_count=missing,
            missing_percent=round(missing_pct, 2),
            mode=mode,
        )

    def compute_feature_profiles(self) -> List[FeatureProfile]:
        """Build a FeatureProfile for every column in the dataset.

        Returns
        -------
        List[FeatureProfile]
        """
        profiles: List[FeatureProfile] = []

        # Variance thresholds for constant detection
        variances = self.df[self._numeric_cols].var(numeric_only=True)
        quasi_const_mask = variances < self.config.constant_variance_threshold
        quasi_const = variances[quasi_const_mask].index.tolist()

        # Constant columns (nunique <= 1) across all columns
        const_cols = self.df.columns[self.df.nunique() <= 1].tolist()

        for col in self.df.columns:
            if col == self.target:
                continue  # target handled separately

            dtype = str(self.df[col].dtype)
            is_const = col in const_cols
            is_quasi = col in quasi_const

            if pd.api.types.is_numeric_dtype(self.df[col]):
                num_profile = self._compute_numeric_profile(col)
                profiles.append(
                    FeatureProfile(
                        column=col,
                        dtype=dtype,
                        numeric_profile=num_profile,
                        is_constant=is_const,
                        is_quasi_constant=is_quasi,
                    )
                )
            else:
                cat_profile = self._compute_categorical_profile(col)
                profiles.append(
                    FeatureProfile(
                        column=col,
                        dtype=dtype,
                        categorical_profile=cat_profile,
                        is_constant=is_const,
                        is_quasi_constant=False,
                    )
                )

        return profiles

    # ------------------------------------------------------------------
    # Correlation
    # ------------------------------------------------------------------
    def compute_correlation_pairs(self) -> List[CorrelationPair]:
        """Identify all pairwise Pearson correlations exceeding the
        configured threshold.

        Returns
        -------
        List[CorrelationPair]
        """
        if len(self._numeric_cols) < 2:
            return []

        corr_mat = self.df[self._numeric_cols].corr()
        pairs: List[CorrelationPair] = []
        threshold = self.config.correlation_threshold

        for i in range(len(corr_mat.columns)):
            for j in range(i + 1, len(corr_mat.columns)):
                val = corr_mat.iloc[i, j]
                if abs(val) >= threshold:
                    pairs.append(
                        CorrelationPair(
                            feature_a=corr_mat.columns[i],
                            feature_b=corr_mat.columns[j],
                            coefficient=round(val, 4),
                        )
                    )

        return pairs

    # ------------------------------------------------------------------
    # Target profile
    # ------------------------------------------------------------------
    def compute_target_profile(self) -> Optional[TargetProfile]:
        """Analyse the target column, if defined.

        Returns
        -------
        TargetProfile or None
        """
        if self.target is None:
            return None

        series = self.df[self.target]
        missing = series.isna().sum()
        missing_pct = (100.0 * missing / len(series)) if len(series) > 0 else 0.0
        dtype = str(series.dtype)

        if pd.api.types.is_numeric_dtype(series):
            # Determine if regression or binary classification
            unique_vals = series.dropna().nunique()
            is_binary = unique_vals == 2
            # A continuous numeric target is considered regression, but
            # if it has very few unique values (< 20) it might be classification.
            is_regression = unique_vals >= 20  # heuristic
            class_dist = {}
            if not is_regression:
                class_dist = (
                    series.value_counts().sort_index().to_dict()
                )
            return TargetProfile(
                column=self.target,
                dtype=dtype,
                n_unique=unique_vals,
                missing_count=int(missing),
                missing_percent=round(missing_pct, 2),
                is_regression=is_regression,
                is_binary=is_binary,
                class_distribution=class_dist,
            )
        else:
            # Categorical target
            unique_vals = series.nunique()
            class_dist = series.value_counts().to_dict()
            return TargetProfile(
                column=self.target,
                dtype=dtype,
                n_unique=unique_vals,
                missing_count=int(missing),
                missing_percent=round(missing_pct, 2),
                is_regression=False,
                is_binary=unique_vals == 2,
                class_distribution=class_dist,
            )

    # ------------------------------------------------------------------
    # Full analysis convenience
    # ------------------------------------------------------------------
    def run_full_analysis(self) -> Dict[str, any]:
        """Execute all statistical computations and return a dictionary
        of dataclass objects.

        Returns
        -------
        dict
            Keys include:
                - 'metadata' : DatasetMetadata
                - 'duplicates' : DuplicateReport
                - 'infinite' : InfiniteReport
                - 'missing' : MissingReport
                - 'outliers' : List[OutlierReport]
                - 'feature_profiles' : List[FeatureProfile]
                - 'correlation_pairs' : List[CorrelationPair]
                - 'target_profile' : TargetProfile or None
        """
        return {
            "metadata": self.compute_dataset_metadata(),
            "duplicates": self.compute_duplicate_report(),
            "infinite": self.compute_infinite_report(),
            "missing": self.compute_missing_report(),
            "outliers": self.compute_outlier_report(),
            "feature_profiles": self.compute_feature_profiles(),
            "correlation_pairs": self.compute_correlation_pairs(),
            "target_profile": self.compute_target_profile(),
        }