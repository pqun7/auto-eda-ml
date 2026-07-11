"""Smoke and unit tests for the remaining preml modules.

These tests focus on public helpers and end-to-end module behavior that was
not covered by the original test set: configuration defaults, normalization
helpers, model utilities, feature engineering, report generation, and
visualization smoke tests.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from preml.config import MLToolkitConfig
from preml.eda import EDAAnalyzer
from preml.feature_engineering import FeatureEngineering
from preml.model_utils import compute_metrics, cross_validate
from preml.recommendation_utils import normalize_recommendation_items
from preml.report import ReportGenerator
from preml.schema import (
    CategoricalProfile,
    CorrelationPair,
    DatasetMetadata,
    DuplicateReport,
    InfiniteReport,
    MissingColumnReport,
    MissingReport,
    NumericDistributionProfile,
    OutlierReport,
    FeatureProfile,
    Recommendation,
    TargetProfile,
)
from preml.visualization import (
    explain_visualizations,
    plot_missing_heatmap,
    plot_outlier_summary,
    plot_target_distribution,
)


@pytest.fixture
def minimal_analysis():
    return {
        "metadata": DatasetMetadata(4, 3, 0.01, {"float64": 1, "object": 1, "int64": 1}),
        "duplicates": DuplicateReport(0, 0.0),
        "infinite": InfiniteReport(),
        "missing": MissingReport(
            2,
            columns_with_missing=["num"],
            column_reports=[MissingColumnReport("num", 2, 50.0)],
        ),
        "outliers": [OutlierReport("num", "iqr", 1, 25.0, 0.0, 10.0)],
        "feature_profiles": [
            FeatureProfile(
                column="num",
                dtype="float64",
                numeric_profile=NumericDistributionProfile(
                    column="num",
                    count=4,
                    mean=2.5,
                    median=2.5,
                    std=1.0,
                    cv=0.4,
                    min=1.0,
                    max=4.0,
                    skewness=0.0,
                    kurtosis=0.0,
                    zero_percent=0.0,
                    negative_percent=0.0,
                    unique_count=4,
                ),
            ),
            FeatureProfile(
                column="cat",
                dtype="object",
                categorical_profile=CategoricalProfile(column="cat", unique_count=3),
            ),
        ],
        "correlation_pairs": [CorrelationPair("num", "num2", 0.95)],
        "target_profile": TargetProfile(
            column="target",
            dtype="int64",
            n_unique=2,
            is_binary=True,
            is_regression=False,
            class_distribution={0: 3, 1: 1},
        ),
        "recommendations": {
            "imputation": [
                Recommendation(
                    category="imputation",
                    action="Impute num with median.",
                    confidence=0.8,
                    evidence=[],
                )
            ],
            "outlier_handling": [],
            "transformation": [],
            "scaling": Recommendation(
                category="scaling",
                action="Scale numeric features.",
                confidence=0.9,
                evidence=[],
            ),
            "encoding": [],
            "feature_engineering": [],
            "feature_selection": [],
        },
        "data_quality_score": 82.5,
        "data_quality_notes": ["Missing values detected."],
    }


class TestConfig:
    def test_low_cardinality_threshold_is_available(self):
        cfg = MLToolkitConfig()
        assert cfg.low_cardinality_threshold == 10


class TestRecommendationUtils:
    def test_normalize_recommendation_items(self):
        rec = Recommendation(category="scaling", action="Scale features", confidence=0.9, evidence=[])
        assert normalize_recommendation_items(None) == []
        assert normalize_recommendation_items(rec) == [rec]
        assert normalize_recommendation_items([rec, "ignored"]) == [rec]


class TestModelUtils:
    def test_compute_metrics_regression(self):
        metrics = compute_metrics(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.5, 2.0]),
            task_type="regression",
        )
        assert set(metrics) == {"rmse", "mae", "r2"}
        assert metrics["rmse"] >= 0

    def test_compute_metrics_classification(self):
        metrics = compute_metrics(
            np.array([0, 1, 1, 0]),
            np.array([0, 1, 0, 0]),
            task_type="binary_classification",
        )
        assert set(metrics) == {"accuracy", "precision", "recall", "f1"}
        assert 0 <= metrics["accuracy"] <= 1

    def test_cross_validate_accepts_random_state(self):
        X = pd.DataFrame({"x": [0, 1, 0, 1, 0, 1, 0, 1]})
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        model = Pipeline([("estimator", LogisticRegression(max_iter=1000))])

        scores = cross_validate(model, X, y, cv=2, scoring="accuracy", random_state=42)

        assert "accuracy" in scores
        assert len(scores["accuracy"]) == 2


class TestFeatureEngineering:
    def test_datetime_feature_suggestion(self):
        df = pd.DataFrame({"event_time": pd.to_datetime(["2026-01-01", "2026-01-02"])})
        engineering = FeatureEngineering({"feature_profiles": [], "correlation_pairs": []}, df=df)

        suggestions = engineering.suggest_features()

        assert any("datetime" in rec.action.lower() for rec in suggestions)


class TestReportGenerator:
    def test_generate_text_markdown_html(self, minimal_analysis, tmp_path):
        report = ReportGenerator(minimal_analysis)

        text = report.generate_text()
        markdown = report.generate_markdown()
        html = report.generate_html(embed_plots=False)

        assert "EDA REPORT" in text
        assert "# PreML" in markdown
        assert "<html>" in html.lower()
        assert "Scale numeric features." in text

        output_path = tmp_path / "eda_report"
        report.save_report(str(output_path), format="txt", embed_plots=False)
        assert Path(str(output_path) + ".txt").exists()

    def test_accepts_eda_analyzer_instance(self):
        df = pd.DataFrame(
            {
                "num": [1.0, 2.0, 3.0, 4.0],
                "target": [0, 1, 0, 1],
            }
        )
        analyzer = EDAAnalyzer(df, target="target")

        report = ReportGenerator(analyzer, df=df)

        text = report.generate_text()

        assert "EDA REPORT" in text
        assert "Dataset Overview" in text


class TestVisualization:
    def test_visualization_smoke(self, minimal_analysis):
        df = pd.DataFrame(
            {
                "num": [1.0, np.nan, 3.0, 4.0],
                "num2": [1.0, 2.0, 3.0, 4.0],
                "cat": ["A", "B", "A", "C"],
                "target": [0, 1, 0, 1],
            }
        )

        fig_missing = plot_missing_heatmap(df)
        fig_outliers = plot_outlier_summary(minimal_analysis)
        fig_target = plot_target_distribution(df, minimal_analysis)
        explanations = explain_visualizations(minimal_analysis, recommendations=minimal_analysis["recommendations"])

        assert fig_missing is not None
        assert fig_outliers is not None
        assert fig_target is not None
        assert "numeric_distributions" in explanations
        assert "target_distribution" in explanations
