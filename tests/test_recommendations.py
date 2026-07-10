"""
Unit tests for ml_toolkit.recommendation_engine.

Tests cover:
- Imputation, outlier, transformation, scaling, encoding recommendations
- Feature engineering and feature selection recommendations
- Model recommendations based on target profile
- Pipeline suggestion generation
- Edge cases (no numeric features, empty data)
- Integration with a minimal analysis_result dict
"""

import pytest
import pandas as pd
import numpy as np

from ml_toolkit.config import MLToolkitConfig
from ml_toolkit.recommendation_engine import RecommendationEngine
from ml_toolkit.schema import (
    DatasetMetadata,
    DuplicateReport,
    InfiniteReport,
    MissingReport,
    MissingColumnReport,
    OutlierReport,
    FeatureProfile,
    NumericDistributionProfile,
    CategoricalProfile,
    CorrelationPair,
    TargetProfile,
    Recommendation,
    PipelineSuggestion,
    ModelRecommendation,
)


# ------------------- Helpers -------------------
def _make_minimal_analysis(**overrides):
    """Create a minimal analysis dict with sensible defaults."""
    base = {
        "metadata": DatasetMetadata(100, 5, 0.5, {}),
        "duplicates": DuplicateReport(0, 0.0),
        "infinite": InfiniteReport(),
        "missing": MissingReport(0),
        "outliers": [],
        "feature_profiles": [],
        "correlation_pairs": [],
        "target_profile": None,
    }
    base.update(overrides)
    return base


# ------------------- Fixtures -------------------
@pytest.fixture
def engine():
    return RecommendationEngine()


@pytest.fixture
def numeric_profile():
    return NumericDistributionProfile(
        column="num",
        count=100,
        mean=10.0,
        median=9.0,
        std=2.0,
        cv=0.2,
        min=5.0,
        max=15.0,
        skewness=1.8,  # highly skewed
        kurtosis=2.0,
        zero_percent=0.0,
        negative_percent=0.0,
        unique_count=85,
    )


@pytest.fixture
def categorical_profile():
    return CategoricalProfile(
        column="cat",
        unique_count=5,
        missing_count=0,
        missing_percent=0.0,
        mode="A",
    )


@pytest.fixture
def feature_profiles(numeric_profile, categorical_profile):
    return [
        FeatureProfile(column="num", dtype="float64", numeric_profile=numeric_profile),
        FeatureProfile(column="cat", dtype="object", categorical_profile=categorical_profile),
    ]


@pytest.fixture
def target_profile_binary():
    return TargetProfile(
        column="target",
        dtype="int64",
        n_unique=2,
        is_regression=False,
        is_binary=True,
    )


@pytest.fixture
def target_profile_regression():
    return TargetProfile(
        column="target",
        dtype="float64",
        n_unique=100,
        is_regression=True,
        is_binary=False,
    )


# ------------------- Tests -------------------
class TestImputation:
    def test_no_missing_returns_empty(self, engine):
        analysis = _make_minimal_analysis(missing=MissingReport(0))
        recs = engine._imputation_recommendations([], analysis["missing"])
        assert recs == []

    def test_missing_numeric_with_outliers(self, engine, numeric_profile):
        missing = MissingReport(10, columns_with_missing=["num"], column_reports=[
            MissingColumnReport("num", 10, 10.0)
        ])
        engine._outlier_columns = ["num"]  # simulate outlier presence
        recs = engine._imputation_recommendations(
            [FeatureProfile(column="num", dtype="float64", numeric_profile=numeric_profile)],
            missing
        )
        assert len(recs) >= 1
        assert "median" in recs[0].action.lower()

    def test_missing_categorical(self, engine, categorical_profile):
        missing = MissingReport(5, columns_with_missing=["cat"], column_reports=[
            MissingColumnReport("cat", 5, 5.0)
        ])
        recs = engine._imputation_recommendations(
            [FeatureProfile(column="cat", dtype="object", categorical_profile=categorical_profile)],
            missing
        )
        assert any("mode" in r.action.lower() for r in recs)

    def test_high_missing_triggers_investigation(self, engine):
        missing = MissingReport(50, columns_with_missing=["num"], column_reports=[
            MissingColumnReport("num", 50, 50.0)
        ])
        # threshold 0.4 -> 40%, so 50% triggers high missing
        recs = engine._imputation_recommendations(
            [],
            missing
        )
        assert any("investigate" in r.action.lower() for r in recs)


class TestOutlierRecommendations:
    def test_no_outliers(self, engine):
        recs = engine._outlier_recommendations([])
        assert recs == []

    def test_minor_outliers(self, engine):
        out = OutlierReport("col", "iqr", 1, 1.0)
        recs = engine._outlier_recommendations([out])
        assert len(recs) == 1
        assert recs[0].confidence < 0.8

    def test_significant_outliers(self, engine):
        out = OutlierReport("col", "iqr", 10, 10.0)
        recs = engine._outlier_recommendations([out])
        assert recs[0].confidence > 0.8


class TestTransformationRecommendations:
    def test_no_skewed_features(self, engine):
        prof = NumericDistributionProfile(
            column="x", count=100, mean=0, median=0, std=1, cv=0.0,
            min=-3, max=3, skewness=0.5
        )
        fp = FeatureProfile(column="x", dtype="float64", numeric_profile=prof)
        recs = engine._transformation_recommendations([fp])
        assert recs == []

    def test_right_skewed_positive(self, engine):
        prof = NumericDistributionProfile(
            column="x", count=100, mean=10, median=5, std=5, cv=0.5,
            min=1, max=50, skewness=3.0
        )
        fp = FeatureProfile(column="x", dtype="float64", numeric_profile=prof)
        recs = engine._transformation_recommendations([fp])
        assert len(recs) == 1
        assert "log" in recs[0].action.lower()

    def test_right_skewed_negative(self, engine):
        prof = NumericDistributionProfile(
            column="x", count=100, mean=-10, median=-5, std=5, cv=0.0,
            min=-50, max=0, skewness=3.0
        )
        fp = FeatureProfile(column="x", dtype="float64", numeric_profile=prof)
        recs = engine._transformation_recommendations([fp])
        assert any("yeo" in r.action.lower() for r in recs)


class TestScalingRecommendation:
    def test_no_numeric(self, engine):
        rec = engine._scaling_recommendations([], None)
        assert "no numeric features" in rec.action.lower()

    def test_with_numeric(self, engine, numeric_profile):
        fp = FeatureProfile(column="num", dtype="float64", numeric_profile=numeric_profile)
        rec = engine._scaling_recommendations([fp], None)
        assert "scale" in rec.action.lower()


class TestEncodingRecommendations:
    def test_categorical_low_cardinality(self, engine):
        cp = CategoricalProfile(column="cat", unique_count=3)
        fp = FeatureProfile(column="cat", dtype="object", categorical_profile=cp)
        recs = engine._encoding_recommendations([fp])
        assert any("one-hot" in r.action.lower() for r in recs)

    def test_categorical_high_cardinality(self, engine):
        cp = CategoricalProfile(column="cat", unique_count=100)
        fp = FeatureProfile(column="cat", dtype="object", categorical_profile=cp)
        recs = engine._encoding_recommendations([fp])
        assert any("frequency" in r.action.lower() for r in recs)

    def test_numeric_categorical_like(self, engine):
        np = NumericDistributionProfile(
            column="x", count=100, mean=1, median=1, std=0.5, cv=0.0,
            min=0, max=2, unique_count=3, is_categorical_like=True
        )
        fp = FeatureProfile(column="x", dtype="int64", numeric_profile=np)
        recs = engine._encoding_recommendations([fp])
        assert any("categorical" in r.action.lower() for r in recs)


class TestFeatureSelectionRecommendations:
    def test_no_correlation(self, engine):
        recs = engine._correlation_recommendations([])
        assert recs == []

    def test_high_correlation(self, engine):
        cp = CorrelationPair("a", "b", 0.95)
        recs = engine._correlation_recommendations([cp])
        assert any("drop" in r.action.lower() for r in recs)


class TestModelRecommendations:
    def test_no_target_profile(self, engine):
        recs = engine._model_recommendations(None)
        assert recs == []

    def test_regression_models(self, engine, target_profile_regression):
        recs = engine._model_recommendations(target_profile_regression)
        assert any("LinearRegression" in r.model_name for r in recs)
        assert any("RandomForest" in r.model_name for r in recs)

    def test_binary_classification_models(self, engine, target_profile_binary):
        recs = engine._model_recommendations(target_profile_binary)
        assert any("LogisticRegression" in r.model_name for r in recs)

    def test_multiclass_models(self, engine):
        tp = TargetProfile(column="target", dtype="int64", n_unique=5, is_regression=False, is_binary=False)
        recs = engine._model_recommendations(tp)
        assert any("Logistic" in r.model_name for r in recs)


class TestGenerateRecommendations:
    def test_returns_all_categories(self, engine, feature_profiles, target_profile_binary):
        analysis = _make_minimal_analysis(
            feature_profiles=feature_profiles,
            target_profile=target_profile_binary,
        )
        result = engine.generate_recommendations(analysis)
        expected_cats = [
            "imputation", "outlier_handling", "transformation", "scaling",
            "encoding", "feature_engineering", "feature_selection",
            "pipeline", "models", "data_quality_notes"
        ]
        for cat in expected_cats:
            assert cat in result

    def test_pipeline_suggestion_type(self, engine, feature_profiles):
        analysis = _make_minimal_analysis(feature_profiles=feature_profiles)
        result = engine.generate_recommendations(analysis)
        assert isinstance(result["pipeline"], PipelineSuggestion)

    def test_model_recommendations_type(self, engine, feature_profiles, target_profile_binary):
        analysis = _make_minimal_analysis(
            feature_profiles=feature_profiles,
            target_profile=target_profile_binary
        )
        result = engine.generate_recommendations(analysis)
        assert all(isinstance(m, ModelRecommendation) for m in result["models"])