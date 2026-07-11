"""Executable documentation example tests.

These tests validate that primary README/Usage Guide workflow snippets
remain aligned with the current public API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from preml import quick_eda
from preml.eda import EDAAnalyzer
from preml.preprocessing import PreprocessingBuilder
from preml.recommendation_engine import EvaluationResult, ModelCandidate, RecommendationEngine
from preml.report import ReportGenerator
from preml.model_utils import BaselineTrainer
from sklearn.ensemble import HistGradientBoostingRegressor


def _make_dataframe(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "num1": rng.normal(0, 1, n),
            "num2": rng.normal(5, 2, n),
            "cat": rng.choice(["A", "B", "C"], n),
            "target": rng.choice([0, 1], n),
        }
    )
    df.loc[:4, "num1"] = np.nan
    return df


def test_readme_quickstart_workflow_executes():
    df = _make_dataframe()

    analysis = quick_eda(df, target="target")
    assert "recommendations" in analysis

    analyzer = EDAAnalyzer(df, target="target")
    analysis2 = analyzer.run()
    assert "data_quality_score" in analysis2

    builder = PreprocessingBuilder(analysis)
    X_train = df.drop(columns=["target"])
    X = builder.fit_transform(X_train)
    X_new = builder.transform(X_train.head(5))

    assert X.shape[0] == len(df)
    assert X_new.shape[0] == 5


def test_usage_guide_end_to_end_baseline_executes():
    df = _make_dataframe()
    analyzer = EDAAnalyzer(df, target="target")
    analysis = analyzer.run()

    builder = PreprocessingBuilder(analysis)
    preprocessor = builder.build_pipeline()

    trainer = BaselineTrainer()
    results = trainer.train_baselines(
        analysis_result=analysis,
        df=df,
        target_col="target",
        preprocessing_pipeline=preprocessor,
        cv=3,
    )

    assert isinstance(results, list)
    assert results
    assert "cv_scores" in results[0]


def test_usage_guide_report_generation_executes(tmp_path):
    df = _make_dataframe()
    analysis = quick_eda(df, target="target")
    report = ReportGenerator(analysis, df=df)

    html_path = tmp_path / "eda_report"
    report.save_report(str(html_path), format="html", embed_plots=False)
    report.save_report(str(html_path), format="md")
    report.save_report(str(html_path), format="txt")

    assert (tmp_path / "eda_report.html").exists()
    assert (tmp_path / "eda_report.md").exists()
    assert (tmp_path / "eda_report.txt").exists()


def test_usage_guide_recommendation_engine_examples_execute(monkeypatch):
    df = pd.DataFrame(
        {
            "feature1": np.linspace(0.0, 1.0, 24),
            "feature2": np.linspace(1.0, 2.0, 24),
            "category": np.array(["A", "B", "C"] * 8),
        }
    )
    y = df["feature1"] * 0.5 + df["feature2"] * 0.2 + np.random.default_rng(42).normal(0, 0.01, len(df))

    engine = RecommendationEngine(random_state=42)
    candidate = ModelCandidate(
        name="HistGradientBoostingRegressor",
        estimator_class=HistGradientBoostingRegressor,
        priority=1.0,
        hyperparams={"random_state": 42},
        supports_categorical=True,
        supports_missing=True,
        needs_scaling=False,
    )

    monkeypatch.setattr(engine, "_generate_candidates", lambda: [candidate])
    monkeypatch.setattr(
        engine,
        "_fast_cv_selector",
        lambda X, y, candidates, time_budget: [
            EvaluationResult(
                model=candidate,
                cv_score=0.91,
                cv_std=0.02,
                training_time=0.0,
                n_folds_completed=5,
                extrapolated_score=0.92,
            )
        ],
    )
    monkeypatch.setattr(
        engine,
        "_lccv_evaluate",
        lambda X, y, cand: EvaluationResult(
            model=cand,
            cv_score=0.93,
            cv_std=0.0,
            training_time=0.0,
            n_folds_completed=3,
            extrapolated_score=0.93,
        ),
    )
    monkeypatch.setattr(engine, "_successive_halving_optimize", lambda *args, **kwargs: {})
    monkeypatch.setattr(engine, "_bayesian_optimization_finetune", lambda *args, **kwargs: {})

    result = engine.fit(df, y, time_budget_seconds=5.0)
    summary = engine.summarize(result)
    recommendation = engine.get_recommendation(df, y)

    assert "Best model" in summary
    assert recommendation["model"] == "HistGradientBoostingRegressor"
    assert recommendation["pipeline"] is not None
    assert recommendation["cv_score"] is not None
