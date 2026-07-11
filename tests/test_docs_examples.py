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
from preml.report import ReportGenerator
from preml.model_utils import BaselineTrainer


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
