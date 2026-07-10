# ML Toolkit – Complete Usage Guide

> **ML Toolkit** is a Python library that automates Exploratory Data Analysis (EDA), generates evidence-based recommendations, builds production-ready scikit-learn preprocessing pipelines, suggests feature engineering opportunities, trains baseline models, and creates comprehensive reports.

---

# Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Configuration](#configuration)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Statistics Engine](#statistics-engine)
- [Recommendation Engine](#recommendation-engine)
- [Visualization](#visualization)
- [Preprocessing](#preprocessing)
- [Feature Engineering](#feature-engineering)
- [Model Utilities](#model-utilities)
- [Reporting](#reporting)
- [Complete Workflow Example](#complete-workflow-example)
- [Exception Handling](#exception-handling)
- [Advanced Tips](#advanced-tips)

---

# Introduction

ML Toolkit provides an end-to-end workflow for classical machine learning projects.

It helps you:

- Perform automated Exploratory Data Analysis (EDA)
- Generate data-driven recommendations
- Build preprocessing pipelines automatically
- Suggest statistically supported feature engineering
- Train baseline machine learning models
- Generate Markdown, HTML, and text reports

Each module can be used independently or integrated into a complete ML workflow.

---

# Installation

## Install from PyPI

```bash
pip install ml-toolkit
```

## Install from source

```bash
git clone https://github.com/alinazer30/ml-toolkit.git

cd ml-toolkit

pip install -e .
```

---

# Configuration

All configurable thresholds and runtime settings are defined inside `MLToolkitConfig`.

Use the default configuration:

```python
from ml_toolkit.config import default_config

config = default_config
```

Or create your own:

```python
from ml_toolkit.config import MLToolkitConfig

config = MLToolkitConfig(
    missing_threshold=0.30,
    correlation_threshold=0.85,
    outlier_method="iqr",      # "iqr" or "zscore"
    random_state=42
)
```

Pass the configuration to any component that accepts it:

```python
from ml_toolkit.eda import EDAAnalyzer

analyzer = EDAAnalyzer(df, config=config)
```

---

# Exploratory Data Analysis (EDA)

`EDAAnalyzer` orchestrates the complete analysis pipeline.

It computes:

- Dataset metadata
- Missing values
- Duplicate rows
- Infinite values
- Outliers
- Feature profiles
- Correlation analysis
- Target analysis
- Recommendations
- Data quality score

## Run a complete analysis

```python
from ml_toolkit.eda import EDAAnalyzer

analyzer = EDAAnalyzer(
    df,
    target="price"
)

analysis = analyzer.run()
```

## Quick helper

```python
from ml_toolkit.eda import quick_eda

analysis = quick_eda(
    df,
    target="price"
)
```

## Available analysis outputs

```python
analysis.keys()
```

```python
dict_keys([
    "metadata",
    "duplicates",
    "infinite",
    "missing",
    "outliers",
    "feature_profiles",
    "correlation_pairs",
    "target_profile",
    "recommendations",
    "data_quality_score",
    "data_quality_notes",
])
```

## Generate a textual summary

```python
print(analyzer.summary())
```

---

# Statistics Engine

If you only need statistical analysis without recommendations, use `StatisticsEngine`.

```python
from ml_toolkit.statistics_engine import StatisticsEngine

engine = StatisticsEngine(
    df,
    target="price"
)
```

Run individual analyses:

```python
metadata = engine.compute_dataset_metadata()

duplicates = engine.compute_duplicate_report()

missing = engine.compute_missing_report()

outliers = engine.compute_outlier_report()

profiles = engine.compute_feature_profiles()

corr_pairs = engine.compute_correlation_pairs()

target_profile = engine.compute_target_profile()
```

Run everything at once:

```python
all_stats = engine.run_full_analysis()
```

Every output is returned as a strongly typed dataclass, for example:

- `MissingReport`
- `OutlierReport`
- `NumericDistributionProfile`

---

# Recommendation Engine

`RecommendationEngine` converts statistical evidence into actionable ML recommendations.

```python
from ml_toolkit.recommendation_engine import RecommendationEngine

rec_engine = RecommendationEngine(
    config=config
)

recommendations = rec_engine.generate_recommendations(
    analysis
)
```

Typical recommendation categories include:

- Imputation
- Scaling
- Encoding
- Feature Engineering
- Feature Selection
- Model Selection

## Example: Scaling recommendation

```python
scaling = recommendations["scaling"]

print(scaling.action)
```

## Example: Recommended models

```python
for model in recommendations["models"]:
    print(
        model.model_name,
        model.suitability,
        model.reason
    )
```

---

# Visualization

Visualization functions consume the existing EDA results.

They **never recompute statistics**, making them lightweight and efficient.

```python
from ml_toolkit.visualization import (
    plot_numeric_distributions,
    plot_missing_heatmap,
    plot_correlation_heatmap,
    plot_outlier_summary,
    plot_target_distribution,
    plot_target_correlations,
    plot_top_correlations_bar,
)
```

## Numeric distributions

```python
fig = plot_numeric_distributions(
    df,
    analysis,
    max_cols=10
)

fig.savefig("distributions.png")
```

---

## Missing values heatmap

```python
fig = plot_missing_heatmap(df)

fig.savefig("missing_heatmap.png")
```

---

## Correlation heatmap

```python
fig = plot_correlation_heatmap(
    df,
    analysis
)

fig.savefig("corr_heatmap.png")
```

---

## Outlier summary

```python
fig = plot_outlier_summary(analysis)

fig.savefig("outliers.png")
```

---

## Target distribution

```python
fig = plot_target_distribution(
    df,
    analysis
)

fig.savefig("target.png")
```

---

## Feature–target correlations

```python
fig = plot_target_correlations(
    df,
    analysis,
    top_n=10
)

fig.savefig("target_corrs.png")
```

---

## Strongest feature correlations

```python
fig = plot_top_correlations_bar(
    analysis,
    top_n=10
)

fig.savefig("top_corrs.png")
```

> All visualization functions accept an optional `ax` parameter and return a Matplotlib `Figure`.

---

# Preprocessing

`PreprocessingBuilder` automatically converts the EDA output into a production-ready `ColumnTransformer`.

```python
from ml_toolkit.preprocessing import PreprocessingBuilder

builder = PreprocessingBuilder(
    analysis
)

pipeline = builder.build_pipeline()

X_transformed = builder.fit_transform(df)
```

Use directly with scikit-learn:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(
    X_transformed,
    y
)
```

## Automatic preprocessing steps

The generated pipeline automatically:

- Removes constant and quasi-constant features
- Handles missing values
- Detects categorical numeric columns
- Applies power transformations to skewed variables
- Scales numerical features
- Encodes categorical variables

### Missing values

| Feature Type | Strategy |
|--------------|----------|
| Numeric with outliers | Median |
| Numeric without outliers | Mean |
| Categorical | Most frequent value |

### Scaling

| Condition | Scaler |
|-----------|--------|
| Outliers present | RobustScaler |
| Otherwise | StandardScaler |

### Encoding

| Category Type | Encoder |
|---------------|----------|
| Low cardinality | OneHotEncoder |
| High cardinality | OrdinalEncoder |

---

# Feature Engineering

`FeatureEngineering` analyzes statistical evidence and proposes useful engineered features.

```python
from ml_toolkit.feature_engineering import FeatureEngineering

fe = FeatureEngineering(
    analysis,
    df=df
)

suggestions = fe.suggest_features()
```

```python
for suggestion in suggestions:
    print(
        suggestion.action,
        suggestion.confidence
    )
```

Possible suggestions include:

- Ratio features
- Interaction features
- Feature binning
- Power transformations
- Datetime decomposition
- Crossed categorical variables

---

# Model Utilities

`BaselineTrainer` creates complete machine learning pipelines and evaluates them using cross-validation.

```python
from ml_toolkit.model_utils import BaselineTrainer

trainer = BaselineTrainer(
    config=config
)
```

## Build a model pipeline

```python
model_pipeline = trainer.build_model_pipeline(
    preprocessing_pipeline=pipeline,
    task_type="regression"
)
```

---

## Evaluate a baseline model

```python
evaluation = trainer.evaluate_baseline(
    model_pipeline,
    X=df.drop(columns=["target"]),
    y=df["target"],
    task_type="regression",
    cv=5
)

print(evaluation["mean_scores"])
```

---

## Train recommended baseline models

```python
results = trainer.train_baselines(
    analysis,
    df,
    target_col="target",
    preprocessing_pipeline=pipeline,
    cv=5
)

for result in results:
    print(
        result["model_name"],
        result["mean_scores"]
    )
```

---

## Standalone utilities

Compute metrics:

```python
metrics = compute_metrics(
    y_true,
    y_pred,
    task_type="regression"
)
```

Custom metrics:

```python
metrics = compute_metrics(
    y_true,
    y_pred,
    task_type="classification",
    extra_metrics={
        "f1_macro": f1_score_macro
    }
)
```

Cross-validation:

```python
scores = cross_validate(
    estimator,
    X,
    y,
    cv=5,
    scoring=[
        "accuracy",
        "f1_macro"
    ]
)
```

---

# Reporting

`ReportGenerator` creates professional reports from EDA results.

```python
from ml_toolkit.report import ReportGenerator

report = ReportGenerator(
    analysis,
    df=df
)
```

## Plain text

```python
text = report.generate_text()

print(text)
```

---

## Markdown

```python
markdown = report.generate_markdown()

with open("report.md", "w") as f:
    f.write(markdown)
```

---

## HTML

```python
html = report.generate_html(
    embed_plots=True
)

with open(
    "report.html",
    "w",
    encoding="utf-8"
) as f:
    f.write(html)
```

---

## Save automatically

```python
report.save_report(
    "report.html",
    format="html"
)

report.save_report(
    "report.md",
    format="md"
)

report.save_report(
    "report.txt",
    format="txt"
)
```

The generated HTML report is self-contained and includes:

- Statistics
- Recommendations
- Correlation plots
- Missing value heatmaps
- Distribution plots
- Data quality summary

---

# Complete Workflow Example

```python
import pandas as pd

from ml_toolkit.eda import EDAAnalyzer
from ml_toolkit.preprocessing import PreprocessingBuilder
from ml_toolkit.model_utils import BaselineTrainer
from ml_toolkit.report import ReportGenerator

df = pd.read_csv("housing.csv")

# EDA
analyzer = EDAAnalyzer(
    df,
    target="SalePrice"
)

analysis = analyzer.run()

print(
    analysis["data_quality_score"]
)

# Preprocessing
builder = PreprocessingBuilder(
    analysis
)

preprocessor = builder.build_pipeline()

X = builder.fit_transform(df)

# Baseline models
trainer = BaselineTrainer()

results = trainer.train_baselines(
    analysis,
    df,
    target_col="SalePrice",
    preprocessing_pipeline=preprocessor,
    cv=5
)

for result in results:
    print(
        result["model_name"],
        result["mean_scores"]
    )

# Generate report
report = ReportGenerator(
    analysis,
    df=df
)

report.save_report(
    "housing_report.html",
    format="html"
)
```

---

# Exception Handling

All library exceptions inherit from `MLToolkitError`.

```python
from ml_toolkit.exceptions import (
    MLToolkitError,
    DataValidationError,
    StatisticsError,
    PreprocessingError,
    ModelError,
    ReportError,
    VisualizationError,
)
```

Example:

```python
try:
    analysis = quick_eda(df)

except DataValidationError as error:
    print(error)

except MLToolkitError as error:
    print(error)
```

---

# Advanced Tips

## Customize the preprocessing pipeline

The generated preprocessing pipeline is a standard scikit-learn `ColumnTransformer`.

Feel free to modify it before training your model.

---

## Validate feature engineering suggestions

Feature engineering recommendations are statistically supported, but should always be validated on your own validation set before production use.

---

## Share one configuration object

Create a single `MLToolkitConfig` instance and reuse it across all components for consistent behavior.

---

## Working with very large datasets

`StatisticsEngine` operates on a DataFrame copy.

For very large datasets, consider sampling before running EDA to reduce memory usage and improve execution time.

---

# Recommended Workflow

```text
Dataset
   │
   ▼
EDAAnalyzer
   │
   ▼
Recommendations
   │
   ▼
PreprocessingBuilder
   │
   ▼
FeatureEngineering
   │
   ▼
BaselineTrainer
   │
   ▼
ReportGenerator
```

---

**ML Toolkit** is designed to automate the repetitive parts of machine learning workflows while keeping every recommendation transparent, interpretable, and grounded in statistical evidence.