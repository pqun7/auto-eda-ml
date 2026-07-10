"""
report.py — Automated report generation from EDA results.

This module creates rich, self‑contained reports in multiple formats
(HTML, Markdown, plain text) from the output of
:class:`ml_toolkit.eda.EDAAnalyzer`.

Reports include statistical summaries, recommendations, data quality scores,
and optional embedded visualisations.  The module depends solely on the
public API of the package and standard library components; it never
recomputes statistics or modifies the analysis.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.figure
import pandas as pd

from ml_toolkit.config import MLToolkitConfig, default_config
from ml_toolkit.exceptions import ReportError
from ml_toolkit.schema import (
    Recommendation,
    FeatureProfile,
    CorrelationPair,
    MissingColumnReport,
    OutlierReport,
    TargetProfile,
)
from ml_toolkit.visualization import (
    plot_correlation_heatmap,
    plot_missing_heatmap,
    plot_numeric_distributions,
    plot_outlier_summary,
    plot_target_correlations,
    plot_target_distribution,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_REPORT_CSS = """
body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
       margin: 40px; color: #333; }
h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 5px; }
h2 { color: #2980b9; margin-top: 30px; }
table { border-collapse: collapse; width: 100%; margin: 15px 0; }
th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
th { background-color: #3498db; color: white; }
tr:nth-child(even) { background-color: #f2f2f2; }
pre { background: #f4f4f4; padding: 10px; border-radius: 5px; }
.quality-score { font-size: 2em; font-weight: bold; }
.warning { color: #e74c3c; }
.good { color: #27ae60; }
"""

# ---------------------------------------------------------------------------
# Helper: convert matplotlib figure to HTML img tag
# ---------------------------------------------------------------------------
def _fig_to_html(fig: matplotlib.figure.Figure) -> str:
    """Encode a matplotlib figure as a base64 PNG data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)  # prevent memory leaks
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return f'<img src="data:image/png;base64,{img_base64}" style="max-width:100%;" />'


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------
class ReportGenerator:
    """Generates formatted EDA reports.

    The generator consumes the complete analysis dictionary produced by
    :meth:`EDAAnalyzer.run() <ml_toolkit.eda.EDAAnalyzer.run>` and
    optionally the original DataFrame for embedded plots.

    Parameters
    ----------
    analysis_result : dict
        The full analysis result dictionary.
    df : pd.DataFrame, optional
        Original dataset. Required for generating plots. If omitted, the
        report will contain only textual/statistical information.
    config : MLToolkitConfig, optional
        Configuration object; used for visualisation settings.
    """

    def __init__(
        self,
        analysis_result: Dict[str, Any],
        df: Optional[pd.DataFrame] = None,
        config: Optional[MLToolkitConfig] = None,
    ) -> None:
        self.analysis = analysis_result
        self.df = df
        self.config = config or default_config

        # Extract commonly used sections
        self.metadata = analysis_result.get("metadata")
        self.duplicates = analysis_result.get("duplicates")
        self.infinite = analysis_result.get("infinite")
        self.missing = analysis_result.get("missing")
        self.outliers: List[OutlierReport] = analysis_result.get("outliers", [])
        self.feature_profiles: List[FeatureProfile] = analysis_result.get("feature_profiles", [])
        self.correlation_pairs: List[CorrelationPair] = analysis_result.get("correlation_pairs", [])
        self.target_profile: Optional[TargetProfile] = analysis_result.get("target_profile")
        self.recommendations: Dict[str, Any] = analysis_result.get("recommendations", {})
        self.quality_score = analysis_result.get("data_quality_score", 0.0)
        self.quality_notes: List[str] = analysis_result.get("data_quality_notes", [])

    # ------------------------------------------------------------------
    # Text report (plain text)
    # ------------------------------------------------------------------
    def generate_text(self) -> str:
        """Generate a plain‑text EDA report.

        Returns
        -------
        str
            Formatted plain‑text report.
        """
        lines = []
        lines.append("=" * 70)
        lines.append("                    ML TOOLKIT – EDA REPORT")
        lines.append("=" * 70)
        lines.append("")

        # Dataset overview
        if self.metadata:
            lines.append("[Dataset Overview]")
            lines.append(f"  Rows: {self.metadata.n_rows}")
            lines.append(f"  Columns: {self.metadata.n_columns}")
            lines.append(f"  Memory: {self.metadata.memory_mb:.2f} MB")
            lines.append("")

        # Data quality
        lines.append("[Data Quality]")
        lines.append(f"  Score: {self.quality_score:.1f}/100")
        for note in self.quality_notes:
            lines.append(f"  • {note}")
        lines.append("")

        # Duplicates
        if self.duplicates:
            lines.append("[Duplicates]")
            lines.append(f"  Total duplicates: {self.duplicates.total_duplicates} "
                         f"({self.duplicates.duplicate_percent:.2f}%)")
            lines.append("")

        # Infinite
        if self.infinite and self.infinite.columns_with_inf:
            lines.append("[Infinite Values]")
            for col, cnt in self.infinite.counts.items():
                lines.append(f"  {col}: {cnt} infinite values")
            lines.append("")

        # Missing values
        if self.missing and self.missing.total_missing > 0:
            lines.append("[Missing Values]")
            lines.append(f"  Total missing cells: {self.missing.total_missing}")
            lines.append("  Top columns:")
            top_miss = sorted(
                self.missing.column_reports,
                key=lambda x: x.missing_percent, reverse=True
            )[:5]
            for col_rpt in top_miss:
                lines.append(f"    {col_rpt.column}: {col_rpt.missing_percent:.2f}%")
            lines.append("")

        # Outliers
        if self.outliers:
            outlier_cols = [o for o in self.outliers if o.outlier_count > 0]
            if outlier_cols:
                lines.append("[Outliers (IQR)]")
                for o in outlier_cols[:10]:
                    lines.append(f"  {o.column}: {o.outlier_count} ({o.outlier_percent:.2f}%)")
                if len(outlier_cols) > 10:
                    lines.append(f"  ... and {len(outlier_cols)-10} more columns.")
                lines.append("")

        # Feature profiles summary
        if self.feature_profiles:
            lines.append("[Feature Profiles Summary]")
            const = [p.column for p in self.feature_profiles if p.is_constant]
            quasi = [p.column for p in self.feature_profiles if p.is_quasi_constant]
            if const:
                lines.append(f"  Constant columns: {', '.join(const)}")
            if quasi:
                lines.append(f"  Quasi-constant columns: {', '.join(quasi)}")
            lines.append("")

        # Correlations
        if self.correlation_pairs:
            lines.append("[Highly Correlated Feature Pairs]")
            for pair in self.correlation_pairs[:10]:
                lines.append(f"  {pair.feature_a} vs {pair.feature_b}: r={pair.coefficient:.2f}")
            if len(self.correlation_pairs) > 10:
                lines.append(f"  ... and {len(self.correlation_pairs)-10} more pairs.")
            lines.append("")

        # Recommendations summary
        if self.recommendations:
            lines.append("[Recommendations Summary]")
            for category in ["imputation", "outlier_handling", "transformation",
                             "scaling", "encoding", "feature_engineering", "feature_selection"]:
                recs = self.recommendations.get(category, [])
                if isinstance(recs, list):
                    for r in recs[:3]:
                        lines.append(f"  [{category}] {r.action}")
                elif hasattr(recs, 'action'):  # single Recommendation for scaling
                    lines.append(f"  [{category}] {recs.action}")
            lines.append("")

        # Models
        model_recs = self.recommendations.get("models", [])
        if model_recs:
            lines.append("[Recommended Baseline Models]")
            for m in model_recs:
                lines.append(f"  {m.model_name} ({m.suitability})")
            lines.append("")

        lines.append("=" * 70)
        lines.append("END OF REPORT")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------------------
    def generate_markdown(self) -> str:
        """Generate a Markdown formatted EDA report.

        Returns
        -------
        str
            Markdown string.
        """
        md = []
        md.append("# ML Toolkit – EDA Report\n")

        # Dataset overview
        if self.metadata:
            md.append("## Dataset Overview\n")
            md.append(f"- **Rows:** {self.metadata.n_rows}")
            md.append(f"- **Columns:** {self.metadata.n_columns}")
            md.append(f"- **Memory:** {self.metadata.memory_mb:.2f} MB\n")

        # Data quality
        md.append("## Data Quality\n")
        score_color = "green" if self.quality_score >= 70 else "orange" if self.quality_score >= 40 else "red"
        md.append(f"**Score:** <span style='color:{score_color};font-size:1.5em;'>{self.quality_score:.1f}/100</span>\n")
        if self.quality_notes:
            for note in self.quality_notes:
                md.append(f"- {note}")
        md.append("")

        # Duplicates
        if self.duplicates:
            md.append("## Duplicates\n")
            md.append(f"Total duplicate rows: {self.duplicates.total_duplicates} "
                      f"({self.duplicates.duplicate_percent:.2f}%)\n")

        # Infinite
        if self.infinite and self.infinite.columns_with_inf:
            md.append("## Infinite Values\n")
            md.append("| Column | Count |")
            md.append("|--------|-------|")
            for col, cnt in self.infinite.counts.items():
                md.append(f"| {col} | {cnt} |")
            md.append("")

        # Missing values
        if self.missing and self.missing.total_missing > 0:
            md.append("## Missing Values\n")
            md.append(f"Total missing cells: {self.missing.total_missing}\n")
            md.append("| Column | Missing Count | Missing % |")
            md.append("|--------|---------------|-----------|")
            for col_rpt in self.missing.column_reports:
                md.append(f"| {col_rpt.column} | {col_rpt.missing_count} | "
                          f"{col_rpt.missing_percent:.2f}% |")
            md.append("")

        # Outliers
        if self.outliers:
            outlier_cols = [o for o in self.outliers if o.outlier_count > 0]
            if outlier_cols:
                md.append("## Outliers (IQR)\n")
                md.append("| Column | Outlier Count | Outlier % |")
                md.append("|--------|---------------|-----------|")
                for o in outlier_cols:
                    md.append(f"| {o.column} | {o.outlier_count} | {o.outlier_percent:.2f}% |")
                md.append("")

        # Feature profiles
        if self.feature_profiles:
            const = [p.column for p in self.feature_profiles if p.is_constant]
            quasi = [p.column for p in self.feature_profiles if p.is_quasi_constant]
            if const or quasi:
                md.append("## Feature Status\n")
                if const:
                    md.append(f"**Constant columns:** {', '.join(const)}")
                if quasi:
                    md.append(f"**Quasi-constant columns:** {', '.join(quasi)}")
                md.append("")

        # Correlations
        if self.correlation_pairs:
            md.append("## Highly Correlated Pairs\n")
            md.append("| Feature A | Feature B | Coefficient |")
            md.append("|-----------|-----------|-------------|")
            for pair in self.correlation_pairs:
                md.append(f"| {pair.feature_a} | {pair.feature_b} | {pair.coefficient:.2f} |")
            md.append("")

        # Recommendations
        if self.recommendations:
            md.append("## Recommendations\n")
            for cat in ["imputation", "outlier_handling", "transformation",
                        "scaling", "encoding", "feature_engineering", "feature_selection"]:
                recs = self.recommendations.get(cat, [])
                if isinstance(recs, list) and recs:
                    md.append(f"### {cat.replace('_', ' ').title()}\n")
                    for r in recs:
                        md.append(f"- **{r.action}** (confidence: {r.confidence:.2f})")
                elif hasattr(recs, 'action'):
                    md.append(f"### {cat.replace('_', ' ').title()}\n")
                    md.append(f"- **{recs.action}** (confidence: {recs.confidence:.2f})")
            md.append("")

        # Models
        model_recs = self.recommendations.get("models", [])
        if model_recs:
            md.append("## Recommended Models\n")
            for m in model_recs:
                md.append(f"- **{m.model_name}** ({m.suitability}): {m.reason}")
            md.append("")

        return "\n".join(md)

    # ------------------------------------------------------------------
    # HTML report (with optional plots)
    # ------------------------------------------------------------------
    def generate_html(self, embed_plots: bool = True) -> str:
        """Generate a self‑contained HTML EDA report.

        Parameters
        ----------
        embed_plots : bool
            If True and *df* was provided, embed base64‑encoded
            visualisations. Plots are generated only for the first
            ``max_plot_cols`` numeric columns.

        Returns
        -------
        str
            Complete HTML document as a string.
        """
        html_parts = []
        html_parts.append("<!DOCTYPE html>")
        html_parts.append("<html><head><meta charset='utf-8'><title>EDA Report</title>")
        html_parts.append(f"<style>{_REPORT_CSS}</style></head><body>")

        # Title & quality score
        html_parts.append("<h1>ML Toolkit – EDA Report</h1>")
        score_class = "good" if self.quality_score >= 70 else "warning"
        html_parts.append(
            f"<p>Data Quality Score: <span class='quality-score {score_class}'>"
            f"{self.quality_score:.1f}</span>/100</p>"
        )

        # Dataset metadata
        if self.metadata:
            html_parts.append("<h2>Dataset Overview</h2>")
            html_parts.append("<ul>")
            html_parts.append(f"<li><strong>Rows:</strong> {self.metadata.n_rows}</li>")
            html_parts.append(f"<li><strong>Columns:</strong> {self.metadata.n_columns}</li>")
            html_parts.append(f"<li><strong>Memory:</strong> {self.metadata.memory_mb:.2f} MB</li>")
            html_parts.append("</ul>")

        # Quality notes
        if self.quality_notes:
            html_parts.append("<h2>Quality Notes</h2>")
            html_parts.append("<ul>")
            for note in self.quality_notes:
                html_parts.append(f"<li>{note}</li>")
            html_parts.append("</ul>")

        # Duplicates
        if self.duplicates and self.duplicates.total_duplicates > 0:
            html_parts.append("<h2>Duplicates</h2>")
            html_parts.append(
                f"<p>Total duplicates: {self.duplicates.total_duplicates} "
                f"({self.duplicates.duplicate_percent:.2f}%)</p>"
            )

        # Infinite
        if self.infinite and self.infinite.columns_with_inf:
            html_parts.append("<h2>Infinite Values</h2>")
            html_parts.append("<table><tr><th>Column</th><th>Count</th></tr>")
            for col, cnt in self.infinite.counts.items():
                html_parts.append(f"<tr><td>{col}</td><td>{cnt}</td></tr>")
            html_parts.append("</table>")

        # Missing values
        if self.missing and self.missing.total_missing > 0:
            html_parts.append("<h2>Missing Values</h2>")
            html_parts.append(
                f"<p>Total missing cells: {self.missing.total_missing}</p>"
            )
            html_parts.append(
                "<table><tr><th>Column</th><th>Missing Count</th><th>Missing %</th></tr>"
            )
            for col_rpt in self.missing.column_reports:
                html_parts.append(
                    f"<tr><td>{col_rpt.column}</td><td>{col_rpt.missing_count}</td>"
                    f"<td>{col_rpt.missing_percent:.2f}%</td></tr>"
                )
            html_parts.append("</table>")

        # Outliers
        if self.outliers:
            outlier_cols = [o for o in self.outliers if o.outlier_count > 0]
            if outlier_cols:
                html_parts.append("<h2>Outliers (IQR)</h2>")
                html_parts.append(
                    "<table><tr><th>Column</th><th>Outlier Count</th><th>Outlier %</th></tr>"
                )
                for o in outlier_cols:
                    html_parts.append(
                        f"<tr><td>{o.column}</td><td>{o.outlier_count}</td>"
                        f"<td>{o.outlier_percent:.2f}%</td></tr>"
                    )
                html_parts.append("</table>")

        # Feature status
        if self.feature_profiles:
            const = [p.column for p in self.feature_profiles if p.is_constant]
            quasi = [p.column for p in self.feature_profiles if p.is_quasi_constant]
            if const or quasi:
                html_parts.append("<h2>Feature Status</h2>")
                if const:
                    html_parts.append(f"<p><strong>Constant columns:</strong> {', '.join(const)}</p>")
                if quasi:
                    html_parts.append(f"<p><strong>Quasi-constant columns:</strong> {', '.join(quasi)}</p>")

        # Correlations
        if self.correlation_pairs:
            html_parts.append("<h2>Highly Correlated Pairs</h2>")
            html_parts.append(
                "<table><tr><th>Feature A</th><th>Feature B</th><th>Coefficient</th></tr>"
            )
            for pair in self.correlation_pairs:
                html_parts.append(
                    f"<tr><td>{pair.feature_a}</td><td>{pair.feature_b}</td>"
                    f"<td>{pair.coefficient:.2f}</td></tr>"
                )
            html_parts.append("</table>")

        # Recommendations
        if self.recommendations:
            html_parts.append("<h2>Recommendations</h2>")
            for cat in ["imputation", "outlier_handling", "transformation",
                        "scaling", "encoding", "feature_engineering", "feature_selection"]:
                recs = self.recommendations.get(cat, [])
                if isinstance(recs, list) and recs:
                    html_parts.append(f"<h3>{cat.replace('_', ' ').title()}</h3><ul>")
                    for r in recs:
                        html_parts.append(f"<li><strong>{r.action}</strong> (confidence: {r.confidence:.2f})</li>")
                    html_parts.append("</ul>")
                elif hasattr(recs, 'action'):
                    html_parts.append(f"<h3>{cat.replace('_', ' ').title()}</h3><ul>")
                    html_parts.append(f"<li><strong>{recs.action}</strong> (confidence: {recs.confidence:.2f})</li></ul>")

        # Model recommendations
        model_recs = self.recommendations.get("models", [])
        if model_recs:
            html_parts.append("<h2>Recommended Models</h2><ul>")
            for m in model_recs:
                html_parts.append(f"<li><strong>{m.model_name}</strong> ({m.suitability}): {m.reason}</li>")
            html_parts.append("</ul>")

        # Embedded visualisations (if requested and DataFrame available)
        if embed_plots and self.df is not None:
            html_parts.append("<h2>Visualisations</h2>")
            # Missing heatmap
            try:
                fig = plot_missing_heatmap(self.df, config=self.config)
                if fig is not None:
                    html_parts.append("<h3>Missing Values Heatmap</h3>")
                    html_parts.append(_fig_to_html(fig))
            except Exception:
                pass  # don't break entire report for a single plot

            # Numeric distributions (sample up to max_plot_cols)
            try:
                fig = plot_numeric_distributions(
                    self.df, self.analysis,
                    max_cols=self.config.max_plot_cols,
                    config=self.config,
                )
                if fig is not None:
                    html_parts.append("<h3>Numeric Distributions</h3>")
                    html_parts.append(_fig_to_html(fig))
            except Exception:
                pass

            # Outlier summary
            try:
                fig = plot_outlier_summary(self.analysis, config=self.config)
                if fig is not None:
                    html_parts.append("<h3>Outlier Summary</h3>")
                    html_parts.append(_fig_to_html(fig))
            except Exception:
                pass

            # Correlation heatmap
            try:
                fig = plot_correlation_heatmap(self.df, self.analysis, config=self.config)
                if fig is not None:
                    html_parts.append("<h3>Correlation Heatmap</h3>")
                    html_parts.append(_fig_to_html(fig))
            except Exception:
                pass

            # Target plots
            if self.target_profile is not None:
                try:
                    fig = plot_target_distribution(self.df, self.analysis, config=self.config)
                    if fig is not None:
                        html_parts.append("<h3>Target Distribution</h3>")
                        html_parts.append(_fig_to_html(fig))
                except Exception:
                    pass
                if self.target_profile.is_regression:
                    try:
                        fig = plot_target_correlations(self.df, self.analysis, config=self.config)
                        if fig is not None:
                            html_parts.append("<h3>Feature Correlations with Target</h3>")
                            html_parts.append(_fig_to_html(fig))
                    except Exception:
                        pass

        html_parts.append("</body></html>")
        return "\n".join(html_parts)

    # ------------------------------------------------------------------
    # Save reports to files
    # ------------------------------------------------------------------
    def save_report(self, filepath: str, format: str = "html", embed_plots: bool = True) -> None:
        """Generate a report and save it to a file.

        Parameters
        ----------
        filepath : str
            Path to the output file (extension will be forced to match format
            if not already correct).
        format : str
            One of ``'html'``, ``'md'``, ``'txt'``.
        embed_plots : bool
            Only relevant for HTML; see :meth:`generate_html`.

        Raises
        ------
        ReportError
            If the format is unsupported.
        """
        format = format.lower()
        if format == "html":
            content = self.generate_html(embed_plots=embed_plots)
        elif format in ("md", "markdown"):
            content = self.generate_markdown()
            if not filepath.endswith(".md"):
                filepath += ".md"
        elif format in ("txt", "text"):
            content = self.generate_text()
            if not filepath.endswith(".txt"):
                filepath += ".txt"
        else:
            raise ReportError(f"Unsupported report format: '{format}'.")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)