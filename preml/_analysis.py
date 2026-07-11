"""Helpers for working with EDA analysis results."""

from __future__ import annotations

from typing import Any, Dict


def resolve_analysis_result(analysis_result: Any) -> Dict[str, Any]:
    """Return a dictionary analysis result from either a dict or analyzer object."""
    if isinstance(analysis_result, dict):
        return analysis_result

    run = getattr(analysis_result, "run", None)
    if callable(run):
        resolved = run()
        if isinstance(resolved, dict):
            return resolved

    raise TypeError(
        "analysis_result must be a dictionary or an object with a run() method returning a dictionary."
    )