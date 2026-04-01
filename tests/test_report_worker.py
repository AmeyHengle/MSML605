"""Failing test stubs for report_worker (ANALYSIS-01/02/04/05/06).

All tests raise pytest.fail() — RED phase.
Implementation in Plan 03-02 (Wave 1/2).
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# ANALYSIS-01: SHAP computation
# ---------------------------------------------------------------------------


def test_shap_computed_for_tree_model() -> None:
    """report_worker computes SHAP values for a RandomForestRegressor and stores non-None shap_top_features in state."""
    pytest.fail("not implemented — Wave 1/2 plan implements this")


def test_shap_fallback_for_ridge() -> None:
    """report_worker falls back gracefully when shap.Explainer raises for a Ridge model; report_path is still returned."""
    pytest.fail("not implemented — Wave 1/2 plan implements this")


def test_shap_artifact_logged() -> None:
    """report_worker calls mlflow.log_artifact with a path ending in .png (SHAP bar chart)."""
    pytest.fail("not implemented — Wave 1/2 plan implements this")


# ---------------------------------------------------------------------------
# ANALYSIS-02: HTML report sections
# ---------------------------------------------------------------------------


def test_html_sections_present() -> None:
    """Rendered HTML output contains all 7 section headings (Performance Metrics, Forecast vs. Actual, Feature Importance, Drift Analysis, Model Comparison, Pipeline Summary, report header)."""
    pytest.fail("not implemented — Wave 1/2 plan implements this")


# ---------------------------------------------------------------------------
# ANALYSIS-04: Forecast chart embedded as base64
# ---------------------------------------------------------------------------


def test_forecast_chart_embedded() -> None:
    """Rendered HTML contains 'data:image/png;base64,' (forecast chart embedded inline)."""
    pytest.fail("not implemented — Wave 1/2 plan implements this")


# ---------------------------------------------------------------------------
# ANALYSIS-05: LLM summary (Groq)
# ---------------------------------------------------------------------------


def test_llm_summary_included() -> None:
    """Mocked Groq client response text appears in the rendered HTML report."""
    pytest.fail("not implemented — Wave 1/2 plan implements this")


def test_llm_fallback_on_error() -> None:
    """When Groq raises groq.APIConnectionError, fallback text appears in HTML and pipeline does not crash."""
    pytest.fail("not implemented — Wave 1/2 plan implements this")


# ---------------------------------------------------------------------------
# ANALYSIS-06: Report file output
# ---------------------------------------------------------------------------


def test_report_path_returned() -> None:
    """report_worker returns a dict with 'report_path' key pointing to an existing .html file."""
    pytest.fail("not implemented — Wave 1/2 plan implements this")


def test_report_artifact_logged() -> None:
    """report_worker calls mlflow.log_artifact with a path ending in .html."""
    pytest.fail("not implemented — Wave 1/2 plan implements this")
