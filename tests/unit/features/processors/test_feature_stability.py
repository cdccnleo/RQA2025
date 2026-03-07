import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def stability_analyzer(monkeypatch):
    from src.features.processors.feature_stability import FeatureStabilityAnalyzer

    analyzer = FeatureStabilityAnalyzer(
        config={
            "stability_threshold": 0.8,
            "drift_threshold": 0.1,
            "time_window_size": 3,
            "min_samples": 6,
        }
    )
    return analyzer


@pytest.fixture
def time_series_features():
    dates = pd.date_range(datetime(2024, 1, 1), periods=12, freq="D")
    data = {
        "f1": np.linspace(0, 1, 12),
        "f2": np.sin(np.linspace(0, np.pi, 12)),
    }
    return pd.DataFrame(data, index=dates)


def test_analyze_feature_stability_with_time(stability_analyzer, time_series_features):
    results = stability_analyzer.analyze_feature_stability(
        time_series_features, time_series_features.index
    )

    combined = results["combined_stability"]
    assert set(combined.keys()) == {"f1", "f2"}
    assert results["analysis_report"]["summary"]["total_features"] == 2


def test_analyze_feature_stability_without_time(stability_analyzer, time_series_features):
    results = stability_analyzer.analyze_feature_stability(time_series_features)
    temporal_scores = results["analysis_results"]["temporal_stability"]
    assert temporal_scores["f1"] == 0.5


def test_get_stability_recommendations(stability_analyzer, time_series_features):
    stability_analyzer.analyze_feature_stability(
        time_series_features, time_series_features.index
    )
    recommendations = stability_analyzer.get_stability_recommendations()

    assert set(recommendations.keys()) == {"stable", "unstable", "monitor"}
    assert set(recommendations.keys()) == {"stable", "unstable", "monitor"}


def test_get_stability_summary(stability_analyzer, time_series_features):
    stability_analyzer.analyze_feature_stability(
        time_series_features, time_series_features.index
    )
    summary = stability_analyzer.analyze_feature_stability(
        time_series_features, time_series_features.index
    )["analysis_report"]["summary"]
    assert summary["total_features"] == 2


def test_detect_feature_drift_thresholds(stability_analyzer, time_series_features):
    results = stability_analyzer._detect_feature_drift(
        time_series_features, time_series_features.index
    )

    assert set(results["drift_severity"].keys()) == {"f1", "f2"}
    assert all(severity in {"low", "medium", "high", "unknown"} for severity in results["drift_severity"].values())

