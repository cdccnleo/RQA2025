import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.processors.feature_correlation import FeatureCorrelationAnalyzer


@pytest.fixture(autouse=True)
def silence_logger(monkeypatch):
    monkeypatch.setattr(
        "src.features.processors.feature_correlation.logger",
        logging.getLogger(__name__),
    )


@pytest.fixture
def correlated_features():
    rng = np.random.default_rng(42)
    base = rng.normal(loc=0, scale=1, size=200)
    feature_1 = base + rng.normal(0, 0.01, size=200)
    feature_2 = base * 2 + 0.05
    feature_3 = rng.normal(0, 1, size=200)
    feature_4 = feature_3 * 0.9 + rng.normal(0, 0.02, size=200)
    return pd.DataFrame(
        {
            "f1": feature_1,
            "f2": feature_2,
            "f3": feature_3,
            "f4": feature_4,
        }
    )


def test_analyze_feature_correlation_outputs_expected_sections(correlated_features):
    analyzer = FeatureCorrelationAnalyzer(
        config={
            "correlation_threshold": 0.75,
            "vif_threshold": 5.0,
            "pca_variance_threshold": 0.9,
            "max_features": 10,
            "random_state": 0,
        }
    )

    results = analyzer.analyze_feature_correlation(correlated_features)
    analysis = results["analysis_results"]

    assert set(analysis.keys()) == {
        "correlation_matrix",
        "vif_analysis",
        "pca_analysis",
        "feature_selection_analysis",
        "multicollinearity_detection",
    }
    assert analyzer.correlation_matrix is not None
    assert analyzer.vif_scores
    assert analyzer.multicollinearity_groups


def test_get_feature_recommendations_tracks_vif_and_groups(correlated_features):
    analyzer = FeatureCorrelationAnalyzer(
        config={"correlation_threshold": 0.7, "vif_threshold": 5.0}
    )
    analyzer.analyze_feature_correlation(correlated_features)

    recommendations = analyzer.get_feature_recommendations()

    assert set(recommendations.keys()) == {"keep", "remove", "merge"}
    assert "f1" in recommendations["keep"] or "f2" in recommendations["remove"]


def test_plot_correlation_heatmap_uses_saved_path(monkeypatch, correlated_features, tmp_path):
    analyzer = FeatureCorrelationAnalyzer()
    analyzer.analyze_feature_correlation(correlated_features)

    heatmap_calls = {}
    class _DummyAxes:
        def set_title(self, *_, **__):
            return None

    class _DummyFigure:
        def gca(self):
            return _DummyAxes()

    monkeypatch.setattr("matplotlib.pyplot.figure", lambda *args, **kwargs: _DummyFigure())
    monkeypatch.setattr("matplotlib.pyplot.gca", lambda: _DummyAxes())
    monkeypatch.setattr("matplotlib.pyplot.close", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "matplotlib.pyplot.savefig",
        lambda path, *_, **__: Path(path).write_bytes(b"ok"),
    )
    monkeypatch.setattr(
        "seaborn.heatmap",
        lambda *args, **kwargs: heatmap_calls.setdefault("called", True),
    )
    monkeypatch.setattr("matplotlib.pyplot.tight_layout", lambda: None)
    captured_path = tmp_path / "corr.png"

    analyzer.plot_correlation_heatmap(save_path=str(captured_path))

    assert heatmap_calls.get("called", False) is True
    assert captured_path.exists()


def test_plot_correlation_heatmap_without_analysis(monkeypatch):
    analyzer = FeatureCorrelationAnalyzer()
    monkeypatch.setattr("matplotlib.pyplot.figure", lambda *args, **kwargs: None)

    # 不应抛异常，即便尚未分析
    analyzer.plot_correlation_heatmap()


def test_export_correlation_report(tmp_path, correlated_features):
    analyzer = FeatureCorrelationAnalyzer()
    analyzer.analyze_feature_correlation(correlated_features)

    export_path = tmp_path / "correlation_report.json"
    analyzer.export_correlation_report(str(export_path))

    assert export_path.exists()
    data = json.loads(export_path.read_text(encoding="utf-8"))
    assert "correlation_matrix" in data
    assert "vif_scores" in data

