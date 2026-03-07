import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.processors.feature_quality_assessor import (
    FeatureQualityAssessor,
    FeatureQualityConfig,
)


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "price": [100, 101, 102, 103, 104],
            "volume": [1000, 1005, 1010, 1015, 1020],
            "momentum": [0.1, 0.15, 0.2, 0.25, 0.3],
        }
    )


def _sample_target() -> pd.Series:
    return pd.Series([1, 0, 1, 0, 1], name="signal")


def _build_assessor() -> FeatureQualityAssessor:
    config = FeatureQualityConfig(
        importance_weight=0.5,
        correlation_weight=0.3,
        stability_weight=0.2,
        quality_threshold=0.6,
    )
    return FeatureQualityAssessor(config)


def test_assess_feature_quality_updates_rankings_and_scores():
    assessor = _build_assessor()
    frame = _sample_frame()
    report = assessor.assess_feature_quality(frame, _sample_target())

    assert set(report["quality_scores"]) == set(frame.columns)
    assert assessor.quality_scores == report["quality_scores"]
    assert set(assessor.feature_rankings) == {"importance", "correlation", "stability", "quality"}
    assert assessor.feature_rankings["quality"][0][1] >= assessor.feature_rankings["quality"][-1][1]

    comprehensive = report["comprehensive_report"]["summary"]
    assert comprehensive["total_features"] == len(frame.columns)
    assert comprehensive["average_quality_score"] == pytest.approx(
        np.mean(list(report["quality_scores"].values()))
    )


def test_get_recommendations_and_summary(tmp_path: Path):
    assessor = _build_assessor()
    assessor.assess_feature_quality(_sample_frame(), _sample_target())

    recommendations = assessor.get_feature_recommendations(threshold=0.55)
    assert set(recommendations) == {"keep", "improve", "remove"}
    assert sum(len(v) for v in recommendations.values()) == len(_sample_frame().columns)

    summary = assessor.get_feature_quality_summary()
    assert summary["total_features"] == len(_sample_frame().columns)
    assert summary["min_quality"] <= summary["max_quality"]

    output_file = tmp_path / "quality_report.json"
    assessor.export_quality_report(str(output_file))
    data = json.loads(output_file.read_text(encoding="utf-8"))
    assert sorted(data.keys()) == ["quality_scores", "quality_summary", "recommendations"]


def test_batch_evaluate_and_series_support():
    assessor = _build_assessor()
    series = _sample_frame()["price"]

    single_report = assessor.evaluate_feature(series, _sample_target())
    assert single_report["quality_scores"]

    results = assessor.batch_evaluate(_sample_frame(), _sample_target())
    assert set(results) == set(_sample_frame().columns)
    assert all(report["quality_scores"] for report in results.values())


def test_top_and_low_quality_helpers_reflect_scores():
    assessor = _build_assessor()
    assessor.assess_feature_quality(_sample_frame(), _sample_target())

    top_features = assessor.get_top_features(n=2)
    assert len(top_features) == 2
    assert top_features[0][1] >= top_features[1][1]

    low_features = assessor.get_low_quality_features(threshold=0.9)
    assert all(score < 0.9 for _, score in low_features)


def test_empty_dataframe_returns_empty_report():
    assessor = _build_assessor()
    report = assessor.assess_feature_quality(pd.DataFrame(), _sample_target())
    assert report["quality_scores"] == {}
    assert report["comprehensive_report"]["summary"] == {}

