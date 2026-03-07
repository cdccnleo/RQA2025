#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征质量评估器的边界与质量保障测试

聚焦于架构设计文档强调的“特征质量评估 / 自适应优化”能力，
补充空数据、批量评估、报告导出与稳定性等关键分支的覆盖率。
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from src.features.processors.feature_quality_assessor import FeatureQualityAssessor


def _build_simple_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "alpha": [1.0, 2.0, 3.0, 4.0],
            "beta": [4.0, 3.0, 2.0, 1.0],
            "gamma": [10.0, 15.0, 10.0, 5.0],
        }
    )


def test_assess_feature_quality_with_empty_dataframe_returns_empty_report():
    assessor = FeatureQualityAssessor()

    result = assessor.assess_feature_quality(pd.DataFrame())

    assert result["quality_scores"] == {}
    assert result["comprehensive_report"]["summary"] == {}


def test_evaluate_and_batch_evaluate_share_pipeline():
    assessor = FeatureQualityAssessor()
    frame = _build_simple_frame()
    target = frame.sum(axis=1)

    single = assessor.evaluate_feature(frame["alpha"])
    batch = assessor.batch_evaluate(frame, target)

    assert "quality_scores" in single and single["quality_scores"]
    assert set(batch.keys()) == set(frame.columns)
    assert all("comprehensive_report" in value for value in batch.values())


def test_recommendations_and_summary_without_scores():
    assessor = FeatureQualityAssessor()

    recommendations = assessor.get_feature_recommendations()
    summary = assessor.get_feature_quality_summary()

    assert recommendations == {"keep": [], "improve": [], "remove": []}
    assert summary == {}


def test_export_quality_report_without_scores_emits_warning(tmp_path, caplog):
    assessor = FeatureQualityAssessor()
    destination = tmp_path / "empty.json"

    with caplog.at_level("WARNING"):
        assessor.export_quality_report(destination.as_posix())

    assert not destination.exists()
    assert "跳过导出" in caplog.text


def test_export_quality_report_persists_scores(tmp_path):
    assessor = FeatureQualityAssessor()
    assessor.quality_scores = {"alpha": 0.9, "beta": 0.1}
    destination = tmp_path / "quality.json"

    assessor.export_quality_report(destination.as_posix())

    assert destination.exists()
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["quality_scores"] == assessor.quality_scores
    assert payload["quality_summary"]["total_features"] == 2


def test_top_and_low_quality_helpers_use_cached_scores():
    assessor = FeatureQualityAssessor()
    assessor.quality_scores = {"good": 0.95, "medium": 0.55, "bad": 0.2}

    assert assessor.get_top_features(1)[0][0] == "good"
    assert ("bad", pytest.approx(0.2)) in assessor.get_low_quality_features(0.5)


def test_get_feature_recommendations_with_scores():
    assessor = FeatureQualityAssessor()
    assessor.quality_scores = {"strong": 0.9, "medium": 0.65, "weak": 0.2}

    recommendations = assessor.get_feature_recommendations(threshold=0.8)

    assert recommendations["keep"] == ["strong"]
    assert recommendations["improve"] == ["medium"]
    assert recommendations["remove"] == ["weak"]


def test_ensure_dataframe_invalid_input_raises_type_error():
    assessor = FeatureQualityAssessor()
    with pytest.raises(TypeError):
        assessor._ensure_dataframe([1, 2, 3])  # type: ignore[arg-type]


def test_assess_stability_handles_empty_and_zero_mean_series():
    assessor = FeatureQualityAssessor()
    frame = pd.DataFrame(
        {
            "empty": [np.nan, np.nan, np.nan],
            "zero_mean": [1.0, -1.0, 0.0],
            "normal": [5.0, 6.0, 7.0],
        }
    )

    result = assessor._assess_stability(frame, None)
    stability = result["combined_stability"]

    assert stability["empty"] == 0.0
    assert pytest.approx(stability["zero_mean"], rel=1e-6) == 1.0


@pytest.mark.parametrize(
    "scores,expected",
    [
        ({}, {}),
        ({"only": 0.0}, {"only": 0.0}),
    ],
)
def test_normalize_scores_handles_empty_and_zero_values(scores, expected):
    assessor = FeatureQualityAssessor()
    assert assessor._normalize_scores(scores) == expected

