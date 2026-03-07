#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pytest

from src.features.quality_assessor import QualityAssessor, QualityAssessorConfig


def _build_sample_frame():
    return pd.DataFrame(
        {
            "stable_col": [5, 5, 5, 5, 5],
            "with_missing": [1.0, 2.0, np.nan, 4.0, 5.0],
            "with_outlier": [0.0, 0.2, 0.1, 0.3, 100.0],
            "category": ["a", "b", None, "b", "b"],
        }
    )


def test_assess_quality_reports_issues_and_score(monkeypatch):
    assessor = QualityAssessor(QualityAssessorConfig(outlier_zscore=1.0))
    fake_detail = {
        "quality_scores": {"stable_col": 0.8, "with_missing": 0.7},
        "comprehensive_report": {"summary": {"dummy": 1}},
    }
    monkeypatch.setattr(assessor.feature_assessor, "assess_feature_quality", lambda df: fake_detail)

    raw_issues = assessor._detect_issues(_build_sample_frame())
    assert any("异常值比例" in issue for issue in raw_issues)

    result = assessor.assess_quality(_build_sample_frame())

    assert pytest.approx(result["score"], rel=1e-3) == 0.75
    assert "quality_scores" in result and result["quality_scores"] == fake_detail["quality_scores"]
    for expected in ("缺失值", "常量列", "异常值比例"):
        assert any(expected in issue for issue in result["issues"])


def test_improve_quality_resolves_missing_and_clips():
    config = QualityAssessorConfig(missing_value_strategy="mean", clip_quantiles=0.2)
    assessor = QualityAssessor(config)

    improved = assessor.improve_quality(_build_sample_frame())

    assert improved.isnull().sum().sum() == 0
    original = _build_sample_frame()
    upper_bound = original["with_outlier"].quantile(0.8)
    assert improved["with_outlier"].max() <= upper_bound + 1e-6
    assert set(improved["category"].unique()) == {"a", "b"}


def test_ensure_dataframe_rejects_invalid_type():
    assessor = QualityAssessor()
    with pytest.raises(TypeError):
        assessor._ensure_dataframe([1, 2, 3])  # type: ignore[arg-type]


def test_improve_quality_zero_strategy_handles_numeric_and_categorical():
    config = QualityAssessorConfig(missing_value_strategy="zero", clip_quantiles=0.0)
    assessor = QualityAssessor(config)

    improved = assessor.improve_quality(_build_sample_frame())

    assert 0.0 in improved["with_missing"].values
    assert 0.0 in improved["with_outlier"].values
    assert improved["category"].isnull().sum() == 0


def test_ensure_dataframe_accepts_series_and_dataframe():
    assessor = QualityAssessor()
    series = pd.Series([1, 2, 3], name="metric")
    dataframe = pd.DataFrame({"col": [10, 20]})

    converted_series = assessor._ensure_dataframe(series)
    converted_df = assessor._ensure_dataframe(dataframe)

    assert list(converted_series.columns) == ["metric"]
    assert converted_df.equals(dataframe)
