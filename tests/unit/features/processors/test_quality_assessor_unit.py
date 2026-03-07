#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FeatureQualityAssessor 精准单测，聚焦评分与报告生成。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.processors.quality_assessor import (
    AssessmentConfig,
    FeatureQualityAssessor,
    QualityMetrics,
)

pytestmark = pytest.mark.features


@pytest.fixture(autouse=True)
def stub_sklearn(monkeypatch):
    class DummyForest:
        def __init__(self, *args, **kwargs):
            self.feature_importances_ = None

        def fit(self, X, y):
            cols = X.shape[1]
            self.feature_importances_ = np.array([1.0 / cols] * cols)
            return self

    def _mi_stub(X, y, random_state=None):
        return np.array([0.2] * X.shape[1])

    monkeypatch.setattr(
        "src.features.processors.quality_assessor.RandomForestRegressor", DummyForest
    )
    monkeypatch.setattr(
        "src.features.processors.quality_assessor.RandomForestClassifier", DummyForest
    )
    monkeypatch.setattr(
        "src.features.processors.quality_assessor.mutual_info_regression", _mi_stub
    )
    monkeypatch.setattr(
        "src.features.processors.quality_assessor.mutual_info_classi", _mi_stub
    )


@pytest.fixture()
def feature_dataset():
    features = pd.DataFrame(
        {
            "feature_a": np.linspace(0, 1, 60),
            "feature_b": np.linspace(1, 2, 60),
            "constant": 1.0,
        }
    )
    target = pd.Series(np.linspace(0, 1, 60))
    return features, target


def test_assess_feature_quality_generates_report(feature_dataset):
    features, target = feature_dataset
    assessor = FeatureQualityAssessor(AssessmentConfig(n_estimators=10))
    report = assessor.assess_feature_quality(features, target)

    assert "feature_scores" in report
    assert set(report["feature_scores"].keys()) == {"feature_a", "feature_b"}
    assert report["summary"]["total_features"] == 2

    quality_report = assessor.get_quality_report()
    assert quality_report["summary"]["total_features"] == 2
    assert quality_report["top_features"]


def test_preprocess_and_redundancy_detection(feature_dataset):
    features, _ = feature_dataset
    assessor = FeatureQualityAssessor()
    processed = assessor._preprocess_features(features)
    assert "constant" not in processed.columns

    redundant = assessor._identify_redundant_features(processed.assign(feature_c=processed["feature_a"]))
    assert "feature_c" in redundant


def test_generate_recommendations_from_metrics():
    assessor = FeatureQualityAssessor()
    assessor.quality_metrics = {
        "strong": QualityMetrics(importance_score=0.9, correlation_score=0.2, stability_score=0.8, information_score=0.7, redundancy_score=0.1, overall_score=0.9),
        "weak": QualityMetrics(importance_score=0.1, correlation_score=0.1, stability_score=0.2, information_score=0.2, redundancy_score=0.8, overall_score=0.2),
    }
    recommendations = assessor._generate_recommendations()
    assert recommendations
    assert any("特征" in rec for rec in recommendations)


def test_error_handling_on_invalid_inputs():
    assessor = FeatureQualityAssessor()
    with pytest.raises(ValueError):
        assessor.assess_feature_quality(pd.DataFrame(), pd.Series(dtype=float))


def test_assess_quality_raises_on_empty_target(feature_dataset):
    features, _ = feature_dataset
    assessor = FeatureQualityAssessor()
    with pytest.raises(ValueError):
        assessor.assess_feature_quality(features, pd.Series(dtype=float))


def test_assess_quality_raises_on_length_mismatch(feature_dataset):
    features, target = feature_dataset
    assessor = FeatureQualityAssessor()
    with pytest.raises(ValueError):
        assessor.assess_feature_quality(features, target.iloc[:-1])


def test_assess_quality_raises_when_no_features_after_preprocess():
    assessor = FeatureQualityAssessor()
    features = pd.DataFrame({"constant": [1, 1, 1]})
    target = pd.Series([1, 2, 3])
    with pytest.raises(ValueError):
        assessor.assess_feature_quality(features, target)


def test_filter_and_redundancy_helpers():
    assessor = FeatureQualityAssessor()
    assessor.quality_metrics = {
        "top": QualityMetrics(overall_score=0.9),
        "mid": QualityMetrics(overall_score=0.7),
        "low": QualityMetrics(overall_score=0.2),
    }
    features = pd.DataFrame({"top": [1, 2], "mid": [3, 4], "low": [5, 6]})
    filtered = assessor.filter_features(features, min_score=0.8)
    assert list(filtered.columns) == ["top"]

    redundant = assessor._identify_redundant_features(pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]}), threshold=0.9)
    assert "b" in redundant


def test_overall_score_computation():
    assessor = FeatureQualityAssessor()
    metrics = QualityMetrics(
        importance_score=0.8,
        correlation_score=0.6,
        stability_score=0.7,
        information_score=0.5,
        redundancy_score=0.2,
    )
    score = assessor._calculate_overall_score(metrics)
    assert 0.0 <= score <= 1.0


def test_filter_features_without_metrics_returns_original():
    assessor = FeatureQualityAssessor()
    df = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
    filtered = assessor.filter_features(df, min_score=0.9)
    pd.testing.assert_frame_equal(filtered, df)


def test_filter_features_respects_max_features():
    assessor = FeatureQualityAssessor()
    assessor.quality_metrics = {
        "f1": QualityMetrics(overall_score=0.9),
        "f2": QualityMetrics(overall_score=0.85),
        "f3": QualityMetrics(overall_score=0.8),
    }
    df = pd.DataFrame({"f1": [1, 2], "f2": [2, 3], "f3": [3, 4]})
    filtered = assessor.filter_features(df, min_score=0.7, max_features=2)
    assert list(filtered.columns) == ["f1", "f2"]


def test_filter_features_returns_original_when_columns_missing():
    assessor = FeatureQualityAssessor()
    assessor.quality_metrics = {"ghost": QualityMetrics(overall_score=0.95)}
    df = pd.DataFrame({"real": [1, 2]})
    filtered = assessor.filter_features(df, min_score=0.8)
    pd.testing.assert_frame_equal(filtered, df)


def test_calculate_redundancy_single_column():
    assessor = FeatureQualityAssessor()
    redundancy = assessor._calculate_redundancy(pd.DataFrame({"solo": [1, 2, 3]}))
    assert redundancy["solo"] == 0.0

