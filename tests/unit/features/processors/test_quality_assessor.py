import pandas as pd
import pytest

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from src.features.core.config import FeatureRegistrationConfig, FeatureType
from src.features.processors.quality_assessor import (
    FeatureQualityAssessor,
    AssessmentConfig,
)


@pytest.fixture
def regression_data():
    X, y = make_regression(
        n_samples=200, n_features=4, noise=0.1, random_state=42
    )
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    features = pd.DataFrame(X, columns=feature_names)
    target = pd.Series(y, name="target")
    return features, target, feature_names


@pytest.fixture
def assessor():
    config = AssessmentConfig(
        use_mutual_info=False,  # 避免在小样本下随机波动过大
        use_random_forest=True,
        n_estimators=50,
        random_state=42,
    )
    return FeatureQualityAssessor(config)


def test_assess_feature_quality_produces_scores(assessor, regression_data):
    features, target, feature_names = regression_data
    feature_configs = [
        FeatureRegistrationConfig(
            name=name,
            feature_type=FeatureType.TECHNICAL,
            params={},
            dependencies=[],
        )
        for name in feature_names
    ]

    report = assessor.assess_feature_quality(features, target, feature_configs)

    assert "feature_scores" in report
    assert "summary" in report
    assert set(report["feature_scores"].keys()) == set(feature_names)
    for metrics in report["feature_scores"].values():
        assert 0.0 <= metrics.overall_score <= 1.0


def test_get_quality_report(after_assessment):
    assessor, feature_names = after_assessment
    report = assessor.get_quality_report()

    assert report["summary"]["total_features"] == len(feature_names)
    assert len(report["top_features"]) > 0


@pytest.fixture
def after_assessment(assessor, regression_data):
    features, target, feature_names = regression_data
    assessor.assess_feature_quality(features, target)
    return assessor, feature_names


def test_filter_features(after_assessment, regression_data):
    assessor, feature_names = after_assessment
    features, _, _ = regression_data

    filtered = assessor.filter_features(features, min_score=0.4)
    assert set(filtered.columns).issubset(set(feature_names))


def test_generate_recommendations(after_assessment):
    assessor, _ = after_assessment
    recommendations = assessor._generate_recommendations()
    assert isinstance(recommendations, list)


def test_assess_feature_quality_handles_empty():
    assessor = FeatureQualityAssessor()
    features = pd.DataFrame()
    with pytest.raises(ValueError):
        assessor.assess_feature_quality(features, pd.Series(dtype=float))


def test_assess_feature_quality_handles_importance_failure(monkeypatch, regression_data, caplog):
    features, target, feature_names = regression_data
    assessor = FeatureQualityAssessor(AssessmentConfig(use_mutual_info=False, use_random_forest=True))

    class BoomRF:
        def fit(self, *_args, **_kwargs):
            raise RuntimeError("rf fail")

        @property
        def feature_importances_(self):
            return [0.0] * features.shape[1]

    monkeypatch.setattr(
        "src.features.processors.quality_assessor.RandomForestRegressor",
        lambda *args, **kwargs: BoomRF(),
    )
    monkeypatch.setattr(
        "src.features.processors.quality_assessor.RandomForestClassifier",
        lambda *args, **kwargs: BoomRF(),
    )
    with caplog.at_level("ERROR"):
        results = assessor.assess_feature_quality(features, target)
    assert results
    assert any("计算特征重要性失败" in message for message in caplog.messages)


def test_filter_features_no_assessment_warns(regression_data, caplog):
    features, _, _ = regression_data
    assessor = FeatureQualityAssessor()
    with caplog.at_level("WARNING"):
        filtered = assessor.filter_features(features, min_score=0.9)
    pd.testing.assert_frame_equal(filtered, features)
    assert any("未进行质量评估" in message for message in caplog.messages)

