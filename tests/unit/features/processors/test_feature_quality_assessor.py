import json
import sys
import types

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def stub_quality_dependencies(monkeypatch):
    features = ["f1", "f2"]

    class StubImportanceAnalyzer:
        def analyze_feature_importance(self, X, target, target_type):
            return {
                "combined_importance": {"f1": 0.8, "f2": 0.2},
                "feature_ranking": features,
                "analysis_report": {"summary": {"low_importance_features": 0}},
            }

    class StubCorrelationAnalyzer:
        def analyze_feature_correlation(self, X):
            return {
                "analysis_results": {"vif_analysis": {"f1": 0.1, "f2": 0.5}},
                "analysis_report": {
                    "summary": {"multicollinearity_groups": 0},
                    "feature_groups": [["f1"], ["f2"]],
                },
            }

    class StubStabilityAnalyzer:
        def analyze_feature_stability(self, X, time_index):
            return {
                "combined_stability": {"f1": 0.9, "f2": 0.4},
                "analysis_report": {
                    "summary": {"low_stability_features": 1},
                    "stability_ranking": ["f1", "f2"],
                },
            }

    modules = {
        "feature_importance": StubImportanceAnalyzer,
        "feature_correlation": StubCorrelationAnalyzer,
        "feature_stability": StubStabilityAnalyzer,
    }

    for name, cls in modules.items():
        module = types.ModuleType(name)
        analyzer_name = "".join(part.title() for part in name.split("_")) + "Analyzer"
        setattr(module, analyzer_name, cls)
        monkeypatch.setitem(sys.modules, name, module)

    yield

    for name in modules:
        sys.modules.pop(name, None)


@pytest.fixture
def quality_assessor():
    from src.features.processors.feature_quality_assessor import FeatureQualityAssessor

    return FeatureQualityAssessor()


@pytest.fixture
def sample_features():
    return pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [4.0, 3.0, 2.0, 1.0],
        }
    )


@pytest.fixture
def sample_target():
    return pd.Series([0.1, 0.2, 0.3, 0.4])


def test_assess_feature_quality_combines_scores(quality_assessor, sample_features, sample_target):
    results = quality_assessor.assess_feature_quality(sample_features, sample_target)

    scores = results["quality_scores"]
    assert "f1" in scores and "f2" in scores
    assert 0.0 <= scores["f1"] <= 1.0
    assert 0.0 <= scores["f2"] <= 1.0
    assert scores["f1"] > scores["f2"]
    assert quality_assessor.feature_rankings["quality"][0][0] == "f1"


def test_get_feature_recommendations(quality_assessor, sample_features, sample_target):
    quality_assessor.assess_feature_quality(sample_features, sample_target)
    recommendations = quality_assessor.get_feature_recommendations()

    assert "f2" in recommendations["remove"]
    assert "f1" not in recommendations["remove"]


def test_get_feature_quality_summary(quality_assessor, sample_features, sample_target):
    quality_assessor.assess_feature_quality(sample_features, sample_target)
    summary = quality_assessor.get_feature_quality_summary()

    assert summary["total_features"] == 2
    assert summary["max_quality"] >= summary["min_quality"]


def test_export_quality_report(tmp_path, quality_assessor, sample_features, sample_target):
    quality_assessor.assess_feature_quality(sample_features, sample_target)
    filepath = tmp_path / "report.json"

    quality_assessor.export_quality_report(filepath.as_posix())
    assert filepath.exists()

    content = json.loads(filepath.read_text(encoding="utf-8"))
    assert "quality_scores" in content
    assert "f1" in content["quality_scores"]


def test_get_top_features(quality_assessor, sample_features, sample_target):
    quality_assessor.assess_feature_quality(sample_features, sample_target)
    top = quality_assessor.get_top_features(1)
    assert len(top) == 1
    assert top[0][0] == "f1"

