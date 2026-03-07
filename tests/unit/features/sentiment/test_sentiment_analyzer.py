import importlib
import sys

import pandas as pd
import pytest


@pytest.fixture
def analyzer_module():
    sys.modules.pop("src.features.sentiment.analyzer", None)
    importlib.invalidate_caches()
    module = importlib.import_module("src.features.sentiment.analyzer")
    return importlib.reload(module)


@pytest.fixture
def analyzer(analyzer_module):
    return analyzer_module.SentimentAnalyzer()


def test_analyze_text_requires_string(analyzer):
    with pytest.raises(TypeError):
        analyzer.analyze_text(123)


def test_analyze_returns_positive_label(analyzer):
    result = analyzer.analyze("公司业绩优秀，增长强劲")
    assert result["label"] == "positive"
    assert result["score"] == pytest.approx(0.7)


def test_analyze_handles_negative(analyzer):
    result = analyzer.analyze("市场下跌，亏损严重")
    assert result["label"] == "negative"
    assert result["score"] == pytest.approx(-0.3)


def test_batch_analyze_validates_inputs(analyzer):
    with pytest.raises(TypeError):
        analyzer.batch_analyze("not-list")
    with pytest.raises(TypeError):
        analyzer.batch_analyze(["ok", 123])


def test_batch_analyze_returns_results(analyzer):
    texts = ["上涨趋势积极", "下跌带来亏损"]
    results = analyzer.batch_analyze(texts)
    assert len(results) == 2
    assert results[0]["label"] == "positive"
    assert results[1]["label"] == "negative"


def test_generate_features_requires_column(analyzer):
    data = pd.DataFrame({"title": ["正面消息"]})
    with pytest.raises(ValueError):
        analyzer.generate_features(data, text_col="content")


def test_generate_features_returns_dataframe(analyzer):
    data = pd.DataFrame(
        {
            "content": [
                "公司表现优秀，增长率高",
                "市场下跌，负面情绪蔓延",
                "",
            ]
        }
    )
    features = analyzer.generate_features(data, text_col="content")
    assert list(features.columns) == ["sentiment_score", "sentiment_label"]
    assert len(features) == 3
    assert "positive" in features["sentiment_label"].values
    assert "negative" in features["sentiment_label"].values


def test_register_features_invokes_feature_manager(monkeypatch, analyzer):
    captured = {}

    class DummyManager:
        def register(self, **kwargs):
            captured.update(kwargs)

    analyzer.feature_manager = DummyManager()
    analyzer.register_features()
    assert captured["name"] == "sentiment"

