import configparser
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.features.core.config_integration import ConfigScope
from src.features.sentiment import sentiment_analyzer as sentiment_module


class DummyConfigManager:
    def __init__(self, config=None):
        self.config = config or {
            "use_bert": False,
            "bert_model_path": "default/path",
            "default_language": "zh",
            "batch_size": 16,
        }
        self.watchers = []
        self.requested_scopes = []

    def get_config(self, scope):
        self.requested_scopes.append(scope)
        return self.config

    def register_config_watcher(self, scope, callback):
        self.watchers.append((scope, callback))


@pytest.fixture
def analyzer(monkeypatch):
    manager = DummyConfigManager()
    monkeypatch.setattr(
        sentiment_module,
        "get_config_integration_manager",
        lambda: manager,
    )
    instance = sentiment_module.SentimentAnalyzer(skip_config=True)
    return instance, manager


def test_init_applies_config_and_registers_watcher(analyzer):
    instance, manager = analyzer
    assert manager.requested_scopes == [ConfigScope.SENTIMENT]
    assert manager.watchers
    assert instance.config.use_bert is False
    assert instance.config.batch_size == 16


def test_on_config_change_updates_attribute(analyzer):
    instance, manager = analyzer
    scope, callback = manager.watchers[0]
    callback(scope, "default_language", "zh", "en")
    assert instance.config.default_language == "en"


def test_clean_and_segment_text(analyzer):
    instance, _ = analyzer
    assert instance.clean_text(" 你好 \n") == "你好"
    assert instance.segment_text("a b c") == ["a", "b", "c"]
    assert instance.clean_text(None) == ""
    assert instance.segment_text("") == []


def test_detect_language(analyzer):
    instance, _ = analyzer
    assert instance.detect_language("增长强劲") == "zh"
    assert instance.detect_language("profit going up") == "en"
    assert instance.detect_language("") == "en"


def test_generate_features_missing_column(analyzer):
    instance, _ = analyzer
    with pytest.raises(ValueError):
        instance.generate_features(pd.DataFrame({"title": ["news"]}))


def test_generate_features_computes_average(monkeypatch, analyzer):
    instance, _ = analyzer
    monkeypatch.setattr(instance, "snownlp_sentiment", lambda text: 0.8)
    monkeypatch.setattr(instance, "textblob_sentiment", lambda text: 0.2)
    frame = pd.DataFrame({"content": ["good", "bad"]})
    features = instance.generate_features(frame)
    assert list(features.columns) == ["sentiment_score", "snownlp_score", "textblob_score"]
    assert features.iloc[0]["sentiment_score"] == pytest.approx(0.5)
    assert len(features) == 2


def test_generate_features_handles_empty(analyzer):
    instance, _ = analyzer
    assert instance.generate_features(pd.DataFrame()).empty


def test_snownlp_sentiment_fallback(monkeypatch, analyzer):
    import builtins

    instance, _ = analyzer

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "snownlp":
            raise ImportError("mock snownlp missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import, raising=False)
    assert instance.snownlp_sentiment("无法导入 snownlp") == pytest.approx(0.5)


def test_textblob_sentiment_fallback(monkeypatch, analyzer):
    import builtins

    instance, _ = analyzer

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "textblob":
            raise ImportError("mock textblob missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import, raising=False)
    assert instance.textblob_sentiment("no textblob module available") == pytest.approx(0.0)


def test_load_config_updates_fields(tmp_path, analyzer):
    instance, _ = analyzer
    config_path = tmp_path / "sentiment.ini"
    parser = configparser.ConfigParser()
    parser["Paths"] = {"bert_model": "models/bert.bin"}
    parser["Settings"] = {
        "use_bert": "false",
        "default_language": "en",
        "batch_size": "64",
    }
    with config_path.open("w", encoding="utf-8") as f:
        parser.write(f)

    instance._load_config(str(config_path))
    assert instance.config.bert_model_path == "models/bert.bin"
    assert instance.config.use_bert is False
    assert instance.config.default_language == "en"
    assert instance.config.batch_size == 64


def test_bert_batch_predict_without_model(analyzer):
    instance, _ = analyzer
    assert instance._bert_batch_predict(["a", "b"]) == [0.5, 0.5]


def test_bert_batch_predict_with_model(analyzer):
    instance, _ = analyzer
    instance.bert_model = object()
    instance.bert_tokenizer = object()
    assert instance._bert_batch_predict(["a"]) == [0.5]

