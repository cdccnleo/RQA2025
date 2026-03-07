import sys
from importlib import import_module

import pandas as pd
import pytest

sys.modules.setdefault("src.sentiment", import_module("src.features.sentiment"))

from src.features.processors.sentiment import SentimentProcessor


class DummyAnalyzer:
    def __init__(self):
        self.process_calls = []
        self.analyze_calls = []
        self.batch_calls = []

    def generate_features(self, data, text_col="content", **kwargs):
        self.process_calls.append((data, text_col, kwargs))
        return pd.DataFrame({"score": [1.0]})

    def analyze(self, text, **kwargs):
        self.analyze_calls.append((text, kwargs))
        return {"score": 0.9}

    def batch_analyze(self, texts, **kwargs):
        self.batch_calls.append((texts, kwargs))
        return [{"score": 0.8} for _ in texts]


@pytest.fixture
def processor(monkeypatch):
    dummy = DummyAnalyzer()
    monkeypatch.setattr(
        "src.features.processors.sentiment.SentimentAnalyzer",
        lambda **kwargs: dummy,
    )
    return SentimentProcessor(), dummy


def test_processor_process_delegates_to_analyzer(processor):
    proc, dummy = processor
    df = pd.DataFrame({"content": ["积极表现"]})
    result = proc.process(df)
    assert not result.empty
    assert dummy.process_calls[0][0].equals(df)


def test_processor_analyze_sentiment(processor):
    proc, dummy = processor
    result = proc.analyze_sentiment("上涨趋势")
    assert result["score"] == 0.9
    assert dummy.analyze_calls[0][0] == "上涨趋势"


def test_processor_batch_analyze(processor):
    proc, dummy = processor
    texts = ["积极", "消极"]
    results = proc.batch_analyze(texts)
    assert len(results) == 2
    assert dummy.batch_calls[0][0] == texts

