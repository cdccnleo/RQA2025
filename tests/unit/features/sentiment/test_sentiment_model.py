from src.features.sentiment.models import load_pretrained_model
from src.features.sentiment.models.sentiment_model import SentimentModel


def test_analyze_sentiment_counts_positive_and_negative_words():
    model = SentimentModel({"model": "demo"})
    result = model.analyze_sentiment("这是一条利好消息，股价上涨")
    assert result["sentiment_score"] > 0
    assert result["positive_score"] >= 1
    assert result["negative_score"] == 0
    assert result["neutral_score"] >= 0


def test_analyze_sentiment_with_negative_bias():
    model = SentimentModel({})
    result = model.analyze_sentiment("业绩糟糕，股价下跌，利空消息")
    assert result["sentiment_score"] < 0
    assert result["negative_score"] >= 1


def test_batch_analyze_returns_list_of_results():
    model = SentimentModel({})
    texts = ["利好上涨", "利空下跌"]
    batch_results = model.batch_analyze(texts)
    assert len(batch_results) == 2
    assert isinstance(batch_results[0], dict)
    assert batch_results[0]["sentiment_score"] != batch_results[1]["sentiment_score"]


def test_load_pretrained_model_uses_config_path():
    model = load_pretrained_model("models/demo.bin")
    assert isinstance(model, SentimentModel)
    assert model.model_config["model_path"] == "models/demo.bin"

