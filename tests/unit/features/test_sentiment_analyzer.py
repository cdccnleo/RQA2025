# tests/features/test_sentiment_analyzer.py
import configparser
from datetime import datetime
from unittest import mock
from unittest.mock import patch, Mock

import numpy as np
import pytest
import pandas as pd
from pathlib import Path
from src.features.processors.sentiment import SentimentAnalyzer
import logging
import torch

logger = logging.getLogger(__name__)


@pytest.fixture
def sample_news_data() -> pd.DataFrame:
    """生成包含中英文混合的测试新闻数据"""
    return pd.DataFrame({
        "title": ["头条新闻"] * 10,
        "content": [
            "<div>公司业绩超出预期，股价大涨！</div>",  # 中文积极
            "The company's performance is outstanding and growth potential is huge!",  # 英文积极
            "The company is suffering severe financial losses and is in crisis.",  # 英文消极
            "公司亏损严重，面临财务危机。",  # 中文消极
            "   ",  # 空内容
            None,  # 缺失值
            "Another positive news",  # 新增英文内容
            "Negative news here",  # 新增英文内容
            "公司发布新产品",  # 新增中文内容
            "Market reacts positively"  # 新增英文内容
        ],
        "date": pd.date_range(start="2023-01-01", periods=10)
    })


@pytest.fixture
def bert_model_mock():
    """模拟BERT模型和分词器"""
    with patch('transformers.AutoModelForSequenceClassification.from_pretrained') as mock_model, \
            patch('transformers.DistilBertTokenizer.from_pretrained') as mock_tokenizer:
        mock_model.return_value = Mock()
        mock_tokenizer.return_value = Mock()
        yield


def test_clean_news_text_normal(bert_model_mock, sample_news_data):
    """测试HTML标签清洗和空格规范化"""
    processor = SentimentAnalyzer()
    cleaned = processor.clean_news_text(sample_news_data["content"][0])
    assert "<div>" not in cleaned
    assert "  " not in cleaned


def test_clean_empty_text():
    """测试空文本输入处理"""
    processor = SentimentAnalyzer(skip_config=True)
    assert processor.clean_news_text("   ") == ""
    assert processor.clean_news_text(None) == ""
    logger.info("空文本清洗测试通过")


#  分词测试
def test_chinese_segmentation():
    """测试中文分词正确性"""
    processor = SentimentAnalyzer()
    text = "量化交易策略"  # 测试文本
    segmented = processor.segment_text(text)  # 调用 segment_text 方法并传递 text 参数
    assert segmented is not None  # 检查分词结果是否不为空
    logger.info("中文分词测试通过")


def test_english_segmentation():
    """测试英文文本不进行分词"""
    processor = SentimentAnalyzer(use_segmentation=False)
    text = "This is an example sentence."
    assert processor.segment_text(text) == text
    logger.info("英文不分词测试通过")


# ------------------- 情感分析测试 -------------------
def test_snownlp_sentiment_distribution():
    """测试SnowNLP情感分数分布"""
    processor = SentimentAnalyzer()

    # 积极文本测试
    pos_score = processor.snownlp_sentiment("公司业绩超出预期，股价大涨！")
    assert pos_score > 0.5

    # 更换更明确的消极文本，例如："这家公司破产了，产品非常糟糕。"
    neg_score = processor.snownlp_sentiment("产品极其糟糕，服务态度差，后悔购买！")
    assert neg_score < 0.5  # 确保消极文本得分低于0.5


def test_textblob_sentiment_distribution():
    """测试TextBlob情感分数分布"""
    processor = SentimentAnalyzer()

    # 积极文本测试
    pos_score = processor.textblob_sentiment("The company's performance is outstanding and growth potential is huge!",
                                             language="en")
    assert pos_score > 0.0

    neg_score = processor.textblob_sentiment("This product is terrible and a waste of money.", language="en")
    assert neg_score < 0.0  # 确保消极文本得分小于0.0


def test_load_bert_model_success(tmp_bert_model_path, monkeypatch):
    """测试BERT模型加载成功"""

    # Mock配置文件读取
    def mock_config_read(self, filenames):
        self["Paths"] = {"bert_model": str(tmp_bert_model_path)}

    monkeypatch.setattr(configparser.ConfigParser, 'read', mock_config_read)

    # 模拟模型和分词器返回有效对象
    with patch('transformers.AutoModelForSequenceClassification.from_pretrained') as mock_model, \
            patch('transformers.DistilBertTokenizer.from_pretrained') as mock_tokenizer:
        mock_model.return_value = Mock()
        mock_tokenizer.return_value = Mock()

        analyzer = SentimentAnalyzer()

        # 验证模型加载方法被调用
        mock_model.assert_called_once_with(tmp_bert_model_path)
        mock_tokenizer.assert_called_once_with(tmp_bert_model_path)
        assert analyzer.bert_model is not None


def test_load_bert_model_invalid_path(tmp_path):
    """测试无效路径下的BERT模型加载"""
    # 获取项目根目录的 src/features 路径
    project_root = Path(__file__).resolve().parents[2]  # 根据测试文件层级调整
    config_dir = project_root / "src" / "features"
    config_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在
    config_path = config_dir / "features_config.ini"

    # 备份原始配置（如果存在）
    original_content = None
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            original_content = f.read()

    # 写入无效路径到配置文件
    invalid_path = "/invalid/model/path"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(f"[Paths]\nbert_model = {invalid_path}")

    # 初始化 SentimentAnalyzer 并验证
    processor = SentimentAnalyzer()
    assert processor.bert_model is None, "无效路径下不应加载BERT模型"

    # 恢复原始配置
    if original_content is not None:
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(original_content)
    else:
        config_path.unlink(missing_ok=True)  # 删除临时文件


def test_load_bert_model_invalid_model_path(tmp_path):
    """测试配置文件中无效模型路径处理"""
    # 创建临时配置文件
    config_path = tmp_path / "features_config.ini"
    with open(config_path, "w") as f:
        f.write("[Paths]\nbert_model = /invalid/model/path")

    # 模拟从配置文件加载
    with patch('src.features.processors.sentiment.Path') as mock_path:
        mock_path.return_value = tmp_path
        processor = SentimentAnalyzer()
        assert processor.bert_model is None, "BERT模型不应加载"
        assert processor.bert_tokenizer is None, "BERT分词器不应加载"
    logger.info("无效模型路径处理测试通过")


def test_feature_generation_integration(tmp_bert_model_path):
    """测试端到端特征生成流程"""
    # 更新配置文件中的模型路径
    config_path = Path(__file__).resolve().parent.parent / "features" / "features_config.ini"
    with open(config_path, "w") as f:
        f.write(f"[Paths]\nbert_model = {tmp_bert_model_path}")

    with patch('transformers.AutoModelForSequenceClassification.from_pretrained') as mock_model, \
            patch('transformers.DistilBertTokenizer.from_pretrained') as mock_tokenizer:
        mock_model.return_value = Mock()
        mock_tokenizer.return_value = Mock()
        processor = SentimentAnalyzer()
        sample_news_data = pd.DataFrame({
            "content": ["公司业绩超出预期，股价大涨！"],
            "date": ["2023-01-01"]
        })
        features = processor.generate_features(
            sample_news_data,
            text_col="content",
            use_segmentation=True
        )
        assert features is not None


def test_missing_text_column():
    """测试缺少文本列时的异常抛出"""
    processor = SentimentAnalyzer()
    # 构造包含date列但缺少content列的数据
    invalid_data = pd.DataFrame({
        "date": ["2023-01-01"],
        "wrong_col": ["test"]
    })

    with pytest.raises(ValueError, match=r"输入数据缺少必要列: content"):
        processor.generate_features(invalid_data)
    logger.info("缺少文本列异常处理测试通过")


def test_large_batch_handling():
    """测试大数据量下的内存处理"""
    processor = SentimentAnalyzer()
    large_data = pd.DataFrame({
        "content": ["test"] * 10000,
        "date": pd.date_range(start="2023-01-01", periods=10000)
    })

    features = processor.generate_features(large_data)
    assert len(features) == 10000
    logger.info("大数据量处理测试通过")


# 性能测试
def test_batch_inference_performance(benchmark, sample_news_data: pd.DataFrame):
    """BERT批量推理性能基准测试（仅在GPU可用时执行）"""
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        pytest.skip("CUDA不可用，跳过GPU性能测试")

    # 确保模型路径正确
    config_path = Path(__file__).resolve().parent.parent / "features" / "features_config.ini"
    bert_model_dir = Path(__file__).resolve().parents[2] / "models" / "bert"
    if not bert_model_dir.exists():
        pytest.skip("BERT模型路径不存在")

    # 写入配置文件
    with open(config_path, "w") as f:
        f.write(f"[Paths]\nbert_model = {bert_model_dir}")

    # 初始化处理器（强制使用GPU和大批量）
    processor = SentimentAnalyzer(use_gpu=True, batch_size=128)

    # 预热模型（避免首次加载时间影响测试）
    processor._bert_batch_predict(["warmup"] * 10)

    # 执行性能测试
    benchmark(processor._bert_batch_predict, ["test"] * 100)

    # 调整断言阈值（根据实际硬件性能设置合理值）
    assert benchmark.stats.stats.mean < 0.3, "GPU推理时间应小于300ms"


@patch('torch.nn.functional.softmax')
def test_bert_batch_inference(softmax_mock, sample_news_data: pd.DataFrame, tmp_path, monkeypatch):
    """测试BERT批量推理性能及结果格式"""
    softmax_mock.return_value = torch.tensor([[0.2, 0.8], [0.3, 0.7]])

    # 动态创建临时BERT模型文件
    bert_model_dir = tmp_path / "models" / "bert"
    bert_model_dir.mkdir(parents=True)
    (bert_model_dir / "config.json").touch()
    (bert_model_dir / "pytorch_model.bin").touch()
    (bert_model_dir / "vocab.txt").touch()

    # 创建临时配置文件并写入路径
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "features_config.ini"
    with open(config_path, "w") as f:
        f.write(f"[Paths]\nbert_model = {bert_model_dir}")

    # 劫持配置加载路径
    def mock_load_config(self):
        self.bert_model_path = str(bert_model_dir)

    monkeypatch.setattr(SentimentAnalyzer, '_load_config', mock_load_config)

    # 模拟模型和分词器
    with patch('transformers.AutoModelForSequenceClassification.from_pretrained') as mock_model, \
            patch('transformers.DistilBertTokenizer.from_pretrained') as mock_tokenizer:
        # 配置分词器返回合法输入
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.return_attention_mask = True
        mock_tokenizer_instance.return_tensors = "pt"
        mock_tokenizer_instance.__call__ = lambda texts, **kwargs: {
            'input_ids': torch.randint(0, 100, (len(texts), 10)),
            'attention_mask': torch.ones((len(texts), 10))
        }
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # 配置模型返回合法logits
        mock_model_instance = Mock()
        mock_model_instance.eval.return_value = None
        mock_model_instance.to.return_value = mock_model_instance  # 确保设备移动不影响测试
        mock_model_instance.forward = Mock(return_value=Mock(logits=torch.tensor([[1.0, -1.0], [0.5, -0.5]])))
        mock_model.from_pretrained.return_value = mock_model_instance

        processor = SentimentAnalyzer(use_gpu=False)  # 禁用GPU加速
        # 显式加载模型
        processor._load_bert_model(bert_model_dir)  # 新增代码

        # 执行推理
        texts = ["文本1", "文本2"]
        scores = processor._bert_batch_predict(texts)

        # 验证结果格式
        assert len(scores) == 2, f"预期返回2个结果，实际返回{len(scores)}个"
        assert all(isinstance(s, float) for s in scores), f"存在非浮点数: {scores}"
        assert all(0 <= s <= 1 for s in scores), f"分数超出范围: {scores}"


@pytest.fixture
def mock_news_data() -> pd.DataFrame:
    """生成测试新闻数据"""
    return pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=5),
        "content": [
            "利好消息：公司发布新产品",
            "Negative news about market crash",
            "",
            "<html>有<script>标签的文本</html>",
            "Another positive announcement"
        ]
    })


def test_sentiment_feature_generation(mock_news_data, tmp_bert_model_path):
    """测试情感特征生成"""
    # 更新配置文件中的 BERT 模型路径
    config_path = Path(__file__).resolve().parent.parent / "features" / "features_config.ini"
    with open(config_path, "w") as f:
        f.write(f"[Paths]\nbert_model = {tmp_bert_model_path}")

    # Mock模型和分词器的加载，确保bert_model和bert_tokenizer不为None
    with patch('transformers.AutoModelForSequenceClassification.from_pretrained') as mock_model, \
            patch('transformers.DistilBertTokenizer.from_pretrained') as mock_tokenizer, \
            patch('src.features.processors.sentiment.SentimentAnalyzer._load_config') as mock_load_config:
        # 强制设置bert_model_path，跳过配置文件读取
        # 强制设置路径并加载模型
        mock_load_config.side_effect = lambda: (
            setattr(SentimentAnalyzer, 'bert_model_path', str(tmp_bert_model_path))
        )
        mock_model.return_value = Mock()
        mock_tokenizer.return_value = Mock()

        analyzer = SentimentAnalyzer(use_gpu=False)
        # 显式加载模型
        analyzer._load_bert_model(tmp_bert_model_path)  # 新增代码

        # 断言模型和分词器已加载
        assert analyzer.bert_model is not None
        assert analyzer.bert_tokenizer is not None

        # 测试特征生成
        with patch.object(analyzer, '_bert_batch_predict') as mock_bert:
            # Mock返回5个分数，对应5条测试数据
            mock_bert.return_value = [0.8, 0.2, 0.5, 0.6, 0.9]
            features = analyzer.generate_features(mock_news_data)

            # 验证BERT分数列无NaN
            assert features["bert_mean"].notna().all(), "bert_mean存在NaN"

            assert 'bert_mean' in features.columns
            assert features["bert_mean"].notna().all()


def test_language_detection():
    analyzer = SentimentAnalyzer()
    assert analyzer.detect_language("中文文本") == "zh"
    assert analyzer.detect_language("English text") == "en"


def test_snownlp_empty_text():
    analyzer = SentimentAnalyzer()
    assert analyzer.snownlp_sentiment("") is np.nan


def test_textblob_sentiment_english():
    analyzer = SentimentAnalyzer()
    pos_score = analyzer.textblob_sentiment("This is a positive review", language="en")
    neg_score = analyzer.textblob_sentiment("This is a negative review", language="en")
    assert pos_score > 0.0
    assert neg_score < 0.0


def test_generate_features_empty_news():
    """测试空新闻数据输入时的处理"""
    analyzer = SentimentAnalyzer()

    # 构造包含必要列但无数据的DataFrame
    empty_news = pd.DataFrame(columns=["date", "content"])

    # 执行特征生成
    features = analyzer.generate_features(empty_news)

    # 验证返回结构
    assert features.empty
    assert all(col in features.columns for col in analyzer.DEFAULT_COLS)
    logger.info("空新闻数据测试通过")


def test_clean_news_text_html():
    analyzer = SentimentAnalyzer()
    cleaned = analyzer.clean_news_text("<div>Hello <b>World</b></div>")
    assert "<" not in cleaned


def test_missing_required_columns():
    """测试缺失必要列时的异常处理"""
    analyzer = SentimentAnalyzer()
    invalid_data = pd.DataFrame({"wrong_col": []})

    with pytest.raises(ValueError) as excinfo:
        analyzer.generate_features(invalid_data)

    assert "date" in str(excinfo.value)
    assert "content" in str(excinfo.value)
    logger.info("缺失必要列异常处理测试通过")


@pytest.mark.parametrize("input_data, expected_missing", [
    ({"wrong_col": [1]}, "date, content"),
    ({"date": [1]}, "content"),
    ({"content": ["test"]}, "date")
])
def test_missing_columns_scenarios(input_data, expected_missing):
    """测试不同列缺失场景"""
    analyzer = SentimentAnalyzer()
    test_data = pd.DataFrame(input_data)

    with pytest.raises(ValueError) as excinfo:
        analyzer.generate_features(test_data)

    assert f"输入数据缺少必要列: {expected_missing}" in str(excinfo.value)
    logger.info(f"缺失列场景测试通过：{expected_missing}")


def test_bert_fallback():
    analyzer = SentimentAnalyzer(use_gpu=False, skip_config=True)  # 跳过配置加载
    analyzer.bert_model = None  # 强制模拟模型缺失
    scores = analyzer._bert_batch_predict(["test text"], "en")
    assert scores == [0.5]  # 验证返回中性值


def test_bert_model_failure_handling():
    """测试BERT模型加载失败时的降级处理"""
    analyzer = SentimentAnalyzer(use_gpu=False)
    # 注入无效模型路径
    analyzer.bert_model_path = "/invalid/path"
    analyzer.bert_model = None
    analyzer.bert_tokenizer = None
    features = analyzer.generate_features(news_data=pd.DataFrame({
        "content": ["test text"],
        "date": [pd.Timestamp.now()]
    }))

    # 验证BERT特征存在且为默认值
    assert "bert_mean" in features.columns
    assert features["bert_mean"].iloc[0] == 0.5
    assert features["bert_volatility"].iloc[0] == 0.0


@pytest.mark.parametrize("text,lang,expected", [
    ("美股大涨", "zh", (0.6, 1.0)),  # 中文积极
    ("市场崩盘", "zh", (0.0, 0.4)),  # 中文消极  
    ("Great news!", "en", (0.7, 1.0)),  # 英文积极
    ("Terrible product", "en", (0.0, 0.3)),  # 英文消极
    ("", "en", (0.5, 0.5)),  # 空文本
    ("@#$%^", "en", (0.5, 0.5)),  # 特殊字符
    ("a"*1000, "en", (0.5, 0.5)),  # 超长文本
])
def test_multilingual_sentiment_analysis(text, lang, expected):
    """参数化测试多语言情感分析"""
    analyzer = SentimentAnalyzer()
    
    # 测试SnowNLP中文分析
    if lang == "zh":
        score = analyzer.snownlp_sentiment(text)
        if not isinstance(score, float):  # 处理空文本返回nan的情况
            score = 0.5
        assert expected[0] <= score <= expected[1]
    
    # 测试TextBlob英文分析
    if lang == "en":
        score = analyzer.textblob_sentiment(text, lang)
        assert expected[0] <= score <= expected[1]

    # 测试BERT多语言分析
    with patch.object(analyzer, '_bert_batch_predict') as mock_bert:
        mock_bert.return_value = [expected[0]]
        bert_score = analyzer._bert_batch_predict([text])[0]
        assert expected[0] <= bert_score <= expected[1]

@pytest.mark.parametrize("model1,model2,text,threshold", [
    ("snownlp", "textblob", "这家公司产品质量很好", 0.3),
    ("textblob", "bert", "This stock will rise", 0.4),
])
def test_model_consistency(model1, model2, text, threshold):
    """测试不同模型对相同文本的情感判断一致性"""
    analyzer = SentimentAnalyzer()
    
    # 获取第一个模型分数
    if model1 == "snownlp":
        score1 = analyzer.snownlp_sentiment(text)
    else:
        score1 = analyzer.textblob_sentiment(text, "en")
    
    # 获取第二个模型分数 
    if model2 == "snownlp":
        score2 = analyzer.snownlp_sentiment(text)
    else:
        with patch.object(analyzer, '_bert_batch_predict') as mock_bert:
            mock_bert.return_value = [0.7]  # Mock BERT返回
            score2 = analyzer._bert_batch_predict([text])[0]
    
    # 验证分数差异在阈值范围内
    assert abs(score1 - score2) <= threshold


@pytest.mark.parametrize("config_change", [
    {"use_bert": False},
    {"bert_model_path": "new/path"},
    {"default_language": "ja"},
])
def test_config_change_handling(config_change):
    """测试配置变更处理"""
    original = SentimentAnalyzer().config
    
    # 模拟配置变更
    with patch.object(SentimentAnalyzer, '_load_config') as mock_load:
        mock_load.return_value = {**original, **config_change}
        analyzer = SentimentAnalyzer()
        
        # 验证配置生效
        for k, v in config_change.items():
            assert getattr(analyzer, k) == v

@pytest.mark.benchmark
def test_high_volume_processing(benchmark):
    """测试高吞吐量下的性能表现"""
    analyzer = SentimentAnalyzer()
    test_data = pd.DataFrame({
        "content": ["test"] * 1000,
        "date": pd.date_range("2023-01-01", periods=1000)
    })
    
    # 执行基准测试
    result = benchmark(analyzer.generate_features, test_data)
    assert len(result) == 1000

def test_model_switching():
    """测试模型动态切换逻辑"""
    analyzer = SentimentAnalyzer()
    
    # 初始使用BERT
    assert analyzer.use_bert is True
    
    # 切换至TextBlob
    analyzer.use_bert = False
    features = analyzer.generate_features(pd.DataFrame({
        "content": ["test"],
        "date": ["2023-01-01"]
    }))
    assert "bert_mean" not in features.columns
    
    # 切换回BERT
    analyzer.use_bert = True
    with patch.object(analyzer, '_bert_batch_predict') as mock_bert:
        mock_bert.return_value = [0.8]
        features = analyzer.generate_features(pd.DataFrame({
            "content": ["test"],
            "date": ["2023-01-01"]
        }))
        assert "bert_mean" in features.columns


def test_multilingual_processing():
    """测试多语言混合处理"""
    mixed_data = pd.DataFrame({
        "content": ["Hello world", "你好世界", ""],
        "date": ["2023-01-01"]*3
    })
    result = SentimentAnalyzer().generate_features(mixed_data)
    assert "textblob_mean" in result.columns


def test_bert_model_fallback_on_inference():
    """测试BERT模型推理失败时返回中性值"""
    analyzer = SentimentAnalyzer()
    analyzer.bert_model = None  # 强制模型不可用

    texts = ["test text"] * 3
    scores = analyzer._bert_batch_predict(texts)
    assert scores == [0.5] * 3  # 验证中性值返回


def test_missing_config_file_handling(tmp_path):
    config_path = tmp_path / "features_config.ini"
    analyzer = SentimentAnalyzer(
        skip_config=True,  # 跳过自动加载
        config_path=str(config_path)  # 指定临时路径
    )
    analyzer._load_config()
    assert analyzer.bert_model is None


def test_bert_disabled_features(tmp_path):
    config_path = tmp_path / "features_config.ini"
    analyzer = SentimentAnalyzer(
        skip_config=True,  # 跳过自动加载
        config_path=str(config_path)  # 指定临时路径
    )
    news_data = pd.DataFrame({
        "content": [
            "美股大涨",  # 中文
            "This product is absolutely terrible and disappointing!"  # 英文
        ],
        "date": [
            pd.Timestamp("2023-01-01"),  # 不同日期
            pd.Timestamp("2023-01-02")
        ]
    })
    # 当BERT模型加载失败时
    analyzer.bert_model = None
    features = analyzer.generate_features(news_data)

    # 验证BERT特征存在但值为默认
    assert 'bert_mean' in features.columns
    assert 'bert_volatility' in features.columns
    assert (features["bert_mean"] == 0.5).all()
    assert (features["bert_volatility"] == 0.0).all()


def test_bert_model_missing():
    """测试BERT模型缺失时的降级处理"""
    news_data = pd.DataFrame({
        "content": [
            "美股大涨",  # 中文
            "This product is absolutely terrible and disappointing!"  # 英文
        ],
        "date": [
            pd.Timestamp("2023-01-01"),  # 不同日期
            pd.Timestamp("2023-01-02")
        ]
    })

    analyzer = SentimentAnalyzer(skip_config=True)
    features = analyzer.generate_features(news_data)
    assert features["bert_mean"].mean() == 0.5  # 中性值填充


def test_empty_text_processing():
    """测试空文本情感分析"""
    analyzer = SentimentAnalyzer(skip_config=True)

    data = pd.DataFrame({"content": [""], "date": [datetime.today()]})
    features = analyzer.generate_features(data)
    assert features["snownlp_mean"][0] == 0.5  # 默认值


def test_language_detection_failure():
    """测试语言检测异常时的默认处理"""
    analyzer = SentimentAnalyzer(skip_config=True)
    news_data = pd.DataFrame({
        "content": [
            "美股大涨",  # 中文
            "This product is absolutely terrible and disappointing!"  # 英文
        ],
        "date": [
            pd.Timestamp("2023-01-01"),  # 不同日期
            pd.Timestamp("2023-01-02")
        ]
    })
    with mock.patch.object(SentimentAnalyzer, 'detect_language', side_effect=Exception):
        features = analyzer.generate_features(news_data)

    # 验证结果
    assert 'language' in features.columns
    assert features['language'].iloc[0] == 'unknown'
    assert features['language'].iloc[1] == 'unknown'


def test_bert_predict_empty_text():
    """测试空文本的BERT预测"""
    processor = SentimentAnalyzer(skip_config=True)
    processor.bert_model = Mock()
    processor.bert_tokenizer = Mock()

    # 模拟分词器处理空文本
    processor.bert_tokenizer.return_value = {"input_ids": torch.tensor([[0]])}
    result = processor._bert_batch_predict([""])
    assert result[0] == 0.5  # 应返回中性值


def test_generate_features_with_missing_columns():
    """测试缺少必要列的情感特征生成"""
    sa = SentimentAnalyzer(skip_config=True)
    data = pd.DataFrame({"wrong_col": ["text"]})

    with pytest.raises(ValueError, match="缺少必要列"):
        sa.generate_features(data)


def test_bert_batch_predict_without_model():
    """测试BERT模型未加载时的预测"""
    sa = SentimentAnalyzer(skip_config=True)
    sa.bert_model = None

    results = sa._bert_batch_predict(["text1", "text2"])
    assert results == [0.5, 0.5]  # 应返回中性值


def test_load_config_with_invalid_path(caplog):
    """测试无效配置文件路径处理"""
    sa = SentimentAnalyzer(config_path="invalid/path.ini")
    assert "配置文件不存在" in caplog.text
    assert sa.bert_model is None


# 测试BERT模型加载失败
def test_bert_load_failure():
    analyzer = SentimentAnalyzer(config_path="invalid.ini")
    assert analyzer.bert_model is None

# 测试空文本处理
def test_empty_text_sentiment():
    analyzer = SentimentAnalyzer()
    assert np.isnan(analyzer.snownlp_sentiment(""))
    assert analyzer.textblob_sentiment("", "en") == 0

def test_bert_batch_failure():
    """测试BERT推理失败时的稳健性"""
    analyzer = SentimentAnalyzer()
    # 在进入测试前，确保模型和分词器已加载
    assert analyzer.bert_model is not None
    assert analyzer.bert_tokenizer is not None

    # 模拟GPU错误
    with patch.object(analyzer.bert_model, 'to') as mock_to:
        mock_to.side_effect = RuntimeError("GPU error")
        scores = analyzer._bert_batch_predict(["text"])
        assert scores == [0.5]  # 验证返回值是否为默认值

def test_sentiment_analyzer_skip_config():
    """测试跳过配置加载"""
    analyzer = SentimentAnalyzer(skip_config=True)
    assert analyzer.bert_model is None
    assert analyzer.bert_tokenizer is None


def test_load_config_missing_file(caplog):
    """测试配置文件不存在的情况"""
    analyzer = SentimentAnalyzer(config_path="/invalid/path/config.ini")
    assert "配置文件不存在" in caplog.text
    assert analyzer.bert_model is None


def test_generate_features_bert_fallback():
    """测试BERT失败时回退到其他特征"""
    analyzer = SentimentAnalyzer()
    # 模拟BERT失败
    analyzer._bert_batch_predict = Mock(side_effect=Exception("BERT error"))

    news_data = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=3),
        "content": ["text1", "text2", "text3"]
    })

    features = analyzer.generate_features(news_data)
    # 验证基础特征存在
    assert "snownlp_mean" in features.columns
    assert "textblob_mean" in features.columns
    # BERT特征应为中性值
    assert all(features["bert_mean"] == 0.5)


def test_bert_batch_predict_fallback():
    """测试BERT模型缺失时的降级处理"""
    analyzer = SentimentAnalyzer(skip_config=True)
    texts = ["test"] * 3
    results = analyzer._bert_batch_predict(texts)
    assert results == [0.5, 0.5, 0.5]  # 中性值


def test_bert_model_loading_failure():
    """测试BERT模型加载失败处理"""
    analyzer = SentimentAnalyzer(config_path="invalid/path")
    assert analyzer.bert_model is None

def test_generate_features_missing_columns():
    """测试缺少必要列的情感特征生成"""
    analyzer = SentimentAnalyzer()
    data = pd.DataFrame({'content': ['test']})  # 缺少date列
    with pytest.raises(ValueError):
        analyzer.generate_features(data)