import pandas as pd
from typing import List, Any
from ..core.config_integration import get_config_integration_manager, ConfigScope
import logging

logger = logging.getLogger(__name__)


class SentimentConfig:

    """情感分析配置类"""

    def __init__(self, **kwargs):

        self.use_bert = kwargs.get('use_bert', True)
        self.bert_model_path = kwargs.get('bert_model_path', None)
        self.default_language = kwargs.get('default_language', 'zh')
        self.batch_size = kwargs.get('batch_size', 32)


class SentimentAnalyzer:

    """情感分析器"""

    def __init__(self, feature_manager=None, config_path=None, skip_config=False, **kwargs):

        self.feature_manager = feature_manager
        self.bert_model = None
        self.bert_tokenizer = None
        # 配置集成
        self.config_manager = get_config_integration_manager()
        sentiment_config = self.config_manager.get_config(ConfigScope.SENTIMENT)
        if sentiment_config:
            self.config = SentimentConfig(**sentiment_config)
        else:
            self.config = SentimentConfig()
        if not skip_config and config_path:
            self._load_config(config_path)
        # 注册配置变更监听器
        self.config_manager.register_config_watcher(ConfigScope.SENTIMENT, self._on_config_change)

    def _on_config_change(self, scope: ConfigScope, key: str, old_value: Any, new_value: Any):

        if scope == ConfigScope.SENTIMENT:
            setattr(self.config, key, new_value)

    def clean_text(self, text: str) -> str:
        """清理文本"""
        if not text:
            return ""
        # 基础文本清理
        text = str(text).strip()
        return text

    def segment_text(self, text: str) -> List[str]:
        """文本分词"""
        if not text:
            return []
        # 简单分词实现
        return text.split()

    def snownlp_sentiment(self, text: str) -> float:
        """使用SnowNLP进行情感分析"""
        try:
            from snownlp import SnowNLP
            cleaned_text = self.clean_text(text)
            if not cleaned_text:
                return 0.5
            s = SnowNLP(cleaned_text)
            return s.sentiments
        except ImportError:
            logger.warning("SnowNLP not available, returning default sentiment")
            return 0.5

    def textblob_sentiment(self, text: str, language: str = "en") -> float:
        """使用TextBlob进行情感分析"""
        try:
            from textblob import TextBlob
            cleaned_text = self.clean_text(text)
            if not cleaned_text:
                return 0.0
            blob = TextBlob(cleaned_text)
            return float(blob.sentiment.polarity)
        except ImportError:
            logger.warning("TextBlob not available, returning default sentiment")
            return 0.0

    def detect_language(self, text: str) -> str:
        """检测文本语言"""
        if not text:
            return "en"
        # 简单的中文检测
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        if chinese_chars > len(text) * 0.3:
            return "zh"
        return "en"

    def generate_features(self, news_data: pd.DataFrame) -> pd.DataFrame:
        """生成情感特征"""
        if len(news_data) == 0:
            return pd.DataFrame()

        # 确保必要的列存在
        required_columns = ['content']
        missing_columns = [col for col in required_columns if col not in news_data.columns]
        if missing_columns:
            raise ValueError(f"缺少必要列: {missing_columns}")

        features = []
        for idx, row in news_data.iterrows():
            content = row.get('content', '')
            if pd.isna(content):
                content = ''

            # 计算情感分数
            snownlp_score = self.snownlp_sentiment(str(content))
            textblob_score = self.textblob_sentiment(str(content))

            features.append({
                'sentiment_score': (snownlp_score + textblob_score) / 2,
                'snownlp_score': snownlp_score,
                'textblob_score': textblob_score
            })

        return pd.DataFrame(features)

    def _load_config(self, config_path: str):
        """加载配置文件"""
        try:
            import configparser
            config = configparser.ConfigParser()
            config.read(config_path)

            if 'Paths' in config:
                self.config.bert_model_path = config.get('Paths', 'bert_model', fallback=None)

            if 'Settings' in config:
                self.config.use_bert = config.getboolean('Settings', 'use_bert', fallback=True)
                self.config.default_language = config.get(
                    'Settings', 'default_language', fallback='zh')
                self.config.batch_size = config.getint('Settings', 'batch_size', fallback=32)

        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")

    def _bert_batch_predict(self, texts: List[str]) -> List[float]:
        """BERT批量预测"""
        if not self.bert_model or not self.bert_tokenizer:
            return [0.5] * len(texts)

        try:
            pass
            # 简化的BERT预测实现
            return [0.5] * len(texts)
        except Exception as e:
            logger.warning(f"BERT prediction failed: {e}")
            return [0.5] * len(texts)
