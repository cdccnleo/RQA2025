"""情感分析器模块"""
from typing import Dict, List, Optional
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dataclasses import dataclass
from src.features.feature_config import FeatureType, FeatureConfig

@dataclass
class SentimentConfig:
    """情感分析配置"""
    model_name: str = "FinBERT-ZH"  # 预训练模型名称
    policy_keywords: List[str] = None  # 政策关键词列表
    industry_terms: Dict[str, List[str]] = None  # 行业术语映射

class SentimentAnalyzer:
    """中文财经文本情感分析器"""

    def __init__(self, register_feature_func, config: SentimentConfig = None):
        self.register_feature_func = register_feature_func
        self.config = config or SentimentConfig()
        self.model, self.tokenizer = self._load_pretrained_model(self.config.model_name)
        self._register_sentiment_features()

    def _load_pretrained_model(self, model_name: str):
        """加载预训练模型"""
        # 实际项目中应从模型仓库加载
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer

    def _register_sentiment_features(self):
        """注册情感特征到特征引擎"""
        # 基础情感特征
        self.register_feature_func(FeatureConfig(
            name="SENTIMENT_SCORE",
            feature_type=FeatureType.SENTIMENT,
            params={"model": self.config.model_name},
            dependencies=[]
        ))

        # 政策情感特征
        if self.config.policy_keywords:
            self.register_feature_func(FeatureConfig(
                name="POLICY_SENTIMENT",
                feature_type=FeatureType.SENTIMENT,
                params={"keywords": self.config.policy_keywords},
                dependencies=[],
                a_share_specific=True
            ))

        # 行业情感特征
        if self.config.industry_terms:
            for industry, terms in self.config.industry_terms.items():
                self.register_feature_func(FeatureConfig(
                    name=f"{industry}_SENTIMENT",
                    feature_type=FeatureType.SENTIMENT,
                    params={"terms": terms},
                    dependencies=[],
                    a_share_specific=True
                ))

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """分析文本情感"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model(**inputs)
        probs = outputs.logits.softmax(dim=1)

        return {
            "positive": probs[0][2].item(),  # 正面概率
            "neutral": probs[0][1].item(),   # 中性概率
            "negative": probs[0][0].item()  # 负面概率
        }

    def detect_policy_keywords(self, text: str) -> Dict[str, int]:
        """检测政策关键词"""
        if not self.config.policy_keywords:
            return {}

        keywords_found = {}
        for keyword in self.config.policy_keywords:
            if keyword in text:
                keywords_found[keyword] = text.count(keyword)

        return keywords_found

    def analyze_industry_sentiment(self, text: str) -> Dict[str, Dict[str, float]]:
        """分析行业特定情感"""
        if not self.config.industry_terms:
            return {}

        results = {}
        for industry, terms in self.config.industry_terms.items():
            # 检查是否包含行业术语
            contains_terms = any(term in text for term in terms)
            if contains_terms:
                sentiment = self.analyze_sentiment(text)
                results[industry] = sentiment

        return results

    def calculate_sentiment_features(self, text_data: List[str]) -> Dict[str, np.ndarray]:
        """批量计算情感特征"""
        # 基础情感得分
        sentiment_scores = np.array([self.analyze_sentiment(text)["positive"] for text in text_data])

        # 政策情感得分
        policy_scores = np.zeros(len(text_data))
        if self.config.policy_keywords:
            for i, text in enumerate(text_data):
                keywords = self.detect_policy_keywords(text)
                if keywords:
                    sentiment = self.analyze_sentiment(text)
                    policy_scores[i] = sentiment["positive"] - sentiment["negative"]

        # 行业情感得分
        industry_scores = {}
        if self.config.industry_terms:
            for industry in self.config.industry_terms:
                industry_scores[f"{industry}_SENTIMENT"] = np.zeros(len(text_data))

            for i, text in enumerate(text_data):
                industry_sentiments = self.analyze_industry_sentiment(text)
                for industry, sentiment in industry_sentiments.items():
                    industry_scores[f"{industry}_SENTIMENT"][i] = (
                        sentiment["positive"] - sentiment["negative"]
                    )

        return {
            "SENTIMENT_SCORE": sentiment_scores,
            "POLICY_SENTIMENT": policy_scores,
            **industry_scores
        }

class AShareSentimentMixin:
    """A股特有情感分析功能"""

    @staticmethod
    def calculate_margin_sentiment(margin_data: Dict[str, List[str]]) -> Dict[str, float]:
        """计算融资融券相关文本情感"""
        # 实现融资融券特定情感分析逻辑
        pass

    @staticmethod
    def calculate_dragon_board_sentiment(dragon_data: Dict[str, List[str]]) -> Dict[str, float]:
        """计算龙虎榜相关文本情感"""
        # 实现龙虎榜特定情感分析逻辑
        pass

    @staticmethod
    def calculate_northbound_sentiment(northbound_data: Dict[str, List[str]]) -> Dict[str, float]:
        """计算北向资金相关文本情感"""
        # 实现北向资金特定情感分析逻辑
        pass

class FpgaSentimentAnalyzer(SentimentAnalyzer):
    """FPGA加速的情感分析器"""

    def __init__(self, feature_engine: 'FeatureEngineer', config: SentimentConfig):
        super().__init__(feature_engine, config)
        # 初始化FPGA加速器
        self.fpga_accelerator = self._init_fpga_accelerator()

    def _init_fpga_accelerator(self):
        """初始化FPGA加速器"""
        # 实际项目中应连接FPGA设备
        return None

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """使用FPGA加速的情感分析"""
        if self.fpga_accelerator:
            # FPGA加速路径
            return self.fpga_accelerator.analyze(text)
        else:
            # 降级到CPU处理
            return super().analyze_sentiment(text)
