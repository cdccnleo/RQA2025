"""
特征配置模块
提供各种特征相关的配置类
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

# 从当前目录导入配置类


@dataclass
class TechnicalConfig:

    """技术指标配置类"""

    # 移动平均线配置
    sma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    ema_periods: List[int] = field(default_factory=lambda: [12, 26])

    # RSI配置
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0

    # MACD配置
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # 布林带配置
    bollinger_period: int = 20
    bollinger_std: float = 2.0

    # 其他指标配置
    atr_period: int = 14
    stoch_k_period: int = 14
    stoch_d_period: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'sma_periods': self.sma_periods,
            'ema_periods': self.ema_periods,
            'rsi_period': self.rsi_period,
            'rsi_overbought': self.rsi_overbought,
            'rsi_oversold': self.rsi_oversold,
            'macd_fast': self.macd_fast,
            'macd_slow': self.macd_slow,
            'macd_signal': self.macd_signal,
            'bollinger_period': self.bollinger_period,
            'bollinger_std': self.bollinger_std,
            'atr_period': self.atr_period,
            'stoch_k_period': self.stoch_k_period,
            'stoch_d_period': self.stoch_d_period
        }


@dataclass
class SentimentConfig:

    """情感分析配置类"""

    # 模型配置
    model_type: str = "bert"
    model_path: Optional[str] = None
    max_length: int = 512

    # 分析配置
    enable_sentiment_analysis: bool = True
    enable_emotion_analysis: bool = False
    enable_topic_analysis: bool = False

    # 阈值配置
    positive_threshold: float = 0.6
    negative_threshold: float = 0.4

    # 缓存配置
    enable_caching: bool = True
    cache_ttl: int = 3600

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'model_type': self.model_type,
            'model_path': self.model_path,
            'max_length': self.max_length,
            'enable_sentiment_analysis': self.enable_sentiment_analysis,
            'enable_emotion_analysis': self.enable_emotion_analysis,
            'enable_topic_analysis': self.enable_topic_analysis,
            'positive_threshold': self.positive_threshold,
            'negative_threshold': self.negative_threshold,
            'enable_caching': self.enable_caching,
            'cache_ttl': self.cache_ttl
        }


@dataclass
class FeatureProcessingConfig:

    """特征处理配置类"""

    # 处理模式
    processing_mode: str = "batch"  # batch, streaming, hybrid
    enable_parallel: bool = True
    max_workers: int = 4

    # 数据配置
    batch_size: int = 1000
    timeout: int = 300

    # 缓存配置
    enable_caching: bool = True
    cache_ttl: int = 3600

    # 验证配置
    enable_validation: bool = True
    strict_mode: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'processing_mode': self.processing_mode,
            'enable_parallel': self.enable_parallel,
            'max_workers': self.max_workers,
            'batch_size': self.batch_size,
            'timeout': self.timeout,
            'enable_caching': self.enable_caching,
            'cache_ttl': self.cache_ttl,
            'enable_validation': self.enable_validation,
            'strict_mode': self.strict_mode
        }
