import logging
"""特征核心配置模块"""
import json
from typing import Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

# 直接定义FeatureType枚举，避免循环导入


class FeatureType(Enum):

    """特征类型枚举"""
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    ORDERBOOK = "orderbook"
    LEVEL2 = "level2"
    HIGH_FREQUENCY = "high_frequency"
    CUSTOM = "custom"


logger = logging.getLogger(__name__)


class TechnicalIndicatorType(Enum):

    """技术指标类型枚举"""
    # 趋势指标
    SMA = "sma"  # 简单移动平均
    EMA = "ema"  # 指数移动平均
    MACD = "macd"  # MACD
    RSI = "rsi"  # 相对强弱指数
    ADX = "adx"  # 平均趋向指数

    # 动量指标
    STOCH = "stoch"  # 随机指标
    CCI = "cci"  # 商品通道指数
    ROC = "roc"  # 变化率

    # 波动率指标
    ATR = "atr"  # 平均真实波幅
    BBANDS = "bbands"  # 布林带
    KC = "kc"  # 肯特纳通道

    # 成交量指标
    OBV = "obv"  # 能量潮
    VWAP = "vwap"  # 成交量加权平均价
    MFI = "mfi"  # 资金流量指数

    # 价格模式
    DOJI = "doji"  # 十字星
    HAMMER = "hammer"  # 锤子线
    ENGULFING = "engulfing"  # 吞没形态


class SentimentType(Enum):

    """情感分析类型枚举"""
    NEWS_SENTIMENT = "news_sentiment"
    SOCIAL_SENTIMENT = "social_sentiment"
    EARNINGS_SENTIMENT = "earnings_sentiment"
    ANALYST_SENTIMENT = "analyst_sentiment"


class OrderBookType(Enum):

    """订单簿类型枚举"""
    LEVEL1 = "level1"
    LEVEL2 = "level2"
    LEVEL3 = "level3"


@dataclass
class OrderBookConfig:

    """订单簿配置类 - 单一来源定义

    此配置类在core层定义，作为整个特征层的单一来源。
    所有其他模块都应该从core层导入此配置类。
    """

    # 基础配置
    orderbook_type: OrderBookType = OrderBookType.LEVEL2
    depth: int = 10  # 订单簿深度
    update_frequency: float = 1.0  # 更新频率（秒）

    # 分析配置
    enable_imbalance_analysis: bool = True
    enable_skew_analysis: bool = True
    enable_spread_analysis: bool = True
    enable_depth_analysis: bool = True

    # 指标配置
    imbalance_threshold: float = 0.1
    skew_threshold: float = 0.05
    spread_threshold: float = 0.001

    # 窗口配置
    window_size: int = 20  # 滑动窗口大小，用于时间序列分析

    # A股特有配置
    a_share_specific: bool = False  # 是否启用A股特有特征

    # 缓存配置
    enable_caching: bool = True
    cache_ttl: int = 60  # 缓存时间（秒）

    # 性能配置
    max_workers: int = 4
    batch_size: int = 1000

    # 自定义配置
    custom_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'orderbook_type': self.orderbook_type.value,
            'depth': self.depth,
            'update_frequency': self.update_frequency,
            'enable_imbalance_analysis': self.enable_imbalance_analysis,
            'enable_skew_analysis': self.enable_skew_analysis,
            'enable_spread_analysis': self.enable_spread_analysis,
            'enable_depth_analysis': self.enable_depth_analysis,
            'imbalance_threshold': self.imbalance_threshold,
            'skew_threshold': self.skew_threshold,
            'spread_threshold': self.spread_threshold,
            'window_size': self.window_size,
            'a_share_specific': self.a_share_specific,
            'enable_caching': self.enable_caching,
            'cache_ttl': self.cache_ttl,
            'max_workers': self.max_workers,
            'batch_size': self.batch_size,
            'custom_config': self.custom_config
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'OrderBookConfig':
        """从字典创建配置"""
        # 处理枚举类型
        if 'orderbook_type' in config_dict and isinstance(config_dict['orderbook_type'], str):
            config_dict['orderbook_type'] = OrderBookType(config_dict['orderbook_type'])

        return cls(**config_dict)

    def validate(self) -> bool:
        """验证配置有效性"""
        try:
            if self.depth <= 0:
                raise ValueError("订单簿深度必须大于0")
            if self.update_frequency <= 0:
                raise ValueError("更新频率必须大于0")
            if self.max_workers <= 0:
                raise ValueError("最大工作线程数必须大于0")
            if self.batch_size <= 0:
                raise ValueError("批处理大小必须大于0")
            return True
        except Exception as e:
            print(f"OrderBookConfig验证失败: {e}")
            return False


@dataclass
class TechnicalParams:

    """技术指标参数配置"""
    sma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    ema_periods: List[int] = field(default_factory=lambda: [12, 26])
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    rsi_period: int = 14
    adx_period: int = 14
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    cci_period: int = 20
    roc_period: int = 10
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    kc_period: int = 20
    kc_multiplier: float = 2.0
    mfi_period: int = 14


@dataclass
class SentimentParams:

    """情感分析参数配置"""
    news_lookback_days: int = 30
    social_lookback_days: int = 7
    min_confidence: float = 0.6
    max_keywords: int = 100
    language: str = "zh - cn"


@dataclass
class FeatureProcessingConfig:

    """特征处理配置类"""

    # 处理参数
    batch_size: int = 1000
    timeout: float = 30.0
    retry_count: int = 3
    retry_delay: float = 1.0

    # 内存管理
    max_memory_usage: float = 1024.0  # MB
    enable_memory_monitoring: bool = True

    # 错误处理
    continue_on_error: bool = False
    log_errors: bool = True

    # 性能优化
    enable_parallel_processing: bool = True
    max_workers: int = 4
    chunk_size: int = 1000

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'batch_size': self.batch_size,
            'timeout': self.timeout,
            'retry_count': self.retry_count,
            'retry_delay': self.retry_delay,
            'max_memory_usage': self.max_memory_usage,
            'enable_memory_monitoring': self.enable_memory_monitoring,
            'continue_on_error': self.continue_on_error,
            'log_errors': self.log_errors,
            'enable_parallel_processing': self.enable_parallel_processing,
            'max_workers': self.max_workers,
            'chunk_size': self.chunk_size,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureProcessingConfig':
        """从字典创建配置"""
        return cls(**data)


@dataclass
class FeatureRegistrationConfig:

    """单个特征注册配置类"""

    # 基本配置
    name: str  # 特征名称
    feature_type: FeatureType  # 特征类型
    params: Dict[str, Any] = field(default_factory=dict)  # 特征参数
    dependencies: List[str] = field(default_factory=list)  # 依赖特征
    enabled: bool = True  # 是否启用
    version: str = "1.0"  # 版本号
    a_share_specific: bool = False  # 是否为A股特有特征
    description: str = ""  # 特征描述
    tags: List[str] = field(default_factory=list)  # 标签列表


@dataclass
class FeatureConfig:

    """特征配置类"""

    # 基本配置
    feature_types: List[FeatureType] = field(default_factory=lambda: [FeatureType.TECHNICAL])
    enable_feature_selection: bool = True
    enable_standardization: bool = True
    enable_feature_saving: bool = False

    # 技术指标配置
    technical_indicators: List[str] = field(default_factory=lambda: [
        "sma", "ema", "rsi", "macd", "bbands", "atr"
    ])
    technical_params: TechnicalParams = field(default_factory=TechnicalParams)

    # 情感分析配置
    sentiment_types: List[str] = field(default_factory=lambda: [
        "news_sentiment", "social_sentiment"
    ])
    sentiment_params: SentimentParams = field(default_factory=SentimentParams)

    # 特征选择配置
    feature_selection_method: str = "mutual_info"
    max_features: int = 50
    min_features: int = 5

    # 标准化配置
    standardization_method: str = "zscore"  # zscore, minmax, robust
    handle_outliers: bool = True
    outlier_method: str = "iqr"  # iqr, zscore, isolation_forest

    # 缓存配置
    enable_caching: bool = True
    cache_dir: str = "./feature_cache"
    cache_ttl: int = 3600  # 1小时

    # 性能配置
    parallel_processing: bool = True
    max_workers: int = 4
    chunk_size: int = 1000

    # 验证配置
    enable_validation: bool = True
    strict_validation: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'feature_types': [ft.value for ft in self.feature_types],
            'enable_feature_selection': self.enable_feature_selection,
            'enable_standardization': self.enable_standardization,
            'enable_feature_saving': self.enable_feature_saving,
            'technical_indicators': self.technical_indicators,
            'technical_params': {
                'sma_periods': self.technical_params.sma_periods,
                'ema_periods': self.technical_params.ema_periods,
                'macd_fast': self.technical_params.macd_fast,
                'macd_slow': self.technical_params.macd_slow,
                'macd_signal': self.technical_params.macd_signal,
                'rsi_period': self.technical_params.rsi_period,
                'adx_period': self.technical_params.adx_period,
                'stoch_k_period': self.technical_params.stoch_k_period,
                'stoch_d_period': self.technical_params.stoch_d_period,
                'cci_period': self.technical_params.cci_period,
                'roc_period': self.technical_params.roc_period,
                'atr_period': self.technical_params.atr_period,
                'bb_period': self.technical_params.bb_period,
                'bb_std': self.technical_params.bb_std,
                'kc_period': self.technical_params.kc_period,
                'kc_multiplier': self.technical_params.kc_multiplier,
                'mfi_period': self.technical_params.mfi_period,
            },
            'sentiment_types': self.sentiment_types,
            'sentiment_params': {
                'news_lookback_days': self.sentiment_params.news_lookback_days,
                'social_lookback_days': self.sentiment_params.social_lookback_days,
                'min_confidence': self.sentiment_params.min_confidence,
                'max_keywords': self.sentiment_params.max_keywords,
                'language': self.sentiment_params.language,
            },
            'feature_selection_method': self.feature_selection_method,
            'max_features': self.max_features,
            'min_features': self.min_features,
            'standardization_method': self.standardization_method,
            'handle_outliers': self.handle_outliers,
            'outlier_method': self.outlier_method,
            'enable_caching': self.enable_caching,
            'cache_dir': self.cache_dir,
            'cache_ttl': self.cache_ttl,
            'parallel_processing': self.parallel_processing,
            'max_workers': self.max_workers,
            'chunk_size': self.chunk_size,
            'enable_validation': self.enable_validation,
            'strict_validation': self.strict_validation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureConfig':
        """从字典创建配置"""
        # 处理枚举类型
        if 'feature_types' in data:
            data['feature_types'] = [FeatureType(ft) for ft in data['feature_types']]

        # 处理技术指标参数
        if 'technical_params' in data:
            tech_params = data['technical_params']
            data['technical_params'] = TechnicalParams(**tech_params)

        # 处理情感分析参数
        if 'sentiment_params' in data:
            sent_params = data['sentiment_params']
            data['sentiment_params'] = SentimentParams(**sent_params)

        return cls(**data)

    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'FeatureConfig':
        """从JSON字符串创建配置"""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def validate(self) -> bool:
        """验证配置有效性"""
        try:
            # 验证特征类型
            if not self.feature_types:
                raise ValueError("至少需要指定一种特征类型")

            # 验证技术指标
            if FeatureType.TECHNICAL in self.feature_types:
                if not self.technical_indicators:
                    raise ValueError("技术指标类型需要指定指标列表")

                # 验证指标参数
                if self.technical_params.sma_periods and min(self.technical_params.sma_periods) <= 0:
                    raise ValueError("SMA周期必须大于0")
                if self.technical_params.ema_periods and min(self.technical_params.ema_periods) <= 0:
                    raise ValueError("EMA周期必须大于0")

            # 验证特征选择参数
            if self.max_features < self.min_features:
                raise ValueError("最大特征数不能小于最小特征数")

            # 验证性能参数
            if self.max_workers <= 0:
                raise ValueError("最大工作线程数必须大于0")
            if self.chunk_size <= 0:
                raise ValueError("数据块大小必须大于0")

            return True

        except Exception as e:
            print(f"配置验证失败: {e}")
            return False

    def get_supported_indicators(self) -> List[str]:
        """获取支持的技术指标列表"""
        return [indicator.value for indicator in TechnicalIndicatorType]

    def get_supported_sentiment_types(self) -> List[str]:
        """获取支持的情感分析类型列表"""
        return [sentiment.value for sentiment in SentimentType]


# 预定义配置
@dataclass
class DefaultConfigs:

    """预定义配置"""

    @staticmethod
    def get_basic_config() -> 'FeatureConfig':
        """获取基础配置"""
        return FeatureConfig(
            feature_types=[FeatureType.TECHNICAL],
            technical_indicators=["sma", "rsi", "macd"],
            enable_feature_selection=False,
            enable_standardization=True
        )

    @staticmethod
    def basic_technical() -> FeatureConfig:
        """基础技术指标配置"""
        return FeatureConfig(
            feature_types=[FeatureType.TECHNICAL],
            technical_indicators=["sma", "rsi", "macd"],
            enable_feature_selection=False,
            enable_standardization=True
        )

    @staticmethod
    def comprehensive_technical() -> FeatureConfig:
        """全面技术指标配置"""
        return FeatureConfig(
            feature_types=[FeatureType.TECHNICAL],
            technical_indicators=["sma", "ema", "rsi", "macd", "bbands", "atr", "stoch", "cci"],
            enable_feature_selection=True,
            enable_standardization=True,
            max_features=30
        )

    @staticmethod
    def sentiment_analysis() -> FeatureConfig:
        """情感分析配置"""
        return FeatureConfig(
            feature_types=[FeatureType.SENTIMENT],
            sentiment_types=["news_sentiment", "social_sentiment"],
            enable_feature_selection=True,
            enable_standardization=True
        )

    @staticmethod
    def full_feature() -> FeatureConfig:
        """完整特征配置"""
        return FeatureConfig(
            feature_types=[FeatureType.TECHNICAL, FeatureType.SENTIMENT],
            technical_indicators=["sma", "ema", "rsi", "macd", "bbands", "atr"],
            sentiment_types=["news_sentiment"],
            enable_feature_selection=True,
            enable_standardization=True,
            max_features=50
        )


# 导出主要类
__all__ = [
    'FeatureType',
    'TechnicalIndicatorType',
    'SentimentType',
    'TechnicalParams',
    'SentimentParams',
    'FeatureRegistrationConfig',
    'FeatureConfig',
    'FeatureProcessingConfig',
    'DefaultConfigs'
]
