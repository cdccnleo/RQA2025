"""高频特征优化器"""
import numpy as np
from typing import Dict, List
from dataclasses import dataclass
from .feature_config import FeatureType, FeatureRegistrationConfig
from .feature_engineer import FeatureEngineer
from .orderbook.level2_analyzer import Level2Analyzer

# 兼容性处理：检查numba是否可用
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    # 如果numba不可用，创建一个装饰器来保持API兼容性

    def njit(*args, **kwargs):

        # 如果第一个参数是函数，说明没有参数调用
        if args and callable(args[0]):
            return args[0]
        # 如果有参数调用，返回装饰器

        def decorator(func):

            return func
        return decorator
    NUMBA_AVAILABLE = False


@dataclass
class HighFreqConfig:

    """高频特征优化配置"""
    batch_size: int = 100
    prealloc_memory: int = 256
    parallel_threshold: int = 1000
    use_simd: bool = True
    max_retries: int = 3
    enable_fallback: bool = False


class HighFreqOptimizer:

    """高频特征提取优化器"""

    def __init__(self, feature_engine: FeatureEngineer, config: HighFreqConfig = None):

        self.engine = feature_engine
        self.config = config or HighFreqConfig()

        # 初始化Level2分析器
        self.level2_analyzer = Level2Analyzer(feature_engine)

        # 预分配内存
        if self.config.prealloc_memory:
            self._preallocate_memory()

        # 注册优化后的特征
        self._register_optimized_features()

    def _preallocate_memory(self):
        """预分配内存池"""
        self.feature_buffer = np.zeros((self.config.batch_size, 10))  # 10个特征
        self.temp_buffer = np.zeros((self.config.batch_size, 5))  # 临时计算缓冲区

    def _register_optimized_features(self):
        """注册优化后的特征"""
        # 高频动量特征
        self.engine.register_feature(FeatureRegistrationConfig(
            name="HF_MOMENTUM",
            feature_type=FeatureType.HIGH_FREQUENCY,
            params={"window": 10},
            dependencies=["price", "volume"]
        ))

        # 订单流不平衡特征
        self.engine.register_feature(FeatureRegistrationConfig(
            name="ORDER_FLOW_IMBALANCE",
            feature_type=FeatureType.HIGH_FREQUENCY,
            params={"depth": 5},
            dependencies=["bid", "ask", "trade"]
        ))

        # 瞬时波动率特征
        self.engine.register_feature(FeatureRegistrationConfig(
            name="INSTANT_VOLATILITY",
            feature_type=FeatureType.HIGH_FREQUENCY,
            params={"window": 20},
            dependencies=["price"]
        ))

    @staticmethod
    @njit(parallel=True)
    def _batch_calculate_momentum_numba(prices: np.ndarray, volumes: np.ndarray,


                                        window: int) -> np.ndarray:
        """批量计算动量特征(Numba加速)"""
        n = len(prices)
        result = np.zeros(n)

        for i in range(n):
            if i < window:
                result[i] = 0.0
                continue

            returns = np.log(prices[i - window + 1:i + 1] / prices[i - window:i])
            vol = np.sum(volumes[i - window + 1:i + 1])
            result[i] = np.sum(returns) * vol

        return result

    def calculate_hf_momentum(self, prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """计算高频动量特征"""
        if len(prices) > self.config.parallel_threshold:
            return self._batch_calculate_momentum_numba(prices, volumes, 10)

        # 小批量处理
        return self._calculate_momentum(prices, volumes)

    def _calculate_momentum(self, prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """计算动量特征(非并行版本)"""
        n = len(prices)
        result = np.zeros(n)

        for i in range(n):
            if i < 10:
                result[i] = 0.0
                continue

            returns = np.log(prices[i - 9:i + 1] / prices[i - 10:i])
            vol = np.sum(volumes[i - 9:i + 1])
            result[i] = np.sum(returns) * vol

        return result

    @staticmethod
    @njit
    def _calculate_order_flow_imbalance(bid: np.ndarray, ask: np.ndarray,


                                        trades: np.ndarray, depth: int) -> float:
        """计算订单流不平衡"""
        bid_vol = np.sum(bid[:depth, 1])
        ask_vol = np.sum(ask[:depth, 1])

        buy_trades = trades[trades[:, 1] > 0]
        sell_trades = trades[trades[:, 1] < 0]

        trade_imbalance = np.sum(buy_trades[:, 2]) - np.sum(sell_trades[:, 2])

        return (bid_vol - ask_vol) * 0.3 + trade_imbalance * 0.7

    def calculate_order_flow_imbalance(self, order_book: Dict[str, np.ndarray]) -> float:
        """计算订单流不平衡"""
        return self._calculate_order_flow_imbalance(
            order_book["bid"],
            order_book["ask"],
            order_book.get("trade", np.empty((0, 3))),
            5
        )

    @staticmethod
    @njit
    def _calculate_instant_volatility(prices: np.ndarray, window: int) -> float:
        """计算瞬时波动率"""
        if len(prices) < window:
            return 0.0

        returns = np.log(prices[1:] / prices[:-1])
        recent_returns = returns[-window:]

        return np.std(recent_returns) * np.sqrt(252)

    def calculate_instant_volatility(self, prices: np.ndarray) -> float:
        """计算瞬时波动率"""
        return self._calculate_instant_volatility(prices, 20)

    def batch_calculate_features(self, data_batch: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """批量计算高频特征"""
        batch_size = len(data_batch)

        # 初始化结果数组
        results = {
            "HF_MOMENTUM": np.zeros(batch_size),
            "ORDER_FLOW_IMBALANCE": np.zeros(batch_size),
            "INSTANT_VOLATILITY": np.zeros(batch_size)
        }

        # 批量处理
        for i, data in enumerate(data_batch):
            # 计算Level2特征
            self.level2_analyzer.calculate_all_features(data)

            # 计算高频动量
            prices = data.get("price", np.array([]))
            volumes = data.get("volume", np.array([]))
            if len(prices) > 0:
                results["HF_MOMENTUM"][i] = self.calculate_hf_momentum(prices, volumes)[-1]

            # 计算订单流不平衡
            results["ORDER_FLOW_IMBALANCE"][i] = self.calculate_order_flow_imbalance(data)

            # 计算瞬时波动率
            if len(prices) > 1:
                results["INSTANT_VOLATILITY"][i] = self.calculate_instant_volatility(prices)

        return results


class CppHighFreqOptimizer(HighFreqOptimizer):

    """C++加速的高频特征优化器"""

    def __init__(self, feature_engine: FeatureEngineer, config: HighFreqConfig = None):

        super().__init__(feature_engine, config)
        # 加载C++扩展
        # TODO: 实现C++扩展以提高性能
        self.cpp_optimizer = None  # 暂时禁用C++扩展

    def batch_calculate_features(self, data_batch: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """使用C++加速的批量特征计算"""
        if self.cpp_optimizer:
            return self.cpp_optimizer.batch_calculate(data_batch)
        return super().batch_calculate_features(data_batch)
