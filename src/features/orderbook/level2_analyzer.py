"""Level2行情深度分析模块"""
import numpy as np
from numba import jit
from typing import Dict, List, Tuple
from dataclasses import dataclass
from ..processors.feature_engineer import FeatureEngineer
from ..feature_config import FeatureType, FeatureConfig
from .order_book_analyzer import OrderBookAnalyzer

@dataclass
class Level2Config:
    """Level2行情分析配置"""
    depth: int = 10  # 分析深度
    window_size: int = 20  # 滑动窗口大小
    tick_buffer_size: int = 1000  # tick数据缓冲区大小
    a_share_specific: bool = True  # 是否启用A股特有特征

class Level2Analyzer:
    """高性能Level2行情分析器"""

    def __init__(self, feature_engine: FeatureEngineer, config: Level2Config = None):
        self.engine = feature_engine
        self.config = config or Level2Config()

        # 初始化订单簿分析器
        self.order_book_analyzer = OrderBookAnalyzer(
            feature_engine,
            OrderBookConfig(
                depth=self.config.depth,
                window_size=self.config.window_size,
                a_share_specific=self.config.a_share_specific
            )
        )

        # 初始化tick数据缓冲区
        self.tick_buffer = np.zeros((self.config.tick_buffer_size, 4))  # [price, volume, timestamp, direction]
        self.buffer_index = 0

        # 注册Level2特征
        self._register_level2_features()

    def _register_level2_features(self):
        """注册Level2行情特征"""
        # 高频买卖压力特征
        self.engine.register_feature(FeatureConfig(
            name="HIGH_FREQ_PRESSURE",
            feature_type=FeatureType.LEVEL2,
            params={"window": self.config.window_size},
            dependencies=["bid", "ask", "trade"]
        ))

        # 瞬时流动性特征
        self.engine.register_feature(FeatureConfig(
            name="INSTANT_LIQUIDITY",
            feature_type=FeatureType.LEVEL2,
            params={"depth": self.config.depth},
            dependencies=["bid", "ask"],
            a_share_specific=self.config.a_share_specific
        ))

        # 大单追踪特征
        self.engine.register_feature(FeatureConfig(
            name="LARGE_ORDER_TRACKING",
            feature_type=FeatureType.LEVEL2,
            params={"threshold": 100000},  # 10万以上算大单
            dependencies=["trade"],
            a_share_specific=self.config.a_share_specific
        ))

    def update_tick_buffer(self, tick_data: Dict[str, float]):
        """更新tick数据缓冲区"""
        if self.buffer_index >= self.config.tick_buffer_size:
            self.buffer_index = 0  # 循环缓冲区

        self.tick_buffer[self.buffer_index] = [
            tick_data["price"],
            tick_data["volume"],
            tick_data["timestamp"],
            1 if tick_data["direction"] == "buy" else -1
        ]
        self.buffer_index += 1

    @staticmethod
    @jit(nopython=True)
    def _calculate_pressure_numba(tick_buffer: np.ndarray, window: int) -> float:
        """计算高频买卖压力"""
        if len(tick_buffer) < window:
            return 0.0

        recent = tick_buffer[-window:]
        buy_vol = np.sum(recent[recent[:, 3] > 0, 1])
        sell_vol = np.sum(recent[recent[:, 3] < 0, 1])
        return (buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-6)

    def calculate_high_freq_pressure(self) -> float:
        """计算高频买卖压力"""
        return self._calculate_pressure_numba(
            self.tick_buffer[:self.buffer_index],
            self.config.window_size
        )

    def calculate_instant_liquidity(self, order_book: Dict[str, np.ndarray]) -> float:
        """计算瞬时流动性"""
        bid, ask = order_book["bid"], order_book["ask"]
        bid_vol = np.sum(bid[:self.config.depth, 1])
        ask_vol = np.sum(ask[:self.config.depth, 1])
        return (bid_vol + ask_vol) / 2

    def track_large_orders(self, trades: np.ndarray) -> Dict[str, float]:
        """追踪大单交易"""
        large_buys = trades[(trades[:, 1] > 0) & (trades[:, 2] > 100000)]
        large_sells = trades[(trades[:, 1] < 0) & (trades[:, 2] > 100000)]

        return {
            "large_buy_vol": np.sum(large_buys[:, 2]) if len(large_buys) > 0 else 0.0,
            "large_sell_vol": np.sum(large_sells[:, 2]) if len(large_sells) > 0 else 0.0
        }

    def calculate_a_share_features(self, order_book: Dict[str, np.ndarray]) -> Dict[str, float]:
        """计算A股特有Level2特征"""
        if not self.config.a_share_specific:
            return {}

        # 涨跌停板附近挂单分析
        upper_limit = order_book.get("upper_limit", 0)
        lower_limit = order_book.get("lower_limit", 0)

        bid_pressure = np.sum(order_book["bid"][:, 0] >= upper_limit * 0.99)
        ask_pressure = np.sum(order_book["ask"][:, 0] <= lower_limit * 1.01)

        return {
            "limit_order_pressure": (bid_pressure - ask_pressure) / self.config.depth
        }

    def calculate_all_features(self, order_book: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """批量计算所有Level2特征"""
        # 基础订单簿特征
        order_book_features = self.order_book_analyzer.calculate_all_features(order_book)

        # Level2特有特征
        pressure = self.calculate_high_freq_pressure()
        liquidity = self.calculate_instant_liquidity(order_book)
        large_orders = self.track_large_orders(order_book.get("trade", np.empty((0, 3))))

        # A股特有特征
        a_share_features = self.calculate_a_share_features(order_book)

        return {
            **order_book_features,
            "HIGH_FREQ_PRESSURE": {"value": pressure},
            "INSTANT_LIQUIDITY": {"value": liquidity},
            "LARGE_ORDER_TRACKING": large_orders,
            **{k: {"value": v} for k, v in a_share_features.items()}
        }

class CppLevel2Analyzer(Level2Analyzer):
    """C++加速的Level2分析器"""

    def __init__(self, feature_engine: FeatureEngineer, config: Level2Config = None):
        super().__init__(feature_engine, config)
        # 加载C++扩展
        try:
            from .level2_cpp import Level2Analyzer as CppAnalyzer
            self.cpp_analyzer = CppAnalyzer(
                depth=config.depth,
                window_size=config.window_size,
                buffer_size=config.tick_buffer_size
            )
        except ImportError:
            self.cpp_analyzer = None

    def calculate_all_features(self, order_book: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """使用C++加速的批量特征计算"""
        if self.cpp_analyzer:
            return self.cpp_analyzer.calculate_all(order_book)
        return super().calculate_all_features(order_book)
