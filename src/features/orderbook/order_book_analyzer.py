"""订单簿分析器模块"""
import numpy as np
from numba import jit
from typing import Dict, List, Tuple
from dataclasses import dataclass
from ..processors.feature_engineer import FeatureEngineer
from ..feature_config import FeatureType, FeatureConfig

@dataclass
class OrderBookConfig:
    """订单簿分析配置"""
    depth: int = 10  # 分析深度
    window_size: int = 20  # 滑动窗口大小
    a_share_specific: bool = True  # 是否启用A股特有特征

class OrderBookAnalyzer:
    """高性能订单簿分析器"""

    def __init__(self, feature_engine: FeatureEngineer, config: OrderBookConfig = None):
        self.engine = feature_engine
        self.config = config or OrderBookConfig()

        # 注册订单簿特征
        self._register_order_book_features()

    def _register_order_book_features(self):
        """注册订单簿特征到特征引擎"""
        # 基础订单簿特征
        self.engine.register_feature(FeatureConfig(
            name="ORDER_BOOK_IMBALANCE",
            feature_type=FeatureType.ORDER_BOOK,
            params={"depth": self.config.depth},
            dependencies=["bid", "ask"]
        ))

        # 大单冲击成本
        self.engine.register_feature(FeatureConfig(
            name="LARGE_ORDER_IMPACT",
            feature_type=FeatureType.ORDER_BOOK,
            params={"window": self.config.window_size},
            dependencies=["bid_volume", "ask_volume"],
            a_share_specific=self.config.a_share_specific
        ))

        # 隐藏流动性检测
        self.engine.register_feature(FeatureConfig(
            name="HIDDEN_LIQUIDITY",
            feature_type=FeatureType.ORDER_BOOK,
            params={"threshold": 0.3},
            dependencies=["bid", "ask", "trade"],
            a_share_specific=self.config.a_share_specific
        ))

        # A股特有特征
        if self.config.a_share_specific:
            self._register_a_share_features()

    def _register_a_share_features(self):
        """注册A股特有订单簿特征"""
        # 涨跌停压力
        self.engine.register_feature(FeatureConfig(
            name="PRICE_LIMIT_PRESSURE",
            feature_type=FeatureType.ORDER_BOOK,
            params={"depth": 5},
            dependencies=["bid", "ask", "prev_close"],
            a_share_specific=True
        ))

        # 主力资金流向
        self.engine.register_feature(FeatureConfig(
            name="MAIN_CAPITAL_FLOW",
            feature_type=FeatureType.ORDER_BOOK,
            params={"threshold": 100000},  # 10万以上算大单
            dependencies=["bid", "ask", "trade"],
            a_share_specific=True
        ))

    @staticmethod
    @jit(nopython=True)
    def _calculate_imbalance_numba(bid: np.ndarray, ask: np.ndarray, depth: int) -> float:
        """使用numba加速的订单簿不平衡度计算"""
        bid_vol = np.sum(bid[:depth, 1])
        ask_vol = np.sum(ask[:depth, 1])
        return (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-6)

    def calculate_imbalance(self, bid: np.ndarray, ask: np.ndarray) -> float:
        """计算订单簿不平衡度"""
        return self._calculate_imbalance_numba(bid, ask, self.config.depth)

    @staticmethod
    @jit(nopython=True)
    def _calculate_impact_cost_numba(bid_vol: np.ndarray, ask_vol: np.ndarray,
                                   window: int) -> Tuple[float, float]:
        """使用numba加速的冲击成本计算"""
        # 计算买卖方大单比例
        bid_large = np.sum(bid_vol[-window:] > 100000) / window
        ask_large = np.sum(ask_vol[-window:] > 100000) / window
        return bid_large, ask_large

    def calculate_impact_cost(self, bid_vol: np.ndarray, ask_vol: np.ndarray) -> Dict[str, float]:
        """计算大单冲击成本"""
        bid_large, ask_large = self._calculate_impact_cost_numba(
            bid_vol, ask_vol, self.config.window_size
        )
        return {
            "bid_impact": bid_large,
            "ask_impact": ask_large
        }

    @staticmethod
    @jit(nopython=True)
    def _detect_hidden_liquidity_numba(bid: np.ndarray, ask: np.ndarray,
                                     trades: np.ndarray, threshold: float) -> float:
        """使用numba加速的隐藏流动性检测"""
        mid_price = (bid[0, 0] + ask[0, 0]) / 2
        large_trades = trades[trades[:, 1] > threshold * mid_price]
        if len(large_trades) == 0:
            return 0.0
        return len(large_trades) / len(trades)

    def detect_hidden_liquidity(self, bid: np.ndarray, ask: np.ndarray,
                               trades: np.ndarray) -> float:
        """检测隐藏流动性"""
        return self._detect_hidden_liquidity_numba(
            bid, ask, trades, 0.3  # 30%阈值
        )

    def calculate_a_share_features(self, order_book: Dict[str, np.ndarray],
                                 prev_close: float) -> Dict[str, float]:
        """计算A股特有订单簿特征"""
        if not self.config.a_share_specific:
            return {}

        bid, ask = order_book["bid"], order_book["ask"]
        trades = order_book.get("trade", np.empty((0, 2)))

        # 涨跌停压力
        upper_limit = prev_close * 1.1
        lower_limit = prev_close * 0.9
        bid_pressure = np.sum(bid[:5, 0] >= upper_limit * 0.99) / 5
        ask_pressure = np.sum(ask[:5, 0] <= lower_limit * 1.01) / 5

        # 主力资金流向
        main_flow = 0.0
        if len(trades) > 0:
            large_buys = np.sum(trades[(trades[:, 1] > 0) & (trades[:, 2] > 100000), 2])
            large_sells = np.sum(trades[(trades[:, 1] < 0) & (trades[:, 2] > 100000), 2])
            main_flow = (large_buys - large_sells) / (large_buys + large_sells + 1e-6)

        return {
            "PRICE_LIMIT_PRESSURE": (bid_pressure - ask_pressure),
            "MAIN_CAPITAL_FLOW": main_flow
        }

    def calculate_all_features(self, order_book: Dict[str, np.ndarray],
                             prev_close: float = None) -> Dict[str, Dict[str, float]]:
        """批量计算所有订单簿特征"""
        bid, ask = order_book["bid"], order_book["ask"]
        bid_vol = order_book.get("bid_volume", np.zeros_like(bid[:, 1]))
        ask_vol = order_book.get("ask_volume", np.zeros_like(ask[:, 1]))
        trades = order_book.get("trade", np.empty((0, 2)))

        # 基础特征
        imbalance = self.calculate_imbalance(bid, ask)
        impact_cost = self.calculate_impact_cost(bid_vol, ask_vol)
        hidden_liquidity = self.detect_hidden_liquidity(bid, ask, trades)

        # A股特有特征
        a_share_features = self.calculate_a_share_features(order_book, prev_close) if prev_close else {}

        return {
            "ORDER_BOOK_IMBALANCE": {"value": imbalance},
            "LARGE_ORDER_IMPACT": impact_cost,
            "HIDDEN_LIQUIDITY": {"value": hidden_liquidity},
            **{k: {"value": v} for k, v in a_share_features.items()}
        }

class CppOrderBookAnalyzer(OrderBookAnalyzer):
    """C++加速的订单簿分析器"""

    def __init__(self, feature_engine: FeatureEngineer, config: OrderBookConfig = None):
        super().__init__(feature_engine, config)
        # 加载C++扩展
        try:
            from .order_book_cpp import OrderBookAnalyzer as CppAnalyzer
            self.cpp_analyzer = CppAnalyzer(
                depth=config.depth,
                window_size=config.window_size
            )
        except ImportError:
            self.cpp_analyzer = None

    def calculate_all_features(self, order_book: Dict[str, np.ndarray],
                             prev_close: float = None) -> Dict[str, Dict[str, float]]:
        """使用C++加速的批量特征计算"""
        if self.cpp_analyzer:
            return self.cpp_analyzer.calculate_all(order_book, prev_close)
        return super().calculate_all_features(order_book, prev_close)
