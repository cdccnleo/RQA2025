#!/usr/bin/env python3
"""
RQA2025 订单簿分析器
分析订单簿数据，提取市场微观结构特征和交易信号
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from collections import deque


logger = logging.getLogger(__name__)


class OrderBookSignal(Enum):

    """订单簿信号枚举"""
    LIQUIDITY_IMBALANCE = "liquidity_imbalance"      # 流动性不平衡
    LARGE_ORDER_IMMINENT = "large_order_imminent"    # 大单临近
    MARKET_MAKING_OPPORTUNITY = "mm_opportunity"     # 做市机会
    SPREAD_EXPANSION = "spread_expansion"           # 价差扩大
    SPREAD_CONTRACTION = "spread_contraction"       # 价差缩小
    ORDER_BOOK_SKEW = "order_book_skew"            # 订单簿倾斜
    DEPTH_IMBALANCE = "depth_imbalance"            # 深度不平衡


@dataclass
class OrderBookAnalysis:

    """订单簿分析结果"""
    symbol: str
    timestamp: datetime

    # 基本指标
    spread_bps: float
    mid_price: float
    weighted_mid_price: float

    # 流动性指标
    bid_liquidity: float
    ask_liquidity: float
    liquidity_ratio: float
    market_depth_5: float

    # 不平衡指标
    volume_imbalance: float
    order_imbalance: float
    price_imbalance: float

    # 市场压力指标
    buying_pressure: float
    selling_pressure: float

    # 波动性指标
    spread_volatility: float
    price_volatility: float

    # 信号
    signals: List[Dict[str, Any]]

    # 预测指标
    predicted_price_move: float
    confidence_score: float


class OrderBookAnalyzer:

    """订单簿分析器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        self.config = config or {}

        # 配置参数
        self.analysis_window = self.config.get('analysis_window', 100)  # 分析窗口
        self.signal_threshold = self.config.get('signal_threshold', 0.7)  # 信号阈值
        self.depth_levels = self.config.get('depth_levels', 5)  # 深度层数

        # 历史数据存储
        self.order_book_history: Dict[str, deque] = {}
        self.analysis_history: Dict[str, deque] = {}

        # 统计指标
        self.stats = {
            'total_analyses': 0,
            'signals_generated': 0,
            'prediction_accuracy': 0.0,
            'average_confidence': 0.0
        }

        # 信号检测器
        self.signal_detectors = {
            'liquidity_imbalance': self._detect_liquidity_imbalance,
            'large_order_imminent': self._detect_large_order,
            'spread_expansion': self._detect_spread_expansion,
            'order_book_skew': self._detect_order_book_skew,
            'depth_imbalance': self._detect_depth_imbalance
        }

        logger.info("订单簿分析器初始化完成")

    def analyze_order_book(self, symbol: str, bids: List[Tuple[float, float]],


                           asks: List[Tuple[float, float]], timestamp: datetime) -> OrderBookAnalysis:
        """
        分析订单簿

        Args:
            symbol: 交易标的
            bids: 买单列表 [(price, quantity), ...]
            asks: 卖单列表 [(price, quantity), ...]
            timestamp: 时间戳

        Returns:
            订单簿分析结果
        """
        # 存储历史数据
        if symbol not in self.order_book_history:
            self.order_book_history[symbol] = deque(maxlen=self.analysis_window)

        order_book_data = {
            'timestamp': timestamp,
            'bids': bids.copy(),
            'asks': asks.copy()
        }

        self.order_book_history[symbol].append(order_book_data)

        # 计算基本指标
        spread_bps = self._calculate_spread_bps(bids, asks)
        mid_price = self._calculate_mid_price(bids, asks)
        weighted_mid_price = self._calculate_weighted_mid_price(bids, asks)

        # 计算流动性指标
        bid_liquidity = sum(qty for _, qty in bids[:self.depth_levels])
        ask_liquidity = sum(qty for _, qty in asks[:self.depth_levels])
        liquidity_ratio = bid_liquidity / ask_liquidity if ask_liquidity > 0 else float('inf')
        market_depth_5 = self._calculate_market_depth(bids, asks, 5)

        # 计算不平衡指标
        volume_imbalance = self._calculate_volume_imbalance(bids, asks)
        order_imbalance = self._calculate_order_imbalance(bids, asks)
        price_imbalance = self._calculate_price_imbalance(bids, asks)

        # 计算市场压力
        buying_pressure = self._calculate_buying_pressure(bids, asks)
        selling_pressure = self._calculate_selling_pressure(bids, asks)

        # 计算波动性
        spread_volatility = self._calculate_spread_volatility(symbol)
        price_volatility = self._calculate_price_volatility(symbol)

        # 生成信号
        signals = self._generate_signals(symbol, bids, asks, spread_bps, volume_imbalance)

        # 预测价格走势
        predicted_price_move, confidence_score = self._predict_price_movement(
            symbol, spread_bps, volume_imbalance, order_imbalance
        )

        # 创建分析结果
        analysis = OrderBookAnalysis(
            symbol=symbol,
            timestamp=timestamp,
            spread_bps=spread_bps,
            mid_price=mid_price,
            weighted_mid_price=weighted_mid_price,
            bid_liquidity=bid_liquidity,
            ask_liquidity=ask_liquidity,
            liquidity_ratio=liquidity_ratio,
            market_depth_5=market_depth_5,
            volume_imbalance=volume_imbalance,
            order_imbalance=order_imbalance,
            price_imbalance=price_imbalance,
            buying_pressure=buying_pressure,
            selling_pressure=selling_pressure,
            spread_volatility=spread_volatility,
            price_volatility=price_volatility,
            signals=signals,
            predicted_price_move=predicted_price_move,
            confidence_score=confidence_score
        )

        # 存储分析历史
        if symbol not in self.analysis_history:
            self.analysis_history[symbol] = deque(maxlen=self.analysis_window)

        self.analysis_history[symbol].append(analysis)

        # 更新统计
        self.stats['total_analyses'] += 1
        self.stats['signals_generated'] += len(signals)

        if signals:
            avg_confidence = sum(signal.get('confidence', 0) for signal in signals) / len(signals)
            self.stats['average_confidence'] = (
                self.stats['average_confidence'] * 0.9 + avg_confidence * 0.1
            )

        logger.debug(f"订单簿分析完成: {symbol}, 信号数量: {len(signals)}")
        return analysis

    def _calculate_spread_bps(self, bids: List[Tuple[float, float]],


                              asks: List[Tuple[float, float]]) -> float:
        """计算价差（基点）"""
        if not bids or not asks:
            return 0.0

        best_bid = bids[0][0]
        best_ask = asks[0][0]

        if best_bid > 0:
            return (best_ask - best_bid) / best_bid * 10000  # 转换为基点

        return 0.0

    def _calculate_mid_price(self, bids: List[Tuple[float, float]],


                             asks: List[Tuple[float, float]]) -> float:
        """计算中间价"""
        if not bids or not asks:
            return 0.0

        best_bid = bids[0][0]
        best_ask = asks[0][0]

        return (best_bid + best_ask) / 2

    def _calculate_weighted_mid_price(self, bids: List[Tuple[float, float]],


                                      asks: List[Tuple[float, float]]) -> float:
        """计算加权中间价"""
        if not bids or not asks:
            return 0.0

        # 使用前3个价位计算加权价格
        bid_levels = bids[:3]
        ask_levels = asks[:3]

        bid_weighted = sum(price * qty for price, qty in bid_levels) / \
            sum(qty for _, qty in bid_levels)
        ask_weighted = sum(price * qty for price, qty in ask_levels) / \
            sum(qty for _, qty in ask_levels)

        return (bid_weighted + ask_weighted) / 2

    def _calculate_volume_imbalance(self, bids: List[Tuple[float, float]],


                                    asks: List[Tuple[float, float]]) -> float:
        """计算成交量不平衡"""
        bid_volume = sum(qty for _, qty in bids[:self.depth_levels])
        ask_volume = sum(qty for _, qty in asks[:self.depth_levels])

        total_volume = bid_volume + ask_volume

        if total_volume > 0:
            return (bid_volume - ask_volume) / total_volume

        return 0.0

    def _calculate_order_imbalance(self, bids: List[Tuple[float, float]],


                                   asks: List[Tuple[float, float]]) -> float:
        """计算订单不平衡"""
        bid_orders = len(bids[:self.depth_levels])
        ask_orders = len(asks[:self.depth_levels])

        total_orders = bid_orders + ask_orders

        if total_orders > 0:
            return (bid_orders - ask_orders) / total_orders

        return 0.0

    def _calculate_price_imbalance(self, bids: List[Tuple[float, float]],


                                   asks: List[Tuple[float, float]]) -> float:
        """计算价格不平衡"""
        if not bids or not asks:
            return 0.0

        bid_price_range = bids[0][0] - bids[-1][0] if len(bids) > 1 else 0
        ask_price_range = asks[-1][0] - asks[0][0] if len(asks) > 1 else 0

        if ask_price_range > 0:
            return (bid_price_range - ask_price_range) / ask_price_range

        return 0.0

    def _calculate_buying_pressure(self, bids: List[Tuple[float, float]],


                                   asks: List[Tuple[float, float]]) -> float:
        """计算买入压力"""
        # 基于买单数量和挂单量计算
        bid_intensity = sum(qty / (price ** 2) for price, qty in bids[:3])  # 价格权重
        ask_intensity = sum(qty / (price ** 2) for price, qty in asks[:3])

        if ask_intensity > 0:
            return bid_intensity / ask_intensity

        return float('inf') if bid_intensity > 0 else 0.0

    def _calculate_selling_pressure(self, bids: List[Tuple[float, float]],


                                    asks: List[Tuple[float, float]]) -> float:
        """计算卖出压力"""
        # 基于卖单数量和挂单量计算
        bid_intensity = sum(qty / (price ** 2) for price, qty in bids[:3])
        ask_intensity = sum(qty / (price ** 2) for price, qty in asks[:3])

        if bid_intensity > 0:
            return ask_intensity / bid_intensity

        return float('inf') if ask_intensity > 0 else 0.0

    def _calculate_market_depth(self, bids: List[Tuple[float, float]],


                                asks: List[Tuple[float, float]], levels: int) -> float:
        """计算市场深度"""
        bid_depth = sum(qty for _, qty in bids[:levels])
        ask_depth = sum(qty for _, qty in asks[:levels])

        return (bid_depth + ask_depth) / 2

    def _calculate_spread_volatility(self, symbol: str) -> float:
        """计算价差波动率"""
        history = self.analysis_history.get(symbol, [])
        if len(history) < 10:
            return 0.0

        spreads = [analysis.spread_bps for analysis in list(history)[-10:]]
        return np.std(spreads) if spreads else 0.0

    def _calculate_price_volatility(self, symbol: str) -> float:
        """计算价格波动率"""
        history = self.analysis_history.get(symbol, [])
        if len(history) < 10:
            return 0.0

        prices = [analysis.mid_price for analysis in list(history)[-10:]]
        if len(prices) > 1 and prices[0] > 0:
            returns = np.diff(prices) / prices[:-1]
            return np.std(returns) if returns.size > 0 else 0.0

        return 0.0

    def _generate_signals(self, symbol: str, bids: List[Tuple[float, float]],


                          asks: List[Tuple[float, float]], spread_bps: float,
                          volume_imbalance: float) -> List[Dict[str, Any]]:
        """生成交易信号"""
        signals = []

        # 检测各种信号
        for signal_type, detector in self.signal_detectors.items():
            try:
                signal = detector(symbol, bids, asks, spread_bps, volume_imbalance)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"信号检测失败 {signal_type}: {e}")

        return signals

    def _detect_liquidity_imbalance(self, symbol: str, bids: List[Tuple[float, float]],


                                    asks: List[Tuple[float, float]], spread_bps: float,
                                    volume_imbalance: float) -> Optional[Dict[str, Any]]:
        """检测流动性不平衡"""
        if abs(volume_imbalance) > 0.3:
            return {
                'type': OrderBookSignal.LIQUIDITY_IMBALANCE.value,
                'symbol': symbol,
                'direction': 'bullish' if volume_imbalance > 0 else 'bearish',
                'strength': abs(volume_imbalance),
                'confidence': min(abs(volume_imbalance) * 2, 1.0),
                'description': f"流动性{'买入' if volume_imbalance > 0 else '卖出'}偏向"
            }

        return None

    def _detect_large_order(self, symbol: str, bids: List[Tuple[float, float]],


                            asks: List[Tuple[float, float]], spread_bps: float,
                            volume_imbalance: float) -> Optional[Dict[str, Any]]:
        """检测大单临近"""
        # 检测订单簿前几级是否有异常大的挂单
        if bids and asks:
            bid_largest = max(qty for _, qty in bids[:3])
            ask_largest = max(qty for _, qty in asks[:3])

            # 计算平均挂单量
            bid_avg = sum(qty for _, qty in bids[:10]) / min(10, len(bids))
            ask_avg = sum(qty for _, qty in asks[:10]) / min(10, len(asks))

        if bid_largest > bid_avg * 3:
            return {
                'type': OrderBookSignal.LARGE_ORDER_IMMINENT.value,
                'symbol': symbol,
                'direction': 'bullish',
                'strength': bid_largest / bid_avg,
                'confidence': 0.8,
                'description': "检测到异常大买单，可能存在大资金活动"
            }

        if ask_largest > ask_avg * 3:
            return {
                'type': OrderBookSignal.LARGE_ORDER_IMMINENT.value,
                'symbol': symbol,
                'direction': 'bearish',
                'strength': ask_largest / ask_avg,
                'confidence': 0.8,
                'description': "检测到异常大卖单，可能存在大资金活动"
            }

        return None

    def _detect_spread_expansion(self, symbol: str, bids: List[Tuple[float, float]],


                                 asks: List[Tuple[float, float]], spread_bps: float,
                                 volume_imbalance: float) -> Optional[Dict[str, Any]]:
        """检测价差扩大"""
        history = self.analysis_history.get(symbol, [])
        if len(history) < 5:
            return None

        # 计算价差趋势
        recent_spreads = [analysis.spread_bps for analysis in list(history)[-5:]]
        spread_trend = np.polyfit(range(len(recent_spreads)), recent_spreads, 1)[0]

        if spread_trend > 2:  # 价差快速扩大
            return {
                'type': OrderBookSignal.SPREAD_EXPANSION.value,
                'symbol': symbol,
                'direction': 'neutral',
                'strength': spread_trend,
                'confidence': 0.6,
                'description': "订单簿价差扩大，可能存在流动性问题"
            }

        return None

    def _detect_order_book_skew(self, symbol: str, bids: List[Tuple[float, float]],


                                asks: List[Tuple[float, float]], spread_bps: float,
                                volume_imbalance: float) -> Optional[Dict[str, Any]]:
        """检测订单簿倾斜"""
        if len(bids) < 3 or len(asks) < 3:
            return None

        # 计算订单分布偏度
        bid_distribution = [qty for _, qty in bids[:5]]
        ask_distribution = [qty for _, qty in asks[:5]]

        bid_skewness = self._calculate_skewness(bid_distribution)
        ask_skewness = self._calculate_skewness(ask_distribution)

        if abs(bid_skewness) > 1.5:
            return {
                'type': OrderBookSignal.ORDER_BOOK_SKEW.value,
                'symbol': symbol,
                'direction': 'bullish' if bid_skewness > 0 else 'bearish',
                'strength': abs(bid_skewness),
                'confidence': 0.7,
                'description': f"买单分布{'集中' if bid_skewness > 0 else '分散'}"
            }

        if abs(ask_skewness) > 1.5:
            return {
                'type': OrderBookSignal.ORDER_BOOK_SKEW.value,
                'symbol': symbol,
                'direction': 'bearish' if ask_skewness > 0 else 'bullish',
                'strength': abs(ask_skewness),
                'confidence': 0.7,
                'description': f"卖单分布{'集中' if ask_skewness > 0 else '分散'}"
            }

        return None

    def _detect_depth_imbalance(self, symbol: str, bids: List[Tuple[float, float]],


                                asks: List[Tuple[float, float]], spread_bps: float,
                                volume_imbalance: float) -> Optional[Dict[str, Any]]:
        """检测深度不平衡"""
        if len(bids) < 2 or len(asks) < 2:
            return None

        # 计算深度分布
        bid_depth_ratio = bids[0][1] / (bids[1][1] + 1e-8)
        ask_depth_ratio = asks[0][1] / (asks[1][1] + 1e-8)

        if bid_depth_ratio > 5:
            return {
                'type': OrderBookSignal.DEPTH_IMBALANCE.value,
                'symbol': symbol,
                'direction': 'bullish',
                'strength': bid_depth_ratio,
                'confidence': 0.75,
                'description': "买单深度集中，可能存在大资金买入"
            }

        if ask_depth_ratio > 5:
            return {
                'type': OrderBookSignal.DEPTH_IMBALANCE.value,
                'symbol': symbol,
                'direction': 'bearish',
                'strength': ask_depth_ratio,
                'confidence': 0.75,
                'description': "卖单深度集中，可能存在大资金卖出"
            }

        return None

    def _calculate_skewness(self, data: List[float]) -> float:
        """计算偏度"""
        if len(data) < 3:
            return 0.0

        mean_val = np.mean(data)
        std_val = np.std(data)

        if std_val == 0:
            return 0.0

        return np.mean(((np.array(data) - mean_val) / std_val) ** 3)

    def _predict_price_movement(self, symbol: str, spread_bps: float,


                                volume_imbalance: float, order_imbalance: float) -> Tuple[float, float]:
        """预测价格走势"""
        # 简化的价格预测模型
        # 在实际应用中，这应该是一个训练好的机器学习模型

        # 基于订单簿特征的简单线性模型
        spread_factor = -spread_bps * 0.001  # 价差越大，预测价格变化越小
        volume_factor = volume_imbalance * 0.005  # 成交量不平衡影响
        order_factor = order_imbalance * 0.003  # 订单不平衡影响

        predicted_move = spread_factor + volume_factor + order_factor

        # 计算置信度
        confidence = min(1.0, abs(predicted_move) * 10)

        return predicted_move, confidence

    def get_analysis_history(self, symbol: str, n: int = 10) -> List[OrderBookAnalysis]:
        """获取分析历史"""
        history = self.analysis_history.get(symbol, [])
        return list(history)[-n:] if history else []

    def get_signal_statistics(self, symbol: str) -> Dict[str, Any]:
        """获取信号统计"""
        history = self.analysis_history.get(symbol, [])
        if not history:
            return {}

        all_signals = []
        for analysis in history:
            all_signals.extend(analysis.signals)

        signal_counts = {}
        for signal in all_signals:
            signal_type = signal.get('type', 'unknown')
            signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1

        return {
            'total_signals': len(all_signals),
            'signal_types': signal_counts,
            'most_common_signal': max(signal_counts, key=signal_counts.get) if signal_counts else None
        }

    def get_market_health_score(self, symbol: str) -> float:
        """获取市场健康评分"""
        history = self.analysis_history.get(symbol, [])
        if len(history) < 5:
            return 0.5

        recent = list(history)[-5:]

        # 基于多个指标计算健康评分
        spread_score = 1.0 / (1.0 + np.mean([r.spread_bps for r in recent]) / 100)
        liquidity_score = np.mean([r.liquidity_score for r in recent]) if hasattr(
            recent[0], 'liquidity_score') else 0.5
        imbalance_score = 1.0 - np.mean([abs(r.volume_imbalance) for r in recent])

        health_score = (spread_score + liquidity_score + imbalance_score) / 3

        return max(0.0, min(1.0, health_score))

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()
