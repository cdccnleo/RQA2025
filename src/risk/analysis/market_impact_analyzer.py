import time
#!/usr/bin/env python3
"""
# 市场冲击分析误
分析大额交易对市场价格和波动性的影响效应，提供市场冲击成本估误"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict, deque
import json
from scipy import stats
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)


class MarketImpactType(Enum):

    """市场冲击类型"""
    PRICE_IMPACT = "price_impact"           # 价格冲击
    VOLATILITY_IMPACT = "volatility_impact"  # 波动率冲误
    LIQUIDITY_IMPACT = "liquidity_impact"   # 流动性冲误
    MOMENTUM_IMPACT = "momentum_impact"     # 动量冲击
    INFORMATION_IMPACT = "information_impact"  # 信息冲击


class OrderType(Enum):

    """订单类型"""
    MARKET_ORDER = "market_order"           # 市价误
    LIMIT_ORDER = "limit_order"            # 限价误
    ICEBERG_ORDER = "iceberg_order"        # 冰山订单
    TWAP_ORDER = "twap_order"              # 时间加权平均价格订单
    VWAP_ORDER = "vwap_order"              # 成交量加权平均价格订误


@dataclass
class MarketImpactConfig:

    """市场冲击分析配置"""
    analysis_window_minutes: int = 60        # 分析时间窗口（分钟）
    impact_decay_factor: float = 0.1         # 冲击衰减因子
    min_order_size_threshold: float = 0.01   # 最小订单规模阈值（占日均成交量比例误
    max_impact_lookback_days: int = 30       # 最大历史冲击数据回溯天误
    enable_real_time_analysis: bool = True   # 启用实时分析
    enable_historical_calibration: bool = True  # 启用历史数据校准


@dataclass
class MarketImpactResult:

    """市场冲击分析结果"""
    order_id: str
    symbol: str
    order_type: OrderType
    order_size: float
    market_cap: float
    analysis_timestamp: datetime

    # 冲击分析结果
    price_impact: float = 0.0                # 价格冲击（基点）
    volatility_impact: float = 0.0           # 波动率冲击（百分比）
    liquidity_impact: float = 0.0            # 流动性冲击（百分比）
    momentum_impact: float = 0.0             # 动量冲击（百分比误
    # 成本估算
    estimated_cost: float = 0.0              # 估算总成本（元）
    slippage_cost: float = 0.0               # 滑点成本（元误
    market_impact_cost: float = 0.0          # 市场冲击成本（元误
    # 置信区间
    price_impact_ci: Tuple[float, float] = (0.0, 0.0)
    cost_estimate_ci: Tuple[float, float] = (0.0, 0.0)

    # 元数误    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketImpactMetrics:

    """市场冲击指标"""
    symbol: str
    timestamp: datetime

    # 市场深度指标
    bid_ask_spread: float = 0.0              # 买卖价差
    market_depth_level1: float = 0.0         # 一级市场深误
    market_depth_level5: float = 0.0         # 五级市场深度

    # 成交量指误
    daily_volume: float = 0.0                # 日成交量
    hourly_volume: float = 0.0               # 小时成交误
    volume_volatility: float = 0.0           # 成交量波动率

    # 价格指标
    price_volatility: float = 0.0            # 价格波动误
    price_trend: float = 0.0                 # 价格趋势
    momentum_indicator: float = 0.0          # 动量指标

    # 冲击敏感误
    impact_sensitivity: float = 0.0           # 冲击敏感度系误


class MarketImpactAnalyzer:

    """市场冲击分析误"""

    def __init__(self, config: Optional[MarketImpactConfig] = None):
        self.config = config or MarketImpactConfig()
        self.lock = threading.RLock()

        # 数据缓存
        self.market_data_cache = defaultdict(lambda: deque(maxlen=1000))
        self.impact_history = defaultdict(lambda: deque(maxlen=5000))
        self.symbol_metrics = {}

        # 冲击模型参数
        self.calibration_data = defaultdict(list)

        logger.info("市场冲击分析器初始化完成")
    def analyze_market_impact(self, order_data: Dict[str, Any]) -> MarketImpactResult:
        with self.lock:
            try:
                logger.info(f"开始分析订单{order_data.get('order_id')} 的市场冲击")

                # 解析订单数据
                order_info = self._parse_order_data(order_data)

                # 获取市场数据
                market_data = self._get_market_data(order_info['symbol'])

                # 计算市场冲击
                impact_result = self._calculate_market_impact(order_info, market_data)

                # 估算交易成本
                cost_estimate = self._estimate_transaction_cost(impact_result, market_data)

                # 更新冲击历史
                self._update_impact_history(order_info['symbol'], impact_result)

                # 创建完整结果
                result = MarketImpactResult(
                    order_id=order_info['order_id'],
                    symbol=order_info['symbol'],
                    order_type=order_info['order_type'],
                    order_size=order_info['order_size'],
                    market_cap=market_data.get('market_cap', 0),
                    analysis_timestamp=datetime.now(),
                    price_impact=impact_result['price_impact'],
                    volatility_impact=impact_result['volatility_impact'],
                    liquidity_impact=impact_result['liquidity_impact'],
                    momentum_impact=impact_result['momentum_impact'],
                    estimated_cost=cost_estimate['total_cost'],
                    slippage_cost=cost_estimate['slippage_cost'],
                    market_impact_cost=cost_estimate['impact_cost'],
                    price_impact_ci=impact_result['price_impact_ci'],
                    cost_estimate_ci=cost_estimate['cost_ci'],
                    metadata={
                        'analysis_window': self.config.analysis_window_minutes,
                        'model_version': 'v1.0',
                        'market_conditions': market_data.get('market_conditions', {})
                    }
                )

                logger.info(f"市场冲击分析完成: 订单 {order_info['order_id']}, "
                           f"价格冲击 {result.price_impact:.2f} 基点, "
                           f"估算成本 {result.estimated_cost:.2f} 元")

                return result

            except Exception as e:
                logger.error(f"市场冲击分析失败: {e}")
                raise


    def _parse_order_data(self, order_data: Dict[str, Any]) -> Dict[str, Any]:

        """解析订单数据"""
        try:
            order_info = {
                'order_id': order_data.get('order_id', f"order_{datetime.now().timestamp()}"),
                'symbol': order_data.get('symbol', ''),
                'order_type': OrderType(order_data.get('order_type', 'market_order')),
                'order_size': float(order_data.get('order_size', 0)),
                'order_price': float(order_data.get('order_price', 0)),
                'urgency': order_data.get('urgency', 'normal'),  # normal, high, low
                'time_horizon': int(order_data.get('time_horizon', 1))  # 执行时间（分钟）
                }

            if not order_info['symbol']:
                raise ValueError("订单数据中缺少交易标误")

            if order_info['order_size'] <= 0:
                raise ValueError("订单规模必须大于0")

            return order_info

        except Exception as e:
            logger.error(f"解析订单数据失败: {e}")
            raise


    def _get_market_data(self, symbol: str) -> Dict[str, Any]:

        """获取市场数据"""
        try:
            # 从缓存获取最近的市场数据
            if symbol in self.market_data_cache and self.market_data_cache[symbol]:
                recent_data = self.market_data_cache[symbol][-1]
                return recent_data

            # 如果缓存中没有数据，返回默认值
            return {
                'symbol': symbol,
                'last_price': 100.0,
                'daily_volume': 1000000,
                'market_cap': 1000000000,
                'bid_ask_spread': 0.001,
                'market_depth_level1': 10000,
                'volatility': 0.02,
                'market_conditions': {
                    'trend': 'sideways',
                    'liquidity': 'normal',
                    'volatility_regime': 'normal'
                }
            }

        except Exception as e:
            logger.error(f"获取市场数据失败: {e}")
            return {}


    def _calculate_market_impact(self, order_info: Dict[str, Any],
                                 market_data: Dict[str, Any]) -> Dict[str, Any]:
        """计算市场冲击"""
        try:
            symbol = order_info['symbol']
            order_size = order_info['order_size']
            order_type = order_info['order_type']
            time_horizon = order_info['time_horizon']

            # 获取市场指标
            market_metrics = self._calculate_market_metrics(symbol, market_data)

            # 计算订单相对规模
            relative_size = self._calculate_relative_order_size(order_size, market_data)

            # 计算基础价格冲击
            price_impact = self._calculate_price_impact(
                relative_size, market_metrics, order_type, time_horizon
            )

            # 计算波动率冲击
            volatility_impact = self._calculate_volatility_impact(
                relative_size, market_metrics, order_type
            )

            # 计算流动性冲击
            liquidity_impact = self._calculate_liquidity_impact(
                relative_size, market_metrics, time_horizon
            )

            # 计算动量冲击
            momentum_impact = self._calculate_momentum_impact(
                relative_size, market_metrics, order_type
            )

            # 计算置信区间
            price_impact_ci = self._calculate_impact_confidence_interval(
                price_impact, relative_size, symbol
            )

            return {
                'price_impact': price_impact,
                'volatility_impact': volatility_impact,
                'liquidity_impact': liquidity_impact,
                'momentum_impact': momentum_impact,
                'relative_size': relative_size,
                'price_impact_ci': price_impact_ci
            }

        except Exception as e:
            logger.error(f"计算市场冲击失败: {e}")
            return {
                'price_impact': 0.0,
                'volatility_impact': 0.0,
                'liquidity_impact': 0.0,
                'momentum_impact': 0.0,
                'relative_size': 0.0,
                'price_impact_ci': (0.0, 0.0)
            }


    def _calculate_market_metrics(self, symbol: str, market_data: Dict[str, Any]) -> MarketImpactMetrics:

        """计算市场指标"""
        try:
            metrics = MarketImpactMetrics(
                symbol=symbol,
                timestamp=datetime.now()
            )

            # 基础指标
            metrics.bid_ask_spread = market_data.get('bid_ask_spread', 0.001)
            metrics.daily_volume = market_data.get('daily_volume', 1000000)
            metrics.hourly_volume = market_data.get('hourly_volume', metrics.daily_volume / 24)
            metrics.price_volatility = market_data.get('volatility', 0.02)

            # 市场深度
            metrics.market_depth_level1 = market_data.get('market_depth_level1', 10000)
            metrics.market_depth_level5 = market_data.get('market_depth_level5', 50000)

            # 成交量波动率（简化计算）
            metrics.volume_volatility = 0.15  # 默认15%

            # 价格趋势（简化计算）
            metrics.price_trend = market_data.get('price_trend', 0.0)

            # 动量指标（简化计算）
            metrics.momentum_indicator = market_data.get('momentum', 0.0)

            # 冲击敏感度
            metrics.impact_sensitivity = self._calculate_impact_sensitivity(metrics)

            # 保存指标
            self.symbol_metrics[symbol] = metrics

            return metrics

        except Exception as e:
            logger.error(f"计算市场指标失败: {e}")
            return MarketImpactMetrics(symbol=symbol, timestamp=datetime.now())


    def _calculate_relative_order_size(self, order_size: float, market_data: Dict[str, Any]) -> float:

        """计算订单相对规模"""
        try:
            daily_volume = market_data.get('daily_volume', 1000000)
            if daily_volume <= 0:
                return 0.0

            # 计算订单规模占日均成交量的比例
            relative_size = order_size / daily_volume

            # 对大额订单进行调整（考虑市场深度限制）
            if relative_size > 0.1:  # 大于10%的订单
                market_depth = market_data.get('market_depth_level1', 10000)
                depth_ratio = min(order_size / market_depth, 1.0)
                relative_size = relative_size * (1 + depth_ratio)

            return relative_size

        except Exception as e:
            logger.error(f"计算相对订单规模失败: {e}")
            return 0.0


    def _calculate_price_impact(self, relative_size: float, market_metrics: MarketImpactMetrics,
                                order_type: OrderType, time_horizon: int) -> float:
        """计算价格冲击"""
        try:
            # 基础冲击模型：平方根模型
            base_impact = self.config.impact_decay_factor * np.sqrt(relative_size)

            # 订单类型调整因子
            type_factor = {
                OrderType.MARKET_ORDER: 1.5,      # 市价单冲击最大
                OrderType.LIMIT_ORDER: 0.8,       # 限价单冲击较小
                OrderType.ICEBERG_ORDER: 0.6,     # 冰山订单冲击最小
                OrderType.TWAP_ORDER: 0.7,        # TWAP订单中等冲击
                OrderType.VWAP_ORDER: 0.7         # VWAP订单中等冲击
            }.get(order_type, 1.0)

            # 时间跨度调整因子
            time_factor = 1.0 / np.sqrt(max(time_horizon, 1))

            # 市场条件调整因子
            market_factor = self._calculate_market_condition_factor(market_metrics)

            # 计算总价格冲击（基点）
            price_impact = base_impact * type_factor * time_factor * market_factor * 100  # 转换为基点
            return max(0.0, price_impact)

        except Exception as e:
            logger.error(f"计算价格冲击失败: {e}")
            return 0.0


    def _calculate_volatility_impact(self, relative_size: float, market_metrics: MarketImpactMetrics,
                                     order_type: OrderType) -> float:
        """计算波动率冲击"""
        try:
            # 大额订单可能增加市场波动性
            base_volatility_impact = relative_size * 0.5  # 50%的基础冲击

            # 订单类型调整
            if order_type == OrderType.MARKET_ORDER:
                base_volatility_impact *= 1.2

            # 市场波动率调整
            volatility_factor = 1 + market_metrics.price_volatility * 2

            return base_volatility_impact * volatility_factor

        except Exception as e:
            logger.error(f"计算波动率冲击失败 {e}")
            return 0.0


    def _calculate_liquidity_impact(self, relative_size: float, market_metrics: MarketImpactMetrics,
                                    time_horizon: int) -> float:
        """计算流动性冲击"""
        try:
            # 流动性冲击主要影响买卖价差
            base_liquidity_impact = relative_size * 0.3

            # 时间跨度调整（较长执行时间降低冲击）
            time_factor = 1.0 / np.sqrt(max(time_horizon, 1))

            # 市场深度调整（深度越好，冲击越小）
            depth_factor = 1.0
            if market_metrics.market_depth_level1 > 0:
                depth_factor = 1 / (1 + market_metrics.market_depth_level1 / 10000)

            return base_liquidity_impact * time_factor * depth_factor

        except Exception as e:
            logger.error(f"计算流动性冲击失败 {e}")
            return 0.0


    def _calculate_momentum_impact(self, relative_size: float, market_metrics: MarketImpactMetrics,
                                   order_type: OrderType) -> float:
        """计算动量冲击"""
        try:
            # 动量冲击基于市场趋势和订单方向
            base_momentum = relative_size * 0.2

            # 趋势调整
            trend_factor = 1 + abs(market_metrics.price_trend) * 0.5

            # 动量调整
            momentum_factor = 1 + market_metrics.momentum_indicator * 0.3

            return base_momentum * trend_factor * momentum_factor

        except Exception as e:
            logger.error(f"计算动量冲击失败: {e}")
            return 0.0


    def _calculate_market_condition_factor(self, market_metrics: MarketImpactMetrics) -> float:

        """计算市场条件调整因子"""
        try:
            factor = 1.0

            # 波动率调整
            if market_metrics.price_volatility > 0.03:  # 高波动
                factor *= 1.2
            elif market_metrics.price_volatility < 0.01:  # 低波动
                factor *= 0.8

            # 流动性调整
            if market_metrics.bid_ask_spread > 0.005:  # 价差大
                factor *= 1.3
            elif market_metrics.bid_ask_spread < 0.001:  # 价差小
                factor *= 0.9

            # 成交量调整
            volume_ratio = market_metrics.hourly_volume / max(market_metrics.daily_volume / 24, 1)
            if volume_ratio < 0.5:  # 成交量低
                factor *= 1.4
            elif volume_ratio > 1.5:  # 成交量高
                factor *= 0.8

            return factor

        except Exception as e:
            logger.error(f"计算市场条件因子失败: {e}")
            return 1.0


    def _calculate_impact_sensitivity(self, market_metrics: MarketImpactMetrics) -> float:

        """计算冲击敏感度"""
        try:
            # 基于市场条件的敏感度计算
            volatility_sensitivity = market_metrics.price_volatility * 2
            liquidity_sensitivity = market_metrics.bid_ask_spread * 100
            volume_sensitivity = 1 / (market_metrics.hourly_volume / 1000 + 1)

            sensitivity = (volatility_sensitivity + liquidity_sensitivity + volume_sensitivity) / 3
            return min(max(sensitivity, 0.1), 2.0)  # 限制在0.1 - 2.0之间

        except Exception as e:
            logger.error(f"计算冲击敏感度失败 {e}")
            return 1.0


    def _calculate_impact_confidence_interval(self, price_impact: float, relative_size: float,
                                             symbol: str) -> Tuple[float, float]:
        """计算冲击置信区间"""
        try:
            # 使用历史数据估算置信区间
            historical_impacts = []
            for impact_data in self.impact_history[symbol]:
                if abs(impact_data.get('relative_size', 0) - relative_size) < 0.1:  # 相似规模
                    historical_impacts.append(impact_data.get('price_impact', 0))

            if len(historical_impacts) >= 5:
                # 使用历史数据的标准差估算区间
                std_dev = np.std(historical_impacts)
                lower_bound = price_impact - 1.96 * std_dev
                upper_bound = price_impact + 1.96 * std_dev
            else:
                # 使用经验值估算区间
                lower_bound = price_impact * 0.7
                upper_bound = price_impact * 1.3

            return max(0.0, lower_bound), upper_bound

        except Exception as e:
            logger.error(f"计算冲击置信区间失败: {e}")
            return price_impact * 0.8, price_impact * 1.2


    def _estimate_transaction_cost(self, impact_result: Dict[str, Any],
                                   market_data: Dict[str, Any]) -> Dict[str, Any]:
        """估算交易成本"""
        try:
            order_size = impact_result.get('order_size', 0)
            price_impact = impact_result.get('price_impact', 0)
            last_price = market_data.get('last_price', 100.0)

            # 市场冲击成本
            impact_cost = order_size * last_price * (price_impact / 10000)  # 基点转换为价格
            # 滑点成本（估算）
            slippage_cost = impact_cost * 0.3

            # 其他成本（交易费用等）
            other_costs = order_size * last_price * 0.0003  # 0.03%的交易费用
            # 总成本
            total_cost = impact_cost + slippage_cost + other_costs

            # 成本置信区间
            cost_uncertainty = total_cost * 0.2
            cost_ci = (total_cost - cost_uncertainty, total_cost + cost_uncertainty)

            return {
                'total_cost': total_cost,
                'slippage_cost': slippage_cost,
                'impact_cost': impact_cost,
                'other_costs': other_costs,
                'cost_ci': cost_ci
            }

        except Exception as e:
            logger.error(f"估算交易成本失败: {e}")
            return {
                'total_cost': 0.0,
                'slippage_cost': 0.0,
                'impact_cost': 0.0,
                'other_costs': 0.0,
                'cost_ci': (0.0, 0.0)
            }


    def _update_impact_history(self, symbol: str, impact_result: Dict[str, Any]):
        """更新冲击历史"""
        try:
            history_record = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                **impact_result
            }

            self.impact_history[symbol].append(history_record)

            # 限制历史记录数量
            if len(self.impact_history[symbol]) > 5000:
                self.impact_history[symbol].popleft()

        except Exception as e:
            logger.error(f"更新冲击历史失败: {e}")


    def update_market_data(self, symbol: str, market_data: Dict[str, Any]):
        """更新市场数据"""
        try:
            data_record = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                **market_data
            }

            self.market_data_cache[symbol].append(data_record)

        except Exception as e:
            logger.error(f"更新市场数据失败: {e}")


    def calibrate_models(self, historical_orders: List[Dict[str, Any]]):
        """校准冲击模型"""
        try:
            logger.info("开始校准市场冲击模型")

            # 使用历史订单数据校准模型参数
            for order_data in historical_orders:
                result = self.analyze_market_impact(order_data)
                self.calibration_data[order_data['symbol']].append({
                    'order_size': result.order_size,
                    'price_impact': result.price_impact,
                    'market_conditions': result.metadata.get('market_conditions', {})
                })

            # 基于校准数据更新模型参数
            self._update_model_parameters()

            logger.info("市场冲击模型校准完成")

        except Exception as e:
            logger.error(f"校准冲击模型失败: {e}")


    def _update_model_parameters(self):
        """更新模型参数"""
        try:
            # 基于校准数据更新冲击衰减因子
            for symbol, calibration_records in self.calibration_data.items():
                if len(calibration_records) >= 10:
                    impacts = [r['price_impact'] for r in calibration_records]
                    sizes = [r['order_size'] for r in calibration_records]

                    # 拟合冲击模型参数
                    # 这里可以实现更复杂的参数估计算法
                    avg_impact = np.mean(impacts)
                    avg_size = np.mean(sizes)

                    if avg_size > 0:
                        estimated_decay = avg_impact / (100 * np.sqrt(avg_size))
                        self.config.impact_decay_factor = (
                            self.config.impact_decay_factor * 0.8 + estimated_decay * 0.2
                        )

        except Exception as e:
            logger.error(f"更新模型参数失败: {e}")


    def get_impact_statistics(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """获取冲击统计信息"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)

            recent_impacts = [
                record for record in self.impact_history[symbol]
                if record['timestamp'] > cutoff_time
            ]

            if not recent_impacts:
                return {'error': '没有足够的历史数据'}

            price_impacts = [r['price_impact'] for r in recent_impacts]

            return {
                'symbol': symbol,
                'period_days': days,
                'total_orders': len(recent_impacts),
                'avg_price_impact': np.mean(price_impacts),
                'median_price_impact': np.median(price_impacts),
                'max_price_impact': np.max(price_impacts),
                'min_price_impact': np.min(price_impacts),
                'price_impact_std': np.std(price_impacts),
                'last_updated': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"获取冲击统计失败: {e}")
            return {'error': str(e)}


    def get_market_impact_sensitivity(self, symbol: str) -> float:
        """获取市场冲击敏感度"""
        try:
            if symbol in self.symbol_metrics:
                return self.symbol_metrics[symbol].impact_sensitivity

            # 默认敏感度
            return 1.0

        except Exception as e:
            logger.error(f"获取冲击敏感度失败 {e}")
            return 1.0
