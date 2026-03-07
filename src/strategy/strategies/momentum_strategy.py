import time
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
动量策略实现
Momentum Strategy Implementation

基于价格动量的交易策略，通过识别价格趋势进行交易决策。
"""

import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
from .base_strategy import BaseStrategy
from ..interfaces.strategy_interfaces import StrategyConfig, StrategySignal

logger = logging.getLogger(__name__)


class MomentumStrategy(BaseStrategy):

    """
    动量策略
    Momentum Strategy

    基于价格动量识别趋势，进行趋势跟随交易。
    当价格上涨趋势明显时买入，下跌趋势明显时卖出。
    """

    def __init__(self, strategy_id: str, name: str = "Momentum", strategy_type: str = "momentum", config: Optional[StrategyConfig] = None):
        """
        初始化动量策略

        Args:
            strategy_id: 策略ID
            name: 策略名称
            strategy_type: 策略类型
            config: 策略配置
        """
        super().__init__(strategy_id, name, strategy_type)

        # 设置默认参数
        self.lookback_period = 20
        self.momentum_threshold = 0.05
        self.volume_threshold = 1.5
        self.min_trend_period = 5
        self.max_hold_period = 10

        # 初始化config属性
        self.config = config

        # 如果提供了配置，初始化策略参数
        if config and hasattr(config, 'parameters'):
            self.initialize({"parameters": config.parameters})
            self.lookback_period = config.parameters.get('lookback_period', 20)
            self.momentum_threshold = config.parameters.get('momentum_threshold', 0.05)
            self.volume_threshold = config.parameters.get('volume_threshold', 1.5)
            self.min_trend_period = config.parameters.get('min_trend_period', 5)
            self.max_hold_period = config.parameters.get('max_hold_period', 10)

        # 策略状态
        self.entry_times: Dict[str, datetime] = {}
        self.positions: Dict[str, Dict[str, Any]] = {}  # 持仓信息
        self.risk_limits: Dict[str, Any] = {
            'max_position': 10000,
            'risk_per_trade': 0.02
        }  # 风险限制

        logger.info(f"动量策略 {name} 初始化完成，参数: "
                    f"lookback_period={self.lookback_period}, "
                    f"momentum_threshold={self.momentum_threshold}")

    def execute(self, data: Dict[str, Any]) -> StrategySignal:
        """执行策略逻辑"""
        try:
            # 初始化属性
            if not hasattr(self, 'entry_price'):
                self.entry_price = 100.0
            if not hasattr(self, 'position'):
                self.position = 0

            # 简化实现
            price = data.get('price', 0)
            volume = data.get('volume', 0)
            symbol = data.get('symbol', 'unknown')

            # 简单的动量判断逻辑
            signal_type = 'HOLD'
            if price > self.entry_price and self.position == 0:
                signal_type = 'BUY'
                self.entry_price = price
                self.position = 1
            elif price < self.entry_price and self.position > 0:
                signal_type = 'SELL'
                self.position = 0

            strategy_name = self.config.strategy_name if self.config else self.name
            signal = StrategySignal(
                strategy_id=strategy_name,
                signal_type=signal_type,
                symbol=symbol,
                price=price,
                quantity=1,
                confidence=0.7,
                timestamp=datetime.now()
            )

            return {
                'signals': [signal],
                'status': 'success',
                'strategy_name': strategy_name
            }

        except Exception as e:
            logger.error(f"动量策略执行失败: {e}")
            strategy_name = self.config.strategy_name if self.config else self.name
            signal = StrategySignal(
                strategy_id=strategy_name,
                signal_type='HOLD',
                symbol=data.get('symbol', 'unknown'),
                price=data.get('price', 0),
                quantity=0,
                confidence=0.0,
                timestamp=datetime.now()
            )

            return {
                'signals': [signal],
                'status': 'error',
                'error_message': str(e),
                'strategy_name': strategy_name
            }

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """设置策略参数"""
        # 先验证参数
        validated_params = self.validate_parameters(parameters)

        # 设置验证后的参数
        for key, value in validated_params.items():
            if key == 'momentum_threshold':
                self.momentum_threshold = value
            elif key == 'lookback_period':
                self.lookback_period = value
            elif key == 'volume_threshold':
                self.volume_threshold = value
            elif key == 'min_trend_period':
                self.min_trend_period = value
            elif key == 'max_hold_period':
                self.max_hold_period = value

        # 更新配置
        if hasattr(self, 'config') and self.config and hasattr(self.config, 'parameters'):
            self.config.parameters.update(validated_params)

    def get_parameters(self) -> Dict[str, Any]:
        """获取策略参数"""
        return {
            'lookback_period': self.lookback_period,
            'momentum_period': self.lookback_period,  # 兼容测试期望的参数名
            'momentum_threshold': self.momentum_threshold,
            'signal_threshold': self.momentum_threshold,  # 兼容测试期望的参数名
            'volume_threshold': self.volume_threshold,
            'min_trend_period': self.min_trend_period,
            'max_hold_period': self.max_hold_period
        }

    def _validate_strategy_specific_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """验证动量策略特定参数"""
        validated = {}

        # lookback_period: 回望周期
        if 'lookback_period' in parameters:
            period = parameters['lookback_period']
            if not isinstance(period, int) or period <= 0:
                raise ValueError("lookback_period must be a positive integer")
            if period < 5 or period > 100:
                raise ValueError("lookback_period must be between 5 and 100")
            validated['lookback_period'] = period

        # momentum_threshold: 动量阈值
        if 'momentum_threshold' in parameters:
            threshold = parameters['momentum_threshold']
            if not isinstance(threshold, (int, float)) or not 0 < threshold <= 1:
                raise ValueError("momentum_threshold must be between 0 and 1")
            validated['momentum_threshold'] = float(threshold)

        # volume_threshold: 成交量阈值
        if 'volume_threshold' in parameters:
            volume = parameters['volume_threshold']
            if not isinstance(volume, (int, float)) or volume <= 0:
                raise ValueError("volume_threshold must be a positive number")
            if volume < 1.0 or volume > 5.0:
                raise ValueError("volume_threshold must be between 1.0 and 5.0")
            validated['volume_threshold'] = float(volume)

        # min_trend_period: 最小趋势周期
        if 'min_trend_period' in parameters:
            trend_period = parameters['min_trend_period']
            if not isinstance(trend_period, int) or trend_period <= 0:
                raise ValueError("min_trend_period must be a positive integer")
            if trend_period < 3 or trend_period > 20:
                raise ValueError("min_trend_period must be between 3 and 20")
            validated['min_trend_period'] = trend_period

        # max_hold_period: 最大持仓周期
        if 'max_hold_period' in parameters:
            hold_period = parameters['max_hold_period']
            if not isinstance(hold_period, int) or hold_period <= 0:
                raise ValueError("max_hold_period must be a positive integer")
            if hold_period < 5 or hold_period > 50:
                raise ValueError("max_hold_period must be between 5 and 50")
            validated['max_hold_period'] = hold_period

        return validated

    def _generate_signals_from_market_data(self, market_data) -> List[StrategySignal]:
        """从MarketData对象生成交易信号"""
        # 将MarketData转换为字典格式供现有逻辑使用
        data_dict = {
            'symbol': market_data.symbol,
            'price': market_data.price,
            'volume': market_data.volume,
            'high': market_data.high,
            'low': market_data.low,
            'open': market_data.open_price,
            'close': market_data.close_price,
            'timestamp': market_data.timestamp.isoformat() if market_data.timestamp else None,
            'vwap': (market_data.high + market_data.low + market_data.close_price) / 3 if market_data.high and market_data.low and market_data.close_price else market_data.price,
            'turnover': market_data.price * market_data.volume
        }

        return self._generate_signals_impl(data_dict)

    def _generate_signals_impl(self, market_data: Dict[str, Any]) -> List[StrategySignal]:
        """
        生成动量交易信号的实际实现

        Args:
            market_data: 市场数据字典

        Returns:
            List[StrategySignal]: 交易信号列表
        """
        signals = []

        for symbol, data in market_data.items():
            try:
                signal = self._analyze_symbol(symbol, data)
                if signal:
                    signals.append(signal)

            except Exception as e:
                logger.error(f"分析股票 {symbol} 时发生错误: {e}")
                continue

        strategy_name = self.config.strategy_name if self.config else self.name
        logger.info(f"动量策略 {strategy_name} 生成 {len(signals)} 个信号")
        return signals

    def _validate_market_data(self, market_data: Dict[str, Any]) -> None:
        """
        验证市场数据格式

        Args:
            market_data: 市场数据字典

        Raises:
            ValueError: 当数据格式不正确时
        """
        if not isinstance(market_data, dict):
            raise ValueError("Market data must be a dictionary")

        if not market_data:
            raise ValueError("Market data cannot be empty")

        # 验证数据结构
        for symbol, data in market_data.items():
            if not isinstance(data, list):
                raise ValueError(f"Data for symbol {symbol} must be a list")

            if len(data) < self.lookback_period:
                raise ValueError(
                    f"Insufficient data for symbol {symbol}: need at least {self.lookback_period} periods")

            # 验证数据点结构
            for point in data:
                if not isinstance(point, dict):
                    raise ValueError(f"Data point for symbol {symbol} must be a dictionary")

                required_fields = ['close', 'volume']
                for field in required_fields:
                    if field not in point:
                        raise ValueError(f"Missing required field '{field}' for symbol {symbol}")

    def generate_signals(self, market_data: Dict[str, Any]) -> List[StrategySignal]:
        """
        生成动量交易信号 (兼容性方法)

        Args:
            market_data: 市场数据字典

        Returns:
            List[StrategySignal]: 交易信号列表
        """
        return self._generate_signals_impl(market_data)

    def _analyze_symbol(self, symbol: str, data: List[Dict[str, Any]]) -> Optional[StrategySignal]:
        """
        分析单个股票的动量信号

        Args:
            symbol: 股票代码
            data: 股票历史数据

        Returns:
            Optional[StrategySignal]: 交易信号
        """
        if len(data) < self.lookback_period + 1:
            return None

        # 提取价格和成交量数据
        prices = [d.get('close', d.get('price', 0)) for d in data[-self.lookback_period - 1:]]
        volumes = [d.get('volume', 0) for d in data[-self.lookback_period - 1:]]

        if len(prices) < self.lookback_period + 1:
            return None

        # 计算动量指标
        momentum = self._calculate_momentum(prices)
        volume_ratio = self._calculate_volume_ratio(volumes)
        trend_strength = self._calculate_trend_strength(prices)

        # 获取当前价格
        current_price = prices[-1]

        # 检查是否有未平仓持仓
        current_position = self.positions.get(symbol, {}).get('quantity', 0)
        entry_time = self.entry_times.get(symbol)

        # 卖出信号检查
        if current_position > 0:
            # 检查是否需要平仓
            if self._should_exit_position(symbol, prices, entry_time):
                return StrategySignal(
                    symbol=symbol,
                    signal_type='SELL',
                    quantity=current_position,
                    price=current_price,
                    confidence=self._calculate_exit_confidence(prices),
                    timestamp=datetime.now(),
                    strategy_id=self.strategy_id,
                    metadata={
                        'momentum': momentum,
                        'trend_strength': trend_strength,
                        'exit_reason': 'trend_reversal'
                    }
                )

        # 买入信号检查
        elif current_position == 0:
            # 检查动量买入条件
            if self._should_enter_position(momentum, volume_ratio, trend_strength):
                position_size = self._calculate_position_size(symbol, current_price)

                if position_size > 0:
                    return StrategySignal(
                        symbol=symbol,
                        signal_type='BUY',
                        quantity=position_size,
                        price=current_price,
                        confidence=self._calculate_entry_confidence(
                            momentum, volume_ratio, trend_strength),
                        timestamp=datetime.now(),
                        strategy_id=self.strategy_id,
                        metadata={
                            'momentum': momentum,
                            'volume_ratio': volume_ratio,
                            'trend_strength': trend_strength
                        }
                    )

        return None

    def _calculate_momentum(self, prices: List[float]) -> float:
        """
        计算动量指标

        Args:
            prices: 价格序列

        Returns:
            float: 动量值
        """
        if len(prices) < 2:
            return 0.0

        # 计算价格动量 (当前价格相对lookback_period前的变化)
        current_price = prices[-1]
        past_price = prices[0]

        if past_price == 0:
            return 0.0

        momentum = (current_price - past_price) / past_price
        return momentum

    def _calculate_volume_ratio(self, volumes: List[float]) -> float:
        """
        计算成交量比率

        Args:
            volumes: 成交量序列

        Returns:
            float: 成交量比率
        """
        if len(volumes) < 2:
            return 1.0

        # 计算最近成交量相对平均成交量的比率
        recent_volume = volumes[-1]
        avg_volume = np.mean(volumes[:-1]) if len(volumes) > 1 else volumes[0]

        if avg_volume == 0:
            return 1.0

        volume_ratio = recent_volume / avg_volume
        return volume_ratio

    def _calculate_trend_strength(self, prices: List[float]) -> float:
        """
        计算趋势强度

        Args:
            prices: 价格序列

        Returns:
            float: 趋势强度 (0 - 1)
        """
        if len(prices) < self.min_trend_period:
            return 0.0

        # 计算价格变化的方向一致性
        price_changes = np.diff(prices)
        positive_changes = np.sum(price_changes > 0)
        total_changes = len(price_changes)

        if total_changes == 0:
            return 0.0

        trend_strength = positive_changes / total_changes

        # 如果是下跌趋势，取负值
        if prices[-1] < prices[0]:
            trend_strength = -trend_strength

        return trend_strength

    def _should_enter_position(self, momentum: float, volume_ratio: float,


                               trend_strength: float) -> bool:
        """
        判断是否应该开仓

        Args:
            momentum: 动量值
            volume_ratio: 成交量比率
            trend_strength: 趋势强度

        Returns:
            bool: 是否应该开仓
        """
        # 动量阈值检查
        if momentum < self.momentum_threshold:
            return False

        # 成交量确认
        if volume_ratio < self.volume_threshold:
            return False

        # 趋势强度检查
        if trend_strength < 0.6:  # 需要至少60 % 的上涨趋势
            return False

        return True

    def _should_exit_position(self, symbol: str, prices: List[float],


                              entry_time: Optional[datetime]) -> bool:
        """
        判断是否应该平仓

        Args:
            symbol: 股票代码
            prices: 价格序列
            entry_time: 开仓时间

        Returns:
            bool: 是否应该平仓
        """
        if not entry_time:
            return False

        # 检查持仓时间
        hold_duration = datetime.now() - entry_time
        if hold_duration.days > self.max_hold_period:
            return True

        # 检查趋势反转
        if len(prices) >= 5:
            recent_trend = self._calculate_trend_strength(prices[-5:])
            if recent_trend < -0.4:  # 下跌趋势明显
                return True

        # 检查动量衰减
        current_momentum = self._calculate_momentum(prices)
        if current_momentum < self.momentum_threshold * 0.5:
            return True

        return False

    def _calculate_position_size(self, symbol: str, current_price: float) -> int:
        """
        计算仓位大小

        Args:
            symbol: 股票代码
            current_price: 当前价格

        Returns:
            int: 仓位大小（股数）
        """
        if current_price <= 0:
            return 0

        # 获取风险限额
        max_position = self.risk_limits.get('max_position', 1000)
        risk_per_trade = self.risk_limits.get('risk_per_trade', 0.02)

        # 计算基础仓位
        base_position = int(max_position * 0.1)  # 默认10 % 的最大仓位

        # 根据风险调整仓位
        if risk_per_trade > 0:
            risk_adjusted_position = int((max_position * risk_per_trade) / current_price)
            base_position = min(base_position, risk_adjusted_position)

        return max(100, base_position)  # 至少100股

    def _calculate_entry_confidence(self, momentum: float, volume_ratio: float,


                                    trend_strength: float) -> float:
        """
        计算开仓置信度

        Args:
            momentum: 动量值
            volume_ratio: 成交量比率
            trend_strength: 趋势强度

        Returns:
            float: 置信度 (0 - 1)
        """
        # 基于多个因素计算置信度
        momentum_score = min(abs(momentum) / (self.momentum_threshold * 2), 1.0)
        volume_score = min(volume_ratio / (self.volume_threshold * 1.5), 1.0)
        trend_score = abs(trend_strength)

        confidence = (momentum_score * 0.4 + volume_score * 0.3 + trend_score * 0.3)
        return min(confidence, 1.0)

    def _calculate_exit_confidence(self, prices: List[float]) -> float:
        """
        计算平仓置信度

        Args:
            prices: 价格序列

        Returns:
            float: 置信度 (0 - 1)
        """
        if len(prices) < 5:
            return 0.5

        # 基于价格变化计算置信度
        recent_changes = np.diff(prices[-5:])
        negative_changes = np.sum(recent_changes < 0)
        change_ratio = negative_changes / len(recent_changes)

        return min(change_ratio * 1.2, 1.0)

    def update_entry_time(self, symbol: str):
        """
        更新开仓时间

        Args:
            symbol: 股票代码
        """
        self.entry_times[symbol] = datetime.now()

    def clear_entry_time(self, symbol: str):
        """
        清除开仓时间

        Args:
            symbol: 股票代码
        """
        if symbol in self.entry_times:
            del self.entry_times[symbol]


# 工厂函数

def create_momentum_strategy(config: StrategyConfig) -> MomentumStrategy:
    """
    创建动量策略实例

    Args:
        config: 策略配置

    Returns:
        MomentumStrategy: 动量策略实例
    """
    return MomentumStrategy(config)


# 导出类和函数
__all__ = [
    'MomentumStrategy',
    'create_momentum_strategy'
]
