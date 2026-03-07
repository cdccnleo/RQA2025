import time
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
均值回归策略实现
Mean Reversion Strategy Implementation

基于价格偏离均值的回归特性进行交易决策。
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from .base_strategy import BaseStrategy
from ..interfaces.strategy_interfaces import StrategyConfig, StrategySignal

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):

    """
    均值回归策略
    Mean Reversion Strategy

    基于价格偏离均值的回归特性进行交易。
    当价格偏离均值过远时，预期价格会回归到均值水平。
    """

    def __init__(self, strategy_id: str, name: str = "Mean Reversion", strategy_type: str = "mean_reversion", config: Optional[StrategyConfig] = None):
        """
        初始化均值回归策略

        Args:
            strategy_id: 策略ID
            name: 策略名称
            strategy_type: 策略类型
            config: 策略配置
        """
        super().__init__(strategy_id, name, strategy_type)

        # 设置默认参数
        self.lookback_period = 20
        self.entry_threshold = 2.0
        self.exit_threshold = 0.5

        # 设置更多默认参数
        self.min_holding_period = 5
        self.max_holding_period = 20

        # 如果提供了配置，初始化策略参数
        if config and hasattr(config, 'parameters'):
            self.initialize({"parameters": config.parameters})
            self.lookback_period = config.parameters.get('lookback_period', 20)
            self.entry_threshold = config.parameters.get('entry_threshold', 2.0)
            self.exit_threshold = config.parameters.get('exit_threshold', 0.5)
            self.min_holding_period = config.parameters.get('min_holding_period', 5)
            self.max_holding_period = config.parameters.get('max_holding_period', 20)

        # 策略状态
        self.entry_times: Dict[str, datetime] = {}

        logger.info(f"均值回归策略 {name} 初始化完成，参数: "
                    f"lookback_period={self.lookback_period}, "
                    f"entry_threshold={self.entry_threshold}")

    def execute(self, data: Dict[str, Any]) -> StrategySignal:
        """执行策略逻辑"""
        try:
            # 初始化属性
            if not hasattr(self, 'position'):
                self.position = 0

            # 简化实现
            price = data.get('price', 0)
            volume = data.get('volume', 0)
            symbol = data.get('symbol', 'unknown')

            # 简单的均值回归判断逻辑
            signal_type = 'HOLD'
            if price < self.entry_threshold and self.position == 0:
                signal_type = 'BUY'
                self.position = 1
            elif price > self.exit_threshold and self.position > 0:
                signal_type = 'SELL'
                self.position = 0

            signal = StrategySignal(
                signal_id=f"{self.config.strategy_name}_{int(time.time())}",
                strategy_id=self.config.strategy_name,
                signal_type=signal_type,
                symbol=symbol,
                price=price,
                quantity=1,
                timestamp=datetime.now(),
                confidence=0.6
            )

            return {
                'signals': [signal],
                'status': 'success',
                'strategy_name': self.config.strategy_name
            }

        except Exception as e:
            logger.error(f"均值回归策略执行失败: {e}")
            signal = StrategySignal(
                signal_id=f"{self.config.strategy_name}_error_{int(time.time())}",
                strategy_id=self.config.strategy_name,
                signal_type='HOLD',
                symbol=data.get('symbol', 'unknown'),
                price=data.get('price', 0),
                quantity=0,
                timestamp=datetime.now(),
                confidence=0.0
            )

            return {
                'signals': [signal],
                'status': 'error',
                'error_message': str(e),
                'strategy_name': self.config.strategy_name
            }

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """设置策略参数"""
        for key, value in parameters.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # 更新配置
        if self.config and self.config.parameters:
            self.config.parameters.update(parameters)

    def get_parameters(self) -> Dict[str, Any]:
        """获取策略参数"""
        return {
            'lookback_period': self.lookback_period,
            'entry_threshold': self.entry_threshold,
            'exit_threshold': self.exit_threshold,
            'min_holding_period': self.min_holding_period,
            'max_holding_period': self.max_holding_period
        }

    def _generate_signals_impl(self, market_data: Dict[str, Any]) -> List[StrategySignal]:
        """
        生成均值回归交易信号的实际实现

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

        logger.info(f"均值回归策略 {self.config.strategy_name} 生成 {len(signals)} 个信号")
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

        if 'close' not in point:
            raise ValueError(f"Missing required field 'close' for symbol {symbol}")

    def _analyze_symbol(self, symbol: str, data: List[Dict[str, Any]]) -> Optional[StrategySignal]:
        """
        分析单个股票的均值回归信号

        Args:
            symbol: 股票代码
            data: 股票历史数据

        Returns:
            Optional[StrategySignal]: 交易信号，如果没有则返回None
        """
        if len(data) < self.lookback_period:
            return None

        # 提取价格数据
        prices = [point['close'] for point in data[-self.lookback_period:]]

        # 计算均值和标准差
        mean_price = np.mean(prices)
        std_price = np.std(prices)
        current_price = data[-1]['close']

        if std_price == 0:
            return None

        # 计算z - score
        z_score = (current_price - mean_price) / std_price

        # 检查是否已有持仓
        current_time = datetime.now()
        if symbol in self.entry_times:
            entry_time = self.entry_times[symbol]
            holding_period = (current_time - entry_time).days

            # 检查退出条件
        if abs(z_score) < self.exit_threshold and holding_period >= self.min_holding_period:
            # 平仓信号
            return StrategySignal(
                signal_id=f"mr_exit_{symbol}_{int(current_time.timestamp())}",
                strategy_id=self.config.strategy_id,
                signal_type='SELL' if z_score > 0 else 'BUY',  # 反向平仓
                symbol=symbol,
                price=current_price,
                quantity=self._calculate_position_size(symbol, current_price),
                timestamp=current_time,
                confidence=self._calculate_exit_confidence(z_score, holding_period)
            )

            # 检查最大持仓时间
        if holding_period >= self.max_holding_period:
            return StrategySignal(
                signal_id=f"mr_force_exit_{symbol}_{int(current_time.timestamp())}",
                strategy_id=self.config.strategy_id,
                signal_type='SELL' if z_score > 0 else 'BUY',  # 反向平仓
                symbol=symbol,
                price=current_price,
                quantity=self._calculate_position_size(symbol, current_price),
                timestamp=current_time,
                confidence=0.5
            )
        else:
            # 检查开仓条件
            if abs(z_score) > self.entry_threshold:
                # 开仓信号
                signal_type = 'SELL' if z_score > 0 else 'BUY'  # 反向开仓
                confidence = self._calculate_entry_confidence(z_score)

                # 记录开仓时间
                self.entry_times[symbol] = current_time

                return StrategySignal(
                    signal_id=f"mr_entry_{symbol}_{int(current_time.timestamp())}",
                    strategy_id=self.config.strategy_id,
                    signal_type=signal_type,
                    symbol=symbol,
                    price=current_price,
                    quantity=self._calculate_position_size(symbol, current_price),
                    timestamp=current_time,
                    confidence=confidence
                )

        return None

    def _calculate_position_size(self, symbol: str, current_price: float) -> int:
        """
        计算仓位大小

        Args:
            symbol: 股票代码
            current_price: 当前价格

        Returns:
            int: 仓位大小
        """
        # 基础仓位计算 - 使用风险控制参数
        risk_per_trade = self.config.risk_limits.get('risk_per_trade', 0.01)  # 1 % 风险
        total_capital = self.config.risk_limits.get('total_capital', 100000)  # 默认10万

        position_size = int((total_capital * risk_per_trade) / current_price)
        return max(100, min(position_size, 10000))  # 限制在100 - 10000股之间

    def _calculate_entry_confidence(self, z_score: float) -> float:
        """
        计算开仓置信度

        Args:
            z_score: z分数

        Returns:
            float: 置信度 (0 - 1)
        """
        # 置信度基于z - score的绝对值
        confidence = min(abs(z_score) / 4.0, 1.0)  # z - score=4时达到最大置信度
        return max(0.5, confidence)  # 最小置信度0.5

    def _calculate_exit_confidence(self, z_score: float, holding_period: int) -> float:
        """
        计算平仓置信度

        Args:
            z_score: z分数
            holding_period: 持仓天数

        Returns:
            float: 置信度 (0 - 1)
        """
        # 基于z - score回归程度和持仓时间的置信度
        z_confidence = 1.0 - min(abs(z_score) / self.entry_threshold, 1.0)
        time_confidence = min(holding_period / 10.0, 1.0)  # 10天后达到最大时间置信度

        return (z_confidence + time_confidence) / 2.0
