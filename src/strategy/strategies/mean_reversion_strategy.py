#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
均值回归策略实现
Mean Reversion Strategy Implementation

基于价格偏离均值的回归特性进行交易决策。
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
from .base_strategy import BaseStrategy, StrategySignal

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):

    """
    均值回归策略
    Mean Reversion Strategy

    基于价格偏离均值的回归特性进行交易。
    当价格偏离均值过远时，预期价格会回归到均值水平。
    """

    def __init__(self, strategy_id: str, name: str = "Mean Reversion", strategy_type: str = "mean_reversion"):
        """
        初始化均值回归策略

        Args:
            strategy_id: 策略ID
            name: 策略名称
            strategy_type: 策略类型
        """
        super().__init__(strategy_id, name, strategy_type)

        # 设置默认参数
        self.lookback_period = 20
        self.entry_threshold = 2.0
        self.exit_threshold = 0.5
        self.min_holding_period = 5
        self.max_holding_period = 20

        # 策略状态
        self.entry_times: Dict[str, datetime] = {}

        logger.info(f"均值回归策略 {name} 初始化完成，参数: "
                    f"lookback_period={self.lookback_period}, "
                    f"entry_threshold={self.entry_threshold}")

    def set_parameters(self, parameters: Dict[str, Any]) -> bool:
        """设置策略参数"""
        try:
            if 'lookback_period' in parameters:
                self.lookback_period = parameters['lookback_period']
            if 'entry_threshold' in parameters:
                self.entry_threshold = parameters['entry_threshold']
            if 'exit_threshold' in parameters:
                self.exit_threshold = parameters['exit_threshold']
            if 'min_holding_period' in parameters:
                self.min_holding_period = parameters['min_holding_period']
            if 'max_holding_period' in parameters:
                self.max_holding_period = parameters['max_holding_period']
            return True
        except Exception as e:
            logger.error(f"设置参数失败: {e}")
            return False

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """验证策略参数"""
        try:
            if 'lookback_period' in parameters:
                lp = parameters['lookback_period']
                if not isinstance(lp, int) or lp <= 0:
                    return False

            if 'entry_threshold' in parameters:
                et = parameters['entry_threshold']
                if not isinstance(et, (int, float)) or et <= 0:
                    return False

            if 'exit_threshold' in parameters:
                et = parameters['exit_threshold']
                if not isinstance(et, (int, float)) or et <= 0:
                    return False

            return True
        except Exception as e:
            logger.error(f"参数验证失败: {e}")
            return False

    def _calculate_z_score(self, prices: List[float], period: int) -> List[float]:
        """计算Z-Score"""
        if not prices:
            raise ValueError("价格数据不能为空")

        z_scores = []
        for i in range(len(prices)):
            if i < period - 1:
                z_scores.append(0.0)
            else:
                window = prices[i-period+1:i+1]
                mean_val = np.mean(window)
                std_val = np.std(window) if len(window) > 1 else 1.0

                if std_val > 0:
                    z_score = (prices[i] - mean_val) / std_val
                else:
                    z_score = 0.0
                z_scores.append(z_score)

        return z_scores

    def _calculate_moving_average(self, prices: List[float], period: int) -> List[float]:
        """计算移动平均"""
        if not prices:
            raise ValueError("价格数据不能为空")

        ma = []
        for i in range(len(prices)):
            if i < period - 1:
                ma.append(np.nan)
            else:
                window = prices[i-period+1:i+1]
                ma.append(np.mean(window))
        return ma

    def _calculate_bollinger_bands(self, prices: List[float], period: int, std_dev: float) -> tuple:
        """计算布林带"""
        ma = self._calculate_moving_average(prices, period)
        upper = []
        lower = []

        for i in range(len(prices)):
            if i < period - 1:
                upper.append(np.nan)
                lower.append(np.nan)
            else:
                window = prices[i-period+1:i+1]
                mean_val = np.mean(window)
                std_val = np.std(window) if len(window) > 1 else 0.0

                upper.append(mean_val + std_dev * std_val)
                lower.append(mean_val - std_dev * std_val)

        return upper, ma, lower

    def should_enter_position(self, data: pd.DataFrame, symbol: str) -> Optional[StrategySignal]:
        """判断是否应该入场"""
        try:
            if len(data) < self.lookback_period:
                return None

            prices = data['close'].values.tolist()
            z_scores = self._calculate_z_score(prices, self.lookback_period)

            if not z_scores:
                return None

            current_z = z_scores[-1]

            # 检查是否有持仓
            current_positions = self.get_current_positions()
            has_position = any(p.symbol == symbol for p in current_positions)

            if not has_position:
                # 价格偏离均值过远（负值表示低于均值太多，是买入机会）
                if current_z <= -self.entry_threshold:
                    return StrategySignal(
                        signal_type='BUY',
                        symbol=symbol,
                        price=data['close'].iloc[-1],
                        quantity=100,
                        confidence=min(abs(current_z) / 3.0, 0.9),  # Z-Score越大，信心越高
                        timestamp=datetime.now(),
                        strategy_id=self.strategy_id
                    )
                # 价格高于均值太多（正值表示高于均值太多，是卖出机会）
                elif current_z >= self.entry_threshold:
                    return StrategySignal(
                        signal_type='SELL',
                        symbol=symbol,
                        price=data['close'].iloc[-1],
                        quantity=100,
                        confidence=min(abs(current_z) / 3.0, 0.9),
                        timestamp=datetime.now(),
                        strategy_id=self.strategy_id
                    )

            return None

        except Exception as e:
            logger.error(f"均值回归策略入场判断失败: {e}")
            return None

    def should_exit_position(self, data: pd.DataFrame, symbol: str) -> bool:
        """判断是否应该出场"""
        try:
            if len(data) < self.lookback_period:
                return False

            prices = data['close'].values.tolist()
            z_scores = self._calculate_z_score(prices, self.lookback_period)

            if not z_scores:
                return False

            current_z = z_scores[-1]

            # 检查是否有持仓
            current_positions = self.get_current_positions()
            for position in current_positions:
                if position.symbol == symbol:
                    # 如果Z-Score接近0，说明价格回归到均值附近，可以考虑出场
                    if abs(current_z) <= self.exit_threshold:
                        return True

            return False

        except Exception as e:
            logger.error(f"均值回归策略出场判断失败: {e}")
            return False

    def execute(self, data: pd.DataFrame) -> Dict[str, Any]:
        """执行策略"""
        try:
            if len(data) < self.lookback_period:
                return {
                    'status': 'error',
                    'message': '数据不足，无法执行策略'
                }

            signals = []

            # 检查入场信号
            entry_signal = self.should_enter_position(data, 'DEFAULT')
            if entry_signal:
                signals.append({
                    'type': entry_signal.signal_type,
                    'symbol': entry_signal.symbol,
                    'price': entry_signal.price,
                    'quantity': entry_signal.quantity,
                    'confidence': entry_signal.confidence,
                    'timestamp': entry_signal.timestamp.isoformat()
                })

            # 检查出场信号
            exit_needed = self.should_exit_position(data, 'DEFAULT')
            if exit_needed:
                signals.append({
                    'type': 'EXIT',
                    'symbol': 'DEFAULT',
                    'reason': '价格回归均值附近出场'
                })

            return {
                'status': 'success',
                'signals': signals,
                'strategy_info': {
                    'lookback_period': self.lookback_period,
                    'entry_threshold': self.entry_threshold,
                    'exit_threshold': self.exit_threshold
                }
            }

        except Exception as e:
            logger.error(f"均值回归策略执行失败: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def get_strategy_description(self) -> str:
        """获取策略描述"""
        return f"均值回归策略：当价格偏离{self.lookback_period}日均值超过{self.entry_threshold}个标准差时，预期价格会回归到均值水平。当Z-Score小于-{self.entry_threshold}时买入，大于{self.entry_threshold}时卖出。"

    def get_risk_management_rules(self) -> Dict[str, Any]:
        """获取风险管理规则"""
        return {
            'max_position_size': 500,
            'max_drawdown': 0.05,
            'stop_loss': 0.03,
            'take_profit': 0.05,
            'max_holding_period': self.max_holding_period
        }
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

        logger.info(f"均值回归策略 {self.name} 生成 {len(signals)} 个信号")
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
