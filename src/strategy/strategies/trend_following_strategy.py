#!/usr/bin/env python3
"""
趋势跟踪策略实现
Trend Following Strategy Implementation

基于趋势跟踪的交易决策策略。
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
from .base_strategy_fixed import BaseStrategy, StrategySignal, MarketData

logger = logging.getLogger(__name__)


class TrendFollowingStrategy(BaseStrategy):

    """趋势跟踪策略 - 均线交叉"""

    def __init__(self, strategy_id: str, name: str = "Trend Following", strategy_type: str = "trend_following"):
        """
        初始化趋势跟踪策略

        Args:
            strategy_id: 策略ID
            name: 策略名称
            strategy_type: 策略类型
        """
        super().__init__(strategy_id, name, strategy_type)
        self.fast_period = 5
        self.slow_period = 20
        self.signal_strength_threshold = 0.02

    def set_parameters(self, parameters: Dict[str, Any]) -> bool:
        """设置策略参数"""
        try:
            if 'fast_period' in parameters:
                self.fast_period = parameters['fast_period']
            if 'slow_period' in parameters:
                self.slow_period = parameters['slow_period']
            if 'signal_strength_threshold' in parameters:
                self.signal_strength_threshold = parameters['signal_strength_threshold']
            return True
        except Exception as e:
            logger.error(f"设置参数失败: {e}")
            return False

    def get_parameters(self) -> Dict[str, Any]:
        """获取策略参数"""
        return {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'signal_strength_threshold': self.signal_strength_threshold
        }

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """验证策略参数"""
        try:
            if 'fast_period' in parameters:
                fp = parameters['fast_period']
                if not isinstance(fp, int) or fp <= 0:
                    return False

            if 'slow_period' in parameters:
                sp = parameters['slow_period']
                if not isinstance(sp, int) or sp <= 0:
                    return False
                if 'fast_period' in parameters and sp <= parameters['fast_period']:
                    return False

            if 'signal_strength_threshold' in parameters:
                threshold = parameters['signal_strength_threshold']
                if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
                    return False

            return True
        except Exception as e:
            logger.error(f"参数验证失败: {e}")
            return False

    def _calculate_moving_averages(self, prices: List[float], fast_period: int, slow_period: int) -> tuple:
        """计算移动平均线"""
        prices_array = np.array(prices)

        if len(prices_array) < slow_period:
            return [], []

        fast_ma = []
        slow_ma = []

        for i in range(len(prices_array)):
            if i >= fast_period - 1:
                fast_ma.append(np.mean(prices_array[i-fast_period+1:i+1]))
            else:
                fast_ma.append(np.nan)

            if i >= slow_period - 1:
                slow_ma.append(np.mean(prices_array[i-slow_period+1:i+1]))
            else:
                slow_ma.append(np.nan)

        return fast_ma, slow_ma

    def _calculate_trend_strength(self, prices: List[float], period: int) -> float:
        """计算趋势强度"""
        if len(prices) < period:
            return 0.0

        # 计算价格变化
        changes = []
        for i in range(period, len(prices)):
            change = (prices[i] - prices[i-period]) / prices[i-period]
            changes.append(change)

        if not changes:
            return 0.0

        # 计算平均变化率
        avg_change = np.mean(changes)

        # 计算标准差
        if len(changes) > 1:
            std_change = np.std(changes)
            if std_change > 0:
                return avg_change / std_change

        return avg_change

    def _detect_trend_direction(self, prices: List[float], period: int) -> str:
        """检测趋势方向"""
        strength = self._calculate_trend_strength(prices, period)

        if strength > self.signal_strength_threshold:
            return 'UP'
        elif strength < -self.signal_strength_threshold:
            return 'DOWN'
        else:
            return 'SIDEWAYS'

    def should_enter_position(self, data: pd.DataFrame, symbol: str) -> Optional[StrategySignal]:
        """判断是否应该入场"""
        try:
            if len(data) < self.slow_period:
                return None

            prices = data['close'].values.tolist()
            fast_ma, slow_ma = self._calculate_moving_averages(prices, self.fast_period, self.slow_period)

            if len(fast_ma) < 2 or len(slow_ma) < 2:
                return None

            # 检查金叉或死叉
            current_fast = fast_ma[-1]
            current_slow = slow_ma[-1]
            prev_fast = fast_ma[-2]
            prev_slow = slow_ma[-2]

            if pd.isna(current_fast) or pd.isna(current_slow) or pd.isna(prev_fast) or pd.isna(prev_slow):
                return None

            # 金叉：短期均线上穿长期均线
            if prev_fast <= prev_slow and current_fast > current_slow:
                trend_direction = self._detect_trend_direction(prices, self.fast_period)
                if trend_direction == 'UP':
                    return StrategySignal(
                        signal_type='BUY',
                        symbol=symbol,
                        price=data['close'].iloc[-1],
                        quantity=100,
                        confidence=0.7,
                        timestamp=datetime.now(),
                        strategy_id=self.strategy_id
                    )

            # 死叉：短期均线下穿长期均线
            elif prev_fast >= prev_slow and current_fast < current_slow:
                trend_direction = self._detect_trend_direction(prices, self.fast_period)
                if trend_direction == 'DOWN':
                    return StrategySignal(
                        signal_type='SELL',
                        symbol=symbol,
                        price=data['close'].iloc[-1],
                        quantity=100,
                        confidence=0.7,
                        timestamp=datetime.now(),
                        strategy_id=self.strategy_id
                    )

            return None

        except Exception as e:
            logger.error(f"趋势跟踪策略入场判断失败: {e}")
            return None

    def should_exit_position(self, data: pd.DataFrame, symbol: str) -> bool:
        """判断是否应该出场"""
        try:
            if len(data) < self.slow_period:
                return False

            prices = data['close'].values.tolist()
            fast_ma, slow_ma = self._calculate_moving_averages(prices, self.fast_period, self.slow_period)

            if len(fast_ma) < 2 or len(slow_ma) < 2:
                return False

            # 检查反向交叉信号
            current_fast = fast_ma[-1]
            current_slow = slow_ma[-1]
            prev_fast = fast_ma[-2]
            prev_slow = slow_ma[-2]

            if pd.isna(current_fast) or pd.isna(current_slow) or pd.isna(prev_fast) or pd.isna(prev_slow):
                return False

            # 检查是否有反向交叉（可能是出场信号）
            trend_direction = self._detect_trend_direction(prices, self.fast_period)

            # 如果当前有持仓，根据趋势变化决定是否出场
            current_positions = self.get_current_positions()
            for position in current_positions:
                if position.symbol == symbol:
                    # 如果趋势反转，考虑出场
                    if position.quantity > 0 and trend_direction == 'DOWN':  # 多头持仓，趋势向下
                        return True
                    elif position.quantity < 0 and trend_direction == 'UP':  # 空头持仓，趋势向上
                        return True

            return False

        except Exception as e:
            logger.error(f"趋势跟踪策略出场判断失败: {e}")
            return False

    def execute(self, data: pd.DataFrame) -> Dict[str, Any]:
        """执行策略"""
        try:
            if len(data) < self.slow_period:
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
                    'reason': '趋势反转出场'
                })

            return {
                'status': 'success',
                'signals': signals,
                'strategy_info': {
                    'fast_period': self.fast_period,
                    'slow_period': self.slow_period,
                    'signal_strength_threshold': self.signal_strength_threshold
                }
            }

        except Exception as e:
            logger.error(f"趋势跟踪策略执行失败: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def get_strategy_description(self) -> str:
        """获取策略描述"""
        return f"趋势跟踪策略：使用{self.fast_period}日和{self.slow_period}日移动平均线交叉来识别趋势变化。当短期均线上穿长期均线时买入，下穿时卖出。"

    def get_risk_management_rules(self) -> Dict[str, Any]:
        """获取风险管理规则"""
        return {
            'max_position_size': 1000,
            'max_drawdown': 0.1,
            'stop_loss': 0.05,
            'take_profit': 0.1,
            'max_holding_period': 30
        }
