"""智能股票池更新器"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class UpdateTrigger(Enum):
    """更新触发类型"""
    TIME_BASED = "time_based"
    MARKET_STATE_CHANGE = "market_state_change"
    PERFORMANCE_DEVIATION = "performance_deviation"
    VOLATILITY_SPIKE = "volatility_spike"
    LIQUIDITY_CHANGE = "liquidity_change"
    MANUAL = "manual"


@dataclass
class UpdateDecision:
    """更新决策"""
    should_update: bool
    trigger: Optional[UpdateTrigger] = None
    reason: str = ""
    urgency_level: int = 1  # 1-5, 5为最高紧急度
    estimated_impact: float = 0.0  # 预期影响程度 0-1


class IntelligentUniverseUpdater:
    """智能股票池更新器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.last_update_time = None
        self.update_frequency = config.get('update_frequency', 'daily')
        self.max_update_interval = config.get('max_update_interval', 24)  # 小时
        self.performance_threshold = config.get('performance_threshold', 0.1)
        self.volatility_threshold = config.get('volatility_threshold', 0.3)
        self.liquidity_threshold = config.get('liquidity_threshold', 0.2)

        # 市场状态跟踪
        self.last_market_state = None
        self.market_state_history = []

        # 性能跟踪
        self.performance_history = []
        self.max_performance_history = config.get('max_performance_history', 30)

        # 更新统计
        self.update_statistics = {
            'total_updates': 0,
            'time_based_updates': 0,
            'market_state_updates': 0,
            'performance_updates': 0,
            'volatility_updates': 0,
            'liquidity_updates': 0,
            'manual_updates': 0
        }

    def should_update_universe(self,
                               current_time: datetime,
                               market_data: pd.DataFrame,
                               current_performance: Optional[float] = None,
                               current_market_state: Optional[str] = None) -> UpdateDecision:
        """判断是否应该更新股票池"""
        logger.info("评估是否需要更新股票池")

        # 1. 时间基础更新检查
        time_decision = self._check_time_based_update(current_time)
        if time_decision.should_update:
            return time_decision

        # 2. 市场状态变化检查
        if current_market_state:
            market_decision = self._check_market_state_change(current_market_state)
            if market_decision.should_update:
                return market_decision

        # 3. 性能偏差检查
        if current_performance is not None:
            performance_decision = self._check_performance_deviation(current_performance)
            if performance_decision.should_update:
                return performance_decision

        # 4. 波动率异常检查
        volatility_decision = self._check_volatility_spike(market_data)
        if volatility_decision.should_update:
            return volatility_decision

        # 5. 流动性变化检查
        liquidity_decision = self._check_liquidity_change(market_data)
        if liquidity_decision.should_update:
            return liquidity_decision

        return UpdateDecision(should_update=False, reason="无需更新")

    def _check_time_based_update(self, current_time: datetime) -> UpdateDecision:
        """检查时间基础更新"""
        if self.last_update_time is None:
            return UpdateDecision(
                should_update=True,
                trigger=UpdateTrigger.TIME_BASED,
                reason="首次更新",
                urgency_level=3
            )

        time_diff = current_time - self.last_update_time
        hours_diff = time_diff.total_seconds() / 3600

        if hours_diff >= self.max_update_interval:
            return UpdateDecision(
                should_update=True,
                trigger=UpdateTrigger.TIME_BASED,
                reason=f"距离上次更新已过{hours_diff:.1f}小时",
                urgency_level=2
            )

        return UpdateDecision(should_update=False)

    def _check_market_state_change(self, current_market_state: str) -> UpdateDecision:
        """检查市场状态变化"""
        if self.last_market_state is None:
            self.last_market_state = current_market_state
            return UpdateDecision(should_update=False)

        if current_market_state != self.last_market_state:
            # 记录市场状态变化
            from_state = self.last_market_state
            self.market_state_history.append({
                'timestamp': datetime.now(),
                'from_state': from_state,
                'to_state': current_market_state
            })

            # 限制历史记录长度
            if len(self.market_state_history) > 50:
                self.market_state_history.pop(0)

            self.last_market_state = current_market_state

            return UpdateDecision(
                should_update=True,
                trigger=UpdateTrigger.MARKET_STATE_CHANGE,
                reason=f"市场状态从{from_state}变为{current_market_state}",
                urgency_level=4
            )

        return UpdateDecision(should_update=False)

    def _check_performance_deviation(self, current_performance: float) -> UpdateDecision:
        """检查性能偏差"""
        if not self.performance_history:
            self.performance_history.append(current_performance)
            return UpdateDecision(should_update=False)

        # 计算性能变化
        avg_performance = np.mean(self.performance_history)
        if avg_performance == 0:
            performance_change = 0
        else:
            performance_change = abs(current_performance - avg_performance) / avg_performance

        # 更新性能历史
        self.performance_history.append(current_performance)
        if len(self.performance_history) > self.max_performance_history:
            self.performance_history.pop(0)

        if performance_change > self.performance_threshold:
            return UpdateDecision(
                should_update=True,
                trigger=UpdateTrigger.PERFORMANCE_DEVIATION,
                reason=f"性能偏差{performance_change:.2%}超过阈值{self.performance_threshold:.2%}",
                urgency_level=5,
                estimated_impact=min(performance_change, 1.0)
            )

        return UpdateDecision(should_update=False)

    def _check_volatility_spike(self, market_data: pd.DataFrame) -> UpdateDecision:
        """检查波动率异常"""
        if market_data.empty:
            return UpdateDecision(should_update=False)

        # 计算市场整体波动率
        if 'volatility' in market_data.columns:
            avg_volatility = market_data['volatility'].mean()
            volatility_std = market_data['volatility'].std()

            # 检查是否有异常高波动率
            high_volatility_stocks = market_data[market_data['volatility']
                                                 > avg_volatility + 2 * volatility_std]

            if len(high_volatility_stocks) > len(market_data) * 0.1:  # 超过10%的股票波动率异常
                return UpdateDecision(
                    should_update=True,
                    trigger=UpdateTrigger.VOLATILITY_SPIKE,
                    reason=f"检测到{len(high_volatility_stocks)}只股票波动率异常",
                    urgency_level=4,
                    estimated_impact=0.3
                )
            elif avg_volatility > self.volatility_threshold:  # 平均波动率超过阈值
                return UpdateDecision(
                    should_update=True,
                    trigger=UpdateTrigger.VOLATILITY_SPIKE,
                    reason=f"市场平均波动率{avg_volatility:.2f}超过阈值{self.volatility_threshold:.2f}",
                    urgency_level=4,
                    estimated_impact=0.3
                )

        return UpdateDecision(should_update=False)

    def _check_liquidity_change(self, market_data: pd.DataFrame) -> UpdateDecision:
        """检查流动性变化"""
        if market_data.empty:
            return UpdateDecision(should_update=False)

        # 计算流动性指标变化
        if 'turnover_rate' in market_data.columns:
            avg_turnover = market_data['turnover_rate'].mean()
            low_liquidity_stocks = market_data[market_data['turnover_rate'] < avg_turnover * 0.5]

            if len(low_liquidity_stocks) > len(market_data) * 0.2:  # 超过20%的股票流动性不足
                return UpdateDecision(
                    should_update=True,
                    trigger=UpdateTrigger.LIQUIDITY_CHANGE,
                    reason=f"检测到{len(low_liquidity_stocks)}只股票流动性不足",
                    urgency_level=3,
                    estimated_impact=0.2
                )
            elif avg_turnover < self.liquidity_threshold:  # 平均换手率低于阈值
                return UpdateDecision(
                    should_update=True,
                    trigger=UpdateTrigger.LIQUIDITY_CHANGE,
                    reason=f"市场平均换手率{avg_turnover:.2%}低于阈值{self.liquidity_threshold:.2%}",
                    urgency_level=3,
                    estimated_impact=0.2
                )

        return UpdateDecision(should_update=False)

    def record_update(self, trigger: UpdateTrigger, reason: str = ""):
        """记录更新事件"""
        self.last_update_time = datetime.now()
        self.update_statistics['total_updates'] += 1

        if trigger == UpdateTrigger.TIME_BASED:
            self.update_statistics['time_based_updates'] += 1
        elif trigger == UpdateTrigger.MARKET_STATE_CHANGE:
            self.update_statistics['market_state_updates'] += 1
        elif trigger == UpdateTrigger.PERFORMANCE_DEVIATION:
            self.update_statistics['performance_updates'] += 1
        elif trigger == UpdateTrigger.VOLATILITY_SPIKE:
            self.update_statistics['volatility_updates'] += 1
        elif trigger == UpdateTrigger.LIQUIDITY_CHANGE:
            self.update_statistics['liquidity_updates'] += 1
        elif trigger == UpdateTrigger.MANUAL:
            self.update_statistics['manual_updates'] += 1

        logger.info(f"记录更新事件: {trigger.value}, 原因: {reason}")

    def get_update_statistics(self) -> Dict[str, Any]:
        """获取更新统计信息"""
        return {
            'update_statistics': self.update_statistics,
            'last_update_time': self.last_update_time,
            'market_state_history_count': len(self.market_state_history),
            'performance_history_count': len(self.performance_history),
            'current_market_state': self.last_market_state
        }

    def get_market_state_history(self) -> List[Dict[str, Any]]:
        """获取市场状态历史"""
        return self.market_state_history.copy()

    def get_performance_history(self) -> List[float]:
        """获取性能历史"""
        return self.performance_history.copy()

    def reset_statistics(self):
        """重置统计信息"""
        self.update_statistics = {
            'total_updates': 0,
            'time_based_updates': 0,
            'market_state_updates': 0,
            'performance_updates': 0,
            'volatility_updates': 0,
            'liquidity_updates': 0,
            'manual_updates': 0
        }
        self.market_state_history.clear()
        self.performance_history.clear()
        self.last_update_time = None
        self.last_market_state = None
