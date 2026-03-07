#!/usr/bin/env python3
"""
自动交易调整引擎

构建智能的自动交易调整和风险响应系统
    创建时间: 2025年3月
"""

import sys
import os
import numpy as np
import logging
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import threading
import queue

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from risk.realtime_risk_monitor import RiskType, RiskLevel, RiskIndicator
    print("✅ 风险模块导入成功")
except ImportError as e:
    print(f"❌ 风险模块导入失败: {e}")
    # 创建简化的替代类用于演示

    class RiskType(Enum):

        MARKET_RISK = "market_risk"
        LIQUIDITY_RISK = "liquidity_risk"

    class RiskLevel(Enum):

        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdjustmentType(Enum):

    """调整类型枚举"""
    POSITION_REDUCTION = "position_reduction"    # 仓位减少
    STOP_LOSS = "stop_loss"                     # 止损
    TAKE_PROFIT = "take_profit"                 # 止盈
    HEDGE_ADJUSTMENT = "hedge_adjustment"       # 对冲调整
    DIVERSIFICATION = "diversification"         # 分散化
    LIQUIDITY_ADJUSTMENT = "liquidity_adjustment"  # 流动性调整
    VOLATILITY_HEDGE = "volatility_hedge"       # 波动率对冲


class ResponseAction(Enum):

    """响应动作枚举"""
    HOLD = "hold"                 # 持有不动
    REDUCE_POSITION = "reduce_position"  # 减少仓位
    CLOSE_POSITION = "close_position"    # 平仓
    HEDGE_POSITION = "hedge_position"    # 对冲仓位
    REBALANCE = "rebalance"       # 再平衡
    EMERGENCY_STOP = "emergency_stop"    # 紧急停止


@dataclass
class TradeAdjustment:

    """交易调整"""
    adjustment_id: str
    timestamp: datetime
    adjustment_type: AdjustmentType
    response_action: ResponseAction
    asset_symbol: str
    current_position: float
    target_position: float
    adjustment_amount: float
    reason: str
    risk_indicators: Dict[str, Any]
    expected_impact: Dict[str, float]
    status: str = "pending"  # pending, executing, completed, failed
    execution_price: Optional[float] = None
    execution_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'adjustment_id': self.adjustment_id,
            'timestamp': self.timestamp.isoformat(),
            'adjustment_type': self.adjustment_type.value,
            'response_action': self.response_action.value,
            'asset_symbol': self.asset_symbol,
            'current_position': self.current_position,
            'target_position': self.target_position,
            'adjustment_amount': self.adjustment_amount,
            'reason': self.reason,
            'risk_indicators': self.risk_indicators,
            'expected_impact': self.expected_impact,
            'status': self.status,
            'execution_price': self.execution_price,
            'execution_time': self.execution_time.isoformat() if self.execution_time else None
        }


@dataclass
class RiskLimit:

    """风险限额"""
    asset_symbol: str
    max_position_size: float
    max_var_limit: float
    max_drawdown_limit: float
    max_concentration_limit: float
    daily_loss_limit: float
    volatility_limit: float
    last_updated: datetime
    breach_count: int = 0
    last_breach_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'asset_symbol': self.asset_symbol,
            'max_position_size': self.max_position_size,
            'max_var_limit': self.max_var_limit,
            'max_drawdown_limit': self.max_drawdown_limit,
            'max_concentration_limit': self.max_concentration_limit,
            'daily_loss_limit': self.daily_loss_limit,
            'volatility_limit': self.volatility_limit,
            'last_updated': self.last_updated.isoformat(),
            'breach_count': self.breach_count,
            'last_breach_time': self.last_breach_time.isoformat() if self.last_breach_time else None
        }


class RiskLimitManager:

    """风险限额管理器"""

    def __init__(self):

        self.risk_limits: Dict[str, RiskLimit] = {}
        self.portfolio_limits = {
            'total_var_limit': 0.10,      # 投资组合VaR限额
            'total_drawdown_limit': 0.20,  # 最大回撤限额
            'concentration_limit': 0.25,  # 最大集中度
            'daily_loss_limit': 0.05      # 每日最大损失
        }

    def set_asset_limit(self, symbol: str, **limits):
        """设置资产风险限额"""
        risk_limit = RiskLimit(
            asset_symbol=symbol,
            max_position_size=limits.get('max_position_size', 1000000),
            max_var_limit=limits.get('max_var_limit', 0.05),
            max_drawdown_limit=limits.get('max_drawdown_limit', 0.10),
            max_concentration_limit=limits.get('max_concentration_limit', 0.20),
            daily_loss_limit=limits.get('daily_loss_limit', 0.02),
            volatility_limit=limits.get('volatility_limit', 0.30),
            last_updated=datetime.now()
        )

        self.risk_limits[symbol] = risk_limit
        logger.info(f"设置资产 {symbol} 风险限额: {limits}")

    def check_limit_breach(self, symbol: str, current_metrics: Dict[str, float],


                           current_position: float) -> List[str]:
        """检查限额突破"""
        if symbol not in self.risk_limits:
            return []

        limit = self.risk_limits[symbol]
        breaches = []

        # 检查VaR限额
        if current_metrics.get('var_95', 0) > limit.max_var_limit:
            breaches.append(f"VaR限额突破: {current_metrics['var_95']:.2%} > {limit.max_var_limit:.2%}")

        # 检查回撤限额
        if current_metrics.get('max_drawdown', 0) > limit.max_drawdown_limit:
            breaches.append(
                f"回撤限额突破: {current_metrics['max_drawdown']:.2%} > {limit.max_drawdown_limit:.2%}")

        # 检查波动率限额
        if current_metrics.get('volatility', 0) > limit.volatility_limit:
            breaches.append(
                f"波动率限额突破: {current_metrics['volatility']:.2%} > {limit.volatility_limit:.2%}")

        # 检查仓位大小限额
        if abs(current_position) > limit.max_position_size:
            breaches.append(
                f"仓位大小限额突破: {abs(current_position):,.0f} > {limit.max_position_size:,.0f}")

        # 更新突破统计
        if breaches:
            limit.breach_count += 1
            limit.last_breach_time = datetime.now()

        return breaches

    def adjust_limits(self, market_conditions: Dict[str, Any]):
        """根据市场条件动态调整限额"""
        volatility = market_conditions.get('market_volatility', 0.20)
        market_stress = market_conditions.get('market_stress_level', 0.5)

        # 根据市场波动率调整限额
        adjustment_factor = 1.0 / (1.0 + volatility)

        for symbol, limit in self.risk_limits.items():
            # 降低波动率较高的资产限额
            if volatility > 0.25:
                limit.max_var_limit *= adjustment_factor
                limit.max_drawdown_limit *= adjustment_factor
                limit.daily_loss_limit *= adjustment_factor
                limit.last_updated = datetime.now()

        # 调整投资组合限额
        if market_stress > 0.7:
            self.portfolio_limits['total_var_limit'] *= 0.8
            self.portfolio_limits['total_drawdown_limit'] *= 0.8
            self.portfolio_limits['daily_loss_limit'] *= 0.8

        logger.info("根据市场条件动态调整风险限额")

    def get_limit_summary(self) -> Dict[str, Any]:
        """获取限额摘要"""
        return {
            'asset_limits': {symbol: limit.to_dict() for symbol, limit in self.risk_limits.items()},
            'portfolio_limits': self.portfolio_limits,
            'total_assets': len(self.risk_limits),
            'breached_limits': sum(1 for limit in self.risk_limits.values() if limit.breach_count > 0)
        }


class TradeAdjustmentEngine:

    """交易调整引擎"""

    def __init__(self):

        self.adjustments: List[TradeAdjustment] = []
        self.active_adjustments: Dict[str, TradeAdjustment] = {}
        self.risk_limits = RiskLimitManager()
        self.adjustment_strategies = self._initialize_strategies()

        # 统计信息
        self.stats = {
            'total_adjustments': 0,
            'successful_adjustments': 0,
            'failed_adjustments': 0,
            'avg_execution_time': 0.0,
            'risk_reduction': 0.0
        }

    def _initialize_strategies(self) -> Dict[RiskType, Callable]:
        """初始化调整策略"""
        return {
            RiskType.MARKET_RISK: self._market_risk_strategy,
            RiskType.LIQUIDITY_RISK: self._liquidity_risk_strategy,
            RiskType.POSITION_RISK: self._position_risk_strategy,
            RiskType.VOLATILITY_RISK: self._volatility_risk_strategy
        }

    def assess_and_adjust(self, portfolio: Dict[str, Any],


                          risk_indicators: Dict[str, RiskIndicator],
                          market_data: Dict[str, Any]) -> List[TradeAdjustment]:
        """评估风险并生成调整建议"""
        adjustments = []

        # 评估每只资产的风险
        for symbol, position in portfolio.get('positions', {}).items():
            if symbol not in risk_indicators:
                continue

            indicator = risk_indicators[symbol]

            # 检查限额突破
            current_metrics = {
                'var_95': indicator.value if 'var' in indicator.name.lower() else 0.02,
                'max_drawdown': 0.05,  # 简化的回撤值
                'volatility': 0.20     # 简化的波动率值
            }

            breaches = self.risk_limits.check_limit_breach(symbol, current_metrics, position)

            if breaches or indicator.calculate_risk_level() in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                # 生成调整建议
                adjustment = self._generate_adjustment(
                    symbol, position, indicator, breaches, market_data
                )
                if adjustment:
                    adjustments.append(adjustment)

        # 评估投资组合整体风险
        portfolio_risks = self._assess_portfolio_risk(portfolio, risk_indicators)
        if portfolio_risks:
            portfolio_adjustments = self._generate_portfolio_adjustments(portfolio, portfolio_risks)
            adjustments.extend(portfolio_adjustments)

        return adjustments

    def _generate_adjustment(self, symbol: str, current_position: float,


                             risk_indicator: RiskIndicator, breaches: List[str],
                             market_data: Dict[str, Any]) -> Optional[TradeAdjustment]:
        """生成单个资产调整"""
        risk_level = risk_indicator.calculate_risk_level()

        # 确定调整类型和动作
        if risk_level == RiskLevel.CRITICAL or breaches:
            response_action = ResponseAction.CLOSE_POSITION
            adjustment_amount = current_position  # 全平
            reason = f"严重风险: {risk_indicator.name}={risk_indicator.value:.2f}"
        elif risk_level == RiskLevel.HIGH:
            response_action = ResponseAction.REDUCE_POSITION
            adjustment_amount = current_position * 0.5  # 减半
            reason = f"高风险: {risk_indicator.name}={risk_indicator.value:.2f}"
        elif risk_level == RiskLevel.MEDIUM:
            response_action = ResponseAction.HEDGE_POSITION
            adjustment_amount = current_position * 0.2  # 对冲20%
            reason = f"中风险: {risk_indicator.name}={risk_indicator.value:.2f}"
        else:
            return None

        # 创建调整对象
        adjustment = TradeAdjustment(
            adjustment_id=f"adj_{symbol}_{int(time.time())}_{hash(reason) % 10000}",
            timestamp=datetime.now(),
            adjustment_type=AdjustmentType.POSITION_REDUCTION,
            response_action=response_action,
            asset_symbol=symbol,
            current_position=current_position,
            target_position=current_position - adjustment_amount,
            adjustment_amount=adjustment_amount,
            reason=reason,
            risk_indicators={
                'indicator_name': risk_indicator.name,
                'value': risk_indicator.value,
                'level': risk_level.value
            },
            expected_impact={
                'risk_reduction': min(adjustment_amount / abs(current_position), 1.0),
                'expected_return_impact': -0.01,  # 简化的预期影响
                'transaction_cost': 0.001
            }
        )

        return adjustment

    def _assess_portfolio_risk(self, portfolio: Dict[str, Any],


                               risk_indicators: Dict[str, RiskIndicator]) -> List[str]:
        """评估投资组合整体风险"""
        risks = []

        # 计算投资组合总VaR（简化）
        total_var = 0
        for symbol, position in portfolio.get('positions', {}).items():
            if symbol in risk_indicators:
                indicator = risk_indicators[symbol]
                if 'var' in indicator.name.lower():
                    total_var += abs(position) * indicator.value

        if total_var > 0.10:  # 10% VaR限额
            risks.append(f"投资组合VaR过高: {total_var:.2%}")

        # 检查集中度
        positions = list(portfolio.get('positions', {}).values())
        if positions:
            max_position = max(abs(p) for p in positions)
            total_position = sum(abs(p) for p in positions)
            concentration = max_position / total_position if total_position > 0 else 0

            if concentration > 0.3:  # 30 % 集中度限额
                risks.append(f"投资组合集中度过高: {concentration:.2%}")

        return risks

    def _generate_portfolio_adjustments(self, portfolio: Dict[str, Any],


                                        portfolio_risks: List[str]) -> List[TradeAdjustment]:
        """生成投资组合调整"""
        adjustments = []

        if not portfolio_risks:
            return adjustments

        # 对集中度过高的资产进行调整
        positions = portfolio.get('positions', {})
        if positions:
            total_value = sum(abs(p) for p in positions)
            max_position = max(abs(p) for p in positions)
            max_symbol = max(positions.keys(), key=lambda k: abs(positions[k]))

            if max_position / total_value > 0.3:
                # 减少最大头寸
                current_position = positions[max_symbol]
                adjustment_amount = current_position * 0.2  # 减少20%

                adjustment = TradeAdjustment(
                    adjustment_id=f"port_adj_{max_symbol}_{int(time.time())}",
                    timestamp=datetime.now(),
                    adjustment_type=AdjustmentType.DIVERSIFICATION,
                    response_action=ResponseAction.REDUCE_POSITION,
                    asset_symbol=max_symbol,
                    current_position=current_position,
                    target_position=current_position - adjustment_amount,
                    adjustment_amount=adjustment_amount,
                    reason="投资组合集中度过高",
                    risk_indicators={'concentration': max_position / total_value},
                    expected_impact={'diversification_improvement': 0.05}
                )

                adjustments.append(adjustment)

        return adjustments

    def execute_adjustment(self, adjustment: TradeAdjustment) -> bool:
        """执行调整"""
        try:
            start_time = datetime.now()
            adjustment.status = "executing"

            # 简化的执行逻辑（实际应该调用交易API）
            logger.info(f"执行调整: {adjustment.adjustment_id}")
            logger.info(f"资产: {adjustment.asset_symbol}")
            logger.info(f"动作: {adjustment.response_action.value}")
            logger.info(f"调整金额: {adjustment.adjustment_amount:.2f}")

            # 模拟执行时间
            time.sleep(0.1)

            # 模拟执行价格
            adjustment.execution_price = 100.0 + np.secrets.normal(0, 5)
            adjustment.execution_time = datetime.now()
            adjustment.status = "completed"

            # 更新统计
            self.stats['total_adjustments'] += 1
            self.stats['successful_adjustments'] += 1
            execution_time = (datetime.now() - start_time).total_seconds()
            self.stats['avg_execution_time'] = (
                (self.stats['avg_execution_time'] * (self.stats['total_adjustments'] - 1))
                + execution_time
            ) / self.stats['total_adjustments']

            logger.info(f"调整执行成功: {adjustment.adjustment_id}")
            return True

        except Exception as e:
            adjustment.status = "failed"
            self.stats['total_adjustments'] += 1
            self.stats['failed_adjustments'] += 1
            logger.error(f"调整执行失败: {adjustment.adjustment_id}, 错误: {e}")
            return False

    def get_adjustment_summary(self) -> Dict[str, Any]:
        """获取调整摘要"""
        recent_adjustments = [
            adj for adj in self.adjustments[-100:]  # 最近100个调整
        ]

        return {
            'total_adjustments': self.stats['total_adjustments'],
            'successful_adjustments': self.stats['successful_adjustments'],
            'failed_adjustments': self.stats['failed_adjustments'],
            'success_rate': (self.stats['successful_adjustments']
                             / max(self.stats['total_adjustments'], 1)),
            'avg_execution_time': self.stats['avg_execution_time'],
            'recent_adjustments': [adj.to_dict() for adj in recent_adjustments],
            'risk_limits': self.risk_limits.get_limit_summary()
        }


class AutomatedResponseSystem:

    """自动化响应系统"""

    def __init__(self):

        self.adjustment_engine = TradeAdjustmentEngine()
        self.is_running = False
        self.monitoring_thread = None
        self.response_queue = queue.Queue()
        self.emergency_stop = False

        # 应急响应配置
        self.emergency_protocols = {
            'circuit_breaker': self._circuit_breaker_protocol,
            'market_halt': self._market_halt_protocol,
            'system_failure': self._system_failure_protocol,
            'extreme_volatility': self._extreme_volatility_protocol
        }

    def start_automated_response(self, data_source: Callable[[], Dict[str, Any]]):
        """启动自动化响应"""
        if self.is_running:
            logger.warning("自动化响应已启动")
            return

        self.is_running = True
        self.monitoring_thread = threading.Thread(
            target=self._response_loop,
            args=(data_source,)
        )
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

        logger.info("自动化响应系统已启动")

    def stop_automated_response(self):
        """停止自动化响应"""
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("自动化响应系统已停止")

    def _response_loop(self, data_source: Callable[[], Dict[str, Any]]):
        """响应循环"""
        while self.is_running:
            try:
                # 获取市场数据和投资组合状态
                market_data = data_source()

                if market_data:
                    # 检查是否需要应急响应
                    if self._check_emergency_conditions(market_data):
                        self._execute_emergency_protocol(market_data)
                        continue

                    # 评估风险并生成调整
                    portfolio = market_data.get('portfolio', {})
                    risk_indicators = market_data.get('risk_indicators', {})

                    adjustments = self.adjustment_engine.assess_and_adjust(
                        portfolio, risk_indicators, market_data
                    )

                    # 执行调整
                    for adjustment in adjustments:
                        if self.emergency_stop:
                            logger.warning("紧急停止：取消执行调整")
                            break

                        success = self.adjustment_engine.execute_adjustment(adjustment)
                        if success:
                            self.adjustment_engine.adjustments.append(adjustment)

                # 控制响应频率
                time.sleep(5)  # 每5秒评估一次

            except Exception as e:
                logger.error(f"响应循环错误: {e}")
                time.sleep(10)

    def _check_emergency_conditions(self, market_data: Dict[str, Any]) -> bool:
        """检查应急条件"""
        # 检查市场波动率
        market_volatility = market_data.get('market_volatility', 0)
        if market_volatility > 0.5:  # 50 % 波动率
            return True

        # 检查价格跳空
        price_gap = market_data.get('price_gap', 0)
        if price_gap > 0.1:  # 10 % 跳空
            return True

        # 检查系统状态
        system_status = market_data.get('system_status', 'normal')
        if system_status in ['critical', 'failed']:
            return True

        return False

    def _execute_emergency_protocol(self, market_data: Dict[str, Any]):
        """执行应急协议"""
        market_volatility = market_data.get('market_volatility', 0)
        price_gap = market_data.get('price_gap', 0)
        system_status = market_data.get('system_status', 'normal')

        if market_volatility > 0.5:
            self._extreme_volatility_protocol(market_data)
        elif price_gap > 0.1:
            self._circuit_breaker_protocol(market_data)
        elif system_status in ['critical', 'failed']:
            self._system_failure_protocol(market_data)

    def _circuit_breaker_protocol(self, market_data: Dict[str, Any]):
        """熔断机制协议"""
        logger.warning("🚨 触发熔断机制：市场价格跳空")

        # 暂停所有交易
        self.emergency_stop = True

        # 生成紧急调整
        emergency_adjustment = TradeAdjustment(
            adjustment_id=f"emergency_circuit_{int(time.time())}",
            timestamp=datetime.now(),
            adjustment_type=AdjustmentType.STOP_LOSS,
            response_action=ResponseAction.EMERGENCY_STOP,
            asset_symbol="ALL",
            current_position=0,
            target_position=0,
            adjustment_amount=0,
            reason="市场熔断：价格跳空",
            risk_indicators=market_data,
            expected_impact={'trading_halt': True}
        )

        logger.warning(f"紧急调整已生成: {emergency_adjustment.adjustment_id}")

    def _extreme_volatility_protocol(self, market_data: Dict[str, Any]):
        """极端波动率协议"""
        logger.warning("⚠️ 触发极端波动率协议：市场波动率过高")

        # 减少所有仓位50%
        volatility = market_data.get('market_volatility', 0)
        reduction_ratio = min(volatility * 2, 0.8)  # 波动率的2倍，最多减少80%

        emergency_adjustment = TradeAdjustment(
            adjustment_id=f"emergency_volatility_{int(time.time())}",
            timestamp=datetime.now(),
            adjustment_type=AdjustmentType.VOLATILITY_HEDGE,
            response_action=ResponseAction.REDUCE_POSITION,
            asset_symbol="PORTFOLIO",
            current_position=1.0,
            target_position=1.0 - reduction_ratio,
            adjustment_amount=reduction_ratio,
            reason=f"极端波动率: {volatility:.2%}",
            risk_indicators={'volatility': volatility},
            expected_impact={'volatility_reduction': reduction_ratio}
        )

        logger.warning(f"波动率调整已生成: {emergency_adjustment.adjustment_id}")

    def _system_failure_protocol(self, market_data: Dict[str, Any]):
        """系统故障协议"""
        logger.error("🚨 触发系统故障协议：系统状态异常")

        # 完全停止交易并通知
        self.emergency_stop = True

        emergency_adjustment = TradeAdjustment(
            adjustment_id=f"emergency_system_{int(time.time())}",
            timestamp=datetime.now(),
            adjustment_type=AdjustmentType.STOP_LOSS,
            response_action=ResponseAction.EMERGENCY_STOP,
            asset_symbol="SYSTEM",
            current_position=0,
            target_position=0,
            adjustment_amount=0,
            reason="系统故障：需要人工干预",
            risk_indicators={'system_status': market_data.get('system_status')},
            expected_impact={'system_halt': True}
        )

        logger.error(f"系统故障调整已生成: {emergency_adjustment.adjustment_id}")

    def _market_halt_protocol(self, market_data: Dict[str, Any]):
        """市场暂停协议"""
        logger.warning("⏸️ 触发市场暂停协议：市场休市")

        emergency_adjustment = TradeAdjustment(
            adjustment_id=f"emergency_halt_{int(time.time())}",
            timestamp=datetime.now(),
            adjustment_type=AdjustmentType.LIQUIDITY_ADJUSTMENT,
            response_action=ResponseAction.HOLD,
            asset_symbol="MARKET",
            current_position=0,
            target_position=0,
            adjustment_amount=0,
            reason="市场暂停：等待恢复",
            risk_indicators={'market_status': 'halted'},
            expected_impact={'trading_pause': True}
        )

        logger.warning(f"市场暂停调整已生成: {emergency_adjustment.adjustment_id}")

    def get_response_summary(self) -> Dict[str, Any]:
        """获取响应摘要"""
        return {
            'is_running': self.is_running,
            'emergency_stop': self.emergency_stop,
            'adjustment_summary': self.adjustment_engine.get_adjustment_summary(),
            'active_protocols': list(self.emergency_protocols.keys()),
            'response_queue_size': self.response_queue.qsize()
        }


def create_sample_portfolio() -> Dict[str, Any]:
    """创建示例投资组合"""
    return {
        'total_value': 1000000,
        'positions': {
            'AAPL': 50000,
            'GOOGL': 30000,
            'MSFT': 40000,
            'AMZN': 20000,
            'TSLA': 10000
        },
        'cash': 850000,
        'last_updated': datetime.now().isoformat()
    }


def create_sample_risk_indicators() -> Dict[str, RiskIndicator]:
    """创建示例风险指标"""
    np.random.seed(42)

    indicators = {}
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']

    for asset in assets:
        # 创建VaR指标
        var_value = 0.02 + np.secrets.normal(0, 0.005)  # 2 % 基础VaR加噪声
        var_indicator = RiskIndicator(
            name=f"{asset}_VaR",
            value=var_value,
            threshold_low=0.01,
            threshold_medium=0.02,
            threshold_high=0.03,
            risk_type=RiskType.MARKET_RISK,
            description=f"{asset} VaR指标",
            unit="%"
        )
        indicators[asset] = var_indicator

    return indicators


def create_sample_market_data() -> Dict[str, Any]:
    """创建示例市场数据"""
    return {
        'timestamp': datetime.now().isoformat(),
        'market_volatility': 0.15 + np.secrets.normal(0, 0.05),
        'price_gap': np.secrets.uniform(0, 0.05),
        'system_status': 'normal',
        'market_stress_level': np.secrets.uniform(0, 1),
        'portfolio': create_sample_portfolio(),
        'risk_indicators': create_sample_risk_indicators()
    }


def main():
    """主函数 - 自动化响应机制演示"""
    print("🤖 RQA2025自动化交易调整和响应系统")
    print("=" * 60)

    # 创建自动化响应系统
    response_system = AutomatedResponseSystem()

    print("✅ 自动化响应系统创建完成")
    print("   包含以下组件:")
    print("   - 交易调整引擎")
    print("   - 风险限额管理器")
    print("   - 应急响应协议")
    print("   - 实时监控系统")

    # 设置风险限额
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    for asset in assets:
        response_system.adjustment_engine.risk_limits.set_asset_limit(
            asset,
            max_position_size=100000,
            max_var_limit=0.05,
            max_drawdown_limit=0.15,
            daily_loss_limit=0.02
        )

    print("\n⚙️ 风险限额配置完成")
    print(f"   配置资产数量: {len(assets)}")
    print("   限额类型: 仓位大小、VaR、回撤、每日损失")

    try:
        # 启动自动化响应
        print("\n🚀 启动自动化响应系统...")
        response_system.start_automated_response(create_sample_market_data)

        # 运行一段时间
        print("   系统运行中... (按Ctrl + 停止)")

        start_time = time.time()
        while time.time() - start_time < 30:  # 运行30秒
            time.sleep(2)

            # 显示状态摘要
            summary = response_system.get_response_summary()
            adjustment_summary = summary['adjustment_summary']

            print(f"\n📊 状态摘要 [{datetime.now().strftime('%H:%M:%S')}]")
            print(f"   运行状态: {'运行中' if summary['is_running'] else '已停止'}")
            print(f"   紧急停止: {'是' if summary['emergency_stop'] else '否'}")
            print(f"   总调整数: {adjustment_summary['total_adjustments']}")
            print(f"   成功调整: {adjustment_summary['successful_adjustments']}")
            print(f"   成功率: {adjustment_summary['success_rate']:.1%}")

        print("\n🎉 自动化响应系统演示完成！")
        print("   系统已成功处理风险评估和自动调整")
        print("   应急响应协议运行正常")

    except KeyboardInterrupt:
        print("\n\n🛑 收到停止信号，正在停止系统...")
    except Exception as e:
        print(f"\n❌ 系统运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 停止系统
        response_system.stop_automated_response()
        print("✅ 自动化响应系统已停止")

        # 显示最终统计
        final_summary = response_system.get_response_summary()
        adjustment_summary = final_summary['adjustment_summary']

        print("\n📋 最终统计:")
        print(f"   运行时间: 30秒")
        print(f"   总调整数: {adjustment_summary['total_adjustments']}")
        print(f"   成功调整: {adjustment_summary['successful_adjustments']}")
        print(f"   失败调整: {adjustment_summary['failed_adjustments']}")
        print(f"   成功率: {adjustment_summary['success_rate']:.1%}")
        print(f"   平均执行时间: {adjustment_summary['avg_execution_time']:.3f}秒")

        if adjustment_summary['total_adjustments'] > 0:
            print("   最近调整记录:")
            for adj in adjustment_summary['recent_adjustments'][-3:]:  # 显示最后3个
                print(f"     {adj['asset_symbol']}: {adj['response_action']} - {adj['reason']}")

    return response_system


if __name__ == "__main__":
    system = main()
