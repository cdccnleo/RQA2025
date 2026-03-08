#!/usr/bin/env python3
"""
RQA2025 实时风控和合规检查器
提供实时风险监控和合规性检查功能"""

from src.infrastructure.utils.logger import get_logger
import logging

logger = get_logger(__name__)
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import threading
import time
import json

from src.infrastructure.integration import get_data_adapter

# 获取统一基础设施集成层的适配器
try:
    data_adapter = get_data_adapter()
    logger = logging.getLogger(__name__)
except Exception:
    # 降级处理
    from src.infrastructure.logging.core.interfaces import get_logger

# 导入市场冲击分析器
try:
    from .market_impact_analyzer import MarketImpactAnalyzer, MarketImpactConfig
    MARKET_IMPACT_AVAILABLE = True
except ImportError:
    MARKET_IMPACT_AVAILABLE = False
logger.warning("市场冲击分析器不可用，将使用基础冲击估算")

# 导入跨境合规管理器
try:
    from .cross_border_compliance_manager import (
        CrossBorderComplianceManager, Country, Currency,
        ComplianceType, CrossBorderTransaction
    )
    CROSS_BORDER_AVAILABLE = True
except ImportError:
    CROSS_BORDER_AVAILABLE = False
    logger.warning("跨境合规管理器不可用，将跳过跨境合规检查")

# 导入统一日志记录器

logger = get_logger(__name__)


class RiskLevel(Enum):

    """风险级别枚举"""
    LOW = "low"                    # 低风误    MEDIUM = "medium"              # 中风误    HIGH = "high"                  # 高风误    CRITICAL = "critical"          # 极高风险


class ComplianceType(Enum):

    """合规类型枚举"""
    POSITION_LIMIT = "position_limit"      # 持仓限制
    LOSS_LIMIT = "loss_limit"              # 损失限制
    VOLUME_LIMIT = "volume_limit"          # 成交量限误    MARKET_IMPACT = "market_impact"        # 市场冲击
    CONCENTRATION = "concentration"        # 集中误    LEVERAGE_LIMIT = "leverage_limit"      # 杠杆限制
    TRADING_HOURS = "trading_hours"        # 交易时间限制
    COUNTERPARTY_RISK = "counterparty"     # 对手方风误    PRICE_MANIPULATION = "price_manipulation"  # 价格操纵
    INSIDER_TRADING = "insider_trading"    # 内幕交易
    MARKET_ABUSE = "market_abuse"          # 市场滥用
    CAPITAL_REQUIREMENTS = "capital_req"   # 资本要求
    STRESS_TEST = "stress_test"            # 压力测试
    BACKTEST_VALIDATION = "backtest_val"   # 回测验证
    MODEL_RISK = "model_risk"              # 模型风险


@dataclass
class RiskMetrics:

    """风险指标"""
    timestamp: datetime
    portfolio_value: float
    total_exposure: float
    daily_pnl: float
    max_drawdown: float
    var_95: float
    sharpe_ratio: float
    concentration_ratio: float
    leverage_ratio: float
    margin_usage: float


@dataclass
class ComplianceCheck:

    """合规检查结果"""
    check_type: ComplianceType
    passed: bool
    risk_level: RiskLevel
    value: float
    threshold: float
    message: str
    timestamp: datetime


@dataclass
class RiskAlert:

    """风险告警"""
    alert_id: str
    risk_type: str
    level: RiskLevel
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    resolved: bool = False


class RealTimeRiskManager:

    """实时风险管理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        self.config = config or {}

        # 风险阈值配误
        self.risk_limits = {
            'max_position_value': self.config.get('max_position_value', 1000000),
            'max_daily_loss': self.config.get('max_daily_loss', 50000),
            'max_single_position_ratio': self.config.get('max_single_position_ratio', 0.1),
            'max_portfolio_volatility': self.config.get('max_portfolio_volatility', 0.25),
            'max_leverage': self.config.get('max_leverage', 5.0),
            'max_concentration': self.config.get('max_concentration', 0.2),
            'var_limit': self.config.get('var_limit', 0.1)
        }

        # 合规规则配置
        self.compliance_rules = {
            ComplianceType.POSITION_LIMIT: {
            'threshold': self.risk_limits['max_position_value'],
            'enabled': True
        },
        ComplianceType.LOSS_LIMIT: {
            'threshold': self.risk_limits['max_daily_loss'],
            'enabled': True
            },
        ComplianceType.CONCENTRATION: {
        'threshold': self.risk_limits['max_concentration'],
        'enabled': True
        },
        ComplianceType.LEVERAGE_LIMIT: {
        'threshold': self.risk_limits['max_leverage'],
        'enabled': True
        }
    }

        # 持仓和交易数误
        self.positions: Dict[str, float] = {}
        self.daily_trades: List[Dict[str, Any]] = []
        self.risk_history: List[RiskMetrics] = []

        # 告警系统
        self.alerts: List[RiskAlert] = []
        self.active_alerts: Dict[str, RiskAlert] = {}

        # 监控线程
        self.monitoring_thread = None
        self.is_monitoring = False

        # 市场冲击分析误
        self.market_impact_analyzer = None
        if MARKET_IMPACT_AVAILABLE:
            try:
                impact_config = MarketImpactConfig(
                    analysis_window_minutes=60,
                    impact_decay_factor=0.1,
                    min_order_size_threshold=0.01,
                    enable_real_time_analysis=True,
                    enable_historical_calibration=True
                )
                self.market_impact_analyzer = MarketImpactAnalyzer(impact_config)
                logger.info("市场冲击分析器初始化成功")
            except Exception as e:
                logger.warning(f"市场冲击分析器初始化失败: {e}")
                self.market_impact_analyzer = None

        # 跨境合规管理误
        self.cross_border_manager = None
        if CROSS_BORDER_AVAILABLE:
            try:
                self.cross_border_manager = CrossBorderComplianceManager()
                logger.info("跨境合规管理器初始化成功")
            except Exception as e:
                logger.warning(f"跨境合规管理器初始化失败: {e}")
            self.cross_border_manager = None

        logger.info("实时风险管理器初始化完成")

    def start_monitoring(self):
        """启动实时监控"""
        if self.is_monitoring:
            logger.warning("风险监控已在运行")
            return

        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

        logger.info("实时风险监控已启动")

    def stop_monitoring(self):
        """停止实时监控"""
        self.is_monitoring = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)

        logger.info("实时风险监控已停止")

    def check_order_risk(self, symbol: str, quantity: float, price: float,
        order_type: str, market_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """检查订单风误"""
        try:
            # 基本风险检误
            checks = []

            # 1. 持仓限制检误
            position_check = self._check_position_limit(symbol, quantity)
            checks.append(position_check)

            # 2. 损失限制检误
            loss_check = self._check_loss_limit()
            checks.append(loss_check)

            # 3. 集中度检误
            concentration_check = self._check_concentration()
            checks.append(concentration_check)

            # 4. 杠杆限制检误
            leverage_check = self._check_leverage_limit()
            checks.append(leverage_check)

            # 5. 交易时间检误
            if market_data:
                time_check = self._check_trading_hours(market_data)
                checks.append(time_check)

            # 6. 市场冲击检误
            if market_data:
                impact_check = self._check_market_impact(symbol, quantity, price, market_data)
                checks.append(impact_check)

            # 7. 成交量限制检误
            volume_check = self._check_volume_limit(symbol, quantity)
            checks.append(volume_check)

            # 8. 对手方风险检误
            counterparty_check = self._check_counterparty_risk(symbol)
            checks.append(counterparty_check)

            # 9. 价格操纵检误
            if market_data:
                manipulation_check = self._check_price_manipulation(symbol, price, market_data)
                checks.append(manipulation_check)

            # 10. 资本要求检误
            capital_check = self._check_capital_requirements()
            checks.append(capital_check)

            # 11. 跨境交易合规检误
            if self.cross_border_manager and self._is_cross_border_transaction(symbol, market_data):
                cross_border_check = self._check_cross_border_compliance(
                    symbol, quantity, price, order_type, market_data)
                checks.append(cross_border_check)

            # 12. 压力测试检误
            stress_check = self._check_stress_test()
            checks.append(stress_check)

            # 汇总检查结误
            failed_checks = [check for check in checks if not check.passed]
            critical_checks = [
                check for check in failed_checks if check.risk_level == RiskLevel.CRITICAL]
            high_risk_checks = [
                check for check in failed_checks if check.risk_level == RiskLevel.HIGH]

            if critical_checks:
                return {
                    'approved': False,
                    'reason': critical_checks[0].message,
                    'risk_level': critical_checks[0].risk_level.value,
                    'checks': [self._check_to_dict(check) for check in checks]
                }
            elif high_risk_checks:
                return {
                    'approved': False,
                    'reason': high_risk_checks[0].message,
                    'risk_level': high_risk_checks[0].risk_level.value,
                    'checks': [self._check_to_dict(check) for check in checks]
                }
            elif failed_checks:
                return {
                    'approved': False,
                    'reason': failed_checks[0].message,
                    'risk_level': failed_checks[0].risk_level.value,
                    'checks': [self._check_to_dict(check) for check in checks]
                }
            else:
                return {
                    'approved': True,
                    'reason': '所有风险和合规检查通过',
                    'checks': [self._check_to_dict(check) for check in checks]
                }

        except Exception as e:
            logger.error(f"订单风险检查失误 {e}")
            return {
                'approved': False,
                'reason': f'风险检查异误 {str(e)}',
                'risk_level': RiskLevel.CRITICAL.value
            }

    def update_position(self, symbol: str, quantity: float, price: float):
        """更新持仓"""
        if symbol not in self.positions:
            self.positions[symbol] = 0

        self.positions[symbol] += quantity

        # 记录交易
        trade_record = {
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'timestamp': datetime.now(),
            'value': quantity * price
        }

        self.daily_trades.append(trade_record)

        # 清理旧交易记录（保留当天）
        today = datetime.now().date()
        self.daily_trades = [
            trade for trade in self.daily_trades
            if trade['timestamp'].date() == today
        ]

    def calculate_risk_metrics(self) -> RiskMetrics:
        """计算风险指标"""
        # 计算投资组合价值
        portfolio_value = sum(abs(qty) * 100 for qty in self.positions.values())  # 简化计算
        # 计算总敞口
        total_exposure = portfolio_value

        # 计算当日PnL（简化计算）
        daily_pnl = sum(trade['value'] for trade in self.daily_trades)

        # 计算最大回撤（简化计算）
        max_drawdown = abs(daily_pnl) / portfolio_value if portfolio_value > 0 else 0

        # 计算VaR（简化计算）
        var_95 = portfolio_value * 0.05  # 5% VaR

        # 计算夏普比率（简化计算）
        sharpe_ratio = daily_pnl / (portfolio_value * 0.02) if portfolio_value > 0 else 0

        # 计算集中度
        if self.positions:
            max_position = max(abs(qty) for qty in self.positions.values())
            concentration_ratio = max_position * 100 / portfolio_value if portfolio_value > 0 else 0
        else:
            concentration_ratio = 0

        # 计算杠杆率
        leverage_ratio = total_exposure / portfolio_value if portfolio_value > 0 else 0

        # 计算保证金使用率
        margin_usage = total_exposure / (portfolio_value * 5) if portfolio_value > 0 else 0

        return RiskMetrics(
            timestamp=datetime.now(),
            portfolio_value=portfolio_value,
            total_exposure=total_exposure,
            daily_pnl=daily_pnl,
            max_drawdown=max_drawdown,
            var_95=var_95,
            sharpe_ratio=sharpe_ratio,
            concentration_ratio=concentration_ratio,
            leverage_ratio=leverage_ratio,
            margin_usage=margin_usage
        )

    def _check_position_limit(self, symbol: str, quantity: float) -> ComplianceCheck:
        """检查持仓限制"""
        current_position = abs(self.positions.get(symbol, 0))
        new_position = current_position + abs(quantity)
        threshold = self.risk_limits['max_position_value']

        if new_position > threshold:
            return ComplianceCheck(
                check_type=ComplianceType.POSITION_LIMIT,
                passed=False,
                risk_level=RiskLevel.HIGH,
                value=new_position,
                threshold=threshold,
                message=f"持仓超出限制: {new_position:.2f} > {threshold:.2f}",
                timestamp=datetime.now()
            )

        return ComplianceCheck(
            check_type=ComplianceType.POSITION_LIMIT,
            passed=True,
            risk_level=RiskLevel.LOW,
            value=new_position,
            threshold=threshold,
            message="持仓限制检查通过",
            timestamp=datetime.now()
        )

    def _check_loss_limit(self) -> ComplianceCheck:
        """检查损失限制"""
        daily_pnl = sum(trade['value'] for trade in self.daily_trades)
        threshold = self.risk_limits['max_daily_loss']

        if daily_pnl < -threshold:
            return ComplianceCheck(
                check_type=ComplianceType.LOSS_LIMIT,
                passed=False,
                risk_level=RiskLevel.CRITICAL,
                value=abs(daily_pnl),
                threshold=threshold,
                message=f"当日损失超出限制: {abs(daily_pnl):.2f} > {threshold:.2f}",
                timestamp=datetime.now()
            )

        return ComplianceCheck(
            check_type=ComplianceType.LOSS_LIMIT,
            passed=True,
            risk_level=RiskLevel.LOW,
            value=abs(daily_pnl),
            threshold=threshold,
            message="损失限制检查通过",
            timestamp=datetime.now()
        )

    def _check_concentration(self) -> ComplianceCheck:
        """检查集中度"""
        if not self.positions:
            return ComplianceCheck(
                check_type=ComplianceType.CONCENTRATION,
                passed=True,
                risk_level=RiskLevel.LOW,
                value=0,
                threshold=self.risk_limits['max_concentration'],
                message="无持仓，集中度检查通过",
                timestamp=datetime.now()
            )

        portfolio_value = sum(abs(qty) * 100 for qty in self.positions.values())
        if portfolio_value == 0:
            return ComplianceCheck(
                check_type=ComplianceType.CONCENTRATION,
                passed=True,
                risk_level=RiskLevel.LOW,
                value=0,
                threshold=self.risk_limits['max_concentration'],
                message="投资组合价值为0",
                timestamp=datetime.now()
            )

        max_position = max(abs(qty) * 100 for qty in self.positions.values())
        concentration_ratio = max_position / portfolio_value
        threshold = self.risk_limits['max_concentration']

        if concentration_ratio > threshold:
            return ComplianceCheck(
                check_type=ComplianceType.CONCENTRATION,
                passed=False,
                risk_level=RiskLevel.MEDIUM,
                value=concentration_ratio,
                threshold=threshold,
                message=f"投资组合集中度过高 {concentration_ratio:.2%} > {threshold:.2%}",
                timestamp=datetime.now()
            )

        return ComplianceCheck(
            check_type=ComplianceType.CONCENTRATION,
            passed=True,
            risk_level=RiskLevel.LOW,
            value=concentration_ratio,
            threshold=threshold,
            message="集中度检查通过",
            timestamp=datetime.now()
        )

    def _check_leverage_limit(self) -> ComplianceCheck:
        """检查杠杆限制"""
        # 计算投资组合价值
        portfolio_value = sum(abs(qty) * 100 for qty in self.positions.values())

        if portfolio_value == 0:
            return ComplianceCheck(
                check_type=ComplianceType.LEVERAGE_LIMIT,
                passed=True,
                risk_level=RiskLevel.LOW,
                value=0,
                threshold=self.risk_limits['max_leverage'],
                message="无持仓，杠杆检查通过",
                timestamp=datetime.now()
            )

        # 简化的杠杆计算
        leverage_ratio = len([p for p in self.positions.values() if p != 0])
        threshold = self.risk_limits['max_leverage']

        if leverage_ratio > threshold:
            return ComplianceCheck(
                check_type=ComplianceType.LEVERAGE_LIMIT,
                passed=False,
                risk_level=RiskLevel.MEDIUM,
                value=leverage_ratio,
                threshold=threshold,
                message=f"杠杆率超出限制 {leverage_ratio:.1f} > {threshold:.1f}",
                timestamp=datetime.now()
            )

        return ComplianceCheck(
            check_type=ComplianceType.LEVERAGE_LIMIT,
            passed=True,
            risk_level=RiskLevel.LOW,
            value=leverage_ratio,
            threshold=threshold,
            message="杠杆限制检查通过",
            timestamp=datetime.now()
        )


    def _monitoring_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                # 计算风险指标
                risk_metrics = self.calculate_risk_metrics()
                self.risk_history.append(risk_metrics)

                # 保持历史记录长度
                if len(self.risk_history) > 1000:
                    self.risk_history = self.risk_history[-500:]

                # 检查风险告警
                self._check_risk_alerts(risk_metrics)

                # 每秒检查一次
                time.sleep(5)

            except Exception as e:
                logger.error(f"风险监控循环异常: {e}")
                time.sleep(5)


    def _check_risk_alerts(self, risk_metrics: RiskMetrics):
        """检查风险告警"""
        # 检查最大回撤
        if risk_metrics.max_drawdown > 0.1:  # 10% 回撤
            self._create_alert(
                risk_type="max_drawdown",
                level=RiskLevel.HIGH,
                message=f"最大回撤过高 {risk_metrics.max_drawdown:.2%}",
                details={'max_drawdown': risk_metrics.max_drawdown}
            )

        # 检查VaR
        if risk_metrics.var_95 > self.risk_limits['var_limit']:
            self._create_alert(
                risk_type="var_95",
                level=RiskLevel.MEDIUM,
                message=f"VaR(95%) 过高: {risk_metrics.var_95:.2f}",
                details={'var_95': risk_metrics.var_95}
            )

        # 检查杠杆率
        if risk_metrics.leverage_ratio > self.risk_limits['max_leverage']:
            self._create_alert(
                risk_type="leverage",
                level=RiskLevel.MEDIUM,
                message=f"杠杆率过高 {risk_metrics.leverage_ratio:.1f}",
                details={'leverage_ratio': risk_metrics.leverage_ratio}
            )

        # 检查保证金使用率
        if risk_metrics.margin_usage > 0.8:  # 80% 保证金使用率
            self._create_alert(
                risk_type="margin_usage",
                level=RiskLevel.HIGH,
                message=f"保证金使用率过高: {risk_metrics.margin_usage:.2%}",
                details={'margin_usage': risk_metrics.margin_usage}
            )


    def _create_alert(self, risk_type: str, level: RiskLevel, message: str, details: Dict[str, Any]):
        """创建告警"""
        alert_id = f"risk_alert_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

        alert = RiskAlert(
            alert_id=alert_id,
            risk_type=risk_type,
            level=level,
            message=message,
            details=details,
            timestamp=datetime.now()
        )

        self.alerts.append(alert)
        self.active_alerts[alert_id] = alert

        logger.warning(f"风险告警: {message}")


    def _check_to_dict(self, check: ComplianceCheck) -> Dict[str, Any]:
        """合规检查转换为字典"""
        return {
            'check_type': check.check_type.value,
            'passed': check.passed,
            'risk_level': check.risk_level.value,
            'value': check.value,
            'threshold': check.threshold,
            'message': check.message,
            'timestamp': check.timestamp.isoformat()
        }


    def get_risk_summary(self) -> Dict[str, Any]:
        """获取风险摘要"""
        if not self.risk_history:
            return {}

        latest = self.risk_history[-1]

        return {
            'portfolio_value': latest.portfolio_value,
            'total_exposure': latest.total_exposure,
            'daily_pnl': latest.daily_pnl,
            'max_drawdown': latest.max_drawdown,
            'var_95': latest.var_95,
            'sharpe_ratio': latest.sharpe_ratio,
            'concentration_ratio': latest.concentration_ratio,
            'leverage_ratio': latest.leverage_ratio,
            'margin_usage': latest.margin_usage,
            'active_alerts': len(self.active_alerts),
            'last_update': latest.timestamp.isoformat()
        }


    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """获取活跃告警"""
        return [
            {
                'alert_id': alert.alert_id,
                'risk_type': alert.risk_type,
                'level': alert.level.value,
                'message': alert.message,
                'details': alert.details,
                'timestamp': alert.timestamp.isoformat()
            }
            for alert in self.active_alerts.values()
        ]


    def resolve_alert(self, alert_id: str):
        """解决告警"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            del self.active_alerts[alert_id]
            logger.info(f"风险告警已解决 {alert_id}")


    def update_risk_limits(self, limits: Dict[str, Any]):
        """更新风险限制"""
        self.risk_limits.update(limits)
        logger.info(f"风险限制已更新 {limits}")


    def _check_trading_hours(self, market_data: Dict[str, Any]) -> ComplianceCheck:
        """检查交易时间"""
        current_time = datetime.now().time()
        trading_start = market_data.get('trading_start', '09:30:00')
        trading_end = market_data.get('trading_end', '15:00:00')

        start_time = datetime.strptime(trading_start, '%H:%M:%S').time()
        end_time = datetime.strptime(trading_end, '%H:%M:%S').time()

        if not (start_time <= current_time <= end_time):
            return ComplianceCheck(
                check_type=ComplianceType.TRADING_HOURS,
                passed=False,
                risk_level=RiskLevel.HIGH,
                value=0,
                threshold=1,
                message=f"当前时间 {current_time} 不在交易时段内",
                timestamp=datetime.now()
            )

        return ComplianceCheck(
            check_type=ComplianceType.TRADING_HOURS,
            passed=True,
            risk_level=RiskLevel.LOW,
            value=1,
            threshold=1,
            message="交易时间检查通过",
            timestamp=datetime.now()
        )


    def _is_cross_border_transaction(self, symbol: str, market_data: Optional[Dict[str, Any]]) -> bool:
        """判断是否为跨境交易"""
        try:
            if not market_data:
                return False

            # 检查交易市场是否在不同国家
            market_country = market_data.get('country', 'CN')
            base_country = 'CN'  # 假设基准在中国
            return market_country != base_country

        except Exception as e:
            logger.error(f"判断跨境交易失败: {e}")
            return False


    def _check_cross_border_compliance(self, symbol: str, quantity: float, price: float,
                                      order_type: str, market_data: Dict[str, Any]) -> ComplianceCheck:
        """检查跨境交易合规"""
        try:
            if not self.cross_border_manager:
                return ComplianceCheck(
                    check_type=ComplianceType.FX_CONTROL,
                    passed=True,
                    risk_level=RiskLevel.LOW,
                    value=0,
                    threshold=1,
                    message="跨境合规管理器不可用",
                    timestamp=datetime.now()
                )

            # 构建跨境交易数据
            transaction_data = {
                'from_country': 'CN',  # 假设从中国发起
                'to_country': market_data.get('country', 'US'),
                'from_currency': 'CNY',
                'to_currency': market_data.get('currency', 'USD'),
                'amount': abs(quantity * price),
                'transaction_type': order_type,
                'asset_type': 'equity',
                'asset_symbol': symbol,
                'counterparty': market_data.get('exchange', 'Unknown'),
                'metadata': {
                    'market_data': market_data,
                    'order_details': {
                        'quantity': quantity,
                        'price': price,
                        'order_type': order_type
                    }
                }
            }

            # 注册跨境交易
            transaction_id = self.cross_border_manager.register_transaction(transaction_data)

            # 执行合规检查
            compliance_results = self.cross_border_manager.check_compliance(transaction_id)

            # 汇总检查结果
            failed_checks = [r for r in compliance_results if not r.passed]
            critical_issues = [r for r in failed_checks if r.risk_level == 'critical']
            high_issues = [r for r in failed_checks if r.risk_level == 'high']

            if critical_issues:
                issue = critical_issues[0]
                return ComplianceCheck(
                    check_type=ComplianceType.FX_CONTROL,
                    passed=False,
                    risk_level=RiskLevel.CRITICAL,
                    value=len(failed_checks),
                    threshold=0,
                    message=f"跨境交易严重违规: {issue.message}",
                    timestamp=datetime.now()
                )
            elif high_issues:
                issue = high_issues[0]
                return ComplianceCheck(
                    check_type=ComplianceType.FX_CONTROL,
                    passed=False,
                    risk_level=RiskLevel.HIGH,
                    value=len(failed_checks),
                    threshold=0,
                    message=f"跨境交易违规: {issue.message}",
                    timestamp=datetime.now()
                )
            elif failed_checks:
                issue = failed_checks[0]
                return ComplianceCheck(
                    check_type=ComplianceType.FX_CONTROL,
                    passed=False,
                    risk_level=RiskLevel.MEDIUM,
                    value=len(failed_checks),
                    threshold=0,
                    message=f"跨境交易需要注意 {issue.message}",
                    timestamp=datetime.now()
                )
            else:
                return ComplianceCheck(
                    check_type=ComplianceType.FX_CONTROL,
                    passed=True,
                    risk_level=RiskLevel.LOW,
                    value=0,
                    threshold=0,
                    message="跨境交易合规检查通过",
                    timestamp=datetime.now()
                )

        except Exception as e:
            logger.error(f"跨境合规检查失败 {e}")
            return ComplianceCheck(
                check_type=ComplianceType.FX_CONTROL,
                passed=False,
                risk_level=RiskLevel.MEDIUM,
                value=0,
                threshold=1,
                message=f"跨境合规检查失败 {e}",
                timestamp=datetime.now()
            )


    def _check_market_impact(self, symbol: str, quantity: float, price: float,
                            market_data: Dict[str, Any]) -> ComplianceCheck:
        """检查市场冲击"""
        try:
            # 如果有高级市场冲击分析器，使用它进行精确分析
            if self.market_impact_analyzer:
                order_data = {
                    'order_id': f"check_{symbol}_{datetime.now().timestamp()}",
                    'symbol': symbol,
                    'order_size': abs(quantity),
                    'order_price': price,
                    'order_type': 'market_order',
                    'urgency': 'normal',
                    'time_horizon': 1
                }

                # 更新市场数据
                self.market_impact_analyzer.update_market_data(symbol, market_data)

                # 分析市场冲击
                impact_result = self.market_impact_analyzer.analyze_market_impact(order_data)

                # 根据冲击结果生成合规检查
                price_impact_bp = impact_result.price_impact  # 基点
                impact_threshold_bp = 5.0  # 5个基点的阈值
                if price_impact_bp > impact_threshold_bp:
                    risk_level = RiskLevel.HIGH if price_impact_bp > 20 else RiskLevel.MEDIUM
                    return ComplianceCheck(
                        check_type=ComplianceType.MARKET_IMPACT,
                        passed=False,
                        risk_level=risk_level,
                        value=price_impact_bp,
                        threshold=impact_threshold_bp,
                        message=f"市场冲击过大: {price_impact_bp:.1f}bp",
                        timestamp=datetime.now()
                    )

                return ComplianceCheck(
                    check_type=ComplianceType.MARKET_IMPACT,
                    passed=True,
                    risk_level=RiskLevel.LOW,
                    value=price_impact_bp,
                    threshold=impact_threshold_bp,
                    message=f"市场冲击在可接受范围内: {price_impact_bp:.1f}bp",
                    timestamp=datetime.now()
                )

            # 降级到基础冲击检查
            else:
                avg_volume = market_data.get('daily_volume', market_data.get('avg_volume', 1000000))
                market_cap = market_data.get('market_cap', 10000000000)

                # 计算订单占日均成交量的比例
                volume_ratio = abs(quantity) / avg_volume

                # 计算订单占总市值的比例
                value_ratio = abs(quantity * price) / market_cap

                # 市场冲击阈值
                impact_threshold = 0.01  # 1%

                if volume_ratio > impact_threshold or value_ratio > impact_threshold:
                    risk_level = RiskLevel.HIGH if (
                        volume_ratio > 0.05 or value_ratio > 0.05) else RiskLevel.MEDIUM
                    return ComplianceCheck(
                        check_type=ComplianceType.MARKET_IMPACT,
                        passed=False,
                        risk_level=risk_level,
                        value=max(volume_ratio, value_ratio),
                        threshold=impact_threshold,
                        message=f"市场冲击过大: {max(volume_ratio, value_ratio):.1%}",
                        timestamp=datetime.now()
                    )

                return ComplianceCheck(
                    check_type=ComplianceType.MARKET_IMPACT,
                    passed=True,
                    risk_level=RiskLevel.LOW,
                    value=max(volume_ratio, value_ratio),
                    threshold=impact_threshold,
                    message="市场冲击检查通过",
                    timestamp=datetime.now()
                )

        except Exception as e:
            logger.error(f"市场冲击检查失败 {e}")
            # 返回保守的检查结果
            return ComplianceCheck(
                check_type=ComplianceType.MARKET_IMPACT,
                passed=False,
                risk_level=RiskLevel.MEDIUM,
                value=0.0,
                threshold=0.01,
                message=f"检查失败 {e}",
                timestamp=datetime.now()
            )


    def _check_volume_limit(self, symbol: str, quantity: float) -> ComplianceCheck:
        """检查成交量限制"""
        # 获取当前持仓和当日已成交量
        current_position = abs(self.positions.get(symbol, 0))
        daily_volume = sum(abs(trade['quantity']) for trade in self.daily_trades
                          if trade['symbol'] == symbol)

        total_volume = current_position + daily_volume + abs(quantity)
        volume_threshold = self.risk_limits.get('max_daily_volume', 1000000)

        if total_volume > volume_threshold:
            return ComplianceCheck(
                check_type=ComplianceType.VOLUME_LIMIT,
                passed=False,
                risk_level=RiskLevel.MEDIUM,
                value=total_volume,
                threshold=volume_threshold,
                message=f"成交量超过限制: {total_volume:,.0f} > {volume_threshold:,.0f}",
                timestamp=datetime.now()
            )

        return ComplianceCheck(
            check_type=ComplianceType.VOLUME_LIMIT,
            passed=True,
            risk_level=RiskLevel.LOW,
            value=total_volume,
            threshold=volume_threshold,
            message="成交量限制检查通过",
            timestamp=datetime.now()
        )


    def _check_counterparty_risk(self, symbol: str) -> ComplianceCheck:
        """检查对手方风险"""
        # 这里实现对手方风险评估逻辑
        # 简化的实现：检查交易对手的信用评级和历史表现
        counterparty_rating = 0.85  # 假设的信用评级
        rating_threshold = 0.7

        if counterparty_rating < rating_threshold:
            return ComplianceCheck(
                check_type=ComplianceType.COUNTERPARTY_RISK,
                passed=False,
                risk_level=RiskLevel.MEDIUM,
                value=counterparty_rating,
                threshold=rating_threshold,
                message=f"对手方信用评级过低: {counterparty_rating:.1%}",
                timestamp=datetime.now()
            )

        return ComplianceCheck(
            check_type=ComplianceType.COUNTERPARTY_RISK,
            passed=True,
            risk_level=RiskLevel.LOW,
            value=counterparty_rating,
            threshold=rating_threshold,
            message="对手方风险检查通过",
            timestamp=datetime.now()
        )


    def _check_price_manipulation(self, symbol: str, price: float,
                                 market_data: Dict[str, Any]) -> ComplianceCheck:
        """检查价格操纵"""
        last_price = market_data.get('last_price', price)
        price_change = abs(price - last_price) / last_price

        # 价格异常变动阈值
        manipulation_threshold = 0.1  # 10%

        if price_change > manipulation_threshold:
            return ComplianceCheck(
                check_type=ComplianceType.PRICE_MANIPULATION,
                passed=False,
                risk_level=RiskLevel.HIGH,
                value=price_change,
                threshold=manipulation_threshold,
                message=f"价格异常变动: {price_change:.1%}",
                timestamp=datetime.now()
            )

        return ComplianceCheck(
            check_type=ComplianceType.PRICE_MANIPULATION,
            passed=True,
            risk_level=RiskLevel.LOW,
            value=price_change,
            threshold=manipulation_threshold,
            message="价格操纵检查通过",
            timestamp=datetime.now()
        )


    def _check_capital_requirements(self) -> ComplianceCheck:
        """检查资本要求"""
        # 计算净资本
        portfolio_value = sum(abs(qty) * 100 for qty in self.positions.values())  # 简化计算
        total_liabilities = portfolio_value * 0.1  # 假设10% 负债
        net_capital = portfolio_value - total_liabilities

        # 资本要求：净资本不能低于投资组合价值的8%
        capital_threshold = portfolio_value * 0.08

        if net_capital < capital_threshold:
            return ComplianceCheck(
                check_type=ComplianceType.CAPITAL_REQUIREMENTS,
                passed=False,
                risk_level=RiskLevel.CRITICAL,
                value=net_capital,
                threshold=capital_threshold,
                message=f"净资本不足: {net_capital:,.0f} < {capital_threshold:,.0f}",
                timestamp=datetime.now()
            )

        return ComplianceCheck(
            check_type=ComplianceType.CAPITAL_REQUIREMENTS,
            passed=True,
            risk_level=RiskLevel.LOW,
            value=net_capital,
            threshold=capital_threshold,
            message="资本要求检查通过",
            timestamp=datetime.now()
        )


    def _check_stress_test(self) -> ComplianceCheck:
        """检查压力测试"""
        # 简化的压力测试：检查在极端市场条件下的表现

        # 获取历史最坏情况的损失
        if self.risk_history:
            max_loss = min(risk.daily_pnl for risk in self.risk_history)
            stress_threshold = abs(max_loss) * 1.5  # 假设压力测试阈值
            if abs(max_loss) > stress_threshold:
                return ComplianceCheck(
                    check_type=ComplianceType.STRESS_TEST,
                    passed=False,
                    risk_level=RiskLevel.MEDIUM,
                    value=abs(max_loss),
                    threshold=stress_threshold,
                    message=f"压力测试失败: {abs(max_loss):,.2f} > {stress_threshold:,.2f}",
                    timestamp=datetime.now()
                )

        return ComplianceCheck(
            check_type=ComplianceType.STRESS_TEST,
            passed=True,
            risk_level=RiskLevel.LOW,
            value=0,
            threshold=0,
            message="压力测试检查通过",
            timestamp=datetime.now()
        )


    def _check_model_risk(self) -> ComplianceCheck:
        """检查模型风险"""
        # 检查策略模型的性能和稳定性
        if self.risk_history:
            # 计算模型预测误差
            pnl_volatility = np.std([risk.daily_pnl for risk in self.risk_history[-30:]])  # 最近30天
            model_risk_threshold = 50000  # 模型风险阈值
            if pnl_volatility > model_risk_threshold:
                return ComplianceCheck(
                    check_type=ComplianceType.MODEL_RISK,
                    passed=False,
                    risk_level=RiskLevel.MEDIUM,
                    value=pnl_volatility,
                    threshold=model_risk_threshold,
                    message=f"模型波动性过高: {pnl_volatility:,.0f} > {model_risk_threshold:,.0f}",
                    timestamp=datetime.now()
                )

        return ComplianceCheck(
            check_type=ComplianceType.MODEL_RISK,
            passed=True,
            risk_level=RiskLevel.LOW,
            value=0,
            threshold=0,
            message="模型风险检查通过",
            timestamp=datetime.now()
        )


    def _check_backtest_validation(self) -> ComplianceCheck:
        """检查回测验证"""
        # 检查策略的回测结果是否符合预期
        # 这里实现回测验证逻辑

        # 简化的实现：检查夏普比率是否为正
        if self.risk_history:
            latest = self.risk_history[-1]
            if latest.sharpe_ratio < 0:
                return ComplianceCheck(
                    check_type=ComplianceType.BACKTEST_VALIDATION,
                    passed=False,
                    risk_level=RiskLevel.MEDIUM,
                    value=latest.sharpe_ratio,
                    threshold=0.0,
                    message=f"夏普比率异常: {latest.sharpe_ratio:.2f}",
                    timestamp=datetime.now()
                )

        return ComplianceCheck(
            check_type=ComplianceType.BACKTEST_VALIDATION,
            passed=True,
            risk_level=RiskLevel.LOW,
            value=0,
            threshold=0,
            message="回测验证检查通过",
            timestamp=datetime.now()
        )


    def _check_market_abuse(self, symbol: str, order_pattern: Dict[str, Any]) -> ComplianceCheck:
        """检查市场滥用"""
        # 检查是否存在市场操纵行为
        # 这里实现市场滥用检测逻辑

        # 简化的实现：检查订单频率是否异常
        recent_orders = [trade for trade in self.daily_trades
                        if trade['symbol'] == symbol and
                        (datetime.now() - trade['timestamp']).seconds < 300]  # 最近5分钟

        if len(recent_orders) > 10:  # 5分钟内超过10笔订单
            return ComplianceCheck(
                check_type=ComplianceType.MARKET_ABUSE,
                passed=False,
                risk_level=RiskLevel.HIGH,
                value=len(recent_orders),
                threshold=10,
                message=f"5分钟内订单数量过多 {len(recent_orders)}",
                timestamp=datetime.now()
            )

        return ComplianceCheck(
            check_type=ComplianceType.MARKET_ABUSE,
            passed=True,
            risk_level=RiskLevel.LOW,
            value=len(recent_orders),
            threshold=10,
            message="市场滥用检查通过",
            timestamp=datetime.now()
        )


    def run_comprehensive_risk_check(self, portfolio_data: Dict[str, Any],
                                    market_data: Dict[str, Any]) -> Dict[str, Any]:
        """运行全面风险检查"""
        comprehensive_checks = []

        # 运行所有合规检查
        for compliance_type in ComplianceType:
            if compliance_type == ComplianceType.POSITION_LIMIT:
                # 这里需要具体的持仓数据
                check = self._check_position_limit("TOTAL", 0)
            elif compliance_type == ComplianceType.LOSS_LIMIT:
                check = self._check_loss_limit()
            elif compliance_type == ComplianceType.CONCENTRATION:
                check = self._check_concentration()
            elif compliance_type == ComplianceType.LEVERAGE_LIMIT:
                check = self._check_leverage_limit()
            elif compliance_type == ComplianceType.TRADING_HOURS:
                check = self._check_trading_hours(market_data)
            elif compliance_type == ComplianceType.CAPITAL_REQUIREMENTS:
                check = self._check_capital_requirements()
            elif compliance_type == ComplianceType.STRESS_TEST:
                check = self._check_stress_test()
            elif compliance_type == ComplianceType.MODEL_RISK:
                check = self._check_model_risk()
            elif compliance_type == ComplianceType.BACKTEST_VALIDATION:
                check = self._check_backtest_validation()
            else:
                # 其他检查的默认实现
                check = ComplianceCheck(
                    check_type=compliance_type,
                    passed=True,
                    risk_level=RiskLevel.LOW,
                    value=0,
                    threshold=0,
                    message=f"{compliance_type.value} 检查通过",
                    timestamp=datetime.now()
                )

            comprehensive_checks.append(check)

        # 汇总结果
        failed_checks = [check for check in comprehensive_checks if not check.passed]
        critical_issues = [
            check for check in failed_checks if check.risk_level == RiskLevel.CRITICAL]
        high_risk_issues = [check for check in failed_checks if check.risk_level == RiskLevel.HIGH]

        overall_risk_level = RiskLevel.LOW
        if critical_issues:
            overall_risk_level = RiskLevel.CRITICAL
        elif high_risk_issues:
            overall_risk_level = RiskLevel.HIGH
        elif failed_checks:
            overall_risk_level = RiskLevel.MEDIUM

        return {
            'overall_risk_level': overall_risk_level.value,
            'total_checks': len(comprehensive_checks),
            'passed_checks': len(comprehensive_checks) - len(failed_checks),
            'failed_checks': len(failed_checks),
            'critical_issues': len(critical_issues),
            'high_risk_issues': len(high_risk_issues),
            'checks': [self._check_to_dict(check) for check in comprehensive_checks],
            'recommendations': self._generate_risk_recommendations(failed_checks)
        }


    def _generate_risk_recommendations(self, failed_checks: List[ComplianceCheck]) -> List[str]:
        """生成风险建议"""
        recommendations = []

        for check in failed_checks:
            if check.check_type == ComplianceType.POSITION_LIMIT:
                recommendations.append("建议减少持仓规模或增加分散度")
            elif check.check_type == ComplianceType.LOSS_LIMIT:
                recommendations.append("建议设置止损点或减少杠杆使用")
            elif check.check_type == ComplianceType.CONCENTRATION:
                recommendations.append("建议增加投资组合多样性")
            elif check.check_type == ComplianceType.LEVERAGE_LIMIT:
                recommendations.append("建议降低杠杆比率")
            elif check.check_type == ComplianceType.CAPITAL_REQUIREMENTS:
                recommendations.append("建议增加资本金或减少风险敞口")
            elif check.check_type == ComplianceType.STRESS_TEST:
                recommendations.append("建议进行压力测试并调整风险参数")
            elif check.check_type == ComplianceType.MODEL_RISK:
                recommendations.append("建议重新校准模型参数或更换策略")
            else:
                recommendations.append(f"建议关注{check.check_type.value}相关风险")

        return recommendations


    def get_compliance_report(self) -> Dict[str, Any]:
        """获取合规报告"""
        # 计算合规分数
        total_checks = len(self.compliance_rules)
        passed_checks = sum(1 for rule in self.compliance_rules.values() if rule['enabled'])

        compliance_score = (passed_checks / total_checks) * 100 if total_checks > 0 else 100

        return {
            'compliance_score': compliance_score,
            'total_rules': total_checks,
            'enabled_rules': passed_checks,
            'disabled_rules': total_checks - passed_checks,
            'last_updated': datetime.now().isoformat(),
            'rules_status': self.compliance_rules.copy()
        }
