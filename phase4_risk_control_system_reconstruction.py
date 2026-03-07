#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 4: 风险控制系统重建

修复技术债务: 风险控制系统重建
解决业务验收测试中发现的风险控制功能严重缺失的问题
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
import threading
import numpy as np
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

# 配置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 风险控制数据结构


class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskType(Enum):
    """风险类型"""
    POSITION = "position_risk"  # 仓位风险
    VOLATILITY = "volatility_risk"  # 波动率风险
    LIQUIDITY = "liquidity_risk"  # 流动性风险
    CONCENTRATION = "concentration_risk"  # 集中度风险
    MARKET = "market_risk"  # 市场风险
    OPERATIONAL = "operational_risk"  # 操作风险
    COMPLIANCE = "compliance_risk"  # 合规风险


class AlertSeverity(Enum):
    """告警严重程度"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class InterventionType(Enum):
    """干预类型"""
    NONE = "none"
    WARNING = "warning"
    PAUSE_TRADING = "pause_trading"
    FORCE_CLOSE = "force_close"
    SUSPEND_ACCOUNT = "suspend_account"


@dataclass
class RiskRule:
    """风险规则"""
    rule_id: str
    rule_name: str
    risk_type: RiskType
    threshold: float
    severity: AlertSeverity
    intervention: InterventionType
    description: str
    enabled: bool = True
    cooldown_minutes: int = 30


@dataclass
class RiskMetric:
    """风险指标"""
    metric_id: str
    risk_type: RiskType
    symbol: Optional[str]
    value: float
    threshold: float
    risk_level: RiskLevel
    timestamp: datetime
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskAlert:
    """风险告警"""
    alert_id: str
    rule_id: str
    risk_type: RiskType
    severity: AlertSeverity
    message: str
    metric_value: float
    threshold: float
    timestamp: datetime
    symbol: Optional[str] = None
    resolved: bool = False
    resolved_time: Optional[datetime] = None
    intervention_taken: InterventionType = InterventionType.NONE


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    quantity: float
    average_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float


@dataclass
class Account:
    """账户信息"""
    account_id: str
    balance: float
    positions: Dict[str, Position]
    total_value: float


class RiskCalculator:
    """风险计算器"""

    def __init__(self):
        self.calculators = {
            RiskType.POSITION: self._calculate_position_risk,
            RiskType.VOLATILITY: self._calculate_volatility_risk,
            RiskType.LIQUIDITY: self._calculate_liquidity_risk,
            RiskType.CONCENTRATION: self._calculate_concentration_risk,
            RiskType.MARKET: self._calculate_market_risk,
            RiskType.OPERATIONAL: self._calculate_operational_risk,
            RiskType.COMPLIANCE: self._calculate_compliance_risk
        }

    def calculate_risk(self, risk_type: RiskType, data: Dict[str, Any]) -> float:
        """计算指定类型的风险"""
        calculator = self.calculators.get(risk_type)
        if calculator:
            return calculator(data)
        return 0.0

    def _calculate_position_risk(self, data: Dict[str, Any]) -> float:
        """计算仓位风险"""
        positions = data.get('positions', [])
        total_value = data.get('total_value', 1.0)

        if not positions:
            return 0.0

        try:
            # 计算最大单股票仓位占比
            max_position_ratio = 0.0
            for position in positions:
                if hasattr(position, 'market_value'):
                    position_ratio = abs(position.market_value) / total_value
                    max_position_ratio = max(max_position_ratio, position_ratio)
                else:
                    logger.warning(f"Position object missing market_value attribute: {position}")

            # 如果最大仓位超过20%，风险增加
            if max_position_ratio > 0.2:
                return min(1.0, (max_position_ratio - 0.2) / 0.3)
            return 0.0
        except Exception as e:
            logger.error(f"计算仓位风险失败: {e}")
            return 0.0

    def _calculate_volatility_risk(self, data: Dict[str, Any]) -> float:
        """计算波动率风险"""
        returns = data.get('returns', [])
        if returns is None or len(returns) < 10:
            return 0.0

        try:
            # 计算历史波动率
            returns_array = np.array(returns)
            volatility = np.std(returns_array) * np.sqrt(252)  # 年化波动率

            # 如果波动率超过30%，风险增加
            if volatility > 0.3:
                return min(1.0, (volatility - 0.3) / 0.4)
            return 0.0
        except Exception as e:
            logger.error(f"计算波动率风险失败: {e}")
            return 0.0

    def _calculate_liquidity_risk(self, data: Dict[str, Any]) -> float:
        """计算流动性风险"""
        volume = data.get('volume', 0)
        market_cap = data.get('market_cap', 1.0)

        if volume == 0 or market_cap == 0:
            return 1.0  # 无流动性，风险最高

        # 换手率 = 成交量 / 流通市值
        turnover_ratio = volume / market_cap

        # 如果换手率低于1%，流动性风险增加
        if turnover_ratio < 0.01:
            return min(1.0, (0.01 - turnover_ratio) / 0.01)
        return 0.0

    def _calculate_concentration_risk(self, data: Dict[str, Any]) -> float:
        """计算集中度风险"""
        positions = data.get('positions', [])
        if len(positions) <= 1:
            return 0.0

        total_value = sum(abs(p.market_value) for p in positions)
        if total_value == 0:
            return 0.0

        # 计算赫芬达尔-赫希曼指数 (HHI)
        hhi = sum((abs(p.market_value) / total_value) ** 2 for p in positions)

        # HHI > 0.25 表示高度集中，风险增加
        if hhi > 0.25:
            return min(1.0, (hhi - 0.25) / 0.5)
        return 0.0

    def _calculate_market_risk(self, data: Dict[str, Any]) -> float:
        """计算市场风险"""
        market_return = data.get('market_return', 0)
        portfolio_beta = data.get('portfolio_beta', 1.0)

        # 使用CAPM模型估算市场风险
        expected_return = 0.03 + portfolio_beta * (market_return - 0.03)  # 假设无风险利率3%

        # 如果预期收益率过低或beta过高，风险增加
        risk_score = 0.0
        if expected_return < -0.1:  # 预期亏损超过10%
            risk_score += 0.5
        if portfolio_beta > 1.5:  # Beta过高
            risk_score += min(0.5, (portfolio_beta - 1.5) / 1.0)

        return min(1.0, risk_score)

    def _calculate_operational_risk(self, data: Dict[str, Any]) -> float:
        """计算操作风险"""
        error_count = data.get('error_count', 0)
        trade_count = data.get('trade_count', 1)

        if trade_count == 0:
            return 0.0

        # 错误率
        error_rate = error_count / trade_count

        # 如果错误率超过5%，风险增加
        if error_rate > 0.05:
            return min(1.0, (error_rate - 0.05) / 0.1)
        return 0.0

    def _calculate_compliance_risk(self, data: Dict[str, Any]) -> float:
        """计算合规风险"""
        violations = data.get('violations', [])
        compliance_score = data.get('compliance_score', 1.0)

        # 基于违规数量和合规评分计算风险
        violation_risk = min(1.0, len(violations) / 10.0)  # 每10个违规风险增加1.0
        score_risk = max(0.0, (1.0 - compliance_score) * 0.5)  # 合规评分每降低0.2，风险增加0.1

        return min(1.0, violation_risk + score_risk)


class RiskRuleEngine:
    """风险规则引擎"""

    def __init__(self):
        self.rules: Dict[str, RiskRule] = {}
        self.alert_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._init_default_rules()

    def _init_default_rules(self):
        """初始化默认风险规则"""
        default_rules = [
            RiskRule(
                rule_id="position_limit",
                rule_name="单股票仓位限制",
                risk_type=RiskType.POSITION,
                threshold=0.2,
                severity=AlertSeverity.WARNING,
                intervention=InterventionType.WARNING,
                description="单股票仓位不得超过总资产的20%"
            ),
            RiskRule(
                rule_id="high_position_limit",
                rule_name="高风险仓位限制",
                risk_type=RiskType.POSITION,
                threshold=0.3,
                severity=AlertSeverity.ERROR,
                intervention=InterventionType.PAUSE_TRADING,
                description="单股票仓位超过30%时暂停交易"
            ),
            RiskRule(
                rule_id="volatility_alert",
                rule_name="高波动率告警",
                risk_type=RiskType.VOLATILITY,
                threshold=0.4,
                severity=AlertSeverity.WARNING,
                intervention=InterventionType.WARNING,
                description="年化波动率超过40%时发出告警"
            ),
            RiskRule(
                rule_id="liquidity_risk",
                rule_name="流动性风险",
                risk_type=RiskType.LIQUIDITY,
                threshold=0.01,
                severity=AlertSeverity.ERROR,
                intervention=InterventionType.FORCE_CLOSE,
                description="换手率低于1%时强制平仓"
            ),
            RiskRule(
                rule_id="concentration_risk",
                rule_name="集中度风险",
                risk_type=RiskType.CONCENTRATION,
                threshold=0.25,
                severity=AlertSeverity.WARNING,
                intervention=InterventionType.WARNING,
                description="投资组合集中度过高"
            ),
            RiskRule(
                rule_id="compliance_violation",
                rule_name="合规违规",
                risk_type=RiskType.COMPLIANCE,
                threshold=0.1,
                severity=AlertSeverity.CRITICAL,
                intervention=InterventionType.SUSPEND_ACCOUNT,
                description="严重合规违规，暂停账户"
            )
        ]

        for rule in default_rules:
            self.add_rule(rule)

    def add_rule(self, rule: RiskRule):
        """添加风险规则"""
        self.rules[rule.rule_id] = rule
        logger.info(f"添加风险规则: {rule.rule_name}")

    def remove_rule(self, rule_id: str):
        """移除风险规则"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"移除风险规则: {rule_id}")

    def evaluate_metric(self, metric: RiskMetric) -> Optional[RiskAlert]:
        """评估风险指标是否触发告警"""
        for rule in self.rules.values():
            if not rule.enabled or rule.risk_type != metric.risk_type:
                continue

            # 检查是否在冷却期内
            if self._is_in_cooldown(rule.rule_id, metric.symbol):
                continue

            # 检查是否超过阈值
            if self._exceeds_threshold(metric, rule):
                alert = RiskAlert(
                    alert_id=str(uuid.uuid4()),
                    rule_id=rule.rule_id,
                    risk_type=rule.risk_type,
                    severity=rule.severity,
                    message=f"{rule.rule_name}: {metric.description}",
                    metric_value=metric.value,
                    threshold=rule.threshold,
                    timestamp=datetime.now(),
                    symbol=metric.symbol
                )

                # 记录告警历史
                self.alert_history[rule.rule_id].append(alert.timestamp)

                return alert

        return None

    def _is_in_cooldown(self, rule_id: str, symbol: Optional[str]) -> bool:
        """检查是否在冷却期内"""
        if not self.alert_history[rule_id]:
            return False

        last_alert_time = self.alert_history[rule_id][-1]
        cooldown_period = timedelta(minutes=self.rules[rule_id].cooldown_minutes)

        return datetime.now() - last_alert_time < cooldown_period

    def _exceeds_threshold(self, metric: RiskMetric, rule: RiskRule) -> bool:
        """检查是否超过阈值"""
        if rule.risk_type == RiskType.POSITION:
            return metric.value > rule.threshold
        elif rule.risk_type == RiskType.VOLATILITY:
            return metric.value > rule.threshold
        elif rule.risk_type == RiskType.LIQUIDITY:
            return metric.value < rule.threshold  # 流动性风险：值低于阈值
        elif rule.risk_type == RiskType.CONCENTRATION:
            return metric.value > rule.threshold
        elif rule.risk_type == RiskType.COMPLIANCE:
            return metric.value > rule.threshold
        else:
            return metric.value > rule.threshold


class RiskMonitor:
    """风险监控器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.monitoring_interval = self.config.get('monitoring_interval', 5)  # 秒
        self.running = False
        self.monitor_thread = None
        self.risk_calculator = RiskCalculator()
        self.rule_engine = RiskRuleEngine()
        self.alert_handlers: List[Callable] = []
        self.executor = ThreadPoolExecutor(max_workers=4)

        # 数据存储
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.active_alerts: Dict[str, RiskAlert] = {}

        logger.info("风险监控器初始化完成")

    def start_monitoring(self):
        """启动监控"""
        if self.running:
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("风险监控器已启动")

    def stop_monitoring(self):
        """停止监控"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("风险监控器已停止")

    def add_alert_handler(self, handler: Callable):
        """添加告警处理器"""
        self.alert_handlers.append(handler)

    def evaluate_portfolio_risk(self, account: Account, market_data: Dict[str, Any]) -> List[RiskMetric]:
        """评估投资组合风险"""
        logger.debug("开始评估投资组合风险指标")
        metrics = []

        # 仓位风险
        try:
            positions_list = list(account.positions.values())
            logger.debug(f"计算仓位风险，持仓数量: {len(positions_list)}")
            position_risk = self.risk_calculator.calculate_risk(
                RiskType.POSITION,
                {
                    'positions': positions_list,
                    'total_value': account.total_value
                }
            )
            logger.debug(f"仓位风险计算完成: {position_risk}")
            metrics.append(RiskMetric(
                metric_id=str(uuid.uuid4()),
                risk_type=RiskType.POSITION,
                symbol=None,
                value=position_risk,
                threshold=0.2,
                risk_level=self._get_risk_level(position_risk),
                timestamp=datetime.now(),
                description="投资组合仓位集中度风险"
            ))
        except Exception as e:
            logger.error(f"计算仓位风险失败: {e}")
            raise

        # 波动率风险
        returns = market_data.get('returns', [])
        logger.debug(
            f"波动率计算 - returns类型: {type(returns)}, 长度: {len(returns) if hasattr(returns, '__len__') else 0}")
        if returns is not None and len(returns) > 0:
            try:
                logger.debug("开始计算波动率风险")
                volatility_risk = self.risk_calculator.calculate_risk(
                    RiskType.VOLATILITY,
                    {'returns': returns}
                )
                logger.debug(f"波动率风险计算完成: {volatility_risk}")
                metrics.append(RiskMetric(
                    metric_id=str(uuid.uuid4()),
                    risk_type=RiskType.VOLATILITY,
                    symbol=None,
                    value=volatility_risk,
                    threshold=0.3,
                    risk_level=self._get_risk_level(volatility_risk),
                    timestamp=datetime.now(),
                    description="投资组合波动率风险"
                ))
                logger.debug("波动率风险指标添加完成")
            except Exception as e:
                logger.error(f"计算波动率风险失败: {e}")
                raise

        # 集中度风险
        concentration_risk = self.risk_calculator.calculate_risk(
            RiskType.CONCENTRATION,
            {'positions': list(account.positions.values())}
        )
        metrics.append(RiskMetric(
            metric_id=str(uuid.uuid4()),
            risk_type=RiskType.CONCENTRATION,
            symbol=None,
            value=concentration_risk,
            threshold=0.25,
            risk_level=self._get_risk_level(concentration_risk),
            timestamp=datetime.now(),
            description="投资组合集中度风险"
        ))

        # 逐个检查股票的流动性风险
        for symbol, position in account.positions.items():
            stock_data = market_data.get(symbol, {})
            liquidity_risk = self.risk_calculator.calculate_risk(
                RiskType.LIQUIDITY,
                {
                    'volume': stock_data.get('volume', 0),
                    'market_cap': stock_data.get('market_cap', 1.0)
                }
            )
            if liquidity_risk > 0.1:  # 只报告有意义的流动性风险
                metrics.append(RiskMetric(
                    metric_id=str(uuid.uuid4()),
                    risk_type=RiskType.LIQUIDITY,
                    symbol=symbol,
                    value=liquidity_risk,
                    threshold=0.01,
                    risk_level=self._get_risk_level(liquidity_risk),
                    timestamp=datetime.now(),
                    description=f"{symbol}流动性风险"
                ))

        return metrics

    def _get_risk_level(self, risk_score: float) -> RiskLevel:
        """根据风险分数确定风险等级"""
        if risk_score >= 0.8:
            return RiskLevel.CRITICAL
        elif risk_score >= 0.6:
            return RiskLevel.HIGH
        elif risk_score >= 0.3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _monitoring_loop(self):
        """监控循环"""
        while self.running:
            try:
                # 这里可以集成实际的账户和市场数据监控
                # 现在只是定期检查
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"监控循环异常: {e}")

    def process_alert(self, alert: RiskAlert):
        """处理告警"""
        self.active_alerts[alert.alert_id] = alert

        # 通知所有处理器
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"告警处理器异常: {e}")

        logger.warning(f"风险告警: {alert.message} (严重程度: {alert.severity.value})")

    def resolve_alert(self, alert_id: str):
        """解决告警"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_time = datetime.now()
            logger.info(f"告警已解决: {alert.message}")


class RiskInterventionEngine:
    """风险干预引擎"""

    def __init__(self):
        self.intervention_handlers = {
            InterventionType.WARNING: self._handle_warning,
            InterventionType.PAUSE_TRADING: self._handle_pause_trading,
            InterventionType.FORCE_CLOSE: self._handle_force_close,
            InterventionType.SUSPEND_ACCOUNT: self._handle_suspend_account
        }

    def execute_intervention(self, alert: RiskAlert) -> bool:
        """执行干预措施"""
        intervention_type = alert.intervention_taken
        if intervention_type == InterventionType.NONE:
            return True

        handler = self.intervention_handlers.get(intervention_type)
        if handler:
            try:
                return handler(alert)
            except Exception as e:
                logger.error(f"执行干预措施失败: {e}")
                return False
        else:
            logger.error(f"未知的干预类型: {intervention_type}")
            return False

    def _handle_warning(self, alert: RiskAlert) -> bool:
        """处理警告干预"""
        logger.warning(f"风险警告: {alert.message}")
        # 这里可以发送邮件、短信等通知
        return True

    def _handle_pause_trading(self, alert: RiskAlert) -> bool:
        """处理暂停交易干预"""
        logger.warning(f"暂停交易: {alert.message}")
        # 这里应该暂停相关交易活动
        return True

    def _handle_force_close(self, alert: RiskAlert) -> bool:
        """处理强制平仓干预"""
        logger.error(f"强制平仓: {alert.message}")
        # 这里应该执行强制平仓操作
        return True

    def _handle_suspend_account(self, alert: RiskAlert) -> bool:
        """处理暂停账户干预"""
        logger.critical(f"暂停账户: {alert.message}")
        # 这里应该暂停账户所有活动
        return True


class ComprehensiveRiskControlSystem:
    """全面风险控制系统 - Phase 4重建版本"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # 核心组件
        self.monitor = RiskMonitor(self.config)
        self.intervention_engine = RiskInterventionEngine()

        # 统计信息
        self.stats = {
            'total_alerts': 0,
            'active_alerts': 0,
            'resolved_alerts': 0,
            'interventions': 0,
            'risk_assessments': 0,
            'start_time': datetime.now()
        }

        # 设置告警处理器
        self.monitor.add_alert_handler(self._handle_alert)

        logger.info("全面风险控制系统初始化完成")

    def start(self) -> bool:
        """启动风险控制系统"""
        try:
            self.monitor.start_monitoring()
            logger.info("风险控制系统已启动")
            return True
        except Exception as e:
            logger.error(f"启动风险控制系统失败: {e}")
            return False

    def stop(self) -> bool:
        """停止风险控制系统"""
        try:
            self.monitor.stop_monitoring()
            logger.info("风险控制系统已停止")
            return True
        except Exception as e:
            logger.error(f"停止风险控制系统失败: {e}")
            return False

    def assess_portfolio_risk(self, account: Account, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """评估投资组合风险"""
        try:
            logger.debug("开始评估投资组合风险")
            # 计算风险指标
            metrics = self.monitor.evaluate_portfolio_risk(account, market_data)
            logger.debug(f"计算得到 {len(metrics)} 个风险指标")

            # 检查是否触发告警
            alerts = []
            for metric in metrics:
                alert = self.monitor.rule_engine.evaluate_metric(metric)
                if alert:
                    alerts.append(alert)
                    self.monitor.process_alert(alert)

            self.stats['risk_assessments'] += 1

            return {
                'metrics': [self._metric_to_dict(m) for m in metrics],
                'alerts': [self._alert_to_dict(a) for a in alerts],
                'overall_risk_level': self._calculate_overall_risk_level(metrics),
                'assessment_time': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"评估投资组合风险失败: {e}")
            return {
                'metrics': [],
                'alerts': [],
                'overall_risk_level': 'UNKNOWN',
                'error': str(e)
            }

    def _calculate_overall_risk_level(self, metrics: List[RiskMetric]) -> str:
        """计算整体风险等级"""
        if not metrics:
            return 'LOW'

        max_risk_score = max(m.value for m in metrics)
        if max_risk_score >= 0.8:
            return 'CRITICAL'
        elif max_risk_score >= 0.6:
            return 'HIGH'
        elif max_risk_score >= 0.3:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _handle_alert(self, alert: RiskAlert):
        """处理告警"""
        self.stats['total_alerts'] += 1
        self.stats['active_alerts'] += 1

        # 执行干预措施
        if alert.intervention_taken != InterventionType.NONE:
            success = self.intervention_engine.execute_intervention(alert)
            if success:
                self.stats['interventions'] += 1

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'running': self.monitor.running,
            'stats': self.stats.copy(),
            'active_alerts_count': len(self.monitor.active_alerts),
            'rules_count': len(self.monitor.rule_engine.rules),
            'last_update': datetime.now().isoformat()
        }

    def _metric_to_dict(self, metric: RiskMetric) -> Dict[str, Any]:
        """将RiskMetric转换为字典"""
        return {
            'metric_id': metric.metric_id,
            'risk_type': metric.risk_type.value,
            'symbol': metric.symbol,
            'value': metric.value,
            'threshold': metric.threshold,
            'risk_level': metric.risk_level.value,
            'timestamp': metric.timestamp.isoformat(),
            'description': metric.description
        }

    def _alert_to_dict(self, alert: RiskAlert) -> Dict[str, Any]:
        """将RiskAlert转换为字典"""
        return {
            'alert_id': alert.alert_id,
            'rule_id': alert.rule_id,
            'risk_type': alert.risk_type.value,
            'severity': alert.severity.value,
            'message': alert.message,
            'metric_value': alert.metric_value,
            'threshold': alert.threshold,
            'timestamp': alert.timestamp.isoformat(),
            'symbol': alert.symbol,
            'resolved': alert.resolved
        }


def test_risk_control_system():
    """测试风险控制系统"""
    logger.info("测试风险控制系统重建...")

    # 创建风险控制系统
    risk_system = ComprehensiveRiskControlSystem()

    # 启动系统
    if not risk_system.start():
        logger.error("风险控制系统启动失败")
        return

    # 创建模拟账户
    positions = {
        'AAPL': Position(
            symbol='AAPL',
            quantity=1000,
            average_price=150.0,
            current_price=155.0,
            market_value=155000,
            unrealized_pnl=5000
        ),
        'GOOGL': Position(
            symbol='GOOGL',
            quantity=100,
            average_price=2500.0,
            current_price=2550.0,
            market_value=255000,
            unrealized_pnl=5000
        )
    }

    account = Account(
        account_id="test_account",
        balance=100000,
        positions=positions,
        total_value=510000  # 100000 + 155000 + 255000
    )

    # 模拟市场数据
    market_data = {
        'returns': np.random.normal(0.001, 0.02, 100),  # 模拟日收益率
        'AAPL': {
            'volume': 50000000,
            'market_cap': 2500000000000  # 2.5万亿
        },
        'GOOGL': {
            'volume': 20000000,
            'market_cap': 1500000000000  # 1.5万亿
        }
    }

    # 1. 评估投资组合风险
    logger.info("\n1. 评估投资组合风险")
    assessment = risk_system.assess_portfolio_risk(account, market_data)

    logger.info("风险评估结果:")
    logger.info(f"  整体风险等级: {assessment['overall_risk_level']}")
    logger.info(f"  风险指标数量: {len(assessment['metrics'])}")
    logger.info(f"  触发告警数量: {len(assessment['alerts'])}")

    for metric in assessment['metrics']:
        logger.info(
            f"  指标: {metric['description']} - 值: {metric['value']:.3f}, 等级: {metric['risk_level']}")

    for alert in assessment['alerts']:
        logger.info(f"  告警: {alert['message']} (严重程度: {alert['severity']})")

    # 2. 获取系统状态
    logger.info("\n2. 获取系统状态")
    status = risk_system.get_system_status()
    logger.info("系统状态:")
    for key, value in status.items():
        if key != 'stats':
            logger.info(f"  {key}: {value}")

    stats = status['stats']
    logger.info("统计信息:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    # 3. 测试高风险情景
    logger.info("\n3. 测试高风险情景")

    # 创建高风险账户（单股票占比过高）
    high_risk_positions = {
        'AAPL': Position(
            symbol='AAPL',
            quantity=5000,  # 大幅增加持仓
            average_price=150.0,
            current_price=155.0,
            market_value=775000,
            unrealized_pnl=25000
        )
    }

    high_risk_account = Account(
        account_id="high_risk_account",
        balance=50000,
        positions=high_risk_positions,
        total_value=825000
    )

    high_risk_assessment = risk_system.assess_portfolio_risk(high_risk_account, market_data)

    logger.info("高风险情景评估:")
    logger.info(f"  整体风险等级: {high_risk_assessment['overall_risk_level']}")
    logger.info(f"  触发告警数量: {len(high_risk_assessment['alerts'])}")

    for alert in high_risk_assessment['alerts']:
        logger.info(f"  告警: {alert['message']} (严重程度: {alert['severity']})")

    # 停止系统
    risk_system.stop()

    logger.info("\n✅ 风险控制系统重建测试完成")


if __name__ == "__main__":
    test_risk_control_system()
