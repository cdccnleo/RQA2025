# Risk rule attributes will be defined in __init__
try:
    from src.infrastructure.monitoring.alert_system import IntelligentAlertSystem
except ImportError:
    # Fallback implementation
    class IntelligentAlertSystem:
        def __init__(self, *args, **kwargs):
            self.rules = {}
            self.alerts = {}
            self.indicators = []

try:
    from src.infrastructure.integration import get_data_adapter
except ImportError:
    def get_data_adapter():
        return None

try:
    from src.adapters import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
import queue
import threading
import time
import logging
import numpy as np
import os
import sys
try:
    import pandas as pd
except ImportError:
    pd = None
from ..models.risk_rule import RiskRule


name = None
threshold_low = None
threshold_medium = None
threshold_high = None
risk_type = None
description = None
unit = None

#!/usr/bin/env python3


"""
实时风险监控系统

构建实时风险监控、预警和响应系统
创建时间: 2025-08-24 10:13:48
"""


# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 使用统一基础设施集成


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# 在类定义外部处理基础设施集成
try:
    data_adapter = get_data_adapter()
    monitoring = data_adapter.get_monitoring()
    print("统一基础设施集成层导入成功")
except Exception as e:
    print(f"统一基础设施集成层导入失败 {e}")
    monitoring = None


class RiskType(Enum):

    """风险类型枚举"""
    MARKET_RISK = "market_risk"           # 市场风险
    LIQUIDITY_RISK = "liquidity_risk"     # 流动性风险
    CREDIT_RISK = "credit_risk"           # 信用风险
    OPERATIONAL_RISK = "operational_risk"  # 操作风险
    COMPLIANCE_RISK = "compliance_risk"   # 合规风险
    MODEL_RISK = "model_risk"             # 模型风险
    SYSTEM_RISK = "system_risk"           # 系统风险


class RiskLevel(Enum):

    """风险等级枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskStatus(Enum):

    """风险状态枚举"""
    NORMAL = "normal"           # 正常
    WARNING = "warning"         # 警告
    ALERT = "alert"            # 告警
    CRITICAL = "critical"      # 严重
    RESOLVED = "resolved"      # 已解决


@dataclass
class RiskIndicator:

    """风险指标"""
    name: str
    value: float
    threshold_low: float
    threshold_medium: float
    threshold_high: float
    risk_type: RiskType
    description: str
    unit: str = ""
    timestamp: Optional[datetime] = None

    def __post_init__(self):

        if self.timestamp is None:
            self.timestamp = datetime.now()

    def calculate_risk_level(self) -> RiskLevel:
        """计算风险等级"""
        abs_value = abs(self.value)

        if abs_value >= self.threshold_high:
            return RiskLevel.CRITICAL
        elif abs_value >= self.threshold_medium:
            return RiskLevel.HIGH
        elif abs_value >= self.threshold_low:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'value': self.value,
            'threshold_low': self.threshold_low,
            'threshold_medium': self.threshold_medium,
            'threshold_high': self.threshold_high,
            'risk_type': self.risk_type.value,
            'risk_level': self.calculate_risk_level().value,
            'description': self.description,
            'unit': self.unit,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class RiskAlert:

    """风险告警"""
    alert_id: str
    risk_indicator: RiskIndicator
    risk_level: RiskLevel
    message: str
    timestamp: datetime
    status: RiskStatus = RiskStatus.ALERT
    resolved_at: Optional[datetime] = None
    resolution: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'alert_id': self.alert_id,
            'indicator_name': self.risk_indicator.name,
            'risk_type': self.risk_indicator.risk_type.value,
            'risk_level': self.risk_level.value,
            'message': self.message,
            'value': self.risk_indicator.value,
            'threshold': self.risk_indicator.threshold_high,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.value,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'resolution': self.resolution
        }


class RiskRule:

    """风险规则"""

    def __init__(self,
                 rule_id: str,
                 name: str,
                 risk_type: RiskType,
                 condition: Callable[[Dict[str, Any]], bool],
                 alert_level: AlertLevel,
                 message_template: str,
                 enabled: bool = True):
        self.rule_id = rule_id
        self.name = name
        self.risk_type = risk_type
        self.condition = condition
        self.alert_level = alert_level
        self.message_template = message_template
        self.enabled = enabled

    def evaluate(self, context: Dict[str, Any]) -> Optional[RiskAlert]:
        """评估风险规则"""
        if not self.enabled:
            return None

        try:
            if self.condition(context):
                alert_id = f"{self.rule_id}_{int(time.time())}_{hash(str(context)) % 10000}"

                # 创建风险指标(从context中提取)
                indicator = RiskIndicator(
                    name=context.get('indicator_name', self.name),
                    value=context.get('value', 0.0),
                    threshold_low=context.get('threshold_low', 0.1),
                    threshold_medium=context.get('threshold_medium', 0.2),
                    threshold_high=context.get('threshold_high', 0.3),
                    risk_type=self.risk_type,
                    description=context.get('description', self.name),
                    unit=context.get('unit', ''),
                    timestamp=datetime.now()
                )

                # 风险等级映射将在后续版本中使用
                risk_level = indicator.calculate_risk_level()

                message = self.message_template.format(
                    indicator_name=indicator.name,
                    value=indicator.value,
                    threshold=indicator.threshold_high,
                    risk_level=risk_level.value
                )

                alert = RiskAlert(
                    alert_id=alert_id,
                    risk_indicator=indicator,
                    risk_level=risk_level,
                    message=message,
                    timestamp=datetime.now()
                )

                return alert

            return None

        except Exception as e:
            logger.error(f"风险规则评估失败 {self.rule_id}: {e}")
            return None


class RiskEngine:

    """风险引擎"""

    def __init__(self):
        self.indicators: Dict[str, RiskIndicator] = {}
        self.rules: Dict[str, RiskRule] = {}
        self.alerts: Dict[str, RiskAlert] = {}
        self.risk_history: List[Dict[str, Any]] = []
        self.is_monitoring = False

        # 指标计算函数
        self.indicator_functions = {
            'portfolio_volatility': self._calculate_portfolio_volatility,
            'max_drawdown': self._calculate_max_drawdown,
            'sharpe_ratio': self._calculate_sharpe_ratio,
            'var_95': self._calculate_var_95,
            'liquidity_ratio': self._calculate_liquidity_ratio,
            'concentration_risk': self._calculate_concentration_risk,
            'model_accuracy': self._calculate_model_accuracy,
            'system_latency': self._calculate_system_latency,
            'data_quality_score': self._calculate_data_quality_score
        }

        logger.info("风险引擎初始化完成")

    def register_indicator(self,
                           name: str,
                           threshold_low: float,
                           threshold_medium: float,
                           threshold_high: float,
                           risk_type: RiskType,
                           description: str,
                           unit: str = ""):
        """注册风险指标"""
        indicator = RiskIndicator(
            name=name,
            value=0.0,  # 初始值
            threshold_low=threshold_low,
            threshold_medium=threshold_medium,
            threshold_high=threshold_high,
            risk_type=risk_type,
            description=description,
            unit=unit
        )

        self.indicators[name] = indicator
        logger.info(f"风险指标已注册 {name}")

    def register_rule(self, rule: RiskRule):
        """注册风险规则"""
        self.rules[rule.rule_id] = rule
        logger.info(f"风险规则已注册 {rule.rule_id}")

    def update_indicator_value(self, name: str, value: float):
        """更新指标值"""
        if name in self.indicators:
            self.indicators[name].value = value
            self.indicators[name].timestamp = datetime.now()

            # 记录历史
            self.risk_history.append({
                'indicator': name,
                'value': value,
                'timestamp': datetime.now().isoformat(),
                'risk_level': self.indicators[name].calculate_risk_level().value
            })

            # 限制历史记录数量
            if len(self.risk_history) > 10000:
                self.risk_history = self.risk_history[-10000:]

            logger.debug(f"指标值已更新: {name} = {value}")

    def evaluate_risks(self) -> List[RiskAlert]:
        """评估所有风险"""
        alerts = []

        for rule in self.rules.values():
            if not rule.enabled:
                continue

            # 构建评估上下文
            context = {}
            for indicator_name, indicator in self.indicators.items():
                if indicator.risk_type == rule.risk_type:
                    context.update({
                        'indicator_name': indicator_name,
                        'value': indicator.value,
                        'threshold_low': indicator.threshold_low,
                        'threshold_medium': indicator.threshold_medium,
                        'threshold_high': indicator.threshold_high,
                        'description': indicator.description,
                        'unit': indicator.unit
                    })

                    # 添加相关指标的历史数据
                    recent_history = [
                        h for h in self.risk_history[-100:]
                        if h['indicator'] == indicator_name
                    ]
                    context['history'] = recent_history

            if context:
                alert = rule.evaluate(context)
                if alert:
                    alerts.append(alert)
                    self.alerts[alert.alert_id] = alert

        return alerts

    def calculate_all_indicators(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """计算所有风险指标"""
        results = {}

        for indicator_name, calc_function in self.indicator_functions.items():
            try:
                value = calc_function(market_data)
                results[indicator_name] = value

                # 更新指标值
                if indicator_name in self.indicators:
                    self.update_indicator_value(indicator_name, value)

            except Exception as e:
                logger.error(f"指标计算失败 {indicator_name}: {e}")
                results[indicator_name] = 0.0

        return results

    # 指标计算方法

    def _calculate_portfolio_volatility(self, data: Dict[str, Any]) -> float:
        """计算投资组合波动率"""
        returns = data.get('portfolio_returns', [])
        if len(returns) < 2:
            return 0.0
        return np.std(returns) * np.sqrt(252)  # 年化波动率

    def _calculate_max_drawdown(self, data: Dict[str, Any]) -> float:
        """计算最大回撤"""
        portfolio_values = data.get('portfolio_values', [])
        if len(portfolio_values) < 2:
            return 0.0

        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        return abs(np.min(drawdown))

    def _calculate_sharpe_ratio(self, data: Dict[str, Any]) -> float:
        """计算夏普比率"""
        returns = data.get('portfolio_returns', [])
        if len(returns) < 2:
            return 0.0

        avg_return = np.mean(returns)
        volatility = np.std(returns)
        risk_free_rate = data.get('risk_free_rate', 0.02)

        if volatility == 0:
            return 0.0

        return (avg_return - risk_free_rate) / volatility * np.sqrt(252)

    def _calculate_var_95(self, data: Dict[str, Any]) -> float:
        """计算VaR(95%)"""
        returns = data.get('portfolio_returns', [])
        if len(returns) < 30:
            return 0.0

        return abs(np.percentile(returns, 5))

    def _calculate_liquidity_ratio(self, data: Dict[str, Any]) -> float:
        """计算流动性比率"""
        volume = data.get('trading_volume', 0)
        position_size = data.get('position_size', 1)
        return volume / position_size if position_size > 0 else 0.0

    def _calculate_concentration_risk(self, data: Dict[str, Any]) -> float:
        """计算集中度风险"""
        positions = data.get('positions', [])
        if not positions:
            return 0.0

        # 计算最大头寸占比
        total_value = sum(abs(pos) for pos in positions)
        if total_value == 0:
            return 0.0

        max_position = max(abs(pos) for pos in positions)
        return max_position / total_value

    def _calculate_model_accuracy(self, data: Dict[str, Any]) -> float:
        """计算模型准确性"""
        predictions = data.get('predictions', [])
        actuals = data.get('actuals', [])

        if len(predictions) != len(actuals) or len(predictions) == 0:
            return 1.0  # 默认准确性

        # 简化的准确性计算(实际应该更复杂)
        errors = [abs(p - a) for p, a in zip(predictions, actuals)]
        return 1.0 - np.mean(errors)

    def _calculate_system_latency(self, data: Dict[str, Any]) -> float:
        """计算系统延迟"""
        response_times = data.get('response_times', [])
        if not response_times:
            return 0.0
        return np.mean(response_times)

    def _calculate_data_quality_score(self, data: Dict[str, Any]) -> float:
        """计算数据质量评分"""
        # 简化的数据质量评分
        missing_rate = data.get('missing_rate', 0.0)
        outlier_rate = data.get('outlier_rate', 0.0)

        # 数据质量得分 = 1 - (缺失率 + 异常率)
        return max(0.0, 1.0 - missing_rate - outlier_rate)

    def get_risk_summary(self) -> Dict[str, Any]:
        """获取风险摘要"""
        summary = {
            'total_indicators': len(self.indicators),
            'total_rules': len(self.rules),
            'active_alerts': len([a for a in self.alerts.values() if a.status != RiskStatus.RESOLVED]),
            'risk_distribution': {},
            'latest_indicators': {}
        }

        # 风险分布统计
        for alert in self.alerts.values():
            if alert.status != RiskStatus.RESOLVED:
                risk_type = alert.risk_indicator.risk_type.value
                if risk_type not in summary['risk_distribution']:
                    summary['risk_distribution'][risk_type] = 0
                summary['risk_distribution'][risk_type] += 1

        # 最新指标值
        for name, indicator in self.indicators.items():
            summary['latest_indicators'][name] = indicator.to_dict()

        return summary


class RealtimeRiskMonitor:
    """实时风险监控器"""

    def __init__(self):
        self.engine = RiskEngine()
        self.alert_system = IntelligentAlertSystem()
        self.is_monitoring = False
        self.monitoring_thread = None
        self.data_queue = queue.Queue()
        self.alert_handlers = []

        # 初始化默认风险指标
        self._initialize_default_indicators()

        # 初始化默认风险规则
        self._initialize_default_rules()

        logger.info("实时风险监控器初始化完成")

    def _initialize_default_indicators(self):
        """初始化默认风险指标"""
        indicators_config = [
            {
                'name': 'portfolio_volatility',
                'threshold_low': 0.15,
                'threshold_medium': 0.25,
                'threshold_high': 0.35,
                'risk_type': RiskType.MARKET_RISK,
                'description': '投资组合波动率',
                'unit': '%'
            },
            {
                'name': 'max_drawdown',
                'threshold_low': 0.05,
                'threshold_medium': 0.10,
                'threshold_high': 0.20,
                'risk_type': RiskType.MARKET_RISK,
                'description': '最大回撤',
                'unit': '%'
            },
            {
                'name': 'sharpe_ratio',
                'threshold_low': 0.5,
                'threshold_medium': 0.8,
                'threshold_high': 1.2,
                'risk_type': RiskType.MARKET_RISK,
                'description': '夏普比率',
                'unit': ''
            },
            {
                'name': 'var_95',
                'threshold_low': 0.02,
                'threshold_medium': 0.05,
                'threshold_high': 0.10,
                'risk_type': RiskType.MARKET_RISK,
                'description': 'VaR(95%)',
                'unit': '%'
            },
            {
                'name': 'liquidity_ratio',
                'threshold_low': 0.1,
                'threshold_medium': 0.05,
                'threshold_high': 0.02,
                'risk_type': RiskType.LIQUIDITY_RISK,
                'description': '流动性比率',
                'unit': ''
            },
            {
                'name': 'concentration_risk',
                'threshold_low': 0.1,
                'threshold_medium': 0.2,
                'threshold_high': 0.3,
                'risk_type': RiskType.MARKET_RISK,
                'description': '集中度风险',
                'unit': '%'
            },
            {
                'name': 'model_accuracy',
                'threshold_low': 0.8,
                'threshold_medium': 0.6,
                'threshold_high': 0.4,
                'risk_type': RiskType.MODEL_RISK,
                'description': '模型准确性',
                'unit': '%'
            },
            {
                'name': 'system_latency',
                'threshold_low': 1.0,
                'threshold_medium': 2.0,
                'threshold_high': 5.0,
                'risk_type': RiskType.SYSTEM_RISK,
                'description': '系统延迟',
                'unit': '秒'
            },
            {
                'name': 'data_quality_score',
                'threshold_low': 0.9,
                'threshold_medium': 0.7,
                'threshold_high': 0.5,
                'risk_type': RiskType.OPERATIONAL_RISK,
                'description': '数据质量评分',
                'unit': '%'
            }
        ]

        for config in indicators_config:
            self.engine.register_indicator(**config)

    def _initialize_default_rules(self):
        """初始化默认风险规则"""
        rules_config = [
            {
                'rule_id': 'high_volatility_alert',
                'name': '高波动率告警',
                'risk_type': RiskType.MARKET_RISK,
                'condition': lambda ctx: ctx.get('value', 0) > ctx.get('threshold_high', 0.35),
                'alert_level': AlertLevel.ERROR,
                'message_template': '{indicator_name} 超出阈误 {value:.2f} > {threshold:.2f} ({risk_level}风险)'
            },
            {
                'rule_id': 'drawdown_alert',
                'name': '回撤告警',
                'risk_type': RiskType.MARKET_RISK,
                'condition': lambda ctx: ctx.get('value', 0) > ctx.get('threshold_high', 0.20),
                'alert_level': AlertLevel.CRITICAL,
                'message_template': '{indicator_name} 严重超标: {value:.2%} > {threshold:.2%} ({risk_level}风险)'
            },
            {
                'rule_id': 'liquidity_alert',
                'name': '流动性告警',
                'risk_type': RiskType.LIQUIDITY_RISK,
                'condition': lambda ctx: ctx.get('value', 1.0) < ctx.get('threshold_high', 0.02),
                'alert_level': AlertLevel.WARNING,
                'message_template': '{indicator_name} 严重不足: {value:.3f} < {threshold:.3f} ({risk_level}风险)'
            },
            {
                'rule_id': 'model_accuracy_alert',
                'name': '模型准确性告警',
                'risk_type': RiskType.MODEL_RISK,
                'condition': lambda ctx: ctx.get('value', 1.0) < ctx.get('threshold_high', 0.4),
                'alert_level': AlertLevel.ERROR,
                'message_template': '{indicator_name} 严重下降: {value:.2%} < {threshold:.2%} ({risk_level}风险)'
            },
            {
                'rule_id': 'system_latency_alert',
                'name': '系统延迟告警',
                'risk_type': RiskType.SYSTEM_RISK,
                'condition': lambda ctx: ctx.get('value', 0) > ctx.get('threshold_high', 5.0),
                'alert_level': AlertLevel.WARNING,
                'message_template': '{indicator_name} 严重延迟: {value:.2f}s > {threshold:.2f}s ({risk_level}风险)'
            }
        ]

        for config in rules_config:
            rule = RiskRule(**config)
            self.engine.register_rule(rule)

    def start_monitoring(self, data_source: Callable[[], Dict[str, Any]]):
        """启动监控"""
        if self.is_monitoring:
            logger.warning("监控已启动")
            return

        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(data_source,)
        )
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

        logger.info("实时风险监控已启动")

    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("实时风险监控已停止")

    def _monitoring_loop(self, data_source: Callable[[], Dict[str, Any]]):
        """监控循环"""
        while self.is_monitoring:
            try:
                # 获取实时数据
                market_data = data_source()

                if market_data:
                    # 计算风险指标(用于内部状态更新)
                    self.engine.calculate_all_indicators(market_data)

                    # 评估风险规则
                    alerts = self.engine.evaluate_risks()

                    # 处理告警
                    for alert in alerts:
                        self._handle_alert(alert)

                logger.debug(f"监控周期完成，生成告警数量 {len(alerts)}")

                # 控制监控频率
                time.sleep(1)  # 每秒监控一次

            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                time.sleep(5)

    def _handle_alert(self, alert: RiskAlert):
        """处理告警"""
        logger.warning(f"风险告警: {alert.message}")

        # 转换为基础设施告警格式
        infrastructure_alert = {
            'id': alert.alert_id,
            'title': f"风险告警: {alert.risk_indicator.name}",
            'description': alert.message,
            'level': alert.risk_level.name,
            'source': 'realtime_risk_monitor',
            'timestamp': alert.timestamp.isoformat(),
            'data': alert.to_dict()
        }

        # 发送告警
        try:
            self.alert_system.process_alert(infrastructure_alert)
        except Exception as e:
            logger.error(f"告警处理失败: {e}")

        # 调用自定义告警处理器
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"告警处理器错误 {e}")

    def add_alert_handler(self, handler: Callable[[RiskAlert], None]):
        """添加告警处理器"""
        self.alert_handlers.append(handler)

    def get_monitoring_status(self) -> Dict[str, Any]:
        """获取监控状态"""
        return {
            'is_monitoring': self.is_monitoring,
            'risk_summary': self.engine.get_risk_summary(),
            'active_alerts': [
                alert.to_dict() for alert in self.engine.alerts.values()
                if alert.status != RiskStatus.RESOLVED
            ],
            'recent_indicators': [
                indicator.to_dict() for indicator in list(self.engine.indicators.values())[-10:]
            ]
        }

    def calculate_all_risks(self, portfolio: Dict[str, Dict[str, Any]], market_data: pd.DataFrame) -> Dict[RiskType, float]:
        """计算所有风险类型

        Args:
            portfolio: 投资组合数据
            market_data: 市场数据

        Returns:
            包含7种风险类型的字典，以RiskType枚举为键
        """
        risk_scores = {}

        # 计算各种风险并映射到RiskType枚举
        risk_scores[RiskType.MARKET_RISK] = self._calculate_market_risk_from_data(market_data)
        risk_scores[RiskType.LIQUIDITY_RISK] = self._calculate_liquidity_risk_from_data(portfolio, market_data)
        risk_scores[RiskType.CREDIT_RISK] = self._calculate_position_risk_from_data(portfolio)  # 使用头寸风险作为信用风险的代理
        risk_scores[RiskType.OPERATIONAL_RISK] = self._calculate_operational_risk_from_data({})
        risk_scores[RiskType.COMPLIANCE_RISK] = 0.1  # 简化的合规风险
        risk_scores[RiskType.MODEL_RISK] = self._calculate_volatility_risk_from_data(market_data) * 0.5  # 模型风险基于波动率
        risk_scores[RiskType.SYSTEM_RISK] = self._calculate_concentration_risk_from_data(portfolio) * 0.3  # 系统风险基于集中度

        return risk_scores

    def _calculate_position_risk_from_data(self, portfolio: Dict[str, Dict[str, Any]]) -> float:
        """基于投资组合数据计算头寸风险"""
        if not portfolio:
            return 0.0

        total_value = sum(pos.get('value', pos.get('quantity', 0) * pos.get('price', 0))
                         for pos in portfolio.values())
        if total_value == 0:
            return 0.0

        # 计算最大头寸占比
        max_position = max(pos.get('value', pos.get('quantity', 0) * pos.get('price', 0))
                          for pos in portfolio.values())
        return min(max_position / total_value, 1.0)

    def _calculate_volatility_risk_from_data(self, market_data: pd.DataFrame) -> float:
        """基于市场数据计算波动率风险"""
        if market_data.empty or 'price' not in market_data.columns:
            return 0.0

        prices = market_data['price'].values
        if len(prices) < 2:
            return 0.0

        # 计算收益率波动率
        returns = np.diff(np.log(prices))
        volatility = np.std(returns)
        return min(volatility * 100, 1.0)  # 转换为百分比并限制在0-1范围内

    def _calculate_liquidity_risk_from_data(self, portfolio: Dict[str, Dict[str, Any]], market_data: pd.DataFrame) -> float:
        """基于数据计算流动性风险"""
        if not portfolio:
            return 0.0

        # 简化的流动性风险计算：基于持仓规模和交易量
        total_position_size = sum(abs(pos.get('quantity', 0)) for pos in portfolio.values())

        # 假设市场数据包含交易量信息
        avg_volume = market_data.get('volume', pd.Series([1000000])).mean() if hasattr(market_data, 'get') else 1000000

        if avg_volume == 0:
            return 1.0

        liquidity_ratio = avg_volume / total_position_size if total_position_size > 0 else 1.0
        return max(0.0, min(1.0 - (liquidity_ratio / 100), 1.0))  # 流动性越差风险越高

    def _calculate_concentration_risk_from_data(self, portfolio: Dict[str, Dict[str, Any]]) -> float:
        """基于投资组合数据计算集中度风险"""
        if len(portfolio) < 2:
            return 0.0

        # 简化的集中度风险：基于持仓多样性
        num_positions = len(portfolio)
        diversity_factor = 1.0 / num_positions if num_positions > 0 else 1.0

        # 假设集中度风险与多样性成反比
        return 1.0 - diversity_factor

    def _calculate_correlation_risk_from_data(self, portfolio: Dict[str, Dict[str, Any]]) -> float:
        """基于投资组合数据计算相关性风险"""
        if len(portfolio) < 2:
            return 0.0

        # 简化的相关性风险：基于持仓多样性
        num_positions = len(portfolio)
        diversity_factor = 1.0 / num_positions if num_positions > 0 else 1.0

        # 假设相关性风险与集中度成反比
        return 1.0 - diversity_factor

    def _calculate_market_risk_from_data(self, market_data: pd.DataFrame) -> float:
        """基于市场数据计算市场风险"""
        if market_data.empty:
            return 0.5

        # 简化的市场风险：基于价格波动
        if 'price' in market_data.columns:
            prices = market_data['price'].values
            if len(prices) > 1:
                price_change = (prices[-1] - prices[0]) / prices[0]
                return min(abs(price_change), 1.0)

        return 0.3

    def _calculate_operational_risk_from_data(self, data: Dict[str, Any]) -> float:
        """基于数据计算操作风险"""
        # 简化的操作风险：基于系统状态
        system_status = data.get('system_status', 'normal')
        if system_status == 'error':
            return 1.0
        elif system_status == 'warning':
            return 0.5
        else:
            return 0.1

    def simulate_market_data(self) -> Dict[str, Any]:
        """模拟市场数据(用于演示)"""
        # 模拟投资组合数据
        np.random.seed(int(time.time()) % 10000)

        # 模拟收益率序列
        n_periods = 100
        base_returns = np.secrets.normal(0.001, 0.02, n_periods)
        portfolio_returns = base_returns + np.secrets.normal(0, 0.005, n_periods)

        # 计算投资组合价值
        portfolio_values = [1000000]  # 初始价值
        for ret in portfolio_returns:
            new_value = portfolio_values[-1] * (1 + ret)
            portfolio_values.append(new_value)

        # 模拟其他指标
        trading_volume = np.secrets.uniform(10000, 100000)
        position_size = np.secrets.uniform(100000, 500000)
        positions = np.secrets.uniform(-0.1, 0.1, 10) * portfolio_values[-1]

        # 模拟模型预测数据
        predictions = np.secrets.uniform(-0.02, 0.02, 20)
        actuals = predictions + np.secrets.normal(0, 0.005, 20)

        # 模拟系统性能数据
        response_times = np.secrets.uniform(0.1, 2.0, 50)

        # 模拟数据质量指标
        missing_rate = np.secrets.uniform(0, 0.1)
        outlier_rate = np.secrets.uniform(0, 0.05)

        return {
            'portfolio_returns': portfolio_returns,
            'portfolio_values': portfolio_values,
            'risk_free_rate': 0.02,
            'trading_volume': trading_volume,
            'position_size': position_size,
            'positions': positions.tolist(),
            'predictions': predictions.tolist(),
            'actuals': actuals.tolist(),
            'response_times': response_times.tolist(),
            'missing_rate': missing_rate,
            'outlier_rate': outlier_rate,
            'timestamp': datetime.now().isoformat()
        }


def create_default_risk_monitor() -> RealtimeRiskMonitor:
    """创建默认风险监控器"""
    monitor = RealtimeRiskMonitor()

    # 添加自定义告警处理器

    def custom_alert_handler(alert: RiskAlert):
        """自定义告警处理器"""
        print(f"🔔 自定义告警处理器: {alert.message}")

        # 这里可以添加更多处理逻辑，比如：
        # - 发送邮件通知
        # - 触发交易暂停
        # - 更新风险限额
        # - 记录到数据库

    monitor.add_alert_handler(custom_alert_handler)

    return monitor


def main():
    """主函数 - 实时风险监控演示"""
    print("🛡 RQA2025实时风险监控系统")
    print("="*50)

    # 创建风险监控器
    monitor = create_default_risk_monitor()

    print("风险监控器创建完成")
    print(f"   风险指标数量: {len(monitor.engine.indicators)}")
    print(f"   风险规则数量: {len(monitor.engine.rules)}")

    # 显示配置的风险指标
    print("\n🔍 配置的风险指标")
    for name, indicator in monitor.engine.indicators.items():
        print(f"   {name}: {indicator.description}")
    print(
        f"     阈值: {indicator.threshold_low}, {indicator.threshold_medium}, {indicator.threshold_high}")

    # 显示配置的风险规则
    print("\n⚖️ 配置的风险规则")
    for rule_id, rule in monitor.engine.rules.items():
        print(f"   {rule_id}: {rule.name}")

    try:
        # 启动监控
        print("\n📊 启动实时监控...")
        monitor.start_monitoring(monitor.simulate_market_data)

        # 运行一段时间
        print("   监控运行中.. (按Ctrl + C停止)")

        while True:
            time.sleep(5)

            # 每5秒显示一次状态摘要
            status = monitor.get_monitoring_status()
            risk_summary = status['risk_summary']

            print(f"\n📈 风险状态摘要[{datetime.now().strftime('%H:%M:%S')}]")
            print(f"   活跃告警: {risk_summary['active_alerts']}")
            print(f"   风险分布: {risk_summary['risk_distribution']}")

            # 显示最新的几个指标
            latest_indicators = status.get('recent_indicators', [])
            if latest_indicators:
                print("   最新指标")
                for indicator in latest_indicators[-3:]:
                    print(
                        f"     {indicator['name']}: {indicator['value']:.4f} ({indicator['risk_level']})")

    except KeyboardInterrupt:
        print("\n\n🛑 收到停止信号，正在停止监控..")
    except Exception as e:
        print(f"\n监控过程中出错 {e}")
    finally:
        # 停止监控
        monitor.stop_monitoring()
        print("风险监控已停止")

        # 显示最终状态
        final_status = monitor.get_monitoring_status()
        print("📋 最终监控状态")
        print(f"   监控运行: {'是' if final_status['is_monitoring'] else '否'}")
        print(f"   活跃告警: {len(final_status['active_alerts'])}")

        if final_status['active_alerts']:
            print("   活跃告警列表:")
            for alert in final_status['active_alerts'][:5]:  # 显示前5个
                print(f"     {alert['indicator_name']}: {alert['message']}")

        return monitor


if __name__ == "__main__":
    monitor = main()
