"""
智能告警系统
基于机器学习的异常检测、告警分级、多渠道告警
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """告警级别"""
    P0 = "P0"  # 紧急 - 立即处理
    P1 = "P1"  # 重要 - 1小时内处理
    P2 = "P2"  # 一般 - 24小时内处理
    P3 = "P3"  # 提示 - 记录即可


class AlertChannel(Enum):
    """告警渠道"""
    EMAIL = "email"
    SMS = "sms"
    WECHAT = "wechat"
    DINGTALK = "dingtalk"
    WEBHOOK = "webhook"
    CONSOLE = "console"


@dataclass
class Alert:
    """告警"""
    alert_id: str
    level: AlertLevel
    title: str
    message: str
    source: str  # 告警来源
    metric_name: str
    metric_value: float
    threshold: float
    timestamp: datetime
    channels: List[AlertChannel] = field(default_factory=list)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertRule:
    """告警规则"""
    rule_id: str
    name: str
    description: str
    metric_pattern: str  # 指标名称或正则表达式
    condition: str  # "gt", "lt", "eq", "between", "anomaly"
    threshold: float
    threshold_max: Optional[float] = None  # 用于between条件
    level: AlertLevel = AlertLevel.P2
    channels: List[AlertChannel] = field(default_factory=lambda: [AlertChannel.CONSOLE])
    enabled: bool = True
    cooldown_minutes: int = 30  # 冷却时间
    last_triggered: Optional[datetime] = None


class AnomalyDetector:
    """
    异常检测器
    
    基于统计方法和机器学习检测异常
    """
    
    def __init__(self, window_size: int = 100):
        """
        初始化异常检测器
        
        Args:
            window_size: 历史数据窗口大小
        """
        self.window_size = window_size
        self._history: Dict[str, List[float]] = {}
        self._thresholds: Dict[str, Dict] = {}
    
    def update(self, metric_name: str, value: float):
        """更新历史数据"""
        if metric_name not in self._history:
            self._history[metric_name] = []
        
        self._history[metric_name].append(value)
        
        # 限制历史数据大小
        if len(self._history[metric_name]) > self.window_size:
            self._history[metric_name] = self._history[metric_name][-self.window_size:]
        
        # 更新阈值
        self._update_thresholds(metric_name)
    
    def _update_thresholds(self, metric_name: str):
        """更新阈值"""
        data = self._history[metric_name]
        if len(data) < 10:
            return
        
        # 使用3-sigma原则
        mean = np.mean(data)
        std = np.std(data)
        
        self._thresholds[metric_name] = {
            'mean': mean,
            'std': std,
            'upper': mean + 3 * std,
            'lower': mean - 3 * std,
            'upper_warning': mean + 2 * std,
            'lower_warning': mean - 2 * std
        }
    
    def detect(self, metric_name: str, value: float) -> Optional[Dict]:
        """
        检测异常
        
        Returns:
            异常信息或None
        """
        if metric_name not in self._thresholds:
            return None
        
        thresholds = self._thresholds[metric_name]
        
        # 严重异常（3-sigma）
        if value > thresholds['upper']:
            return {
                'severity': 'critical',
                'type': 'upper_outlier',
                'value': value,
                'threshold': thresholds['upper'],
                'deviation': (value - thresholds['mean']) / thresholds['std']
            }
        
        if value < thresholds['lower']:
            return {
                'severity': 'critical',
                'type': 'lower_outlier',
                'value': value,
                'threshold': thresholds['lower'],
                'deviation': (thresholds['mean'] - value) / thresholds['std']
            }
        
        # 警告（2-sigma）
        if value > thresholds['upper_warning']:
            return {
                'severity': 'warning',
                'type': 'upper_warning',
                'value': value,
                'threshold': thresholds['upper_warning'],
                'deviation': (value - thresholds['mean']) / thresholds['std']
            }
        
        if value < thresholds['lower_warning']:
            return {
                'severity': 'warning',
                'type': 'lower_warning',
                'value': value,
                'threshold': thresholds['lower_warning'],
                'deviation': (thresholds['mean'] - value) / thresholds['std']
            }
        
        return None
    
    def get_baseline(self, metric_name: str) -> Optional[Dict]:
        """获取基线信息"""
        return self._thresholds.get(metric_name)


class AlertRouter:
    """告警路由器"""
    
    def __init__(self):
        self._handlers: Dict[AlertChannel, Callable] = {}
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """注册默认处理器"""
        self._handlers[AlertChannel.CONSOLE] = self._handle_console
        self._handlers[AlertChannel.EMAIL] = self._handle_email
        self._handlers[AlertChannel.SMS] = self._handle_sms
        self._handlers[AlertChannel.WEBHOOK] = self._handle_webhook
    
    def register_handler(self, channel: AlertChannel, handler: Callable):
        """注册处理器"""
        self._handlers[channel] = handler
    
    async def route(self, alert: Alert):
        """路由告警"""
        for channel in alert.channels:
            handler = self._handlers.get(channel)
            if handler:
                try:
                    await handler(alert)
                except Exception as e:
                    logger.error(f"处理告警失败 {channel}: {e}")
    
    async def _handle_console(self, alert: Alert):
        """控制台输出"""
        level_color = {
            AlertLevel.P0: "\033[91m",  # 红色
            AlertLevel.P1: "\033[93m",  # 黄色
            AlertLevel.P2: "\033[94m",  # 蓝色
            AlertLevel.P3: "\033[90m"   # 灰色
        }
        reset = "\033[0m"
        
        color = level_color.get(alert.level, "")
        print(f"{color}[{alert.level.value}] {alert.title}{reset}")
        print(f"  消息: {alert.message}")
        print(f"  来源: {alert.source}")
        print(f"  时间: {alert.timestamp}")
    
    async def _handle_email(self, alert: Alert):
        """邮件告警"""
        # 这里应该集成邮件发送服务
        logger.info(f"发送邮件告警: {alert.title}")
    
    async def _handle_sms(self, alert: Alert):
        """短信告警"""
        # 这里应该集成短信服务
        logger.info(f"发送短信告警: {alert.title}")
    
    async def _handle_webhook(self, alert: Alert):
        """Webhook回调"""
        # 这里应该发送HTTP请求
        logger.info(f"发送Webhook告警: {alert.title}")


class IntelligentAlertSystem:
    """
    智能告警系统
    
    职责：
    1. 基于机器学习的异常检测
    2. 动态阈值调整
    3. 告警降噪
    4. 告警分级
    5. 多渠道告警
    """
    
    def __init__(self):
        """初始化智能告警系统"""
        self.anomaly_detector = AnomalyDetector()
        self.alert_router = AlertRouter()
        self._rules: Dict[str, AlertRule] = {}
        self._alerts: List[Alert] = []
        self._alert_counter = 0
        
        # 初始化默认规则
        self._init_default_rules()
        
        logger.info("智能告警系统初始化完成")
    
    def _init_default_rules(self):
        """初始化默认告警规则"""
        default_rules = [
            AlertRule(
                rule_id="signal_low_score",
                name="信号评分过低",
                description="信号综合评分低于阈值",
                metric_pattern="signal_score",
                condition="lt",
                threshold=30,
                level=AlertLevel.P1,
                channels=[AlertChannel.CONSOLE, AlertChannel.EMAIL],
                cooldown_minutes=60
            ),
            AlertRule(
                rule_id="system_high_latency",
                name="系统延迟过高",
                description="API响应延迟超过阈值",
                metric_pattern="api_latency",
                condition="gt",
                threshold=1000,  # 1秒
                level=AlertLevel.P0,
                channels=[AlertChannel.CONSOLE, AlertChannel.SMS],
                cooldown_minutes=15
            ),
            AlertRule(
                rule_id="data_source_down",
                name="数据源异常",
                description="数据源连接失败",
                metric_pattern="data_source_status",
                condition="eq",
                threshold=0,  # 0表示异常
                level=AlertLevel.P0,
                channels=[AlertChannel.CONSOLE, AlertChannel.EMAIL, AlertChannel.SMS],
                cooldown_minutes=5
            ),
            AlertRule(
                rule_id="memory_high_usage",
                name="内存使用率过高",
                description="系统内存使用率超过阈值",
                metric_pattern="memory_usage",
                condition="gt",
                threshold=85,  # 85%
                level=AlertLevel.P1,
                channels=[AlertChannel.CONSOLE, AlertChannel.EMAIL],
                cooldown_minutes=30
            ),
            AlertRule(
                rule_id="disk_high_usage",
                name="磁盘使用率过高",
                description="磁盘使用率超过阈值",
                metric_pattern="disk_usage",
                condition="gt",
                threshold=90,  # 90%
                level=AlertLevel.P1,
                channels=[AlertChannel.CONSOLE, AlertChannel.EMAIL],
                cooldown_minutes=60
            )
        ]
        
        for rule in default_rules:
            self._rules[rule.rule_id] = rule
        
        logger.info(f"初始化 {len(default_rules)} 个默认告警规则")
    
    def add_rule(self, rule: AlertRule):
        """添加告警规则"""
        self._rules[rule.rule_id] = rule
        logger.info(f"添加告警规则: {rule.rule_id}")
    
    def remove_rule(self, rule_id: str):
        """移除告警规则"""
        if rule_id in self._rules:
            del self._rules[rule_id]
            logger.info(f"移除告警规则: {rule_id}")
    
    def enable_rule(self, rule_id: str):
        """启用告警规则"""
        if rule_id in self._rules:
            self._rules[rule_id].enabled = True
            logger.info(f"启用告警规则: {rule_id}")
    
    def disable_rule(self, rule_id: str):
        """禁用告警规则"""
        if rule_id in self._rules:
            self._rules[rule_id].enabled = False
            logger.info(f"禁用告警规则: {rule_id}")
    
    async def check_metric(self, metric_name: str, value: float, context: Dict = None):
        """
        检查指标
        
        Args:
            metric_name: 指标名称
            value: 指标值
            context: 上下文信息
        """
        context = context or {}
        
        # 更新异常检测器
        self.anomaly_detector.update(metric_name, value)
        
        # 检查规则
        for rule in self._rules.values():
            if not rule.enabled:
                continue
            
            # 检查指标匹配
            if not self._match_metric(rule.metric_pattern, metric_name):
                continue
            
            # 检查冷却时间
            if rule.last_triggered:
                cooldown = timedelta(minutes=rule.cooldown_minutes)
                if datetime.now() - rule.last_triggered < cooldown:
                    continue
            
            # 检查条件
            triggered = self._check_condition(rule.condition, value, rule.threshold, rule.threshold_max)
            
            if triggered:
                # 创建告警
                alert = self._create_alert(rule, metric_name, value, context)
                
                # 路由告警
                await self.alert_router.route(alert)
                
                # 保存告警
                self._alerts.append(alert)
                
                # 更新规则触发时间
                rule.last_triggered = datetime.now()
        
        # 异常检测
        anomaly = self.anomaly_detector.detect(metric_name, value)
        if anomaly and anomaly['severity'] == 'critical':
            # 创建异常告警
            alert = Alert(
                alert_id=self._generate_alert_id(),
                level=AlertLevel.P1,
                title=f"指标异常: {metric_name}",
                message=f"检测到严重异常，偏离度: {anomaly['deviation']:.2f}σ",
                source="anomaly_detector",
                metric_name=metric_name,
                metric_value=value,
                threshold=anomaly['threshold'],
                timestamp=datetime.now(),
                channels=[AlertChannel.CONSOLE, AlertChannel.EMAIL],
                context=context
            )
            
            await self.alert_router.route(alert)
            self._alerts.append(alert)
    
    def _match_metric(self, pattern: str, metric_name: str) -> bool:
        """匹配指标名称"""
        import re
        return bool(re.match(pattern, metric_name))
    
    def _check_condition(
        self,
        condition: str,
        value: float,
        threshold: float,
        threshold_max: Optional[float] = None
    ) -> bool:
        """检查条件"""
        if condition == "gt":
            return value > threshold
        elif condition == "lt":
            return value < threshold
        elif condition == "eq":
            return value == threshold
        elif condition == "between":
            return threshold <= value <= threshold_max
        elif condition == "anomaly":
            # 异常检测由专门的检测器处理
            return False
        return False
    
    def _create_alert(
        self,
        rule: AlertRule,
        metric_name: str,
        value: float,
        context: Dict
    ) -> Alert:
        """创建告警"""
        self._alert_counter += 1
        
        return Alert(
            alert_id=f"alert_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._alert_counter}",
            level=rule.level,
            title=rule.name,
            message=f"{rule.description}\n当前值: {value:.2f}, 阈值: {rule.threshold}",
            source=rule.rule_id,
            metric_name=metric_name,
            metric_value=value,
            threshold=rule.threshold,
            timestamp=datetime.now(),
            channels=rule.channels,
            context=context
        )
    
    def _generate_alert_id(self) -> str:
        """生成告警ID"""
        self._alert_counter += 1
        return f"anomaly_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._alert_counter}"
    
    def acknowledge_alert(self, alert_id: str, user: str):
        """确认告警"""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = user
                alert.acknowledged_at = datetime.now()
                logger.info(f"告警已确认: {alert_id} by {user}")
                return True
        return False
    
    def resolve_alert(self, alert_id: str):
        """解决告警"""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                logger.info(f"告警已解决: {alert_id}")
                return True
        return False
    
    def get_alerts(
        self,
        level: Optional[AlertLevel] = None,
        acknowledged: Optional[bool] = None,
        resolved: Optional[bool] = None,
        limit: int = 100
    ) -> List[Alert]:
        """获取告警列表"""
        alerts = self._alerts
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]
        
        if resolved is not None:
            alerts = [a for a in alerts if a.resolved == resolved]
        
        # 按时间倒序
        alerts.sort(key=lambda x: x.timestamp, reverse=True)
        
        return alerts[:limit]
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """获取告警统计"""
        total = len(self._alerts)
        unacknowledged = sum(1 for a in self._alerts if not a.acknowledged)
        unresolved = sum(1 for a in self._alerts if not a.resolved)
        
        level_counts = {}
        for level in AlertLevel:
            level_counts[level.value] = sum(1 for a in self._alerts if a.level == level)
        
        return {
            "total_alerts": total,
            "unacknowledged": unacknowledged,
            "unresolved": unresolved,
            "level_distribution": level_counts,
            "active_rules": sum(1 for r in self._rules.values() if r.enabled)
        }
    
    def get_rules(self) -> List[AlertRule]:
        """获取所有规则"""
        return list(self._rules.values())


# 单例实例
_alert_system: Optional[IntelligentAlertSystem] = None


def get_intelligent_alert_system() -> IntelligentAlertSystem:
    """获取智能告警系统实例"""
    global _alert_system
    if _alert_system is None:
        _alert_system = IntelligentAlertSystem()
    return _alert_system
