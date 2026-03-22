"""
告警通知服务

提供多种告警通知渠道，包括邮件、微信、Slack等。
支持告警聚合、静默期管理和通知模板。
"""

import os
import time
import logging
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """告警严重级别"""
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    CRITICAL = 'critical'


class AlertChannel(Enum):
    """告警通知渠道"""
    EMAIL = 'email'
    WECHAT = 'wechat'
    SLACK = 'slack'
    WEBHOOK = 'webhook'
    LOG = 'log'


@dataclass
class Alert:
    """告警数据类"""
    alert_id: str
    alert_type: str
    severity: AlertSeverity
    title: str
    message: str
    source: str
    details: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[float] = None
    silenced: bool = False
    silence_until: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type,
            'severity': self.severity.value,
            'title': self.title,
            'message': self.message,
            'source': self.source,
            'details': self.details,
            'created_at': datetime.fromtimestamp(self.created_at).isoformat(),
            'acknowledged': self.acknowledged,
            'acknowledged_by': self.acknowledged_by,
            'silenced': self.silenced
        }


@dataclass
class AlertRule:
    """告警规则"""
    rule_id: str
    name: str
    alert_type: str
    condition: Callable[[Dict[str, Any]], bool]
    severity: AlertSeverity
    channels: List[AlertChannel]
    cooldown_minutes: int = 5
    enabled: bool = True
    template: str = "告警: {title}\n\n详情: {message}\n\n来源: {source}"


class AlertNotificationService:
    """
    告警通知服务
    
    提供多渠道告警通知能力，支持告警聚合和静默期管理。
    """
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls, *args, **kwargs):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        email_config: Optional[Dict[str, Any]] = None,
        wechat_config: Optional[Dict[str, Any]] = None,
        slack_config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化告警通知服务
        
        Args:
            email_config: 邮件配置
            wechat_config: 微信配置
            slack_config: Slack配置
        """
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.email_config = email_config or self._get_default_email_config()
        self.wechat_config = wechat_config or {}
        self.slack_config = slack_config or {}
        
        self._alerts_lock = threading.RLock()
        self._alerts: List[Alert] = []
        self._silence_rules: Dict[str, float] = {}
        self._last_notification: Dict[str, float] = {}
        
        self._rules: Dict[str, AlertRule] = {}
        self._register_default_rules()
        
        self._initialized = True
        logger.info("告警通知服务初始化完成")
    
    def _get_default_email_config(self) -> Dict[str, Any]:
        """获取默认邮件配置"""
        return {
            'smtp_host': os.getenv('SMTP_HOST', 'smtp.example.com'),
            'smtp_port': int(os.getenv('SMTP_PORT', '587')),
            'smtp_user': os.getenv('SMTP_USER', ''),
            'smtp_password': os.getenv('SMTP_PASSWORD', ''),
            'from_addr': os.getenv('SMTP_FROM', 'alerts@rqa2025.com'),
            'to_addrs': os.getenv('SMTP_TO', '').split(',') if os.getenv('SMTP_TO') else [],
            'use_tls': os.getenv('SMTP_TLS', 'true').lower() == 'true'
        }
    
    def _register_default_rules(self):
        """注册默认告警规则"""
        default_rules = [
            AlertRule(
                rule_id='collection_failure',
                name='数据采集失败告警',
                alert_type='collection_failure',
                condition=lambda d: d.get('consecutive_failures', 0) >= 3,
                severity=AlertSeverity.HIGH,
                channels=[AlertChannel.EMAIL, AlertChannel.LOG],
                cooldown_minutes=5
            ),
            AlertRule(
                rule_id='low_success_rate',
                name='采集成功率低告警',
                alert_type='low_success_rate',
                condition=lambda d: d.get('success_rate', 100) < 80,
                severity=AlertSeverity.MEDIUM,
                channels=[AlertChannel.EMAIL, AlertChannel.LOG],
                cooldown_minutes=10
            ),
            AlertRule(
                rule_id='api_connection_failure',
                name='API连接失败告警',
                alert_type='api_connection_failure',
                condition=lambda d: not d.get('api_accessible', True),
                severity=AlertSeverity.HIGH,
                channels=[AlertChannel.EMAIL, AlertChannel.LOG],
                cooldown_minutes=5
            ),
            AlertRule(
                rule_id='data_delay',
                name='数据延迟告警',
                alert_type='data_delay',
                condition=lambda d: d.get('delay_minutes', 0) > 60,
                severity=AlertSeverity.MEDIUM,
                channels=[AlertChannel.LOG],
                cooldown_minutes=15
            ),
            AlertRule(
                rule_id='storage_space',
                name='存储空间告警',
                alert_type='storage_space',
                condition=lambda d: d.get('disk_usage', 0) > 80,
                severity=AlertSeverity.LOW,
                channels=[AlertChannel.LOG],
                cooldown_minutes=30
            )
        ]
        
        for rule in default_rules:
            self._rules[rule.rule_id] = rule
    
    def create_alert(
        self,
        alert_type: str,
        severity: AlertSeverity,
        title: str,
        message: str,
        source: str,
        details: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """
        创建告警
        
        Args:
            alert_type: 告警类型
            severity: 严重级别
            title: 告警标题
            message: 告警消息
            source: 告警来源
            details: 详细信息
            
        Returns:
            告警对象
        """
        alert_id = f"{alert_type}_{int(time.time() * 1000)}"
        
        alert = Alert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            source=source,
            details=details or {}
        )
        
        with self._alerts_lock:
            self._alerts.append(alert)
            if len(self._alerts) > 1000:
                self._alerts = self._alerts[-500:]
        
        logger.info(f"创建告警: [{severity.value}] {title}")
        
        return alert
    
    def send_alert(
        self,
        alert: Alert,
        channels: Optional[List[AlertChannel]] = None
    ) -> Dict[str, bool]:
        """
        发送告警通知
        
        Args:
            alert: 告警对象
            channels: 通知渠道列表
            
        Returns:
            各渠道发送结果
        """
        if channels is None:
            channels = [AlertChannel.LOG]
        
        if alert.alert_type in self._silence_rules:
            if time.time() < self._silence_rules[alert.alert_type]:
                logger.info(f"告警 {alert.alert_id} 在静默期内，跳过通知")
                return {'silenced': True}
        
        cooldown_key = f"{alert.alert_type}_{alert.source}"
        if cooldown_key in self._last_notification:
            elapsed = time.time() - self._last_notification[cooldown_key]
            rule = self._rules.get(alert.alert_type)
            if rule and elapsed < rule.cooldown_minutes * 60:
                logger.debug(f"告警 {alert.alert_id} 在冷却期内，跳过通知")
                return {'cooldown': True}
        
        results = {}
        
        for channel in channels:
            try:
                if channel == AlertChannel.EMAIL:
                    results['email'] = self._send_email(alert)
                elif channel == AlertChannel.WECHAT:
                    results['wechat'] = self._send_wechat(alert)
                elif channel == AlertChannel.SLACK:
                    results['slack'] = self._send_slack(alert)
                elif channel == AlertChannel.LOG:
                    results['log'] = self._send_log(alert)
                elif channel == AlertChannel.WEBHOOK:
                    results['webhook'] = self._send_webhook(alert)
            except Exception as e:
                logger.error(f"发送告警到 {channel.value} 失败: {e}")
                results[channel.value] = False
        
        self._last_notification[cooldown_key] = time.time()
        
        return results
    
    def _send_email(self, alert: Alert) -> bool:
        """
        发送邮件通知
        
        Args:
            alert: 告警对象
            
        Returns:
            是否发送成功
        """
        config = self.email_config
        
        if not config.get('smtp_user') or not config.get('to_addrs'):
            logger.warning("邮件配置不完整，跳过邮件通知")
            return False
        
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            msg['From'] = config['from_addr']
            msg['To'] = ', '.join(config['to_addrs'])
            
            text_content = f"""
告警通知

严重级别: {alert.severity.value.upper()}
告警类型: {alert.alert_type}
告警来源: {alert.source}
告警时间: {datetime.fromtimestamp(alert.created_at).strftime('%Y-%m-%d %H:%M:%S')}

标题: {alert.title}

详情:
{alert.message}

详细信息:
{json.dumps(alert.details, indent=2, ensure_ascii=False)}

---
RQA2025 量化交易系统
            """
            
            msg.attach(MIMEText(text_content, 'plain', 'utf-8'))
            
            with smtplib.SMTP(config['smtp_host'], config['smtp_port']) as server:
                if config.get('use_tls'):
                    server.starttls()
                if config.get('smtp_user'):
                    server.login(config['smtp_user'], config['smtp_password'])
                server.sendmail(
                    config['from_addr'],
                    config['to_addrs'],
                    msg.as_string()
                )
            
            logger.info(f"✅ 邮件告警发送成功: {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 邮件发送失败: {e}")
            return False
    
    def _send_wechat(self, alert: Alert) -> bool:
        """
        发送微信通知
        
        Args:
            alert: 告警对象
            
        Returns:
            是否发送成功
        """
        config = self.wechat_config
        
        if not config.get('webhook_url'):
            logger.debug("微信 Webhook 未配置，跳过微信通知")
            return False
        
        try:
            import requests
            
            content = f"""
【{alert.severity.value.upper()}】{alert.title}

类型: {alert.alert_type}
来源: {alert.source}
时间: {datetime.fromtimestamp(alert.created_at).strftime('%H:%M:%S')}

{alert.message}
            """.strip()
            
            payload = {
                "msgtype": "text",
                "text": {
                    "content": content
                }
            }
            
            response = requests.post(
                config['webhook_url'],
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"✅ 微信告警发送成功: {alert.alert_id}")
                return True
            else:
                logger.error(f"❌ 微信告警发送失败: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ 微信通知发送失败: {e}")
            return False
    
    def _send_slack(self, alert: Alert) -> bool:
        """
        发送 Slack 通知
        
        Args:
            alert: 告警对象
            
        Returns:
            是否发送成功
        """
        config = self.slack_config
        
        if not config.get('webhook_url'):
            logger.debug("Slack Webhook 未配置，跳过 Slack 通知")
            return False
        
        try:
            import requests
            
            color_map = {
                AlertSeverity.LOW: '#36a64f',
                AlertSeverity.MEDIUM: '#ff9900',
                AlertSeverity.HIGH: '#ff6600',
                AlertSeverity.CRITICAL: '#ff0000'
            }
            
            payload = {
                "attachments": [{
                    "color": color_map.get(alert.severity, '#808080'),
                    "title": alert.title,
                    "text": alert.message,
                    "fields": [
                        {"title": "严重级别", "value": alert.severity.value.upper(), "short": True},
                        {"title": "来源", "value": alert.source, "short": True},
                        {"title": "时间", "value": datetime.fromtimestamp(alert.created_at).strftime('%Y-%m-%d %H:%M:%S'), "short": True}
                    ]
                }]
            }
            
            response = requests.post(
                config['webhook_url'],
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Slack 告警发送成功: {alert.alert_id}")
                return True
            else:
                logger.error(f"❌ Slack 告警发送失败: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Slack 通知发送失败: {e}")
            return False
    
    def _send_log(self, alert: Alert) -> bool:
        """
        记录日志告警
        
        Args:
            alert: 告警对象
            
        Returns:
            是否记录成功
        """
        log_methods = {
            AlertSeverity.LOW: logger.info,
            AlertSeverity.MEDIUM: logger.warning,
            AlertSeverity.HIGH: logger.warning,
            AlertSeverity.CRITICAL: logger.error
        }
        
        log_method = log_methods.get(alert.severity, logger.info)
        
        log_message = (
            f"🚨 告警 [{alert.severity.value.upper()}] {alert.title}\n"
            f"   类型: {alert.alert_type}\n"
            f"   来源: {alert.source}\n"
            f"   消息: {alert.message}\n"
            f"   详情: {json.dumps(alert.details, ensure_ascii=False)}"
        )
        
        log_method(log_message)
        return True
    
    def _send_webhook(self, alert: Alert) -> bool:
        """
        发送 Webhook 通知
        
        Args:
            alert: 告警对象
            
        Returns:
            是否发送成功
        """
        webhook_url = os.getenv('ALERT_WEBHOOK_URL')
        
        if not webhook_url:
            logger.debug("Webhook URL 未配置，跳过 Webhook 通知")
            return False
        
        try:
            import requests
            
            payload = alert.to_dict()
            
            response = requests.post(
                webhook_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Webhook 告警发送成功: {alert.alert_id}")
                return True
            else:
                logger.error(f"❌ Webhook 告警发送失败: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Webhook 通知发送失败: {e}")
            return False
    
    def set_silence(self, alert_type: str, duration_minutes: int):
        """
        设置告警静默期
        
        Args:
            alert_type: 告警类型
            duration_minutes: 静默时长（分钟）
        """
        self._silence_rules[alert_type] = time.time() + duration_minutes * 60
        logger.info(f"设置告警静默: {alert_type}, 时长: {duration_minutes} 分钟")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """
        确认告警
        
        Args:
            alert_id: 告警ID
            acknowledged_by: 确认人
            
        Returns:
            是否确认成功
        """
        with self._alerts_lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    alert.acknowledged_by = acknowledged_by
                    alert.acknowledged_at = time.time()
                    logger.info(f"告警 {alert_id} 已被 {acknowledged_by} 确认")
                    return True
        return False
    
    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        获取活跃告警列表
        
        Args:
            severity: 严重级别过滤
            limit: 返回数量限制
            
        Returns:
            告警列表
        """
        with self._alerts_lock:
            alerts = self._alerts[-limit:]
            
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            
            return [a.to_dict() for a in alerts if not a.acknowledged]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """
        获取告警统计
        
        Returns:
            告警统计数据
        """
        with self._alerts_lock:
            total = len(self._alerts)
            by_severity = defaultdict(int)
            by_type = defaultdict(int)
            acknowledged = 0
            
            for alert in self._alerts:
                by_severity[alert.severity.value] += 1
                by_type[alert.alert_type] += 1
                if alert.acknowledged:
                    acknowledged += 1
            
            return {
                'total_alerts': total,
                'acknowledged': acknowledged,
                'unacknowledged': total - acknowledged,
                'by_severity': dict(by_severity),
                'by_type': dict(by_type),
                'silence_rules': len(self._silence_rules)
            }


_alert_service_instance = None
_alert_service_lock = threading.RLock()


def get_alert_notification_service() -> AlertNotificationService:
    """
    获取告警通知服务实例（单例模式）
    
    Returns:
        AlertNotificationService 实例
    """
    global _alert_service_instance
    if _alert_service_instance is None:
        with _alert_service_lock:
            if _alert_service_instance is None:
                _alert_service_instance = AlertNotificationService()
    return _alert_service_instance
