"""
告警规则管理器组件

负责管理告警规则的添加、删除和查询。
"""

import time
from datetime import datetime
from typing import Dict, Any, Optional, List

# 导入告警相关类型
try:
    from ..services.alert_service import AlertRule, AlertLevel, AlertChannel
except ImportError:
    try:
        from ..alert_system import AlertRule, AlertLevel, AlertChannel
    except ImportError:
        # 如果无法导入，定义基础类型
        from dataclasses import dataclass
        from enum import Enum
        
        class AlertLevel(Enum):
            INFO = "info"
            WARNING = "warning"
            ERROR = "error"
            CRITICAL = "critical"
        
        class AlertChannel(Enum):
            EMAIL = "email"
            SMS = "sms"
            WEBHOOK = "webhook"
            SLACK = "slack"
            WECHAT = "wechat"
            CONSOLE = "console"
        
        @dataclass
        class AlertRule:
            rule_id: str
            name: str
            description: str
            condition: Dict[str, Any]
            level: AlertLevel
            channels: List[AlertChannel]
            enabled: bool = True
            cooldown: int = 300
            metadata: Optional[Dict[str, Any]] = None
            created_at: Optional[datetime] = None
            updated_at: Optional[datetime] = None


class AlertRuleManager:
    """告警规则管理器"""
    
    def __init__(self):
        """初始化规则管理器"""
        self.rules: Dict[str, AlertRule] = {}
        self.rule_last_triggered: Dict[str, datetime] = {}
    
    def add_alert_rule(self, rule: AlertRule) -> bool:
        """添加告警规则"""
        try:
            self.rules[rule.rule_id] = rule
            return True
        except Exception as e:
            print(f"添加告警规则失败: {e}")
            return False
    
    def remove_alert_rule(self, rule_id: str) -> bool:
        """移除告警规则"""
        try:
            if rule_id in self.rules:
                del self.rules[rule_id]
                # 同时移除触发时间记录
                if rule_id in self.rule_last_triggered:
                    del self.rule_last_triggered[rule_id]
                return True
            return False
        except Exception as e:
            print(f"移除告警规则失败: {e}")
            return False
    
    def get_alert_rule(self, rule_id: str) -> Optional[AlertRule]:
        """获取告警规则"""
        return self.rules.get(rule_id)
    
    def get_all_rules(self) -> Dict[str, AlertRule]:
        """获取所有告警规则"""
        return self.rules.copy()
    
    def get_enabled_rules(self) -> Dict[str, AlertRule]:
        """获取启用的告警规则"""
        return {rule_id: rule for rule_id, rule in self.rules.items() if rule.enabled}
    
    def update_rule_last_triggered(self, rule_id: str, trigger_time: Optional[datetime] = None) -> None:
        """更新规则最后触发时间"""
        self.rule_last_triggered[rule_id] = trigger_time or datetime.now()
    
    def is_rule_in_cooldown(self, rule: AlertRule, current_time: Optional[datetime] = None) -> bool:
        """检查规则是否在冷却时间内"""
        if current_time is None:
            current_time = datetime.now()
        
        last_triggered = self.rule_last_triggered.get(rule.rule_id)
        if last_triggered is None:
            return False
        
        time_diff = (current_time - last_triggered).total_seconds()
        return time_diff < rule.cooldown
    
    def create_rule_from_template(self, template_name: str, config: Dict[str, Any]) -> Optional[AlertRule]:
        """从模板创建规则"""
        templates = self._get_alert_templates(config)
        template = self._validate_and_get_template(template_name, templates)
        if not template:
            return None
        
        rule_id = self._generate_rule_id(template_name, config)
        return self._build_alert_rule(rule_id, template, config)
    
    def _get_alert_templates(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """获取告警模板"""
        return {
            "performance_threshold": self._create_performance_template(config),
            "error_rate_monitor": self._create_error_rate_template(config),
            "security_alert": self._create_security_template(config)
        }
    
    def _create_performance_template(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """创建性能阈值模板"""
        return {
            "name": "性能阈值告警",
            "description": "监控系统性能指标阈值",
            "condition": {
                "operator": "gt",
                "field": config.get('metric', 'cpu_usage'),
                "value": config.get('threshold', 80)
            },
            "level": AlertLevel(config.get('level', 'warning')),
            "channels": [AlertChannel.CONSOLE, AlertChannel.EMAIL]
        }
    
    def _create_error_rate_template(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """创建错误率监控模板"""
        return {
            "name": "错误率监控",
            "description": "监控系统错误率",
            "condition": {
                "operator": "gt",
                "field": "error_rate",
                "value": config.get('threshold', 5)
            },
            "level": AlertLevel.ERROR,
            "channels": [AlertChannel.CONSOLE, AlertChannel.EMAIL, AlertChannel.SLACK]
        }
    
    def _create_security_template(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """创建安全告警模板"""
        return {
            "name": "安全告警",
            "description": "监控安全相关事件",
            "condition": {
                "operator": "eq",
                "field": "event_type",
                "value": config.get('event_type', 'unauthorized_access')
            },
            "level": AlertLevel.CRITICAL,
            "channels": [AlertChannel.CONSOLE, AlertChannel.EMAIL, AlertChannel.SLACK]
        }
    
    def _validate_and_get_template(self, template_name: str, templates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """验证并获取模板"""
        template = templates.get(template_name)
        if not template:
            print(f"模板不存在: {template_name}")
            return None
        return template
    
    def _generate_rule_id(self, template_name: str, config: Dict[str, Any]) -> str:
        """生成规则ID"""
        return config.get('rule_id', f"{template_name}_{int(time.time())}")
    
    def _build_alert_rule(self, rule_id: str, template: Dict[str, Any], config: Dict[str, Any]) -> AlertRule:
        """构建告警规则对象"""
        return AlertRule(
            rule_id=rule_id,
            name=config.get('name', template['name']),
            description=config.get('description', template['description']),
            condition=template['condition'],
            level=template['level'],
            channels=config.get('channels', template['channels']),
            cooldown=config.get('cooldown', 300)
        )
    
    def get_rules_count(self) -> int:
        """获取规则数量"""
        return len(self.rules)
    
    def clear_all_rules(self) -> None:
        """清空所有规则"""
        self.rules.clear()
        self.rule_last_triggered.clear()
