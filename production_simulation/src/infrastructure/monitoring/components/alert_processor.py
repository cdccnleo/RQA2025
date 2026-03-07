"""
告警处理器组件

负责创建告警和处理告警队列。
"""

import queue
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List

# 导入告警相关类型
try:
    from ..services.alert_service import Alert, AlertRule, AlertStatus
except ImportError:
    try:
        from ..alert_system import Alert, AlertRule, AlertStatus
    except ImportError:
        # 如果无法导入，定义基础类型
        from dataclasses import dataclass
        from enum import Enum
        
        class AlertStatus(Enum):
            ACTIVE = "active"
            RESOLVED = "resolved"
            ACKNOWLEDGED = "acknowledged"
            SUPPRESSED = "suppressed"
        
        @dataclass
        class AlertRule:
            rule_id: str
            name: str
            description: str
            condition: Dict[str, Any]
            level: Any
            enabled: bool = True
            cooldown: int = 300
            metadata: Optional[Dict[str, Any]] = None
        
        @dataclass
        class Alert:
            alert_id: str
            rule_id: str
            title: str
            message: str
            level: Any
            status: AlertStatus = AlertStatus.ACTIVE
            data: Optional[Dict[str, Any]] = None
            created_at: Optional[datetime] = None
            source: str = "system"
            resolved_at: Optional[datetime] = None
            acknowledged_at: Optional[datetime] = None
            acknowledged_by: Optional[str] = None


class AlertProcessor:
    """告警处理器"""
    
    def __init__(self):
        """初始化告警处理器"""
        self.alerts: Dict[str, Alert] = {}
        self.alert_queue = queue.Queue()
        self.alert_counter = 0
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None
    
    def validate_rule_condition(self, condition: str) -> bool:
        """
        验证规则条件（兼容性方法）
        
        Args:
            condition: 条件表达式
            
        Returns:
            条件是否有效
        """
        if not condition or not isinstance(condition, str):
            return False
        # 简单的语法检查
        try:
            compile(condition, '<string>', 'eval')
            return True
        except SyntaxError:
            return False
        
    def start_processing(self) -> None:
        """启动告警处理线程"""
        if self.running:
            return
            
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_alerts, daemon=True)
        self.worker_thread.start()
    
    def stop_processing(self) -> None:
        """停止告警处理线程"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
    
    def create_alert(self, rule: AlertRule, data: Dict[str, Any], source: str = "system") -> Optional[Alert]:
        """创建告警"""
        try:
            self.alert_counter += 1
            alert_id = f"alert_{self.alert_counter:06d}"

            alert = Alert(
                alert_id=alert_id,
                rule_id=rule.rule_id,
                title=rule.name,
                message=rule.description,
                level=rule.level,
                status=AlertStatus.ACTIVE,
                data=data,
                created_at=datetime.now(),
                source=source
            )

            self.alerts[alert_id] = alert
            return alert

        except Exception as e:
            print(f"告警创建失败: {e}")
            return None
    
    def queue_alert_for_processing(self, alert: Alert) -> None:
        """将告警加入处理队列"""
        try:
            self.alert_queue.put(alert)
        except Exception as e:
            print(f"告警加入队列失败: {e}")
    
    def _process_alerts(self) -> None:
        """处理告警队列"""
        while self.running:
            try:
                alert = self.alert_queue.get(timeout=1)
                # 这里可以添加具体的告警处理逻辑
                # 比如发送通知、记录日志等
                print(f"处理告警: {alert.alert_id} - {alert.title}")
                
                # 标记队列任务完成
                self.alert_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"告警处理失败: {e}")
    
    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """获取告警"""
        return self.alerts.get(alert_id)
    
    def get_all_alerts(self) -> Dict[str, Alert]:
        """获取所有告警"""
        return self.alerts.copy()
    
    def get_active_alerts(self) -> Dict[str, Alert]:
        """获取活跃告警"""
        return {alert_id: alert for alert_id, alert in self.alerts.items() 
                if alert.status == AlertStatus.ACTIVE}
    
    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """确认告警"""
        alert = self.alerts.get(alert_id)
        if alert and alert.status == AlertStatus.ACTIVE:
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now()
            alert.acknowledged_by = user
            return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """解决告警"""
        alert = self.alerts.get(alert_id)
        if alert and alert.status in [AlertStatus.ACTIVE, AlertStatus.ACKNOWLEDGED]:
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()
            return True
        return False
    
    def get_alerts_by_status(self, status: AlertStatus) -> Dict[str, Alert]:
        """根据状态获取告警"""
        return {alert_id: alert for alert_id, alert in self.alerts.items() 
                if alert.status == status}
    
    def get_alerts_by_rule(self, rule_id: str) -> Dict[str, Alert]:
        """根据规则ID获取告警"""
        return {alert_id: alert for alert_id, alert in self.alerts.items() 
                if alert.rule_id == rule_id}
    
    def clear_resolved_alerts(self, older_than_hours: int = 24) -> int:
        """清理已解决的告警"""
        from datetime import timedelta
        
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        alerts_to_remove = []
        
        for alert_id, alert in self.alerts.items():
            if (alert.status == AlertStatus.RESOLVED and 
                alert.resolved_at and 
                alert.resolved_at < cutoff_time):
                alerts_to_remove.append(alert_id)
        
        for alert_id in alerts_to_remove:
            del self.alerts[alert_id]
        
        return len(alerts_to_remove)
    
    def get_alerts_statistics(self) -> Dict[str, Any]:
        """获取告警统计信息"""
        total_alerts = len(self.alerts)
        active_alerts = len(self.get_active_alerts())
        resolved_alerts = len(self.get_alerts_by_status(AlertStatus.RESOLVED))
        acknowledged_alerts = len(self.get_alerts_by_status(AlertStatus.ACKNOWLEDGED))
        
        return {
            'total_alerts': total_alerts,
            'active_alerts': active_alerts,
            'resolved_alerts': resolved_alerts,
            'acknowledged_alerts': acknowledged_alerts,
            'queue_size': self.alert_queue.qsize()
        }
