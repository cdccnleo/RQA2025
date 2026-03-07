"""
生产环境告警管理器组件

负责检查告警条件和发送通知。
"""

import time
from datetime import datetime
from typing import Dict, Any, List, Optional


class ProductionAlertManager:
    """生产环境告警管理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化告警管理器"""
        self.config = config or {
            'alert_thresholds': {
                'cpu_percent': 80.0,
                'memory_percent': 85.0,
                'disk_percent': 90.0,
                'response_time': 5000,  # 5秒
                'error_rate': 5.0
            },
            'alert_cooldown': 300  # 5分钟告警冷却
        }
        self.last_alert_times: Dict[str, float] = {}
    
    def check_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检查告警条件"""
        alerts = []
        thresholds = self.config['alert_thresholds']
        
        # CPU使用率告警
        cpu_alert = self._check_cpu_alert(metrics, thresholds)
        if cpu_alert:
            alerts.append(cpu_alert)
        
        # 内存使用率告警
        memory_alert = self._check_memory_alert(metrics, thresholds)
        if memory_alert:
            alerts.append(memory_alert)
        
        # 磁盘使用率告警
        disk_alert = self._check_disk_alert(metrics, thresholds)
        if disk_alert:
            alerts.append(disk_alert)
        
        return alerts
    
    def _check_cpu_alert(self, metrics: Dict[str, Any], thresholds: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """检查CPU告警"""
        cpu_percent = metrics.get('cpu', {}).get('percent', 0)
        threshold = thresholds['cpu_percent']
        
        if cpu_percent > threshold:
            return {
                'type': 'cpu_high',
                'level': 'warning',
                'message': f"CPU使用率过高: {cpu_percent:.1f}%",
                'value': cpu_percent,
                'threshold': threshold,
                'timestamp': datetime.now().isoformat()
            }
        return None
    
    def _check_memory_alert(self, metrics: Dict[str, Any], thresholds: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """检查内存告警"""
        memory_percent = metrics.get('memory', {}).get('percent', 0)
        threshold = thresholds['memory_percent']
        
        if memory_percent > threshold:
            return {
                'type': 'memory_high',
                'level': 'warning',
                'message': f"内存使用率过高: {memory_percent:.1f}%",
                'value': memory_percent,
                'threshold': threshold,
                'timestamp': datetime.now().isoformat()
            }
        return None
    
    def _check_disk_alert(self, metrics: Dict[str, Any], thresholds: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """检查磁盘告警"""
        disk_percent = metrics.get('disk', {}).get('percent', 0)
        threshold = thresholds['disk_percent']
        
        if disk_percent > threshold:
            return {
                'type': 'disk_high',
                'level': 'error',
                'message': f"磁盘使用率过高: {disk_percent:.1f}%",
                'value': disk_percent,
                'threshold': threshold,
                'timestamp': datetime.now().isoformat()
            }
        return None
    
    def send_alerts(self, alerts: List[Dict[str, Any]]) -> List[str]:
        """发送告警通知"""
        sent_alerts = []
        
        for alert in alerts:
            if self._should_send_alert(alert):
                self._send_alert_notification(alert)
                self._update_alert_cooldown(alert)
                sent_alerts.append(alert['type'])
        
        return sent_alerts
    
    def _should_send_alert(self, alert: Dict[str, Any]) -> bool:
        """判断是否应该发送告警（考虑冷却时间）"""
        alert_key = f"{alert['type']}_{alert['level']}"
        current_time = time.time()
        last_alert_time = self.last_alert_times.get(alert_key, 0)
        
        return current_time - last_alert_time > self.config['alert_cooldown']
    
    def _update_alert_cooldown(self, alert: Dict[str, Any]) -> None:
        """更新告警冷却时间"""
        alert_key = f"{alert['type']}_{alert['level']}"
        self.last_alert_times[alert_key] = time.time()
    
    def _send_alert_notification(self, alert: Dict[str, Any]) -> None:
        """发送告警通知"""
        try:
            # 控制台输出
            print(f"🚨 告警通知: [{alert['level'].upper()}] {alert['message']}")
            
            # 记录到日志文件
            self._log_alert_to_file(alert)
            
        except Exception as e:
            print(f"❌ 发送告警通知失败: {e}")
    
    def _log_alert_to_file(self, alert: Dict[str, Any]) -> None:
        """记录告警到日志文件"""
        try:
            with open('alerts.log', 'a', encoding='utf-8') as f:
                timestamp = alert.get('timestamp', datetime.now().isoformat())
                f.write(f"{timestamp} | {alert['level']} | {alert['message']}\n")
        except Exception as e:
            print(f"❌ 写入告警日志失败: {e}")
    
    def update_threshold(self, metric_type: str, threshold: float) -> bool:
        """更新告警阈值"""
        if metric_type in self.config['alert_thresholds']:
            self.config['alert_thresholds'][metric_type] = threshold
            return True
        return False
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """获取告警统计"""
        return {
            'thresholds': self.config['alert_thresholds'],
            'cooldown_seconds': self.config['alert_cooldown'],
            'active_cooldowns': len(self.last_alert_times)
        }
    
    def clear_alert_cooldowns(self) -> None:
        """清空告警冷却时间"""
        self.last_alert_times.clear()

