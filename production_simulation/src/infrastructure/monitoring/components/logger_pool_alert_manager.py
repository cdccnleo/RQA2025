"""
Logger池告警管理器组件

负责Logger池的告警检查和触发。
"""

import time
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime


try:
    from ...monitoring.logger_pool_monitor import LoggerPoolStats
except ImportError:
    # 如果没有导入成功，定义基础的数据类
    from dataclasses import dataclass
    
    @dataclass
    class LoggerPoolStats:
        pool_size: int
        max_size: int
        created_count: int
        hit_count: int
        hit_rate: float
        logger_count: int
        total_access_count: int
        avg_access_time: float
        memory_usage_mb: float
        timestamp: float


class LoggerPoolAlertManager:
    """Logger池告警管理器"""
    
    def __init__(self, pool_name: str = "default", 
                 alert_thresholds: Optional[Dict[str, Any]] = None):
        """初始化告警管理器"""
        self.pool_name = pool_name
        
        # 默认告警阈值
        self.alert_thresholds = alert_thresholds or {
            'hit_rate_low': 0.8,      # 命中率低于80%告警
            'pool_usage_high': 0.9,   # 池使用率高于90%告警
            'memory_high': 100.0,     # 内存使用高于100MB告警
        }
        
        # 告警回调
        self.alert_callbacks: List[Callable] = []
        self.alert_history: List[Dict[str, Any]] = []
        self.max_alert_history = 100
    
    def check_alerts(self, stats: LoggerPoolStats) -> List[Dict[str, Any]]:
        """检查告警条件"""
        if not stats:
            return []
        
        alerts = []
        
        # 命中率告警
        hit_rate_alert = self._check_hit_rate_alert(stats)
        if hit_rate_alert:
            alerts.append(hit_rate_alert)
        
        # 池使用率告警
        usage_alert = self._check_pool_usage_alert(stats)
        if usage_alert:
            alerts.append(usage_alert)
        
        # 内存使用告警
        memory_alert = self._check_memory_alert(stats)
        if memory_alert:
            alerts.append(memory_alert)
        
        # 触发告警回调
        for alert in alerts:
            self._trigger_alert(alert)
        
        return alerts
    
    def _check_hit_rate_alert(self, stats: LoggerPoolStats) -> Optional[Dict[str, Any]]:
        """检查命中率告警"""
        if stats.hit_rate < self.alert_thresholds['hit_rate_low']:
            return {
                'alert_type': 'hit_rate_low',
                'message': f"Logger池命中率过低: {stats.hit_rate:.2f} < {self.alert_thresholds['hit_rate_low']}",
                'severity': 'warning',
                'threshold': self.alert_thresholds['hit_rate_low'],
                'current_value': stats.hit_rate,
                'timestamp': datetime.now(),
                'pool_name': self.pool_name
            }
        return None
    
    def _check_pool_usage_alert(self, stats: LoggerPoolStats) -> Optional[Dict[str, Any]]:
        """检查池使用率告警"""
        usage_rate = stats.pool_size / stats.max_size if stats.max_size > 0 else 0
        if usage_rate > self.alert_thresholds['pool_usage_high']:
            return {
                'alert_type': 'pool_usage_high',
                'message': f"Logger池使用率过高: {usage_rate:.2f} > {self.alert_thresholds['pool_usage_high']}",
                'severity': 'warning',
                'threshold': self.alert_thresholds['pool_usage_high'],
                'current_value': usage_rate,
                'timestamp': datetime.now(),
                'pool_name': self.pool_name
            }
        return None
    
    def _check_memory_alert(self, stats: LoggerPoolStats) -> Optional[Dict[str, Any]]:
        """检查内存使用告警"""
        if stats.memory_usage_mb > self.alert_thresholds['memory_high']:
            return {
                'alert_type': 'memory_high',
                'message': f"Logger池内存使用过高: {stats.memory_usage_mb:.1f}MB > {self.alert_thresholds['memory_high']}MB",
                'severity': 'error',
                'threshold': self.alert_thresholds['memory_high'],
                'current_value': stats.memory_usage_mb,
                'timestamp': datetime.now(),
                'pool_name': self.pool_name
            }
        return None
    
    def _trigger_alert(self, alert_data: Dict[str, Any]) -> None:
        """触发告警"""
        try:
            # 记录告警历史
            self.alert_history.append(alert_data.copy())
            if len(self.alert_history) > self.max_alert_history:
                self.alert_history.pop(0)
            
            # 打印告警信息
            severity = alert_data.get('severity', 'unknown')
            message = alert_data.get('message', 'Unknown alert')
            print(f"🔥 Logger池告警 [{severity.upper()}]: {message}")
            
            # 调用注册的回调函数
            for callback in self.alert_callbacks:
                try:
                    callback(alert_data)
                except Exception as e:
                    print(f"告警回调执行失败: {e}")
                    
        except Exception as e:
            print(f"触发告警失败: {e}")
    
    def register_alert_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """注册告警回调函数"""
        self.alert_callbacks.append(callback)
    
    def unregister_alert_callback(self, callback: Callable[[Dict[str, Any]], None]) -> bool:
        """注销告警回调函数"""
        try:
            self.alert_callbacks.remove(callback)
            return True
        except ValueError:
            return False
    
    def update_threshold(self, alert_type: str, threshold: Any) -> bool:
        """更新告警阈值"""
        if alert_type in self.alert_thresholds:
            self.alert_thresholds[alert_type] = threshold
            return True
        return False
    
    def get_alert_status(self, stats: LoggerPoolStats) -> Dict[str, bool]:
        """获取告警状态"""
        if not stats:
            return {}
        
        usage_rate = stats.pool_size / stats.max_size if stats.max_size > 0 else 0
        
        return {
            'hit_rate_low': stats.hit_rate < self.alert_thresholds['hit_rate_low'],
            'pool_usage_high': usage_rate > self.alert_thresholds['pool_usage_high'],
            'memory_high': stats.memory_usage_mb > self.alert_thresholds['memory_high'],
        }
    
    def get_alert_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取告警历史"""
        if limit is None:
            return self.alert_history.copy()
        return self.alert_history[-limit:] if limit > 0 else []
    
    def clear_alert_history(self) -> None:
        """清空告警历史"""
        self.alert_history.clear()

