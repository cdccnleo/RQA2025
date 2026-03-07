"""
生产环境数据管理器组件

负责存储和管理监控指标和告警数据。
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional


class ProductionDataManager:
    """生产环境数据管理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化数据管理器"""
        self.config = config or {
            'retention_period': 3600,  # 1小时数据保留
            'max_metrics_history': 1000,  # 最大指标历史数量
            'max_alerts_history': 500  # 最大告警历史数量
        }
        
        self.metrics_history: List[Dict[str, Any]] = []
        self.alerts_history: List[Dict[str, Any]] = []
    
    def store_metrics(self, metrics: Dict[str, Any]) -> None:
        """存储指标数据"""
        try:
            self.metrics_history.append(metrics.copy())
            
            # 限制历史数据大小
            max_history = self.config.get('max_metrics_history', 1000)
            if len(self.metrics_history) > max_history:
                self.metrics_history = self.metrics_history[-max_history:]
                
        except Exception as e:
            print(f"❌ 存储指标数据失败: {e}")
    
    def store_alerts(self, alerts: List[Dict[str, Any]]) -> None:
        """存储告警信息"""
        try:
            for alert in alerts:
                alert_copy = alert.copy()
                alert_copy['timestamp'] = datetime.now().isoformat()
                self.alerts_history.append(alert_copy)
            
            # 限制告警历史大小
            max_alerts = self.config.get('max_alerts_history', 500)
            if len(self.alerts_history) > max_alerts:
                self.alerts_history = self.alerts_history[-max_alerts:]
                
        except Exception as e:
            print(f"❌ 存储告警数据失败: {e}")
    
    def cleanup_old_data(self) -> Dict[str, int]:
        """清理过期数据"""
        try:
            current_time = datetime.now()
            retention_period = timedelta(seconds=self.config.get('retention_period', 3600))
            
            # 清理过期指标
            initial_metrics_count = len(self.metrics_history)
            self.metrics_history = [
                metric for metric in self.metrics_history
                if self._is_within_retention_period(metric.get('timestamp'), current_time, retention_period)
            ]
            cleaned_metrics = initial_metrics_count - len(self.metrics_history)
            
            # 清理过期告警
            initial_alerts_count = len(self.alerts_history)
            self.alerts_history = [
                alert for alert in self.alerts_history
                if self._is_within_retention_period(alert.get('timestamp'), current_time, retention_period)
            ]
            cleaned_alerts = initial_alerts_count - len(self.alerts_history)
            
            return {
                'cleaned_metrics': cleaned_metrics,
                'cleaned_alerts': cleaned_alerts,
                'remaining_metrics': len(self.metrics_history),
                'remaining_alerts': len(self.alerts_history)
            }
            
        except Exception as e:
            print(f"❌ 清理过期数据失败: {e}")
            return {'error': str(e)}
    
    def _is_within_retention_period(self, timestamp_str: str, current_time: datetime, retention_period: timedelta) -> bool:
        """检查时间戳是否在保留期内"""
        try:
            if not timestamp_str:
                return False
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return current_time - timestamp < retention_period
        except Exception:
            return False
    
    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """获取最新指标"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取指标历史"""
        if limit is None:
            return self.metrics_history.copy()
        return self.metrics_history[-limit:] if limit > 0 else []
    
    def get_alerts_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取告警历史"""
        if limit is None:
            return self.alerts_history.copy()
        return self.alerts_history[-limit:] if limit > 0 else []
    
    def get_recent_alerts(self, hours: int = 1) -> List[Dict[str, Any]]:
        """获取最近N小时的告警"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            return [
                alert for alert in self.alerts_history
                if self._parse_timestamp(alert.get('timestamp')) > cutoff_time
            ]
        except Exception as e:
            print(f"❌ 获取最近告警失败: {e}")
            return []
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """解析时间戳字符串"""
        try:
            if not timestamp_str:
                return datetime.min
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except Exception:
            return datetime.min
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """获取数据统计信息"""
        try:
            # 计算告警类型统计
            alert_types = {}
            for alert in self.alerts_history:
                alert_type = alert.get('type', 'unknown')
                alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
            
            # 计算时间范围
            time_range = {}
            if self.metrics_history:
                time_range['start'] = self.metrics_history[0].get('timestamp')
                time_range['end'] = self.metrics_history[-1].get('timestamp')
            
            return {
                'total_metrics': len(self.metrics_history),
                'total_alerts': len(self.alerts_history),
                'alert_types': alert_types,
                'time_range': time_range,
                'config': {
                    'retention_period_seconds': self.config.get('retention_period', 3600),
                    'max_metrics_history': self.config.get('max_metrics_history', 1000),
                    'max_alerts_history': self.config.get('max_alerts_history', 500)
                }
            }
        except Exception as e:
            return {'error': str(e)}
    
    def clear_all_data(self) -> None:
        """清空所有数据"""
        self.metrics_history.clear()
        self.alerts_history.clear()
    
    def export_data(self) -> Dict[str, Any]:
        """导出所有数据"""
        return {
            'metrics_history': self.metrics_history.copy(),
            'alerts_history': self.alerts_history.copy(),
            'export_time': datetime.now().isoformat(),
            'statistics': self.get_data_statistics()
        }

