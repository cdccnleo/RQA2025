"""
数据持久化组件

负责监控数据的持久化存储和管理。
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional


class DataPersistence:
    """数据持久化管理器"""
    
    def __init__(self, max_history_items: int = 1000, data_file: str = 'monitoring_data.json'):
        """初始化数据持久化管理器"""
        self.max_history_items = max_history_items
        self.data_file = data_file
        self.metrics_history: List[Dict[str, Any]] = []
    
    def save_monitoring_data(self, timestamp: datetime, data: Dict[str, Any]) -> None:
        """保存监控数据"""
        monitoring_record = {
            'timestamp': timestamp.isoformat(),
            'data': data
        }

        self.metrics_history.append(monitoring_record)

        # 限制历史记录数量
        if len(self.metrics_history) > self.max_history_items:
            self.metrics_history = self.metrics_history[-self.max_history_items:]
    
    def persist_monitoring_data(self, config: Dict[str, Any], 
                              alerts_history: List[Dict[str, Any]], 
                              optimization_suggestions: List[Dict[str, Any]]) -> None:
        """持久化监控数据到文件"""
        try:
            monitoring_data = {
                'config': config,
                'metrics_history': self._format_metrics_history(),
                'alerts_history': alerts_history[-50:] if alerts_history else [],  # 只保存最近50条告警
                'optimization_suggestions': optimization_suggestions[-20:] if optimization_suggestions else [],  # 只保存最近20条建议
                'last_updated': datetime.now().isoformat()
            }

            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(monitoring_data, f, ensure_ascii=False, indent=2, default=str)

        except Exception as e:
            print(f"❌ 保存监控数据失败: {e}")
    
    def _format_metrics_history(self) -> List[Dict[str, Any]]:
        """格式化指标历史数据"""
        formatted_records = []
        
        for record in self.metrics_history[-100:]:  # 只保存最近100条
            try:
                data = record.get('data', {})
                formatted_record = {
                    'timestamp': record.get('timestamp', ''),
                    'coverage_percent': data.get('coverage', {}).get('coverage_percent', 0),
                    'memory_usage_mb': data.get('performance', {}).get('memory_usage_mb', 0),
                    'cpu_usage_percent': data.get('performance', {}).get('cpu_usage_percent', 0),
                    'overall_health': data.get('health', {}).get('overall_status', 'unknown')
                }
                formatted_records.append(formatted_record)
            except (KeyError, TypeError, AttributeError) as e:
                print(f"❌ 格式化指标记录时发生错误: {e}")
                continue
        
        return formatted_records
    
    def load_monitoring_data(self) -> Optional[Dict[str, Any]]:
        """加载监控数据"""
        try:
            if not os.path.exists(self.data_file):
                return None
            
            with open(self.data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 加载监控数据失败: {e}")
            return None
    
    def export_data(self, export_file: Optional[str] = None) -> str:
        """导出监控数据"""
        if export_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            export_file = f'monitoring_export_{timestamp}.json'
        
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'metrics_history': self.metrics_history,
                'export_info': {
                    'total_records': len(self.metrics_history),
                    'max_history_items': self.max_history_items,
                    'data_file': self.data_file
                }
            }
            
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
            
            return export_file
        except Exception as e:
            print(f"❌ 导出监控数据失败: {e}")
            return ""
    
    def get_metrics_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取指标历史"""
        if limit is None:
            return self.metrics_history.copy()
        return self.metrics_history[-limit:] if limit > 0 else []
    
    def clear_history(self) -> None:
        """清空历史数据"""
        self.metrics_history.clear()
    
    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """获取最新的指标数据"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_count(self) -> int:
        """获取指标记录数量"""
        return len(self.metrics_history)

