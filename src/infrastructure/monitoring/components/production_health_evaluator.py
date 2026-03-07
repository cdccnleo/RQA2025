"""
生产环境健康评估器组件

负责计算系统健康状态和生成性能报告。
"""

from datetime import datetime
from typing import Dict, Any, List


class ProductionHealthEvaluator:
    """生产环境健康评估器"""
    
    def __init__(self):
        """初始化健康评估器"""
        pass
    
    def evaluate_health_status(self, metrics_history: List[Dict[str, Any]], 
                             system_info: Dict[str, Any], 
                             is_monitoring: bool) -> Dict[str, Any]:
        """评估系统健康状态"""
        if not metrics_history:
            return {'status': 'no_data', 'message': 'No metrics available'}
        
        try:
            latest_metrics = metrics_history[-1]
            health_score = self._calculate_health_score(latest_metrics)
            status = self._determine_health_status(health_score)
            
            return {
                'status': status,
                'health_score': health_score,
                'latest_metrics': latest_metrics,
                'system_info': system_info,
                'monitoring_active': is_monitoring,
                'evaluation_time': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Health evaluation failed: {str(e)}',
                'evaluation_time': datetime.now().isoformat()
            }
    
    def generate_performance_report(self, metrics_history: List[Dict[str, Any]], 
                                  alerts_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成性能报告"""
        if not metrics_history:
            return {'error': 'No metrics data available'}
        
        try:
            return {
                'time_range': self._get_time_range_info(metrics_history),
                'cpu_stats': self._calculate_cpu_statistics(metrics_history),
                'memory_stats': self._calculate_memory_statistics(metrics_history),
                'disk_stats': self._calculate_disk_statistics(metrics_history),
                'alerts_summary': self._generate_alerts_summary(alerts_history),
                'report_generated_at': datetime.now().isoformat()
            }
        except Exception as e:
            return {'error': f'Performance report generation failed: {str(e)}'}
    
    def _calculate_health_score(self, metrics: Dict[str, Any]) -> int:
        """计算健康评分"""
        health_score = 100
        
        try:
            # CPU健康评分
            cpu_percent = metrics.get('cpu', {}).get('percent', 0)
            health_score -= self._calculate_cpu_penalty(cpu_percent)
            
            # 内存健康评分
            memory_percent = metrics.get('memory', {}).get('percent', 0)
            health_score -= self._calculate_memory_penalty(memory_percent)
            
            # 磁盘健康评分
            disk_percent = metrics.get('disk', {}).get('percent', 0)
            health_score -= self._calculate_disk_penalty(disk_percent)
            
            # 确保分数在0-100范围内
            return max(0, min(100, health_score))
            
        except Exception:
            return 0
    
    def _calculate_cpu_penalty(self, cpu_percent: float) -> int:
        """计算CPU使用率惩罚分数"""
        if cpu_percent > 80:
            return 20
        elif cpu_percent > 60:
            return 10
        return 0
    
    def _calculate_memory_penalty(self, memory_percent: float) -> int:
        """计算内存使用率惩罚分数"""
        if memory_percent > 85:
            return 20
        elif memory_percent > 70:
            return 10
        return 0
    
    def _calculate_disk_penalty(self, disk_percent: float) -> int:
        """计算磁盘使用率惩罚分数"""
        if disk_percent > 90:
            return 30
        elif disk_percent > 80:
            return 15
        return 0
    
    def _determine_health_status(self, health_score: int) -> str:
        """根据健康评分确定健康状态"""
        if health_score >= 80:
            return 'healthy'
        elif health_score >= 60:
            return 'warning'
        else:
            return 'critical'
    
    def _get_time_range_info(self, metrics_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """获取时间范围信息"""
        if not metrics_history:
            return {}
        
        return {
            'start': metrics_history[0].get('timestamp'),
            'end': metrics_history[-1].get('timestamp'),
            'data_points': len(metrics_history)
        }
    
    def _calculate_cpu_statistics(self, metrics_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算CPU统计信息"""
        try:
            cpu_percents = [
                m.get('cpu', {}).get('percent', 0) 
                for m in metrics_history 
                if m.get('cpu', {}).get('percent') is not None
            ]
            
            if not cpu_percents:
                return {}
            
            return {
                'avg_percent': sum(cpu_percents) / len(cpu_percents),
                'max_percent': max(cpu_percents),
                'min_percent': min(cpu_percents),
                'data_points': len(cpu_percents)
            }
        except Exception:
            return {}
    
    def _calculate_memory_statistics(self, metrics_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算内存统计信息"""
        try:
            memory_percents = [
                m.get('memory', {}).get('percent', 0) 
                for m in metrics_history 
                if m.get('memory', {}).get('percent') is not None
            ]
            
            if not memory_percents:
                return {}
            
            return {
                'avg_percent': sum(memory_percents) / len(memory_percents),
                'max_percent': max(memory_percents),
                'min_percent': min(memory_percents),
                'data_points': len(memory_percents)
            }
        except Exception:
            return {}
    
    def _calculate_disk_statistics(self, metrics_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算磁盘统计信息"""
        try:
            disk_percents = [
                m.get('disk', {}).get('percent', 0) 
                for m in metrics_history 
                if m.get('disk', {}).get('percent') is not None
            ]
            
            if not disk_percents:
                return {}
            
            return {
                'avg_percent': sum(disk_percents) / len(disk_percents),
                'max_percent': max(disk_percents),
                'min_percent': min(disk_percents),
                'data_points': len(disk_percents)
            }
        except Exception:
            return {}
    
    def _generate_alerts_summary(self, alerts_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成告警摘要"""
        try:
            if not alerts_history:
                return {
                    'total_alerts': 0,
                    'alert_types': [],
                    'recent_alerts': []
                }
            
            # 统计告警类型
            alert_types = list(set(a.get('type', 'unknown') for a in alerts_history))
            
            return {
                'total_alerts': len(alerts_history),
                'alert_types': alert_types,
                'recent_alerts': alerts_history[-5:],
                'alert_levels': self._count_alert_levels(alerts_history)
            }
        except Exception:
            return {}
    
    def _count_alert_levels(self, alerts_history: List[Dict[str, Any]]) -> Dict[str, int]:
        """统计告警级别"""
        try:
            level_counts = {}
            for alert in alerts_history:
                level = alert.get('level', 'unknown')
                level_counts[level] = level_counts.get(level, 0) + 1
            return level_counts
        except Exception:
            return {}
    
    def get_health_recommendations(self, health_score: int, metrics: Dict[str, Any]) -> List[str]:
        """根据健康状态和指标生成建议"""
        recommendations = []
        
        try:
            if health_score < 60:
                recommendations.append("系统健康状态严重，建议立即检查系统资源使用情况")
            
            cpu_percent = metrics.get('cpu', {}).get('percent', 0)
            if cpu_percent > 80:
                recommendations.append("CPU使用率过高，建议优化程序性能或增加计算资源")
            
            memory_percent = metrics.get('memory', {}).get('percent', 0)
            if memory_percent > 85:
                recommendations.append("内存使用率过高，建议检查内存泄漏或增加内存")
            
            disk_percent = metrics.get('disk', {}).get('percent', 0)
            if disk_percent > 90:
                recommendations.append("磁盘空间不足，建议清理临时文件或扩展存储")
            
            if health_score > 90:
                recommendations.append("系统运行良好，保持当前状态")
                
        except Exception as e:
            recommendations.append(f"生成建议时出错: {str(e)}")
        
        return recommendations

