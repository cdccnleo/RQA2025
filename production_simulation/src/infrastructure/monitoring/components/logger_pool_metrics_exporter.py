"""
Logger池指标导出器组件

负责导出Logger池的监控指标，支持Prometheus格式。
"""

from typing import Dict, Any, Optional, List


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


class LoggerPoolMetricsExporter:
    """Logger池指标导出器"""
    
    def __init__(self, pool_name: str = "default"):
        """初始化指标导出器"""
        self.pool_name = pool_name
    
    def export_prometheus_metrics(self, stats: LoggerPoolStats) -> str:
        """
        生成Prometheus格式的指标
        
        Args:
            stats: Logger池统计数据
            
        Returns:
            Prometheus格式的指标字符串
        """
        if not stats:
            return ""
        
        pool_name = self.pool_name.replace('-', '_')
        
        # 收集所有指标
        metrics = []
        metrics.extend(self._generate_core_metrics(stats, pool_name))
        metrics.extend(self._generate_performance_metrics(stats, pool_name))
        metrics.extend(self._generate_memory_metrics(stats, pool_name))
        
        return '\n'.join(metrics) + '\n'
    
    def _generate_core_metrics(self, stats: LoggerPoolStats, pool_name: str) -> List[str]:
        """生成核心池指标"""
        return [
            f'# HELP logger_pool_size Logger pool current size',
            f'# TYPE logger_pool_size gauge',
            f'logger_pool_size{{pool="{pool_name}"}} {stats.pool_size}',
            
            f'# HELP logger_pool_max_size Logger pool maximum size',
            f'# TYPE logger_pool_max_size gauge',
            f'logger_pool_max_size{{pool="{pool_name}"}} {stats.max_size}',
            
            f'# HELP logger_pool_logger_count Current logger instances count',
            f'# TYPE logger_pool_logger_count gauge',
            f'logger_pool_logger_count{{pool="{pool_name}"}} {stats.logger_count}',
        ]
    
    def _generate_performance_metrics(self, stats: LoggerPoolStats, pool_name: str) -> List[str]:
        """生成性能相关指标"""
        return [
            f'# HELP logger_pool_created_count Total loggers created',
            f'# TYPE logger_pool_created_count counter',
            f'logger_pool_created_count{{pool="{pool_name}"}} {stats.created_count}',
            
            f'# HELP logger_pool_hit_count Total cache hits',
            f'# TYPE logger_pool_hit_count counter',
            f'logger_pool_hit_count{{pool="{pool_name}"}} {stats.hit_count}',
            
            f'# HELP logger_pool_hit_rate Cache hit rate (0.0-1.0)',
            f'# TYPE logger_pool_hit_rate gauge',
            f'logger_pool_hit_rate{{pool="{pool_name}"}} {stats.hit_rate}',
            
            f'# HELP logger_pool_total_access_count Total access count',
            f'# TYPE logger_pool_total_access_count counter',
            f'logger_pool_total_access_count{{pool="{pool_name}"}} {stats.total_access_count}',
            
            f'# HELP logger_pool_avg_access_time_seconds Average access time in seconds',
            f'# TYPE logger_pool_avg_access_time_seconds gauge',
            f'logger_pool_avg_access_time_seconds{{pool="{pool_name}"}} {stats.avg_access_time}',
        ]
    
    def _generate_memory_metrics(self, stats: LoggerPoolStats, pool_name: str) -> List[str]:
        """生成内存相关指标"""
        return [
            f'# HELP logger_pool_memory_usage_mb Memory usage in MB',
            f'# TYPE logger_pool_memory_usage_mb gauge',
            f'logger_pool_memory_usage_mb{{pool="{pool_name}"}} {stats.memory_usage_mb}',
        ]
    
    def export_json_metrics(self, stats: LoggerPoolStats) -> Dict[str, Any]:
        """
        导出JSON格式的指标
        
        Args:
            stats: Logger池统计数据
            
        Returns:
            JSON格式的指标字典
        """
        if not stats:
            return {}
        
        return {
            'pool_name': self.pool_name,
            'timestamp': stats.timestamp,
            'metrics': {
                'pool_size': stats.pool_size,
                'max_size': stats.max_size,
                'created_count': stats.created_count,
                'hit_count': stats.hit_count,
                'hit_rate': stats.hit_rate,
                'logger_count': stats.logger_count,
                'total_access_count': stats.total_access_count,
                'avg_access_time': stats.avg_access_time,
                'memory_usage_mb': stats.memory_usage_mb,
            },
            'derived_metrics': {
                'pool_utilization': stats.pool_size / stats.max_size if stats.max_size > 0 else 0,
                'miss_rate': 1.0 - stats.hit_rate,
                'memory_efficiency_score': max(0, 100 - (stats.memory_usage_mb / stats.max_size * 100)) if stats.max_size > 0 else 0
            }
        }
    
    def export_summary_report(self, stats: LoggerPoolStats, 
                            alert_status: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
        """
        导出汇总报告
        
        Args:
            stats: Logger池统计数据
            alert_status: 告警状态
            
        Returns:
            汇总报告字典
        """
        if not stats:
            return {}
        
        pool_utilization = stats.pool_size / stats.max_size if stats.max_size > 0 else 0
        
        return {
            'pool_name': self.pool_name,
            'timestamp': stats.timestamp,
            'current_stats': {
                'pool_size': stats.pool_size,
                'max_size': stats.max_size,
                'hit_rate': stats.hit_rate,
                'memory_usage_mb': stats.memory_usage_mb,
                'avg_access_time_ms': stats.avg_access_time * 1000,
            },
            'performance_metrics': {
                'pool_utilization': pool_utilization,
                'memory_efficiency': 'good' if stats.memory_usage_mb < 50 else 'high',
                'performance_score': self._calculate_performance_score(stats)
            },
            'alert_status': alert_status or {},
            'recommendations': self._generate_recommendations(stats, alert_status)
        }
    
    def _calculate_performance_score(self, stats: LoggerPoolStats) -> float:
        """计算性能评分"""
        try:
            # 综合考虑命中率、内存使用和访问时间
            hit_rate_score = stats.hit_rate * 40  # 40%权重
            memory_score = max(0, (100 - stats.memory_usage_mb) / 100) * 30  # 30%权重
            access_time_score = max(0, (0.01 - stats.avg_access_time) / 0.01) * 30  # 30%权重
            
            return min(100, hit_rate_score + memory_score + access_time_score)
        except Exception:
            return 0.0
    
    def _generate_recommendations(self, stats: LoggerPoolStats, 
                                alert_status: Optional[Dict[str, bool]] = None) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        if not stats:
            return recommendations
        
        if alert_status:
            if alert_status.get('hit_rate_low'):
                recommendations.append("考虑增加池大小以提高命中率")
            
            if alert_status.get('pool_usage_high'):
                recommendations.append("池使用率过高，考虑增加max_size")
            
            if alert_status.get('memory_high'):
                recommendations.append("内存使用过高，考虑优化Logger实例")
        
        # 基于统计数据生成建议
        if stats.hit_rate > 0.95:
            recommendations.append("命中率优秀，性能表现良好")
        elif stats.hit_rate < 0.7:
            recommendations.append("命中率偏低，建议检查池配置")
        
        pool_utilization = stats.pool_size / stats.max_size if stats.max_size > 0 else 0
        if pool_utilization > 0.8:
            recommendations.append("池使用率较高，建议监控并考虑扩容")
        
        if stats.avg_access_time > 0.01:  # 10ms
            recommendations.append("访问时间较长，建议优化Logger实例创建流程")
        
        return recommendations

