#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强型特征工程监控指标收集器

提供全面的监控指标收集功能，包括：
- 任务执行时间统计
- 特征计算性能指标
- 数据质量评分趋势
- 系统资源使用情况
"""

import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading

logger = logging.getLogger(__name__)


class EnhancedFeatureMetricsCollector:
    """
    增强型特征工程监控指标收集器
    
    功能：
    1. 任务执行时间统计
    2. 特征计算性能指标
    3. 数据质量评分趋势
    4. 系统资源使用情况
    5. 实时监控告警
    """
    
    def __init__(self, max_history_size: int = 1000):
        """
        初始化监控指标收集器
        
        Args:
            max_history_size: 历史数据最大保留数量
        """
        self.max_history_size = max_history_size
        
        # 任务执行时间统计
        self.task_execution_times: Dict[str, List[float]] = defaultdict(list)
        self.task_start_times: Dict[str, float] = {}
        
        # 特征计算性能指标
        self.feature_calculation_times: Dict[str, List[float]] = defaultdict(list)
        self.feature_counts: deque = deque(maxlen=max_history_size)
        
        # 数据质量评分趋势
        self.quality_scores: deque = deque(maxlen=max_history_size)
        self.quality_history: List[Dict[str, Any]] = []
        
        # 系统资源使用
        self.resource_usage: deque = deque(maxlen=max_history_size)
        
        # 错误统计
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.error_history: deque = deque(maxlen=100)
        
        # 线程锁
        self._lock = threading.Lock()
        
        # 启动时间
        self.start_time = time.time()
        
        logger.info("增强型监控指标收集器已初始化")
    
    def record_task_start(self, task_id: str, task_type: str = None) -> None:
        """
        记录任务开始时间
        
        Args:
            task_id: 任务ID
            task_type: 任务类型
        """
        with self._lock:
            self.task_start_times[task_id] = time.time()
            logger.debug(f"任务开始时间已记录: {task_id}")
    
    def record_task_completion(self, task_id: str, success: bool = True, 
                              feature_count: int = 0, error_message: str = None) -> None:
        """
        记录任务完成
        
        Args:
            task_id: 任务ID
            success: 是否成功
            feature_count: 生成的特征数量
            error_message: 错误信息（如果失败）
        """
        with self._lock:
            if task_id in self.task_start_times:
                execution_time = time.time() - self.task_start_times[task_id]
                
                # 记录执行时间
                task_type = 'success' if success else 'failed'
                self.task_execution_times[task_type].append(execution_time)
                
                # 限制历史数据大小
                if len(self.task_execution_times[task_type]) > self.max_history_size:
                    self.task_execution_times[task_type] = \
                        self.task_execution_times[task_type][-self.max_history_size:]
                
                # 记录特征数量
                if success and feature_count > 0:
                    self.feature_counts.append({
                        'task_id': task_id,
                        'count': feature_count,
                        'timestamp': time.time(),
                        'execution_time': execution_time
                    })
                
                # 记录错误
                if not success and error_message:
                    self.error_counts[error_message[:100]] += 1
                    self.error_history.append({
                        'task_id': task_id,
                        'error': error_message,
                        'timestamp': time.time()
                    })
                
                # 清理开始时间记录
                del self.task_start_times[task_id]
                
                logger.info(f"任务完成已记录: {task_id}, 执行时间: {execution_time:.2f}s, "
                           f"成功: {success}, 特征数: {feature_count}")
    
    def record_feature_calculation(self, feature_name: str, calculation_time: float) -> None:
        """
        记录特征计算时间
        
        Args:
            feature_name: 特征名称
            calculation_time: 计算时间（秒）
        """
        with self._lock:
            self.feature_calculation_times[feature_name].append(calculation_time)
            
            # 限制历史数据大小
            if len(self.feature_calculation_times[feature_name]) > 100:
                self.feature_calculation_times[feature_name] = \
                    self.feature_calculation_times[feature_name][-100:]
    
    def record_quality_score(self, task_id: str, score: float, details: Dict[str, Any] = None) -> None:
        """
        记录数据质量评分
        
        Args:
            task_id: 任务ID
            score: 质量评分（0-100）
            details: 详细信息
        """
        with self._lock:
            quality_record = {
                'task_id': task_id,
                'score': score,
                'timestamp': time.time(),
                'datetime': datetime.now().isoformat(),
                'details': details or {}
            }
            
            self.quality_scores.append(quality_record)
            self.quality_history.append(quality_record)
            
            # 限制历史数据大小
            if len(self.quality_history) > self.max_history_size:
                self.quality_history = self.quality_history[-self.max_history_size:]
            
            logger.debug(f"质量评分已记录: {task_id}, 评分: {score:.2f}")
    
    def record_resource_usage(self, cpu_percent: float, memory_percent: float, 
                             memory_mb: float) -> None:
        """
        记录系统资源使用情况
        
        Args:
            cpu_percent: CPU使用率
            memory_percent: 内存使用率
            memory_mb: 内存使用（MB）
        """
        with self._lock:
            self.resource_usage.append({
                'timestamp': time.time(),
                'datetime': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_mb': memory_mb
            })
    
    def get_task_execution_stats(self) -> Dict[str, Any]:
        """
        获取任务执行统计
        
        Returns:
            任务执行统计信息
        """
        with self._lock:
            stats = {}
            
            for task_type, times in self.task_execution_times.items():
                if times:
                    stats[task_type] = {
                        'count': len(times),
                        'avg_time': sum(times) / len(times),
                        'min_time': min(times),
                        'max_time': max(times),
                        'total_time': sum(times)
                    }
            
            # 计算成功率
            success_count = len(self.task_execution_times.get('success', []))
            failed_count = len(self.task_execution_times.get('failed', []))
            total_count = success_count + failed_count
            
            if total_count > 0:
                stats['success_rate'] = (success_count / total_count) * 100
                stats['total_tasks'] = total_count
            else:
                stats['success_rate'] = 0
                stats['total_tasks'] = 0
            
            return stats
    
    def get_feature_calculation_stats(self) -> Dict[str, Any]:
        """
        获取特征计算统计
        
        Returns:
            特征计算统计信息
        """
        with self._lock:
            stats = {}
            
            # 计算每个特征的平均计算时间
            for feature_name, times in self.feature_calculation_times.items():
                if times:
                    stats[feature_name] = {
                        'avg_time': sum(times) / len(times),
                        'min_time': min(times),
                        'max_time': max(times),
                        'calculation_count': len(times)
                    }
            
            # 总体统计
            if self.feature_counts:
                total_features = sum(item['count'] for item in self.feature_counts)
                avg_features_per_task = total_features / len(self.feature_counts)
                stats['overall'] = {
                    'total_features_generated': total_features,
                    'avg_features_per_task': avg_features_per_task,
                    'tasks_with_features': len(self.feature_counts)
                }
            
            return stats
    
    def get_quality_trend(self, hours: int = 24) -> Dict[str, Any]:
        """
        获取质量评分趋势
        
        Args:
            hours: 查询小时数
            
        Returns:
            质量评分趋势
        """
        with self._lock:
            cutoff_time = time.time() - (hours * 3600)
            recent_scores = [
                record for record in self.quality_history 
                if record['timestamp'] > cutoff_time
            ]
            
            if not recent_scores:
                return {'message': '没有质量评分数据'}
            
            scores = [record['score'] for record in recent_scores]
            
            return {
                'period_hours': hours,
                'data_points': len(scores),
                'avg_score': sum(scores) / len(scores),
                'min_score': min(scores),
                'max_score': max(scores),
                'latest_score': scores[-1] if scores else None,
                'trend': 'up' if len(scores) > 1 and scores[-1] > scores[0] else 'down',
                'history': recent_scores[-50:]  # 最近50条记录
            }
    
    def get_resource_usage_stats(self, minutes: int = 60) -> Dict[str, Any]:
        """
        获取资源使用统计
        
        Args:
            minutes: 查询分钟数
            
        Returns:
            资源使用统计
        """
        with self._lock:
            cutoff_time = time.time() - (minutes * 60)
            recent_usage = [
                record for record in self.resource_usage 
                if record['timestamp'] > cutoff_time
            ]
            
            if not recent_usage:
                return {'message': '没有资源使用数据'}
            
            cpu_values = [r['cpu_percent'] for r in recent_usage]
            memory_values = [r['memory_percent'] for r in recent_usage]
            memory_mb_values = [r['memory_mb'] for r in recent_usage]
            
            return {
                'period_minutes': minutes,
                'data_points': len(recent_usage),
                'cpu': {
                    'avg': sum(cpu_values) / len(cpu_values),
                    'min': min(cpu_values),
                    'max': max(cpu_values),
                    'current': cpu_values[-1]
                },
                'memory': {
                    'avg_percent': sum(memory_values) / len(memory_values),
                    'max_percent': max(memory_values),
                    'avg_mb': sum(memory_mb_values) / len(memory_mb_values),
                    'max_mb': max(memory_mb_values),
                    'current_percent': memory_values[-1]
                }
            }
    
    def get_error_stats(self) -> Dict[str, Any]:
        """
        获取错误统计
        
        Returns:
            错误统计信息
        """
        with self._lock:
            return {
                'total_errors': sum(self.error_counts.values()),
                'error_types': dict(self.error_counts),
                'recent_errors': list(self.error_history)[-10:]
            }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """
        获取所有监控指标
        
        Returns:
            完整的监控指标数据
        """
        return {
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'uptime_seconds': time.time() - self.start_time,
            'task_execution': self.get_task_execution_stats(),
            'feature_calculation': self.get_feature_calculation_stats(),
            'quality_trend': self.get_quality_trend(),
            'resource_usage': self.get_resource_usage_stats(),
            'errors': self.get_error_stats()
        }
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """
        检查告警条件
        
        Returns:
            告警列表
        """
        alerts = []
        
        # 检查成功率
        task_stats = self.get_task_execution_stats()
        if task_stats.get('success_rate', 100) < 80:
            alerts.append({
                'level': 'warning',
                'type': 'low_success_rate',
                'message': f"任务成功率过低: {task_stats['success_rate']:.1f}%",
                'timestamp': time.time()
            })
        
        # 检查质量评分
        quality_trend = self.get_quality_trend(hours=1)
        if quality_trend.get('avg_score', 100) < 60:
            alerts.append({
                'level': 'warning',
                'type': 'low_quality_score',
                'message': f"数据质量评分过低: {quality_trend['avg_score']:.1f}",
                'timestamp': time.time()
            })
        
        # 检查资源使用
        resource_stats = self.get_resource_usage_stats(minutes=5)
        if resource_stats.get('cpu', {}).get('max', 0) > 90:
            alerts.append({
                'level': 'critical',
                'type': 'high_cpu_usage',
                'message': f"CPU使用率过高: {resource_stats['cpu']['max']:.1f}%",
                'timestamp': time.time()
            })
        
        if resource_stats.get('memory', {}).get('max_percent', 0) > 90:
            alerts.append({
                'level': 'critical',
                'type': 'high_memory_usage',
                'message': f"内存使用率过高: {resource_stats['memory']['max_percent']:.1f}%",
                'timestamp': time.time()
            })
        
        return alerts


# 全局实例
_enhanced_metrics_collector: Optional[EnhancedFeatureMetricsCollector] = None


def get_enhanced_metrics_collector() -> EnhancedFeatureMetricsCollector:
    """
    获取全局监控指标收集器实例
    
    Returns:
        监控指标收集器实例
    """
    global _enhanced_metrics_collector
    if _enhanced_metrics_collector is None:
        _enhanced_metrics_collector = EnhancedFeatureMetricsCollector()
    return _enhanced_metrics_collector
