"""
健康状态评估器

提供复杂的健康状态评估逻辑的简化重构。
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from enum import Enum

from src.infrastructure.logging.core.unified_logger import get_unified_logger

logger = get_unified_logger(__name__)


class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    ERROR = "error"


@dataclass
class ThresholdConfig:
    """阈值配置"""
    warning: float
    critical: float


@dataclass
class MetricValue:
    """指标值"""
    name: str
    value: float
    unit: Optional[str] = None


class HealthStatusEvaluator:
    """
    健康状态评估器
    
    职责：
    - 简化复杂的条件判断逻辑
    - 统一的阈值检查和状态评估
    - 生成问题描述和建议
    """
    
    def __init__(self, thresholds: Optional[Dict[str, ThresholdConfig]] = None):
        """
        初始化评估器
        
        Args:
            thresholds: 阈值配置字典
        """
        self.thresholds = thresholds or {}
    
    def evaluate_metric(self, 
                       metric_value: float, 
                       metric_name: str,
                       threshold_config: ThresholdConfig) -> Tuple[HealthStatus, List[str], List[str]]:
        """
        评估单个指标的健康状态
        
        Args:
            metric_value: 指标值
            metric_name: 指标名称
            threshold_config: 阈值配置
            
        Returns:
            (状态, 问题列表, 建议列表)
        """
        issues = []
        recommendations = []
        status = HealthStatus.HEALTHY
        
        if metric_value > threshold_config.critical:
            status = HealthStatus.CRITICAL
            issues.append(f"{metric_name}过高: {metric_value:.1%}")
            recommendations.append(self._get_critical_recommendation(metric_name))
        elif metric_value > threshold_config.warning:
            status = HealthStatus.WARNING
            issues.append(f"{metric_name}较高: {metric_value:.1%}")
            recommendations.append(self._get_warning_recommendation(metric_name))
        
        return status, issues, recommendations
    
    def evaluate_multiple_metrics(self, metrics: List[MetricValue]) -> Tuple[HealthStatus, List[str], List[str]]:
        """
        评估多个指标的综合健康状态
        
        Args:
            metrics: 指标列表
            
        Returns:
            (综合状态, 所有问题列表, 所有建议列表)
        """
        all_issues = []
        all_recommendations = []
        statuses = []
        
        for metric in metrics:
            if metric.name in self.thresholds:
                threshold_config = self.thresholds[metric.name]
                status, issues, recommendations = self.evaluate_metric(
                    metric.value, metric.name, threshold_config
                )
                statuses.append(status)
                all_issues.extend(issues)
                all_recommendations.extend(recommendations)
            else:
                logger.warning(f"未找到指标 {metric.name} 的阈值配置")
        
        # 确定综合状态（取最严重的状态）
        overall_status = self._determine_overall_status(statuses)
        
        return overall_status, all_issues, all_recommendations
    
    def _determine_overall_status(self, statuses: List[HealthStatus]) -> HealthStatus:
        """确定综合状态"""
        if not statuses:
            return HealthStatus.HEALTHY
        
        # 状态优先级：CRITICAL > ERROR > WARNING > HEALTHY
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.ERROR in statuses:
            return HealthStatus.ERROR
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def _get_critical_recommendation(self, metric_name: str) -> str:
        """获取严重问题的建议"""
        recommendations = {
            "error_rate": "检查错误日志，排查问题",
            "memory_usage": "考虑增加内存或优化内存使用",
            "cpu_usage": "考虑增加CPU资源或优化查询",
            "disk_usage": "考虑清理数据或扩容磁盘",
            "response_time": "优化查询性能或增加资源"
        }
        return recommendations.get(metric_name, "需要立即处理")
    
    def _get_warning_recommendation(self, metric_name: str) -> str:
        """获取警告的建议"""
        recommendations = {
            "error_rate": "监控错误趋势",
            "memory_usage": "监控内存使用趋势",
            "cpu_usage": "监控CPU使用趋势",
            "disk_usage": "监控磁盘使用趋势",
            "response_time": "监控响应时间趋势"
        }
        return recommendations.get(metric_name, "持续监控")


class ComponentHealthChecker:
    """
    组件健康检查器
    
    用于检查各种组件的基础配置和状态
    """
    
    @staticmethod
    def check_attributes_exist(obj: object, required_attrs: List[str]) -> Tuple[bool, List[str]]:
        """
        检查对象是否具有必需的属性
        
        Args:
            obj: 要检查的对象
            required_attrs: 必需的属性列表
            
        Returns:
            (是否所有属性都存在, 缺失的属性列表)
        """
        missing_attrs = []
        for attr in required_attrs:
            if not hasattr(obj, attr):
                missing_attrs.append(attr)
        
        return len(missing_attrs) == 0, missing_attrs
    
    @staticmethod
    def check_component_state(executor=None, semaphore=None, shutdown_attr: str = "_shutdown") -> Tuple[bool, List[str]]:
        """
        检查组件状态的统一方法
        
        Args:
            executor: 执行器对象
            semaphore: 信号量对象
            shutdown_attr: 关闭状态属性名
            
        Returns:
            (是否健康, 问题描述列表)
        """
        issues = []
        
        # 检查执行器状态
        if executor:
            if hasattr(executor, shutdown_attr) and getattr(executor, shutdown_attr):
                issues.append("执行器已关闭")
        
        # 检查信号量状态
        if semaphore and hasattr(semaphore, '_value'):
            if semaphore._value < 0:
                issues.append("信号量值异常")
        
        return len(issues) == 0, issues


class ConditionalLogicSimplifier:
    """
    条件逻辑简化器
    
    用于简化复杂的条件判断和嵌套逻辑
    """
    
    @staticmethod
    def safe_get_nested_dict(dict_obj: dict, keys: List[str], default: Any = None) -> Any:
        """
        安全地获取嵌套字典的值
        
        Args:
            dict_obj: 字典对象
            keys: 键路径列表
            default: 默认值
            
        Returns:
            获取到的值或默认值
        """
        try:
            current = dict_obj
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    @staticmethod
    def evaluate_and_conditions(*conditions) -> bool:
        """评估多个AND条件"""
        return all(conditions)
    
    @staticmethod
    def evaluate_or_conditions(*conditions) -> bool:
        """评估多个OR条件"""
        return any(conditions)
    
    @staticmethod
    def chain_conditions(conditions: List[Tuple[bool, str]]) -> Tuple[bool, List[str]]:
        """
        链式评估条件
        
        Args:
            conditions: (条件结果, 错误消息) 的列表
            
        Returns:
            (是否全部通过, 失败的消息列表)
        """
        failed_messages = []
        all_passed = True
        
        for condition, message in conditions:
            if not condition:
                all_passed = False
                failed_messages.append(message)
        
        return all_passed, failed_messages

