"""
性能和质量管理工具模块

提供性能监控、质量检查和统计相关的工具函数。
"""

import logging
from typing import Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)


def check_data_quality(data: pd.DataFrame, identifier: str, quality_monitor=None):
    """
    检查数据质量
    
    Args:
        data: 数据DataFrame
        identifier: 数据标识符（如股票代码）
        quality_monitor: 质量监控器实例
    
    Returns:
        质量指标对象或None
    """
    if data is None or data.empty:
        return None

    # 如果提供了质量监控器，使用它检查质量
    if quality_monitor is not None:
        try:
            return quality_monitor.check_quality(data, identifier)
        except Exception as e:
            logger.warning(f"质量检查失败 {identifier}: {e}")
            return None
    
    # 简化实现，返回None表示未检查
    return None


def update_avg_response_time(
    performance_metrics: Dict[str, Any], response_time: float
):
    """
    更新平均响应时间
    
    使用指数移动平均算法更新平均响应时间。
    
    Args:
        performance_metrics: 性能指标字典（会被修改）
        response_time: 新的响应时间（毫秒）
    """
    performance_metrics["total_requests"] = performance_metrics.get("total_requests", 0) + 1
    current_avg = performance_metrics.get("avg_response_time", 0.0)

    # 使用指数移动平均
    alpha = 0.1
    performance_metrics["avg_response_time"] = (
        alpha * response_time + (1 - alpha) * current_avg
    )


def monitor_performance():
    """
    监控性能
    
    注意：这是一个占位函数，实际实现需要上下文对象（如类实例）。
    """
    logger.warning("性能监控功能需要集成上下文对象")


def get_integration_stats(
    performance_metrics: Dict[str, Any], cache_strategy=None, parallel_manager=None
) -> Dict[str, Any]:
    """
    获取集成统计信息
    
    Args:
        performance_metrics: 性能指标字典
        cache_strategy: 缓存策略实例
        parallel_manager: 并行管理器实例
    
    Returns:
        包含所有统计信息的字典
    """
    # 获取各组件统计信息
    cache_stats = {}
    if cache_strategy is not None and hasattr(cache_strategy, 'get_stats'):
        cache_stats = cache_strategy.get_stats()
    
    parallel_stats = {}
    if parallel_manager is not None and hasattr(parallel_manager, 'get_stats'):
        parallel_stats = parallel_manager.get_stats()

    # 构建兼容测试的统计信息
    return {
        "total_requests": performance_metrics.get("total_requests", 0),
        "successful_requests": performance_metrics.get("successful_requests", 0),
        "failed_requests": performance_metrics.get("failed_requests", 0),
        "avg_response_time": performance_metrics.get("avg_response_time", 0.0),
        "cache_hit_rate": cache_stats.get("hit_rate", 0.0),
        "memory_usage": performance_metrics.get("memory_usage", 0.0),
        "quality_score": performance_metrics.get("quality_score", 0.0),
        "performance_metrics": performance_metrics.copy(),
        "cache_stats": cache_stats,
        "parallel_stats": parallel_stats,
    }


def shutdown(parallel_manager=None, cache_strategy=None, quality_monitor=None):
    """
    关闭集成管理器
    
    安全地关闭并行管理器、清理缓存等资源。
    
    Args:
        parallel_manager: 并行管理器实例
        cache_strategy: 缓存策略实例
        quality_monitor: 质量监控器实例（不需要特殊清理）
    """
    logger.info("关闭增强版数据层集成管理器")

    # 关闭并行管理器
    if parallel_manager is not None:
        try:
            if hasattr(parallel_manager, 'shutdown'):
                parallel_manager.shutdown()
                logger.debug("并行管理器已关闭")
            else:
                logger.warning("并行管理器没有shutdown方法，跳过关闭")
        except Exception as e:
            logger.error(f"关闭并行管理器失败: {e}")

    # 清理缓存
    if cache_strategy is not None:
        try:
            if hasattr(cache_strategy, 'cleanup'):
                cache_strategy.cleanup()
                logger.debug("缓存策略已清理")
            else:
                logger.warning("缓存策略没有cleanup方法，跳过清理")
        except Exception as e:
            logger.error(f"清理缓存失败: {e}")

    # 清理质量监控器
    # quality_monitor 不需要特别清理
    if quality_monitor is not None:
        logger.debug("质量监控器无需特殊清理")
    
    logger.info("集成管理器关闭完成")

