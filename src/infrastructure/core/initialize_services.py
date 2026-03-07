#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施服务统一初始化

使用InfrastructureServiceRegistry统一初始化所有基础设施服务
"""

import logging
from typing import Dict, Any, Optional
from .service_registry import InfrastructureServiceRegistry, get_service_registry


def initialize_infrastructure_services() -> Dict[str, bool]:
    """
    统一初始化所有基础设施服务
    
    使用InfrastructureServiceRegistry注册和初始化所有核心基础设施服务
    
    Returns:
        Dict[str, bool]: 服务名称到初始化结果的映射
    """
    logger = logging.getLogger(__name__)
    registry = get_service_registry()
    results = {}
    
    logger.info("开始初始化基础设施服务...")
    
    # 注册配置管理器
    try:
        if not registry.is_service_registered('config_manager'):
            from src.infrastructure.config.core.unified_manager import UnifiedConfigManager
            registry.register_singleton('config_manager', service_class=UnifiedConfigManager)
            logger.info("注册配置管理器成功")
        results['config_manager'] = True
    except Exception as e:
        logger.warning(f"注册配置管理器失败: {e}")
        results['config_manager'] = False
    
    # 注册缓存管理器
    try:
        if not registry.is_service_registered('cache_manager'):
            from src.infrastructure.cache.core.cache_manager import UnifiedCacheManager
            registry.register_singleton('cache_manager', service_class=UnifiedCacheManager)
            logger.info("注册缓存管理器成功")
        results['cache_manager'] = True
    except Exception as e:
        logger.warning(f"注册缓存管理器失败: {e}")
        results['cache_manager'] = False
    
    # 注册监控服务
    try:
        if not registry.is_service_registered('monitoring'):
            from src.infrastructure.monitoring import ContinuousMonitoringSystem
            registry.register_singleton('monitoring', service_class=ContinuousMonitoringSystem)
            logger.info("注册监控服务成功")
        results['monitoring'] = True
    except Exception as e:
        logger.warning(f"注册监控服务失败: {e}")
        results['monitoring'] = False
    
    # 注册健康检查器
    try:
        if not registry.is_service_registered('health_checker'):
            from src.infrastructure.health.components.enhanced_health_checker import EnhancedHealthChecker
            registry.register_singleton('health_checker', service_class=EnhancedHealthChecker)
            logger.info("注册健康检查器成功")
        results['health_checker'] = True
    except Exception as e:
        logger.warning(f"注册健康检查器失败: {e}")
        results['health_checker'] = False
    
    # 初始化所有已注册的单例服务
    init_results = registry.initialize_all_services()
    results.update(init_results)
    
    # 预热Logger池（优化性能）
    try:
        from src.infrastructure.logging.core.logger_pool import LoggerPool
        logger_pool = LoggerPool.get_instance()
        logger_pool.warmup()
        logger.info("Logger池预热完成")
        results['logger_pool_warmup'] = True
    except Exception as e:
        logger.debug(f"Logger池预热失败（非关键）: {e}")
        results['logger_pool_warmup'] = False
    
    # 统计初始化结果
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    logger.info(f"基础设施服务初始化完成: {success_count}/{total_count} 成功")
    
    return results


def get_infrastructure_service(service_name: str) -> Optional[Any]:
    """
    获取基础设施服务实例
    
    Args:
        service_name: 服务名称
        
    Returns:
        服务实例，如果不存在则返回None
    """
    registry = get_service_registry()
    return registry.get_service(service_name)