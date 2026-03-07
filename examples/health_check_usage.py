#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
健康检查模块使用示例
演示如何使用健康检查模块的各种功能
"""

from src.infrastructure.health import (
    get_enhanced_health_checker,
    get_cache_manager,
    get_prometheus_exporter,
    get_alert_manager,
    AlertRule,
    AlertSeverity,
    NotificationChannel
)
import asyncio
import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入健康检查模块


async def basic_health_check_example():
    """基础健康检查示例"""
    logger.info("=== 基础健康检查示例 ===")

    # 获取健康检查器实例
    checker = get_enhanced_health_checker()

    # 执行健康检查
    result = await checker.perform_health_check('example_service', 'availability')
    logger.info(f"健康检查结果: {result}")

    # 获取综合健康状态
    status = await checker.get_comprehensive_health_status()
    logger.info(f"综合健康状态: {status['overall_status']}")


def cache_management_example():
    """缓存管理示例"""
    logger.info("=== 缓存管理示例 ===")

    # 获取缓存管理器
    cache_manager = get_cache_manager()

    # 设置缓存
    cache_manager.set('example_key', 'example_value', ttl=600, priority=10)

    # 获取缓存
    value = cache_manager.get('example_key')
    logger.info(f"缓存值: {value}")

    # 获取或计算缓存
    def compute_function():
        logger.info("计算函数被调用")
        return {'computed': 'value', 'timestamp': time.time()}

    cached_value = cache_manager.get_or_compute('computed_key', compute_function, ttl=300)
    logger.info(f"计算缓存值: {cached_value}")

    # 再次获取，应该从缓存返回
    cached_value_again = cache_manager.get_or_compute('computed_key', compute_function, ttl=300)
    logger.info(f"再次获取缓存值: {cached_value_again}")

    # 获取缓存统计
    stats = cache_manager.get_stats()
    logger.info(f"缓存统计: {stats}")


def alert_management_example():
    """告警管理示例"""
    logger.info("=== 告警管理示例 ===")

    # 获取告警管理器
    alert_manager = get_alert_manager()

    # 创建自定义告警规则
    custom_rule = AlertRule(
        name="custom_metric_alert",
        description="自定义指标告警",
        metric_name="custom_metric",
        threshold=100,
        comparison=">",
        severity=AlertSeverity.WARNING,
        duration=60,
        notification_channels=[NotificationChannel.WEBHOOK],
        escalation_delay=300
    )

    # 添加告警规则
    alert_manager.add_alert_rule(custom_rule)

    # 检查告警条件
    alerts = alert_manager.check_alert_condition('custom_metric', 150)
    logger.info(f"触发的告警数量: {len(alerts)}")

    # 获取活跃告警
    active_alerts = alert_manager.get_active_alerts()
    logger.info(f"活跃告警数量: {len(active_alerts)}")


def prometheus_integration_example():
    """Prometheus集成示例"""
    logger.info("=== Prometheus集成示例 ===")

    # 获取Prometheus导出器
    exporter = get_prometheus_exporter()

    # 记录健康检查指标
    exporter.record_health_check(
        service='example_service',
        check_type='availability',
        status='healthy',
        response_time=0.05
    )

    # 记录系统指标
    exporter.record_system_metrics(
        host='example-server',
        cpu_percent=45.2,
        memory_bytes=8589934592,  # 8GB
        disk_usage={'/': 75.5, '/data': 60.2}
    )

    # 记录缓存指标
    exporter.record_cache_metrics(
        cache_type='health_check',
        hit_rate=95.5,
        total_entries=150,
        evictions=5,
        policy='lru'
    )

    # 获取指标摘要
    summary = exporter.get_metrics_summary()
    logger.info(f"指标摘要: {summary}")


async def advanced_health_check_example():
    """高级健康检查示例"""
    logger.info("=== 高级健康检查示例 ===")

    # 获取健康检查器
    checker = get_enhanced_health_checker()

    # 注册自定义健康检查
    async def custom_database_check():
        """模拟数据库健康检查"""
        await asyncio.sleep(0.1)  # 模拟检查时间
        return {
            'status': 'healthy',
            'connection_pool_size': 10,
            'active_connections': 3,
            'response_time': 0.05
        }

    async def custom_cache_check():
        """模拟缓存健康检查"""
        await asyncio.sleep(0.05)
        return {
            'status': 'healthy',
            'hit_rate': 95.5,
            'memory_usage': '256MB'
        }

    # 注册检查函数
    checker.register_health_check('database', custom_database_check)
    checker.register_health_check('cache', custom_cache_check)

    # 执行所有健康检查
    comprehensive_status = await checker.get_comprehensive_health_status()
    logger.info(f"综合健康状态: {comprehensive_status}")


def cache_policy_example():
    """缓存策略示例"""
    logger.info("=== 缓存策略示例 ===")

    # 创建不同策略的缓存管理器
    lru_cache = get_cache_manager()

    # 设置预加载键
    lru_cache.set_preload_keys(['frequently_used', 'system_config'])

    # 模拟预加载函数
    def get_frequently_used_data():
        return {'data': 'frequently accessed', 'timestamp': time.time()}

    def get_system_config():
        return {'version': '2.1.0', 'environment': 'production'}

    # 预加载缓存
    lru_cache.preload_cache({
        'frequently_used': get_frequently_used_data,
        'system_config': get_system_config
    })

    # 获取预加载的数据
    frequent_data = lru_cache.get('frequently_used')
    system_config = lru_cache.get('system_config')

    logger.info(f"预加载数据: {frequent_data}")
    logger.info(f"系统配置: {system_config}")


async def performance_monitoring_example():
    """性能监控示例"""
    logger.info("=== 性能监控示例 ===")

    # 获取健康检查器
    checker = get_enhanced_health_checker()

    # 模拟性能测试
    start_time = time.time()

    # 执行多次健康检查
    for i in range(5):
        result = await checker.perform_health_check('performance_test', 'response_time')
        logger.info(f"第{i+1}次检查: {result.response_time:.3f}s")
        await asyncio.sleep(0.1)

    end_time = time.time()
    total_time = end_time - start_time

    # 获取系统指标
    system_metrics = await checker.get_system_metrics()
    logger.info(f"系统指标: CPU {system_metrics.cpu_percent}%, 内存 {system_metrics.memory_percent}%")

    # 获取性能摘要
    performance_summary = checker.get_metrics_summary()
    logger.info(f"性能摘要: {performance_summary}")

    logger.info(f"总执行时间: {total_time:.3f}s")


async def main():
    """主函数"""
    logger.info("开始健康检查模块使用示例")

    try:
        # 基础示例
        await basic_health_check_example()

        # 缓存管理
        cache_management_example()

        # 告警管理
        alert_management_example()

        # Prometheus集成
        prometheus_integration_example()

        # 高级健康检查
        await advanced_health_check_example()

        # 缓存策略
        cache_policy_example()

        # 性能监控
        await performance_monitoring_example()

        logger.info("所有示例执行完成")

    except Exception as e:
        logger.error(f"示例执行出错: {e}")
        raise

if __name__ == "__main__":
    # 运行示例
    asyncio.run(main())
