#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证关键模块导入是否修复成功
"""

import os
import sys

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_import(module_name, import_statement):
    """测试单个导入"""
    try:
        exec(import_statement)
        print(f"✅ {module_name}: 导入成功")
        return True
    except ImportError as e:
        print(f"❌ {module_name}: 导入失败 - {e}")
        return False
    except Exception as e:
        print(f"⚠️ {module_name}: 其他错误 - {e}")
        return False

def main():
    """主函数"""
    print("🔍 验证关键模块导入修复效果")
    print("=" * 50)

    # 测试关键模块
    test_cases = [
        ("缓存策略", "from infrastructure.cache.strategies.cache_strategy_manager import LRUStrategy"),
        ("配置验证器", "from infrastructure.config.validators import ValidationSeverity"),
        ("健康检查器", "from infrastructure.health.components.health_checker import AsyncHealthCheckerComponent"),
        ("日志器", "from infrastructure.logging.core.base_logger import BaseLogger"),
        ("安全检查器", "from infrastructure.security.access.components.access_checker import AccessChecker"),
        ("连接池", "from infrastructure.utils.components.connection_pool import ConnectionPool"),
        ("分布式锁", "from infrastructure.distributed.distributed_lock import DistributedLockManager"),
        ("指标收集器", "from infrastructure.monitoring.components.metrics_collector import MetricsCollector"),
        ("事件驱动系统", "from infrastructure.events.event_driven_system import EventDrivenSystem"),
        ("消息队列", "from infrastructure.messaging.async_message_queue import AsyncMessageQueue"),
        ("多级缓存", "from infrastructure.cache.core.multi_level_cache import MultiLevelCache"),
        ("性能监控", "from infrastructure.cache.monitoring.performance_monitor import CachePerformanceMonitor"),
        ("文本格式化器", "from infrastructure.logging.formatters.text import TextFormatter"),
        ("增强验证器", "from infrastructure.config.validators.enhanced_validators import EnhancedConfigValidator"),
        ("ZooKeeper发现", "from infrastructure.distributed.zookeeper_service_discovery import ZooKeeperServiceDiscovery"),
        ("Consul发现", "from infrastructure.distributed.consul_service_discovery import ConsulServiceDiscovery"),
        ("健康监控", "from infrastructure.health.monitoring.health_checker import SystemHealthChecker")
    ]

    successful_imports = 0
    total_tests = len(test_cases)

    print("测试导入中...")
    for module_name, import_stmt in test_cases:
        if test_import(module_name, import_stmt):
            successful_imports += 1

    print()
    print("=" * 50)
    print("📊 导入验证结果:"    print(f"   • 总测试模块: {total_tests}")
    print(f"   • 导入成功: {successful_imports}")
    print(f"   • 导入失败: {total_tests - successful_imports}")
    print(".1f"
    if successful_imports == total_tests:
        print("🎉 所有关键模块导入修复成功！可以进行测试覆盖率统计。")
        return True
    else:
        print("⚠️ 部分模块导入仍需修复。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
