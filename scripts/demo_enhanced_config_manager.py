#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增强配置管理器演示脚本
展示性能优化、监控增强、安全加固等功能
"""

from src.infrastructure.config.monitoring import (
    get_config_monitor, get_audit_logger, get_health_checker
)
from src.infrastructure.config.unified_manager import (
    UnifiedConfigManager, ConfigScope,
    CachePolicy
)
import sys
import os
import time
import threading
import tempfile

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def demo_enhanced_config_manager():
    """增强配置管理器演示"""
    print("🚀 开始增强配置管理器演示")
    print("=" * 50)

    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    print(f"📁 临时目录: {temp_dir}")

    try:
        # 1️⃣ 测试性能优化功能
        print("\n1️⃣ 测试性能优化功能")
        print("-" * 30)

        # 创建带缓存的配置管理器
        config_manager = UnifiedConfigManager(
            cache_policy=CachePolicy.LRU,
            cache_size=100,
            enable_encryption=True
        )

        # 性能测试：大量配置操作
        start_time = time.time()
        for i in range(1000):
            config_manager.set(f"perf_key_{i}", f"perf_value_{i}")

        # 缓存命中测试
        for i in range(1000):
            config_manager.get(f"perf_key_{i}")

        end_time = time.time()
        duration = end_time - start_time

        print(f"✅ 性能测试完成")
        print(f"   操作数量: 2000 (1000次写入 + 1000次读取)")
        print(f"   总耗时: {duration:.3f}秒")
        print(f"   平均耗时: {duration/2000*1000:.2f}ms/操作")

        # 获取性能指标
        metrics = config_manager.get_performance_metrics()
        print(f"   总请求数: {metrics['total_requests']}")
        print(f"   缓存命中率: {metrics['cache_hit_rate']*100:.1f}%")
        print(f"   平均响应时间: {metrics['avg_response_time']*1000:.2f}ms")

        # 2️⃣ 测试监控增强功能
        print("\n2️⃣ 测试监控增强功能")
        print("-" * 30)

        # 获取监控器
        monitor = get_config_monitor(config_manager)

        # 添加告警处理器
        alert_events = []

        def alert_handler(alert):
            alert_events.append(alert)
            print(f"   告警: [{alert.level.value.upper()}] {alert.message}")

        monitor.add_alert_handler(alert_handler)

        # 启动监控
        monitor.start_monitoring(interval=1.0)

        # 模拟一些操作触发告警
        for i in range(20):
            config_manager.set(f"monitor_key_{i}", f"monitor_value_{i}")
            time.sleep(0.1)

        time.sleep(2)  # 等待监控检查

        # 获取监控报告
        report = monitor.get_monitoring_report()
        print(f"✅ 监控报告:")
        print(f"   监控状态: {'活跃' if report['monitoring_active'] else '停止'}")
        print(f"   配置变更: {report['metrics']['config_changes']}")
        print(f"   告警处理器: {report['alert_handlers_count']}个")
        print(f"   性能历史: {report['performance_history_count']}个快照")

        # 停止监控
        monitor.stop_monitoring()

        # 3️⃣ 测试安全加固功能
        print("\n3️⃣ 测试安全加固功能")
        print("-" * 30)

        # 设置敏感配置
        sensitive_configs = {
            "database.password": "secret_password_123",
            "api.key": "api_key_secret_456",
            "redis.password": "redis_secret_789"
        }

        for key, value in sensitive_configs.items():
            config_manager.set(key, value)
            print(f"   设置敏感配置: {key}")

        # 验证加密存储
        with config_manager._lock:
            for key, original_value in sensitive_configs.items():
                stored_value = config_manager._config.get(key)
                if stored_value != original_value:
                    print(f"   ✅ {key} 已加密存储")
                else:
                    print(f"   ❌ {key} 未加密存储")

        # 验证解密读取
        for key, expected_value in sensitive_configs.items():
            retrieved_value = config_manager.get(key)
            if retrieved_value == expected_value:
                print(f"   ✅ {key} 解密读取正常")
            else:
                print(f"   ❌ {key} 解密读取失败")

        # 4️⃣ 测试审计日志功能
        print("\n4️⃣ 测试审计日志功能")
        print("-" * 30)

        audit_logger = get_audit_logger()

        # 记录配置变更
        audit_logger.log_config_change(
            key="test.key",
            old_value="old_value",
            new_value="new_value",
            user="demo_user",
            scope="global"
        )

        # 记录配置访问
        audit_logger.log_config_access(
            key="database.url",
            user="demo_user",
            scope="global",
            success=True
        )

        # 记录安全事件
        audit_logger.log_security_event(
            event_type="encryption_error",
            details={"key": "test.key", "error": "encryption_failed"},
            user="demo_user"
        )

        print("   ✅ 审计日志记录完成")

        # 5️⃣ 测试健康检查功能
        print("\n5️⃣ 测试健康检查功能")
        print("-" * 30)

        health_checker = get_health_checker(config_manager)
        health_status = health_checker.check_config_health()

        print(f"   整体状态: {health_status['overall_status']}")
        print(f"   检查项目: {len(health_status['checks'])}个")

        for check_name, check_result in health_status['checks'].items():
            status_icon = "✅" if check_result['status'] == 'passed' else "❌"
            print(f"   {status_icon} {check_name}: {check_result['status']}")

        # 6️⃣ 测试并发安全功能
        print("\n6️⃣ 测试并发安全功能")
        print("-" * 30)

        results = []
        errors = []

        def concurrent_reader():
            """并发读取线程"""
            try:
                for i in range(100):
                    value = config_manager.get(f"concurrent_key_{i}", default=f"default_{i}")
                    results.append(value)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def concurrent_writer():
            """并发写入线程"""
            try:
                for i in range(50):
                    config_manager.set(f"concurrent_key_{i}", f"value_{i}")
                    time.sleep(0.002)
            except Exception as e:
                errors.append(e)

        # 启动多个线程
        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=concurrent_reader))
        threads.append(threading.Thread(target=concurrent_writer))

        start_time = time.time()
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()
        end_time = time.time()

        print(f"   ✅ 并发测试完成")
        print(f"   线程数: {len(threads)}")
        print(f"   耗时: {end_time - start_time:.3f}秒")
        print(f"   错误数: {len(errors)}")
        print(f"   结果数: {len(results)}")

        # 7️⃣ 测试缓存优化功能
        print("\n7️⃣ 测试缓存优化功能")
        print("-" * 30)

        # 获取缓存统计
        cache_stats = config_manager.get_cache_stats()
        print(f"   缓存状态: {'启用' if cache_stats['enabled'] else '禁用'}")
        if cache_stats['enabled']:
            print(f"   缓存大小: {cache_stats['size']}/{cache_stats['max_size']}")
            print(f"   缓存策略: {cache_stats['policy']}")

        # 测试缓存淘汰
        if config_manager.cache:
            # 添加超出缓存容量的配置
            for i in range(200):
                config_manager.set(f"cache_test_{i}", f"cache_value_{i}")

            # 检查缓存大小
            final_cache_stats = config_manager.get_cache_stats()
            print(f"   最终缓存大小: {final_cache_stats['size']}")
            print(f"   缓存命中率: {config_manager.metrics.get_cache_hit_rate()*100:.1f}%")

        # 8️⃣ 测试配置验证增强
        print("\n8️⃣ 测试配置验证增强")
        print("-" * 30)

        # 测试有效配置
        valid_config = {
            "database.port": 5432,
            "cache.max_size": 1000,
            "risk.max_drawdown": 0.1,
            "risk.stop_loss": 0.05
        }

        is_valid, errors = config_manager.validate(valid_config)
        print(f"   有效配置验证: {'通过' if is_valid else '失败'}")

        # 测试无效配置
        invalid_config = {
            "database.port": "not_a_number",
            "risk.max_drawdown": 1.5,  # 超出范围
            "database.url": None        # 必需项为空
        }

        is_valid, errors = config_manager.validate(invalid_config)
        print(f"   无效配置验证: {'通过' if is_valid else '失败'}")
        if errors:
            print(f"   验证错误数: {len(errors)}")

        # 9️⃣ 测试作用域配置加密
        print("\n9️⃣ 测试作用域配置加密")
        print("-" * 30)

        # 设置作用域配置
        scope_config = {
            "feature.enabled": True,
            "feature.api_key": "scope_secret_key",
            "feature.endpoint": "https://api.example.com"
        }

        config_manager.set_scope_config(ConfigScope.FEATURES, scope_config)

        # 获取作用域配置
        retrieved_config = config_manager.get_scope_config(ConfigScope.FEATURES)

        # 验证配置
        for key, expected_value in scope_config.items():
            if retrieved_config.get(key) == expected_value:
                print(f"   ✅ {key}: 配置正确")
            else:
                print(f"   ❌ {key}: 配置错误")

        # 🔟 测试导出导入加密配置
        print("\n🔟 测试导出导入加密配置")
        print("-" * 30)

        # 导出配置
        exported_config = config_manager.export_config()

        # 创建新的配置管理器
        new_config_manager = UnifiedConfigManager(enable_encryption=True)

        # 导入配置
        success = new_config_manager.import_config(exported_config)
        print(f"   导入结果: {'成功' if success else '失败'}")

        # 验证导入的配置
        for key, expected_value in sensitive_configs.items():
            imported_value = new_config_manager.get(key)
            if imported_value == expected_value:
                print(f"   ✅ {key}: 导入正确")
            else:
                print(f"   ❌ {key}: 导入错误")

        print("\n✅ 所有测试通过!")

    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 清理临时目录
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\n🧹 已清理临时目录: {temp_dir}")


def demo_performance_comparison():
    """性能对比演示"""
    print("\n🎯 性能对比演示")
    print("=" * 50)

    # 测试无缓存配置管理器
    print("\n📊 无缓存配置管理器性能测试")
    no_cache_manager = UnifiedConfigManager(cache_policy=CachePolicy.NO_CACHE)

    start_time = time.time()
    for i in range(1000):
        no_cache_manager.set(f"perf_key_{i}", f"perf_value_{i}")
        no_cache_manager.get(f"perf_key_{i}")
    no_cache_time = time.time() - start_time

    # 测试有缓存配置管理器
    print("📊 有缓存配置管理器性能测试")
    cache_manager = UnifiedConfigManager(cache_policy=CachePolicy.LRU, cache_size=1000)

    start_time = time.time()
    for i in range(1000):
        cache_manager.set(f"perf_key_{i}", f"perf_value_{i}")
        cache_manager.get(f"perf_key_{i}")
    cache_time = time.time() - start_time

    # 性能对比
    print(f"\n📈 性能对比结果:")
    print(f"   无缓存耗时: {no_cache_time:.3f}秒")
    print(f"   有缓存耗时: {cache_time:.3f}秒")
    print(f"   性能提升: {((no_cache_time - cache_time) / no_cache_time * 100):.1f}%")

    # 缓存命中率
    cache_metrics = cache_manager.get_performance_metrics()
    print(f"   缓存命中率: {cache_metrics['cache_hit_rate']*100:.1f}%")


def demo_monitoring_dashboard():
    """监控面板演示"""
    print("\n📊 监控面板演示")
    print("=" * 50)

    config_manager = UnifiedConfigManager(
        cache_policy=CachePolicy.LRU,
        cache_size=100,
        enable_encryption=True
    )

    monitor = get_config_monitor(config_manager)
    health_checker = get_health_checker(config_manager)

    # 模拟一些操作
    for i in range(50):
        config_manager.set(f"dashboard_key_{i}", f"dashboard_value_{i}")
        config_manager.get(f"dashboard_key_{i}")

    # 获取监控数据
    metrics = config_manager.get_performance_metrics()
    cache_stats = config_manager.get_cache_stats()
    health_status = health_checker.check_config_health()

    print("📊 实时监控数据:")
    print(f"   总请求数: {metrics['total_requests']}")
    print(f"   缓存命中率: {metrics['cache_hit_rate']*100:.1f}%")
    print(f"   平均响应时间: {metrics['avg_response_time']*1000:.2f}ms")
    print(f"   缓存大小: {cache_stats['size']}/{cache_stats['max_size']}")
    print(f"   健康状态: {health_status['overall_status']}")

    # 显示健康检查详情
    print("\n🏥 健康检查详情:")
    for check_name, check_result in health_status['checks'].items():
        status_icon = "✅" if check_result['status'] == 'passed' else "❌"
        print(f"   {status_icon} {check_name}: {check_result['status']}")


if __name__ == "__main__":
    print("🎉 增强配置管理器演示")
    print("=" * 60)

    try:
        # 主演示
        demo_enhanced_config_manager()

        # 性能对比演示
        demo_performance_comparison()

        # 监控面板演示
        demo_monitoring_dashboard()

        print("\n🎉 演示完成!")
        print("📖 迁移指南: docs/migration/config_management_migration.md")
        print("🧪 测试文件: tests/unit/infrastructure/config/test_unified_config_manager_enhanced.py")

    except Exception as e:
        print(f"\n💥 演示失败: {e}")
        import traceback
        traceback.print_exc()
