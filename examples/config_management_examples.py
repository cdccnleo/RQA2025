#!/usr/bin/env python3
"""
配置管理模块使用示例

本文件展示了配置管理模块的各种使用场景和最佳实践。
"""

import time
import threading
from src.infrastructure.config import (
    UnifiedConfigManager,
    CachePolicy,
    ConfigScope
)


def basic_usage_example():
    """基础使用示例"""
    print("=== 基础使用示例 ===")

    # 创建配置管理器
    config_manager = UnifiedConfigManager(
        cache_policy=CachePolicy.LRU,
        cache_size=100,
        enable_encryption=True
    )

    # 设置配置
    config_manager.set("database.host", "localhost")
    config_manager.set("database.port", 5432)
    config_manager.set("database.password", "secret123")
    config_manager.set("cache.max_size", 1000)
    config_manager.set("risk.max_drawdown", 0.1)

    # 获取配置
    host = config_manager.get("database.host")
    port = config_manager.get("database.port")
    password = config_manager.get("database.password")  # 自动解密

    print(f"数据库主机: {host}")
    print(f"数据库端口: {port}")
    print(f"数据库密码: {password}")

    # 验证配置
    config = {
        "database.host": host,
        "database.port": port,
        "cache.max_size": 1000,
        "risk.max_drawdown": 0.1
    }

    is_valid, errors = config_manager.validate(config)
    if is_valid:
        print("✅ 配置验证通过")
    else:
        print("❌ 配置验证失败:", errors)


def encryption_example():
    """加密功能示例"""
    print("\n=== 加密功能示例 ===")

    # 创建启用加密的配置管理器
    config_manager = UnifiedConfigManager(enable_encryption=True)

    # 设置敏感配置
    sensitive_configs = {
        "database.password": "secret_password",
        "api.key": "api_secret_key",
        "redis.password": "redis_secret"
    }

    for key, value in sensitive_configs.items():
        config_manager.set(key, value)
        print(f"设置敏感配置: {key}")

    # 获取配置（自动解密）
    for key in sensitive_configs.keys():
        value = config_manager.get(key)
        print(f"获取配置 {key}: {value}")

    # 验证内部存储是加密的
    with config_manager._lock:
        stored_value = config_manager._config_manager._configs[ConfigScope.GLOBAL].get(
            "database.password")
        if stored_value != "secret_password":
            print("✅ 敏感配置已正确加密存储")


def cache_performance_example():
    """缓存性能示例"""
    print("\n=== 缓存性能示例 ===")

    # 创建配置管理器
    config_manager = UnifiedConfigManager(
        cache_policy=CachePolicy.LRU,
        cache_size=50
    )

    # 预热缓存
    print("预热缓存...")
    for i in range(100):
        config_manager.set(f"key_{i}", f"value_{i}")

    # 性能测试
    print("开始性能测试...")
    start_time = time.time()

    # 执行大量读取操作
    for _ in range(1000):
        for i in range(50):  # 只访问前50个键，应该大部分命中缓存
            config_manager.get(f"key_{i}")

    end_time = time.time()
    duration = end_time - start_time

    # 获取性能指标
    metrics = config_manager.get_performance_metrics()
    cache_stats = config_manager.get_cache_stats()

    print(f"测试耗时: {duration:.3f}秒")
    print(f"总请求数: {metrics['total_requests']}")
    print(f"缓存命中数: {metrics['cache_hits']}")
    print(f"缓存未命中数: {metrics['cache_misses']}")
    print(f"缓存命中率: {metrics['cache_hit_rate']:.2%}")
    print(f"平均响应时间: {metrics['avg_response_time']:.6f}秒")
    print(f"缓存大小: {cache_stats.get('size', 0)}")


def validation_example():
    """配置验证示例"""
    print("\n=== 配置验证示例 ===")

    config_manager = UnifiedConfigManager()

    # 有效配置
    valid_config = {
        "database.port": 5432,
        "cache.max_size": 1000,
        "risk.max_drawdown": 0.1,
        "risk.stop_loss": 0.05,
        "database.url": "localhost"
    }

    is_valid, errors = config_manager.validate(valid_config)
    if is_valid:
        print("✅ 有效配置验证通过")
    else:
        print("❌ 有效配置验证失败:", errors)

    # 无效配置
    invalid_config = {
        "database.port": "not_a_number",  # 应该是整数
        "cache.max_size": -1,             # 应该是正整数
        "risk.max_drawdown": 1.5,         # 超出范围
        "risk.stop_loss": -0.1,           # 超出范围
        "database.url": None              # 不能为空
    }

    is_valid, errors = config_manager.validate(invalid_config)
    if not is_valid:
        print("❌ 无效配置验证失败:")
        for key, error in errors.items():
            print(f"  {key}: {error}")


def concurrent_access_example():
    """并发访问示例"""
    print("\n=== 并发访问示例 ===")

    config_manager = UnifiedConfigManager()
    results = []
    errors = []

    def reader_thread():
        """读取线程"""
        try:
            for i in range(100):
                value = config_manager.get(f"concurrent.key_{i}", default=f"default_{i}")
                results.append(value)
                time.sleep(0.001)
        except Exception as e:
            errors.append(e)

    def writer_thread():
        """写入线程"""
        try:
            for i in range(50):
                config_manager.set(f"concurrent.key_{i}", f"value_{i}")
                time.sleep(0.002)
        except Exception as e:
            errors.append(e)

    # 启动多个线程
    threads = []
    for _ in range(5):
        threads.append(threading.Thread(target=reader_thread))
    threads.append(threading.Thread(target=writer_thread))

    print("启动并发测试...")
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    # 验证结果
    print(f"并发测试完成")
    print(f"读取结果数量: {len(results)}")
    print(f"错误数量: {len(errors)}")

    if len(errors) == 0:
        print("✅ 并发访问测试通过")
    else:
        print("❌ 并发访问测试失败:", errors)


def export_import_example():
    """配置导入导出示例"""
    print("\n=== 配置导入导出示例 ===")

    # 创建源配置管理器
    source_config = UnifiedConfigManager(enable_encryption=True)

    # 设置一些配置
    test_configs = {
        "database.host": "source-host",
        "database.port": 5432,
        "database.password": "source_password",
        "cache.max_size": 1000,
        "api.key": "source_api_key"
    }

    for key, value in test_configs.items():
        source_config.set(key, value)

    # 导出配置
    exported_config = source_config.export_config()
    print(f"导出配置项数量: {len(exported_config)}")

    # 创建目标配置管理器
    target_config = UnifiedConfigManager(enable_encryption=True)

    # 导入配置
    success = target_config.import_config(exported_config)

    if success:
        print("✅ 配置导入成功")

        # 验证导入的配置
        for key, expected_value in test_configs.items():
            actual_value = target_config.get(key)
            if actual_value == expected_value:
                print(f"✅ {key}: {actual_value}")
            else:
                print(f"❌ {key}: 期望 {expected_value}, 实际 {actual_value}")
    else:
        print("❌ 配置导入失败")


def monitoring_example():
    """监控示例"""
    print("\n=== 监控示例 ===")

    config_manager = UnifiedConfigManager(
        cache_policy=CachePolicy.LRU,
        cache_size=100
    )

    # 执行一些操作
    for i in range(100):
        config_manager.set(f"monitor.key_{i}", f"value_{i}")

    for i in range(200):
        config_manager.get(f"monitor.key_{i % 100}")

    # 获取监控报告
    report = config_manager.get_monitoring_report()

    print("监控报告:")
    print(f"  性能指标:")
    for key, value in report['performance'].items():
        print(f"    {key}: {value}")

    print(f"  缓存统计:")
    for key, value in report['cache'].items():
        print(f"    {key}: {value}")

    print(f"  配置信息:")
    for key, value in report['config'].items():
        print(f"    {key}: {value}")


def error_handling_example():
    """错误处理示例"""
    print("\n=== 错误处理示例 ===")

    config_manager = UnifiedConfigManager()

    # 测试各种错误情况
    test_cases = [
        ("获取None键", lambda: config_manager.get(None)),
        ("获取不存在的键", lambda: config_manager.get("nonexistent.key")),
        ("设置None值", lambda: config_manager.set("test.key", None)),
        ("验证无效配置", lambda: config_manager.validate({"invalid": "config"}))
    ]

    for test_name, test_func in test_cases:
        try:
            result = test_func()
            print(f"✅ {test_name}: {result}")
        except Exception as e:
            print(f"❌ {test_name}: {type(e).__name__}: {e}")


def best_practices_example():
    """最佳实践示例"""
    print("\n=== 最佳实践示例 ===")

    # 1. 配置命名规范
    print("1. 配置命名规范:")
    config_manager = UnifiedConfigManager()

    # 推荐: 使用点分隔的层次结构
    config_manager.set("database.host", "localhost")
    config_manager.set("database.port", 5432)
    config_manager.set("cache.max_size", 1000)
    config_manager.set("risk.max_drawdown", 0.1)

    print("   ✅ 使用点分隔的层次结构")

    # 2. 敏感信息处理
    print("\n2. 敏感信息处理:")
    secure_config = UnifiedConfigManager(enable_encryption=True)
    secure_config.set("database.password", "secret123")
    secure_config.set("api.key", "api_secret_key")

    password = secure_config.get("database.password")
    api_key = secure_config.get("api.key")

    print(f"   ✅ 敏感配置已加密: password={password}, api_key={api_key}")

    # 3. 性能优化
    print("\n3. 性能优化:")
    optimized_config = UnifiedConfigManager(
        cache_policy=CachePolicy.LRU,
        cache_size=1000
    )

    # 预热缓存
    for i in range(100):
        optimized_config.set(f"optimized.key_{i}", f"value_{i}")

    # 测试缓存效果
    start_time = time.time()
    for i in range(1000):
        optimized_config.get(f"optimized.key_{i % 100}")
    end_time = time.time()

    metrics = optimized_config.get_performance_metrics()
    print(f"   ✅ 缓存命中率: {metrics['cache_hit_rate']:.2%}")
    print(f"   ✅ 平均响应时间: {metrics['avg_response_time']:.6f}秒")

    # 4. 监控和告警
    print("\n4. 监控和告警:")
    metrics = optimized_config.get_performance_metrics()

    if metrics['cache_hit_rate'] < 0.8:
        print("   ⚠️  警告: 缓存命中率过低")
    else:
        print("   ✅ 缓存命中率正常")

    if metrics['avg_response_time'] > 0.01:
        print("   ⚠️  警告: 响应时间过长")
    else:
        print("   ✅ 响应时间正常")


def main():
    """主函数"""
    print("配置管理模块使用示例")
    print("=" * 50)

    try:
        # 运行所有示例
        basic_usage_example()
        encryption_example()
        cache_performance_example()
        validation_example()
        concurrent_access_example()
        export_import_example()
        monitoring_example()
        error_handling_example()
        best_practices_example()

        print("\n" + "=" * 50)
        print("✅ 所有示例运行完成")

    except Exception as e:
        print(f"\n❌ 示例运行失败: {e}")
        raise


if __name__ == "__main__":
    main()
