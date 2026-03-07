"""
分布式同步功能示例
演示如何使用配置管理模块的分布式同步功能
"""

import time
import json
import os
import tempfile

from src.infrastructure.config import (
    UnifiedConfigManager, CachePolicy, SyncConfig
)


def create_test_config_file(file_path: str, config_data: dict):
    """创建测试配置文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)


def distributed_sync_basic_example():
    """基础分布式同步示例"""
    print("=== 基础分布式同步示例 ===")

    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = os.path.join(temp_dir, "distributed_config.json")

        # 创建初始配置文件
        initial_config = {
            "app": {
                "name": "分布式量化交易系统",
                "version": "1.0.0",
                "debug": True
            },
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "trading_db"
            },
            "trading": {
                "max_positions": 10,
                "risk_limit": 0.02
            }
        }

        create_test_config_file(config_file, initial_config)
        print(f"创建配置文件: {config_file}")

        # 创建同步配置
        sync_config = SyncConfig(
            sync_interval=30,  # 30秒同步间隔
            retry_count=3,     # 重试3次
            conflict_resolution="merge",  # 冲突解决策略
            enable_encryption=True,       # 启用加密
            sync_metadata=True           # 同步元数据
        )

        # 创建启用分布式同步的配置管理器
        config_manager = UnifiedConfigManager(
            enable_distributed_sync=True,
            sync_config=sync_config,
            cache_policy=CachePolicy.LRU,
            cache_size=100
        )

        # 加载初始配置
        config_manager.load(config_file)
        print("加载初始配置...")

        # 注册同步节点
        nodes = [
            ("node1", "192.168.1.100", 8080),
            ("node2", "192.168.1.101", 8080),
            ("node3", "192.168.1.102", 8080)
        ]

        for node_id, address, port in nodes:
            success = config_manager.register_sync_node(node_id, address, port)
            if success:
                print(f"注册节点成功: {node_id} ({address}:{port})")
            else:
                print(f"注册节点失败: {node_id}")

        # 获取同步状态
        status = config_manager.get_sync_status()
        print(f"同步状态: {status}")

        # 同步配置到所有节点
        print("\n开始同步配置到所有节点...")
        sync_result = config_manager.sync_config_to_nodes()
        print(f"同步结果: {sync_result}")

        # 启动自动同步
        print("\n启动自动同步...")
        config_manager.start_auto_sync()

        # 等待一段时间
        time.sleep(2)

        # 检查同步状态
        status = config_manager.get_sync_status()
        print(f"自动同步状态: {status['auto_sync']}")

        # 获取同步历史
        history = config_manager.get_sync_history(limit=5)
        print(f"同步历史记录数: {len(history)}")

        # 停止自动同步
        config_manager.stop_auto_sync()
        print("停止自动同步")


def distributed_sync_conflict_resolution_example():
    """分布式同步冲突解决示例"""
    print("\n=== 分布式同步冲突解决示例 ===")

    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = os.path.join(temp_dir, "conflict_config.json")

        # 创建初始配置文件
        initial_config = {
            "app": {
                "name": "冲突测试应用",
                "version": "1.0.0"
            },
            "trading": {
                "max_positions": 10,
                "risk_limit": 0.02,
                "strategy": "momentum"
            }
        }

        create_test_config_file(config_file, initial_config)
        print(f"创建配置文件: {config_file}")

        # 创建配置管理器
        config_manager = UnifiedConfigManager(enable_distributed_sync=True)

        # 加载配置
        config_manager.load(config_file)

        # 注册节点
        config_manager.register_sync_node("node1", "192.168.1.100", 8080)
        config_manager.register_sync_node("node2", "192.168.1.101", 8080)

        # 模拟配置冲突
        print("模拟配置冲突...")

        # 获取冲突
        conflicts = config_manager.get_conflicts()
        print(f"检测到的冲突数量: {len(conflicts)}")

        # 解决冲突
        print("解决配置冲突...")
        resolution_result = config_manager.resolve_conflicts(strategy="merge")
        print(f"冲突解决结果: {resolution_result}")

        # 再次检查冲突
        conflicts_after = config_manager.get_conflicts()
        print(f"解决后的冲突数量: {len(conflicts_after)}")


def distributed_sync_callback_example():
    """分布式同步回调示例"""
    print("\n=== 分布式同步回调示例 ===")

    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = os.path.join(temp_dir, "callback_config.json")

        # 创建初始配置文件
        initial_config = {
            "app": {
                "name": "回调测试应用",
                "version": "1.0.0"
            }
        }

        create_test_config_file(config_file, initial_config)
        print(f"创建配置文件: {config_file}")

        # 创建配置管理器
        config_manager = UnifiedConfigManager(enable_distributed_sync=True)

        # 加载配置
        config_manager.load(config_file)

        # 注册节点
        config_manager.register_sync_node("node1", "192.168.1.100", 8080)

        # 同步回调函数
        def sync_callback(node_id: str, result: dict):
            print(f"同步回调 - 节点: {node_id}, 结果: {result}")

        # 冲突回调函数
        def conflict_callback(conflicts: list):
            print(f"冲突回调 - 冲突数量: {len(conflicts)}")
            for conflict in conflicts:
                print(f"  - 冲突路径: {conflict.get('path', 'unknown')}")

        # 添加回调函数
        config_manager.add_sync_callback(sync_callback)
        config_manager.add_conflict_callback(conflict_callback)

        print("已添加同步和冲突回调函数")

        # 执行同步操作
        print("执行同步操作...")
        sync_result = config_manager.sync_config_to_nodes()
        print(f"同步完成: {sync_result}")

        # 检查冲突
        conflicts = config_manager.get_conflicts()
        if conflicts:
            print(f"发现 {len(conflicts)} 个冲突")


def distributed_sync_performance_example():
    """分布式同步性能示例"""
    print("\n=== 分布式同步性能示例 ===")

    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = os.path.join(temp_dir, "performance_config.json")

        # 创建大型配置文件
        large_config = {
            "app": {
                "name": "性能测试应用",
                "version": "1.0.0",
                "features": ["feature1", "feature2", "feature3"] * 100
            },
            "database": {
                "host": "localhost",
                "port": 5432,
                "connections": list(range(1000)),
                "pools": {f"pool_{i}": {"size": 10} for i in range(50)}
            },
            "trading": {
                "strategies": {f"strategy_{i}": {"enabled": True, "params": {"param1": i}} for i in range(100)},
                "risk_limits": {f"limit_{i}": 0.01 * i for i in range(50)}
            },
            "monitoring": {
                "metrics": {f"metric_{i}": {"enabled": True, "interval": 5} for i in range(200)}
            }
        }

        create_test_config_file(config_file, large_config)
        print(f"创建大型配置文件: {config_file}")

        # 创建配置管理器
        config_manager = UnifiedConfigManager(
            enable_distributed_sync=True,
            cache_policy=CachePolicy.LRU,
            cache_size=1000
        )

        # 加载大型配置
        config_manager.load(config_file)
        print("加载大型配置...")

        # 注册多个节点
        nodes = []
        for i in range(10):
            node_id = f"node{i}"
            address = f"192.168.1.{100 + i}"
            port = 8080 + i
            nodes.append((node_id, address, port))
            config_manager.register_sync_node(node_id, address, port)

        print(f"注册了 {len(nodes)} 个节点")

        # 性能测试：同步到所有节点
        print("\n开始性能测试：同步到所有节点...")
        start_time = time.time()

        sync_result = config_manager.sync_config_to_nodes()

        end_time = time.time()
        duration = end_time - start_time

        print(f"性能测试完成:")
        print(f"- 同步节点数: {len(nodes)}")
        print(f"- 配置大小: {len(json.dumps(large_config))} 字节")
        print(f"- 同步耗时: {duration:.3f}秒")
        print(f"- 平均每节点耗时: {duration/len(nodes):.3f}秒")

        # 获取性能指标
        metrics = config_manager.get_performance_metrics()
        cache_stats = config_manager.get_cache_stats()

        print(f"\n性能指标:")
        print(f"- 总请求数: {metrics.get('total_requests', 0)}")
        print(f"- 平均响应时间: {metrics.get('avg_response_time', 0):.6f}秒")
        print(f"- 缓存命中率: {metrics.get('cache_hit_rate', 0):.2%}")
        print(f"- 缓存大小: {cache_stats.get('size', 0)}")

        # 获取同步状态
        sync_status = config_manager.get_sync_status()
        print(f"\n同步状态:")
        print(f"- 启用状态: {sync_status['enabled']}")
        print(f"- 节点数量: {len(sync_status['nodes'])}")
        print(f"- 同步历史: {len(sync_status['sync_history'])}")
        print(f"- 冲突数量: {len(sync_status['conflicts'])}")


def distributed_sync_error_handling_example():
    """分布式同步错误处理示例"""
    print("\n=== 分布式同步错误处理示例 ===")

    # 创建配置管理器
    config_manager = UnifiedConfigManager(enable_distributed_sync=True)

    # 测试1：注册无效节点
    print("测试1：注册无效节点")
    result = config_manager.register_sync_node("", "invalid", -1)
    print(f"注册无效节点结果: {result}")

    # 测试2：注销不存在的节点
    print("\n测试2：注销不存在的节点")
    result = config_manager.unregister_sync_node("nonexistent")
    print(f"注销不存在节点结果: {result}")

    # 测试3：同步到不存在的节点
    print("\n测试3：同步到不存在的节点")
    result = config_manager.sync_config_to_nodes(["nonexistent"])
    print(f"同步到不存在节点结果: {result}")

    # 测试4：获取同步状态（无节点）
    print("\n测试4：获取同步状态（无节点）")
    status = config_manager.get_sync_status()
    print(f"同步状态: {status}")

    # 测试5：解决不存在的冲突
    print("\n测试5：解决不存在的冲突")
    result = config_manager.resolve_conflicts(strategy="merge")
    print(f"解决冲突结果: {result}")

    # 测试6：禁用分布式同步时的操作
    print("\n测试6：禁用分布式同步时的操作")
    disabled_config_manager = UnifiedConfigManager(enable_distributed_sync=False)

    # 尝试注册节点
    result = disabled_config_manager.register_sync_node("node1", "192.168.1.100", 8080)
    print(f"禁用状态下注册节点结果: {result}")

    # 尝试同步配置
    result = disabled_config_manager.sync_config_to_nodes()
    print(f"禁用状态下同步配置结果: {result}")

    # 获取状态
    status = disabled_config_manager.get_sync_status()
    print(f"禁用状态下同步状态: {status}")


def main():
    """主函数"""
    print("分布式同步功能演示")
    print("=" * 50)

    try:
        # 基础分布式同步示例
        distributed_sync_basic_example()

        # 冲突解决示例
        distributed_sync_conflict_resolution_example()

        # 回调示例
        distributed_sync_callback_example()

        # 性能示例
        distributed_sync_performance_example()

        # 错误处理示例
        distributed_sync_error_handling_example()

        print("\n" + "=" * 50)
        print("所有分布式同步示例执行完成！")

    except Exception as e:
        print(f"执行示例时发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
