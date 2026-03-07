"""
配置热重载示例
演示如何使用配置管理模块的热重载功能
"""

import time
import json
import os
import tempfile

from src.infrastructure.config import UnifiedConfigManager, CachePolicy, ConfigScope


def create_test_config_file(file_path: str, config_data: dict):
    """创建测试配置文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)


def hot_reload_basic_example():
    """基础热重载示例"""
    print("=== 基础热重载示例 ===")

    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = os.path.join(temp_dir, "app_config.json")

        # 创建初始配置文件
        initial_config = {
            "app": {
                "name": "量化交易系统",
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

        # 创建启用热重载的配置管理器
        config_manager = UnifiedConfigManager(
            enable_hot_reload=True,
            cache_policy=CachePolicy.LRU,
            cache_size=100
        )

        # 加载初始配置
        config_manager.load(config_file)
        print("加载初始配置...")

        # 获取初始配置
        app_name = config_manager.get("app.name")
        max_positions = config_manager.get("trading.max_positions")
        print(f"初始配置 - 应用名称: {app_name}, 最大持仓: {max_positions}")

        # 启动热重载服务
        config_manager.start_hot_reload()
        print("启动热重载服务...")

        # 监听配置文件
        config_manager.watch_file(config_file)
        print(f"开始监听配置文件: {config_file}")

        # 模拟配置文件变更
        print("\n模拟配置文件变更...")
        time.sleep(2)  # 等待服务启动

        # 修改配置文件
        updated_config = {
            "app": {
                "name": "量化交易系统 v2.0",
                "version": "2.0.0",
                "debug": False
            },
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "trading_db"
            },
            "trading": {
                "max_positions": 20,  # 增加最大持仓
                "risk_limit": 0.03    # 调整风险限制
            }
        }

        create_test_config_file(config_file, updated_config)
        print("配置文件已更新")

        # 等待配置重新加载
        time.sleep(3)

        # 检查配置是否已更新
        new_app_name = config_manager.get("app.name")
        new_max_positions = config_manager.get("trading.max_positions")
        new_risk_limit = config_manager.get("trading.risk_limit")

        print(f"更新后配置 - 应用名称: {new_app_name}, 最大持仓: {new_max_positions}, 风险限制: {new_risk_limit}")

        # 检查热重载状态
        status = config_manager.get_hot_reload_status()
        print(f"热重载状态: {status}")

        # 停止热重载服务
        config_manager.stop_hot_reload()
        print("停止热重载服务")


def hot_reload_directory_example():
    """目录监听热重载示例"""
    print("\n=== 目录监听热重载示例 ===")

    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        config_dir = os.path.join(temp_dir, "configs")
        os.makedirs(config_dir, exist_ok=True)

        # 创建多个配置文件
        configs = {
            "database.json": {
                "host": "localhost",
                "port": 5432,
                "name": "trading_db"
            },
            "trading.json": {
                "max_positions": 10,
                "risk_limit": 0.02,
                "strategy": "momentum"
            },
            "logging.json": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }

        for filename, config_data in configs.items():
            file_path = os.path.join(config_dir, filename)
            create_test_config_file(file_path, config_data)
            print(f"创建配置文件: {file_path}")

        # 创建启用热重载的配置管理器
        config_manager = UnifiedConfigManager(
            enable_hot_reload=True,
            cache_policy=CachePolicy.TTL,
            cache_size=200
        )

        # 启动热重载服务
        config_manager.start_hot_reload()
        print("启动热重载服务...")

        # 监听整个目录
        config_manager.watch_directory(config_dir, "*.json")
        print(f"开始监听目录: {config_dir}")

        # 加载所有配置文件
        for filename in configs.keys():
            file_path = os.path.join(config_dir, filename)
            config_manager.load(file_path)
            print(f"加载配置文件: {filename}")

        # 获取初始配置
        db_host = config_manager.get("host", scope=ConfigScope.INFRASTRUCTURE)
        max_positions = config_manager.get("max_positions", scope=ConfigScope.TRADING)
        log_level = config_manager.get("level", scope=ConfigScope.GLOBAL)

        print(f"初始配置 - 数据库主机: {db_host}, 最大持仓: {max_positions}, 日志级别: {log_level}")

        # 模拟配置文件变更
        print("\n模拟配置文件变更...")
        time.sleep(2)

        # 更新数据库配置
        updated_db_config = {
            "host": "192.168.1.100",
            "port": 5432,
            "name": "trading_db_prod"
        }
        db_config_file = os.path.join(config_dir, "database.json")
        create_test_config_file(db_config_file, updated_db_config)
        print("数据库配置文件已更新")

        # 更新交易配置
        updated_trading_config = {
            "max_positions": 25,
            "risk_limit": 0.05,
            "strategy": "mean_reversion"
        }
        trading_config_file = os.path.join(config_dir, "trading.json")
        create_test_config_file(trading_config_file, updated_trading_config)
        print("交易配置文件已更新")

        # 等待配置重新加载
        time.sleep(3)

        # 检查配置是否已更新
        new_db_host = config_manager.get("host", scope=ConfigScope.INFRASTRUCTURE)
        new_max_positions = config_manager.get("max_positions", scope=ConfigScope.TRADING)

        print(f"更新后配置 - 数据库主机: {new_db_host}, 最大持仓: {new_max_positions}")

        # 获取热重载状态
        status = config_manager.get_hot_reload_status()
        print(f"热重载状态: {status}")

        # 停止热重载服务
        config_manager.stop_hot_reload()
        print("停止热重载服务")


def hot_reload_performance_example():
    """热重载性能示例"""
    print("\n=== 热重载性能示例 ===")

    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = os.path.join(temp_dir, "performance_config.json")

        # 创建大型配置文件
        large_config = {
            "performance": {
                "cache_size": 10000,
                "max_memory_mb": 1024,
                "thread_pool_size": 8
            },
            "monitoring": {
                "metrics_interval": 5,
                "alert_thresholds": {
                    "cpu_usage": 80,
                    "memory_usage": 85,
                    "response_time": 100
                }
            }
        }

        create_test_config_file(config_file, large_config)
        print(f"创建大型配置文件: {config_file}")

        # 创建启用热重载的配置管理器
        config_manager = UnifiedConfigManager(
            enable_hot_reload=True,
            cache_policy=CachePolicy.LRU,
            cache_size=1000
        )

        # 启动热重载服务
        config_manager.start_hot_reload()
        print("启动热重载服务...")

        # 监听配置文件
        config_manager.watch_file(config_file)
        print(f"开始监听配置文件: {config_file}")

        # 加载配置
        config_manager.load(config_file)
        print("加载配置文件...")

        # 性能测试：频繁变更配置文件
        print("\n开始性能测试：频繁变更配置文件...")

        start_time = time.time()
        change_count = 0

        for i in range(10):
            # 更新配置
            updated_config = {
                "performance": {
                    "cache_size": 10000 + i * 1000,
                    "max_memory_mb": 1024 + i * 100,
                    "thread_pool_size": 8 + i
                },
                "monitoring": {
                    "metrics_interval": 5 + i,
                    "alert_thresholds": {
                        "cpu_usage": 80 + i,
                        "memory_usage": 85 + i,
                        "response_time": 100 + i * 10
                    }
                }
            }

            create_test_config_file(config_file, updated_config)
            change_count += 1

            # 等待配置重新加载
            time.sleep(0.5)

        end_time = time.time()
        duration = end_time - start_time

        print(f"性能测试完成:")
        print(f"- 变更次数: {change_count}")
        print(f"- 总耗时: {duration:.2f}秒")
        print(f"- 平均每次变更耗时: {duration/change_count:.3f}秒")

        # 获取性能指标
        metrics = config_manager.get_performance_metrics()
        cache_stats = config_manager.get_cache_stats()

        print(f"\n性能指标:")
        print(f"- 总请求数: {metrics.get('total_requests', 0)}")
        print(f"- 平均响应时间: {metrics.get('avg_response_time', 0):.6f}秒")
        print(f"- 缓存命中率: {metrics.get('cache_hit_rate', 0):.2%}")
        print(f"- 缓存大小: {cache_stats.get('size', 0)}")

        # 停止热重载服务
        config_manager.stop_hot_reload()
        print("停止热重载服务")


def hot_reload_error_handling_example():
    """热重载错误处理示例"""
    print("\n=== 热重载错误处理示例 ===")

    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = os.path.join(temp_dir, "error_test_config.json")

        # 创建初始配置文件
        initial_config = {
            "app": {
                "name": "测试应用",
                "version": "1.0.0"
            }
        }

        create_test_config_file(config_file, initial_config)
        print(f"创建配置文件: {config_file}")

        # 创建启用热重载的配置管理器
        config_manager = UnifiedConfigManager(
            enable_hot_reload=True,
            cache_policy=CachePolicy.LRU,
            cache_size=100
        )

        # 启动热重载服务
        config_manager.start_hot_reload()
        print("启动热重载服务...")

        # 监听配置文件
        config_manager.watch_file(config_file)
        print(f"开始监听配置文件: {config_file}")

        # 加载初始配置
        config_manager.load(config_file)
        print("加载初始配置...")

        # 测试1：创建无效的JSON文件
        print("\n测试1：创建无效的JSON文件...")
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write('{"invalid": json, "syntax": error}')

        time.sleep(2)

        # 检查配置是否仍然有效
        app_name = config_manager.get("app.name")
        print(f"配置仍然有效: {app_name}")

        # 测试2：创建有效的配置文件
        print("\n测试2：创建有效的配置文件...")
        valid_config = {
            "app": {
                "name": "恢复的应用",
                "version": "2.0.0"
            }
        }
        create_test_config_file(config_file, valid_config)

        time.sleep(2)

        # 检查配置是否已恢复
        new_app_name = config_manager.get("app.name")
        print(f"配置已恢复: {new_app_name}")

        # 测试3：删除配置文件
        print("\n测试3：删除配置文件...")
        os.remove(config_file)

        time.sleep(2)

        # 检查配置是否仍然可用
        final_app_name = config_manager.get("app.name")
        print(f"配置仍然可用: {final_app_name}")

        # 停止热重载服务
        config_manager.stop_hot_reload()
        print("停止热重载服务")


def main():
    """主函数"""
    print("配置热重载功能演示")
    print("=" * 50)

    try:
        # 基础热重载示例
        hot_reload_basic_example()

        # 目录监听热重载示例
        hot_reload_directory_example()

        # 热重载性能示例
        hot_reload_performance_example()

        # 热重载错误处理示例
        hot_reload_error_handling_example()

        print("\n" + "=" * 50)
        print("所有热重载示例执行完成！")

    except Exception as e:
        print(f"执行示例时发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
