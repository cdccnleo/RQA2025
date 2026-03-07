#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logger配置使用示例

演示如何使用灵活的Logger配置系统，包括文件配置、环境变量配置和热重载。
"""

from infrastructure.logging.config import (
    EnvironmentLoggerConfig,
    HotReloadLoggerConfig,
    create_config_manager,
    create_default_config_file,
    LoggerConfig
)
import time
import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def demo_file_config():
    """演示文件配置"""
    print("=== 文件配置演示 ===")

    # 创建配置管理器
    config_file = "examples/logging/config/logger_config.yaml"
    manager = create_config_manager(config_file=config_file, auto_reload=False)

    print(f"已加载配置: {config_file}")
    config = manager.get_config()
    print(f"版本: {config.version}")
    print(f"默认日志级别: {config.manager.default_level}")
    print(f"对象池大小: {config.pool.max_size}")
    print(f"处理器数量: {len(config.handlers)}")
    print(f"专用Logger数量: {len(config.loggers)}")
    print()


def demo_env_config():
    """演示环境变量配置"""
    print("=== 环境变量配置演示 ===")

    # 创建环境配置处理器
    env_config = EnvironmentLoggerConfig()

    # 加载环境变量配置
    config_dict = env_config.load_from_env()
    print(f"从环境变量加载了 {len(config_dict)} 个配置项")

    # 打印环境变量帮助
    print("\n环境变量配置帮助:")
    print("=" * 50)
    env_config.print_env_help()
    print("=" * 50)
    print()


def demo_hot_reload():
    """演示热重载配置"""
    print("=== 热重载配置演示 ===")

    # 创建配置管理器
    config_file = "examples/logging/config/logger_config.yaml"
    manager = create_config_manager(config_file=config_file, auto_reload=False)

    # 创建热重载处理器
    hot_reload = HotReloadLoggerConfig(manager, check_interval=2.0)

    # 添加重载回调
    def on_config_reload(new_config):
        print(f"配置已重载! 新版本: {new_config.version}")
        print(f"当前池大小: {new_config.pool.max_size}")

    hot_reload.add_reload_callback(on_config_reload)

    # 启动热重载
    with hot_reload:
        print("热重载监控已启动...")
        print("请修改配置文件 logger_config.yaml 中的版本号，然后保存")
        print("程序将在2秒内检测到变化并自动重载配置")
        print("按Ctrl+C退出...")

        try:
            while True:
                time.sleep(1)
                stats = hot_reload.get_reload_stats()
                print(f"监控状态: 运行中={stats['is_running']}, 重载次数={stats['reload_count']}")
        except KeyboardInterrupt:
            print("\n停止热重载监控")

    print()


def demo_config_creation():
    """演示配置创建"""
    print("=== 配置创建演示 ===")

    # 创建默认YAML配置
    yaml_file = "examples/logging/config/generated_config.yaml"
    create_default_config_file(yaml_file, "yaml")
    print(f"已创建默认YAML配置: {yaml_file}")

    # 创建默认JSON配置
    json_file = "examples/logging/config/generated_config.json"
    create_default_config_file(json_file, "json")
    print(f"已创建默认JSON配置: {json_file}")
    print()


def demo_logger_with_config():
    """演示使用配置的Logger"""
    print("=== 使用配置的Logger演示 ===")

    # 加载配置
    config_file = "examples/logging/config/logger_config.yaml"
    manager = create_config_manager(config_file=config_file, auto_reload=False)

    # 使用配置创建Logger
    # 注意：这里需要扩展实际的Logger类以支持配置注入
    # 目前只是演示配置的加载和使用

    print("配置加载完成，可以在此基础上创建Logger实例")
    print("实际项目中，Logger类会自动从配置管理器获取配置")
    print()


def demo_config_validation():
    """演示配置验证"""
    print("=== 配置验证演示 ===")

    from infrastructure.logging.config import LoggerConfigValidator

    validator = LoggerConfigValidator()

    # 创建一个有效的配置
    config = LoggerConfig()

    # 验证配置
    is_valid = validator.validate(config)

    print(f"配置验证结果: {'通过' if is_valid else '失败'}")
    if validator.get_errors():
        print("错误:")
        for error in validator.get_errors():
            print(f"  - {error}")

    if validator.get_warnings():
        print("警告:")
        for warning in validator.get_warnings():
            print(f"  - {warning}")

    print()


def main():
    """主函数"""
    print("RQA2025 Logger配置系统使用示例")
    print("=" * 50)

    try:
        # 演示各种配置功能
        demo_file_config()
        demo_env_config()
        demo_config_creation()
        demo_config_validation()
        demo_logger_with_config()

        # 热重载演示（需要手动操作）
        print("热重载演示需要手动修改配置文件，暂时跳过")
        print("如需演示热重载，请取消注释下面的代码:")
        print("# demo_hot_reload()")

        # demo_hot_reload()  # 取消注释以运行热重载演示

    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
