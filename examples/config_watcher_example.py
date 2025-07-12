"""
ConfigWatcher 使用示例

本示例展示了如何使用 ConfigWatcher 监控配置文件变化。
"""

import json
import os
import time
from pathlib import Path

# 导入 ConfigWatcher
from src.infrastructure.config.config_watcher import ConfigWatcher

# 创建示例配置目录和文件
def setup_example_config():
    """设置示例配置文件"""
    config_dir = Path("./example_config")
    config_dir.mkdir(exist_ok=True)

    # 创建默认环境配置
    default_config = {
        "database": {
            "host": "localhost",
            "port": 5432,
            "username": "user",
            "password": "password"
        },
        "api": {
            "url": "https://api.example.com",
            "timeout": 30
        },
        "logging": {
            "level": "INFO",
            "file": "app.log"
        }
    }

    with open(config_dir / "default.json", "w") as f:
        json.dump(default_config, f, indent=2)

    # 创建开发环境配置
    dev_config = {
        "database": {
            "host": "localhost",
            "port": 5432,
            "username": "dev_user",
            "password": "dev_password"
        },
        "api": {
            "url": "https://dev-api.example.com",
            "timeout": 60
        }
    }

    dev_dir = config_dir / "dev"
    dev_dir.mkdir(exist_ok=True)

    with open(dev_dir / "config.json", "w") as f:
        json.dump(dev_config, f, indent=2)

    return config_dir

# 回调函数
def handle_database_change(value):
    """处理数据库配置变更"""
    print(f"数据库配置变更: {value}")

def handle_api_change(value):
    """处理API配置变更"""
    print(f"API配置变更: {value}")

def handle_logging_level_change(value):
    """处理日志级别变更"""
    print(f"日志级别变更为: {value}")

def global_config_change(file_path):
    """全局配置变更回调"""
    print(f"配置文件变更: {file_path}")

def main():
    """主函数"""
    # 设置示例配置
    config_dir = setup_example_config()
    print(f"示例配置已创建在: {config_dir.absolute()}")

    # 创建 ConfigWatcher
    watcher = ConfigWatcher(config_dir=str(config_dir))

    # 设置全局配置变更回调
    watcher.set_config_change_callback(global_config_change)

    # 监控特定配置项
    watcher.watch("database", handle_database_change)
    watcher.watch("api.url", handle_api_change)
    watcher.watch("logging.level", handle_logging_level_change)

    # 监控开发环境配置
    watcher.watch("database", lambda v: print(f"开发环境数据库配置: {v}"), env="dev")

    # 启动监控
    watcher.start()

    print("配置监控已启动。尝试修改配置文件以触发回调...")
    print(f"默认配置文件: {config_dir / 'default.json'}")
    print(f"开发环境配置文件: {config_dir / 'dev' / 'config.json'}")

    try:
        # 等待用户修改配置文件
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n停止监控...")
        watcher.stop()
        print("监控已停止")

if __name__ == "__main__":
    main()
