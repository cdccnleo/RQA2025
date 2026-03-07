#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一配置热重载功能演示

展示如何使用 UnifiedHotReload 来监控配置文件变更并自动重载。
"""

import time
import json
import tempfile
from pathlib import Path
from src.infrastructure.core.config.services.unified_hot_reload import UnifiedHotReload


def on_config_change(file_path: str, new_config: dict):
    """配置文件变更回调函数"""
    print(f"🔄 配置文件已更新: {file_path}")
    print(f"📊 新配置内容: {json.dumps(new_config, indent=2, ensure_ascii=False)}")
    print("-" * 50)


def main():
    """主演示函数"""
    print("🚀 统一配置热重载功能演示")
    print("=" * 50)

    # 创建临时配置文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        initial_config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "username": "demo_user"
            },
            "api": {
                "url": "https://api.demo.com",
                "timeout": 30
            },
            "logging": {
                "level": "INFO",
                "format": "json"
            }
        }
        json.dump(initial_config, f, indent=2, ensure_ascii=False)
        config_file = f.name

    print(f"📁 创建临时配置文件: {config_file}")
    print(f"📋 初始配置: {json.dumps(initial_config, indent=2, ensure_ascii=False)}")
    print()

    try:
        # 创建热重载实例
        hot_reload = UnifiedHotReload(enable_hot_reload=True)

        # 启动热重载服务
        print("🔄 启动热重载服务...")
        if hot_reload.start_hot_reload():
            print("✅ 热重载服务启动成功")
        else:
            print("❌ 热重载服务启动失败")
            return

        # 监视配置文件
        print(f"👀 开始监视配置文件: {config_file}")
        if hot_reload.watch_file(config_file, on_config_change):
            print("✅ 文件监视启动成功")
        else:
            print("❌ 文件监视启动失败")
            return

        # 显示状态
        status = hot_reload.get_hot_reload_status()
        print(f"📊 热重载状态: {json.dumps(status, indent=2, ensure_ascii=False)}")
        print()

        # 模拟配置文件变更
        print("🔄 模拟配置文件变更...")
        time.sleep(2)

        # 更新配置文件
        updated_config = {
            "database": {
                "host": "prod-server.com",
                "port": 5432,
                "username": "prod_user"
            },
            "api": {
                "url": "https://api.prod.com",
                "timeout": 60
            },
            "logging": {
                "level": "DEBUG",
                "format": "json"
            },
            "new_feature": {
                "enabled": True,
                "version": "2.0.0"
            }
        }

        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(updated_config, f, indent=2, ensure_ascii=False)

        print("✅ 配置文件已更新")
        print("⏳ 等待热重载处理...")

        # 等待热重载处理
        time.sleep(3)

        # 再次更新配置文件
        print("🔄 再次更新配置文件...")
        time.sleep(2)

        final_config = {
            "database": {
                "host": "final-server.com",
                "port": 5432,
                "username": "final_user"
            },
            "api": {
                "url": "https://api.final.com",
                "timeout": 120
            },
            "logging": {
                "level": "WARNING",
                "format": "text"
            },
            "final_feature": {
                "enabled": False,
                "reason": "testing_complete"
            }
        }

        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(final_config, f, indent=2, ensure_ascii=False)

        print("✅ 最终配置文件已更新")
        print("⏳ 等待最终热重载处理...")

        # 等待最终处理
        time.sleep(3)

        # 显示最终状态
        final_status = hot_reload.get_hot_reload_status()
        print(f"📊 最终热重载状态: {json.dumps(final_status, indent=2, ensure_ascii=False)}")

        # 重新加载所有监视的文件
        print("🔄 手动重新加载所有监视的文件...")
        results = hot_reload.reload_all_watched_files()
        print(f"📋 重新加载结果: {json.dumps(results, indent=2, ensure_ascii=False)}")

    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")

    finally:
        # 清理资源
        print("\n🧹 清理资源...")
        if 'hot_reload' in locals():
            hot_reload.stop_hot_reload()
            hot_reload.cleanup()
            print("✅ 热重载服务已停止并清理")

        # 删除临时配置文件
        try:
            Path(config_file).unlink()
            print("✅ 临时配置文件已删除")
        except Exception as e:
            print(f"⚠️ 删除临时配置文件时出现警告: {e}")

        print("\n🎉 演示完成！")


if __name__ == "__main__":
    main()
