#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工作热重载演示脚本

展示实际的文件监控和配置重载功能。
"""

import sys
import os
import time
import json
import tempfile
from pathlib import Path

# 添加src目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, src_dir)


def on_config_change(file_path: str, new_config: dict):
    """配置文件变更回调函数"""
    print(f"🔄 配置文件已更新: {file_path}")
    print(f"📊 新配置内容: {json.dumps(new_config, indent=2, ensure_ascii=False)}")
    print("-" * 50)


def main():
    """主演示函数"""
    print("🚀 工作热重载演示")
    print("=" * 50)

    try:
        # 直接导入核心模块
        sys.path.insert(0, os.path.join(src_dir, 'infrastructure', 'core', 'config', 'services'))
        from hot_reload_service import HotReloadService
        print("✅ 成功导入HotReloadService")

        # 创建临时配置文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            initial_config = {
                "app": {
                    "name": "热重载演示",
                    "version": "1.0.0"
                },
                "database": {
                    "host": "localhost",
                    "port": 5432,
                    "enabled": True
                }
            }
            json.dump(initial_config, f, indent=2, ensure_ascii=False)
            config_file = f.name

        print(f"📁 创建临时配置文件: {config_file}")
        print(f"📋 初始配置: {json.dumps(initial_config, indent=2, ensure_ascii=False)}")
        print()

        # 创建热重载服务
        print("🔧 创建热重载服务...")
        hot_reload = HotReloadService()
        print("✅ 热重载服务创建成功")

        # 启动服务
        print("🔄 启动热重载服务...")
        if hot_reload.start():
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
        status = hot_reload.get_status()
        print(f"📊 服务状态: {json.dumps(status, indent=2, ensure_ascii=False)}")
        print()

        # 模拟配置文件变更
        print("🔄 模拟配置文件变更...")
        time.sleep(2)

        # 更新配置文件
        updated_config = {
            "app": {
                "name": "热重载演示 - 已更新",
                "version": "2.0.0"
            },
            "database": {
                "host": "127.0.0.1",
                "port": 5433,
                "enabled": False
            },
            "new_feature": {
                "enabled": True,
                "description": "新增功能配置"
            }
        }

        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(updated_config, f, indent=2, ensure_ascii=False)

        print("✅ 配置文件已更新")
        print("⏳ 等待热重载处理...")

        # 等待热重载处理
        time.sleep(3)

        # 显示最终状态
        final_status = hot_reload.get_status()
        print(f"📊 最终服务状态: {json.dumps(final_status, indent=2, ensure_ascii=False)}")

        # 重新加载所有监视的文件
        print("🔄 手动重新加载所有监视的文件...")
        results = hot_reload.reload_all()
        print(f"📋 重新加载结果: {json.dumps(results, indent=2, ensure_ascii=False)}")

    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 清理资源
        print("\n🧹 清理资源...")
        if 'hot_reload' in locals():
            try:
                hot_reload.stop()
                print("✅ 热重载服务已停止")
            except Exception as e:
                print(f"⚠️ 停止服务时出现警告: {e}")

        # 删除临时配置文件
        try:
            Path(config_file).unlink()
            print("✅ 临时配置文件已删除")
        except Exception as e:
            print(f"⚠️ 删除临时配置文件时出现警告: {e}")

        print("\n🎉 演示完成！")


if __name__ == "__main__":
    main()
