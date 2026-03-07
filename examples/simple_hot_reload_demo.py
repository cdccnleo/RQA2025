#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化热重载功能演示

直接导入核心模块，避免循环导入问题。
"""

import sys
import time
import json
import tempfile
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 直接导入核心模块
sys.path.insert(0, str(project_root / "src" / "infrastructure" / "core" / "config" / "services"))

try:
    from unified_hot_reload import UnifiedHotReload
    print("✅ 成功导入 UnifiedHotReload")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("尝试直接导入...")

    # 尝试直接导入
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "unified_hot_reload",
        project_root / "src" / "infrastructure" / "core" / "config" / "services" / "unified_hot_reload.py"
    )
    if spec and spec.loader:
        unified_hot_reload = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(unified_hot_reload)
        UnifiedHotReload = unified_hot_reload.UnifiedHotReload
        print("✅ 通过spec导入成功")
    else:
        print("❌ 无法找到模块文件")
        sys.exit(1)


def on_config_change(file_path: str, new_config: dict):
    """配置文件变更回调函数"""
    print(f"🔄 配置文件已更新: {file_path}")
    print(f"📊 新配置内容: {json.dumps(new_config, indent=2, ensure_ascii=False)}")
    print("-" * 50)


def main():
    """主演示函数"""
    print("🚀 简化热重载功能演示")
    print("=" * 50)

    # 创建临时配置文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        initial_config = {
            "demo": {
                "name": "热重载演示",
                "version": "1.0.0"
            },
            "settings": {
                "enabled": True,
                "timeout": 30
            }
        }
        json.dump(initial_config, f, indent=2, ensure_ascii=False)
        config_file = f.name

    print(f"📁 创建临时配置文件: {config_file}")
    print(f"📋 初始配置: {json.dumps(initial_config, indent=2, ensure_ascii=False)}")
    print()

    try:
        # 创建热重载实例
        print("🔧 创建热重载实例...")
        hot_reload = UnifiedHotReload(enable_hot_reload=True)
        print("✅ 热重载实例创建成功")

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
            "demo": {
                "name": "热重载演示 - 已更新",
                "version": "2.0.0"
            },
            "settings": {
                "enabled": False,
                "timeout": 60
            },
            "new_feature": {
                "enabled": True,
                "description": "新增功能"
            }
        }

        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(updated_config, f, indent=2, ensure_ascii=False)

        print("✅ 配置文件已更新")
        print("⏳ 等待热重载处理...")

        # 等待热重载处理
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
        import traceback
        traceback.print_exc()

    finally:
        # 清理资源
        print("\n🧹 清理资源...")
        if 'hot_reload' in locals():
            try:
                hot_reload.stop_hot_reload()
                hot_reload.cleanup()
                print("✅ 热重载服务已停止并清理")
            except Exception as e:
                print(f"⚠️ 清理时出现警告: {e}")

        # 删除临时配置文件
        try:
            Path(config_file).unlink()
            print("✅ 临时配置文件已删除")
        except Exception as e:
            print(f"⚠️ 删除临时配置文件时出现警告: {e}")

        print("\n🎉 演示完成！")


if __name__ == "__main__":
    main()
