#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理界面演示脚本

演示特征层配置管理界面的功能，包括：
- 配置实时预览和编辑
- 配置变更历史记录
- 配置验证和回滚
- 多作用域配置管理
"""

import sys
import time
import threading
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def demo_config_management():
    """演示配置管理功能"""
    try:
        print("🚀 启动配置管理界面演示...")

        # 导入配置管理界面
        from src.features.config_management_interface import ConfigManagementInterface
        from src.features.config_integration import ConfigScope

        # 创建配置管理界面
        interface = ConfigManagementInterface()

        # 演示配置变更
        def demo_config_changes():
            """演示配置变更"""
            time.sleep(3)  # 等待界面启动

            config_manager = interface.config_manager

            print("📝 演示配置变更...")

            # 演示技术指标配置变更
            print("  - 变更RSI周期从14到16")
            config_manager.set_config(ConfigScope.TECHNICAL, "rsi_period", 16)
            config_manager.notify_config_change(ConfigScope.TECHNICAL, "rsi_period", 14, 16)

            time.sleep(2)

            # 演示处理配置变更
            print("  - 变更最大工作线程从4到6")
            config_manager.set_config(ConfigScope.PROCESSING, "max_workers", 6)
            config_manager.notify_config_change(ConfigScope.PROCESSING, "max_workers", 4, 6)

            time.sleep(2)

            # 演示监控配置变更
            print("  - 变更监控级别为detailed")
            config_manager.set_config(ConfigScope.MONITORING, "monitoring_level", "detailed")
            config_manager.notify_config_change(
                ConfigScope.MONITORING, "monitoring_level", "standard", "detailed")

            print("✅ 配置变更演示完成")

        # 启动演示线程
        demo_thread = threading.Thread(target=demo_config_changes, daemon=True)
        demo_thread.start()

        # 运行界面
        interface.run()

    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保已安装所需依赖包")
        return 1
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


def demo_config_integration():
    """演示配置集成功能"""
    try:
        print("🔧 演示配置集成功能...")

        from src.features.config_integration import get_config_integration_manager, ConfigScope

        # 获取配置管理器
        config_manager = get_config_integration_manager()

        # 演示配置获取
        print("  - 获取全局配置")
        global_config = config_manager.get_config(ConfigScope.GLOBAL)
        print(f"    全局配置: {global_config}")

        # 演示配置设置
        print("  - 设置测试配置")
        config_manager.set_config(ConfigScope.GLOBAL, "test_key", "test_value")

        # 演示配置变更通知
        print("  - 模拟配置变更通知")
        config_manager.notify_config_change(ConfigScope.GLOBAL, "test_key", None, "test_value")

        # 演示配置摘要
        print("  - 获取配置摘要")
        summary = config_manager.get_config_summary()
        print(f"    配置摘要: {len(summary)} 个作用域")

        print("✅ 配置集成演示完成")

    except Exception as e:
        print(f"❌ 配置集成演示失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


def main():
    """主函数"""
    print("🎯 特征层配置管理界面演示")
    print("=" * 50)

    # 演示配置集成
    if demo_config_integration() != 0:
        return 1

    print("\n" + "=" * 50)

    # 演示配置管理界面
    if demo_config_management() != 0:
        return 1

    print("\n✅ 所有演示完成！")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
