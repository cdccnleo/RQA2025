#!/usr/bin/env python3
"""
简单修复验证脚本
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

def test_fpga_fix():
    """测试FPGA修复"""
    print("🔧 测试FPGA修复...")
    try:
        from src.features.acceleration.fpga.fpga_manager import FPGAManager
        manager = FPGAManager()

        # 测试get_accelerator方法
        result = manager.get_accelerator()
        print(f"✅ FPGA get_accelerator方法存在，返回: {result is not None}")
        return result is not None
    except Exception as e:
        print(f"❌ FPGA测试失败: {e}")
        return False

def test_infrastructure_singleton():
    """测试基础设施单例"""
    print("🏗️ 测试基础设施单例...")
    try:
        from src.infrastructure.init_infrastructure import Infrastructure

        infra1 = Infrastructure()
        infra2 = Infrastructure()

        is_singleton = infra1 is infra2
        print(f"✅ 基础设施单例模式: {'正常' if is_singleton else '异常'}")
        return is_singleton
    except Exception as e:
        print(f"❌ 基础设施测试失败: {e}")
        return False

def test_event_bus():
    """测试事件总线"""
    print("🔄 测试事件总线...")
    try:
        from src.core.event_bus import get_event_bus
        from src.core.event_bus.types import EventType

        bus = get_event_bus()
        print(f"✅ 全局事件总线获取成功，实例ID: {id(bus)}")

        # 测试订阅者数量
        count = bus.get_subscriber_count(EventType.APPLICATION_STARTUP_COMPLETE)
        print(f"✅ 事件订阅者数量: {count}")
        return True
    except Exception as e:
        print(f"❌ 事件总线测试失败: {e}")
        return False

def main():
    print("🚀 简单修复验证")

    tests = [
        ("FPGA修复", test_fpga_fix),
        ("基础设施单例", test_infrastructure_singleton),
        ("事件总线", test_event_bus)
    ]

    results = []
    for name, test_func in tests:
        print(f"\n--- {name} ---")
        result = test_func()
        results.append((name, result))

    print("\n" + "="*30)
    print("验证结果:")
    all_passed = True
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")
        if not result:
            all_passed = False

    print(f"\n总体: {'✅ 所有测试通过' if all_passed else '❌ 部分测试失败'}")
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)