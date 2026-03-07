#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试模块导入问题的脚本
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 确保src目录在Python路径中
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


def test_module_import(module_name):
    """测试模块导入"""
    print(f"\n🧪 测试 {module_name} 模块导入")
    print("-" * 50)

    try:
        # 直接导入模块
        module = __import__(module_name, fromlist=[''])
        print(f"✅ 直接导入成功: {module}")

        # 检查模块是否有__init__.py
        module_path = module_name.replace('.', '/')
        init_file = src_dir / f"{module_path}/__init__.py"
        if init_file.exists():
            print(f"✅ __init__.py文件存在: {init_file}")
        else:
            print(f"❌ __init__.py文件不存在: {init_file}")

        # 检查模块的基本属性
        print(f"   📁 模块路径: {getattr(module, '__file__', 'N/A')}")
        print(f"   📝 模块描述: {getattr(module, '__doc__', 'N/A')[:100]}...")

        return True

    except ImportError as e:
        print(f"❌ 直接导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 导入异常: {e}")
        return False


def test_layer_validators():
    """测试各层验证器"""
    print("\n🏗️ 测试各层验证器")
    print("=" * 60)

    layers = {
        'core': 'src.core',
        'infrastructure': 'src.infrastructure',
        'data': 'src.data',
        'gateway': 'src.gateway',
        'features': 'src.features',
        'ml': 'src.ml',
        'backtest': 'src.backtest',
        'risk': 'src.risk',
        'trading': 'src.trading',
        'engine': 'src.engine'
    }

    results = {}

    for layer_name, module_name in layers.items():
        print(f"\n🔍 测试 {layer_name} 层 ({module_name})")
        print("-" * 40)

        success = test_module_import(module_name)
        results[layer_name] = success

        if success:
            # 尝试导入一些常见的类
            try:
                module = __import__(module_name, fromlist=[''])
                attrs = [attr for attr in dir(module) if not attr.startswith('_')]
                print(f"   📋 可用属性: {attrs[:10]}..." if len(attrs) > 10 else f"   📋 可用属性: {attrs}")
            except Exception as e:
                print(f"   ❌ 获取属性失败: {e}")

    print(f"\n📊 总结:")
    print(f"   总层数: {len(layers)}")
    passed = sum(1 for success in results.values() if success)
    print(f"   通过: {passed}")
    print(f"   失败: {len(layers) - passed}")
    print(f"   成功率: {passed/len(layers)*100:.1f}%")

    return results


if __name__ == "__main__":
    print("🚀 RQA2025 模块导入调试")
    print("=" * 60)

    results = test_layer_validators()

    print(f"\n🎯 调试完成")
    print("建议:")
    for layer, success in results.items():
        if not success:
            print(f"   • 检查 src.{layer} 模块的 __init__.py 文件")
            print(f"   • 验证模块路径和文件是否存在")
            print(f"   • 检查是否有循环导入问题")
