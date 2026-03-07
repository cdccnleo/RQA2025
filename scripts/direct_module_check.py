#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 直接模块检查脚本

直接检查项目中实际存在的模块和包
"""

import sys
import importlib
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def check_package_exists(package_name):
    """检查包是否存在"""
    try:
        package_path = project_root / package_name.replace('.', '/')
        if package_path.exists() and (package_path / '__init__.py').exists():
            return True, f"包存在: {package_path}"
        return False, f"包不存在或缺少__init__.py: {package_path}"
    except Exception as e:
        return False, f"检查异常: {e}"


def check_module_import(module_name):
    """检查模块是否可以导入"""
    try:
        module = importlib.import_module(module_name)
        return True, f"模块导入成功: {module.__file__ if hasattr(module, '__file__') else 'built-in'}"
    except ImportError as e:
        return False, f"导入失败: {e}"
    except Exception as e:
        return False, f"其他错误: {e}"


def scan_src_directory():
    """扫描src目录结构"""
    print("🔍 扫描src目录结构")
    print("=" * 50)

    src_path = project_root / "src"
    if not src_path.exists():
        print("❌ src目录不存在")
        return

    packages = []
    for item in src_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            init_file = item / '__init__.py'
            if init_file.exists():
                packages.append(item.name)
                print(f"📦 发现包: {item.name}")
            else:
                print(f"📁 目录(无__init__.py): {item.name}")

    print(f"\n✅ 发现 {len(packages)} 个Python包")
    return packages


def test_layer_imports():
    """测试各层级导入"""
    print("\n🧪 测试各层级导入")
    print("=" * 50)

    layers = {
        "核心服务层": ["src.core"],
        "基础设施层": ["src.infrastructure"],
        "数据采集层": ["src.data"],
        "API网关层": ["src.gateway"],
        "特征处理层": ["src.features"],
        "模型推理层": ["src.ml"],
        "策略决策层": ["src.backtest"],
        "风控合规层": ["src.risk"],
        "交易执行层": ["src.trading"],
        "监控反馈层": ["src.engine"]
    }

    results = {}

    for layer_name, modules in layers.items():
        print(f"\n🔍 检查 {layer_name}")
        layer_results = []

        for module in modules:
            # 检查包是否存在
            exists, exist_msg = check_package_exists(module)
            if exists:
                print(f"   📦 {module}: 存在")
                # 尝试导入
                import_success, import_msg = check_module_import(module)
                if import_success:
                    print(f"      ✅ 导入成功")
                    layer_results.append({"module": module, "exists": True, "importable": True})
                else:
                    print(f"      ❌ 导入失败: {import_msg}")
                    layer_results.append({"module": module, "exists": True,
                                         "importable": False, "error": import_msg})
            else:
                print(f"   ❌ {module}: {exist_msg}")
                layer_results.append({"module": module, "exists": False, "error": exist_msg})

        results[layer_name] = layer_results

    return results


def test_key_components():
    """测试关键组件"""
    print("\n🔑 测试关键组件")
    print("=" * 50)

    components = {
        "核心服务": [
            "src.core.event_bus",
            "src.core.container",
            "src.core.business_process_orchestrator"
        ],
        "基础设施": [
            "src.infrastructure.config",
            "src.infrastructure.cache",
            "src.infrastructure.logging"
        ],
        "业务功能": [
            "src.data.adapters",
            "src.features",
            "src.ml.integration"
        ]
    }

    for category, comps in components.items():
        print(f"\n📦 {category}")
        for comp in comps:
            success, msg = check_module_import(comp)
            if success:
                print(f"   ✅ {comp}")
            else:
                print(f"   ❌ {comp}: {msg}")


def generate_summary(results):
    """生成总结"""
    print("\n📊 模块检查总结")
    print("=" * 50)

    total_layers = len(results)
    existing_layers = 0
    importable_layers = 0

    for layer_name, layer_results in results.items():
        exists_count = sum(1 for r in layer_results if r.get("exists", False))
        importable_count = sum(1 for r in layer_results if r.get("importable", False))

        if exists_count > 0:
            existing_layers += 1
        if importable_count > 0:
            importable_layers += 1

        print(f"{layer_name}: 存在 {exists_count}, 可导入 {importable_count}")

    print("\n📈 总体统计:")
    print(f"   总层数: {total_layers}")
    print(f"   存在层数: {existing_layers}")
    print(f"   可导入层数: {importable_layers}")
    if total_layers > 0:
        print(f"   存在率: {existing_layers/total_layers:.1%}")
        print(f"   可导入率: {importable_layers/total_layers:.1%}")


def main():
    """主函数"""
    print("🚀 RQA2025 直接模块检查")
    print("=" * 60)

    try:
        # 扫描目录结构
        packages = scan_src_directory()

        # 测试层级导入
        results = test_layer_imports()

        # 测试关键组件
        test_key_components()

        # 生成总结
        generate_summary(results)

        print("\n" + "=" * 60)
        print("✅ 模块检查完成!")

        return 0

    except Exception as e:
        print(f"❌ 检查过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
