#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块导入诊断脚本

诊断并修复Python模块导入问题
"""

import sys
import importlib
import traceback
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def check_python_path():
    """检查Python路径"""
    print("🔍 检查Python路径")
    print("=" * 50)

    print(f"项目根目录: {project_root}")
    print(f"项目根目录是否存在: {project_root.exists()}")

    print("\nPython路径:")
    for i, path in enumerate(sys.path):
        print("2d")

    print(f"\n项目根目录在Python路径中: {str(project_root) in sys.path}")


def check_package_structure():
    """检查包结构"""
    print("\n📦 检查包结构")
    print("=" * 50)

    packages = [
        'src.core',
        'src.infrastructure',
        'src.data',
        'src.gateway',
        'src.features',
        'src.ml',
        'src.backtest',
        'src.risk',
        'src.trading',
        'src.engine'
    ]

    for package in packages:
        package_path = project_root / package.replace('.', '/')
        init_file = package_path / '__init__.py'

        exists = package_path.exists()
        has_init = init_file.exists()

        status = "✅" if exists and has_init else "❌" if exists else "🚫"

        print(f"{status} {package}")
        print(f"   目录: {package_path} ({exists})")
        print(f"   __init__.py: {init_file} ({has_init})")

        if exists:
            py_files = list(package_path.glob("*.py"))
            print(f"   Python文件: {len(py_files)} 个")

        print()


def diagnose_import_issues():
    """诊断导入问题"""
    print("\n🔧 诊断导入问题")
    print("=" * 50)

    modules_to_test = [
        'src.core',
        'src.infrastructure',
        'src.data',
        'src.gateway',
        'src.features',
        'src.ml',
        'src.backtest',
        'src.risk',
        'src.trading',
        'src.engine'
    ]

    results = {}

    for module in modules_to_test:
        print(f"\n📦 测试导入: {module}")

        try:
            imported_module = importlib.import_module(module)
            print(f"   ✅ 导入成功: {getattr(imported_module, '__file__', 'unknown')}")

            # 检查模块内容
            if hasattr(imported_module, '__all__'):
                exported_items = len(imported_module.__all__)
                print(f"   📋 导出项: {exported_items} 个")
            else:
                print("   ⚠️  无__all__定义")

            results[module] = {"status": "success", "module": imported_module}

        except ImportError as e:
            print(f"   ❌ ImportError: {e}")
            results[module] = {"status": "import_error", "error": str(e)}

        except SyntaxError as e:
            print(f"   ❌ SyntaxError: {e}")
            results[module] = {"status": "syntax_error", "error": str(e)}

        except Exception as e:
            print(f"   ❌ 其他错误: {e}")
            results[module] = {"status": "other_error", "error": str(e)}

    return results


def test_specific_imports():
    """测试特定导入"""
    print("\n🎯 测试特定导入")
    print("=" * 50)

    specific_imports = [
        ('src.core.event_bus', 'EventBus'),
        ('src.core.container', 'DependencyContainer'),
        ('src.infrastructure.config', 'UnifiedConfigManager'),
        ('src.data.adapters', 'BaseDataAdapter'),
        ('src.gateway.api_gateway', 'APIGateway'),
        ('src.features', 'FeatureProcessor'),
        ('src.ml', 'MLModel'),
        ('src.backtest', 'BacktestEngine'),
        ('src.risk', 'RiskManager'),
        ('src.trading', 'TradingEngine'),
        ('src.engine', 'Engine')
    ]

    for module_name, class_name in specific_imports:
        print(f"\n🔍 测试: {module_name}.{class_name}")

        try:
            module = importlib.import_module(module_name)
            if hasattr(module, class_name):
                cls = getattr(module, class_name)
                print(f"   ✅ 找到类: {cls}")
            else:
                print(f"   ❌ 未找到类: {class_name}")
                # 列出模块中的可用属性
                attrs = [attr for attr in dir(module) if not attr.startswith('_')]
                if attrs:
                    print(f"   📋 可用属性: {attrs[:10]}...")

        except ImportError as e:
            print(f"   ❌ ImportError: {e}")

        except Exception as e:
            print(f"   ❌ 其他错误: {e}")


def check_and_fix_init_files():
    """检查并修复__init__.py文件"""
    print("\n🔧 检查并修复__init__.py文件")
    print("=" * 50)

    packages = [
        'src.core',
        'src.infrastructure',
        'src.data',
        'src.gateway',
        'src.features',
        'src.ml',
        'src.backtest',
        'src.risk',
        'src.trading',
        'src.engine'
    ]

    for package in packages:
        package_path = project_root / package.replace('.', '/')
        init_file = package_path / '__init__.py'

        if init_file.exists():
            print(f"\n📦 检查: {package}")

            try:
                # 读取文件内容
                with open(init_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 检查是否有明显的导入问题
                lines = content.split('\n')
                import_lines = [line for line in lines if 'import ' in line or 'from ' in line]

                if import_lines:
                    print(f"   📋 导入语句: {len(import_lines)} 行")

                    # 检查是否有try-except块
                    try_count = content.count('try:')
                    except_count = content.count('except ImportError:')

                    if try_count > 0:
                        print(f"   🛡️  错误处理: {try_count} 个try块, {except_count} 个except块")
                    else:
                        print("   ⚠️  无错误处理")

                # 测试导入
                try:
                    importlib.import_module(package)
                    print("   ✅ 导入测试通过")
                except Exception as e:
                    print(f"   ❌ 导入测试失败: {e}")

            except Exception as e:
                print(f"   ❌ 文件读取错误: {e}")


def create_minimal_init_files():
    """创建最小的__init__.py文件"""
    print("\n📝 创建最小__init__.py文件")
    print("=" * 50)

    packages = [
        'src.core',
        'src.infrastructure',
        'src.data',
        'src.gateway',
        'src.features',
        'src.ml',
        'src.backtest',
        'src.risk',
        'src.trading',
        'src.engine'
    ]

    for package in packages:
        package_path = project_root / package.replace('.', '/')
        init_file = package_path / '__init__.py'

        if not init_file.exists():
            print(f"📦 创建: {package}")

            # 创建最小的__init__.py文件
            minimal_content = f'''"""
{package} 模块

自动创建的最小初始化文件
"""

__version__ = "1.0.0"
__description__ = "{package} 模块"
'''

            try:
                with open(init_file, 'w', encoding='utf-8') as f:
                    f.write(minimal_content)
                print(f"   ✅ 创建成功: {init_file}")
            except Exception as e:
                print(f"   ❌ 创建失败: {e}")


def main():
    """主函数"""
    print("🚀 RQA2025 模块导入诊断")
    print("=" * 60)

    try:
        # 1. 检查Python路径
        check_python_path()

        # 2. 检查包结构
        check_package_structure()

        # 3. 诊断导入问题
        import_results = diagnose_import_issues()

        # 4. 测试特定导入
        test_specific_imports()

        # 5. 检查并修复__init__.py文件
        check_and_fix_init_files()

        # 6. 如果需要，创建最小__init__.py文件
        create_minimal_init_files()

        print("\n" + "=" * 60)
        print("✅ 诊断完成!")

        # 统计结果
        success_count = sum(1 for r in import_results.values() if r['status'] == 'success')
        total_count = len(import_results)

        print("\n📊 统计结果:")
        print(f"   总模块数: {total_count}")
        print(f"   成功导入: {success_count}")
        if total_count > 0:
            print(f"   成功率: {success_count/total_count:.1%}")
        if success_count < total_count:
            print("   ⚠️  存在导入问题，需要进一步修复")
        else:
            print("   ✅ 所有模块导入正常")

        return 0

    except Exception as e:
        print(f"❌ 诊断过程中发生错误: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
