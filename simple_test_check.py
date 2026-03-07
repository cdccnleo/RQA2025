#!/usr/bin/env python3
"""
简单的测试检查脚本
检查测试文件的基本语法和导入
"""

import os
import sys
import ast
import importlib.util

def check_syntax(filepath):
    """检查文件语法"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            source = f.read()
        ast.parse(source)
        return True, "语法正确"
    except SyntaxError as e:
        return False, f"语法错误: {e}"
    except Exception as e:
        return False, f"其他错误: {e}"

def check_imports(filepath):
    """检查导入"""
    try:
        spec = importlib.util.spec_from_file_location("test_module", filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return True, "导入成功"
    except ImportError as e:
        return False, f"导入错误: {e}"
    except Exception as e:
        return False, f"其他错误: {e}"

def main():
    print("🔍 RQA2025测试文件检查")
    print("=" * 40)

    test_files = [
        'tests/unit/core/test_service_factory.py',
        'tests/unit/core/test_data_processor_comprehensive.py',
        'tests/unit/core/foundation/test_unified_exceptions.py',
        'tests/unit/core/integration/test_data_layer_adapter.py'
    ]

    for test_file in test_files:
        print(f"\n📋 检查: {test_file}")

        if not os.path.exists(test_file):
            print("   ❌ 文件不存在")
            continue

        # 检查语法
        syntax_ok, syntax_msg = check_syntax(test_file)
        print(f"   语法检查: {'✅' if syntax_ok else '❌'} {syntax_msg}")

        if syntax_ok:
            # 检查导入
            import_ok, import_msg = check_imports(test_file)
            print(f"   导入检查: {'✅' if import_ok else '❌'} {import_msg}")

    print("\n🎯 检查完成")

if __name__ == "__main__":
    main()
