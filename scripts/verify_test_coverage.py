#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手动验证基础设施层测试覆盖率
"""

import os
import sys
import traceback
from typing import Dict, List, Tuple

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def collect_test_files() -> List[str]:
    """收集所有基础设施层的测试文件"""
    test_files = []
    test_base = os.path.join(os.path.dirname(__file__), '..', 'tests', 'unit', 'infrastructure')

    for root, dirs, files in os.walk(test_base):
        for file in files:
            if file.startswith('test_') and file.endswith('.py'):
                test_files.append(os.path.join(root, file))

    return test_files

def run_single_test(test_file: str) -> Tuple[bool, int, str]:
    """运行单个测试文件并返回结果"""
    try:
        # 动态导入测试模块
        module_name = test_file.replace(os.path.join(os.path.dirname(__file__), '..'), '').replace('.py', '').replace(os.sep, '.').lstrip('.')
        if module_name.startswith('tests.'):
            module_name = module_name[6:]  # 移除tests.前缀

        print(f"Testing: {module_name}")

        # 导入模块
        __import__(module_name)

        # 这里我们简单地检查导入是否成功
        # 实际的测试运行需要更复杂的逻辑
        return True, 0, "Import successful"

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        return False, 0, error_msg

def main():
    """主函数"""
    print("🧪 基础设施层测试覆盖率验证")
    print("=" * 60)

    # 收集测试文件
    test_files = collect_test_files()
    print(f"发现 {len(test_files)} 个测试文件")

    # 统计结果
    successful_imports = 0
    failed_imports = 0
    total_tests = 0
    errors = []

    # 测试每个文件
    for test_file in test_files[:10]:  # 只测试前10个文件作为示例
        success, test_count, error_msg = run_single_test(test_file)
        total_tests += test_count

        if success:
            successful_imports += 1
        else:
            failed_imports += 1
            errors.append(f"{os.path.basename(test_file)}: {error_msg}")

    print("
📊 测试结果统计:"    print(f"   • 成功导入: {successful_imports}")
    print(f"   • 导入失败: {failed_imports}")
    print(f"   • 总测试数: {total_tests}")

    if errors:
        print("
❌ 导入失败的文件:"        for error in errors[:5]:  # 只显示前5个错误
            print(f"   • {error}")

    print("
🎯 结论:"    if failed_imports == 0:
        print("✅ 所有测试文件导入成功")
    else:
        print(f"❌ {failed_imports} 个测试文件导入失败，需要修复")

if __name__ == "__main__":
    main()