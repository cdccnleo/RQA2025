#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 业务流程测试检查脚本

检查业务流程测试文件是否存在和基本结构
"""

from pathlib import Path


def check_business_process_tests():
    """检查业务流程测试文件"""
    print("=== RQA2025 业务流程测试检查 ===")

    project_root = Path(__file__).parent.parent

    # 定义预期的测试文件
    expected_test_files = [
        "tests/business_process/test_strategy_management_flow.py",
        "tests/business_process/test_portfolio_management_flow.py",
        "tests/business_process/test_user_service_management_flow.py",
        "tests/business_process/test_system_monitoring_flow.py"
    ]

    total_files = len(expected_test_files)
    existing_files = 0

    print(f"\n检查 {total_files} 个业务流程测试文件:")
    print("-" * 50)

    for test_file in expected_test_files:
        test_path = project_root / test_file
        if test_path.exists():
            print(f"✅ {test_file} - 存在")
            existing_files += 1

            # 检查文件大小
            size = test_path.stat().st_size
            print(f"   文件大小: {size} bytes")

            # 检查是否包含测试类
            with open(test_path, 'r', encoding='utf-8') as f:
                content = f.read()
                test_classes = content.count('class Test')
                test_methods = content.count('def test_')
                print(f"   测试类: {test_classes}")
                print(f"   测试方法: {test_methods}")

        else:
            print(f"❌ {test_file} - 缺失")

    print("\n总结:")
    print(f"  总文件数: {total_files}")
    print(f"  存在文件数: {existing_files}")
    print(f"  缺失文件数: {total_files - existing_files}")
    print(f"  覆盖率: {existing_files/total_files*100:.1f}%")
    if existing_files == total_files:
        print("🎉 所有业务流程测试文件都已存在!")
        return 0
    else:
        print("⚠️  部分业务流程测试文件缺失")
        return 1


if __name__ == "__main__":
    exit(check_business_process_tests())
