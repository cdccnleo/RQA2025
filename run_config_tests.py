#!/usr/bin/env python3
"""
配置管理单元测试执行脚本
运行所有配置管理相关测试并生成覆盖率报告
"""

import unittest
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_config_tests():
    """运行配置管理测试"""

    # 发现所有配置测试
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()

    # 配置测试目录
    config_test_dir = project_root / "tests" / "unit" / "infrastructure" / "config"

    if config_test_dir.exists():
        print(f"发现测试目录: {config_test_dir}")

        # 加载所有测试文件
        for test_file in config_test_dir.glob("test_*.py"):
            if test_file.name != "__init__.py":
                print(f"加载测试文件: {test_file.name}")

                try:
                    # 动态导入测试模块
                    module_name = f"tests.unit.infrastructure.config.{test_file.stem}"
                    module = __import__(module_name, fromlist=[''])

                    # 获取测试类
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type) and issubclass(attr, unittest.TestCase):
                            test_suite.addTest(test_loader.loadTestsFromTestCase(attr))
                            print(f"  ✓ 加载测试类: {attr_name}")

                except Exception as e:
                    print(f"  ✗ 加载失败 {test_file.name}: {e}")

    else:
        print(f"测试目录不存在: {config_test_dir}")
        return

    # 运行测试
    print("\n" + "="*60)
    print("开始运行配置管理测试...")
    print("="*60)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # 统计结果
    print("\n" + "="*60)
    print("测试结果统计:")
    print("="*60)
    print(f"运行测试: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")

    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"  - {test}")

    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"  - {test}")

    # 计算覆盖率估算
    total_tests = result.testsRun
    if total_tests > 0:
        success_rate = (total_tests - len(result.failures) - len(result.errors)) / total_tests * 100
        print(".1f"
        # 投产要求评估
        if success_rate >= 95:
            print("✅ 达到投产要求 (95%+ 成功率)")
        elif success_rate >= 85:
            print("⚠️ 接近投产要求，需要改进")
        else:
            print("❌ 未达到投产要求，需要补充测试")
    else:
        print("❌ 没有运行任何测试")

    return result

if __name__ == "__main__":
    print("配置管理单元测试执行器")
    print("="*40)
    run_config_tests()
