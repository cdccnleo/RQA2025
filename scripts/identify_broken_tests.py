#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""识别与源代码不匹配的测试文件"""

import subprocess
import re
from pathlib import Path

# 失败的测试列表（从之前的运行中提取）
FAILED_TESTS = [
    "tests/unit/infrastructure/config/test_core_typed_config_comprehensive.py",
    "tests/unit/infrastructure/config/test_factory.py",
    "tests/unit/infrastructure/api/test_api_documentation_enhancer_refactored.py",
    "tests/unit/infrastructure/cache/test_distributed_cache.py",
    "tests/unit/infrastructure/cache/test_cache_manager_concurrent_enhanced.py",
    "tests/unit/infrastructure/cache/test_monitoring.py",
    "tests/unit/infrastructure/cache/test_cache_components_low_coverage.py",
    "tests/unit/infrastructure/config/test_priority_manager.py",
    "tests/unit/infrastructure/config/test_config_loaders.py",
    "tests/unit/infrastructure/config/test_cloud_enhanced_monitoring.py",
    "tests/unit/infrastructure/config/test_core_priority_manager.py",
    "tests/unit/infrastructure/cache/test_performance_monitoring_comprehensive.py",
    "tests/unit/infrastructure/cache/test_lru_strategy_edge_cases.py",
    "tests/unit/infrastructure/cache/test_mixins.py",
    "tests/unit/infrastructure/config/test_config_core_simple.py",
    "tests/unit/infrastructure/config/test_config_event.py",
    "tests/unit/infrastructure/config/test_config_storage.py",
    "tests/unit/infrastructure/config/test_loaders_cloud.py",
    "tests/unit/infrastructure/cache/test_cache_uncovered_modules.py",
]

def test_file(test_path):
    """测试单个文件是否能通过"""
    try:
        result = subprocess.run(
            f'pytest "{test_path}" -q',
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            encoding='utf-8',
            errors='ignore'
        )
        return result.returncode == 0
    except Exception:
        return False

def main():
    """主函数"""
    print("="*80)
    print("🔍 识别有问题的测试文件")
    print("="*80)
    
    broken_tests = []
    working_tests = []
    
    for test_file in FAILED_TESTS:
        path = Path(test_file)
        if not path.exists():
            print(f"❌ 不存在: {test_file}")
            continue
        
        print(f"\n测试: {test_file}")
        if test_file(test_file):
            print("  ✅ 通过")
            working_tests.append(test_file)
        else:
            print("  ❌ 失败")
            broken_tests.append(test_file)
    
    print("\n" + "="*80)
    print("📊 统计结果")
    print("="*80)
    print(f"总计: {len(FAILED_TESTS)} 个测试文件")
    print(f"失败: {len(broken_tests)} 个")
    print(f"通过: {len(working_tests)} 个")
    
    if broken_tests:
        print("\n" + "="*80)
        print("❌ 失败的测试文件列表：")
        print("="*80)
        for test in broken_tests:
            print(f"  - {test}")
        
        # 保存到文件
        with open('test_logs/broken_tests.txt', 'w', encoding='utf-8') as f:
            for test in broken_tests:
                f.write(f"{test}\n")
        
        print(f"\n📄 失败列表已保存: test_logs/broken_tests.txt")
        print(f"\n建议：删除或修复这些测试文件后重新统计覆盖率")

if __name__ == "__main__":
    main()

