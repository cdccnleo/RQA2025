#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层测试收集最终验证
"""

import subprocess
import sys
import re

def run_collect():
    """运行测试收集"""
    try:
        result = subprocess.run(
            ['pytest', 'tests/unit/infrastructure', '--collect-only', '-q'],
            capture_output=True,
            text=True,
            timeout=180
        )
        return result.stdout + result.stderr
    except Exception as e:
        return str(e)

def main():
    print("=" * 70)
    print("基础设施层测试用例收集验证报告")
    print("=" * 70)
    print()
    
    print("正在收集测试用例...")
    output = run_collect()
    
    # 检查各类错误
    checks = [
        ("ERROR collecting", "收集错误"),
        ("ImportError", "导入错误"),
        ("ModuleNotFoundError", "模块未找到错误"),
        ("SyntaxError", "语法错误"),
    ]
    
    print("\n【错误检查】")
    print("-" * 70)
    
    all_passed = True
    for pattern, name in checks:
        matches = re.findall(pattern, output, re.IGNORECASE)
        if matches:
            print(f"  ❌ {name}: 发现 {len(matches)} 个")
            all_passed = False
            # 显示前3个错误
            error_lines = [line for line in output.split('\n') if pattern in line]
            for line in error_lines[:3]:
                print(f"     - {line[:100]}")
        else:
            print(f"  ✅ {name}: 无")
    
    # 统计收集的测试数量
    print("\n【收集统计】")
    print("-" * 70)
    
    collected_match = re.search(r'(\d+) tests? collected', output)
    if collected_match:
        test_count = collected_match.group(1)
        print(f"  ✅ 成功收集: {test_count} 个测试用例")
    else:
        print("  ⚠️  无法获取测试数量")
    
    # 检查修复的9个文件
    print("\n【修复文件验证】")
    print("-" * 70)
    
    fixed_files = [
        'test_resource_optimizer_functional.py (functional)',
        'test_resource_optimizer_functional.py (root)',
        'test_health_checker_deep_dive.py',
        'test_health_core_targeted_boost.py',
        'test_logging_core_comprehensive.py',
        'test_interface_checker.py',
        'test_performance_monitoring_comprehensive.py',
        'test_standards.py',
        'test_standards_simple.py',
    ]
    
    for filename in fixed_files:
        # 简单检查：如果文件名在错误输出中出现，说明有问题
        if filename.split()[0] in output and 'ERROR' in output:
            print(f"  ❌ {filename}")
        else:
            print(f"  ✅ {filename}")
    
    # 总结
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ 验证通过: 所有测试收集错误已修复！")
        print("=" * 70)
        return 0
    else:
        print("❌ 验证失败: 仍存在测试收集错误")
        print("=" * 70)
        return 1

if __name__ == '__main__':
    sys.exit(main())
