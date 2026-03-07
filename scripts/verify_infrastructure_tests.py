#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证基础设施层测试修复
"""

import subprocess
import sys

def check_test_file(filepath, check_type="ERROR"):
    """检查测试文件是否有错误"""
    try:
        result = subprocess.run(
            ['pytest', filepath, '--collect-only', '-q'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        has_error = check_type in result.stdout or check_type in result.stderr
        return not has_error, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "超时"
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 60)
    print("基础设施层测试修复验证")
    print("=" * 60)
    print()
    
    test_cases = [
        ("语法错误修复", [
            ('tests/unit/infrastructure/functional/test_resource_optimizer_functional.py', 'SyntaxError'),
            ('tests/unit/infrastructure/test_resource_optimizer_functional.py', 'SyntaxError'),
        ]),
        ("导入错误修复 - Health模块", [
            ('tests/unit/infrastructure/health/test_health_checker_deep_dive.py', 'ImportError'),
            ('tests/unit/infrastructure/health/test_health_core_targeted_boost.py', 'ImportError'),
        ]),
        ("导入错误修复 - Logging模块", [
            ('tests/unit/infrastructure/logging/test_logging_core_comprehensive.py', 'ImportError'),
            ('tests/unit/infrastructure/logging/test_interface_checker.py', 'ImportError'),
        ]),
        ("模块安装 - msgpack", [
            ('tests/unit/infrastructure/logging/test_standards.py', 'ModuleNotFoundError'),
            ('tests/unit/infrastructure/logging/test_standards_simple.py', 'ModuleNotFoundError'),
        ]),
        ("导入错误修复 - Cache模块", [
            ('tests/unit/infrastructure/cache/test_performance_monitoring_comprehensive.py', 'ImportError'),
        ]),
    ]
    
    total_files = 0
    passed_files = 0
    
    for category, files in test_cases:
        print(f"\n【{category}】")
        print("-" * 60)
        for filepath, error_type in files:
            total_files += 1
            passed, output = check_test_file(filepath, error_type)
            status = "✅ 通过" if passed else "❌ 失败"
            filename = filepath.split('/')[-1]
            print(f"  {status}  {filename}")
            
            if passed:
                passed_files += 1
            else:
                # 显示错误详情
                error_lines = [line for line in output.split('\n') if error_type in line]
                if error_lines:
                    print(f"         错误: {error_lines[0][:80]}")
    
    print()
    print("=" * 60)
    print(f"验证结果: {passed_files}/{total_files} 个文件通过")
    print("=" * 60)
    
    if passed_files == total_files:
        print("\n✅ 所有测试文件修复成功！")
        return 0
    else:
        print(f"\n❌ 还有 {total_files - passed_files} 个文件存在问题")
        return 1

if __name__ == '__main__':
    sys.exit(main())

