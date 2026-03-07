#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试错误分析脚本
分析Infrastructure/utils模块的测试收集错误
"""

import subprocess
import re
from collections import defaultdict
from pathlib import Path


def run_pytest_collect(test_dir='tests/unit/infrastructure/utils/'):
    """运行pytest收集测试"""
    cmd = ['pytest', test_dir, '--co', '-v']
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    return result.stdout + result.stderr


def analyze_errors(output):
    """分析错误信息"""
    errors = defaultdict(lambda: {'count': 0, 'files': []})
    
    # 错误模式
    patterns = {
        'ModuleNotFoundError': r"ModuleNotFoundError: No module named '([^']+)'",
        'ImportError': r"ImportError: cannot import name '([^']+)'",
        'AttributeError': r"AttributeError: module '([^']+)' has no attribute",
        'SyntaxError': r"SyntaxError:",
        'Other': r"ERROR.*"
    }
    
    current_file = None
    lines = output.split('\n')
    
    for line in lines:
        # 识别错误文件
        if 'ERROR collecting' in line or 'ERROR tests' in line:
            match = re.search(r'tests[/\\].*?\.py', line)
            if match:
                current_file = match.group(0)
        
        # 分析错误类型
        for error_type, pattern in patterns.items():
            if error_type == 'Other':
                continue
            match = re.search(pattern, line)
            if match and current_file:
                if error_type in ['ModuleNotFoundError', 'ImportError']:
                    module_name = match.group(1)
                    key = f"{error_type}: {module_name}"
                else:
                    key = error_type
                
                if current_file not in errors[key]['files']:
                    errors[key]['files'].append(current_file)
                    errors[key]['count'] += 1
    
    return errors


def generate_report(errors):
    """生成错误分析报告"""
    print("=" * 80)
    print("Infrastructure/Utils 测试错误分析报告")
    print("=" * 80)
    print()
    
    # 按数量排序
    sorted_errors = sorted(errors.items(), key=lambda x: x[1]['count'], reverse=True)
    
    total_files = sum(e['count'] for e in errors.values())
    print(f"总错误文件数: {total_files}")
    print(f"错误类型数: {len(errors)}")
    print()
    
    print("错误类型分布:")
    print("-" * 80)
    
    for error_key, error_info in sorted_errors[:10]:  # 显示前10个
        print(f"\n{error_key}")
        print(f"  影响文件数: {error_info['count']}")
        print(f"  示例文件:")
        for file in error_info['files'][:3]:  # 显示前3个文件
            print(f"    - {file}")
    
    print("\n" + "=" * 80)
    
    # 生成修复建议
    print("\n修复建议:")
    print("-" * 80)
    
    for error_key, error_info in sorted_errors:
        if 'environment' in error_key.lower():
            print(f"\n✅ 高优先级: {error_key}")
            print(f"   修复方案: 统一修改为 from .components.environment import")
            print(f"   影响文件: {error_info['count']}个")
        elif 'cache' in error_key.lower():
            print(f"\n✅ 高优先级: {error_key}")
            print(f"   修复方案: 更新cache模块导入路径")
            print(f"   影响文件: {error_info['count']}个")
    
    return sorted_errors


def save_detailed_report(errors, output_file='test_logs/error_analysis_report.txt'):
    """保存详细报告"""
    Path('test_logs').mkdir(exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Infrastructure/Utils 测试错误详细分析报告\n")
        f.write("=" * 80 + "\n\n")
        
        sorted_errors = sorted(errors.items(), key=lambda x: x[1]['count'], reverse=True)
        
        for error_key, error_info in sorted_errors:
            f.write(f"\n错误类型: {error_key}\n")
            f.write(f"影响文件数: {error_info['count']}\n")
            f.write(f"所有文件:\n")
            for file in error_info['files']:
                f.write(f"  - {file}\n")
    
    print(f"\n✅ 详细报告已保存到: {output_file}")


def main():
    print("开始分析Infrastructure/utils测试错误...\n")
    
    # 运行pytest收集
    output = run_pytest_collect()
    
    # 分析错误
    errors = analyze_errors(output)
    
    # 生成报告
    sorted_errors = generate_report(errors)
    
    # 保存详细报告
    save_detailed_report(dict(errors))
    
    print("\n分析完成！")


if __name__ == '__main__':
    main()

