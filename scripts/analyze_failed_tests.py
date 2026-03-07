#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析失败测试的模式和原因

功能：
- 收集失败测试列表
- 分析失败模式
- 分类统计
- 生成修复建议
"""

import re
import subprocess
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def run_pytest_collect_failures():
    """运行pytest收集失败测试"""
    cmd = [
        'pytest',
        'tests/unit/infrastructure/utils/',
        '-n', 'auto',
        '--tb=short',
        '-q'
    ]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding='utf-8'
    )
    
    failures = []
    for line in result.stdout.split('\n'):
        if 'FAILED' in line:
            # 提取测试路径
            match = re.search(r'tests[^\s]+::[^\s]+', line)
            if match:
                failures.append(match.group(0))
    
    return failures, result.stdout

def analyze_failure_patterns(failures, full_output):
    """分析失败模式"""
    patterns = defaultdict(list)
    
    # 按文件分组
    file_groups = defaultdict(list)
    for failure in failures:
        file_path = failure.split('::')[0]
        file_groups[file_path].append(failure)
    
    # 识别高频失败文件
    high_freq_files = {
        file: len(tests) 
        for file, tests in file_groups.items() 
        if len(tests) >= 3
    }
    
    # 按错误类型分类（从输出中提取）
    error_patterns = {
        'AttributeError': [],
        'AssertionError': [],
        'TypeError': [],
        'ValueError': [],
        'ImportError': [],
        '其他': []
    }
    
    for line in full_output.split('\n'):
        for error_type in error_patterns.keys():
            if error_type in line and 'FAILED' not in line:
                # 提取相关测试
                for failure in failures:
                    if failure.split('::')[-1] in line or failure in line:
                        if failure not in error_patterns[error_type]:
                            error_patterns[error_type].append(failure)
    
    return {
        'total': len(failures),
        'by_file': dict(file_groups),
        'high_freq_files': high_freq_files,
        'by_error_type': error_patterns
    }

def main():
    """主函数"""
    print("=" * 80)
    print("📊 分析失败测试模式")
    print("=" * 80)
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print("🔍 收集失败测试...")
    failures, full_output = run_pytest_collect_failures()
    
    if not failures:
        print("✅ 没有失败测试！")
        return
    
    print(f"📊 找到 {len(failures)} 个失败测试")
    print()
    
    print("📈 分析失败模式...")
    analysis = analyze_failure_patterns(failures, full_output)
    
    # 输出统计
    print("=" * 80)
    print("📊 失败测试统计")
    print("=" * 80)
    print(f"总失败数: {analysis['total']}")
    print()
    
    # 高频失败文件
    if analysis['high_freq_files']:
        print("🔥 高频失败文件 (≥3个失败):")
        print("-" * 80)
        for file, count in sorted(analysis['high_freq_files'].items(), key=lambda x: x[1], reverse=True):
            print(f"  • {file}: {count}个失败")
        print()
    
    # 按错误类型分类
    print("📋 按错误类型分类:")
    print("-" * 80)
    for error_type, tests in analysis['by_error_type'].items():
        if tests:
            print(f"  • {error_type}: {len(tests)}个")
    print()
    
    # 详细列表
    print("📝 所有失败测试:")
    print("-" * 80)
    for i, failure in enumerate(failures, 1):
        print(f"  {i}. {failure}")
    print()
    
    # 修复建议
    print("💡 修复建议:")
    print("-" * 80)
    if analysis['high_freq_files']:
        top_file = max(analysis['high_freq_files'].items(), key=lambda x: x[1])
        print(f"  1. 优先修复高频失败文件: {top_file[0]} ({top_file[1]}个失败)")
    
    if analysis['by_error_type']['AttributeError']:
        print(f"  2. 修复AttributeError ({len(analysis['by_error_type']['AttributeError'])}个)")
        print("     可能是API变更导致的属性访问问题")
    
    if analysis['by_error_type']['AssertionError']:
        print(f"  3. 修复AssertionError ({len(analysis['by_error_type']['AssertionError'])}个)")
        print("     需要调整断言逻辑")
    
    print()
    print("=" * 80)
    print(f"⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

if __name__ == '__main__':
    main()

