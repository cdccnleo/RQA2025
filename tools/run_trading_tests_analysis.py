#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行Trading层测试并分析结果
质量优先原则：确保100%测试通过率
"""

import subprocess
import sys
import json
import re
from datetime import datetime
from pathlib import Path

def run_trading_tests():
    """运行Trading层测试并分析结果"""
    print("=" * 70)
    print("  Trading层质量验证 - 测试通过率分析")
    print("=" * 70)
    print()
    
    # 运行测试
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/unit/trading/",
        "-n", "auto",
        "-v",
        "--tb=short",
        "--maxfail=10"
    ]
    
    print("运行测试...")
    print("命令:", " ".join(cmd))
    print()
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='ignore'
    )
    
    # 保存完整输出
    output_dir = Path("test_logs")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "trading_tests_raw_output.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("STDOUT:\n")
        f.write(result.stdout)
        f.write("\n\nSTDERR:\n")
        f.write(result.stderr)
    
    # 分析结果
    output = result.stdout + result.stderr
    lines = output.split('\n')
    
    stats = {
        'passed': 0,
        'failed': 0,
        'error': 0,
        'skipped': 0,
        'warnings': 0,
        'total': 0,
        'failed_tests': [],
        'error_tests': [],
        'timestamp': datetime.now().isoformat(),
        'exit_code': result.returncode
    }
    
    # 解析结果行
    for line in lines:
        # 查找总结行，例如: "100 passed, 5 failed, 2 error, 10 skipped in 10.5s"
        summary_match = re.search(
            r'(\d+)\s+passed.*?(\d+)\s+failed.*?(\d+)\s+error.*?(\d+)\s+skipped',
            line
        )
        if summary_match:
            stats['passed'] = int(summary_match.group(1))
            stats['failed'] = int(summary_match.group(2))
            stats['error'] = int(summary_match.group(3))
            stats['skipped'] = int(summary_match.group(4))
            break
        
        # 查找单独的统计信息
        if 'passed' in line.lower() and re.search(r'\d+\s+passed', line):
            match = re.search(r'(\d+)\s+passed', line)
            if match:
                stats['passed'] = int(match.group(1))
        
        if 'failed' in line.lower() and re.search(r'\d+\s+failed', line):
            match = re.search(r'(\d+)\s+failed', line)
            if match:
                stats['failed'] = int(match.group(1))
        
        if 'error' in line.lower() and re.search(r'\d+\s+error', line):
            match = re.search(r'(\d+)\s+error', line)
            if match:
                stats['error'] = int(match.group(1))
        
        if 'skipped' in line.lower() and re.search(r'\d+\s+skipped', line):
            match = re.search(r'(\d+)\s+skipped', line)
            if match:
                stats['skipped'] = int(match.group(1))
        
        # 查找失败的测试
        if 'FAILED' in line:
            test_match = re.search(r'FAILED\s+(.+?)(?:\s+-|\s*$)', line)
            if test_match:
                stats['failed_tests'].append(test_match.group(1).strip())
        
        # 查找错误的测试
        if 'ERROR' in line and 'tests/' in line:
            test_match = re.search(r'ERROR\s+(.+?)(?:\s+-|\s*$)', line)
            if test_match:
                stats['error_tests'].append(test_match.group(1).strip())
    
    stats['total'] = stats['passed'] + stats['failed'] + stats['error'] + stats['skipped']
    
    # 计算通过率
    if stats['total'] > 0:
        stats['pass_rate'] = (stats['passed'] / stats['total']) * 100
    else:
        stats['pass_rate'] = 0.0
    
    # 保存统计结果
    stats_file = output_dir / "trading_tests_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # 打印结果
    print("=" * 70)
    print("  测试统计结果")
    print("=" * 70)
    print(f"总测试数: {stats['total']}")
    print(f"通过: {stats['passed']}")
    print(f"失败: {stats['failed']}")
    print(f"错误: {stats['error']}")
    print(f"跳过: {stats['skipped']}")
    print(f"通过率: {stats['pass_rate']:.2f}%")
    print(f"退出码: {stats['exit_code']}")
    print()
    
    # 质量评估
    if stats['failed'] == 0 and stats['error'] == 0:
        print("=" * 70)
        print("  ✅ 测试通过率: 100% (质量优先原则达成)")
        print("  ✅ 状态: 可以继续推进覆盖率验证")
        print("=" * 70)
    else:
        print("=" * 70)
        print(f"  ❌ 测试通过率: {stats['pass_rate']:.2f}% (未达到100%)")
        print(f"  ❌ 失败测试: {stats['failed']}个")
        print(f"  ❌ 错误测试: {stats['error']}个")
        print("  ⚠️  必须先修复失败的测试才能继续")
        print("=" * 70)
        
        if stats['failed_tests']:
            print("\n失败的测试:")
            for test in stats['failed_tests'][:10]:  # 只显示前10个
                print(f"  - {test}")
            if len(stats['failed_tests']) > 10:
                print(f"  ... 还有 {len(stats['failed_tests']) - 10} 个失败的测试")
        
        if stats['error_tests']:
            print("\n错误的测试:")
            for test in stats['error_tests'][:10]:  # 只显示前10个
                print(f"  - {test}")
            if len(stats['error_tests']) > 10:
                print(f"  ... 还有 {len(stats['error_tests']) - 10} 个错误的测试")
    
    print()
    print(f"完整输出已保存到: {output_file}")
    print(f"统计结果已保存到: {stats_file}")
    print()
    
    return stats

if __name__ == "__main__":
    stats = run_trading_tests()
    sys.exit(0 if stats['failed'] == 0 and stats['error'] == 0 else 1)

