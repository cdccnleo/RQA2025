#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行监控模块测试并生成覆盖率报告
"""

import subprocess
import json
import sys
from pathlib import Path

def run_tests():
    """运行测试并生成覆盖率报告"""
    print("正在运行测试并生成覆盖率报告...")
    
    # 运行pytest
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/unit/infrastructure/monitoring",
        "--cov=src/infrastructure/monitoring",
        "--cov-report=json:test_logs/monitoring_coverage.json",
        "--cov-report=term-missing",
        "-q",
        "--tb=no"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        # 保存输出
        output_file = Path("test_logs/monitoring_test_output.txt")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result.stdout)
            f.write("\n\n=== STDERR ===\n\n")
            f.write(result.stderr)
        
        # 解析测试结果
        test_stats = parse_test_output(result.stdout + result.stderr)
        
        return result.returncode == 0, test_stats
        
    except Exception as e:
        print(f"运行测试时出错: {e}")
        return False, None

def parse_test_output(output):
    """解析测试输出"""
    stats = {
        'total': 0,
        'passed': 0,
        'failed': 0,
        'errors': 0,
        'skipped': 0,
        'pass_rate': 0.0
    }
    
    lines = output.split('\n')
    for line in lines:
        if 'passed' in line.lower() and ('failed' in line.lower() or 'error' in line.lower() or 'skipped' in line.lower()):
            # 解析类似 "2322 passed, 4 failed, 64 errors, 101 skipped"
            parts = line.split(',')
            for part in parts:
                part = part.strip()
                if 'passed' in part:
                    try:
                        stats['passed'] = int(part.split()[0])
                    except:
                        pass
                elif 'failed' in part:
                    try:
                        stats['failed'] = int(part.split()[0])
                    except:
                        pass
                elif 'error' in part.lower():
                    try:
                        stats['errors'] = int(part.split()[0])
                    except:
                        pass
                elif 'skipped' in part:
                    try:
                        stats['skipped'] = int(part.split()[0])
                    except:
                        pass
    
    stats['total'] = stats['passed'] + stats['failed'] + stats['errors'] + stats['skipped']
    if stats['total'] > 0:
        stats['pass_rate'] = (stats['passed'] / stats['total']) * 100
    
    return stats

if __name__ == "__main__":
    success, test_stats = run_tests()
    
    if test_stats:
        print(f"\n测试统计:")
        print(f"  总测试数: {test_stats['total']}")
        print(f"  通过: {test_stats['passed']}")
        print(f"  失败: {test_stats['failed']}")
        print(f"  错误: {test_stats['errors']}")
        print(f"  跳过: {test_stats['skipped']}")
        print(f"  通过率: {test_stats['pass_rate']:.2f}%")
    
    sys.exit(0 if success else 1)

