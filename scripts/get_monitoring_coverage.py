#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
获取监控模块的测试覆盖率和通过率
"""

import subprocess
import json
import sys
import os
from pathlib import Path

# 设置编码环境变量
os.environ['PYTHONIOENCODING'] = 'utf-8'

def run_pytest():
    """运行pytest并生成覆盖率JSON"""
    print("正在运行测试并生成覆盖率报告...")
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/unit/infrastructure/monitoring",
        "--cov=src/infrastructure/monitoring",
        "--cov-report=json:test_logs/monitoring_coverage.json",
        "--cov-report=term",
        "-q",
        "--tb=no",
        "--no-header"
    ]
    
    # 运行pytest，忽略编码错误
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            errors='replace'
        )
        
        # 解析测试统计
        output = result.stdout + result.stderr
        test_stats = parse_test_stats(output)
        
        return test_stats
        
    except Exception as e:
        print(f"运行测试时出错: {e}")
        return None

def parse_test_stats(output):
    """解析测试统计信息"""
    stats = {
        'total': 0,
        'passed': 0,
        'failed': 0,
        'errors': 0,
        'skipped': 0,
        'pass_rate': 0.0
    }
    
    # 查找测试统计行
    lines = output.split('\n')
    for line in lines:
        # 查找包含 "passed" 的行
        if 'passed' in line.lower():
            # 尝试解析各种格式
            parts = line.replace(' in ', ' ').split()
            for i, part in enumerate(parts):
                if 'passed' in part.lower():
                    try:
                        stats['passed'] = int(parts[i-1])
                    except:
                        pass
                elif 'failed' in part.lower():
                    try:
                        stats['failed'] = int(parts[i-1])
                    except:
                        pass
                elif 'error' in part.lower() and i > 0:
                    try:
                        stats['errors'] = int(parts[i-1])
                    except:
                        pass
                elif 'skipped' in part.lower():
                    try:
                        stats['skipped'] = int(parts[i-1])
                    except:
                        pass
    
    stats['total'] = stats['passed'] + stats['failed'] + stats['errors'] + stats['skipped']
    if stats['total'] > 0:
        stats['pass_rate'] = (stats['passed'] / stats['total']) * 100
    
    return stats

def analyze_coverage():
    """分析覆盖率JSON文件"""
    coverage_file = Path("test_logs/monitoring_coverage.json")
    
    if not coverage_file.exists():
        return None
    
    try:
        with open(coverage_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        totals = data['totals']
        files = data['files']
        
        # 按目录分组统计
        dir_stats = {}
        
        for file_path, file_data in files.items():
            if 'src/infrastructure/monitoring' not in file_path:
                continue
            
            # 获取目录
            rel_path = file_path.replace('src/infrastructure/monitoring/', '')
            parts = rel_path.split('/')
            if len(parts) > 1:
                dir_name = parts[0]
            else:
                dir_name = 'root'
            
            if dir_name not in dir_stats:
                dir_stats[dir_name] = {
                    'statements': 0,
                    'covered': 0,
                    'missing': 0,
                    'files': []
                }
            
            statements = file_data['summary']['num_statements']
            covered = file_data['summary']['covered_lines']
            missing = file_data['summary']['missing_lines']
            percent = file_data['summary']['percent_covered']
            
            dir_stats[dir_name]['statements'] += statements
            dir_stats[dir_name]['covered'] += covered
            dir_stats[dir_name]['missing'] += missing
            dir_stats[dir_name]['files'].append({
                'file': rel_path,
                'percent': percent,
                'statements': statements,
                'covered': covered,
                'missing': missing
            })
        
        return {
            'totals': totals,
            'dir_stats': dir_stats
        }
    except Exception as e:
        print(f"分析覆盖率时出错: {e}")
        return None

def print_report(coverage_data, test_stats):
    """打印报告"""
    print("\n" + "=" * 80)
    print("监控模块测试覆盖率和通过率报告")
    print("=" * 80 + "\n")
    
    if test_stats:
        print("✅ 测试通过率统计")
        print("-" * 80)
        print(f"总测试数: {test_stats['total']}")
        print(f"通过: {test_stats['passed']}")
        print(f"失败: {test_stats['failed']}")
        print(f"错误: {test_stats['errors']}")
        print(f"跳过: {test_stats['skipped']}")
        print(f"通过率: {test_stats['pass_rate']:.2f}%")
        print()
    
    if coverage_data:
        totals = coverage_data['totals']
        print("📊 整体覆盖率统计")
        print("-" * 80)
        print(f"覆盖率: {totals['percent_covered']:.2f}%")
        print(f"总行数: {totals['num_statements']}")
        print(f"已覆盖行数: {totals['covered_lines']}")
        print(f"未覆盖行数: {totals['missing_lines']}")
        print(f"总分支数: {totals['num_branches']}")
        print(f"已覆盖分支数: {totals['covered_branches']}")
        print()
        
        print("📁 按目录统计覆盖率")
        print("-" * 80)
        print(f"{'目录':20s} | {'覆盖率':>8s} | {'已覆盖/总行数':>15s} | {'文件数':>6s}")
        print("-" * 80)
        
        dir_stats = coverage_data['dir_stats']
        for dir_name in sorted(dir_stats.keys()):
            stats = dir_stats[dir_name]
            if stats['statements'] > 0:
                percent = (stats['covered'] / stats['statements'] * 100) if stats['statements'] > 0 else 0
                print(f"{dir_name:20s} | {percent:7.2f}% | {stats['covered']:5d}/{stats['statements']:5d} | {len(stats['files']):6d}")
        print()
        
        # 找出低覆盖率文件
        low_coverage_files = []
        for dir_name, stats in dir_stats.items():
            for file_info in stats['files']:
                if file_info['percent'] < 80 and file_info['statements'] > 0:
                    low_coverage_files.append((file_info['file'], file_info['percent'], file_info['covered'], file_info['statements']))
        
        if low_coverage_files:
            print("📄 低覆盖率文件 (< 80%)")
            print("-" * 80)
            low_coverage_files.sort(key=lambda x: x[1])  # 按覆盖率排序
            for file_path, percent, covered, statements in low_coverage_files[:15]:  # 只显示前15个
                print(f"{file_path:50s} | {percent:6.2f}% | {covered:4d}/{statements:4d}")
            if len(low_coverage_files) > 15:
                print(f"... 还有 {len(low_coverage_files) - 15} 个文件")
            print()
    
    print("=" * 80 + "\n")

if __name__ == "__main__":
    # 运行测试
    test_stats = run_pytest()
    
    # 分析覆盖率
    coverage_data = analyze_coverage()
    
    # 打印报告
    print_report(coverage_data, test_stats)

