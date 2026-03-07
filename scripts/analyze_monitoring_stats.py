#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析监控模块的测试覆盖率和通过率
"""

import json
import re
from pathlib import Path

def read_coverage_json():
    """读取覆盖率JSON文件"""
    coverage_file = Path("test_logs/monitoring_coverage.json")
    if not coverage_file.exists():
        return None
    
    with open(coverage_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def read_test_output():
    """读取测试输出文件"""
    output_file = Path("test_logs/pytest_output.txt")
    if not output_file.exists():
        return None
    
    with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def parse_test_stats(output_text):
    """解析测试统计信息"""
    if not output_text:
        return None
    
    stats = {
        'total': 0,
        'passed': 0,
        'failed': 0,
        'errors': 0,
        'skipped': 0,
        'pass_rate': 0.0
    }
    
    # 查找测试统计行
    lines = output_text.split('\n')
    for line in lines:
        # 查找包含测试统计的行
        if 'passed' in line.lower() and ('failed' in line.lower() or 'error' in line.lower() or 'skipped' in line.lower()):
            # 使用正则表达式提取数字
            patterns = [
                r'(\d+)\s+passed',
                r'(\d+)\s+failed',
                r'(\d+)\s+error',
                r'(\d+)\s+skipped'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    num = int(match.group(1))
                    if 'passed' in pattern:
                        stats['passed'] = num
                    elif 'failed' in pattern:
                        stats['failed'] = num
                    elif 'error' in pattern:
                        stats['errors'] = num
                    elif 'skipped' in pattern:
                        stats['skipped'] = num
    
    stats['total'] = stats['passed'] + stats['failed'] + stats['errors'] + stats['skipped']
    if stats['total'] > 0:
        stats['pass_rate'] = (stats['passed'] / stats['total']) * 100
    
    return stats

def analyze_coverage(coverage_data):
    """分析覆盖率数据"""
    if not coverage_data:
        return None
    
    totals = coverage_data['totals']
    files = coverage_data['files']
    
    # 按目录分组统计
    dir_stats = {}
    
    for file_path, file_data in files.items():
        if 'src/infrastructure/monitoring' not in file_path:
            continue
        
        # 获取目录（统一处理Windows和Unix路径）
        normalized_path = file_path.replace('\\', '/')
        # 移除前缀
        if 'src/infrastructure/monitoring/' in normalized_path:
            rel_path = normalized_path.split('src/infrastructure/monitoring/')[-1]
        else:
            rel_path = normalized_path
        
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

def print_report(coverage_analysis, test_stats):
    """打印详细报告"""
    print("\n" + "=" * 80)
    print("监控模块测试覆盖率和通过率详细报告")
    print("=" * 80 + "\n")
    
    if test_stats:
        print("✅ 测试通过率统计")
        print("-" * 80)
        print(f"总测试数: {test_stats['total']}")
        print(f"通过: {test_stats['passed']}")
        print(f"失败: {test_stats['failed']}")
        print(f"错误: {test_stats['errors']}")
        print(f"跳过: {test_stats['skipped']}")
        if test_stats['total'] > 0:
            print(f"通过率: {test_stats['pass_rate']:.2f}%")
        print()
    
    if coverage_analysis:
        totals = coverage_analysis['totals']
        print("📊 整体覆盖率统计")
        print("-" * 80)
        print(f"覆盖率: {totals['percent_covered']:.2f}%")
        print(f"总行数: {totals['num_statements']}")
        print(f"已覆盖行数: {totals['covered_lines']}")
        print(f"未覆盖行数: {totals['missing_lines']}")
        if 'num_branches' in totals:
            print(f"总分支数: {totals['num_branches']}")
            print(f"已覆盖分支数: {totals['covered_branches']}")
            if totals['num_branches'] > 0:
                branch_coverage = (totals['covered_branches'] / totals['num_branches']) * 100
                print(f"分支覆盖率: {branch_coverage:.2f}%")
        print()
        
        print("📁 按目录统计覆盖率")
        print("-" * 80)
        print(f"{'目录':20s} | {'覆盖率':>8s} | {'已覆盖/总行数':>15s} | {'文件数':>6s}")
        print("-" * 80)
        
        dir_stats = coverage_analysis['dir_stats']
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
                    low_coverage_files.append((
                        file_info['file'], 
                        file_info['percent'], 
                        file_info['covered'], 
                        file_info['statements']
                    ))
        
        if low_coverage_files:
            print("📄 低覆盖率文件 (< 80%)")
            print("-" * 80)
            low_coverage_files.sort(key=lambda x: x[1])  # 按覆盖率排序
            for file_path, percent, covered, statements in low_coverage_files[:20]:  # 显示前20个
                print(f"{file_path:55s} | {percent:6.2f}% | {covered:4d}/{statements:4d}")
            if len(low_coverage_files) > 20:
                print(f"... 还有 {len(low_coverage_files) - 20} 个文件")
            print()
    
    print("=" * 80 + "\n")

if __name__ == "__main__":
    # 读取覆盖率数据
    coverage_data = read_coverage_json()
    
    # 读取测试输出
    test_output = read_test_output()
    
    # 解析测试统计
    test_stats = parse_test_stats(test_output)
    
    # 分析覆盖率
    coverage_analysis = analyze_coverage(coverage_data)
    
    # 打印报告
    print_report(coverage_analysis, test_stats)

