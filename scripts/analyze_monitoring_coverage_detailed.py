#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细分析监控模块的测试覆盖率和通过率
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def analyze_coverage():
    """分析覆盖率JSON文件"""
    coverage_file = Path("test_logs/monitoring_coverage.json")
    
    if not coverage_file.exists():
        print(f"错误: 覆盖率文件不存在: {coverage_file}")
        print("请先运行: pytest tests/unit/infrastructure/monitoring --cov=src/infrastructure/monitoring --cov-report=json:test_logs/monitoring_coverage.json")
        return None
    
    with open(coverage_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    totals = data['totals']
    files = data['files']
    
    # 按目录分组统计
    dir_stats = defaultdict(lambda: {'statements': 0, 'covered': 0, 'missing': 0, 'files': []})
    
    for file_path, file_data in files.items():
        if 'src/infrastructure/monitoring' not in file_path:
            continue
        
        # 获取目录
        parts = file_path.replace('src/infrastructure/monitoring/', '').split('/')
        if len(parts) > 1:
            dir_name = parts[0]
        else:
            dir_name = 'root'
        
        statements = file_data['summary']['num_statements']
        covered = file_data['summary']['covered_lines']
        missing = file_data['summary']['missing_lines']
        percent = file_data['summary']['percent_covered']
        
        dir_stats[dir_name]['statements'] += statements
        dir_stats[dir_name]['covered'] += covered
        dir_stats[dir_name]['missing'] += missing
        dir_stats[dir_name]['files'].append({
            'file': file_path,
            'percent': percent,
            'statements': statements,
            'covered': covered,
            'missing': missing
        })
    
    return {
        'totals': totals,
        'dir_stats': dict(dir_stats),
        'files': files
    }

def analyze_test_results():
    """分析测试结果"""
    output_file = Path("test_logs/monitoring_test_output.txt")
    
    if not output_file.exists():
        return None
    
    with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # 提取测试统计信息
    passed = 0
    failed = 0
    errors = 0
    skipped = 0
    
    # 查找测试统计行
    lines = content.split('\n')
    for line in lines:
        if 'passed' in line.lower() and ('failed' in line.lower() or 'error' in line.lower() or 'skipped' in line.lower()):
            # 解析类似 "2322 passed, 4 failed, 64 errors, 101 skipped"
            parts = line.split(',')
            for part in parts:
                part = part.strip()
                if 'passed' in part:
                    try:
                        passed = int(part.split()[0])
                    except:
                        pass
                elif 'failed' in part:
                    try:
                        failed = int(part.split()[0])
                    except:
                        pass
                elif 'error' in part:
                    try:
                        errors = int(part.split()[0])
                    except:
                        pass
                elif 'skipped' in part:
                    try:
                        skipped = int(part.split()[0])
                    except:
                        pass
    
    total = passed + failed + errors + skipped
    pass_rate = (passed / total * 100) if total > 0 else 0
    
    return {
        'total': total,
        'passed': passed,
        'failed': failed,
        'errors': errors,
        'skipped': skipped,
        'pass_rate': pass_rate
    }

def print_report(coverage_data, test_data):
    """打印详细报告"""
    print("=" * 80)
    print("监控模块测试覆盖率和通过率详细报告")
    print("=" * 80)
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
        dir_stats = coverage_data['dir_stats']
        for dir_name in sorted(dir_stats.keys()):
            stats = dir_stats[dir_name]
            if stats['statements'] > 0:
                percent = (stats['covered'] / stats['statements'] * 100) if stats['statements'] > 0 else 0
                print(f"{dir_name:20s} | {percent:6.2f}% | {stats['covered']:5d}/{stats['statements']:5d} | {len(stats['files']):3d} 文件")
        print()
        
        print("📄 低覆盖率文件 (< 80%)")
        print("-" * 80)
        low_coverage_files = []
        for dir_name, stats in dir_stats.items():
            for file_info in stats['files']:
                if file_info['percent'] < 80:
                    low_coverage_files.append(file_info)
        
        low_coverage_files.sort(key=lambda x: x['percent'])
        for file_info in low_coverage_files[:20]:  # 只显示前20个
            file_path = file_info['file'].replace('src/infrastructure/monitoring/', '')
            print(f"{file_path:50s} | {file_info['percent']:6.2f}% | {file_info['covered']:4d}/{file_info['statements']:4d}")
        if len(low_coverage_files) > 20:
            print(f"... 还有 {len(low_coverage_files) - 20} 个文件")
        print()
    
    if test_data:
        print("✅ 测试通过率统计")
        print("-" * 80)
        print(f"总测试数: {test_data['total']}")
        print(f"通过: {test_data['passed']}")
        print(f"失败: {test_data['failed']}")
        print(f"错误: {test_data['errors']}")
        print(f"跳过: {test_data['skipped']}")
        print(f"通过率: {test_data['pass_rate']:.2f}%")
        print()
    
    print("=" * 80)

if __name__ == "__main__":
    coverage_data = analyze_coverage()
    test_data = analyze_test_results()
    print_report(coverage_data, test_data)

