#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成监控模块测试覆盖率和通过率综合报告
"""

import json
from pathlib import Path
from collections import defaultdict

def load_coverage_data():
    """加载覆盖率JSON数据"""
    coverage_file = Path("test_logs/monitoring_coverage.json")
    if not coverage_file.exists():
        print(f"错误: 覆盖率文件不存在: {coverage_file}")
        return None
    
    with open(coverage_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_coverage(coverage_data):
    """分析覆盖率数据"""
    if not coverage_data:
        return None
    
    totals = coverage_data['totals']
    files = coverage_data['files']
    
    # 按目录分组统计
    dir_stats = defaultdict(lambda: {
        'statements': 0,
        'covered': 0,
        'missing': 0,
        'files': []
    })
    
    # 统计所有文件
    all_files = []
    
    for file_path, file_data in files.items():
        # 统一处理路径格式
        normalized_path = file_path.replace('\\', '/')
        
        if 'src/infrastructure/monitoring' not in normalized_path:
            continue
        
        # 获取相对路径
        if 'src/infrastructure/monitoring/' in normalized_path:
            rel_path = normalized_path.split('src/infrastructure/monitoring/')[-1]
        else:
            rel_path = normalized_path
        
        # 获取目录名
        parts = rel_path.split('/')
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
        
        file_info = {
            'file': rel_path,
            'percent': percent,
            'statements': statements,
            'covered': covered,
            'missing': missing
        }
        
        dir_stats[dir_name]['files'].append(file_info)
        all_files.append(file_info)
    
    return {
        'totals': totals,
        'dir_stats': dict(dir_stats),
        'all_files': all_files
    }

def print_report(coverage_analysis):
    """打印详细报告"""
    if not coverage_analysis:
        print("无法生成报告：缺少覆盖率数据")
        return
    
    totals = coverage_analysis['totals']
    dir_stats = coverage_analysis['dir_stats']
    all_files = coverage_analysis['all_files']
    
    print("\n" + "=" * 80)
    print("监控模块测试覆盖率详细报告 (使用pytest-cov统计)")
    print("=" * 80 + "\n")
    
    # 整体统计
    print("📊 整体覆盖率统计")
    print("-" * 80)
    print(f"覆盖率: {totals['percent_covered']:.2f}%")
    print(f"总行数: {totals['num_statements']}")
    print(f"已覆盖行数: {totals['covered_lines']}")
    print(f"未覆盖行数: {totals['missing_lines']}")
    
    if 'num_branches' in totals and totals['num_branches'] > 0:
        branch_coverage = (totals['covered_branches'] / totals['num_branches']) * 100
        print(f"总分支数: {totals['num_branches']}")
        print(f"已覆盖分支数: {totals['covered_branches']}")
        print(f"分支覆盖率: {branch_coverage:.2f}%")
    
    print()
    
    # 按目录统计
    print("📁 按目录统计覆盖率")
    print("-" * 80)
    print(f"{'目录':20s} | {'覆盖率':>8s} | {'已覆盖/总行数':>15s} | {'文件数':>6s}")
    print("-" * 80)
    
    for dir_name in sorted(dir_stats.keys()):
        stats = dir_stats[dir_name]
        if stats['statements'] > 0:
            percent = (stats['covered'] / stats['statements'] * 100) if stats['statements'] > 0 else 0
            print(f"{dir_name:20s} | {percent:7.2f}% | {stats['covered']:5d}/{stats['statements']:5d} | {len(stats['files']):6d}")
    print()
    
    # 低覆盖率文件
    low_coverage_files = [
        f for f in all_files 
        if f['percent'] < 80 and f['statements'] > 0
    ]
    
    if low_coverage_files:
        print("📄 低覆盖率文件 (< 80%)")
        print("-" * 80)
        low_coverage_files.sort(key=lambda x: x['percent'])
        for file_info in low_coverage_files[:25]:  # 显示前25个
            print(f"{file_info['file']:55s} | {file_info['percent']:6.2f}% | {file_info['covered']:4d}/{file_info['statements']:4d}")
        if len(low_coverage_files) > 25:
            print(f"... 还有 {len(low_coverage_files) - 25} 个文件")
        print()
    
    # 高覆盖率文件（>= 80%）
    high_coverage_files = [
        f for f in all_files 
        if f['percent'] >= 80 and f['statements'] > 0
    ]
    
    if high_coverage_files:
        print("✅ 高覆盖率文件 (>= 80%)")
        print("-" * 80)
        high_coverage_files.sort(key=lambda x: x['percent'], reverse=True)
        for file_info in high_coverage_files[:15]:  # 显示前15个
            print(f"{file_info['file']:55s} | {file_info['percent']:6.2f}% | {file_info['covered']:4d}/{file_info['statements']:4d}")
        if len(high_coverage_files) > 15:
            print(f"... 还有 {len(high_coverage_files) - 15} 个文件")
        print()
    
    print("=" * 80 + "\n")
    
    # 总结
    print("📈 覆盖率总结")
    print("-" * 80)
    print(f"总文件数: {len(all_files)}")
    print(f"高覆盖率文件 (>=80%): {len(high_coverage_files)}")
    print(f"低覆盖率文件 (<80%): {len(low_coverage_files)}")
    print(f"整体覆盖率: {totals['percent_covered']:.2f}%")
    print(f"目标覆盖率: 80%")
    if totals['percent_covered'] >= 80:
        print("✅ 已达到目标覆盖率")
    else:
        gap = 80 - totals['percent_covered']
        print(f"⚠️  距离目标还差 {gap:.2f}%")
    print()

if __name__ == "__main__":
    coverage_data = load_coverage_data()
    coverage_analysis = analyze_coverage(coverage_data)
    print_report(coverage_analysis)

