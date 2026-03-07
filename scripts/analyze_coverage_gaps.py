#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析覆盖率缺口，找出具体未覆盖的文件和行
"""

import json
import os
from pathlib import Path
from collections import defaultdict

def analyze_module_coverage_gaps(module_name):
    """分析模块的覆盖率缺口"""
    
    coverage_file = f"test_logs/cov_{module_name}.json"
    
    if not os.path.exists(coverage_file):
        print(f"❌ 覆盖率文件不存在: {coverage_file}")
        return None
    
    with open(coverage_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    files = data.get('files', {})
    
    # 按覆盖率排序
    file_stats = []
    for filepath, stats in files.items():
        if f'/infrastructure/{module_name}/' in filepath or f'\\infrastructure\\{module_name}\\' in filepath:
            coverage = stats['summary']['percent_covered']
            missing_lines = stats['summary']['num_statements'] - stats['summary']['covered_lines']
            
            file_stats.append({
                'file': filepath,
                'coverage': coverage,
                'total_lines': stats['summary']['num_statements'],
                'covered_lines': stats['summary']['covered_lines'],
                'missing_lines': missing_lines,
                'missing_line_numbers': stats.get('missing_lines', [])
            })
    
    return sorted(file_stats, key=lambda x: x['missing_lines'], reverse=True)

def print_coverage_gaps(module_name, top_n=30):
    """打印覆盖率缺口分析"""
    
    print("="*100)
    print(f"📊 {module_name.capitalize()} 模块覆盖率缺口分析")
    print("="*100)
    print()
    
    gaps = analyze_module_coverage_gaps(module_name)
    
    if not gaps:
        return
    
    total_missing = sum(g['missing_lines'] for g in gaps)
    total_lines = sum(g['total_lines'] for g in gaps)
    
    print(f"📈 总体统计")
    print(f"  文件总数: {len(gaps)}")
    print(f"  代码总行数: {total_lines:,}")
    print(f"  未覆盖行数: {total_missing:,}")
    print(f"  整体覆盖率: {((total_lines-total_missing)/total_lines*100):.2f}%")
    print()
    
    # 找出0%覆盖的文件
    zero_coverage = [g for g in gaps if g['coverage'] == 0]
    if zero_coverage:
        print(f"🔴 完全未覆盖的文件 ({len(zero_coverage)}个)")
        print(f"{'文件':<60} {'行数':<10}")
        print("-"*100)
        for g in sorted(zero_coverage, key=lambda x: x['total_lines'], reverse=True)[:15]:
            filename = Path(g['file']).name
            print(f"{filename:<60} {g['total_lines']:<10}")
        print()
    
    # 找出低覆盖率文件（1-30%）
    low_coverage = [g for g in gaps if 0 < g['coverage'] < 30]
    if low_coverage:
        print(f"🟠 低覆盖率文件 (0-30%, {len(low_coverage)}个)")
        print(f"{'文件':<60} {'覆盖率':<10} {'未覆盖行':<10}")
        print("-"*100)
        for g in sorted(low_coverage, key=lambda x: x['missing_lines'], reverse=True)[:15]:
            filename = Path(g['file']).name
            print(f"{filename:<60} {g['coverage']:>6.2f}%   {g['missing_lines']:<10}")
        print()
    
    # Top N未覆盖行数最多的文件
    print(f"🎯 未覆盖行数最多的Top {min(top_n, len(gaps))}文件")
    print(f"{'排名':<6} {'文件':<50} {'覆盖率':<10} {'未覆盖行':<12} {'总行数':<10}")
    print("-"*100)
    
    for idx, g in enumerate(gaps[:top_n], 1):
        filename = Path(g['file']).name
        if len(filename) > 48:
            filename = filename[:45] + "..."
        print(f"{idx:<6} {filename:<50} {g['coverage']:>6.2f}%   {g['missing_lines']:<12} {g['total_lines']:<10}")
    
    print()
    
    # 统计覆盖率分布
    ranges = {
        '0%': 0,
        '1-20%': 0,
        '21-40%': 0,
        '41-60%': 0,
        '61-80%': 0,
        '81-99%': 0,
        '100%': 0
    }
    
    for g in gaps:
        cov = g['coverage']
        if cov == 0:
            ranges['0%'] += 1
        elif cov <= 20:
            ranges['1-20%'] += 1
        elif cov <= 40:
            ranges['21-40%'] += 1
        elif cov <= 60:
            ranges['41-60%'] += 1
        elif cov <= 80:
            ranges['61-80%'] += 1
        elif cov < 100:
            ranges['81-99%'] += 1
        else:
            ranges['100%'] += 1
    
    print("📊 覆盖率分布")
    for range_name, count in ranges.items():
        if count > 0:
            pct = count / len(gaps) * 100
            bar = '█' * int(pct / 2)
            print(f"  {range_name:<10} {count:>4} 个文件  {bar} {pct:.1f}%")
    
    print()
    print("="*100)
    print("💡 改进建议")
    print("="*100)
    
    # 快速收益机会
    quick_wins = [g for g in gaps if 50 < g['coverage'] < 90 and g['missing_lines'] < 50]
    if quick_wins:
        print(f"✅ 快速收益机会 ({len(quick_wins)}个文件，覆盖率50-90%且未覆盖行<50):")
        for g in sorted(quick_wins, key=lambda x: x['missing_lines'], reverse=True)[:10]:
            filename = Path(g['file']).name
            print(f"  • {filename}: {g['coverage']:.1f}% → 补充{g['missing_lines']}行可达高覆盖")
    
    # 高价值目标
    high_value = [g for g in gaps if g['missing_lines'] > 100 and g['coverage'] < 50]
    if high_value:
        print(f"\n🎯 高价值目标 ({len(high_value)}个文件，未覆盖>100行且覆盖率<50%):")
        for g in sorted(high_value, key=lambda x: x['missing_lines'], reverse=True)[:10]:
            filename = Path(g['file']).name
            estimated_tests = g['missing_lines'] // 10  # 粗略估计
            print(f"  • {filename}: {g['missing_lines']}行未覆盖 → 预计需要{estimated_tests}个测试")
    
    # 零覆盖优先
    if zero_coverage:
        print(f"\n🔴 零覆盖优先 ({len(zero_coverage)}个文件):")
        total_zero_lines = sum(g['total_lines'] for g in zero_coverage[:10])
        print(f"  • 前10个文件共{total_zero_lines}行 → 建立基础测试框架")

def main():
    """主函数"""
    modules = ['health', 'logging', 'config', 'utils']
    
    for module in modules:
        print_coverage_gaps(module, top_n=20)
        print("\n" * 2)

if __name__ == "__main__":
    main()

