#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控模块覆盖率分析脚本
分析监控模块的测试覆盖率，识别低覆盖率文件
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple

def analyze_coverage(coverage_file: str = "test_logs/coverage_monitoring.json") -> Dict:
    """分析覆盖率数据"""
    coverage_path = Path(coverage_file)
    if not coverage_path.exists():
        print(f"覆盖率文件不存在: {coverage_file}")
        return {}
    
    with open(coverage_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    files = data.get('files', {})
    monitoring_files = {
        k: v for k, v in files.items() 
        if 'monitoring' in k and not k.endswith('__pycache__')
    }
    
    # 按覆盖率排序
    sorted_files = sorted(
        monitoring_files.items(),
        key=lambda x: x[1]['summary']['percent_covered']
    )
    
    # 分类统计
    low_coverage = []  # < 80%
    medium_coverage = []  # 80-90%
    high_coverage = []  # > 90%
    
    for file_path, file_data in sorted_files:
        coverage = file_data['summary']['percent_covered']
        lines = file_data['summary']['num_statements']
        missing = file_data['summary']['missing_lines']
        
        file_info = {
            'file': file_path,
            'coverage': coverage,
            'lines': lines,
            'missing': missing
        }
        
        if coverage < 80:
            low_coverage.append(file_info)
        elif coverage < 90:
            medium_coverage.append(file_info)
        else:
            high_coverage.append(file_info)
    
    return {
        'low_coverage': low_coverage,
        'medium_coverage': medium_coverage,
        'high_coverage': high_coverage,
        'total_files': len(sorted_files),
        'overall_coverage': data.get('totals', {}).get('percent_covered', 0)
    }

def print_report(analysis: Dict):
    """打印分析报告"""
    print("=" * 80)
    print("监控模块测试覆盖率分析报告")
    print("=" * 80)
    print(f"\n整体覆盖率: {analysis['overall_coverage']:.2f}%")
    print(f"总文件数: {analysis['total_files']}")
    print(f"\n低覆盖率文件 (<80%): {len(analysis['low_coverage'])}")
    print(f"中等覆盖率文件 (80-90%): {len(analysis['medium_coverage'])}")
    print(f"高覆盖率文件 (>90%): {len(analysis['high_coverage'])}")
    
    print("\n" + "=" * 80)
    print("低覆盖率文件详情（需要优先提升）:")
    print("=" * 80)
    for i, file_info in enumerate(analysis['low_coverage'][:30], 1):
        print(f"{i:2d}. {file_info['coverage']:6.2f}% - {file_info['file']}")
        print(f"    缺失行数: {file_info['missing']}, 总行数: {file_info['lines']}")
    
    if len(analysis['low_coverage']) > 30:
        print(f"\n... 还有 {len(analysis['low_coverage']) - 30} 个低覆盖率文件")

if __name__ == "__main__":
    analysis = analyze_coverage()
    if analysis:
        print_report(analysis)
        
        # 保存报告
        report_file = Path("test_logs/monitoring_coverage_analysis.json")
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"\n详细报告已保存到: {report_file}")

