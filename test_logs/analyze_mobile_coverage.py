#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
移动端层测试覆盖率分析脚本
按子模块统计覆盖率
"""

import json
from collections import defaultdict
from pathlib import Path

def analyze_coverage():
    """分析移动端层覆盖率"""
    coverage_file = Path(__file__).parent / "mobile_coverage.json"
    
    if not coverage_file.exists():
        print(f"覆盖率文件不存在: {coverage_file}")
        return
    
    with open(coverage_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    files = data.get('files', {})
    
    # 按子模块分组
    modules = defaultdict(lambda: {'total': 0, 'covered': 0, 'files': []})
    
    for file_path, file_data in files.items():
        if 'mobile' not in file_path:
            continue
        
        # 提取子模块名称 (src/mobile/xxx/...)
        parts = file_path.replace('\\', '/').split('/')
        if len(parts) >= 3 and parts[0] == 'src' and parts[1] == 'mobile':
            if len(parts) > 3:
                module_name = parts[2]  # 子模块名
            else:
                module_name = 'root'  # 根目录文件
        else:
            module_name = 'root'  # 根目录文件
        
        summary = file_data.get('summary', {})
        total = summary.get('num_statements', 0)
        covered = summary.get('covered_statements', 0)
        
        modules[module_name]['total'] += total
        modules[module_name]['covered'] += covered
        modules[module_name]['files'].append(file_path)
    
    # 打印结果
    print("=" * 80)
    print("移动端层测试覆盖率分析报告")
    print("=" * 80)
    print(f"\n总体覆盖率: {data.get('totals', {}).get('percent_covered', 0):.2f}%")
    print(f"总代码行数: {data.get('totals', {}).get('num_statements', 0)}")
    print(f"已覆盖行数: {data.get('totals', {}).get('covered_statements', 0)}")
    print(f"未覆盖行数: {data.get('totals', {}).get('num_statements', 0) - data.get('totals', {}).get('covered_statements', 0)}")
    print("\n" + "=" * 80)
    print("各子模块覆盖率详情:")
    print("=" * 80)
    
    # 按覆盖率排序
    sorted_modules = sorted(
        modules.items(),
        key=lambda x: x[1]['covered'] / x[1]['total'] if x[1]['total'] > 0 else 0,
        reverse=True
    )
    
    print(f"\n{'子模块':<25} {'覆盖率':<12} {'已覆盖/总数':<15} {'文件数':<10} {'状态'}")
    print("-" * 80)
    
    for module_name, module_data in sorted_modules:
        total = module_data['total']
        covered = module_data['covered']
        file_count = len(module_data['files'])
        
        if total > 0:
            coverage_percent = (covered / total) * 100
            status = "✅ 优秀" if coverage_percent >= 80 else "✅ 良好" if coverage_percent >= 50 else "⚠️ 需改进" if coverage_percent >= 20 else "❌ 待提升"
        else:
            coverage_percent = 0
            status = "📝 无代码"
        
        print(f"{module_name:<25} {coverage_percent:>6.2f}%     {covered:>6}/{total:<6} {file_count:>6}      {status}")
    
    print("\n" + "=" * 80)
    print("覆盖率统计:")
    print("=" * 80)
    
    # 统计各覆盖率区间的模块数
    excellent = sum(1 for _, d in modules.items() if d['total'] > 0 and (d['covered'] / d['total']) >= 0.8)
    good = sum(1 for _, d in modules.items() if d['total'] > 0 and 0.5 <= (d['covered'] / d['total']) < 0.8)
    fair = sum(1 for _, d in modules.items() if d['total'] > 0 and 0.2 <= (d['covered'] / d['total']) < 0.5)
    poor = sum(1 for _, d in modules.items() if d['total'] > 0 and (d['covered'] / d['total']) < 0.2)
    empty = sum(1 for _, d in modules.items() if d['total'] == 0)
    
    print(f"✅ 优秀 (≥80%): {excellent} 个模块")
    print(f"✅ 良好 (50-79%): {good} 个模块")
    print(f"⚠️ 需改进 (20-49%): {fair} 个模块")
    print(f"❌ 待提升 (<20%): {poor} 个模块")
    print(f"📝 无代码: {empty} 个模块")
    print(f"总计: {len(modules)} 个子模块")

if __name__ == '__main__':
    analyze_coverage()

