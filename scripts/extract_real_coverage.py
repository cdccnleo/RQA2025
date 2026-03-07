#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""提取真实覆盖率报告"""

import json
from pathlib import Path
from collections import defaultdict

def extract_coverage():
    """提取并分析覆盖率数据"""
    
    with open('test_logs/infrastructure_coverage.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 按模块分组统计
    module_stats = defaultdict(lambda: {'covered': 0, 'total': 0, 'files': []})
    
    for file_path, file_data in data['files'].items():
        if 'src/infrastructure/' in file_path or 'src\\infrastructure\\' in file_path:
            # 提取模块名
            parts = file_path.replace('\\', '/').split('/')
            if len(parts) >= 3:
                module = parts[2]
                
                covered = file_data['summary']['covered_lines']
                total = file_data['summary']['num_statements']
                percent = file_data['summary']['percent_covered']
                
                module_stats[module]['covered'] += covered
                module_stats[module]['total'] += total
                module_stats[module]['files'].append({
                    'path': file_path,
                    'covered': covered,
                    'total': total,
                    'percent': percent
                })
    
    # 生成报告
    print("="*80)
    print("📊 基础设施层真实覆盖率分析")
    print("="*80)
    
    print(f"\n{'模块':<20} {'覆盖率':<10} {'已覆盖行':<12} {'总行数':<10} {'文件数':<10}")
    print("-"*80)
    
    sorted_modules = sorted(module_stats.items(), 
                          key=lambda x: (x[1]['covered']/x[1]['total'] if x[1]['total'] > 0 else 0), 
                          reverse=True)
    
    for module, stats in sorted_modules:
        coverage = (stats['covered'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"{module:<20} {coverage:>6.2f}%  {stats['covered']:>10}  {stats['total']:>10}  {len(stats['files']):>8}")
    
    # 总体统计
    total_covered = data['totals']['covered_lines']
    total_lines = data['totals']['num_statements']
    total_percent = data['totals']['percent_covered']
    
    print("-"*80)
    print(f"{'总体':<20} {total_percent:>6.2f}%  {total_covered:>10}  {total_lines:>10}  {len(data['files']):>8}")
    
    print("\n" + "="*80)
    print("🎯 达标评估")
    print("="*80)
    print(f"当前覆盖率: {total_percent:.2f}%")
    print(f"目标覆盖率: 80.00%")
    print(f"差距: {80 - total_percent:.2f}%")
    
    # 找出低覆盖率模块
    print("\n" + "="*80)
    print("⚠️  低覆盖率模块（<50%）")
    print("="*80)
    
    low_coverage_modules = []
    for module, stats in sorted_modules:
        coverage = (stats['covered'] / stats['total'] * 100) if stats['total'] > 0 else 0
        if coverage < 50:
            low_coverage_modules.append((module, coverage, stats))
            print(f"{module}: {coverage:.2f}%")
    
    # 保存简化报告
    report = {
        'overall': {
            'coverage': total_percent,
            'covered_lines': total_covered,
            'total_lines': total_lines,
            'file_count': len(data['files'])
        },
        'modules': {}
    }
    
    for module, stats in sorted_modules:
        coverage = (stats['covered'] / stats['total'] * 100) if stats['total'] > 0 else 0
        report['modules'][module] = {
            'coverage': coverage,
            'covered_lines': stats['covered'],
            'total_lines': stats['total'],
            'file_count': len(stats['files'])
        }
    
    with open('test_logs/real_coverage_summary.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 简化报告已保存: test_logs/real_coverage_summary.json")
    
    # 生成Markdown报告
    generate_markdown_report(sorted_modules, total_percent, total_covered, total_lines, len(data['files']))

def generate_markdown_report(sorted_modules, total_percent, total_covered, total_lines, file_count):
    """生成Markdown格式报告"""
    
    md_content = f"""# 🎯 基础设施层真实覆盖率评估报告

**生成时间**: 2025-11-02
**测试通过**: 1557 个
**测试失败**: 72 个
**整体覆盖率**: {total_percent:.2f}%

## 📊 整体统计

| 指标 | 数值 |
|------|------|
| 覆盖率 | **{total_percent:.2f}%** |
| 已覆盖行数 | {total_covered:,} |
| 总行数 | {total_lines:,} |
| 文件数 | {file_count} |
| **距离80%目标** | **{80 - total_percent:.2f}%** |

## 📋 各模块覆盖率详情

| 模块 | 覆盖率 | 已覆盖行 | 总行数 | 文件数 | 状态 |
|------|--------|----------|--------|--------|------|
"""
    
    for module, stats in sorted_modules:
        coverage = (stats['covered'] / stats['total'] * 100) if stats['total'] > 0 else 0
        status = "✅" if coverage >= 80 else "⚠️" if coverage >= 50 else "❌"
        md_content += f"| {module} | {coverage:.2f}% | {stats['covered']:,} | {stats['total']:,} | {len(stats['files'])} | {status} |\n"
    
    # 添加问题分析
    md_content += f"""
## 🔍 关键发现

### 1. 真实覆盖率远低于预期

- **实际覆盖率**: {total_percent:.2f}%
- **目标覆盖率**: 80%
- **差距**: {80 - total_percent:.2f}%

### 2. 测试失败分析

- **通过测试**: 1557 个
- **失败测试**: 72 个
- **错误测试**: 5 个
- **失败率**: {72/(1557+72)*100:.1f}%

### 3. 主要问题

#### 测试失败原因：
1. **配置模块测试失败** - TypedConfigBase、ConfigFactory等
2. **缓存模块测试失败** - 分布式缓存、并发测试等
3. **API模块测试失败** - 文档增强器等

#### 低覆盖率模块：
"""
    
    for module, stats in sorted_modules:
        coverage = (stats['covered'] / stats['total'] * 100) if stats['total'] > 0 else 0
        if coverage < 50:
            md_content += f"- **{module}**: {coverage:.2f}%\n"
    
    md_content += """
## 🎯 改进建议

### 立即行动（优先级高）

1. **修复失败的测试**
   - 重点修复config、cache、api模块的72个失败测试
   - 确保测试用例的正确性和稳定性

2. **提升低覆盖率模块**
   - 聚焦<50%覆盖率的模块
   - 创建针对性的测试用例

### 中期目标

1. **分模块达标**
   - 每个模块至少达到60%覆盖率
   - 核心模块（core、config、cache）达到80%

2. **测试质量提升**
   - 减少mock依赖
   - 增加集成测试
   - 提高测试的实际价值

### 长期规划

1. **持续监控**
   - 建立覆盖率CI检查
   - 防止覆盖率下降

2. **全面达标**
   - 整体覆盖率达到80%
   - 满足投产要求

## 📝 总结

当前基础设施层覆盖率为 **{total_percent:.2f}%**，距离80%目标还有 **{80 - total_percent:.2f}%** 的差距。

主要原因是：
1. 存在72个失败的测试，影响实际覆盖效果
2. 部分模块覆盖率极低（<50%）
3. 需要修复测试用例并补充有效的测试

**建议下一步**：优先修复失败的测试，然后针对性地补充低覆盖率模块的测试用例。
"""
    
    with open('test_logs/🎯真实覆盖率评估报告.md', 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"📄 Markdown报告已保存: test_logs/🎯真实覆盖率评估报告.md")

if __name__ == "__main__":
    extract_coverage()

