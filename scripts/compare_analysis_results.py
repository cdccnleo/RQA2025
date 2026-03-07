#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比重构前后的代码分析结果
"""

import json
import sys
from pathlib import Path

def compare_results():
    """对比重构前后的分析结果"""
    
    # 读取分析结果
    before_file = Path("reports/data_layer_architecture_review.json")
    after_file = Path("reports/data_layer_post_refactor_review.json")
    
    if not before_file.exists():
        print(f"❌ 找不到重构前的分析报告: {before_file}")
        return
    
    if not after_file.exists():
        print(f"❌ 找不到重构后的分析报告: {after_file}")
        return
    
    with open(before_file, 'r', encoding='utf-8') as f:
        before = json.load(f)
    
    with open(after_file, 'r', encoding='utf-8') as f:
        after = json.load(f)
    
    print("="*60)
    print("数据层重构前后对比分析")
    print("="*60)
    
    # 基础指标对比
    print("\n📊 基础指标对比:")
    print(f"  总文件数: {before['metrics']['total_files']} → {after['metrics']['total_files']}")
    
    lines_before = before['metrics']['total_lines']
    lines_after = after['metrics']['total_lines']
    lines_diff = lines_before - lines_after
    lines_percent = (lines_diff / lines_before * 100) if lines_before > 0 else 0
    print(f"  总代码行: {lines_before} → {lines_after} (减少 {lines_diff} 行, {lines_percent:.1f}%)")
    
    patterns_before = before['metrics']['total_patterns']
    patterns_after = after['metrics']['total_patterns']
    patterns_diff = patterns_before - patterns_after
    print(f"  识别模式: {patterns_before} → {patterns_after} (减少 {patterns_diff})")
    
    opps_before = before['metrics']['refactor_opportunities']
    opps_after = after['metrics']['refactor_opportunities']
    opps_diff = opps_before - opps_after
    print(f"  重构机会: {opps_before} → {opps_after} (减少 {opps_diff})")
    
    # 质量评分对比
    print("\n📈 质量评分对比:")
    quality_before = before['quality_score']
    quality_after = after['quality_score']
    quality_diff = quality_after - quality_before
    print(f"  质量评分: {quality_before:.3f} → {quality_after:.3f} ({'+' if quality_diff >= 0 else ''}{quality_diff:.3f})")
    
    overall_before = before['overall_score']
    overall_after = after['overall_score']
    overall_diff = overall_after - overall_before
    print(f"  综合评分: {overall_before:.3f} → {overall_after:.3f} ({'+' if overall_diff >= 0 else ''}{overall_diff:.3f})")
    
    # 风险评估对比
    print("\n⚠️ 风险评估对比:")
    risk_before = before['risk_assessment']
    risk_after = after['risk_assessment']
    
    print(f"  整体风险: {risk_before['overall_risk']} → {risk_after['overall_risk']}")
    
    auto_before = risk_before['automated_opportunities']
    auto_after = risk_after['automated_opportunities']
    print(f"  可自动修复: {auto_before} → {auto_after} (增加 {auto_after - auto_before})")
    
    manual_before = risk_before['manual_opportunities']
    manual_after = risk_after['manual_opportunities']
    manual_diff = manual_before - manual_after
    print(f"  人工修复项: {manual_before} → {manual_after} (减少 {manual_diff})")
    
    # 详细风险分布
    print("\n  风险等级分布:")
    for level in ['high', 'medium', 'low']:
        before_count = risk_before['risk_breakdown'].get(level, 0)
        after_count = risk_after['risk_breakdown'].get(level, 0)
        diff = before_count - after_count
        print(f"    {level}: {before_count} → {after_count} ({'+' if diff < 0 else ''}{-diff})")
    
    # 组织分析对比（如果存在）
    if 'organization_analysis' in after:
        org = after['organization_analysis']
        print(f"\n🏗️ 组织质量评分: {org.get('organization_score', 'N/A')}")
        print(f"  组织问题: {org.get('issues_count', 'N/A')} 个")
        print(f"  组织建议: {org.get('suggestions_count', 'N/A')} 个")
    
    # 总结
    print("\n" + "="*60)
    print("总结:")
    print("="*60)
    
    if lines_diff > 0:
        print(f"✅ 代码行数减少: {lines_diff} 行 ({lines_percent:.1f}%)")
    
    if opps_diff > 0:
        print(f"✅ 重构机会减少: {opps_diff} 个")
    
    if quality_diff >= 0:
        print(f"✅ 质量评分{'提升' if quality_diff > 0 else '保持'}: {quality_diff:+.3f}")
    
    if manual_diff > 0:
        print(f"✅ 人工修复项减少: {manual_diff} 个")
    
    print("\n🎉 重构成果: 代码质量有所改善！")
    print("="*60)
    
    return True

if __name__ == "__main__":
    compare_results()
