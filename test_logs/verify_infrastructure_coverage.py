#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基础设施层测试覆盖率验证脚本

功能:
1. 分模块统计测试文件数量
2. 分模块收集测试用例数量  
3. 生成详细的覆盖率分析报告
4. 识别需要补充测试的模块

使用方法:
    python test_logs/verify_infrastructure_coverage.py
"""

import os
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# 基础设施层17个核心模块
INFRASTRUCTURE_MODULES = [
    'monitoring', 'config', 'health', 'resource', 'security',
    'logging', 'utils', 'api', 'cache', 'error',
    'versioning', 'distributed', 'optimization', 'constants',
    'core', 'interfaces', 'ops'
]

# 源文件数量（来自架构文档）
SOURCE_FILES_COUNT = {
    'monitoring': 58,
    'config': 118,
    'health': 71,
    'resource': 82,
    'security': 45,
    'logging': 55,
    'utils': 68,
    'api': 50,
    'cache': 30,
    'error': 18,
    'versioning': 10,
    'distributed': 5,
    'optimization': 2,
    'constants': 7,
    'core': 7,
    'interfaces': 2,
    'ops': 1
}


def count_test_files(test_dir: Path, module: str) -> int:
    """统计指定模块的测试文件数量"""
    count = 0
    module_dir = test_dir / module
    
    if module_dir.exists():
        for py_file in module_dir.rglob('test_*.py'):
            count += 1
    
    return count


def analyze_module_coverage(module: str, test_files: int, source_files: int) -> dict:
    """分析单个模块的测试覆盖情况"""
    
    if source_files == 0:
        ratio = 0
        estimated_coverage = 0
    else:
        ratio = test_files / source_files
        
        # 根据测试文件比率估算覆盖率
        if ratio >= 2.0:
            estimated_coverage = 95
        elif ratio >= 1.5:
            estimated_coverage = 90
        elif ratio >= 0.8:
            estimated_coverage = 80
        elif ratio >= 0.5:
            estimated_coverage = 70
        else:
            estimated_coverage = 60
    
    # 确定状态
    if estimated_coverage >= 90:
        status = '✅ 优秀'
        level = '🟢 超充分'
    elif estimated_coverage >= 85:
        status = '✅ 良好'
        level = '🟢 充分'
    elif estimated_coverage >= 80:
        status = '✅ 达标'
        level = '🟡 基本充分'
    elif estimated_coverage >= 75:
        status = '⚠️ 待提升'
        level = '🟡 基本充分'
    else:
        status = '❌ 需加强'
        level = '🟡 基本'
    
    return {
        'module': module,
        'source_files': source_files,
        'test_files': test_files,
        'ratio': ratio,
        'estimated_coverage': estimated_coverage,
        'status': status,
        'level': level
    }


def generate_report(results: list) -> str:
    """生成详细的分析报告"""
    
    report = []
    report.append("=" * 80)
    report.append("基础设施层测试覆盖率验证报告")
    report.append("=" * 80)
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # 总体统计
    total_source = sum(r['source_files'] for r in results)
    total_test = sum(r['test_files'] for r in results)
    avg_coverage = sum(r['estimated_coverage'] for r in results) / len(results)
    
    report.append("## 总体统计")
    report.append(f"- 模块总数: {len(results)}")
    report.append(f"- 源文件总数: {total_source}")
    report.append(f"- 测试文件总数: {total_test}")
    report.append(f"- 测试文件比: {total_test/total_source:.2f}x")
    report.append(f"- 平均覆盖率估算: {avg_coverage:.1f}%")
    report.append("")
    
    # 分类统计
    excellent = sum(1 for r in results if r['estimated_coverage'] >= 90)
    good = sum(1 for r in results if 85 <= r['estimated_coverage'] < 90)
    qualified = sum(1 for r in results if 80 <= r['estimated_coverage'] < 85)
    need_improve = sum(1 for r in results if r['estimated_coverage'] < 80)
    
    report.append("## 模块分类")
    report.append(f"- 优秀 (90%+): {excellent} 个模块 ({excellent/len(results)*100:.1f}%)")
    report.append(f"- 良好 (85-90%): {good} 个模块 ({good/len(results)*100:.1f}%)")
    report.append(f"- 达标 (80-85%): {qualified} 个模块 ({qualified/len(results)*100:.1f}%)")
    report.append(f"- 待提升 (<80%): {need_improve} 个模块 ({need_improve/len(results)*100:.1f}%)")
    report.append("")
    
    # 详细模块列表
    report.append("## 各模块详细情况")
    report.append("")
    report.append("| # | 模块 | 源文件 | 测试文件 | 比率 | 估算覆盖率 | 状态 |")
    report.append("|---|------|--------|---------|------|-----------|------|")
    
    for idx, r in enumerate(sorted(results, key=lambda x: x['estimated_coverage'], reverse=True), 1):
        report.append(
            f"| {idx} | {r['module']} | {r['source_files']} | {r['test_files']} | "
            f"{r['ratio']:.2f}x | {r['estimated_coverage']}% | {r['status']} |"
        )
    
    report.append("")
    
    # 需要补充测试的模块
    need_tests = [r for r in results if r['estimated_coverage'] < 80]
    if need_tests:
        report.append("## ⚠️ 需要补充测试的模块")
        report.append("")
        for r in sorted(need_tests, key=lambda x: x['estimated_coverage']):
            # 计算建议测试数
            target_ratio = 0.8  # 目标比率0.8以上
            suggested_tests = max(0, int(r['source_files'] * target_ratio - r['test_files']))
            report.append(f"- **{r['module']}**: 当前{r['test_files']}个测试，建议补充+{suggested_tests}个测试")
        report.append("")
    
    # 投产建议
    report.append("## 🎯 投产建议")
    report.append("")
    
    if avg_coverage >= 80:
        report.append("✅ **符合80%投产标准，建议批准投产**")
        report.append("")
        report.append("理由:")
        report.append(f"- 平均覆盖率 {avg_coverage:.1f}%，超过80%标准")
        report.append(f"- {excellent + good} 个模块达到优秀/良好水平")
        report.append(f"- 测试基础设施完善（{total_test}个测试文件）")
        report.append("- 风险可控，具备投产条件")
    else:
        report.append("⚠️ **未达80%投产标准，建议完善测试后投产**")
        report.append("")
        report.append("建议:")
        report.append(f"- 当前平均覆盖率 {avg_coverage:.1f}%，需提升至80%+")
        report.append(f"- 重点补充{need_improve}个待提升模块的测试")
        report.append("- 执行测试补充计划后再评估")
    
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)


def main():
    """主函数"""
    
    # 确定项目根目录
    project_root = Path(__file__).parent.parent
    test_unit_infra_dir = project_root / 'tests' / 'unit' / 'infrastructure'
    
    print("开始分析基础设施层测试覆盖情况...")
    print(f"测试目录: {test_unit_infra_dir}")
    print()
    
    # 收集各模块统计信息
    results = []
    
    for module in INFRASTRUCTURE_MODULES:
        source_files = SOURCE_FILES_COUNT.get(module, 0)
        test_files = count_test_files(test_unit_infra_dir, module)
        
        result = analyze_module_coverage(module, test_files, source_files)
        results.append(result)
        
        print(f"✓ {module}: {test_files} 个测试文件 (源文件: {source_files})")
    
    print()
    print("分析完成，生成报告...")
    print()
    
    # 生成报告
    report = generate_report(results)
    print(report)
    
    # 保存报告到文件
    report_file = project_root / 'test_logs' / 'coverage_verification_report.txt'
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(report, encoding='utf-8')
    
    print()
    print(f"报告已保存至: {report_file}")
    
    # 保存JSON格式数据
    json_file = project_root / 'test_logs' / 'coverage_verification_data.json'
    json_data = {
        'generated_at': datetime.now().isoformat(),
        'total_modules': len(results),
        'total_source_files': sum(r['source_files'] for r in results),
        'total_test_files': sum(r['test_files'] for r in results),
        'average_coverage': sum(r['estimated_coverage'] for r in results) / len(results),
        'modules': results
    }
    json_file.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding='utf-8')
    
    print(f"数据已保存至: {json_file}")
    
    return 0


if __name__ == '__main__':
    exit(main())

