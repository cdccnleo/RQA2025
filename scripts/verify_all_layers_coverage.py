#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
21层级测试覆盖率逐一验证脚本
验证每个层级是否达到80%覆盖率投产标准
"""

import subprocess
import re
from datetime import datetime
from pathlib import Path


# 定义21个层级的测试和源代码路径
LAYERS = [
    # Infrastructure层
    {"name": "versioning", "test_path": "tests/unit/infrastructure/versioning", "src_path": "src/infrastructure/versioning"},
    {"name": "monitoring", "test_path": "tests/unit/infrastructure/monitoring", "src_path": "src/infrastructure/monitoring"},
    {"name": "ops", "test_path": "tests/unit/infrastructure/ops", "src_path": "src/infrastructure/ops"},
    
    # Core层
    {"name": "core", "test_path": "tests/unit/core", "src_path": "src/core"},
    {"name": "data", "test_path": "tests/unit/data", "src_path": "src/data"},
    {"name": "gateway", "test_path": "tests/unit/gateway", "src_path": "src/gateway"},
    
    # Features层
    {"name": "features", "test_path": "tests/unit/features", "src_path": "src/features"},
    
    # ML层
    {"name": "ml", "test_path": "tests/unit/ml", "src_path": "src/ml"},
    
    # Strategy + Trading + Risk
    {"name": "strategy", "test_path": "tests/unit/strategy", "src_path": "src/strategy"},
    {"name": "trading", "test_path": "tests/unit/trading", "src_path": "src/trading"},
    {"name": "risk", "test_path": "tests/unit/risk", "src_path": "src/risk"},
    
    # 其他层级
    {"name": "distributed", "test_path": "tests/unit/distributed", "src_path": "src/distributed"},
    {"name": "adapters", "test_path": "tests/unit/adapters", "src_path": "src/adapters"},
    {"name": "boundary", "test_path": "tests/unit/boundary", "src_path": "src/boundary"},
    {"name": "security", "test_path": "tests/unit/security", "src_path": "src/security"},
    {"name": "utils", "test_path": "tests/unit/utils", "src_path": "src/utils"},
    {"name": "async", "test_path": "tests/unit/async", "src_path": "src/async"},
    {"name": "optimization", "test_path": "tests/unit/optimization", "src_path": "src/optimization"},
    {"name": "automation", "test_path": "tests/unit/automation", "src_path": "src/automation"},
    {"name": "streaming", "test_path": "tests/unit/streaming", "src_path": "src/streaming"},
    {"name": "resilience", "test_path": "tests/unit/resilience", "src_path": "src/resilience"},
    {"name": "mobile", "test_path": "tests/unit/mobile", "src_path": "src/mobile"},
]


def run_coverage_for_layer(layer):
    """为单个层级运行覆盖率测试"""
    name = layer["name"]
    test_path = layer["test_path"]
    src_path = layer["src_path"]
    
    print(f"\n{'='*80}")
    print(f"验证层级: {name}")
    print(f"测试路径: {test_path}")
    print(f"源码路径: {src_path}")
    print(f"{'='*80}")
    
    # 检查路径是否存在
    if not Path(test_path).exists():
        return {
            "name": name,
            "status": "SKIP",
            "reason": "测试路径不存在",
            "coverage": 0,
            "tests_passed": 0,
            "tests_total": 0
        }
    
    if not Path(src_path).exists():
        return {
            "name": name,
            "status": "SKIP",
            "reason": "源码路径不存在",
            "coverage": 0,
            "tests_passed": 0,
            "tests_total": 0
        }
    
    # 运行pytest覆盖率测试
    cmd = [
        "pytest",
        test_path,
        f"--cov={src_path}",
        "--cov-report=term",
        "-v",
        "--tb=no"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        output = result.stdout + result.stderr
        
        # 提取覆盖率
        coverage_match = re.search(r'TOTAL\s+\d+\s+\d+\s+(\d+)%', output)
        coverage = int(coverage_match.group(1)) if coverage_match else 0
        
        # 提取测试结果
        test_match = re.search(r'(\d+) passed', output)
        tests_passed = int(test_match.group(1)) if test_match else 0
        
        failed_match = re.search(r'(\d+) failed', output)
        tests_failed = int(failed_match.group(1)) if failed_match else 0
        
        tests_total = tests_passed + tests_failed
        
        # 判断是否达标
        is_qualified = coverage >= 80 and result.returncode == 0
        
        print(f"\n✅ 测试结果:")
        print(f"   覆盖率: {coverage}%")
        print(f"   测试通过: {tests_passed}/{tests_total}")
        print(f"   达标状态: {'✅ 达标' if is_qualified else '❌ 不达标'}")
        
        return {
            "name": name,
            "status": "PASS" if is_qualified else "FAIL",
            "coverage": coverage,
            "tests_passed": tests_passed,
            "tests_total": tests_total,
            "output": output
        }
        
    except subprocess.TimeoutExpired:
        return {
            "name": name,
            "status": "TIMEOUT",
            "reason": "执行超时",
            "coverage": 0,
            "tests_passed": 0,
            "tests_total": 0
        }
    except Exception as e:
        return {
            "name": name,
            "status": "ERROR",
            "reason": str(e),
            "coverage": 0,
            "tests_passed": 0,
            "tests_total": 0
        }


def generate_report(results):
    """生成验证报告"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"reports/coverage_layered/21层级覆盖率验证结果_{timestamp}.md"
    
    # 统计
    total_layers = len(results)
    passed_layers = sum(1 for r in results if r["status"] == "PASS")
    failed_layers = sum(1 for r in results if r["status"] == "FAIL")
    skipped_layers = sum(1 for r in results if r["status"] == "SKIP")
    
    avg_coverage = sum(r["coverage"] for r in results) / total_layers if total_layers > 0 else 0
    total_tests = sum(r["tests_total"] for r in results)
    passed_tests = sum(r["tests_passed"] for r in results)
    
    # 生成Markdown报告
    report = f"""# 21层级测试覆盖率验证结果

**验证时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**验证工具**: pytest + pytest-cov  
**投产标准**: 覆盖率≥80%, 测试通过率100%

---

## 📊 验证总览

| 指标 | 数值 | 状态 |
|------|------|------|
| 总层级数 | {total_layers} | - |
| 达标层级 | {passed_layers} | {'✅' if passed_layers == total_layers else '⚠️'} |
| 未达标层级 | {failed_layers} | {'✅' if failed_layers == 0 else '❌'} |
| 跳过层级 | {skipped_layers} | - |
| 达标率 | {passed_layers/total_layers*100:.1f}% | {'✅' if passed_layers == total_layers else '❌'} |
| 平均覆盖率 | {avg_coverage:.1f}% | {'✅' if avg_coverage >= 80 else '❌'} |
| 总测试数 | {total_tests} | - |
| 通过测试数 | {passed_tests} | - |
| 测试通过率 | {passed_tests/total_tests*100:.1f}% | {'✅' if passed_tests == total_tests else '❌'} |

---

## 📋 分层级验证结果

| # | 层级 | 覆盖率 | 测试通过 | 状态 |
|---|------|--------|----------|------|
"""
    
    for i, result in enumerate(results, 1):
        status_icon = {
            "PASS": "✅",
            "FAIL": "❌",
            "SKIP": "⏭️",
            "TIMEOUT": "⏱️",
            "ERROR": "⚠️"
        }.get(result["status"], "❓")
        
        coverage_str = f"{result['coverage']}%" if result["coverage"] > 0 else "N/A"
        tests_str = f"{result['tests_passed']}/{result['tests_total']}" if result["tests_total"] > 0 else "N/A"
        
        report += f"| {i} | {result['name']} | {coverage_str} | {tests_str} | {status_icon} {result['status']} |\n"
    
    # 详细结果
    report += "\n---\n\n## 📝 详细验证结果\n\n"
    
    for result in results:
        status_icon = {
            "PASS": "✅",
            "FAIL": "❌",
            "SKIP": "⏭️",
            "TIMEOUT": "⏱️",
            "ERROR": "⚠️"
        }.get(result["status"], "❓")
        
        report += f"### {status_icon} {result['name']}\n\n"
        report += f"- **状态**: {result['status']}\n"
        report += f"- **覆盖率**: {result['coverage']}%\n"
        report += f"- **测试通过**: {result['tests_passed']}/{result['tests_total']}\n"
        
        if result["status"] == "PASS":
            report += f"- **结论**: ✅ 达标（覆盖率≥80%）\n"
        elif result["status"] == "FAIL":
            report += f"- **结论**: ❌ 未达标（覆盖率<80%或测试失败）\n"
        elif result["status"] == "SKIP":
            report += f"- **原因**: {result.get('reason', 'Unknown')}\n"
        
        report += "\n"
    
    # 总结
    report += "---\n\n## 🎯 验证结论\n\n"
    
    if passed_layers == total_layers and avg_coverage >= 80:
        report += "### ✅ 全部达标！\n\n"
        report += f"- ✅ {total_layers}/{total_layers}个层级全部达到80%+覆盖率\n"
        report += f"- ✅ 平均覆盖率{avg_coverage:.1f}%\n"
        report += f"- ✅ 测试通过率{passed_tests/total_tests*100:.1f}%\n"
        report += "\n**项目状态**: 🎉 **Ready for Production**\n"
    else:
        report += "### ⚠️ 存在未达标层级\n\n"
        report += f"- 达标层级: {passed_layers}/{total_layers}\n"
        report += f"- 未达标层级: {failed_layers}\n"
        report += f"- 平均覆盖率: {avg_coverage:.1f}%\n"
        report += "\n**需要**: 继续提升未达标层级的覆盖率\n"
    
    report += f"\n---\n\n*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
    
    # 保存报告
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n\n📄 验证报告已生成: {report_path}")
    
    return report_path


def main():
    """主函数"""
    print("="*80)
    print("RQA2025项目 - 21层级测试覆盖率逐一验证")
    print("="*80)
    print(f"\n开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"验证层级数: {len(LAYERS)}")
    print(f"投产标准: 覆盖率≥80%, 测试通过率100%")
    
    results = []
    
    for i, layer in enumerate(LAYERS, 1):
        print(f"\n\n[{i}/{len(LAYERS)}] 正在验证: {layer['name']}")
        result = run_coverage_for_layer(layer)
        results.append(result)
    
    # 生成报告
    print("\n\n" + "="*80)
    print("验证完成，正在生成报告...")
    print("="*80)
    
    report_path = generate_report(results)
    
    # 输出摘要
    passed = sum(1 for r in results if r["status"] == "PASS")
    total = len(results)
    avg_cov = sum(r["coverage"] for r in results) / total
    
    print(f"\n\n{'='*80}")
    print("验证摘要")
    print(f"{'='*80}")
    print(f"✅ 达标层级: {passed}/{total}")
    print(f"📊 平均覆盖率: {avg_cov:.1f}%")
    print(f"📄 详细报告: {report_path}")
    
    if passed == total:
        print(f"\n🎉 恭喜！所有{total}个层级全部达标，可以投产！")
    else:
        print(f"\n⚠️  还有{total-passed}个层级未达标，需要继续提升")
    
    return results


if __name__ == "__main__":
    main()

