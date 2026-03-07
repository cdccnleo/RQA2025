#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 各层测试覆盖率检查脚本
检查所有架构层的测试覆盖率是否达到投产要求
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
REPORTS_DIR = PROJECT_ROOT / "test_logs"

# 各层投产覆盖率要求（基于PRODUCTION_TEST_PLAN.md）
PRODUCTION_REQUIREMENTS = {
    "基础设施层": {"target": 95, "module": "src/infrastructure"},
    "核心服务层": {"target": 95, "module": "src/core"},
    "数据管理层": {"target": 95, "module": "src/data"},
    "特征分析层": {"target": 90, "module": "src/features"},
    "机器学习层": {"target": 90, "module": "src/ml"},
    "策略服务层": {"target": 85, "module": "src/backtest"},
    "交易层": {"target": 85, "module": "src/trading"},
    "风险控制层": {"target": 90, "module": "src/risk"},
    "监控层": {"target": 90, "module": "src/monitoring"},
    "流处理层": {"target": 80, "module": "src/streaming"},
    "网关层": {"target": 80, "module": "src/gateway"},
    "优化层": {"target": 80, "module": "src/optimization"},
    "适配器层": {"target": 80, "module": "src/adapters"},
    "自动化层": {"target": 80, "module": "src/automation"},
    "弹性层": {"target": 80, "module": "src/resilience"},
    "测试层": {"target": 80, "module": "src/testing"},
    "工具层": {"target": 80, "module": "src/utils"},
    "分布式协调器": {"target": 80, "module": "src/core/distributed"},
    "异步处理器": {"target": 80, "module": "src/core/async"},
    "移动端层": {"target": 80, "module": "src/mobile"},
    "业务边界层": {"target": 80, "module": "src/core/boundary"},
}

def run_coverage_for_layer(layer_name: str, module_path: str, target: int) -> Dict:
    """为指定层运行覆盖率测试"""
    print(f"\n{'='*80}")
    print(f"🔍 检查层级: {layer_name}")
    print(f"📁 模块路径: {module_path}")
    print(f"🎯 目标覆盖率: {target}%")
    print(f"{'='*80}")
    
    # 检查模块是否存在
    module_full_path = PROJECT_ROOT / module_path
    if not module_full_path.exists():
        return {
            "layer": layer_name,
            "status": "SKIPPED",
            "coverage": 0.0,
            "target": target,
            "gap": -target,
            "reason": f"模块路径不存在: {module_path}",
            "tests_total": 0,
            "tests_passed": 0,
            "tests_failed": 0
        }
    
    # 生成覆盖率JSON文件
    json_file = REPORTS_DIR / f"coverage_{layer_name.lower().replace(' ', '_')}.json"
    
    # 运行pytest覆盖率测试
    cmd = [
        sys.executable, "-m", "pytest",
        f"--cov={module_path}",
        "--cov-report=json:" + str(json_file),
        "--cov-report=term-missing",
        "-n", "auto",
        "-k", "not e2e",
        "--tb=short",
        "-q"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        # 解析覆盖率结果
        coverage_data = None
        if json_file.exists():
            with open(json_file, 'r', encoding='utf-8') as f:
                coverage_data = json.load(f)
        
        # 提取覆盖率
        coverage_percent = 0.0
        if coverage_data and 'totals' in coverage_data:
            coverage_percent = coverage_data['totals'].get('percent_covered', 0.0)
        
        # 解析测试结果
        tests_total = 0
        tests_passed = 0
        tests_failed = 0
        
        if result.returncode == 0 or "passed" in result.stdout.lower():
            # 尝试从输出中提取测试统计
            lines = result.stdout.split('\n')
            for line in lines:
                if "passed" in line.lower() and "failed" in line.lower():
                    # 解析类似 "10 passed, 2 failed" 的格式
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.isdigit() and i + 1 < len(parts):
                            if "passed" in parts[i+1].lower():
                                tests_passed = int(part)
                            elif "failed" in parts[i+1].lower():
                                tests_failed = int(part)
                    tests_total = tests_passed + tests_failed
        
        gap = coverage_percent - target
        status = "PASSED" if coverage_percent >= target else "FAILED"
        
        return {
            "layer": layer_name,
            "status": status,
            "coverage": round(coverage_percent, 2),
            "target": target,
            "gap": round(gap, 2),
            "tests_total": tests_total,
            "tests_passed": tests_passed,
            "tests_failed": tests_failed,
            "module_path": module_path
        }
        
    except subprocess.TimeoutExpired:
        return {
            "layer": layer_name,
            "status": "TIMEOUT",
            "coverage": 0.0,
            "target": target,
            "gap": -target,
            "reason": "测试执行超时",
            "tests_total": 0,
            "tests_passed": 0,
            "tests_failed": 0
        }
    except Exception as e:
        return {
            "layer": layer_name,
            "status": "ERROR",
            "coverage": 0.0,
            "target": target,
            "gap": -target,
            "reason": str(e),
            "tests_total": 0,
            "tests_passed": 0,
            "tests_failed": 0
        }

def generate_report(results: List[Dict]) -> str:
    """生成覆盖率检查报告"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 统计信息
    total_layers = len(results)
    passed_layers = sum(1 for r in results if r["status"] == "PASSED")
    failed_layers = sum(1 for r in results if r["status"] == "FAILED")
    skipped_layers = sum(1 for r in results if r["status"] in ["SKIPPED", "ERROR", "TIMEOUT"])
    
    # 计算平均覆盖率
    valid_results = [r for r in results if r["coverage"] > 0]
    avg_coverage = sum(r["coverage"] for r in valid_results) / len(valid_results) if valid_results else 0
    
    report = f"""# RQA2025 各层测试覆盖率检查报告

## 📊 检查概览

**检查时间**: {timestamp}  
**检查层级总数**: {total_layers}  
**达标层级**: {passed_layers} ✅  
**未达标层级**: {failed_layers} ⚠️  
**跳过/错误层级**: {skipped_layers} ⏭️  
**平均覆盖率**: {avg_coverage:.2f}%

## 🎯 投产要求标准

| 层级类型 | 覆盖率要求 |
|---------|-----------|
| 核心业务层 | ≥85-90% |
| 核心支撑层 | ≥90-95% |
| 辅助支撑层 | ≥80% |

## 📈 各层覆盖率详情

| 层级 | 当前覆盖率 | 目标覆盖率 | 差距 | 状态 | 测试用例 |
|------|-----------|-----------|------|------|----------|
"""
    
    for result in results:
        status_icon = {
            "PASSED": "✅",
            "FAILED": "⚠️",
            "SKIPPED": "⏭️",
            "ERROR": "❌",
            "TIMEOUT": "⏱️"
        }.get(result["status"], "❓")
        
        tests_info = f"{result['tests_passed']}/{result['tests_total']}"
        if result["tests_total"] == 0:
            tests_info = "N/A"
        
        gap_str = f"{result['gap']:+.2f}%"
        if result["status"] in ["SKIPPED", "ERROR", "TIMEOUT"]:
            gap_str = result.get("reason", "N/A")
        
        report += f"| {result['layer']} | {result['coverage']:.2f}% | {result['target']}% | {gap_str} | {status_icon} {result['status']} | {tests_info} |\n"
    
    # 未达标层级详情
    failed_results = [r for r in results if r["status"] == "FAILED"]
    if failed_results:
        report += f"\n## ⚠️ 未达标层级详情\n\n"
        for result in failed_results:
            report += f"### {result['layer']}\n"
            report += f"- **当前覆盖率**: {result['coverage']:.2f}%\n"
            report += f"- **目标覆盖率**: {result['target']}%\n"
            report += f"- **差距**: {result['gap']:.2f}%\n"
            report += f"- **模块路径**: {result['module_path']}\n"
            report += f"- **测试用例**: {result['tests_passed']}/{result['tests_total']}\n\n"
    
    # 跳过层级详情
    skipped_results = [r for r in results if r["status"] in ["SKIPPED", "ERROR", "TIMEOUT"]]
    if skipped_results:
        report += f"\n## ⏭️ 跳过/错误层级详情\n\n"
        for result in skipped_results:
            report += f"### {result['layer']}\n"
            report += f"- **状态**: {result['status']}\n"
            report += f"- **原因**: {result.get('reason', 'N/A')}\n"
            report += f"- **模块路径**: {result.get('module_path', 'N/A')}\n\n"
    
    # 总结
    report += f"\n## 📋 检查总结\n\n"
    
    if passed_layers == total_layers:
        report += "✅ **所有层级均已达到投产要求！**\n\n"
    else:
        report += f"⚠️ **共有 {failed_layers} 个层级未达到投产要求**\n\n"
        report += "### 改进建议\n\n"
        report += "1. **优先处理核心业务层**: 策略服务层、交易层、风险控制层\n"
        report += "2. **提升核心支撑层**: 数据管理层、机器学习层、基础设施层\n"
        report += "3. **完善辅助支撑层**: 监控层、优化层、网关层\n"
        report += "4. **补充测试用例**: 针对低覆盖率模块增加测试用例\n"
        report += "5. **优化测试执行**: 提高测试执行效率和稳定性\n\n"
    
    report += f"\n---\n\n"
    report += f"*报告生成时间: {timestamp}*\n"
    report += f"*检查脚本: scripts/check_all_layers_coverage.py*\n"
    
    return report

def main():
    """主函数"""
    print("🚀 RQA2025 各层测试覆盖率检查")
    print("=" * 80)
    
    # 确保报告目录存在
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 检查各层覆盖率
    results = []
    for layer_name, config in PRODUCTION_REQUIREMENTS.items():
        result = run_coverage_for_layer(
            layer_name,
            config["module"],
            config["target"]
        )
        results.append(result)
    
    # 生成报告
    report = generate_report(results)
    
    # 保存报告
    report_file = REPORTS_DIR / f"LAYERS_COVERAGE_CHECK_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n" + "=" * 80)
    print(f"📄 报告已保存: {report_file}")
    print("=" * 80)
    
    # 打印摘要
    print("\n📊 检查摘要:")
    passed = sum(1 for r in results if r["status"] == "PASSED")
    failed = sum(1 for r in results if r["status"] == "FAILED")
    print(f"  ✅ 达标: {passed}/{len(results)}")
    print(f"  ⚠️  未达标: {failed}/{len(results)}")
    
    # 返回状态码
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())










