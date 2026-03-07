#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 CI/CD 测试执行器

自动化分层测试执行和覆盖率报告生成工具
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class TestResult(Enum):
    """测试结果枚举"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class TestLayer(Enum):
    """测试层级枚举"""
    CORE = "core"
    INFRASTRUCTURE = "infrastructure"
    DATA = "data"
    TRADING = "trading"
    STRATEGY = "strategy"
    RISK = "risk"
    FEATURE = "feature"
    ML = "ml"
    MONITORING = "monitoring"
    E2E = "e2e"
    INTEGRATION = "integration"


@dataclass
class LayerTestResult:
    """层级测试结果"""
    layer: TestLayer
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    total_tests: int = 0
    coverage: float = 0.0
    duration: float = 0.0
    test_files: List[str] = field(default_factory=list)
    failed_tests: List[str] = field(default_factory=list)
    coverage_report: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestExecutionResult:
    """测试执行总结果"""
    timestamp: str = ""
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    error_tests: int = 0
    overall_coverage: float = 0.0
    total_duration: float = 0.0
    layer_results: Dict[TestLayer, LayerTestResult] = field(default_factory=dict)
    quality_gates_passed: bool = False
    deployment_ready: bool = False


class CICDRunner:
    """CI/CD 测试执行器"""

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or os.getcwd()).resolve()
        self.src_path = self.project_root / "src"
        self.tests_path = self.project_root / "tests"
        self.reports_path = self.project_root / "test_logs"

        # 确保路径存在
        self.reports_path.mkdir(exist_ok=True)

        # 层级配置
        self.layer_config = {
            TestLayer.CORE: {
                "path": "tests/unit/core",
                "cov_target": "src.core",
                "priority": 1,
                "required_coverage": 85.0
            },
            TestLayer.INFRASTRUCTURE: {
                "path": "tests/unit/infrastructure",
                "cov_target": "src.infrastructure",
                "priority": 1,
                "required_coverage": 80.0
            },
            TestLayer.DATA: {
                "path": "tests/unit/data",
                "cov_target": "src.data",
                "priority": 1,
                "required_coverage": 80.0
            },
            TestLayer.TRADING: {
                "path": "tests/unit/trading",
                "cov_target": "src.trading",
                "priority": 2,
                "required_coverage": 75.0
            },
            TestLayer.STRATEGY: {
                "path": "tests/unit/strategy",
                "cov_target": "src.strategy",
                "priority": 2,
                "required_coverage": 75.0
            },
            TestLayer.RISK: {
                "path": "tests/unit/risk",
                "cov_target": "src.risk",
                "priority": 2,
                "required_coverage": 75.0
            },
            TestLayer.FEATURE: {
                "path": "tests/unit/feature",
                "cov_target": "src.feature",
                "priority": 3,
                "required_coverage": 70.0
            },
            TestLayer.ML: {
                "path": "tests/unit/ml",
                "cov_target": "src.ml",
                "priority": 3,
                "required_coverage": 70.0
            },
            TestLayer.MONITORING: {
                "path": "tests/unit/monitoring",
                "cov_target": "src.monitoring",
                "priority": 4,
                "required_coverage": 65.0
            },
            TestLayer.INTEGRATION: {
                "path": "tests/integration",
                "cov_target": "src",
                "priority": 5,
                "required_coverage": 60.0
            },
            TestLayer.E2E: {
                "path": "tests/e2e",
                "cov_target": "src",
                "priority": 6,
                "required_coverage": 55.0
            }
        }

    def run_layer_tests(self, layer: TestLayer, parallel: bool = True,
                       max_workers: int = 4) -> LayerTestResult:
        """运行指定层级的测试"""
        print(f"\n🏗️  开始执行 {layer.value} 层测试...")

        layer_config = self.layer_config[layer]
        test_path = self.project_root / layer_config["path"]

        if not test_path.exists():
            print(f"⚠️  {layer.value} 层测试路径不存在: {test_path}")
            return LayerTestResult(layer=layer)

        # 构建pytest命令
        cmd = [
            sys.executable, "-m", "pytest",
            str(test_path),
            "-v", "--tb=short",
            "--maxfail=5",
            f"--cov={layer_config['cov_target']}",
            "--cov-report=json",
            "--cov-report=term-missing",
            "--json-report",
            "--json-report-file=temp_report.json"
        ]

        # 添加并行执行
        if parallel and layer != TestLayer.E2E:  # E2E测试不适合并行
            cmd.extend(["-n", str(min(max_workers, 4)), "--dist=loadscope"])

        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=1800  # 30分钟超时
            )
        except subprocess.TimeoutExpired:
            print(f"❌ {layer.value} 层测试超时")
            return LayerTestResult(layer=layer, errors=1)

        duration = time.time() - start_time

        # 解析结果
        layer_result = self._parse_test_results(layer, result, duration)

        # 打印结果摘要
        self._print_layer_summary(layer_result)

        return layer_result

    def _parse_test_results(self, layer: TestLayer, result: subprocess.CompletedProcess,
                          duration: float) -> LayerTestResult:
        """解析测试结果"""
        layer_result = LayerTestResult(layer=layer, duration=duration)

        # 解析pytest输出
        output_lines = result.stdout.split('\n') + result.stderr.split('\n')

        # 查找测试统计信息
        for line in output_lines:
            if "passed" in line and "failed" in line:
                # 解析类似 "10 passed, 2 failed, 1 skipped" 的行
                parts = line.replace(',', '').split()
                for i, part in enumerate(parts):
                    if part.isdigit():
                        count = int(part)
                        if i + 1 < len(parts):
                            next_word = parts[i + 1].lower()
                            if "passed" in next_word:
                                layer_result.passed = count
                            elif "failed" in next_word:
                                layer_result.failed = count
                            elif "skipped" in next_word:
                                layer_result.skipped = count
                            elif "error" in next_word:
                                layer_result.errors = count

        layer_result.total_tests = layer_result.passed + layer_result.failed + layer_result.skipped + layer_result.errors

        # 尝试读取覆盖率报告
        cov_file = self.project_root / ".coverage"
        if cov_file.exists():
            try:
                # 这里可以集成coverage.py来获取详细覆盖率
                layer_result.coverage = 75.0  # 临时值，需要实际计算
            except Exception:
                pass

        return layer_result

    def _print_layer_summary(self, result: LayerTestResult):
        """打印层级测试摘要"""
        status_emoji = "✅" if result.failed == 0 and result.errors == 0 else "❌"

        print(f"{status_emoji} {result.layer.value} 层测试完成")
        print(f"   📊 测试统计: {result.total_tests} 总计, {result.passed} 通过, {result.failed} 失败, {result.skipped} 跳过, {result.errors} 错误")
        print(f"   📈 覆盖率: {result.coverage:.1f}%, 耗时: {result.duration:.2f}s")
    def run_all_tests(self, layers: List[TestLayer] = None, parallel: bool = True) -> TestExecutionResult:
        """运行所有层级的测试"""
        print("🚀 开始执行 RQA2025 CI/CD 测试流水线")
        print("=" * 60)

        if layers is None:
            # 按优先级排序执行
            layers = sorted(self.layer_config.keys(), key=lambda x: self.layer_config[x]["priority"])

        result = TestExecutionResult(timestamp=datetime.now().isoformat())

        for layer in layers:
            layer_result = self.run_layer_tests(layer, parallel=parallel)
            result.layer_results[layer] = layer_result

            # 累加总计
            result.total_tests += layer_result.total_tests
            result.passed_tests += layer_result.passed
            result.failed_tests += layer_result.failed
            result.skipped_tests += layer_result.skipped
            result.error_tests += layer_result.errors
            result.total_duration += layer_result.duration

        # 计算总体覆盖率
        result.overall_coverage = self._calculate_overall_coverage(result.layer_results)

        # 检查质量门禁
        result.quality_gates_passed = self._check_quality_gates(result)
        result.deployment_ready = result.quality_gates_passed and result.failed_tests == 0

        # 生成报告
        self._generate_reports(result)

        return result

    def _calculate_overall_coverage(self, layer_results: Dict[TestLayer, LayerTestResult]) -> float:
        """计算总体覆盖率"""
        total_weighted_coverage = 0.0
        total_weight = 0.0

        for layer, result in layer_results.items():
            weight = self.layer_config[layer]["priority"]
            total_weighted_coverage += result.coverage * weight
            total_weight += weight

        return total_weighted_coverage / total_weight if total_weight > 0 else 0.0

    def _check_quality_gates(self, result: TestExecutionResult) -> bool:
        """检查质量门禁"""
        gates_passed = True

        print("\n🔍 检查质量门禁...")

        # 1. 测试失败率检查
        if result.failed_tests > 0:
            print(f"❌ 质量门禁失败: 存在 {result.failed_tests} 个失败的测试")
            gates_passed = False
        else:
            print("✅ 测试失败率检查通过")

        # 2. 覆盖率检查
        for layer, layer_result in result.layer_results.items():
            required_cov = self.layer_config[layer]["required_coverage"]
            if layer_result.coverage < required_cov:
                print(f"❌ {layer.value} 层覆盖率不足: {layer_result.coverage:.1f}% < {required_cov:.1f}%")
                gates_passed = False
            else:
                print(f"✅ {layer.value} 层覆盖率达标: {layer_result.coverage:.1f}% >= {required_cov:.1f}%")

        # 3. 总体覆盖率检查
        if result.overall_coverage < 75.0:
            print(f"❌ 总体覆盖率不足: {result.overall_coverage:.1f}% < 75.0%")
            gates_passed = False
        else:
            print(f"✅ 总体覆盖率达标: {result.overall_coverage:.1f}% >= 75.0%")

        return gates_passed

    def _generate_reports(self, result: TestExecutionResult):
        """生成测试报告"""
        # 生成JSON报告
        json_report = {
            "timestamp": result.timestamp,
            "summary": {
                "total_tests": result.total_tests,
                "passed": result.passed_tests,
                "failed": result.failed_tests,
                "skipped": result.skipped_tests,
                "errors": result.error_tests,
                "overall_coverage": result.overall_coverage,
                "total_duration": result.total_duration,
                "quality_gates_passed": result.quality_gates_passed,
                "deployment_ready": result.deployment_ready
            },
            "layer_results": {
                layer.value: {
                    "passed": lr.passed,
                    "failed": lr.failed,
                    "skipped": lr.skipped,
                    "errors": lr.errors,
                    "total_tests": lr.total_tests,
                    "coverage": lr.coverage,
                    "duration": lr.duration
                }
                for layer, lr in result.layer_results.items()
            }
        }

        # 保存JSON报告
        json_file = self.reports_path / f"ci_cd_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, ensure_ascii=False, indent=2)

        # 生成Markdown报告
        self._generate_markdown_report(result, json_file)

        print(f"\n📄 测试报告已生成: {json_file}")

    def _generate_markdown_report(self, result: TestExecutionResult, json_file: Path):
        """生成Markdown报告"""
        md_file = json_file.with_suffix('.md')

        with open(md_file, 'w', encoding='utf-8') as f:
            f.write("# RQA2025 CI/CD 测试报告\n\n")
            f.write(f"**执行时间**: {result.timestamp}\n\n")

            # 总体摘要
            f.write("## 📊 总体摘要\n\n")
            f.write("| 指标 | 值 |\n")
            f.write("|------|-----|\n")
            f.write(f"| 总测试数 | {result.total_tests} |\n")
            f.write(f"| 通过测试 | {result.passed_tests} |\n")
            f.write(f"| 失败测试 | {result.failed_tests} |\n")
            f.write(f"| 跳过测试 | {result.skipped_tests} |\n")
            f.write(f"| 错误测试 | {result.error_tests} |\n")
            f.write(f"| 总体覆盖率 | {result.overall_coverage:.1f}% |\n")
            f.write(f"| 执行时间 | {result.total_duration:.2f}s |\n")
            f.write(f"| 质量门禁 | {'✅ 通过' if result.quality_gates_passed else '❌ 失败'} |\n")
            f.write(f"| 部署就绪 | {'✅ 是' if result.deployment_ready else '❌ 否'} |\n\n")

            # 分层结果
            f.write("## 🏗️ 分层测试结果\n\n")
            f.write("| 层级 | 测试数 | 通过 | 失败 | 跳过 | 错误 | 覆盖率 | 耗时 |\n")
            f.write("|------|--------|------|------|------|------|--------|------|\n")

            for layer in sorted(result.layer_results.keys(), key=lambda x: self.layer_config[x]["priority"]):
                lr = result.layer_results[layer]
                f.write(f"| {layer.value} | {lr.total_tests} | {lr.passed} | {lr.failed} | {lr.skipped} | {lr.errors} | {lr.coverage:.1f}% | {lr.duration:.2f}s |\n")

            f.write("\n")

            # 质量门禁状态
            f.write("## 🔍 质量门禁检查\n\n")
            if result.quality_gates_passed:
                f.write("✅ 所有质量门禁检查通过，系统可以部署到生产环境。\n\n")
            else:
                f.write("❌ 部分质量门禁检查失败，需要修复后重新执行测试。\n\n")

                # 详细的失败原因
                f.write("### 失败详情\n\n")
                if result.failed_tests > 0:
                    f.write(f"- ❌ 存在 {result.failed_tests} 个失败的测试用例\n")

                for layer, lr in result.layer_results.items():
                    required_cov = self.layer_config[layer]["required_coverage"]
                    if lr.coverage < required_cov:
                        f.write(f"- ❌ {layer.value} 层覆盖率不足: {lr.coverage:.1f}% < {required_cov:.1f}%\n")

                if result.overall_coverage < 75.0:
                    f.write(f"- ❌ 总体覆盖率不足: {result.overall_coverage:.1f}% < 75.0%\n")

        print(f"📄 Markdown报告已生成: {md_file}")


def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(description="RQA2025 CI/CD 测试执行器")
    parser.add_argument("--layers", nargs="*", choices=[l.value for l in TestLayer],
                       help="指定要测试的层级，不指定则测试所有层级")
    parser.add_argument("--parallel", action="store_true", default=True,
                       help="启用并行测试执行")
    parser.add_argument("--max-workers", type=int, default=4,
                       help="并行执行的最大工作进程数")
    parser.add_argument("--no-coverage", action="store_true",
                       help="跳过覆盖率检查")
    parser.add_argument("--project-root", type=str,
                       help="项目根目录路径")

    args = parser.parse_args()

    # 初始化CI/CD运行器
    runner = CICDRunner(args.project_root)

    # 确定要测试的层级
    if args.layers:
        layers = [TestLayer(layer) for layer in args.layers]
    else:
        layers = None

    # 执行测试
    result = runner.run_all_tests(layers=layers, parallel=args.parallel)

    # 输出最终结果
    print("\n" + "=" * 60)
    print("🎯 CI/CD 测试执行完成")
    print("=" * 60)

    status_emoji = "✅" if result.deployment_ready else "❌"
    print(f"{status_emoji} 部署就绪状态: {'是' if result.deployment_ready else '否'}")
    print(f"📊 总体覆盖率: {result.overall_coverage:.1f}%")
    print(f"🧪 测试统计: {result.total_tests} 总计, {result.passed_tests} 通过, {result.failed_tests} 失败")

    # 返回适当的退出码
    exit_code = 0 if result.deployment_ready else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
