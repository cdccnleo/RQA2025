#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试覆盖率和质量度量监控系统
提供测试覆盖率监控、质量指标计算和趋势分析功能
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import subprocess
import re
import logging

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class CoverageMetrics:
    """覆盖率指标"""
    timestamp: datetime
    total_lines: int
    covered_lines: int
    coverage_percent: float
    missing_lines: int
    files_covered: int
    total_files: int
    branch_coverage: Optional[float] = None
    function_coverage: Optional[float] = None


@dataclass
class QualityMetrics:
    """质量指标"""
    timestamp: datetime
    test_count: int
    test_passed: int
    test_failed: int
    test_error: int
    test_skipped: int
    execution_time: float
    success_rate: float
    average_test_time: float
    flaky_tests: List[str] = field(default_factory=list)
    slow_tests: List[str] = field(default_factory=list)


@dataclass
class LayerCoverage:
    """各层覆盖率"""
    layer_name: str
    coverage_percent: float
    test_count: int
    files_covered: int
    total_files: int
    critical_paths_covered: bool
    timestamp: datetime


class CoverageQualityMonitor:
    """测试覆盖率和质量度量监控系统"""

    def __init__(self):
        self.logger = self._setup_logger()
        self.coverage_history = []
        self.quality_history = []
        self.layer_coverage_history = []
        self.baseline_coverage = {}
        self.quality_thresholds = self._load_quality_thresholds()

    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger("CoverageQualityMonitor")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _load_quality_thresholds(self) -> Dict[str, float]:
        """加载质量阈值"""
        return {
            'minimum_coverage': 75.0,  # 最低覆盖率
            'target_coverage': 85.0,   # 目标覆盖率
            'maximum_test_time': 30.0, # 单个测试最大时间（秒）
            'success_rate_threshold': 95.0,  # 成功率阈值
            'flaky_test_threshold': 5,  # 不稳定测试阈值
            'slow_test_threshold': 10.0  # 慢测试阈值（秒）
        }

    def collect_coverage_metrics(self) -> CoverageMetrics:
        """收集覆盖率指标"""
        self.logger.info("开始收集覆盖率指标")

        try:
            # 运行pytest并收集覆盖率
            cmd = [
                "python", "-m", "pytest",
                "--cov=src",
                "--cov-report=json:coverage.json",
                "--cov-report=term-missing",
                "tests/"
            ]

            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=600  # 10分钟超时
            )

            # 解析覆盖率报告
            coverage_data = self._parse_coverage_report()

            if coverage_data:
                metrics = CoverageMetrics(
                    timestamp=datetime.now(),
                    total_lines=coverage_data.get('totals', {}).get('num_statements', 0),
                    covered_lines=coverage_data.get('totals', {}).get('covered_lines', 0),
                    coverage_percent=coverage_data.get('totals', {}).get('percent_covered', 0.0),
                    missing_lines=coverage_data.get('totals', {}).get('missing_lines', 0),
                    files_covered=len(coverage_data.get('files', {})),
                    total_files=len(coverage_data.get('files', {})),
                    branch_coverage=coverage_data.get('totals', {}).get('branch_percent_covered'),
                    function_coverage=coverage_data.get('totals', {}).get('function_percent_covered')
                )

                self.coverage_history.append(metrics)

                self.logger.info(
                    f"覆盖率指标收集完成: {metrics.coverage_percent:.1f}% "
                    f"({metrics.covered_lines}/{metrics.total_lines} 行)"
                )

                return metrics
            else:
                self.logger.error("无法解析覆盖率报告")
                return None

        except subprocess.TimeoutExpired:
            self.logger.error("覆盖率测试执行超时")
            return None
        except Exception as e:
            self.logger.error(f"收集覆盖率指标时发生错误: {e}")
            return None

    def collect_quality_metrics(self) -> QualityMetrics:
        """收集质量指标"""
        self.logger.info("开始收集质量指标")

        try:
            # 运行pytest并收集结果
            cmd = [
                "python", "-m", "pytest",
                "--json-report",
                "--json-report-file=test_results.json",
                "--tb=no",  # 不显示traceback以减少输出
                "tests/"
            ]

            start_time = time.time()
            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=600
            )
            execution_time = time.time() - start_time

            # 解析测试结果
            test_data = self._parse_test_results()

            if test_data:
                total_tests = test_data.get('summary', {}).get('total', 0)
                passed_tests = test_data.get('summary', {}).get('passed', 0)
                failed_tests = test_data.get('summary', {}).get('failed', 0)
                error_tests = test_data.get('summary', {}).get('error', 0)
                skipped_tests = test_data.get('summary', {}).get('skipped', 0)

                success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0.0
                average_test_time = execution_time / total_tests if total_tests > 0 else 0.0

                # 识别不稳定和慢速测试
                flaky_tests, slow_tests = self._analyze_test_patterns(test_data)

                metrics = QualityMetrics(
                    timestamp=datetime.now(),
                    test_count=total_tests,
                    test_passed=passed_tests,
                    test_failed=failed_tests,
                    test_error=error_tests,
                    test_skipped=skipped_tests,
                    execution_time=execution_time,
                    success_rate=success_rate,
                    average_test_time=average_test_time,
                    flaky_tests=flaky_tests,
                    slow_tests=slow_tests
                )

                self.quality_history.append(metrics)

                self.logger.info(
                    f"质量指标收集完成: {success_rate:.1f}% 成功率, "
                    f"{total_tests} 个测试, 耗时 {execution_time:.2f}秒"
                )

                return metrics
            else:
                self.logger.error("无法解析测试结果")
                return None

        except subprocess.TimeoutExpired:
            self.logger.error("质量测试执行超时")
            return None
        except Exception as e:
            self.logger.error(f"收集质量指标时发生错误: {e}")
            return None

    def collect_layer_coverage(self) -> List[LayerCoverage]:
        """收集各层覆盖率"""
        self.logger.info("开始收集各层覆盖率")

        layers = {
            'infrastructure': 'src/infrastructure',
            'core': 'src/core',
            'data': 'src/data',
            'feature': 'src/feature',
            'ml': 'src/ml',
            'strategy': 'src/strategy',
            'trading': 'src/trading',
            'risk': 'src/risk',
            'monitoring': 'src/monitoring'
        }

        layer_coverages = []

        for layer_name, layer_path in layers.items():
            try:
                # 运行特定层的覆盖率测试
                cmd = [
                    "python", "-m", "pytest",
                    f"--cov={layer_path}",
                    "--cov-report=json",
                    f"--cov-report-file=layer_coverage_{layer_name}.json",
                    f"tests/unit/{layer_name}/" if (project_root / f"tests/unit/{layer_name}").exists() else "tests/",
                    "-k", layer_name
                ]

                result = subprocess.run(
                    cmd,
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    timeout=300
                )

                # 解析层覆盖率
                layer_coverage = self._parse_layer_coverage(layer_name)
                if layer_coverage:
                    layer_coverages.append(layer_coverage)

            except Exception as e:
                self.logger.warning(f"收集 {layer_name} 层覆盖率时发生错误: {e}")

        self.layer_coverage_history.extend(layer_coverages)
        return layer_coverages

    def _parse_coverage_report(self) -> Optional[Dict[str, Any]]:
        """解析覆盖率报告"""
        coverage_file = project_root / "coverage.json"
        if not coverage_file.exists():
            return None

        try:
            with open(coverage_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"解析覆盖率报告失败: {e}")
            return None

    def _parse_test_results(self) -> Optional[Dict[str, Any]]:
        """解析测试结果"""
        results_file = project_root / "test_results.json"
        if not results_file.exists():
            return None

        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"解析测试结果失败: {e}")
            return None

    def _parse_layer_coverage(self, layer_name: str) -> Optional[LayerCoverage]:
        """解析层覆盖率"""
        coverage_file = project_root / f"layer_coverage_{layer_name}.json"
        if not coverage_file.exists():
            return None

        try:
            with open(coverage_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            totals = data.get('totals', {})
            coverage_percent = totals.get('percent_covered', 0.0)
            files_covered = len(data.get('files', {}))

            # 估算总文件数（这里需要更准确的实现）
            total_files = files_covered  # 简化处理

            # 检查关键路径覆盖
            critical_paths_covered = coverage_percent > 80.0  # 简化的检查

            return LayerCoverage(
                layer_name=layer_name,
                coverage_percent=coverage_percent,
                test_count=0,  # 需要从测试结果中提取
                files_covered=files_covered,
                total_files=total_files,
                critical_paths_covered=critical_paths_covered,
                timestamp=datetime.now()
            )

        except Exception as e:
            self.logger.error(f"解析 {layer_name} 层覆盖率失败: {e}")
            return None

    def _analyze_test_patterns(self, test_data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """分析测试模式，识别不稳定和慢速测试"""
        flaky_tests = []
        slow_tests = []

        try:
            tests = test_data.get('tests', [])
            for test in tests:
                test_name = test.get('nodeid', '')
                duration = test.get('duration', 0)
                outcome = test.get('outcome', '')

                # 检查慢速测试
                if duration > self.quality_thresholds['slow_test_threshold']:
                    slow_tests.append(f"{test_name} ({duration:.2f}s)")

                # 这里可以添加更复杂的 flaky 测试检测逻辑
                # 暂时简化处理

        except Exception as e:
            self.logger.warning(f"分析测试模式时发生错误: {e}")

        return flaky_tests, slow_tests

    def check_quality_thresholds(self, coverage: CoverageMetrics,
                               quality: QualityMetrics) -> Dict[str, Any]:
        """检查质量阈值"""
        violations = []
        warnings = []

        # 检查覆盖率
        if coverage.coverage_percent < self.quality_thresholds['minimum_coverage']:
            violations.append({
                'type': 'coverage',
                'metric': 'minimum_coverage',
                'current': coverage.coverage_percent,
                'threshold': self.quality_thresholds['minimum_coverage'],
                'severity': 'critical'
            })

        if coverage.coverage_percent < self.quality_thresholds['target_coverage']:
            warnings.append({
                'type': 'coverage',
                'metric': 'target_coverage',
                'current': coverage.coverage_percent,
                'threshold': self.quality_thresholds['target_coverage'],
                'severity': 'warning'
            })

        # 检查成功率
        if quality.success_rate < self.quality_thresholds['success_rate_threshold']:
            violations.append({
                'type': 'quality',
                'metric': 'success_rate',
                'current': quality.success_rate,
                'threshold': self.quality_thresholds['success_rate_threshold'],
                'severity': 'critical'
            })

        # 检查不稳定测试
        if len(quality.flaky_tests) > self.quality_thresholds['flaky_test_threshold']:
            warnings.append({
                'type': 'quality',
                'metric': 'flaky_tests',
                'current': len(quality.flaky_tests),
                'threshold': self.quality_thresholds['flaky_test_threshold'],
                'severity': 'warning'
            })

        # 检查慢速测试
        if len(quality.slow_tests) > 0:
            warnings.append({
                'type': 'quality',
                'metric': 'slow_tests',
                'current': len(quality.slow_tests),
                'threshold': 0,
                'severity': 'info'
            })

        return {
            'overall_status': 'fail' if violations else ('warning' if warnings else 'pass'),
            'violations': violations,
            'warnings': warnings
        }

    def generate_quality_report(self, coverage: CoverageMetrics,
                              quality: QualityMetrics,
                              layer_coverages: List[LayerCoverage]) -> str:
        """生成质量报告"""
        report_lines = []
        report_lines.append("# 测试质量和覆盖率报告")
        report_lines.append(f"生成时间: {datetime.now().isoformat()}")
        report_lines.append("")

        # 覆盖率概览
        report_lines.append("## 📊 覆盖率概览")
        report_lines.append(f"- 总行数: {coverage.total_lines:,}")
        report_lines.append(f"- 覆盖行数: {coverage.covered_lines:,}")
        report_lines.append(".1")
        report_lines.append(f"- 未覆盖行数: {coverage.missing_lines:,}")
        report_lines.append(f"- 覆盖文件数: {coverage.files_covered}/{coverage.total_files}")
        if coverage.branch_coverage:
            report_lines.append(".1")
        if coverage.function_coverage:
            report_lines.append(".1")
        report_lines.append("")

        # 质量指标概览
        report_lines.append("## 🧪 质量指标概览")
        report_lines.append(f"- 总测试数: {quality.test_count}")
        report_lines.append(f"- 通过测试: {quality.test_passed}")
        report_lines.append(f"- 失败测试: {quality.test_failed}")
        report_lines.append(f"- 错误测试: {quality.test_error}")
        report_lines.append(f"- 跳过测试: {quality.test_skipped}")
        report_lines.append(".1")
        report_lines.append(".3")
        report_lines.append(f"- 平均测试时间: {quality.average_test_time:.3f}秒")

        if quality.flaky_tests:
            report_lines.append(f"- 不稳定测试: {len(quality.flaky_tests)} 个")
        if quality.slow_tests:
            report_lines.append(f"- 慢速测试: {len(quality.slow_tests)} 个")

        report_lines.append("")

        # 各层覆盖率
        report_lines.append("## 🏗️ 各层覆盖率")
        for layer in layer_coverages:
            status = "✅" if layer.coverage_percent >= 80 else "⚠️" if layer.coverage_percent >= 60 else "❌"
            report_lines.append(".1")
        report_lines.append("")

        # 质量检查结果
        quality_check = self.check_quality_thresholds(coverage, quality)
        report_lines.append("## 🔍 质量检查结果")

        status_emoji = {
            'pass': '✅',
            'warning': '⚠️',
            'fail': '❌'
        }.get(quality_check['overall_status'], '❓')

        report_lines.append(f"总体状态: {status_emoji} {quality_check['overall_status'].upper()}")

        if quality_check['violations']:
            report_lines.append("\n### 🚨 违规项目")
            for violation in quality_check['violations']:
                report_lines.append(f"- **{violation['type']}**: {violation['metric']} = {violation['current']:.2f}, 阈值: {violation['threshold']:.2f}")

        if quality_check['warnings']:
            report_lines.append("\n### ⚠️ 警告项目")
            for warning in quality_check['warnings']:
                report_lines.append(f"- **{warning['type']}**: {warning['metric']} = {warning['current']:.2f}, 阈值: {warning['threshold']:.2f}")

        return "\n".join(report_lines)

    def save_quality_report(self, coverage: CoverageMetrics,
                          quality: QualityMetrics,
                          layer_coverages: List[LayerCoverage],
                          report_file: str = "test_quality_report.md"):
        """保存质量报告"""
        report_path = project_root / "test_logs" / report_file
        report_path.parent.mkdir(exist_ok=True)

        report_content = self.generate_quality_report(coverage, quality, layer_coverages)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        self.logger.info(f"测试质量报告已保存到: {report_path}")

    def export_metrics_to_json(self, coverage: CoverageMetrics,
                             quality: QualityMetrics,
                             layer_coverages: List[LayerCoverage],
                             json_file: str = "test_metrics.json"):
        """导出指标到JSON"""
        json_path = project_root / "test_logs" / json_file
        json_path.parent.mkdir(exist_ok=True)

        metrics_data = {
            'timestamp': datetime.now().isoformat(),
            'coverage': {
                'total_lines': coverage.total_lines,
                'covered_lines': coverage.covered_lines,
                'coverage_percent': coverage.coverage_percent,
                'missing_lines': coverage.missing_lines,
                'files_covered': coverage.files_covered,
                'total_files': coverage.total_files,
                'branch_coverage': coverage.branch_coverage,
                'function_coverage': coverage.function_coverage
            },
            'quality': {
                'test_count': quality.test_count,
                'test_passed': quality.test_passed,
                'test_failed': quality.test_failed,
                'test_error': quality.test_error,
                'test_skipped': quality.test_skipped,
                'execution_time': quality.execution_time,
                'success_rate': quality.success_rate,
                'average_test_time': quality.average_test_time,
                'flaky_tests_count': len(quality.flaky_tests),
                'slow_tests_count': len(quality.slow_tests)
            },
            'layer_coverage': [
                {
                    'layer_name': lc.layer_name,
                    'coverage_percent': lc.coverage_percent,
                    'test_count': lc.test_count,
                    'files_covered': lc.files_covered,
                    'total_files': lc.total_files,
                    'critical_paths_covered': lc.critical_paths_covered
                } for lc in layer_coverages
            ]
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"测试指标已导出到: {json_path}")


# 全局监控实例
quality_monitor = CoverageQualityMonitor()


def run_quality_assessment() -> Tuple[CoverageMetrics, QualityMetrics, List[LayerCoverage]]:
    """运行质量评估的便捷函数"""
    coverage = quality_monitor.collect_coverage_metrics()
    quality = quality_monitor.collect_quality_metrics()
    layer_coverages = quality_monitor.collect_layer_coverage()

    return coverage, quality, layer_coverages


def generate_quality_report(coverage: CoverageMetrics,
                          quality: QualityMetrics,
                          layer_coverages: List[LayerCoverage]) -> str:
    """生成质量报告的便捷函数"""
    return quality_monitor.generate_quality_report(coverage, quality, layer_coverages)
