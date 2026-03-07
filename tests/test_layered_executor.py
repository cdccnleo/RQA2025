#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分层测试执行器
根据测试架构配置执行分层测试，确保依赖关系和资源管理
"""

import os
import sys
import time
import asyncio
import subprocess
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import logging

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.test_architecture_config import (
    TestArchitectureConfig,
    TestLayer,
    TestCategory,
    TestExecutionConfig,
    test_architecture_config
)


@dataclass
class TestExecutionResult:
    """测试执行结果"""
    test_name: str
    layer: TestLayer
    status: str  # "passed", "failed", "error", "skipped"
    execution_time: float
    output: str
    error_message: Optional[str] = None
    coverage_data: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None


@dataclass
class LayerExecutionResult:
    """层级执行结果"""
    layer: TestLayer
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    execution_time: float
    test_results: List[TestExecutionResult]


class LayeredTestExecutor:
    """分层测试执行器"""

    def __init__(self, config: TestArchitectureConfig = None):
        self.config = config or test_architecture_config
        self.logger = self._setup_logger()
        self.execution_results = {}

    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger("LayeredTestExecutor")
        logger.setLevel(logging.INFO)

        # 避免重复添加处理器
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def execute_layered_tests(self, target_layers: Optional[List[TestLayer]] = None) -> Dict[TestLayer, LayerExecutionResult]:
        """执行分层测试"""
        if target_layers is None:
            target_layers = self.config.get_layer_hierarchy()

        results = {}

        for layer in target_layers:
            self.logger.info(f"开始执行 {layer.value} 层测试")
            start_time = time.time()

            try:
                layer_result = self._execute_layer(layer)
                execution_time = time.time() - start_time

                # 更新执行时间
                layer_result.execution_time = execution_time

                results[layer] = layer_result

                self.logger.info(
                    f"{layer.value} 层测试完成: {layer_result.passed_tests}/{layer_result.total_tests} 通过, "
                    f"耗时 {execution_time:.2f}秒"
                )

                # 如果是关键层级失败过多，停止执行后续层级
                if self._should_stop_execution(layer_result):
                    self.logger.warning(f"{layer.value} 层测试失败过多，停止执行后续层级")
                    break

            except Exception as e:
                self.logger.error(f"{layer.value} 层测试执行失败: {e}")
                # 创建失败结果
                results[layer] = LayerExecutionResult(
                    layer=layer,
                    total_tests=0,
                    passed_tests=0,
                    failed_tests=0,
                    skipped_tests=0,
                    execution_time=time.time() - start_time,
                    test_results=[]
                )

        return results

    def _execute_layer(self, layer: TestLayer) -> LayerExecutionResult:
        """执行单个层级的测试"""
        strategy = self.config.get_execution_strategy(layer)

        # 获取该层级的测试文件
        test_files = self._discover_test_files(layer)

        if not test_files:
            self.logger.warning(f"{layer.value} 层没有发现测试文件")
            return LayerExecutionResult(
                layer=layer,
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                skipped_tests=0,
                execution_time=0.0,
                test_results=[]
            )

        # 设置依赖
        if strategy.get("require_dependencies", False):
            self._setup_dependencies(layer)

        # 执行测试
        if strategy.get("parallel_execution", True) and len(test_files) > 1:
            return self._execute_parallel(layer, test_files, strategy)
        else:
            return self._execute_sequential(layer, test_files, strategy)

    def _discover_test_files(self, layer: TestLayer) -> List[Path]:
        """发现测试文件"""
        test_root = Path(__file__).parent

        # 根据层级确定测试目录
        layer_dirs = {
            TestLayer.UNIT: ["unit"],
            TestLayer.INTEGRATION: ["integration"],
            TestLayer.E2E: ["e2e"],
            TestLayer.PERFORMANCE: ["performance"],
            TestLayer.SECURITY: ["security"]
        }

        test_files = []
        for dir_name in layer_dirs.get(layer, []):
            layer_dir = test_root / dir_name
            if layer_dir.exists():
                # 查找所有test_*.py文件
                for pattern in ["test_*.py", "*_test.py"]:
                    test_files.extend(list(layer_dir.glob(f"**/{pattern}")))

        return sorted(list(set(test_files)))  # 去重并排序

    def _setup_dependencies(self, layer: TestLayer):
        """设置测试依赖"""
        config = self.config.get_test_config(f"{layer.value}.*")
        for dependency_name in config.dependencies:
            dependency_config = self.config.get_dependency_config(dependency_name)
            if dependency_config:
                self._execute_setup_commands(dependency_config.setup_commands)

    def _execute_parallel(self, layer: TestLayer, test_files: List[Path],
                         strategy: Dict[str, Any]) -> LayerExecutionResult:
        """并行执行测试"""
        max_workers = min(len(test_files), strategy.get("max_workers", 4))
        test_results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有测试任务
            future_to_test = {}
            for test_file in test_files:
                future = executor.submit(self._execute_single_test, test_file, layer, strategy)
                future_to_test[future] = test_file

            # 收集结果
            for future in as_completed(future_to_test):
                test_file = future_to_test[future]
                try:
                    result = future.result()
                    test_results.append(result)
                except Exception as e:
                    self.logger.error(f"执行测试 {test_file} 时发生异常: {e}")
                    # 创建错误结果
                    test_results.append(TestExecutionResult(
                        test_name=str(test_file),
                        layer=layer,
                        status="error",
                        execution_time=0.0,
                        output="",
                        error_message=str(e)
                    ))

        return self._aggregate_results(layer, test_results)

    def _execute_sequential(self, layer: TestLayer, test_files: List[Path],
                           strategy: Dict[str, Any]) -> LayerExecutionResult:
        """顺序执行测试"""
        test_results = []

        for test_file in test_files:
            result = self._execute_single_test(test_file, layer, strategy)
            test_results.append(result)

            # 如果启用了快速失败且测试失败，停止执行
            if strategy.get("fail_fast", False) and result.status in ["failed", "error"]:
                self.logger.warning(f"测试 {test_file} 失败，启用快速失败，停止执行")
                break

        return self._aggregate_results(layer, test_results)

    def _execute_single_test(self, test_file: Path, layer: TestLayer,
                           strategy: Dict[str, Any]) -> TestExecutionResult:
        """执行单个测试文件"""
        test_name = str(test_file.relative_to(Path(__file__).parent))

        self.logger.info(f"执行测试: {test_name}")

        # 获取测试配置
        config = self.config.get_test_config(test_name)

        # 构建pytest命令
        cmd = ["python", "-m", "pytest", str(test_file)]

        # 添加超时
        cmd.extend(["--timeout", str(config.timeout)])

        # 添加其他选项
        if strategy.get("report_coverage", False):
            cmd.extend(["--cov=src", "--cov-report=term-missing"])

        if layer == TestLayer.PERFORMANCE:
            cmd.extend(["--benchmark-only", "--benchmark-json=benchmark.json"])

        # 设置环境变量
        env = os.environ.copy()
        env.update(config.environment_variables)

        start_time = time.time()

        try:
            # 执行测试
            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=config.timeout + 10,  # 额外10秒缓冲
                env=env
            )

            execution_time = time.time() - start_time

            # 解析结果
            if result.returncode == 0:
                status = "passed"
            elif result.returncode == 1:
                status = "failed"
            else:
                status = "error"

            # 提取覆盖率数据（如果有的话）
            coverage_data = None
            if "coverage" in result.stdout:
                coverage_data = self._parse_coverage_output(result.stdout)

            # 提取性能指标（如果有的话）
            performance_metrics = None
            if layer == TestLayer.PERFORMANCE:
                performance_metrics = self._parse_performance_output(result.stdout)

            return TestExecutionResult(
                test_name=test_name,
                layer=layer,
                status=status,
                execution_time=execution_time,
                output=result.stdout + result.stderr,
                error_message=result.stderr if result.returncode != 0 else None,
                coverage_data=coverage_data,
                performance_metrics=performance_metrics
            )

        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return TestExecutionResult(
                test_name=test_name,
                layer=layer,
                status="error",
                execution_time=execution_time,
                output="",
                error_message=f"测试超时 ({config.timeout}秒)"
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return TestExecutionResult(
                test_name=test_name,
                layer=layer,
                status="error",
                execution_time=execution_time,
                output="",
                error_message=str(e)
            )

    def _aggregate_results(self, layer: TestLayer,
                          test_results: List[TestExecutionResult]) -> LayerExecutionResult:
        """聚合测试结果"""
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results if r.status == "passed")
        failed_tests = sum(1 for r in test_results if r.status == "failed")
        skipped_tests = sum(1 for r in test_results if r.status == "skipped")
        execution_time = sum(r.execution_time for r in test_results)

        return LayerExecutionResult(
            layer=layer,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            execution_time=execution_time,
            test_results=test_results
        )

    def _should_stop_execution(self, layer_result: LayerExecutionResult) -> bool:
        """判断是否应该停止执行"""
        if layer_result.total_tests == 0:
            return False

        failure_rate = layer_result.failed_tests / layer_result.total_tests

        # 如果失败率超过20%，停止执行
        return failure_rate > 0.2

    def _execute_setup_commands(self, commands: List[str]):
        """执行设置命令"""
        for cmd in commands:
            try:
                self.logger.info(f"执行设置命令: {cmd}")
                result = subprocess.run(cmd, shell=True, cwd=project_root,
                                      capture_output=True, text=True)
                if result.returncode != 0:
                    self.logger.warning(f"设置命令失败: {cmd}, 错误: {result.stderr}")
            except Exception as e:
                self.logger.error(f"执行设置命令异常: {cmd}, 错误: {e}")

    def _parse_coverage_output(self, output: str) -> Optional[Dict[str, Any]]:
        """解析覆盖率输出"""
        # 这里可以实现覆盖率数据的解析
        # 暂时返回None，具体实现可以根据pytest-cov的输出格式来解析
        return None

    def _parse_performance_output(self, output: str) -> Optional[Dict[str, Any]]:
        """解析性能输出"""
        # 这里可以实现性能数据的解析
        # 暂时返回None，具体实现可以根据benchmark插件的输出格式来解析
        return None

    def generate_execution_report(self, results: Dict[TestLayer, LayerExecutionResult]) -> str:
        """生成执行报告"""
        report_lines = []
        report_lines.append("# 测试执行报告")
        report_lines.append(f"生成时间: {datetime.now().isoformat()}")
        report_lines.append("")

        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_execution_time = 0.0

        for layer, result in results.items():
            report_lines.append(f"## {layer.value.upper()} 层")
            report_lines.append(f"- 总测试数: {result.total_tests}")
            report_lines.append(f"- 通过测试: {result.passed_tests}")
            report_lines.append(f"- 失败测试: {result.failed_tests}")
            report_lines.append(f"- 跳过测试: {result.skipped_tests}")
            report_lines.append(f"- 执行时间: {result.execution_time:.2f}s")
            report_lines.append("")

            total_tests += result.total_tests
            total_passed += result.passed_tests
            total_failed += result.failed_tests
            total_execution_time += result.execution_time

        # 总体统计
        report_lines.append("## 总体统计")
        report_lines.append(f"- 总测试数: {total_tests}")
        report_lines.append(f"- 总通过数: {total_passed}")
        report_lines.append(f"- 总失败数: {total_failed}")
        report_lines.append(f"- 总执行时间: {total_execution_time:.1f}s")
        report_lines.append(f"- 平均执行时间: {total_execution_time/max(total_tests, 1):.2f}s")
        if total_tests > 0:
            success_rate = (total_passed / total_tests) * 100
            report_lines.append(f"- 成功率: {success_rate:.1f}%")
        return "\n".join(report_lines)

    def save_execution_report(self, results: Dict[TestLayer, LayerExecutionResult],
                            report_file: str = "test_execution_report.md"):
        """保存执行报告"""
        report_path = project_root / "test_logs" / report_file
        report_path.parent.mkdir(exist_ok=True)

        report_content = self.generate_execution_report(results)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        self.logger.info(f"测试执行报告已保存到: {report_path}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="分层测试执行器")
    parser.add_argument("--layers", nargs="*", choices=["unit", "integration", "e2e", "performance", "security"],
                       help="要执行的测试层级")
    parser.add_argument("--report", type=str, default="test_execution_report.md",
                       help="报告文件路径")

    args = parser.parse_args()

    # 转换层级参数
    target_layers = None
    if args.layers:
        layer_map = {
            "unit": TestLayer.UNIT,
            "integration": TestLayer.INTEGRATION,
            "e2e": TestLayer.E2E,
            "performance": TestLayer.PERFORMANCE,
            "security": TestLayer.SECURITY
        }
        target_layers = [layer_map[layer] for layer in args.layers if layer in layer_map]

    # 执行测试
    executor = LayeredTestExecutor()
    results = executor.execute_layered_tests(target_layers)

    # 生成报告
    executor.save_execution_report(results, args.report)

    # 输出简要结果
    total_passed = sum(result.passed_tests for result in results.values())
    total_failed = sum(result.failed_tests for result in results.values())

    print("\n测试执行完成:")
    print(f"通过: {total_passed}, 失败: {total_failed}")

    # 如果有失败的测试，返回非零退出码
    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
