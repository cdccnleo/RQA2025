#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试执行加速器

优化pytest并行执行策略，提高测试运行效率：
- 智能并行分组：根据测试类型和依赖关系分组
- 资源利用优化：合理分配CPU和内存资源
- 缓存策略：利用pytest缓存减少重复工作
- 执行顺序优化：优先执行快速测试，失败测试优先重试
"""

import os
import time
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class TestExecutionPlan:
    """测试执行计划"""
    groups: List[List[str]]  # 分组的测试文件列表
    priorities: Dict[str, int]  # 测试文件优先级
    estimated_times: Dict[str, float]  # 预估执行时间
    dependencies: Dict[str, List[str]]  # 依赖关系
    resource_requirements: Dict[str, Dict[str, Any]]  # 资源需求


@dataclass
class ExecutionResult:
    """执行结果"""
    test_file: str
    duration: float
    passed: int
    failed: int
    errors: int
    skipped: int
    success: bool
    output: str


class TestAccelerator:
    """测试执行加速器"""

    def __init__(self, test_dir: str = "tests", workers: Optional[int] = None):
        self.test_dir = Path(test_dir)
        self.workers = workers or min(os.cpu_count() or 4, 8)  # 默认不超过8个worker
        self.execution_history: Dict[str, List[float]] = {}  # 执行历史
        self.cache_dir = Path(".pytest_cache")

        # 系统资源信息
        self.cpu_count = os.cpu_count() or 4
        self.memory_gb = psutil.virtual_memory().total / (1024**3)

        logger.info(f"测试加速器初始化: {self.workers}个工作进程, {self.cpu_count}个CPU核心, {self.memory_gb:.1f}GB内存")

    def optimize_execution(self, test_files: Optional[List[str]] = None) -> TestExecutionPlan:
        """优化测试执行"""
        logger.info("开始优化测试执行计划...")

        # 获取所有测试文件
        if test_files is None:
            test_files = self._discover_test_files()

        # 分析测试文件特征
        file_analysis = self._analyze_test_files(test_files)

        # 创建执行计划
        plan = self._create_execution_plan(file_analysis)

        logger.info(f"执行计划创建完成: {len(plan.groups)}个组, {len(test_files)}个测试文件")
        return plan

    def _discover_test_files(self) -> List[str]:
        """发现测试文件"""
        test_files = []

        # 查找所有测试文件
        for pattern in ["test_*.py", "*_test.py"]:
            test_files.extend([str(f) for f in self.test_dir.rglob(pattern)])

        # 按大小排序，优先执行小的文件
        test_files.sort(key=lambda x: os.path.getsize(x))

        logger.info(f"发现{len(test_files)}个测试文件")
        return test_files

    def _analyze_test_files(self, test_files: List[str]) -> Dict[str, Dict[str, Any]]:
        """分析测试文件特征"""
        analysis = {}

        for test_file in test_files:
            file_path = Path(test_file)

            # 基本文件信息
            file_size = file_path.stat().st_size
            line_count = self._count_lines(file_path)

            # 分析测试类型和复杂度
            test_types = self._analyze_test_types(file_path)
            complexity = self._estimate_complexity(file_path)

            # 预估执行时间（基于历史数据或启发式）
            estimated_time = self._estimate_execution_time(test_file, file_size, line_count, complexity)

            analysis[test_file] = {
                'size': file_size,
                'lines': line_count,
                'types': test_types,
                'complexity': complexity,
                'estimated_time': estimated_time,
                'dependencies': self._analyze_dependencies(file_path),
                'resource_usage': self._estimate_resource_usage(test_types, complexity)
            }

        return analysis

    def _count_lines(self, file_path: Path) -> int:
        """统计文件行数"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return len(f.readlines())
        except:
            return 0

    def _analyze_test_types(self, file_path: Path) -> Set[str]:
        """分析测试类型"""
        types = set()

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().lower()

            # 检测测试标记
            if 'pytest.mark.unit' in content or 'def test_' in content:
                types.add('unit')
            if 'pytest.mark.integration' in content:
                types.add('integration')
            if 'pytest.mark.e2e' in content or 'end_to_end' in content:
                types.add('e2e')
            if 'pytest.mark.slow' in content or 'time.sleep' in content:
                types.add('slow')
            if 'pytest.mark.asyncio' in content or 'async de' in content:
                types.add('async')
            if 'mock' in content or 'patch' in content:
                types.add('mock')

        except Exception as e:
            logger.warning(f"分析测试类型失败 {file_path}: {e}")

        return types

    def _estimate_complexity(self, file_path: Path) -> int:
        """估算测试复杂度"""
        complexity = 1

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 基于各种模式估算复杂度
            complexity += content.count('def test_') * 2  # 每个测试函数
            complexity += content.count('pytest.mark.parametrize') * 3  # 参数化测试
            complexity += content.count('mock')  # Mock使用
            complexity += content.count('patch')  # Patch使用
            complexity += content.count('fixture')  # Fixture使用
            complexity += content.count('asyncio')  # 异步测试

            # 文件大小因子
            size_kb = len(content) / 1024
            if size_kb > 50:
                complexity = int(complexity * 1.5)
            elif size_kb > 100:
                complexity = int(complexity * 2.0)

        except Exception as e:
            logger.warning(f"估算复杂度失败 {file_path}: {e}")

        return max(1, complexity)

    def _estimate_execution_time(self, test_file: str, file_size: int, line_count: int, complexity: int) -> float:
        """预估执行时间"""
        # 基于历史数据或启发式估算
        base_time = 1.0  # 基础时间1秒

        # 文件大小因子
        size_factor = min(file_size / (10 * 1024), 5.0)  # 10KB为基准

        # 复杂度因子
        complexity_factor = min(complexity / 10.0, 3.0)  # 10为基准

        # 历史平均时间（如果有的话）
        history_factor = 1.0
        if test_file in self.execution_history:
            history_times = self.execution_history[test_file]
            if history_times:
                history_factor = sum(history_times) / len(history_times)

        estimated_time = base_time * size_factor * complexity_factor * history_factor

        return max(0.1, min(estimated_time, 300.0))  # 限制在0.1秒到5分钟之间

    def _analyze_dependencies(self, file_path: Path) -> List[str]:
        """分析测试依赖"""
        dependencies = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 查找导入的测试工具
            if 'conftest' in content:
                dependencies.append('conftest')

            # 查找共享fixture
            if 'fixture' in content and ('session' in content or 'module' in content):
                dependencies.append('shared_fixtures')

        except Exception as e:
            logger.warning(f"分析依赖失败 {file_path}: {e}")

        return dependencies

    def _estimate_resource_usage(self, test_types: Set[str], complexity: int) -> Dict[str, Any]:
        """估算资源使用"""
        # 基础资源需求
        cpu = 1
        memory_mb = 100

        # 根据测试类型调整
        if 'async' in test_types:
            cpu = min(self.cpu_count, cpu + 1)
            memory_mb += 50

        if 'integration' in test_types or 'e2e' in test_types:
            cpu = min(self.cpu_count, cpu + 2)
            memory_mb += 200

        if 'slow' in test_types:
            cpu = min(self.cpu_count, cpu + 1)
            memory_mb += 100

        # 根据复杂度调整
        if complexity > 20:
            cpu = min(self.cpu_count, cpu + 1)
            memory_mb += 100

        return {
            'cpu_cores': cpu,
            'memory_mb': memory_mb,
            'io_intensive': 'e2e' in test_types or 'integration' in test_types,
            'network_intensive': 'integration' in test_types or 'e2e' in test_types
        }

    def _create_execution_plan(self, file_analysis: Dict[str, Dict[str, Any]]) -> TestExecutionPlan:
        """创建执行计划"""
        # 按执行时间排序（快的先执行）
        sorted_files = sorted(
            file_analysis.items(),
            key=lambda x: x[1]['estimated_time']
        )

        # 分组策略
        groups = []
        current_group = []
        current_resources = {'cpu': 0, 'memory': 0}
        max_cpu_per_group = self.cpu_count
        max_memory_per_group = self.memory_gb * 1024  # GB to MB

        for test_file, analysis in sorted_files:
            resource_usage = analysis['resource_usage']

            # 检查是否可以加入当前组
            if (current_resources['cpu'] + resource_usage['cpu_cores'] <= max_cpu_per_group and
                current_resources['memory'] + resource_usage['memory_mb'] <= max_memory_per_group):

                current_group.append(test_file)
                current_resources['cpu'] += resource_usage['cpu_cores']
                current_resources['memory'] += resource_usage['memory_mb']
            else:
                # 开始新组
                if current_group:
                    groups.append(current_group)
                current_group = [test_file]
                current_resources = {
                    'cpu': resource_usage['cpu_cores'],
                    'memory': resource_usage['memory_mb']
                }

        # 添加最后一个组
        if current_group:
            groups.append(current_group)

        # 提取其他信息
        priorities = {file: i for i, file in enumerate([f for f, _ in sorted_files])}
        estimated_times = {file: analysis['estimated_time'] for file, analysis in file_analysis.items()}
        dependencies = {file: analysis['dependencies'] for file, analysis in file_analysis.items()}
        resource_requirements = {file: analysis['resource_usage'] for file, analysis in file_analysis.items()}

        return TestExecutionPlan(
            groups=groups,
            priorities=priorities,
            estimated_times=estimated_times,
            dependencies=dependencies,
            resource_requirements=resource_requirements
        )

    def execute_optimized(self, plan: TestExecutionPlan, parallel: bool = True) -> List[ExecutionResult]:
        """执行优化后的测试"""
        logger.info(f"开始执行优化测试: {len(plan.groups)}个组, 并行={parallel}")

        results = []

        if parallel and len(plan.groups) > 1:
            results = self._execute_parallel(plan)
        else:
            results = self._execute_sequential(plan)

        # 更新执行历史
        self._update_execution_history(results)

        # 生成性能报告
        self._generate_performance_report(results, plan)

        return results

    def _execute_parallel(self, plan: TestExecutionPlan) -> List[ExecutionResult]:
        """并行执行"""
        results = []

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            # 提交所有组的任务
            future_to_group = {}
            for group in plan.groups:
                future = executor.submit(self._execute_group, group)
                future_to_group[future] = group

            # 收集结果
            for future in as_completed(future_to_group):
                group = future_to_group[future]
                try:
                    group_results = future.result()
                    results.extend(group_results)
                except Exception as e:
                    logger.error(f"执行组失败 {group}: {e}")
                    # 为失败的组创建错误结果
                    for test_file in group:
                        results.append(ExecutionResult(
                            test_file=test_file,
                            duration=0.0,
                            passed=0,
                            failed=0,
                            errors=1,
                            skipped=0,
                            success=False,
                            output=f"Execution failed: {e}"
                        ))

        return results

    def _execute_sequential(self, plan: TestExecutionPlan) -> List[ExecutionResult]:
        """顺序执行"""
        results = []

        for group in plan.groups:
            group_results = self._execute_group(group)
            results.extend(group_results)

        return results

    def _execute_group(self, group: List[str]) -> List[ExecutionResult]:
        """执行一组测试"""
        results = []

        for test_file in group:
            result = self._execute_single_test(test_file)
            results.append(result)

        return results

    def _execute_single_test(self, test_file: str) -> ExecutionResult:
        """执行单个测试文件"""
        start_time = time.time()

        try:
            # 构建pytest命令
            cmd = [
                "python", "-m", "pytest",
                test_file,
                "--tb=short",
                "--quiet",
                "--disable-warnings",
                "-x"  # 遇到第一个失败就停止
            ]

            # 执行测试
            result = subprocess.run(
                cmd,
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                timeout=300,  # 5分钟超时
                encoding='utf-8',
                errors='replace'
            )

            duration = time.time() - start_time

            # 解析结果
            passed, failed, errors, skipped = self._parse_pytest_output(result.stdout)

            return ExecutionResult(
                test_file=test_file,
                duration=duration,
                passed=passed,
                failed=failed,
                errors=errors,
                skipped=skipped,
                success=result.returncode == 0,
                output=result.stdout + result.stderr
            )

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return ExecutionResult(
                test_file=test_file,
                duration=duration,
                passed=0,
                failed=0,
                errors=1,
                skipped=0,
                success=False,
                output="Test execution timed out"
            )
        except Exception as e:
            duration = time.time() - start_time
            return ExecutionResult(
                test_file=test_file,
                duration=duration,
                passed=0,
                failed=0,
                errors=1,
                skipped=0,
                success=False,
                output=f"Execution error: {e}"
            )

    def _parse_pytest_output(self, output: str) -> Tuple[int, int, int, int]:
        """解析pytest输出"""
        passed = failed = errors = skipped = 0

        # 查找总结行
        lines = output.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if 'passed' in line.lower() and 'failed' in line.lower():
                # 解析类似 "5 passed, 2 failed, 1 error, 3 skipped"
                import re
                match = re.search(r'(\d+)\s*passed.*?(\d+)\s*failed.*?(\d+)\s*error.*?(\d+)\s*skipped', line, re.IGNORECASE)
                if match:
                    passed = int(match.group(1))
                    failed = int(match.group(2))
                    errors = int(match.group(3))
                    skipped = int(match.group(4))
                    break

        return passed, failed, errors, skipped

    def _update_execution_history(self, results: List[ExecutionResult]):
        """更新执行历史"""
        for result in results:
            if result.test_file not in self.execution_history:
                self.execution_history[result.test_file] = []
            self.execution_history[result.test_file].append(result.duration)

            # 只保留最近10次执行的时间
            if len(self.execution_history[result.test_file]) > 10:
                self.execution_history[result.test_file] = self.execution_history[result.test_file][-10:]

    def _generate_performance_report(self, results: List[ExecutionResult], plan: TestExecutionPlan):
        """生成性能报告"""
        report_path = Path("test_logs/test_performance_report.md")

        total_duration = sum(r.duration for r in results)
        total_tests = sum(r.passed + r.failed + r.errors + r.skipped for r in results)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 测试性能报告\n\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## 执行统计\n\n")
            f.write(f"- **总执行时间**: {total_duration:.2f}秒\n")
            f.write(f"- **测试文件数**: {len(results)}\n")
            f.write(f"- **总测试数**: {total_tests}\n")
            f.write(f"- **并行度**: {self.workers}\n")
            f.write(".1")
            f.write(f"- **平均文件执行时间**: {total_duration/len(results):.2f}秒\n\n")

            f.write("## 分组统计\n\n")
            f.write(f"- **执行组数**: {len(plan.groups)}\n")
            f.write(f"- **平均组大小**: {len(results)/len(plan.groups):.1f}个文件\n\n")

            f.write("## 最慢的测试文件\n\n")
            slow_tests = sorted(results, key=lambda x: x.duration, reverse=True)[:10]
            f.write("| 测试文件 | 执行时间 | 状态 |\n")
            f.write("|----------|----------|--------|\n")
            for result in slow_tests:
                status = "✅" if result.success else "❌"
                f.write(f"| `{result.test_file}` | {result.duration:.2f}秒 | {status} |\n")

            f.write("\n## 优化建议\n\n")

            # 分析优化机会
            avg_time = total_duration / len(results)
            slow_files = [r for r in results if r.duration > avg_time * 2]

            if slow_files:
                f.write(f"- **慢测试文件**: {len(slow_files)}个文件执行时间超过平均水平2倍\n")
                f.write("  - 考虑将这些测试分离到单独的执行组\n")
                f.write("  - 检查是否存在不必要的等待或资源竞争\n")

            if total_duration > 300:  # 超过5分钟
                f.write("- **总执行时间较长**: 考虑增加并行度或优化测试实现\n")

            if len(plan.groups) > self.workers * 2:
                f.write(f"- **分组较多**: 当前{len(plan.groups)}个组，建议增加worker数量到{len(plan.groups)//2}\n")

        logger.info(f"性能报告已生成: {report_path}")

    def run_optimized_test_suite(self, test_pattern: str = "*") -> Dict[str, Any]:
        """运行优化后的完整测试套件"""
        logger.info("开始运行优化测试套件...")

        # 创建执行计划
        plan = self.optimize_execution()

        # 执行测试
        results = self.execute_optimized(plan)

        # 汇总结果
        summary = {
            'total_files': len(results),
            'total_duration': sum(r.duration for r in results),
            'successful_files': sum(1 for r in results if r.success),
            'failed_files': sum(1 for r in results if not r.success),
            'total_passed': sum(r.passed for r in results),
            'total_failed': sum(r.failed for r in results),
            'total_errors': sum(r.errors for r in results),
            'total_skipped': sum(r.skipped for r in results),
            'groups_used': len(plan.groups),
            'workers_used': self.workers
        }

        logger.info("优化测试套件执行完成")
        logger.info(".1")
        logger.info(f"成功文件: {summary['successful_files']}/{summary['total_files']}")

        return summary


def main():
    """主函数"""
    accelerator = TestAccelerator()

    print("🎯 测试执行加速器启动")
    print(f"📊 系统配置: {accelerator.cpu_count} CPU核心, {accelerator.memory_gb:.1f}GB内存")
    print(f"⚡ 并行度: {accelerator.workers} 个工作进程")

    # 运行优化测试套件
    summary = accelerator.run_optimized_test_suite()

    print("\n📈 执行结果汇总:")
    print(f"  📁 测试文件: {summary['total_files']}")
    print(".1")
    print(f"  ✅ 成功文件: {summary['successful_files']}")
    print(f"  ❌ 失败文件: {summary['failed_files']}")
    print(f"  🧪 通过测试: {summary['total_passed']}")
    print(f"  💥 失败测试: {summary['total_failed']}")
    print(f"  🚨 错误数: {summary['total_errors']}")
    print(f"  ⏭️ 跳过数: {summary['total_skipped']}")
    print(f"  📦 执行组数: {summary['groups_used']}")
    print(f"  ⚙️ 工作进程: {summary['workers_used']}")

    print("\n💡 性能报告已保存到: test_logs/test_performance_report.md")
    print("\n✅ 测试执行加速器运行完成")


if __name__ == "__main__":
    main()
