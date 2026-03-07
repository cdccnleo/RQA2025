#!/usr/bin/env python3
"""
测试执行器 - 统一测试执行管理

提供统一的测试执行接口，支持分层测试执行、结果统计和报告生成。
"""

import sys
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
import logging

import sys
from pathlib import Path

# 添加framework目录到路径
_framework_dir = Path(__file__).parent
if str(_framework_dir) not in sys.path:
    sys.path.insert(0, str(_framework_dir))

from unified_test_framework import get_unified_framework

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """测试结果数据类"""
    layer_name: str
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    errors: int = 0
    execution_time: float = 0.0
    coverage: Optional[float] = None
    return_code: int = 0
    stdout: str = ""
    stderr: str = ""

    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100

    @property
    def is_success(self) -> bool:
        """是否成功"""
        return self.return_code == 0 and self.failed_tests == 0 and self.errors == 0


class TestExecutor:
    """
    测试执行器

    统一管理所有测试执行任务，提供：
    - 分层测试执行
    - 并行执行支持
    - 结果统计和报告
    - 覆盖率集成
    """

    def __init__(self):
        self.framework = get_unified_framework()
        self.project_root = self.framework.project_root
        self.results: Dict[str, TestResult] = {}

    def execute_layer_tests(self, layer_name: str, **kwargs) -> TestResult:
        """
        执行指定层的测试

        Args:
            layer_name: 层名称
            **kwargs: 执行参数

        Returns:
            TestResult对象
        """
        logger.info(f"开始执行 {layer_name} 层测试")

        # 设置测试环境
        self.framework.setup_layer_environment(layer_name)

        # 构建测试命令
        cmd = self._build_test_command(layer_name, **kwargs)

        # 执行测试
        start_time = time.time()
        result = self._run_command(cmd)
        execution_time = time.time() - start_time

        # 解析结果
        test_result = self._parse_test_result(layer_name, result, execution_time)

        # 保存结果
        self.results[layer_name] = test_result

        logger.info(".2")
        return test_result

    def execute_all_layers(self, layers: List[str] = None, **kwargs) -> Dict[str, TestResult]:
        """
        执行所有层的测试

        Args:
            layers: 要执行的层列表，默认执行所有层
            **kwargs: 执行参数

        Returns:
            各层测试结果字典
        """
        if layers is None:
            layers = [
                'infrastructure', 'core', 'data', 'features', 'ml',
                'optimization', 'strategy', 'trading', 'risk', 'monitoring', 'gateway'
            ]

        results = {}
        for layer in layers:
            try:
                result = self.execute_layer_tests(layer, **kwargs)
                results[layer] = result
            except Exception as e:
                logger.error(f"执行 {layer} 层测试失败: {e}")
                results[layer] = TestResult(
                    layer_name=layer,
                    errors=1,
                    return_code=1,
                    stderr=str(e)
                )

        return results

    def _build_test_command(self, layer_name: str, **kwargs) -> List[str]:
        """构建测试命令"""
        cmd = [
            sys.executable, '-m', 'pytest',
            f'tests/unit/{layer_name}/',
            '--tb=short',
            '--strict-markers'
        ]

        logger.debug(f"初始命令: {cmd}")

        # 添加详细输出
        if kwargs.get('verbose', True):
            cmd.append('-v')

        # 添加覆盖率
        if kwargs.get('coverage', False):
            cmd.extend([
                '--cov', f'src.{layer_name}',
                '--cov-report', 'term-missing',
                '--cov-report', 'html:test_logs/coverage_reports/',
                '--cov-report', 'json:test_logs/coverage_reports/coverage.json'
            ])

        # 强制单进程执行以确保完整的结果收集
        # 确保没有其他-n参数
        cmd = [c for c in cmd if not (c == '-n' or c.startswith('-n='))]
        cmd.extend(['-n', '0'])

        # 允许测试失败继续执行（移除--maxfail限制）
        cmd = [c for c in cmd if not c.startswith('--maxfail')]

        # 移除可能导致过滤的标记选项（-m "not legacy"）
        # 需要小心处理，因为-m后面跟着参数
        new_cmd = []
        skip_next = False
        for c in cmd:
            if skip_next:
                skip_next = False
                continue
            if c == '-m' and len(cmd) > cmd.index(c) + 1 and 'not legacy' in cmd[cmd.index(c) + 1]:
                skip_next = True  # 跳过-m和下一个参数
                continue
            new_cmd.append(c)
        cmd = new_cmd

        # 如果用户明确要求并行，则覆盖单进程设置
        if kwargs.get('parallel', False):
            workers = kwargs.get('workers', 'auto')
            cmd[-1] = str(workers)  # 替换-n参数

        # 添加标记过滤
        if kwargs.get('markers'):
            cmd.extend(['-m', kwargs['markers']])

        # 添加其他选项
        if kwargs.get('no_cov', True):
            cmd.append('--no-cov')

        if kwargs.get('quiet', False):
            cmd.append('-q')

        return cmd

    def _run_command(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """运行命令"""
        logger.debug(f"执行命令: {' '.join(cmd)}")

        try:
            # 设置环境变量确保UTF-8编码
            env = os.environ.copy()
            env.update({
                'PYTHONIOENCODING': 'utf-8',
                'PYTHONUTF8': '1',
                'LANG': 'C.UTF-8',
                'LC_ALL': 'C.UTF-8'
            })

            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',  # 替换无法解码的字符
                timeout=600,  # 10分钟超时
                env=env
            )
            return result
        except subprocess.TimeoutExpired:
            logger.error("测试执行超时")
            return subprocess.CompletedProcess(
                cmd, -1, "", "测试执行超时"
            )
        except UnicodeDecodeError as e:
            logger.error(f"编码错误: {e}")
            return subprocess.CompletedProcess(
                cmd, -1, "", f"编码错误: {e}"
            )

    def _parse_test_result(self, layer_name: str, process_result: subprocess.CompletedProcess,
                          execution_time: float) -> TestResult:
        """解析测试结果"""
        result = TestResult(
            layer_name=layer_name,
            execution_time=execution_time,
            return_code=process_result.returncode,
            stdout=process_result.stdout or "",
            stderr=process_result.stderr or ""
        )

        # 如果执行失败但有输出，尝试解析
        if process_result.returncode != 0:
            # 检查是否有实际的测试执行，即使返回码非0
            if not process_result.stdout and process_result.stderr:
                result.errors = 1
                logger.warning(f"pytest执行失败: {process_result.stderr}")
                return result
            # 如果有stdout，继续解析

        # 解析stdout中的测试统计信息
        stdout = process_result.stdout or ""

        try:
            # 查找测试统计行
            lines = stdout.split('\n')

            # 首先查找"short test summary info"部分
            in_summary = False
            for line in lines:
                line = line.strip()

                # 查找总结信息开始标记
                if 'short test summary info' in line.lower():
                    in_summary = True
                    continue

                if in_summary:
                    # 解析类似 "SKIPPED [1] ...: Core services not available" 的行
                    if 'SKIPPED [' in line and ']:' in line:
                        try:
                            # 提取跳过数量
                            skip_part = line.split('SKIPPED [')[1].split(']')[0]
                            result.skipped_tests += int(skip_part)
                        except (ValueError, IndexError):
                            pass
                    elif line.startswith('=====') and in_summary:
                        # 总结部分结束
                        break

            # 查找最终统计行，通常在最后
            for line in reversed(lines):
                line = line.strip()

                # 查找多种格式的统计行
                # 格式1: "= 4 failed, 762 passed, 5 skipped, 9 warnings, 4 errors in 109.82s (0:01:49) =="
                # 格式2: "============================= 3 errors in 20.60s =============================="
                # 格式3: "4 passed, 3 errors in 25.97s"

                if ('=' in line and ('failed' in line or 'passed' in line or 'skipped' in line or 'errors' in line) and 'in' in line) or \
                   (('skipped' in line or 'passed' in line or 'failed' in line or 'errors' in line) and 'in' in line and ('s' in line or 'warnings' in line)):
                    # 清理等号
                    clean_line = line.replace('=', '').strip()

                    try:
                        # 提取统计信息
                        # 处理类似 "= 4 failed, 762 passed, 5 skipped, 9 warnings, 4 errors in 109.82s (0:01:49) =="
                        if 'failed,' in clean_line or 'passed,' in clean_line or 'skipped,' in clean_line or 'errors,' in clean_line:
                            # 分割并解析逗号分隔的部分
                            time_part = clean_line.split(' in ')[0]  # 获取时间前面的部分
                            stat_parts = [p.strip() for p in time_part.split(',')]

                            for stat_part in stat_parts:
                                if 'failed' in stat_part:
                                    result.failed_tests = int(stat_part.split()[0])
                                elif 'passed' in stat_part:
                                    result.passed_tests = int(stat_part.split()[0])
                                elif 'skipped' in stat_part:
                                    result.skipped_tests = int(stat_part.split()[0])
                                elif 'errors' in stat_part:
                                    result.errors = int(stat_part.split()[0])
                        else:
                            # 处理简单格式: "3 errors in 17.53s" 或 "4 passed, 3 errors in 25.97s"
                            if ', ' in clean_line:
                                parts = clean_line.replace(' in ', ', ').split(', ')
                            else:
                                parts = clean_line.replace(' in ', ' ').split()

                            # 处理分割后的部分，数字和类型词在不同元素中
                            i = 0
                            while i < len(parts):
                                part = parts[i].strip()
                                if part.isdigit():
                                    count = int(part)
                                    if i + 1 < len(parts):
                                        test_type = parts[i + 1].strip()
                                        if 'skipped' in test_type:
                                            result.skipped_tests = count
                                        elif 'errors' in test_type:
                                            result.errors = count
                                        elif 'failed' in test_type:
                                            result.failed_tests = count
                                        elif 'passed' in test_type:
                                            result.passed_tests = count
                                    i += 1  # 跳过下一个类型词
                                i += 1
                    except Exception as e:
                        logger.warning(f"解析统计行时出错: {e}, 行内容: {line}")
                    break

            # 计算总测试数
            result.total_tests = result.passed_tests + result.failed_tests + result.skipped_tests + result.errors

            # 解析覆盖率信息
            for line in lines:
                if 'TOTAL' in line and '%' in line:
                    try:
                        # 查找百分比
                        parts = line.split()
                        for part in parts:
                            if '%' in part:
                                result.coverage = float(part.strip('%'))
                                break
                    except (ValueError, IndexError):
                        pass

        except Exception as e:
            logger.warning(f"解析测试结果时出错: {e}")
            result.errors = 1

        return result

    def generate_report(self, results: Dict[str, TestResult] = None,
                       output_file: str = "test_logs/test_execution_report.md") -> str:
        """
        生成测试报告

        Args:
            results: 测试结果字典，如果为None则使用self.results
            output_file: 输出文件路径

        Returns:
            报告内容
        """
        if results is None:
            results = self.results

        report_lines = [
            "# 测试执行报告\n",
            f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            "---\n"
        ]

        # 总体统计
        total_tests = sum(r.total_tests for r in results.values())
        total_passed = sum(r.passed_tests for r in results.values())
        total_failed = sum(r.failed_tests for r in results.values())
        total_skipped = sum(r.skipped_tests for r in results.values())
        total_errors = sum(r.errors for r in results.values())
        total_time = sum(r.execution_time for r in results.values())

        report_lines.extend([
            "## 总体统计\n",
            f"- 总测试数: {total_tests}",
            f"- 通过测试: {total_passed}",
            f"- 失败测试: {total_failed}",
            f"- 跳过测试: {total_skipped}",
            f"- 错误数: {total_errors}",
            ".2"            ".1"            "---\n"
        ])

        # 分层统计
        report_lines.extend([
            "## 分层统计\n",
            "| 层级 | 总测试 | 通过 | 失败 | 跳过 | 成功率 | 执行时间 |",
            "|------|--------|------|------|------|--------|----------|"
        ])

        for layer_name, result in results.items():
            report_lines.append(
                f"| {layer_name} | {result.total_tests} | {result.passed_tests} | "
                f"{result.failed_tests} | {result.skipped_tests} | "
                ".1"
                ".2"
            )

        report_lines.append("---\n")

        # 详细结果
        report_lines.append("## 详细结果\n")
        for layer_name, result in results.items():
            report_lines.extend([
                f"### {layer_name.upper()} 层",
                f"- **状态**: {'✅ 通过' if result.is_success else '❌ 失败'}",
                f"- **总测试数**: {result.total_tests}",
                f"- **成功率**: {result.success_rate:.1f}%",
                ".2"
            ])

            if result.coverage is not None:
                report_lines.append(f"- **覆盖率**: {result.coverage:.1f}%")

            if result.failed_tests > 0 or result.errors > 0:
                report_lines.extend([
                    f"- **失败**: {result.failed_tests}",
                    f"- **错误**: {result.errors}",
                    f"- **返回码**: {result.return_code}"
                ])

            if result.stderr:
                report_lines.extend([
                    "- **错误信息**:",
                    "```",
                    result.stderr.strip(),
                    "```"
                ])

            report_lines.append("")

        # 保存报告
        report_path = self.project_root / output_file
        report_path.parent.mkdir(parents=True, exist_ok=True)

        report_content = "\n".join(report_lines)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        logger.info(f"测试报告已保存到: {report_path}")
        return report_content

    def save_results_json(self, results: Dict[str, TestResult] = None,
                         output_file: str = "test_logs/test_results.json"):
        """保存结果为JSON格式"""
        if results is None:
            results = self.results

        # 转换为可序列化的字典
        json_data = {}
        for layer_name, result in results.items():
            json_data[layer_name] = {
                'layer_name': result.layer_name,
                'total_tests': result.total_tests,
                'passed_tests': result.passed_tests,
                'failed_tests': result.failed_tests,
                'skipped_tests': result.skipped_tests,
                'errors': result.errors,
                'execution_time': result.execution_time,
                'coverage': result.coverage,
                'return_code': result.return_code,
                'success_rate': result.success_rate,
                'is_success': result.is_success
            }

        # 保存到文件
        output_path = self.project_root / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        logger.info(f"测试结果JSON已保存到: {output_path}")


# 全局测试执行器实例
test_executor = TestExecutor()


def get_test_executor() -> TestExecutor:
    """获取测试执行器实例"""
    return test_executor


def run_layer_tests(layer_name: str, **kwargs) -> TestResult:
    """
    运行层级测试 - 便捷函数

    Args:
        layer_name: 层名称
        **kwargs: 执行参数

    Returns:
        TestResult对象
    """
    return test_executor.execute_layer_tests(layer_name, **kwargs)


def run_all_tests(layers: List[str] = None, **kwargs) -> Dict[str, TestResult]:
    """
    运行所有测试 - 便捷函数

    Args:
        layers: 层列表
        **kwargs: 执行参数

    Returns:
        测试结果字典
    """
    return test_executor.execute_all_layers(layers, **kwargs)


if __name__ == "__main__":
    # 测试执行器
    print("测试执行器功能验证...")

    executor = get_test_executor()
    print(f"项目根目录: {executor.project_root}")

    # 测试基础设施层（不实际运行pytest，避免递归）
    print("测试执行器初始化完成 - 实际测试请使用命令行接口")

    print("测试执行器验证完成!")
