#!/usr/bin/env python3
"""
RQA2025 CI / CD集成工具
CI / CD Integration Tools

提供与CI / CD流水线的集成功能。
"""

import subprocess
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

# 获取统一基础设施集成层的日志适配器
try:
    from src.core.integration import get_models_adapter
    models_adapter = get_models_adapter()
    logger = logging.getLogger(__name__)
except Exception as e:
    # 如果导入失败，使用标准logging
    logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:

    """流水线执行结果"""
    pipeline_id: str
    stage: str
    status: str  # success, failure, running, pending
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    logs: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityGate:

    """质量门禁"""
    gate_id: str
    name: str
    conditions: Dict[str, Any]
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)


class CICDTools:

    """
    CI / CD集成工具
    提供与CI / CD流水线的集成和质量保障功能
    """

    def __init__(self):

        self.pipeline_results: Dict[str, PipelineResult] = {}
        self.quality_gates: Dict[str, QualityGate] = {}

        # 默认质量门禁
        self._setup_default_quality_gates()

        logger.info("CI / CD集成工具已初始化")

    def run_pipeline_stage(self, stage_name: str, commands: List[str],


                           working_dir: Optional[str] = None) -> PipelineResult:
        """运行流水线阶段"""
        pipeline_id = f"pipeline_{stage_name}_{datetime.now().strftime('%Y % m % d % H % M % S % f')}"

        result = PipelineResult(
            pipeline_id=pipeline_id,
            stage=stage_name,
            status="running",
            start_time=datetime.now()
        )

        self.pipeline_results[pipeline_id] = result

        try:
            # 执行命令
            for command in commands:
                result.logs.append(f"Executing: {command}")

                # 运行命令
                process_result = self._run_command(command, working_dir)

                result.logs.extend(process_result['logs'])

                if process_result['return_code'] != 0:
                    result.status = "failure"
                    break
            else:
                result.status = "success"

        except Exception as e:
            result.status = "failure"
            result.logs.append(f"Pipeline execution error: {str(e)}")

        finally:
            result.end_time = datetime.now()
            if result.end_time and result.start_time:
                result.duration_seconds = (result.end_time - result.start_time).total_seconds()

        logger.info(f"流水线阶段 {stage_name} 执行完成，状态: {result.status}")
        return result

    def check_quality_gates(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """检查质量门禁"""
        results = {}

        for gate_id, gate in self.quality_gates.items():
            if not gate.enabled:
                continue

            gate_result = self._evaluate_quality_gate(gate, metrics)
            results[gate_id] = gate_result

        # 汇总结果
        passed_gates = sum(1 for r in results.values() if r['passed'])
        total_gates = len(results)

        summary = {
            'total_gates': total_gates,
            'passed_gates': passed_gates,
            'failed_gates': total_gates - passed_gates,
            'overall_passed': passed_gates == total_gates,
            'gate_results': results
        }

        logger.info(f"质量门禁检查完成: {passed_gates}/{total_gates} 通过")
        return summary

    def generate_test_report(self, test_results: Dict[str, Any]) -> str:
        """生成测试报告"""
        report = f"""# 测试报告

生成时间: {datetime.now().isoformat()}

# # 测试摘要

- 总测试数: {test_results.get('total_tests', 0)}
- 通过测试: {test_results.get('passed_tests', 0)}
- 失败测试: {test_results.get('failed_tests', 0)}
- 跳过测试: {test_results.get('skipped_tests', 0)}
- 成功率: {test_results.get('success_rate', 0):.2f}%

# # 测试详情

"""

        # 添加详细的测试结果
        if 'details' in test_results:
            for test_name, test_result in test_results['details'].items():
                report += f"### {test_name}\n"
                report += f"- 状态: {test_result.get('status', 'unknown')}\n"
                report += f"- 执行时间: {test_result.get('duration_ms', 0)}ms\n"
                if 'error' in test_result:
                    report += f"- 错误: {test_result['error']}\n"
                report += "\n"

        return report

    def generate_coverage_report(self, coverage_data: Dict[str, Any]) -> str:
        """生成覆盖率报告"""
        report = f"""# 代码覆盖率报告

生成时间: {datetime.now().isoformat()}

# # 覆盖率摘要

- 总模块数: {coverage_data.get('total_modules', 0)}
- 总行数: {coverage_data.get('total_lines', 0)}
- 覆盖行数: {coverage_data.get('covered_lines', 0)}
- 整体覆盖率: {coverage_data.get('overall_coverage_percent', 0):.2f}%

# # 模块覆盖率

"""

        # 添加各模块的覆盖率详情
        if 'module_coverage' in coverage_data:
            for module_name, coverage in coverage_data['module_coverage'].items():
                report += f"### {module_name}\n"
                report += f"- 行覆盖率: {coverage.get('line_coverage', 0):.2f}%\n"
                report += f"- 分支覆盖率: {coverage.get('branch_coverage', 0):.2f}%\n"
                report += f"- 函数覆盖率: {coverage.get('function_coverage', 0):.2f}%\n\n"

        return report

    def deploy_to_environment(self, environment: str, artifacts: Dict[str, Any]) -> bool:
        """部署到指定环境"""
        logger.info(f"开始部署到环境: {environment}")

        try:
            # 这里应该实现实际的部署逻辑
            # 例如：调用部署脚本、更新配置等

            # 模拟部署过程
            deployment_commands = [
                f"echo 'Deploying to {environment}'",
                f"echo 'Artifacts: {list(artifacts.keys())}'",
                # 添加实际的部署命令
            ]

            result = self.run_pipeline_stage(
                f"deploy_{environment}",
                deployment_commands
            )

            success = result.status == "success"
            logger.info(f"部署到 {environment} {'成功' if success else '失败'}")

            return success

        except Exception as e:
            logger.error(f"部署失败: {str(e)}")
            return False

    def rollback_deployment(self, environment: str, backup_version: str) -> bool:
        """回滚部署"""
        logger.info(f"开始回滚环境 {environment} 到版本 {backup_version}")

        try:
            # 实现回滚逻辑
            rollback_commands = [
                f"echo 'Rolling back {environment} to {backup_version}'",
                # 添加实际的回滚命令
            ]

            result = self.run_pipeline_stage(
                f"rollback_{environment}",
                rollback_commands
            )

            success = result.status == "success"
            logger.info(f"回滚 {environment} {'成功' if success else '失败'}")

            return success

        except Exception as e:
            logger.error(f"回滚失败: {str(e)}")
            return False

    def _run_command(self, command: str, working_dir: Optional[str] = None) -> Dict[str, Any]:
        """运行命令"""
        try:
            # 设置工作目录
            cwd = Path(working_dir) if working_dir else None

            # 执行命令
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )

            logs = []
            if result.stdout:
                logs.extend(result.stdout.strip().split('\n'))
            if result.stderr:
                logs.extend(result.stderr.strip().split('\n'))

            return {
                'return_code': result.returncode,
                'logs': logs,
                'stdout': result.stdout,
                'stderr': result.stderr
            }

        except subprocess.TimeoutExpired:
            return {
                'return_code': -1,
                'logs': ['Command timed out after 300 seconds'],
                'stdout': '',
                'stderr': 'Timeout'
            }

        except Exception as e:
            return {
                'return_code': -1,
                'logs': [f'Command execution error: {str(e)}'],
                'stdout': '',
                'stderr': str(e)
            }

    def _setup_default_quality_gates(self):
        """设置默认质量门禁"""
        # 单元测试覆盖率门禁
        unit_test_coverage_gate = QualityGate(
            gate_id="unit_test_coverage",
            name="单元测试覆盖率",
            conditions={
                'min_coverage_percent': 80.0,
                'max_failed_tests': 0
            }
        )

        # 集成测试通过率门禁
        integration_test_gate = QualityGate(
            gate_id="integration_test_pass",
            name="集成测试通过",
            conditions={
                'min_pass_rate': 95.0,
                'max_failed_tests': 5
            }
        )

        # 代码质量门禁
        code_quality_gate = QualityGate(
            gate_id="code_quality",
            name="代码质量检查",
            conditions={
                'max_complexity': 10,
                'max_line_length': 120,
                'require_docstrings': True
            }
        )

        # 安全检查门禁
        security_gate = QualityGate(
            gate_id="security_check",
            name="安全检查",
            conditions={
                'block_high_severity': True,
                'max_medium_severity': 5,
                'require_sast_scan': True
            }
        )

        self.quality_gates = {
            unit_test_coverage_gate.gate_id: unit_test_coverage_gate,
            integration_test_gate.gate_id: integration_test_gate,
            code_quality_gate.gate_id: code_quality_gate,
            security_gate.gate_id: security_gate
        }

    def _evaluate_quality_gate(self, gate: QualityGate, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """评估质量门禁"""
        conditions = gate.conditions
        passed = True
        failed_conditions = []

        for condition_key, condition_value in conditions.items():
            if condition_key == 'min_coverage_percent':
                actual_coverage = metrics.get('coverage_percent', 0)
                if actual_coverage < condition_value:
                    passed = False
                    failed_conditions.append(f"覆盖率 {actual_coverage:.2f}% < {condition_value}%")

            elif condition_key == 'max_failed_tests':
                actual_failed = metrics.get('failed_tests', 0)
                if actual_failed > condition_value:
                    passed = False
                    failed_conditions.append(f"失败测试数 {actual_failed} > {condition_value}")

            elif condition_key == 'min_pass_rate':
                actual_rate = metrics.get('pass_rate', 0)
                if actual_rate < condition_value:
                    passed = False
                    failed_conditions.append(f"通过率 {actual_rate:.2f}% < {condition_value}%")

            # 添加更多条件检查...

        return {
            'gate_id': gate.gate_id,
            'gate_name': gate.name,
            'passed': passed,
            'failed_conditions': failed_conditions
        }

    def get_pipeline_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取流水线执行历史"""
        history = list(self.pipeline_results.values())
        history.sort(key=lambda x: x.start_time, reverse=True)

        return [
            {
                'pipeline_id': result.pipeline_id,
                'stage': result.stage,
                'status': result.status,
                'start_time': result.start_time.isoformat(),
                'end_time': result.end_time.isoformat() if result.end_time else None,
                'duration_seconds': result.duration_seconds
            }
            for result in history[:limit]
        ]


# 创建全局CI / CD工具实例
_ci_cd_tools = None


def get_ci_cd_tools() -> CICDTools:
    """获取全局CI / CD工具实例"""
    global _ci_cd_tools
    if _ci_cd_tools is None:
        _ci_cd_tools = CICDTools()
    return _ci_cd_tools


__all__ = [
    'CICDTools', 'PipelineResult', 'QualityGate', 'get_ci_cd_tools'
]
