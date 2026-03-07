"""
质量检查器主控制器

协调和管理所有质量检查器的执行。
"""

from typing import Dict, Any, Optional
import logging

from .base_checker import BaseChecker
from .check_result import CheckResult, IssueSeverity
from ..checkers.duplicate_checker import DuplicateCodeChecker
from ..checkers.interface_checker import InterfaceConsistencyChecker
from ..checkers.complexity_checker import ComplexityChecker


class QualityChecker:
    """
    质量检查器主控制器

    统一管理和执行各种质量检查。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化质量检查器

        Args:
            config: 全局配置
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # 初始化所有检查器
        self.checkers = self._initialize_checkers()

        # 设置默认配置
        self._setup_default_config()

    def _setup_default_config(self) -> None:
        """设置默认配置"""
        defaults = {
            'enabled_checkers': ['duplicate', 'interface', 'complexity'],
            'fail_on_error': True,
            'fail_on_critical': True,
            'parallel_execution': True,
            'max_workers': 4,
            'exclude_patterns': ['__pycache__', '.git', 'node_modules', '*.pyc']
        }

        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value

    def _initialize_checkers(self) -> Dict[str, BaseChecker]:
        """初始化所有检查器"""
        checkers = {}

        # 代码重复检查器
        checkers['duplicate'] = DuplicateCodeChecker(
            self.config.get('duplicate', {})
        )

        # 接口一致性检查器
        checkers['interface'] = InterfaceConsistencyChecker(
            self.config.get('interface', {})
        )

        # 复杂度检查器
        checkers['complexity'] = ComplexityChecker(
            self.config.get('complexity', {})
        )

        return checkers

    def run_checks(self, target_path: str) -> Dict[str, CheckResult]:
        """
        运行所有启用的检查

        Args:
            target_path: 检查目标路径

        Returns:
            Dict[str, CheckResult]: 检查结果字典
        """
        self.logger.info(f"开始质量检查: {target_path}")

        results = {}
        enabled_checkers = self.config.get('enabled_checkers', [])

        for checker_name in enabled_checkers:
            if checker_name in self.checkers:
                self.logger.info(f"执行检查器: {checker_name}")
                checker = self.checkers[checker_name]
                result = checker.check(target_path)
                results[checker_name] = result

                # 记录检查结果统计
                total_issues = result.get_issue_count()
                errors = result.get_issue_count(IssueSeverity.ERROR)
                criticals = result.get_issue_count(IssueSeverity.CRITICAL)

                self.logger.info(
                    f"{checker_name} 检查完成: "
                    f"{total_issues}个问题 ({errors}错误, {criticals}严重) "
                    f"耗时{result.get_duration():.2f}秒"
                )

        self.logger.info(f"质量检查完成，共执行{len(results)}个检查器")
        return results

    def check_should_fail(self, results: Dict[str, CheckResult]) -> bool:
        """
        检查是否应该失败（用于CI/CD）

        Args:
            results: 检查结果

        Returns:
            bool: 是否应该失败
        """
        fail_on_error = self.config.get('fail_on_error', True)
        fail_on_critical = self.config.get('fail_on_critical', True)

        for result in results.values():
            if fail_on_critical and result.has_critical_issues():
                return True
            if fail_on_error and result.has_errors():
                return True

        return False

    def get_summary_report(self, results: Dict[str, CheckResult]) -> Dict[str, Any]:
        """
        生成汇总报告

        Args:
            results: 检查结果

        Returns:
            Dict[str, Any]: 汇总报告
        """
        total_issues = 0
        total_errors = 0
        total_criticals = 0
        total_duration = 0.0

        checker_summaries = {}

        for checker_name, result in results.items():
            summary = result.get_summary()
            checker_summaries[checker_name] = summary

            total_issues += summary['total_issues']
            total_errors += summary['issues_by_severity']['error']
            total_criticals += summary['issues_by_severity']['critical']
            total_duration += summary['duration_seconds']

        return {
            'total_checkers': len(results),
            'total_issues': total_issues,
            'total_errors': total_errors,
            'total_criticals': total_criticals,
            'total_duration_seconds': total_duration,
            'should_fail': self.check_should_fail(results),
            'checker_summaries': checker_summaries
        }

    def run_quality_check(self, target_path: str) -> Dict[str, Any]:
        """
        执行完整的质量检查流程

        Args:
            target_path: 检查目标路径

        Returns:
            Dict[str, Any]: 完整检查报告
        """
        # 运行所有检查
        results = self.run_checks(target_path)

        # 生成汇总报告
        summary = self.get_summary_report(results)

        # 返回完整报告
        return {
            'summary': summary,
            'results': {name: result.to_dict() for name, result in results.items()},
            'config': self.config
        }
