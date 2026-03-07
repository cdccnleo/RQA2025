"""
JSON报告器

将检查结果输出为JSON格式。
"""

import json
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from ..core.check_result import CheckResult


class JsonReporter:
    """
    JSON报告器

    将检查结果序列化为JSON格式，便于程序处理和存储。
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化报告器

        Args:
            config: 配置选项
        """
        self.config = config or {}
        self.output_file = self.config.get('output_file')
        self.pretty_print = self.config.get('pretty_print', True)
        self.include_metadata = self.config.get('include_metadata', True)

    def report(self, results: Dict[str, CheckResult]) -> str:
        """
        生成JSON报告

        Args:
            results: 检查结果字典

        Returns:
            str: JSON字符串
        """
        # 构建报告数据
        report_data = self._build_report_data(results)

        # 序列化为JSON
        json_str = self._serialize_to_json(report_data)

        # 如果指定了输出文件，写入文件
        if self.output_file:
            self._write_to_file(json_str)

        return json_str

    def _build_report_data(self, results) -> Dict[str, Any]:
        """
        构建报告数据结构

        Args:
            results: 检查结果字典（CheckResult对象或字典）

        Returns:
            Dict[str, Any]: 报告数据
        """
        # 计算总体统计
        total_issues = 0
        total_errors = 0
        total_warnings = 0
        total_criticals = 0
        total_duration = 0.0

        for result in results.values():
            if hasattr(result, 'get_issue_count'):
                # CheckResult对象
                total_issues += result.get_issue_count()
                total_errors += result.get_issue_count('ERROR')
                total_warnings += result.get_issue_count('WARNING')
                total_criticals += result.get_issue_count('CRITICAL')
                total_duration += result.get_duration()
            else:
                # 字典格式
                summary = result.get('summary', {})
                total_issues += summary.get('total_issues', 0)
                total_errors += summary.get('issues_by_severity', {}).get('error', 0)
                total_warnings += summary.get('issues_by_severity', {}).get('warning', 0)
                total_criticals += summary.get('issues_by_severity', {}).get('critical', 0)
                total_duration += summary.get('duration_seconds', 0.0)

        report_data = {
            'report_info': {
                'title': '基础设施层质量检查报告',
                'generated_at': datetime.now().isoformat(),
                'version': '1.0.0',
                'checkers_executed': len(results)
            },
            'summary': {
                'total_issues': total_issues,
                'total_errors': total_errors,
                'total_warnings': total_warnings,
                'total_criticals': total_criticals,
                'total_duration_seconds': total_duration,
                'status': self._determine_overall_status(results)
            },
            'results': {}
        }

        # 添加每个检查器的详细结果
        for checker_name, result in results.items():
            if hasattr(result, 'to_dict'):
                # CheckResult对象
                report_data['results'][checker_name] = result.to_dict()
            else:
                # 已经是字典格式
                report_data['results'][checker_name] = result

        # 添加元数据
        if self.include_metadata:
            report_data['metadata'] = {
                'config': self.config,
                'system_info': self._get_system_info()
            }

        return report_data

    def _determine_overall_status(self, results) -> str:
        """
        确定总体状态

        Args:
            results: 检查结果字典（CheckResult对象或字典）

        Returns:
            str: 总体状态
        """
        has_critical = False
        has_errors = False

        for result in results.values():
            if hasattr(result, 'has_critical_issues'):
                # CheckResult对象
                if result.has_critical_issues():
                    has_critical = True
                if result.has_errors():
                    has_errors = True
            else:
                # 字典格式
                summary = result.get('summary', {})
                severity_counts = summary.get('issues_by_severity', {})
                if severity_counts.get('critical', 0) > 0:
                    has_critical = True
                if severity_counts.get('error', 0) > 0:
                    has_errors = True

        if has_critical:
            return 'failed'
        elif has_errors:
            return 'warning'
        else:
            return 'passed'

    def _serialize_to_json(self, data: Dict[str, Any]) -> str:
        """
        序列化为JSON字符串

        Args:
            data: 数据字典

        Returns:
            str: JSON字符串
        """
        indent = 2 if self.pretty_print else None
        return json.dumps(data, indent=indent, ensure_ascii=False)

    def _write_to_file(self, json_str: str) -> None:
        """
        写入到文件

        Args:
            json_str: JSON字符串
        """
        try:
            output_path = Path(self.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_str)

            print(f"JSON报告已保存到: {self.output_file}")

        except Exception as e:
            print(f"保存JSON报告失败: {e}")

    def _get_system_info(self) -> Dict[str, Any]:
        """
        获取系统信息

        Returns:
            Dict[str, Any]: 系统信息
        """
        try:
            import platform
            import sys

            return {
                'platform': platform.platform(),
                'python_version': sys.version,
                'architecture': platform.architecture(),
                'processor': platform.processor()
            }
        except Exception:
            return {'error': '无法获取系统信息'}

    def load_report(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        从文件加载报告

        Args:
            file_path: 文件路径

        Returns:
            Optional[Dict[str, Any]]: 报告数据
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载报告失败: {e}")
            return None

    def compare_reports(self, report1: Dict[str, Any], report2: Dict[str, Any]) -> Dict[str, Any]:
        """
        比较两个报告

        Args:
            report1: 第一个报告
            report2: 第二个报告

        Returns:
            Dict[str, Any]: 比较结果
        """
        def get_summary(report):
            return report.get('summary', {})

        summary1 = get_summary(report1)
        summary2 = get_summary(report2)

        comparison = {
            'issues_change': summary2.get('total_issues', 0) - summary1.get('total_issues', 0),
            'errors_change': summary2.get('total_errors', 0) - summary1.get('total_errors', 0),
            'warnings_change': summary2.get('total_warnings', 0) - summary1.get('total_warnings', 0),
            'criticals_change': summary2.get('total_criticals', 0) - summary1.get('total_criticals', 0),
            'duration_change': summary2.get('total_duration_seconds', 0) - summary1.get('total_duration_seconds', 0)
        }

        # 计算改进趋势
        if comparison['issues_change'] < 0:
            comparison['trend'] = 'improving'
        elif comparison['issues_change'] > 0:
            comparison['trend'] = 'worsening'
        else:
            comparison['trend'] = 'stable'

        return comparison
