"""
控制台报告器

将检查结果输出到控制台。
"""

from typing import Dict, Any, List

from ..core.check_result import CheckResult, IssueSeverity


class ConsoleReporter:
    """
    控制台报告器

    以易读的格式将检查结果输出到控制台。
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化报告器

        Args:
            config: 配置选项
        """
        self.config = config or {}
        self.colors = self.config.get('colors', True)
        self.verbose = self.config.get('verbose', False)

    def report(self, results: Dict[str, CheckResult]) -> None:
        """
        生成控制台报告

        Args:
            results: 检查结果字典
        """
        print("\n" + "="*80)
        print("🎯 基础设施层质量检查报告")
        print("="*80)

        # 总体统计
        self._print_summary(results)

        # 详细结果
        if results:
            print("\n" + "-"*80)
            print("📋 详细检查结果")
            print("-"*80)

            for checker_name, result in results.items():
                self._print_checker_result(checker_name, result)

        print("\n" + "="*80)
        print("✅ 检查完成")
        print("="*80)

    def _print_summary(self, results) -> None:
        """打印总体统计"""
        total_issues = 0
        total_errors = 0
        total_warnings = 0
        total_criticals = 0
        total_duration = 0.0

        for result in results.values():
            # 检查是CheckResult对象还是字典
            if hasattr(result, 'get_issue_count'):
                # CheckResult对象
                total_issues += result.get_issue_count()
                total_errors += result.get_issue_count(IssueSeverity.ERROR)
                total_warnings += result.get_issue_count(IssueSeverity.WARNING)
                total_criticals += result.get_issue_count(IssueSeverity.CRITICAL)
                total_duration += result.get_duration()
            else:
                # 字典格式
                summary = result.get('summary', {})
                total_issues += summary.get('total_issues', 0)
                total_errors += summary.get('issues_by_severity', {}).get('error', 0)
                total_warnings += summary.get('issues_by_severity', {}).get('warning', 0)
                total_criticals += summary.get('issues_by_severity', {}).get('critical', 0)
                total_duration += summary.get('duration_seconds', 0.0)

        print(f"\n📊 检查概览:")
        print(f"   🔍 执行检查器: {len(results)} 个")
        print(f"   📝 发现问题: {total_issues} 个")
        print(f"   ❌ 错误: {self._color_text(str(total_errors), 'red')}")
        print(f"   ⚠️  警告: {self._color_text(str(total_warnings), 'yellow')}")
        print(f"   🚨 严重问题: {self._color_text(str(total_criticals), 'red', bold=True)}")
        print(f"   ⏱️  总耗时: {total_duration:.2f} 秒")

        # 检查状态
        if total_criticals > 0:
            print(f"   🚫 状态: {self._color_text('失败', 'red', bold=True)}")
        elif total_errors > 0:
            print(f"   ⚠️  状态: {self._color_text('有错误', 'yellow')}")
        else:
            print(f"   ✅ 状态: {self._color_text('通过', 'green')}")

    def _print_checker_result(self, checker_name: str, result) -> None:
        """打印单个检查器的结果"""
        # 检查是CheckResult对象还是字典
        if hasattr(result, 'issues'):
            # CheckResult对象
            issues = result.issues
            duration = result.get_duration()
            metadata = result.metadata
        else:
            # 字典格式
            summary = result.get('summary', {})
            issues_data = result.get('issues', [])
            duration = summary.get('duration_seconds', 0.0)
            metadata = summary.get('metadata', {})

            print(f"\n🔍 {checker_name}")
            print(f"   ⏱️  耗时: {duration:.2f}秒")
            print(f"   📝 问题: {len(issues_data)}个")

            # 按严重程度分组显示问题（简化处理）
            severity_counts = summary.get('issues_by_severity', {})
            for severity_name, count in severity_counts.items():
                if count > 0:
                    severity_display = {
                        'critical': '🚨 严重',
                        'error': '❌ 错误',
                        'warning': '⚠️  警告',
                        'info': 'ℹ️  信息'
                    }.get(severity_name, severity_name)
                    print(f"   {severity_display}: {count}个")

            # 显示元数据
            if metadata:
                print(f"   📊 元数据: {metadata}")
            return

        print(f"\n🔍 {checker_name}")
        print(f"   ⏱️  耗时: {duration:.2f}秒")
        print(f"   📝 问题: {len(issues)}个")

        if issues:
            # 按严重程度分组
            issues_by_severity = self._group_issues_by_severity(issues)

            for severity in [IssueSeverity.CRITICAL, IssueSeverity.ERROR,
                             IssueSeverity.WARNING, IssueSeverity.INFO]:
                severity_issues = issues_by_severity.get(severity, [])
                if severity_issues:
                    self._print_issues_by_severity(severity, severity_issues)

        # 显示元数据
        if metadata:
            print(f"   📊 元数据: {metadata}")

    def _group_issues_by_severity(self, issues: List['Issue']) -> Dict[IssueSeverity, List['Issue']]:
        """按严重程度分组问题"""
        grouped = {}
        for issue in issues:
            if issue.severity not in grouped:
                grouped[issue.severity] = []
            grouped[issue.severity].append(issue)
        return grouped

    def _print_issues_by_severity(self, severity: IssueSeverity,
                                  issues: List['Issue']) -> None:
        """打印按严重程度分组的问题"""
        severity_name = {
            IssueSeverity.CRITICAL: "🚨 严重",
            IssueSeverity.ERROR: "❌ 错误",
            IssueSeverity.WARNING: "⚠️  警告",
            IssueSeverity.INFO: "ℹ️  信息"
        }.get(severity, str(severity))

        print(f"   {severity_name} ({len(issues)}个):")

        for issue in issues[:10]:  # 只显示前10个
            line_info = f":{issue.line_number}" if issue.line_number else ""
            print(f"      • {issue.file_path}{line_info}")
            print(f"        {issue.message}")

            if issue.details and self.verbose:
                for key, value in issue.details.items():
                    print(f"        └─ {key}: {value}")

        if len(issues) > 10:
            print(f"      ... 还有 {len(issues) - 10} 个问题")

    def _color_text(self, text: str, color: str, bold: bool = False) -> str:
        """
        为文本添加颜色

        Args:
            text: 文本
            color: 颜色名称
            bold: 是否粗体

        Returns:
            str: 带颜色的文本
        """
        if not self.colors:
            return text

        color_codes = {
            'red': '31',
            'green': '32',
            'yellow': '33',
            'blue': '34',
            'magenta': '35',
            'cyan': '36',
            'white': '37'
        }

        color_code = color_codes.get(color, '37')
        bold_code = '1;' if bold else ''

        return f"\033[{bold_code}{color_code}m{text}\033[0m"
