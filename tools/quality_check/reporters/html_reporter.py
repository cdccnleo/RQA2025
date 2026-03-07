"""
HTML报告器

将检查结果输出为HTML格式的报告。
"""

from typing import Dict, Any
from pathlib import Path
from datetime import datetime

from ..core.check_result import CheckResult, IssueSeverity


class HtmlReporter:
    """
    HTML报告器

    生成美观的HTML格式质量检查报告。
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化报告器

        Args:
            config: 配置选项
        """
        self.config = config or {}
        self.output_file = self.config.get('output_file', 'quality_report.html')
        self.include_charts = self.config.get('include_charts', True)
        self.theme = self.config.get('theme', 'default')

    def report(self, results: Dict[str, CheckResult]) -> str:
        """
        生成HTML报告

        Args:
            results: 检查结果字典

        Returns:
            str: HTML字符串
        """
        # 构建报告数据
        report_data = self._build_report_data(results)

        # 生成HTML
        html_content = self._generate_html(report_data)

        # 写入文件
        if self.output_file:
            self._write_to_file(html_content)

        return html_content

    def _build_report_data(self, results: Dict[str, CheckResult]) -> Dict[str, Any]:
        """构建报告数据"""
        total_issues = sum(result.get_issue_count() for result in results.values())
        total_errors = sum(result.get_issue_count(IssueSeverity.ERROR)
                           for result in results.values())
        total_warnings = sum(result.get_issue_count(IssueSeverity.WARNING)
                             for result in results.values())
        total_criticals = sum(result.get_issue_count(IssueSeverity.CRITICAL)
                              for result in results.values())
        total_duration = sum(result.get_duration() for result in results.values())

        return {
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_checkers': len(results),
                'total_issues': total_issues,
                'total_errors': total_errors,
                'total_warnings': total_warnings,
                'total_criticals': total_criticals,
                'total_duration_seconds': total_duration,
                'status': self._get_status(total_criticals, total_errors)
            },
            'results': results,
            'charts_data': self._prepare_charts_data(results) if self.include_charts else None
        }

    def _get_status(self, criticals: int, errors: int) -> str:
        """获取状态"""
        if criticals > 0:
            return 'failed'
        elif errors > 0:
            return 'warning'
        else:
            return 'passed'

    def _prepare_charts_data(self, results: Dict[str, CheckResult]) -> Dict[str, Any]:
        """准备图表数据"""
        # 按检查器分组的问题数量
        checker_data = {}
        for checker_name, result in results.items():
            checker_data[checker_name] = {
                'issues': result.get_issue_count(),
                'errors': result.get_issue_count(IssueSeverity.ERROR),
                'warnings': result.get_issue_count(IssueSeverity.WARNING),
                'criticals': result.get_issue_count(IssueSeverity.CRITICAL)
            }

        # 按严重程度分组的总体问题
        severity_data = {
            'Critical': sum(r.get_issue_count(IssueSeverity.CRITICAL) for r in results.values()),
            'Error': sum(r.get_issue_count(IssueSeverity.ERROR) for r in results.values()),
            'Warning': sum(r.get_issue_count(IssueSeverity.WARNING) for r in results.values()),
            'Info': sum(r.get_issue_count(IssueSeverity.INFO) for r in results.values())
        }

        return {
            'checker_data': checker_data,
            'severity_data': severity_data
        }

    def _generate_html(self, data: Dict[str, Any]) -> str:
        """生成HTML内容"""
        template = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>基础设施层质量检查报告</title>
    {self._get_css_styles()}
    {self._get_chart_scripts() if self.include_charts and data.get('charts_data') else ''}
</head>
<body>
    <div class="container">
        <header>
            <h1>🎯 基础设施层质量检查报告</h1>
            <div class="report-info">
                <span>生成时间: {data['generated_at'][:19].replace('T', ' ')}</span>
                <span>版本: 1.0.0</span>
            </div>
        </header>

        {self._generate_summary_section(data)}

        {self._generate_charts_section(data.get('charts_data')) if self.include_charts and data.get('charts_data') else ''}

        {self._generate_detailed_results(data['results'])}

        <footer>
            <p>报告由专项修复小组质量检查工具生成</p>
        </footer>
    </div>

    <script>
        {self._get_javascript()}
    </script>
</body>
</html>"""

        return template

    def _get_css_styles(self) -> str:
        """获取CSS样式"""
        return """
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                line-height: 1.6;
                color: #333;
                background-color: #f5f5f5;
            }

            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }

            header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
                text-align: center;
            }

            header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
            }

            .report-info {
                display: flex;
                justify-content: center;
                gap: 20px;
                font-size: 0.9em;
            }

            .summary-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }

            .summary-card {
                background: white;
                padding: 25px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                text-align: center;
            }

            .summary-card h3 {
                font-size: 2em;
                margin-bottom: 5px;
            }

            .summary-card p {
                color: #666;
                font-size: 0.9em;
            }

            .status-badge {
                display: inline-block;
                padding: 5px 15px;
                border-radius: 20px;
                font-weight: bold;
                text-transform: uppercase;
                font-size: 0.8em;
            }

            .status-passed { background-color: #d4edda; color: #155724; }
            .status-warning { background-color: #fff3cd; color: #856404; }
            .status-failed { background-color: #f8d7da; color: #721c24; }

            .issues-section {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 30px;
            }

            .issues-table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }

            .issues-table th,
            .issues-table td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }

            .issues-table th {
                background-color: #f8f9fa;
                font-weight: 600;
            }

            .severity-critical { color: #dc3545; font-weight: bold; }
            .severity-error { color: #dc3545; }
            .severity-warning { color: #ffc107; }
            .severity-info { color: #17a2b8; }

            .charts-section {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 30px;
            }

            .chart-container {
                margin-bottom: 30px;
            }

            .chart-container h3 {
                margin-bottom: 20px;
                color: #333;
            }

            footer {
                text-align: center;
                color: #666;
                margin-top: 50px;
                padding: 20px;
                border-top: 1px solid #ddd;
            }

            @media (max-width: 768px) {
                .container {
                    padding: 10px;
                }

                header {
                    padding: 20px;
                }

                header h1 {
                    font-size: 2em;
                }

                .summary-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        """

    def _get_chart_scripts(self) -> str:
        """获取图表脚本"""
        return """
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        """

    def _get_javascript(self) -> str:
        """获取JavaScript代码"""
        return """
        // 页面加载完成后初始化图表
        document.addEventListener('DOMContentLoaded', function() {
            initializeCharts();
        });

        function initializeCharts() {
            // 图表初始化逻辑会在这里添加
            console.log('质量检查报告已加载');
        }
        """

    def _generate_summary_section(self, data: Dict[str, Any]) -> str:
        """生成汇总部分"""
        summary = data['summary']
        report_info = data.get('report_info', {})

        status_class = f"status-{summary['status']}"
        status_text = {
            'passed': '✅ 通过',
            'warning': '⚠️ 警告',
            'failed': '❌ 失败'
        }.get(summary['status'], summary['status'])

        return f"""
        <div class="summary-grid">
            <div class="summary-card">
                <h3>{report_info.get('checkers_executed', 'N/A')}</h3>
                <p>执行检查器</p>
            </div>
            <div class="summary-card">
                <h3>{summary['total_issues']}</h3>
                <p>发现问题</p>
            </div>
            <div class="summary-card">
                <h3 style="color: #dc3545;">{summary['total_errors']}</h3>
                <p>错误</p>
            </div>
            <div class="summary-card">
                <h3 style="color: #ffc107;">{summary['total_warnings']}</h3>
                <p>警告</p>
            </div>
            <div class="summary-card">
                <h3>{summary['total_duration_seconds']:.2f}s</h3>
                <p>总耗时</p>
            </div>
            <div class="summary-card">
                <span class="status-badge {status_class}">{status_text}</span>
                <p>检查状态</p>
            </div>
        </div>
        """

    def _generate_charts_section(self, charts_data: Dict[str, Any]) -> str:
        """生成图表部分"""
        return f"""
        <div class="charts-section">
            <h2>📊 数据可视化</h2>
            <div class="chart-container">
                <h3>问题严重程度分布</h3>
                <canvas id="severityChart" width="400" height="200"></canvas>
            </div>
            <div class="chart-container">
                <h3>各检查器问题统计</h3>
                <canvas id="checkerChart" width="400" height="200"></canvas>
            </div>
        </div>

        <script>
            // 严重程度图表
            const severityCtx = document.getElementById('severityChart').getContext('2d');
            new Chart(severityCtx, {{
                type: 'doughnut',
                data: {{
                    labels: {list(charts_data['severity_data'].keys())},
                    datasets: [{{
                        data: {list(charts_data['severity_data'].values())},
                        backgroundColor: ['#dc3545', '#fd7e14', '#ffc107', '#17a2b8']
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        legend: {{
                            position: 'bottom'
                        }}
                    }}
                }}
            }});

            // 检查器图表
            const checkerCtx = document.getElementById('checkerChart').getContext('2d');
            const checkerLabels = {list(charts_data['checker_data'].keys())};
            const checkerIssues = checkerLabels.map(function(label) {{
                return charts_data['checker_data'][label]['issues'];
            }});

            new Chart(checkerCtx, {{
                type: 'bar',
                data: {{
                    labels: checkerLabels,
                    datasets: [{{
                        label: '问题数量',
                        data: checkerIssues,
                        backgroundColor: '#667eea'
                    }}]
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        y: {{
                            beginAtZero: true
                        }}
                    }}
                }}
            }});
        </script>
        """

    def _generate_detailed_results(self, results: Dict[str, CheckResult]) -> str:
        """生成详细结果部分"""
        html = '<div class="issues-section"><h2>📋 详细检查结果</h2>'

        for checker_name, result in results.items():
            issues = result.issues
            duration = result.get_duration()

            html += f"""
            <h3>🔍 {checker_name}</h3>
            <p>耗时: {duration:.2f}秒 | 发现问题: {len(issues)}个</p>
            """

            if issues:
                html += '<table class="issues-table">'
                html += '<thead><tr><th>文件</th><th>行号</th><th>严重程度</th><th>规则</th><th>描述</th></tr></thead><tbody>'

                for issue in issues[:50]:  # 限制显示数量
                    severity_class = f"severity-{issue.severity.value}"
                    line_info = issue.line_number or '-'

                    html += f"""
                    <tr>
                        <td>{issue.file_path}</td>
                        <td>{line_info}</td>
                        <td class="{severity_class}">{issue.severity.value.upper()}</td>
                        <td>{issue.rule_id}</td>
                        <td>{issue.message}</td>
                    </tr>
                    """

                html += '</tbody></table>'

                if len(issues) > 50:
                    html += f'<p>... 还有 {len(issues) - 50} 个问题未显示</p>'

        html += '</div>'
        return html

    def _write_to_file(self, html_content: str) -> None:
        """写入HTML到文件"""
        try:
            output_path = Path(self.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            print(f"HTML报告已保存到: {self.output_file}")

        except Exception as e:
            print(f"保存HTML报告失败: {e}")
