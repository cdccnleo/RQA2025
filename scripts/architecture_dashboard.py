#!/usr/bin/env python3
"""
架构质量仪表板

生成交互式架构质量报告和可视化仪表板
"""

import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path
from datetime import datetime


class ArchitectureDashboard:
    """架构质量仪表板"""

    def __init__(self, report_path: str):
        self.report_path = Path(report_path)
        self.output_dir = self.report_path.parent / "dashboard"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(self.report_path, 'r', encoding='utf-8') as f:
            self.report_data = json.load(f)

    def generate_dashboard(self):
        """生成完整的仪表板"""
        print("📊 生成架构质量仪表板...")

        # 生成各个图表
        self._generate_overview_chart()
        self._generate_layer_analysis_chart()
        self._generate_dependency_chart()
        self._generate_quality_metrics_chart()
        self._generate_issues_chart()
        self._generate_trends_chart()

        # 生成HTML报告
        self._generate_html_report()

        print(f"✅ 仪表板已生成: {self.output_dir / 'architecture_dashboard.html'}")

    def _generate_overview_chart(self):
        """生成概览图表"""
        metrics = self.report_data['metrics']

        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('代码规模', '复杂性分布', '测试覆盖率', '模块数量', '可维护性', '质量指标'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]]
        )

        # 代码规模
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=metrics['total_files'],
                title="总文件数",
                domain={'x': [0, 0.33], 'y': [0.5, 1]}
            ),
            row=1, col=1
        )

        # 复杂性
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=metrics['average_complexity'],
                title="平均复杂度",
                delta={'reference': 20, 'relative': True},
                domain={'x': [0.33, 0.66], 'y': [0.5, 1]}
            ),
            row=1, col=2
        )

        # 测试覆盖率
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=metrics.get('test_coverage', 0) * 100,
                title="测试覆盖率",
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "darkblue"}},
                domain={'x': [0.66, 1], 'y': [0.5, 1]}
            ),
            row=1, col=3
        )

        # 模块数量
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=metrics['total_modules'],
                title="模块数量",
                domain={'x': [0, 0.33], 'y': [0, 0.5]}
            ),
            row=2, col=1
        )

        # 可维护性
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=metrics.get('maintainability_index', 0),
                title="可维护性指数",
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "green"}},
                domain={'x': [0.33, 0.66], 'y': [0, 0.5]}
            ),
            row=2, col=2
        )

        # 质量综合指标
        quality_score = self._calculate_quality_score()
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=quality_score,
                title="综合质量评分",
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "orange"}},
                domain={'x': [0.66, 1], 'y': [0, 0.5]}
            ),
            row=2, col=3
        )

        fig.update_layout(title="架构质量概览")
        fig.write_html(self.output_dir / "overview.html")

    def _generate_layer_analysis_chart(self):
        """生成层次分析图表"""
        layers = self.report_data['layers']

        if not layers:
            return

        # 准备数据
        layer_names = []
        component_counts = []
        complexity_scores = []
        test_coverages = []

        for layer_key, layer_data in layers.items():
            layer_names.append(layer_data['layer_name'])
            component_counts.append(layer_data['component_count'])
            complexity_scores.append(layer_data['complexity_score'])
            test_coverages.append(layer_data.get('test_coverage', 0))

        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('组件数量分布', '复杂性对比', '测试覆盖率', '综合评分'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )

        # 组件数量
        fig.add_trace(
            go.Bar(x=layer_names, y=component_counts, name="组件数量"),
            row=1, col=1
        )

        # 复杂性对比
        fig.add_trace(
            go.Bar(x=layer_names, y=complexity_scores, name="复杂性", marker_color='red'),
            row=1, col=2
        )

        # 测试覆盖率
        fig.add_trace(
            go.Scatter(x=layer_names, y=[c*100 for c in test_coverages],
                       mode='lines+markers', name="测试覆盖率(%)", line=dict(color='green')),
            row=2, col=1
        )

        # 综合评分
        composite_scores = []
        for i, layer in enumerate(layer_names):
            score = (component_counts[i] * 10 +
                     (100 - complexity_scores[i]) + test_coverages[i] * 50) / 3
            composite_scores.append(min(score, 100))

        fig.add_trace(
            go.Bar(x=layer_names, y=composite_scores, name="综合评分", marker_color='purple'),
            row=2, col=2
        )

        fig.update_layout(title="架构层次分析", showlegend=False)
        fig.write_html(self.output_dir / "layer_analysis.html")

    def _generate_dependency_chart(self):
        """生成依赖关系图表"""
        dependencies = self.report_data.get('dependencies', {})

        if not dependencies:
            return

        # 创建依赖关系网络图
        modules = list(dependencies.get('centrality', {}).keys())
        centrality_values = list(dependencies.get('centrality', {}).values())

        fig = go.Figure(data=[
            go.Bar(x=modules, y=centrality_values, marker_color='lightblue')
        ])

        fig.update_layout(
            title="模块依赖中心性分析",
            xaxis_title="模块",
            yaxis_title="中心性"
        )

        fig.write_html(self.output_dir / "dependencies.html")

    def _generate_quality_metrics_chart(self):
        """生成质量指标图表"""
        quality = self.report_data.get('quality', {})

        # 准备数据
        categories = []
        values = []

        if 'complexity_analysis' in quality:
            categories.append('复杂度')
            values.append(quality['complexity_analysis'].get('average_complexity', 0))

        if 'coupling_analysis' in quality:
            categories.append('耦合度')
            values.append(quality['coupling_analysis'].get('coupling_score', 0))

        if 'cohesion_analysis' in quality:
            categories.append('内聚度')
            values.append(quality['cohesion_analysis'].get('cohesion_score', 0))

        if categories:
            fig = go.Figure(data=[
                go.Scatterpolar(r=values, theta=categories, fill='toself')
            ])

            fig.update_layout(
                title="架构质量雷达图",
                polar=dict(radialaxis=dict(visible=True, range=[0, max(values) if values else 10]))
            )

            fig.write_html(self.output_dir / "quality_metrics.html")

    def _generate_issues_chart(self):
        """生成问题分析图表"""
        issues = self.report_data.get('issues', [])

        if not issues:
            return

        # 统计问题类型
        severity_count = {}
        category_count = {}

        for issue in issues:
            severity = issue.get('severity', 'unknown')
            category = issue.get('category', 'unknown')

            severity_count[severity] = severity_count.get(severity, 0) + 1
            category_count[category] = category_count.get(category, 0) + 1

        # 创建饼图
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=('问题严重性分布', '问题类别分布'),
                            specs=[[{"type": "domain"}, {"type": "domain"}]])

        fig.add_trace(
            go.Pie(labels=list(severity_count.keys()), values=list(severity_count.values())),
            row=1, col=1
        )

        fig.add_trace(
            go.Pie(labels=list(category_count.keys()), values=list(category_count.values())),
            row=1, col=2
        )

        fig.update_layout(title="架构问题分析")
        fig.write_html(self.output_dir / "issues.html")

    def _generate_trends_chart(self):
        """生成趋势分析图表"""
        # 这里可以加载历史数据进行趋势分析
        # 目前创建一个示例趋势图

        dates = pd.date_range(start='2025-01-01', periods=10, freq='D')
        maintainability_trend = [50 + i*2 for i in range(10)]
        complexity_trend = [25 - i*0.5 for i in range(10)]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(x=dates, y=maintainability_trend, mode='lines+markers', name='可维护性指数')
        )

        fig.add_trace(
            go.Scatter(x=dates, y=complexity_trend, mode='lines+markers', name='平均复杂度')
        )

        fig.update_layout(
            title="架构质量趋势分析",
            xaxis_title="日期",
            yaxis_title="指标值"
        )

        fig.write_html(self.output_dir / "trends.html")

    def _generate_html_report(self):
        """生成HTML综合报告"""
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RQA2025 架构质量仪表板</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            height: 500px;
            overflow: hidden;
        }}
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-label {{
            color: #666;
            margin-top: 10px;
        }}
        .status-good {{ color: #28a745; }}
        .status-warning {{ color: #ffc107; }}
        .status-danger {{ color: #dc3545; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🏗️ RQA2025 架构质量仪表板</h1>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>架构健康度: {self._get_health_status()}</p>
    </div>

    <div class="summary-cards">
        {self._generate_summary_cards()}
    </div>

    <div class="dashboard-grid">
        <div class="chart-container">
            <h3>📊 架构质量概览</h3>
            <iframe src="overview.html" width="100%" height="450" frameborder="0"></iframe>
        </div>

        <div class="chart-container">
            <h3>🏗️ 层次分析</h3>
            <iframe src="layer_analysis.html" width="100%" height="450" frameborder="0"></iframe>
        </div>

        <div class="chart-container">
            <h3>🔗 依赖关系</h3>
            <iframe src="dependencies.html" width="100%" height="450" frameborder="0"></iframe>
        </div>

        <div class="chart-container">
            <h3>⚖️ 质量指标</h3>
            <iframe src="quality_metrics.html" width="100%" height="450" frameborder="0"></iframe>
        </div>

        <div class="chart-container">
            <h3>🚨 问题分析</h3>
            <iframe src="issues.html" width="100%" height="450" frameborder="0"></iframe>
        </div>

        <div class="chart-container">
            <h3>📈 趋势分析</h3>
            <iframe src="trends.html" width="100%" height="450" frameborder="0"></iframe>
        </div>
    </div>

    <div class="dashboard-grid">
        <div class="chart-container">
            <h3>💡 改进建议</h3>
            {self._generate_recommendations_html()}
        </div>

        <div class="chart-container">
            <h3>🎯 架构模式识别</h3>
            {self._generate_patterns_html()}
        </div>
    </div>
</body>
</html>
        """

        with open(self.output_dir / "architecture_dashboard.html", 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _calculate_quality_score(self) -> float:
        """计算综合质量评分"""
        metrics = self.report_data['metrics']

        # 基于多个指标计算综合评分
        maintainability_score = metrics.get('maintainability_index', 50) / 100 * 30
        complexity_score = max(0, (50 - metrics.get('average_complexity', 25)) / 50 * 25)
        coverage_score = metrics.get('test_coverage', 0.5) * 25
        issues_penalty = max(0, 20 - len(self.report_data.get('issues', [])) * 2)

        return maintainability_score + complexity_score + coverage_score + issues_penalty

    def _get_health_status(self) -> str:
        """获取架构健康状态"""
        score = self._calculate_quality_score()

        if score >= 80:
            return "🟢 健康"
        elif score >= 60:
            return "🟡 需要关注"
        else:
            return "🔴 需要改进"

    def _generate_summary_cards(self) -> str:
        """生成摘要卡片"""
        metrics = self.report_data['metrics']

        cards = [
            {
                "title": "总文件数",
                "value": metrics['total_files'],
                "label": "个Python文件"
            },
            {
                "title": "代码行数",
                "value": f"{metrics['total_lines']:,}",
                "label": "行代码"
            },
            {
                "title": "架构层次",
                "value": len(self.report_data.get('layers', {})),
                "label": "个层次"
            },
            {
                "title": "发现问题",
                "value": len(self.report_data.get('issues', [])),
                "label": "个架构问题"
            }
        ]

        html = ""
        for card in cards:
            html += f'''
            <div class="card">
                <div class="metric-value">{card["value"]}</div>
                <div class="metric-label">{card["title"]}</div>
                <div class="metric-label">{card["label"]}</div>
            </div>
            '''

        return html

    def _generate_recommendations_html(self) -> str:
        """生成改进建议HTML"""
        recommendations = self.report_data.get('recommendations', [])

        if not recommendations:
            return "<p>🎉 没有发现需要改进的领域！</p>"

        html = "<ul>"
        for rec in recommendations:
            priority_class = {
                "high": "status-danger",
                "medium": "status-warning",
                "low": "status-good"
            }.get(rec.get('priority', 'medium'), 'status-warning')

            html += f'''
            <li>
                <strong class="{priority_class}">[{rec.get('priority', 'medium').upper()}]</strong>
                <strong>{rec.get('title', '未命名建议')}</strong><br>
                {rec.get('description', '')}<br>
                <small><strong>建议行动:</strong> {', '.join(rec.get('actions', []))}</small>
            </li>
            '''
        html += "</ul>"
        return html

    def _generate_patterns_html(self) -> str:
        """生成架构模式HTML"""
        patterns = self.report_data.get('patterns', {})

        html = ""

        if patterns.get('design_patterns'):
            html += "<h4>设计模式</h4><ul>"
            for pattern in patterns['design_patterns']:
                html += f"<li>✅ {pattern}</li>"
            html += "</ul>"

        if patterns.get('architectural_patterns'):
            html += "<h4>架构模式</h4><ul>"
            for pattern in patterns['architectural_patterns']:
                html += f"<li>🏗️ {pattern}</li>"
            html += "</ul>"

        if patterns.get('anti_patterns'):
            html += "<h4>需要关注的反模式</h4><ul>"
            for pattern in patterns['anti_patterns']:
                html += f"<li>⚠️ {pattern}</li>"
            html += "</ul>"

        if not html:
            html = "<p>📝 正在分析架构模式...</p>"

        return html


def main():
    """主函数"""
    import sys

    if len(sys.argv) != 2:
        print("用法: python architecture_dashboard.py <报告文件路径>")
        sys.exit(1)

    report_path = sys.argv[1]
    dashboard = ArchitectureDashboard(report_path)
    dashboard.generate_dashboard()


if __name__ == "__main__":
    # 默认使用最近的报告文件
    project_root = Path(__file__).parent.parent
    report_path = project_root / "reports" / "architecture_validation_report.json"

    if report_path.exists():
        dashboard = ArchitectureDashboard(str(report_path))
        dashboard.generate_dashboard()
    else:
        print("❌ 未找到架构验证报告文件")
        print("请先运行: python scripts/architecture_validator.py")
