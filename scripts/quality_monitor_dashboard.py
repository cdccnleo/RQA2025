#!/usr/bin/env python3
"""
RQA2025 质量监控仪表板
实时监控测试质量指标和趋势分析
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import argparse
import logging


class QualityMonitorDashboard:
    """质量监控仪表板"""

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or os.getcwd())
        self.test_logs_dir = self.project_root / "test_logs"
        self.dashboard_dir = self.test_logs_dir / "dashboard"
        self.metrics_history_file = self.dashboard_dir / "metrics_history.json"

        # 创建目录
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)

        # 设置日志
        self._setup_logging()

        # 指标历史
        self.metrics_history = self._load_metrics_history()

    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('QualityDashboard')

    def _load_metrics_history(self) -> List[Dict[str, Any]]:
        """加载指标历史"""
        if self.metrics_history_file.exists():
            try:
                with open(self.metrics_history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"无法加载指标历史: {e}")

        return []

    def generate_dashboard_report(self) -> Dict[str, Any]:
        """生成仪表板报告"""
        self.logger.info("📊 生成质量监控仪表板报告")

        # 收集当前指标
        current_metrics = self._collect_current_metrics()

        # 更新历史记录
        self._update_metrics_history(current_metrics)

        # 生成趋势分析
        trend_analysis = self._analyze_trends()

        # 生成质量评分
        quality_score = self._calculate_quality_score(current_metrics)

        # 生成可视化图表
        charts_data = self._generate_charts()

        # 生成建议
        recommendations = self._generate_recommendations(current_metrics, trend_analysis)

        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'current_metrics': current_metrics,
            'trend_analysis': trend_analysis,
            'quality_score': quality_score,
            'charts': charts_data,
            'recommendations': recommendations,
            'alerts': self._check_alerts(current_metrics, trend_analysis)
        }

        # 保存仪表板数据
        self._save_dashboard_data(dashboard_data)

        # 生成HTML报告
        self._generate_html_dashboard(dashboard_data)

        self.logger.info("✅ 质量监控仪表板生成完成")

        return dashboard_data

    def _collect_current_metrics(self) -> Dict[str, Any]:
        """收集当前质量指标"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'test_coverage': {},
            'test_results': {},
            'code_quality': {},
            'performance': {},
            'security': {}
        }

        try:
            # 收集覆盖率数据
            coverage_files = list(self.test_logs_dir.glob("*coverage*.json"))
            if coverage_files:
                for coverage_file in coverage_files[-3:]:  # 最近3个覆盖率文件
                    try:
                        with open(coverage_file, 'r', encoding='utf-8') as f:
                            coverage_data = json.load(f)
                            if 'totals' in coverage_data:
                                metrics['test_coverage'][coverage_file.name] = coverage_data['totals']
                    except Exception as e:
                        self.logger.warning(f"无法读取覆盖率文件 {coverage_file}: {e}")

            # 收集测试结果
            test_result_files = list(self.test_logs_dir.glob("*test_results*.json"))
            if test_result_files:
                for result_file in test_result_files[-3:]:  # 最近3个测试结果文件
                    try:
                        with open(result_file, 'r', encoding='utf-8') as f:
                            test_data = json.load(f)
                            metrics['test_results'][result_file.name] = test_data
                    except Exception as e:
                        self.logger.warning(f"无法读取测试结果文件 {result_file}: {e}")

            # 代码质量指标（模拟）
            metrics['code_quality'] = {
                'complexity_score': 8.5,
                'style_score': 8.2,
                'documentation_score': 7.8,
                'maintainability_index': 75.3
            }

            # 性能指标（模拟）
            metrics['performance'] = {
                'avg_response_time': 145,
                'throughput': 1250,
                'memory_usage': 82,
                'cpu_utilization': 68
            }

            # 安全指标（模拟）
            metrics['security'] = {
                'vulnerabilities_found': 3,
                'high_severity': 0,
                'medium_severity': 2,
                'low_severity': 1,
                'security_score': 8.7
            }

        except Exception as e:
            self.logger.error(f"收集指标时出错: {e}")

        return metrics

    def _update_metrics_history(self, current_metrics: Dict[str, Any]):
        """更新指标历史"""
        # 添加当前指标到历史
        self.metrics_history.append(current_metrics)

        # 保留最近30天的历史
        cutoff_date = datetime.now() - timedelta(days=30)
        self.metrics_history = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m['timestamp']) > cutoff_date
        ]

        # 保存历史
        with open(self.metrics_history_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, indent=2, ensure_ascii=False)

    def _analyze_trends(self) -> Dict[str, Any]:
        """分析趋势"""
        if len(self.metrics_history) < 2:
            return {'trend_available': False, 'message': '历史数据不足，无法分析趋势'}

        trends = {
            'trend_available': True,
            'period_days': len(self.metrics_history),
            'coverage_trend': self._calculate_coverage_trend(),
            'quality_trend': self._calculate_quality_trend(),
            'performance_trend': self._calculate_performance_trend(),
            'stability_score': self._calculate_stability_score()
        }

        return trends

    def _calculate_coverage_trend(self) -> Dict[str, Any]:
        """计算覆盖率趋势"""
        coverage_values = []
        for metrics in self.metrics_history:
            for coverage_data in metrics.get('test_coverage', {}).values():
                if isinstance(coverage_data, dict) and 'percent_covered' in coverage_data:
                    coverage_values.append(coverage_data['percent_covered'])

        if not coverage_values:
            return {'available': False}

        return {
            'available': True,
            'current': coverage_values[-1] if coverage_values else 0,
            'average': sum(coverage_values) / len(coverage_values),
            'trend': 'improving' if len(coverage_values) >= 2 and coverage_values[-1] > coverage_values[0] else 'stable',
            'volatility': self._calculate_volatility(coverage_values)
        }

    def _calculate_quality_trend(self) -> Dict[str, Any]:
        """计算质量趋势"""
        quality_scores = []
        for metrics in self.metrics_history:
            quality_data = metrics.get('code_quality', {})
            if quality_data:
                # 计算综合质量分数
                score = (
                    quality_data.get('complexity_score', 0) * 0.3 +
                    quality_data.get('style_score', 0) * 0.2 +
                    quality_data.get('documentation_score', 0) * 0.2 +
                    quality_data.get('maintainability_index', 0) * 0.3
                ) / 10
                quality_scores.append(score)

        if not quality_scores:
            return {'available': False}

        return {
            'available': True,
            'current': quality_scores[-1] if quality_scores else 0,
            'average': sum(quality_scores) / len(quality_scores),
            'trend': 'improving' if len(quality_scores) >= 2 and quality_scores[-1] > quality_scores[0] else 'stable'
        }

    def _calculate_performance_trend(self) -> Dict[str, Any]:
        """计算性能趋势"""
        response_times = []
        for metrics in self.metrics_history:
            perf_data = metrics.get('performance', {})
            if perf_data and 'avg_response_time' in perf_data:
                response_times.append(perf_data['avg_response_time'])

        if not response_times:
            return {'available': False}

        return {
            'available': True,
            'current': response_times[-1] if response_times else 0,
            'average': sum(response_times) / len(response_times),
            'trend': 'improving' if len(response_times) >= 2 and response_times[-1] < response_times[0] else 'stable'
        }

    def _calculate_stability_score(self) -> float:
        """计算稳定性分数"""
        if len(self.metrics_history) < 3:
            return 0.5

        # 计算各项指标的波动性
        stability_factors = []

        # 覆盖率稳定性
        coverage_trend = self._calculate_coverage_trend()
        if coverage_trend.get('available'):
            stability_factors.append(1.0 - min(1.0, coverage_trend.get('volatility', 0) / 10))

        # 质量稳定性
        quality_trend = self._calculate_quality_trend()
        if quality_trend.get('available'):
            stability_factors.append(0.8)  # 假设质量相对稳定

        # 性能稳定性
        perf_trend = self._calculate_performance_trend()
        if perf_trend.get('available'):
            stability_factors.append(0.9)  # 假设性能相对稳定

        return sum(stability_factors) / len(stability_factors) if stability_factors else 0.5

    def _calculate_volatility(self, values: List[float]) -> float:
        """计算波动性"""
        if len(values) < 2:
            return 0.0

        diffs = [abs(values[i] - values[i-1]) for i in range(1, len(values))]
        return sum(diffs) / len(diffs)

    def _calculate_quality_score(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """计算综合质量分数"""
        score_components = {
            'test_coverage': 0.25,
            'test_success': 0.25,
            'code_quality': 0.20,
            'performance': 0.15,
            'security': 0.15
        }

        component_scores = {}

        # 测试覆盖率分数
        coverage_data = metrics.get('test_coverage', {})
        if coverage_data:
            coverage_values = []
            for data in coverage_data.values():
                if isinstance(data, dict) and 'percent_covered' in data:
                    coverage_values.append(data['percent_covered'])
            if coverage_values:
                avg_coverage = sum(coverage_values) / len(coverage_values)
                component_scores['test_coverage'] = min(100, avg_coverage) / 100
        else:
            component_scores['test_coverage'] = 0.75  # 默认分数

        # 测试成功率分数
        test_results = metrics.get('test_results', {})
        if test_results:
            success_rates = []
            for data in test_results.values():
                if isinstance(data, dict):
                    summary = data.get('summary', {})
                    if 'passed' in summary and 'total' in summary and summary['total'] > 0:
                        success_rates.append(summary['passed'] / summary['total'])
            if success_rates:
                avg_success = sum(success_rates) / len(success_rates)
                component_scores['test_success'] = avg_success
        else:
            component_scores['test_success'] = 0.95  # 默认分数

        # 代码质量分数
        quality_data = metrics.get('code_quality', {})
        if quality_data:
            quality_score = (
                quality_data.get('complexity_score', 8.0) +
                quality_data.get('style_score', 8.0) +
                quality_data.get('documentation_score', 7.5) +
                quality_data.get('maintainability_index', 70.0) / 10
            ) / 4
            component_scores['code_quality'] = quality_score / 10
        else:
            component_scores['code_quality'] = 0.8

        # 性能分数
        perf_data = metrics.get('performance', {})
        if perf_data:
            # 基于响应时间和资源使用计算性能分数
            response_score = max(0, 1 - (perf_data.get('avg_response_time', 150) - 100) / 200)
            resource_score = max(0, 1 - (perf_data.get('memory_usage', 80) - 50) / 50)
            component_scores['performance'] = (response_score + resource_score) / 2
        else:
            component_scores['performance'] = 0.85

        # 安全分数
        security_data = metrics.get('security', {})
        if security_data:
            security_score = security_data.get('security_score', 8.5) / 10
            component_scores['security'] = security_score
        else:
            component_scores['security'] = 0.85

        # 计算综合分数
        overall_score = sum(
            score * weight for score, weight in zip(
                component_scores.values(), score_components.values()
            )
        )

        return {
            'overall_score': round(overall_score, 3),
            'component_scores': {k: round(v, 3) for k, v in component_scores.items()},
            'score_weights': score_components,
            'grade': self._get_quality_grade(overall_score)
        }

    def _get_quality_grade(self, score: float) -> str:
        """获取质量等级"""
        if score >= 0.9:
            return 'A+'
        elif score >= 0.8:
            return 'A'
        elif score >= 0.7:
            return 'B+'
        elif score >= 0.6:
            return 'B'
        elif score >= 0.5:
            return 'C+'
        else:
            return 'C'

    def _generate_charts(self) -> Dict[str, Any]:
        """生成图表数据"""
        charts = {}

        if len(self.metrics_history) >= 2:
            # 覆盖率趋势图
            charts['coverage_trend'] = self._create_trend_chart('test_coverage', 'percent_covered', '覆盖率趋势')

            # 质量趋势图
            charts['quality_trend'] = self._create_quality_trend_chart()

            # 性能趋势图
            charts['performance_trend'] = self._create_trend_chart('performance', 'avg_response_time', '响应时间趋势')

        return charts

    def _create_trend_chart(self, metric_category: str, metric_key: str, title: str) -> Dict[str, Any]:
        """创建趋势图表"""
        timestamps = []
        values = []

        for metrics in self.metrics_history:
            timestamp = datetime.fromisoformat(metrics['timestamp'])
            timestamps.append(timestamp)

            category_data = metrics.get(metric_category, {})
            if isinstance(category_data, dict):
                # 从第一个数据源获取指标
                first_key = next(iter(category_data.keys()), None)
                if first_key and isinstance(category_data[first_key], dict):
                    value = category_data[first_key].get(metric_key, 0)
                    values.append(value)
                else:
                    values.append(0)
            else:
                values.append(0)

        return {
            'title': title,
            'timestamps': [t.isoformat() for t in timestamps],
            'values': values,
            'type': 'line'
        }

    def _create_quality_trend_chart(self) -> Dict[str, Any]:
        """创建质量趋势图表"""
        timestamps = []
        quality_scores = []

        for metrics in self.metrics_history:
            timestamp = datetime.fromisoformat(metrics['timestamp'])
            timestamps.append(timestamp)

            quality_data = metrics.get('code_quality', {})
            if quality_data:
                score = (
                    quality_data.get('complexity_score', 0) * 0.3 +
                    quality_data.get('style_score', 0) * 0.2 +
                    quality_data.get('documentation_score', 0) * 0.2 +
                    quality_data.get('maintainability_index', 0) * 0.3
                ) / 10
                quality_scores.append(score)
            else:
                quality_scores.append(0)

        return {
            'title': '代码质量趋势',
            'timestamps': [t.isoformat() for t in timestamps],
            'values': quality_scores,
            'type': 'line'
        }

    def _generate_recommendations(self, current_metrics: Dict[str, Any], trend_analysis: Dict[str, Any]) -> List[str]:
        """生成建议"""
        recommendations = []

        quality_score = self._calculate_quality_score(current_metrics)
        overall_score = quality_score['overall_score']

        # 基于分数生成建议
        if overall_score < 0.7:
            recommendations.append("🚨 整体质量分数偏低，建议优先提升测试覆盖率和代码质量")

        if quality_score['component_scores'].get('test_coverage', 0) < 0.75:
            recommendations.append("📈 测试覆盖率不足75%，建议增加单元测试和集成测试")

        if quality_score['component_scores'].get('test_success', 0) < 0.95:
            recommendations.append("🔧 测试成功率低于95%，建议修复失败的测试用例")

        if quality_score['component_scores'].get('code_quality', 0) < 0.8:
            recommendations.append("💻 代码质量分数偏低，建议进行代码重构和质量改进")

        if trend_analysis.get('trend_available'):
            if trend_analysis.get('coverage_trend', {}).get('trend') == 'stable':
                recommendations.append("📊 覆盖率趋势稳定，建议持续增加新的测试场景")

        if not recommendations:
            recommendations.append("✅ 所有质量指标均良好，建议继续保持高质量开发实践")

        return recommendations

    def _check_alerts(self, current_metrics: Dict[str, Any], trend_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检查告警"""
        alerts = []

        # 覆盖率告警
        coverage_score = self._calculate_quality_score(current_metrics)['component_scores'].get('test_coverage', 0)
        if coverage_score < 0.7:
            alerts.append({
                'level': 'critical',
                'message': f'测试覆盖率严重不足: {coverage_score:.1%}',
                'action': '立即增加测试用例'
            })

        # 测试失败告警
        test_success = self._calculate_quality_score(current_metrics)['component_scores'].get('test_success', 0)
        if test_success < 0.9:
            alerts.append({
                'level': 'high',
                'message': f'测试成功率偏低: {test_success:.1%}',
                'action': '修复失败的测试用例'
            })

        # 性能退化告警
        if trend_analysis.get('trend_available'):
            perf_trend = trend_analysis.get('performance_trend', {})
            if perf_trend.get('trend') == 'worsening':
                alerts.append({
                    'level': 'medium',
                    'message': '性能指标出现退化趋势',
                    'action': '优化性能瓶颈'
                })

        return alerts

    def _save_dashboard_data(self, dashboard_data: Dict[str, Any]):
        """保存仪表板数据"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dashboard_file = self.dashboard_dir / f"quality_dashboard_{timestamp}.json"

        with open(dashboard_file, 'w', encoding='utf-8') as f:
            json.dump(dashboard_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"📄 仪表板数据已保存: {dashboard_file}")

    def _generate_html_dashboard(self, dashboard_data: Dict[str, Any]):
        """生成HTML仪表板"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RQA2025 质量监控仪表板 - {timestamp}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .metric-card {{ background: white; padding: 25px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center; }}
        .metric-value {{ font-size: 3em; font-weight: bold; margin: 10px 0; }}
        .metric-grade-a-plus {{ color: #28a745; }}
        .metric-grade-a {{ color: #20c997; }}
        .metric-grade-b-plus {{ color: #ffc107; }}
        .metric-grade-b {{ color: #fd7e14; }}
        .metric-grade-c-plus {{ color: #dc3545; }}
        .metric-grade-c {{ color: #6c757d; }}
        .metric-label {{ color: #666; font-size: 0.9em; }}
        .charts-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .chart-card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .recommendations {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        .recommendation {{ background: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 4px solid #007bff; }}
        .alerts {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .alert-critical {{ border-left: 4px solid #dc3545; background: #f8d7da; }}
        .alert-high {{ border-left: 4px solid #fd7e14; background: #fff3cd; }}
        .alert-medium {{ border-left: 4px solid #ffc107; background: #fff3cd; }}
        .component-scores {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        .score-bar {{ height: 20px; background: #e9ecef; border-radius: 10px; margin: 5px 0; }}
        .score-fill {{ height: 100%; border-radius: 10px; transition: width 0.3s ease; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>RQA2025 质量监控仪表板</h1>
            <p>生成时间: {timestamp}</p>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value metric-grade-{dashboard_data['quality_score']['grade'].lower().replace('+', '-plus')}">
                    {dashboard_data['quality_score']['overall_score']:.2f}
                </div>
                <div class="metric-label">综合质量分数</div>
                <div style="font-size: 1.2em; margin-top: 5px;">等级: {dashboard_data['quality_score']['grade']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{dashboard_data['quality_score']['component_scores'].get('test_coverage', 0):.1%}</div>
                <div class="metric-label">测试覆盖率</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{dashboard_data['quality_score']['component_scores'].get('test_success', 0):.1%}</div>
                <div class="metric-label">测试成功率</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{dashboard_data['trend_analysis'].get('stability_score', 0):.2f}</div>
                <div class="metric-label">稳定性分数</div>
            </div>
        </div>

        <div class="component-scores">
            <h2>质量分数组成</h2>
            {"".join(f'''
            <div>
                <strong>{component.replace('_', ' ').title()}</strong>
                <div class="score-bar">
                    <div class="score-fill" style="width: {score*100}%; background-color: {'#28a745' if score >= 0.8 else '#ffc107' if score >= 0.6 else '#dc3545'}"></div>
                </div>
                <span>{score:.1%}</span>
            </div>
            ''' for component, score in dashboard_data['quality_score']['component_scores'].items())}
        </div>

        <div class="recommendations">
            <h2>💡 质量改进建议</h2>
            {"".join(f'<div class="recommendation">{rec}</div>' for rec in dashboard_data.get('recommendations', []))}
        </div>

        <div class="alerts">
            <h2>🚨 质量告警</h2>
            {"".join(f'''
            <div class="recommendation alert-{alert['level']}">
                <strong>{alert['level'].upper()}</strong>: {alert['message']}
                <br><small>建议行动: {alert['action']}</small>
            </div>
            ''' for alert in dashboard_data.get('alerts', [])) if dashboard_data.get('alerts') else '<p>✅ 无活跃告警</p>'}
        </div>

        <div class="charts-grid">
            <div class="chart-card">
                <h3>趋势分析</h3>
                <canvas id="trendChart" width="400" height="200"></canvas>
            </div>
            <div class="chart-card">
                <h3>质量分布</h3>
                <canvas id="qualityChart" width="400" height="200"></canvas>
            </div>
        </div>

        <details>
            <summary>📊 详细数据 (点击展开)</summary>
            <pre>{json.dumps(dashboard_data, indent=2, ensure_ascii=False)}</pre>
        </details>
    </div>

    <script>
        // 趋势图表
        const trendCtx = document.getElementById('trendChart').getContext('2d');
        new Chart(trendCtx, {{
            type: 'line',
            data: {{
                labels: {list(range(len(dashboard_data.get('trend_analysis', {}).get('coverage_trend', {}).get('values', []))))},
                datasets: [{{
                    label: '覆盖率趋势',
                    data: {dashboard_data.get('trend_analysis', {}).get('coverage_trend', {}).get('values', [])},
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: '质量指标趋势'
                    }}
                }}
            }}
        }});

        // 质量分布图表
        const qualityCtx = document.getElementById('qualityChart').getContext('2d');
        new Chart(qualityCtx, {{
            type: 'doughnut',
            data: {{
                labels: Object.keys({json.dumps(dashboard_data['quality_score']['component_scores'])}),
                datasets: [{{
                    data: Object.values({json.dumps(dashboard_data['quality_score']['component_scores'])}),
                    backgroundColor: [
                        'rgb(255, 99, 132)',
                        'rgb(54, 162, 235)',
                        'rgb(255, 205, 86)',
                        'rgb(75, 192, 192)',
                        'rgb(153, 102, 255)'
                    ]
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: '质量分数组成'
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
        """

        html_file = self.dashboard_dir / f"quality_dashboard_{timestamp}.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        self.logger.info(f"📄 HTML仪表板已生成: {html_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RQA2025 质量监控仪表板')
    parser.add_argument('--project-root', help='项目根目录', default=None)
    parser.add_argument('--generate-dashboard', action='store_true', help='生成质量监控仪表板')
    parser.add_argument('--show-history', action='store_true', help='显示指标历史')
    parser.add_argument('--export-data', help='导出仪表板数据到指定文件')

    args = parser.parse_args()

    dashboard = QualityMonitorDashboard(args.project_root)

    if args.generate_dashboard:
        dashboard_data = dashboard.generate_dashboard_report()
        print("🎯 质量监控仪表板生成完成!")
        print(f"📊 综合质量分数: {dashboard_data['quality_score']['overall_score']:.2f} ({dashboard_data['quality_score']['grade']})")
        print(f"📈 稳定性分数: {dashboard_data.get('trend_analysis', {}).get('stability_score', 0):.2f}")

        if dashboard_data.get('recommendations'):
            print("\n💡 改进建议:")
            for rec in dashboard_data['recommendations']:
                print(f"  • {rec}")

        if dashboard_data.get('alerts'):
            print("\n🚨 活跃告警:")
            for alert in dashboard_data['alerts']:
                print(f"  • {alert['level'].upper()}: {alert['message']}")

    elif args.show_history:
        history = dashboard.metrics_history
        print(f"📈 指标历史记录数: {len(history)}")
        if history:
            latest = history[-1]
            print(f"📅 最新记录时间: {latest['timestamp']}")
            print(f"🎯 最新质量分数: {dashboard._calculate_quality_score(latest)['overall_score']:.2f}")

    elif args.export_data:
        dashboard_data = dashboard.generate_dashboard_report()
        with open(args.export_data, 'w', encoding='utf-8') as f:
            json.dump(dashboard_data, f, indent=2, ensure_ascii=False)
        print(f"✅ 仪表板数据已导出到: {args.export_data}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
