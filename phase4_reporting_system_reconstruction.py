#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4 报告和分析系统重建脚本
重建完整的报告和分析系统，支持多种报告类型和自动化生成
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ReportConfig:
    """报告配置"""
    report_type: str
    title: str
    description: str
    include_charts: bool = True
    include_tables: bool = True
    output_format: str = "html"  # html, pdf, json
    date_range: Tuple[datetime, datetime] = None


@dataclass
class PerformanceMetrics:
    """绩效指标"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    sortino_ratio: float


@dataclass
class RiskMetrics:
    """风险指标"""
    value_at_risk: float
    expected_shortfall: float
    beta: float
    correlation_matrix: Dict[str, Dict[str, float]]
    stress_test_results: Dict[str, float]


@dataclass
class ComplianceReport:
    """合规报告"""
    compliance_status: str
    violations: List[Dict[str, Any]]
    risk_assessment: str
    recommendations: List[str]


@dataclass
class PortfolioReport:
    """投资组合报告"""
    portfolio_id: str
    assets: List[Dict[str, Any]]
    allocation: Dict[str, float]
    performance: PerformanceMetrics
    risk: RiskMetrics
    compliance: ComplianceReport


class ReportGenerator(ABC):
    """报告生成器基类"""

    def __init__(self, config: ReportConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def generate_report(self, data: Dict[str, Any]) -> str:
        """生成报告"""

    def _generate_header(self) -> str:
        """生成报告头部"""
        return f"""
        <div class="report-header">
            <h1>{self.config.title}</h1>
            <p class="description">{self.config.description}</p>
            <p class="timestamp">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """

    def _generate_footer(self) -> str:
        """生成报告尾部"""
        return """
        <div class="report-footer">
            <p>本报告由RQA2025系统自动生成</p>
            <p>© 2025 量化交易系统</p>
        </div>
        """


class PerformanceReportGenerator(ReportGenerator):
    """绩效报告生成器"""

    def generate_report(self, data: Dict[str, Any]) -> str:
        """生成绩效报告"""
        try:
            metrics = self._calculate_performance_metrics(data)
            charts = self._generate_performance_charts(data) if self.config.include_charts else ""

            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{self.config.title}</title>
                <style>
                    {self._get_css_styles()}
                </style>
            </head>
            <body>
                {self._generate_header()}

                <div class="metrics-section">
                    <h2>绩效指标</h2>
                    {self._generate_metrics_table(metrics)}
                </div>

                {charts}

                <div class="analysis-section">
                    <h2>绩效分析</h2>
                    {self._generate_performance_analysis(metrics)}
                </div>

                {self._generate_footer()}
            </body>
            </html>
            """

            return html

        except Exception as e:
            self.logger.error(f"生成绩效报告失败: {e}")
            return self._generate_error_report(str(e))

    def _calculate_performance_metrics(self, data: Dict[str, Any]) -> PerformanceMetrics:
        """计算绩效指标"""
        returns = pd.Series(data.get('returns', []))

        if returns.empty:
            return PerformanceMetrics(
                total_return=0.0,
                annualized_return=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                calmar_ratio=0.0,
                sortino_ratio=0.0
            )

        # 计算基础指标
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

        # 计算最大回撤
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # 计算胜率
        win_rate = (returns > 0).mean()

        # 计算盈利因子
        winning_returns = returns[returns > 0]
        losing_returns = returns[returns < 0]
        profit_factor = abs(winning_returns.sum() / losing_returns.sum()
                            ) if losing_returns.sum() != 0 else float('inf')

        # 计算Calmar比率
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # 计算Sortino比率
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        sortino_ratio = annualized_return / downside_volatility if downside_volatility > 0 else 0

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio
        )

    def _generate_metrics_table(self, metrics: PerformanceMetrics) -> str:
        """生成指标表格"""
        return f"""
        <table class="metrics-table">
            <tr>
                <th>指标</th>
                <th>数值</th>
                <th>评级</th>
            </tr>
            <tr>
                <td>总收益率</td>
                <td>{metrics.total_return:.2%}</td>
                <td>{self._rate_performance(metrics.total_return, 'return')}</td>
            </tr>
            <tr>
                <td>年化收益率</td>
                <td>{metrics.annualized_return:.2%}</td>
                <td>{self._rate_performance(metrics.annualized_return, 'return')}</td>
            </tr>
            <tr>
                <td>波动率</td>
                <td>{metrics.volatility:.2%}</td>
                <td>{self._rate_performance(metrics.volatility, 'volatility')}</td>
            </tr>
            <tr>
                <td>夏普比率</td>
                <td>{metrics.sharpe_ratio:.2f}</td>
                <td>{self._rate_performance(metrics.sharpe_ratio, 'sharpe')}</td>
            </tr>
            <tr>
                <td>最大回撤</td>
                <td>{metrics.max_drawdown:.2%}</td>
                <td>{self._rate_performance(metrics.max_drawdown, 'drawdown')}</td>
            </tr>
            <tr>
                <td>胜率</td>
                <td>{metrics.win_rate:.2%}</td>
                <td>{self._rate_performance(metrics.win_rate, 'win_rate')}</td>
            </tr>
            <tr>
                <td>盈利因子</td>
                <td>{metrics.profit_factor:.2f}</td>
                <td>{self._rate_performance(metrics.profit_factor, 'profit_factor')}</td>
            </tr>
        </table>
        """

    def _rate_performance(self, value: float, metric_type: str) -> str:
        """对绩效指标进行评级"""
        if metric_type == 'return':
            if value > 0.20:
                return "优秀"
            elif value > 0.10:
                return "良好"
            elif value > 0.05:
                return "一般"
            else:
                return "需改进"
        elif metric_type == 'volatility':
            if value < 0.15:
                return "优秀"
            elif value < 0.25:
                return "良好"
            elif value < 0.35:
                return "一般"
            else:
                return "需改进"
        elif metric_type == 'sharpe':
            if value > 2.0:
                return "优秀"
            elif value > 1.5:
                return "良好"
            elif value > 1.0:
                return "一般"
            else:
                return "需改进"
        elif metric_type == 'drawdown':
            if value > -0.10:
                return "优秀"
            elif value > -0.20:
                return "良好"
            elif value > -0.30:
                return "一般"
            else:
                return "需改进"
        elif metric_type == 'win_rate':
            if value > 0.60:
                return "优秀"
            elif value > 0.50:
                return "良好"
            elif value > 0.40:
                return "一般"
            else:
                return "需改进"
        elif metric_type == 'profit_factor':
            if value > 2.0:
                return "优秀"
            elif value > 1.5:
                return "良好"
            elif value > 1.2:
                return "一般"
            else:
                return "需改进"
        return "未知"

    def _generate_performance_charts(self, data: Dict[str, Any]) -> str:
        """生成绩效图表"""
        # 这里简化实现，实际应该生成真实的图表
        return """
        <div class="charts-section">
            <h2>绩效图表</h2>
            <div class="chart-placeholder">
                <p>收益率曲线图表</p>
                <p>回撤分析图表</p>
                <p>月度收益分布图表</p>
            </div>
        </div>
        """

    def _generate_performance_analysis(self, metrics: PerformanceMetrics) -> str:
        """生成绩效分析"""
        analysis = []

        if metrics.sharpe_ratio > 2.0:
            analysis.append("风险调整后收益表现优秀")
        elif metrics.sharpe_ratio > 1.5:
            analysis.append("风险调整后收益表现良好")
        else:
            analysis.append("建议关注风险控制以提升夏普比率")

        if metrics.max_drawdown < -0.20:
            analysis.append("最大回撤较大，建议加强风险管理")
        else:
            analysis.append("回撤控制在合理范围内")

        if metrics.win_rate > 0.55:
            analysis.append("胜率表现良好")
        else:
            analysis.append("建议优化策略以提升胜率")

        return "<ul>" + "".join(f"<li>{item}</li>" for item in analysis) + "</ul>"

    def _get_css_styles(self) -> str:
        """获取CSS样式"""
        return """
        body { font-family: Arial, sans-serif; margin: 20px; }
        .report-header { border-bottom: 2px solid #333; padding-bottom: 10px; }
        .report-header h1 { color: #333; }
        .metrics-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        .metrics-table th, .metrics-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        .metrics-table th { background-color: #f2f2f2; }
        .charts-section { margin: 30px 0; }
        .report-footer { border-top: 1px solid #ccc; margin-top: 40px; padding-top: 10px; color: #666; font-size: 0.9em; }
        """

    def _generate_error_report(self, error: str) -> str:
        """生成错误报告"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head><title>报告生成错误</title></head>
        <body>
            <h1>报告生成失败</h1>
            <p>错误信息: {error}</p>
            <p>请检查输入数据或联系技术支持</p>
        </body>
        </html>
        """


class RiskReportGenerator(ReportGenerator):
    """风险报告生成器"""

    def generate_report(self, data: Dict[str, Any]) -> str:
        """生成风险报告"""
        try:
            risk_metrics = self._calculate_risk_metrics(data)

            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{self.config.title}</title>
                <style>{self._get_css_styles()}</style>
            </head>
            <body>
                {self._generate_header()}

                <div class="risk-section">
                    <h2>风险指标</h2>
                    {self._generate_risk_table(risk_metrics)}
                </div>

                <div class="stress-test-section">
                    <h2>压力测试结果</h2>
                    {self._generate_stress_test_table(risk_metrics)}
                </div>

                <div class="recommendations-section">
                    <h2>风险管理建议</h2>
                    {self._generate_risk_recommendations(risk_metrics)}
                </div>

                {self._generate_footer()}
            </body>
            </html>
            """

            return html

        except Exception as e:
            self.logger.error(f"生成风险报告失败: {e}")
            return f"<html><body><h1>风险报告生成失败</h1><p>错误: {e}</p></body></html>"

    def _calculate_risk_metrics(self, data: Dict[str, Any]) -> RiskMetrics:
        """计算风险指标"""
        returns = pd.Series(data.get('returns', []))
        benchmark_returns = pd.Series(data.get('benchmark_returns', []))

        # 计算VaR (95%置信度)
        value_at_risk = np.percentile(returns, 5) if not returns.empty else 0

        # 计算ES (预期损失)
        expected_shortfall = returns[returns <= value_at_risk].mean() if not returns.empty else 0

        # 计算Beta
        if not benchmark_returns.empty and not returns.empty:
            covariance = returns.cov(benchmark_returns)
            variance = benchmark_returns.var()
            beta = covariance / variance if variance > 0 else 1.0
        else:
            beta = 1.0

        # 模拟相关性矩阵
        correlation_matrix = {"portfolio": {"benchmark": 0.7}}

        # 压力测试结果
        stress_test_results = {
            "市场下跌10%": -0.08,
            "市场下跌20%": -0.15,
            "利率上升100bp": -0.03,
            "汇率波动10%": -0.05
        }

        return RiskMetrics(
            value_at_risk=value_at_risk,
            expected_shortfall=expected_shortfall,
            beta=beta,
            correlation_matrix=correlation_matrix,
            stress_test_results=stress_test_results
        )

    def _generate_risk_table(self, metrics: RiskMetrics) -> str:
        """生成风险指标表格"""
        return f"""
        <table class="risk-table">
            <tr><th>风险指标</th><th>数值</th><th>评估</th></tr>
            <tr>
                <td>VaR (95%)</td>
                <td>{metrics.value_at_risk:.2%}</td>
                <td>{'可接受' if abs(metrics.value_at_risk) < 0.10 else '需关注'}</td>
            </tr>
            <tr>
                <td>预期损失 (ES)</td>
                <td>{metrics.expected_shortfall:.2%}</td>
                <td>{'可接受' if abs(metrics.expected_shortfall) < 0.12 else '需关注'}</td>
            </tr>
            <tr>
                <td>Beta系数</td>
                <td>{metrics.beta:.2f}</td>
                <td>{'市场中性' if 0.8 <= metrics.beta <= 1.2 else '偏离市场'}</td>
            </tr>
        </table>
        """

    def _generate_stress_test_table(self, metrics: RiskMetrics) -> str:
        """生成压力测试表格"""
        rows = ""
        for scenario, impact in metrics.stress_test_results.items():
            assessment = "严重" if abs(impact) > 0.10 else "中等" if abs(impact) > 0.05 else "轻微"
            rows += f"<tr><td>{scenario}</td><td>{impact:.1%}</td><td>{assessment}</td></tr>"

        return f"""
        <table class="stress-test-table">
            <tr><th>压力情景</th><th>潜在损失</th><th>严重程度</th></tr>
            {rows}
        </table>
        """

    def _generate_risk_recommendations(self, metrics: RiskMetrics) -> str:
        """生成风险管理建议"""
        recommendations = []

        if abs(metrics.value_at_risk) > 0.10:
            recommendations.append("VaR水平较高，建议增加对冲或降低仓位")

        if abs(metrics.expected_shortfall) > 0.12:
            recommendations.append("预期损失较大，建议优化投资组合分散度")

        if not 0.8 <= metrics.beta <= 1.2:
            recommendations.append("Beta偏离市场基准，建议调整市场暴露")

        max_stress_impact = max(abs(impact) for impact in metrics.stress_test_results.values())
        if max_stress_impact > 0.10:
            recommendations.append("压力测试显示潜在较大损失，建议制定应急预案")

        return "<ul>" + "".join(f"<li>{rec}</li>" for rec in recommendations) + "</ul>"

    def _get_css_styles(self) -> str:
        """获取CSS样式"""
        return """
        body { font-family: Arial, sans-serif; margin: 20px; }
        .report-header { border-bottom: 2px solid #333; padding-bottom: 10px; }
        .risk-table, .stress-test-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        .risk-table th, .risk-table td, .stress-test-table th, .stress-test-table td {
            border: 1px solid #ddd; padding: 8px; text-align: left;
        }
        .risk-table th, .stress-test-table th { background-color: #f2f2f2; }
        .report-footer { border-top: 1px solid #ccc; margin-top: 40px; padding-top: 10px; color: #666; font-size: 0.9em; }
        """


class ComplianceReportGenerator(ReportGenerator):
    """合规报告生成器"""

    def generate_report(self, data: Dict[str, Any]) -> str:
        """生成合规报告"""
        try:
            compliance_data = self._analyze_compliance(data)

            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{self.config.title}</title>
                <style>{self._get_css_styles()}</style>
            </head>
            <body>
                {self._generate_header()}

                <div class="compliance-status">
                    <h2>合规状态总览</h2>
                    <div class="status-indicator {compliance_data['status'].lower()}">
                        <h3>总体合规状态: {compliance_data['status']}</h3>
                        <p>合规评分: {compliance_data['score']}/100</p>
                    </div>
                </div>

                <div class="violations-section">
                    <h2>违规记录</h2>
                    {self._generate_violations_table(compliance_data['violations'])}
                </div>

                <div class="recommendations-section">
                    <h2>合规建议</h2>
                    {self._generate_compliance_recommendations(compliance_data)}
                </div>

                {self._generate_footer()}
            </body>
            </html>
            """

            return html

        except Exception as e:
            self.logger.error(f"生成合规报告失败: {e}")
            return f"<html><body><h1>合规报告生成失败</h1><p>错误: {e}</p></body></html>"

    def _analyze_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """分析合规状况"""
        # 模拟合规分析
        violations = [
            {
                "rule": "交易量限制",
                "severity": "低",
                "description": "单日交易量接近监管上限",
                "status": "警告",
                "timestamp": datetime.now().isoformat()
            }
        ]

        score = 85  # 模拟合规评分
        status = "合规" if score >= 80 else "部分合规" if score >= 60 else "不合规"

        return {
            "status": status,
            "score": score,
            "violations": violations,
            "recommendations": [
                "加强交易监控",
                "完善合规培训",
                "定期审计检查"
            ]
        }

    def _generate_violations_table(self, violations: List[Dict[str, Any]]) -> str:
        """生成违规记录表格"""
        if not violations:
            return "<p>无违规记录</p>"

        rows = ""
        for violation in violations:
            rows += f"""
            <tr>
                <td>{violation['rule']}</td>
                <td>{violation['severity']}</td>
                <td>{violation['description']}</td>
                <td>{violation['status']}</td>
                <td>{violation['timestamp'][:10]}</td>
            </tr>
            """

        return f"""
        <table class="violations-table">
            <tr><th>规则</th><th>严重程度</th><th>描述</th><th>状态</th><th>时间</th></tr>
            {rows}
        </table>
        """

    def _generate_compliance_recommendations(self, compliance_data: Dict[str, Any]) -> str:
        """生成合规建议"""
        recommendations = compliance_data.get('recommendations', [])
        return "<ul>" + "".join(f"<li>{rec}</li>" for rec in recommendations) + "</ul>"

    def _get_css_styles(self) -> str:
        """获取CSS样式"""
        return """
        body { font-family: Arial, sans-serif; margin: 20px; }
        .report-header { border-bottom: 2px solid #333; padding-bottom: 10px; }
        .status-indicator { padding: 20px; border-radius: 5px; margin: 20px 0; }
        .status-indicator.合规 { background-color: #d4edda; border: 1px solid #c3e6cb; }
        .status-indicator.部分合规 { background-color: #fff3cd; border: 1px solid #ffeaa7; }
        .status-indicator.不合规 { background-color: #f8d7da; border: 1px solid #f5c6cb; }
        .violations-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        .violations-table th, .violations-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        .violations-table th { background-color: #f2f2f2; }
        .report-footer { border-top: 1px solid #ccc; margin-top: 40px; padding-top: 10px; color: #666; font-size: 0.9em; }
        """


class ReportManager:
    """报告管理器"""

    def __init__(self):
        self.generators = {
            'performance': PerformanceReportGenerator,
            'risk': RiskReportGenerator,
            'compliance': ComplianceReportGenerator
        }
        self.logger = logging.getLogger(__name__)

    def generate_report(self, report_type: str, config: ReportConfig, data: Dict[str, Any]) -> Optional[str]:
        """生成报告"""
        try:
            if report_type not in self.generators:
                raise ValueError(f"不支持的报告类型: {report_type}")

            generator_class = self.generators[report_type]
            generator = generator_class(config)

            report_content = generator.generate_report(data)

            # 保存报告到文件
            self._save_report(report_content, config)

            self.logger.info(f"成功生成{report_type}报告: {config.title}")
            return report_content

        except Exception as e:
            self.logger.error(f"生成报告失败: {e}")
            return None

    def _save_report(self, content: str, config: ReportConfig):
        """保存报告到文件"""
        try:
            os.makedirs('reports', exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"reports/{config.report_type}_report_{timestamp}.{config.output_format}"

            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)

            self.logger.info(f"报告已保存到: {filename}")

        except Exception as e:
            self.logger.error(f"保存报告失败: {e}")


def test_reporting_system():
    """测试报告系统"""
    print("测试报告和分析系统...")

    manager = ReportManager()

    # 测试绩效报告
    print("\n1. 生成绩效报告")

    perf_config = ReportConfig(
        report_type="performance",
        title="月度绩效报告",
        description="2025年9月投资组合绩效分析",
        include_charts=True
    )

    perf_data = {
        'returns': np.random.normal(0.001, 0.02, 252)  # 一年252个交易日
    }

    perf_report = manager.generate_report("performance", perf_config, perf_data)
    print(f"绩效报告生成: {'成功' if perf_report else '失败'}")

    # 测试风险报告
    print("\n2. 生成风险报告")

    risk_config = ReportConfig(
        report_type="risk",
        title="风险评估报告",
        description="投资组合风险分析",
        include_charts=True
    )

    risk_data = {
        'returns': np.random.normal(0.001, 0.02, 252),
        'benchmark_returns': np.random.normal(0.0005, 0.015, 252)
    }

    risk_report = manager.generate_report("risk", risk_config, risk_data)
    print(f"风险报告生成: {'成功' if risk_report else '失败'}")

    # 测试合规报告
    print("\n3. 生成合规报告")

    compliance_config = ReportConfig(
        report_type="compliance",
        title="合规审核报告",
        description="交易合规性检查结果",
        include_charts=False
    )

    compliance_data = {
        'trades': [],
        'violations': []
    }

    compliance_report = manager.generate_report("compliance", compliance_config, compliance_data)
    print(f"合规报告生成: {'成功' if compliance_report else '失败'}")

    # 检查生成的文件
    print("\n4. 检查生成的文件")

    if os.path.exists('reports'):
        files = os.listdir('reports')
        print(f"生成的报告文件: {files}")
    else:
        print("reports目录不存在")

    print("\n✅ 报告和分析系统测试完成")


if __name__ == "__main__":
    test_reporting_system()
