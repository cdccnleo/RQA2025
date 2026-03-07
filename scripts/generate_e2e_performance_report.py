#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E2E测试性能报告生成器
"""

import json
import glob
from pathlib import Path
from datetime import datetime


def generate_e2e_performance_report():
    """生成E2E测试性能报告"""
    project_root = Path(__file__).parent.parent

    print("📊 生成E2E测试性能报告...")

    # 查找最新的监控报告
    reports_dir = project_root / "tests" / "e2e" / "reports"
    if not reports_dir.exists():
        print("❌ 未找到测试报告目录")
        return False

    report_files = list(reports_dir.glob("*.json"))
    if not report_files:
        print("❌ 未找到测试报告文件")
        return False

    # 读取最新的报告
    latest_report = max(report_files, key=lambda f: f.stat().st_mtime)

    with open(latest_report, 'r', encoding='utf-8') as f:
        report_data = json.load(f)

    # 生成HTML报告
    html_report = generate_html_performance_report(report_data)

    # 保存HTML报告
    html_file = reports_dir / \
        f"e2e_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_report)

    print(f"✅ 性能报告已生成: {html_file}")

    # 输出关键指标
    print("📈 关键性能指标: ")
    monitoring = report_data.get("monitoring_period", {})
    system_metrics = report_data.get("system_metrics", {})
    test_metrics = report_data.get("test_metrics", {})
    performance = report_data.get("performance_analysis", {})

    print(f"  执行时间: {monitoring.get('total_duration_minutes', 0):.1f} 分钟")
    print(
        f"  CPU使用率: 平均 {system_metrics.get('avg_cpu_usage', 0):.1f}%, 最大 {system_metrics.get('max_cpu_usage', 0):.1f}%")
    print(
        f"  内存使用率: 平均 {system_metrics.get('avg_memory_usage', 0):.1f}%, 最大 {system_metrics.get('max_memory_usage', 0):.1f}%")
    print(f"  测试通过率: {test_metrics.get('passed_tests', 0)}/{test_metrics.get('total_tests', 0)}")
    print(f"  平均测试时长: {test_metrics.get('avg_test_duration', 0):.1f} 秒")
    print(f"  效率评级: {performance.get('efficiency_rating', 'unknown')}")

    return True


def generate_html_performance_report(report_data):
    """生成HTML性能报告"""
    monitoring = report_data.get("monitoring_period", {})
    system_metrics = report_data.get("system_metrics", {})
    test_metrics = report_data.get("test_metrics", {})
    performance = report_data.get("performance_analysis", {})

    html_template = f'''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RQA2025 E2E测试性能报告</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 30px;
        }}
        .header {{
            text-align: center;
            border-bottom: 2px solid #007bff;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 6px;
            border-left: 4px solid #007bff;
        }}
        .metric-card h3 {{
            margin: 0 0 10px 0;
            color: #333;
            font-size: 1.2em;
        }}
        .metric-card p {{
            margin: 5px 0;
            color: #666;
        }}
        .good {{ border-left-color: #28a745; }}
        .warning {{ border-left-color: #ffc107; }}
        .danger {{ border-left-color: #dc3545; }}
        .bottlenecks {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 6px;
            padding: 15px;
            margin-top: 20px;
        }}
        .bottleneck-item {{
            background: white;
            margin: 10px 0;
            padding: 10px;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>RQA2025 E2E测试性能报告</h1>
            <p>测试执行时间: {monitoring.get('start_time', 'N/A')} - {monitoring.get('end_time', 'N/A')}</p>
        </div>

        <div class="metrics-grid">
            <div class="metric-card {'good' if monitoring.get('total_duration_minutes', 0) < 2 else 'warning'}">
                <h3>⏱️ 执行时间</h3>
                <p><strong>{monitoring.get('total_duration_minutes', 0):.1f} 分钟</strong></p>
                <p>目标: <2分钟</p>
            </div>

            <div class="metric-card {'good' if system_metrics.get('max_cpu_usage', 0) < 80 else 'warning'}">
                <h3>⚡ CPU使用率</h3>
                <p><strong>平均: {system_metrics.get('avg_cpu_usage', 0):.1f}%</strong></p>
                <p>最大: {system_metrics.get('max_cpu_usage', 0):.1f}%</p>
            </div>

            <div class="metric-card {'good' if system_metrics.get('max_memory_usage', 0) < 70 else 'warning'}">
                <h3>💾 内存使用率</h3>
                <p><strong>平均: {system_metrics.get('avg_memory_usage', 0):.1f}%</strong></p>
                <p>最大: {system_metrics.get('max_memory_usage', 0):.1f}%</p>
            </div>

            <div class="metric-card {'good' if test_metrics.get('passed_tests', 0) == test_metrics.get('total_tests', 0) else 'warning'}">
                <h3>✅ 测试通过率</h3>
                <p><strong>{test_metrics.get('passed_tests', 0)}/{test_metrics.get('total_tests', 0)}</strong></p>
                <p>通过率: {test_metrics.get('passed_tests', 0)/max(test_metrics.get('total_tests', 0), 1)*100:.1f}%</p>
            </div>
        </div>

        <h2>📊 详细指标</h2>

        <div class="metrics-grid">
            <div class="metric-card">
                <h3>🧪 测试指标</h3>
                <p><strong>总测试数:</strong> {test_metrics.get('total_tests', 0)}</p>
                <p><strong>通过测试:</strong> {test_metrics.get('passed_tests', 0)}</p>
                <p><strong>失败测试:</strong> {test_metrics.get('failed_tests', 0)}</p>
                <p><strong>平均时长:</strong> {test_metrics.get('avg_test_duration', 0):.1f}秒</p>
            </div>

            <div class="metric-card">
                <h3>🎯 性能分析</h3>
                <p><strong>效率评级:</strong> {performance.get('efficiency_rating', 'unknown')}</p>
                <p><strong>资源使用:</strong> {performance.get('resource_usage', 'unknown')}</p>
                <p><strong>目标达成:</strong> {'✅' if monitoring.get('total_duration_minutes', 0) < 2 else '❌'}</p>
            </div>
        </div>

        <div class="bottlenecks">
            <h3>🔍 性能瓶颈分析</h3>
            {"".join(f"<div class="bottleneck-item">{b}</div>" for b in performance.get('bottleneck_identified', ['无瓶颈']))}
        </div>
    </div>
</body>
</html>
'''

    return html_template


if __name__ == "__main__":
    success = generate_e2e_performance_report()
    exit(0 if success else 1)
