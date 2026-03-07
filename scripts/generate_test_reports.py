#!/usr/bin/env python3
"""
RQA2025 测试报告生成器
生成详细的测试覆盖率和质量报告
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd

class TestReportGenerator:
    """测试报告生成器"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.reports_dir = project_root / "test_reports"
        self.reports_dir.mkdir(exist_ok=True)

    def generate_coverage_badge(self, coverage_percent: float) -> str:
        """生成覆盖率徽章"""
        if coverage_percent >= 80:
            color = "brightgreen"
        elif coverage_percent >= 70:
            color = "green"
        elif coverage_percent >= 60:
            color = "yellow"
        elif coverage_percent >= 50:
            color = "orange"
        else:
            color = "red"

        badge_url = f"https://img.shields.io/badge/coverage-{coverage_percent:.1f}%25-{color}"
        return badge_url

    def generate_layer_report(self, layer_name: str, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成各层测试报告"""
        return {
            "layer": layer_name,
            "timestamp": datetime.now().isoformat(),
            "test_results": test_results,
            "coverage": test_results.get("coverage", 0),
            "passed": test_results.get("passed", 0),
            "failed": test_results.get("failed", 0),
            "skipped": test_results.get("skipped", 0),
            "errors": test_results.get("errors", 0)
        }

    def generate_comprehensive_report(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成综合测试报告"""
        total_passed = sum(r.get("passed", 0) for r in all_results.values() if isinstance(r, dict))
        total_failed = sum(r.get("failed", 0) for r in all_results.values() if isinstance(r, dict))
        total_skipped = sum(r.get("skipped", 0) for r in all_results.values() if isinstance(r, dict))
        total_errors = sum(r.get("errors", 0) for r in all_results.values() if isinstance(r, dict))

        total_tests = total_passed + total_failed + total_skipped + total_errors
        pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

        # 计算加权覆盖率
        total_weighted_coverage = 0
        total_weight = 0

        layer_weights = {
            "core": 0.25,
            "infrastructure": 0.20,
            "data": 0.15,
            "trading": 0.15,
            "risk": 0.10,
            "ml": 0.10,
            "feature": 0.05
        }

        for layer, results in all_results.items():
            if isinstance(results, dict) and "coverage" in results:
                weight = layer_weights.get(layer, 0.05)
                total_weighted_coverage += results["coverage"] * weight
                total_weight += weight

        weighted_coverage = total_weighted_coverage / total_weight if total_weight > 0 else 0

        report = {
            "report_type": "RQA2025 Comprehensive Test Report",
            "generated_at": datetime.now().isoformat(),
            "project_version": "RQA2025-v2.0",
            "summary": {
                "total_tests": total_tests,
                "passed": total_passed,
                "failed": total_failed,
                "skipped": total_skipped,
                "errors": total_errors,
                "pass_rate": round(pass_rate, 2),
                "weighted_coverage": round(weighted_coverage, 2)
            },
            "layer_results": all_results,
            "quality_metrics": {
                "test_pass_rate": round(pass_rate, 2),
                "coverage_target_achieved": weighted_coverage >= 70,
                "high_quality_threshold": pass_rate >= 95 and weighted_coverage >= 60,
                "enterprise_ready": pass_rate >= 98 and weighted_coverage >= 70
            },
            "recommendations": self.generate_recommendations(pass_rate, weighted_coverage)
        }

        return report

    def generate_recommendations(self, pass_rate: float, coverage: float) -> List[str]:
        """生成改进建议"""
        recommendations = []

        if pass_rate < 95:
            recommendations.append("🔧 提升测试通过率：修复失败的测试用例，改进测试用例质量")

        if coverage < 60:
            recommendations.append("📊 提升代码覆盖率：增加单元测试覆盖关键业务逻辑")

        if coverage < 70:
            recommendations.append("🎯 达到70%覆盖目标：重点覆盖复杂业务流程和边界条件")

        if pass_rate >= 98 and coverage >= 70:
            recommendations.append("✅ 达到企业级质量标准：建议进行性能优化和生产环境验证")

        if not recommendations:
            recommendations.append("🎉 测试质量优秀：继续保持高质量测试实践")

        return recommendations

    def save_report(self, report: Dict[str, Any], filename: str):
        """保存报告"""
        filepath = self.reports_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"📄 报告已保存: {filepath}")

    def generate_html_report(self, report: Dict[str, Any], filename: str):
        """生成HTML报告"""
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RQA2025 测试报告</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #4CAF50, #45a049);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
        }}
        .metric {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #4CAF50;
        }}
        .metric h3 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .metric .value {{
            font-size: 2em;
            font-weight: bold;
            color: #4CAF50;
        }}
        .layers {{
            padding: 30px;
        }}
        .layer {{
            background: #f8f9fa;
            margin: 10px 0;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #2196F3;
        }}
        .recommendations {{
            padding: 30px;
            background: #f8f9fa;
        }}
        .recommendation {{
            background: white;
            margin: 10px 0;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #FF9800;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 RQA2025 测试报告</h1>
            <p>生成时间: {report['generated_at']}</p>
            <p>项目版本: {report['project_version']}</p>
        </div>

        <div class="summary">
            <div class="metric">
                <h3>总测试数</h3>
                <div class="value">{report['summary']['total_tests']}</div>
            </div>
            <div class="metric">
                <h3>通过率</h3>
                <div class="value">{report['summary']['pass_rate']}%</div>
            </div>
            <div class="metric">
                <h3>覆盖率</h3>
                <div class="value">{report['summary']['weighted_coverage']}%</div>
            </div>
            <div class="metric">
                <h3>失败数</h3>
                <div class="value">{report['summary']['failed']}</div>
            </div>
        </div>

        <div class="layers">
            <h2>📊 分层测试结果</h2>
"""

        for layer_name, layer_data in report.get('layer_results', {}).items():
            if isinstance(layer_data, dict):
                html_content += f"""
            <div class="layer">
                <h3>{layer_name.upper()} 层</h3>
                <p>覆盖率: {layer_data.get('coverage', 0):.1f}% | 通过: {layer_data.get('passed', 0)} | 失败: {layer_data.get('failed', 0)}</p>
            </div>
"""

        html_content += """
        </div>

        <div class="recommendations">
            <h2>💡 改进建议</h2>
"""

        for rec in report.get('recommendations', []):
            html_content += f"""
            <div class="recommendation">
                {rec}
            </div>
"""

        html_content += """
        </div>
    </div>
</body>
</html>
"""

        filepath = self.reports_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"🌐 HTML报告已生成: {filepath}")


def main():
    """主函数"""
    project_root = Path(__file__).parent.parent
    generator = TestReportGenerator(project_root)

    # 示例数据（实际使用时从CI/CD获取）
    sample_results = {
        "infrastructure": {"passed": 295, "failed": 0, "skipped": 6, "errors": 0, "coverage": 85.5},
        "data": {"passed": 180, "failed": 0, "skipped": 2, "errors": 0, "coverage": 72.3},
        "core": {"passed": 450, "failed": 5, "skipped": 8, "errors": 2, "coverage": 68.9},
        "risk": {"passed": 320, "failed": 2, "skipped": 5, "errors": 0, "coverage": 65.4},
        "trading": {"passed": 380, "failed": 3, "skipped": 4, "errors": 1, "coverage": 71.2},
        "ml": {"passed": 280, "failed": 1, "skipped": 7, "errors": 0, "coverage": 58.7}
    }

    # 生成综合报告
    comprehensive_report = generator.generate_comprehensive_report(sample_results)

    # 保存JSON报告
    generator.save_report(comprehensive_report, "comprehensive_test_report.json")

    # 生成HTML报告
    generator.generate_html_report(comprehensive_report, "comprehensive_test_report.html")

    # 生成覆盖率徽章
    coverage = comprehensive_report['summary']['weighted_coverage']
    badge_url = generator.generate_coverage_badge(coverage)

    print(f"🎯 综合覆盖率: {coverage:.1f}%")
    print(f"📊 覆盖率徽章: {badge_url}")
    print("✅ 测试报告生成完成！")


if __name__ == "__main__":
    main()
