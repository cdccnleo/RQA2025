#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025自动化测试报告生成器

生成详细的测试报告，包括：
- 测试结果统计
- 覆盖率分析
- 质量指标监控
- 趋势分析
- HTML报告输出
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# 确保项目根目录在路径中
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

class TestReportGenerator:
    """自动化测试报告生成器"""

    def __init__(self, output_dir: str = "test_reports"):
        self.output_dir = Path(PROJECT_ROOT) / output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def run_tests_with_reporting(self, test_scope: str = "tests/unit/",
                               coverage: bool = True) -> Dict[str, Any]:
        """运行测试并生成报告数据"""

        print(f"🚀 开始运行测试: {test_scope}")

        # 构建pytest命令
        cmd = [sys.executable, "-m", "pytest", test_scope, "--tb=short", "-v", "-q"]

        if coverage:
            cmd.extend([
                "--cov=src",
                "--cov-report=json:coverage.json",
                "--cov-report=term-missing"
            ])

        # 执行测试
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=300,  # 5分钟超时
                cwd=PROJECT_ROOT
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error": "测试执行超时"}

        execution_time = time.time() - start_time

        # 解析测试结果
        test_summary = self._parse_test_output(result.stdout, result.stderr)

        # 添加执行信息
        test_summary.update({
            "execution_time": execution_time,
            "exit_code": result.returncode,
            "timestamp": datetime.now().isoformat(),
            "test_scope": test_scope
        })

        # 如果有覆盖率数据，读取并解析
        if coverage and Path(PROJECT_ROOT / "coverage.json").exists():
            test_summary["coverage"] = self._parse_coverage_data()

        return test_summary

    def _parse_test_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """解析pytest输出"""

        summary = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "warnings": 0,
            "duration": 0.0
        }

        lines = stdout.split('\n')

        for line in lines:
            line = line.strip()

            # 查找汇总行
            if "passed" in line and "failed" in line:
                parts = line.split(',')
                for part in parts:
                    part = part.strip()
                    if "passed" in part:
                        try:
                            summary["passed"] = int(part.split()[0])
                        except:
                            pass
                    elif "failed" in part:
                        try:
                            summary["failed"] = int(part.split()[0])
                        except:
                            pass
                    elif "skipped" in part:
                        try:
                            summary["skipped"] = int(part.split()[0])
                        except:
                            pass
                    elif "errors" in part:
                        try:
                            summary["errors"] = int(part.split()[0])
                        except:
                            pass
                    elif "warnings" in part:
                        try:
                            summary["warnings"] = int(part.split()[0])
                        except:
                            pass

            # 查找持续时间
            if "in " in line and "s" in line:
                try:
                    duration_str = line.split("in ")[1].split("s")[0].strip()
                    summary["duration"] = float(duration_str)
                except:
                    pass

        summary["total_tests"] = summary["passed"] + summary["failed"] + summary["skipped"] + summary["errors"]

        return summary

    def _parse_coverage_data(self) -> Dict[str, Any]:
        """解析覆盖率数据"""

        coverage_file = PROJECT_ROOT / "coverage.json"
        if not coverage_file.exists():
            return {}

        try:
            with open(coverage_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 计算总体覆盖率
            total_lines = 0
            covered_lines = 0

            for file_path, file_data in data.items():
                if file_path.startswith(str(PROJECT_ROOT / "src")):
                    total_lines += file_data.get("summary", {}).get("num_statements", 0)
                    covered_lines += file_data.get("summary", {}).get("covered_lines", 0)

            overall_coverage = (covered_lines / total_lines * 100) if total_lines > 0 else 0

            return {
                "overall_percentage": round(overall_coverage, 2),
                "total_lines": total_lines,
                "covered_lines": covered_lines,
                "missing_lines": total_lines - covered_lines,
                "files_analyzed": len([f for f in data.keys() if f.startswith(str(PROJECT_ROOT / "src"))])
            }

        except Exception as e:
            print(f"解析覆盖率数据失败: {e}")
            return {}

    def generate_html_report(self, test_results: Dict[str, Any]) -> str:
        """生成HTML报告"""

        template = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RQA2025测试报告 - {self.timestamp}</title>
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
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .content {{
            padding: 30px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            border-left: 4px solid #3498db;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }}
        .metric-label {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        .status-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
            text-transform: uppercase;
        }}
        .status-passed {{ background: #27ae60; color: white; }}
        .status-failed {{ background: #e74c3c; color: white; }}
        .status-partial {{ background: #f39c12; color: white; }}
        .chart-container {{
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .footer {{
            background: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
            font-size: 0.9em;
        }}
        .success-rate {{
            font-size: 1.2em;
            color: #27ae60;
            font-weight: bold;
        }}
        .failure-rate {{
            font-size: 1.2em;
            color: #e74c3c;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧪 RQA2025测试报告</h1>
            <p>生成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</p>
            <p>测试范围: {test_results.get('test_scope', '未知')}</p>
        </div>

        <div class="content">
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{test_results.get('total_tests', 0)}</div>
                    <div class="metric-label">总测试数</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value success-rate">{test_results.get('passed', 0)}</div>
                    <div class="metric-label">通过测试</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value failure-rate">{test_results.get('failed', 0)}</div>
                    <div class="metric-label">失败测试</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{test_results.get('skipped', 0)}</div>
                    <div class="metric-label">跳过测试</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{test_results.get('execution_time', 0):.2f}s</div>
                    <div class="metric-label">执行时间</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{test_results.get('coverage', {}).get('overall_percentage', 0):.1f}%</div>
                    <div class="metric-label">代码覆盖率</div>
                </div>
            </div>

            <div class="chart-container">
                <h3>📊 测试执行摘要</h3>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="background: #f8f9fa;">
                        <th style="padding: 10px; border: 1px solid #ddd;">指标</th>
                        <th style="padding: 10px; border: 1px solid #ddd;">数值</th>
                        <th style="padding: 10px; border: 1px solid #ddd;">状态</th>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border: 1px solid #ddd;">测试成功率</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">
                            {((test_results.get('passed', 0) / test_results.get('total_tests', 1)) * 100):.1f}%
                        </td>
                        <td style="padding: 10px; border: 1px solid #ddd;">
                            <span class="status-badge {'status-passed' if (test_results.get('passed', 0) / max(test_results.get('total_tests', 1), 1)) > 0.8 else 'status-failed'}">
                                {'优秀' if (test_results.get('passed', 0) / max(test_results.get('total_tests', 1), 1)) > 0.8 else '需改进'}
                            </span>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border: 1px solid #ddd;">代码覆盖率</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">
                            {test_results.get('coverage', {}).get('overall_percentage', 0):.1f}%
                        </td>
                        <td style="padding: 10px; border: 1px solid #ddd;">
                            <span class="status-badge {'status-passed' if test_results.get('coverage', {}).get('overall_percentage', 0) > 40 else 'status-partial'}">
                                {'良好' if test_results.get('coverage', {}).get('overall_percentage', 0) > 40 else '可提升'}
                            </span>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border: 1px solid #ddd;">执行时间</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">
                            {test_results.get('execution_time', 0):.2f}秒
                        </td>
                        <td style="padding: 10px; border: 1px solid #ddd;">
                            <span class="status-badge status-passed">正常</span>
                        </td>
                    </tr>
                </table>
            </div>

            {f'''
            <div class="chart-container">
                <h3>🎯 覆盖率详情</h3>
                <p><strong>覆盖率统计:</strong></p>
                <ul>
                    <li>总代码行数: {test_results.get('coverage', {}).get('total_lines', 0)}</li>
                    <li>覆盖行数: {test_results.get('coverage', {}).get('covered_lines', 0)}</li>
                    <li>未覆盖行数: {test_results.get('coverage', {}).get('missing_lines', 0)}</li>
                    <li>分析文件数: {test_results.get('coverage', {}).get('files_analyzed', 0)}</li>
                </ul>
            </div>
            ''' if test_results.get('coverage') else ''}
        </div>

        <div class="footer">
            <p>🚀 RQA2025 质量保障系统 | 自动生成测试报告</p>
            <p>报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
        """

        return template

    def save_report(self, test_results: Dict[str, Any], report_name: str = None) -> str:
        """保存报告到文件"""

        if report_name is None:
            report_name = f"test_report_{self.timestamp}.html"

        report_path = self.output_dir / report_name
        html_content = self.generate_html_report(test_results)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"📄 测试报告已保存: {report_path}")
        return str(report_path)

def main():
    """主函数"""

    print("🚀 RQA2025自动化测试报告生成器")
    print("=" * 50)

    # 创建报告生成器
    generator = TestReportGenerator()

    # 运行单个测试套件作为示例
    print("\\n🔬 开始执行基础设施层测试...")
    results = generator.run_tests_with_reporting("tests/unit/infrastructure/")

    # 保存报告
    report_path = generator.save_report(results, "infrastructure_test_report.html")

    # 打印摘要
    print("\\n" + "=" * 50)
    print("📊 测试执行摘要:")
    print(f"   总测试数: {results.get('total_tests', 0)}")
    print(f"   通过: {results.get('passed', 0)}")
    print(f"   失败: {results.get('failed', 0)}")
    print(f"   跳过: {results.get('skipped', 0)}")
    print(f"   执行时间: {results.get('execution_time', 0):.2f}秒")
    print(f"   报告路径: {report_path}")

    print("\\n✅ 测试报告生成完成！")
    print("🌐 打开HTML报告查看详细结果")

if __name__ == "__main__":
    main()