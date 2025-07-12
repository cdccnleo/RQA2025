#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RQA2025 测试报告生成工具
生成HTML格式的可视化测试报告
"""

from datetime import datetime
import json
import os
from jinja2 import Environment, FileSystemLoader

class TestHTMLReportGenerator:
    def __init__(self, output_dir="reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 初始化模板引擎
        self.env = Environment(
            loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), 'templates')),
            autoescape=True
        )

    def generate_report(self, test_results, report_name="test_report"):
        """
        生成HTML测试报告
        :param test_results: 测试结果数据
        :param report_name: 报告文件名(不含扩展名)
        :return: 报告文件路径
        """
        # 准备报告数据
        report_data = {
            "title": "RQA2025 测试报告",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": self._generate_summary(test_results),
            "test_cases": test_results.get("test_cases", []),
            "metrics": self._calculate_metrics(test_results),
            "performance": test_results.get("performance", {})
        }

        # 渲染模板
        template = self.env.get_template("report_template.html")
        html_content = template.render(report_data)

        # 保存报告
        report_path = os.path.join(self.output_dir, f"{report_name}.html")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return report_path

    def _generate_summary(self, test_results):
        """生成报告摘要"""
        total = len(test_results.get("test_cases", []))
        passed = sum(1 for case in test_results["test_cases"] if case["status"] == "passed")
        failed = total - passed

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / total * 100, 2) if total > 0 else 0
        }

    def _calculate_metrics(self, test_results):
        """计算各项指标"""
        metrics = {
            "unit_test": {"passed": 0, "failed": 0},
            "integration_test": {"passed": 0, "failed": 0},
            "performance_test": {"latency": []}
        }

        for case in test_results.get("test_cases", []):
            if case["type"] == "unit":
                if case["status"] == "passed":
                    metrics["unit_test"]["passed"] += 1
                else:
                    metrics["unit_test"]["failed"] += 1
            elif case["type"] == "integration":
                if case["status"] == "passed":
                    metrics["integration_test"]["passed"] += 1
                else:
                    metrics["integration_test"]["failed"] += 1
            elif case["type"] == "performance":
                metrics["performance_test"]["latency"].append(case.get("latency", 0))

        # 计算性能指标百分位
        if metrics["performance_test"]["latency"]:
            latencies = sorted(metrics["performance_test"]["latency"])
            metrics["performance_test"].update({
                "p50": latencies[int(len(latencies)*0.5)],
                "p90": latencies[int(len(latencies)*0.9)],
                "p99": latencies[int(len(latencies)*0.99)]
            })

        return metrics

    @staticmethod
    def save_test_results(results, file_path="test_results.json"):
        """保存原始测试结果"""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_test_results(file_path="test_results.json"):
        """加载测试结果"""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)


if __name__ == "__main__":
    # 示例用法
    sample_results = {
        "test_cases": [
            {"name": "熔断机制测试", "type": "unit", "status": "passed", "duration": 0.12},
            {"name": "FPGA一致性测试", "type": "unit", "status": "passed", "duration": 0.25},
            {"name": "交易全流程测试", "type": "integration", "status": "failed", "duration": 1.32},
            {"name": "订单延迟测试", "type": "performance", "status": "passed", "latency": 0.045}
        ],
        "performance": {
            "throughput": 1250,
            "error_rate": 0.001
        }
    }

    generator = TestHTMLReportGenerator()
    generator.save_test_results(sample_results)
    report_path = generator.generate_report(sample_results)
    print(f"测试报告已生成: {report_path}")
