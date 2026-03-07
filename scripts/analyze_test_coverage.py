#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试覆盖率分析脚本
分析测试结果，生成覆盖率报告和质量评估
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import statistics


class CoverageAnalyzer:
    """覆盖率分析器"""

    def __init__(self):
        self.analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "layers": {},
            "quality_metrics": {},
            "recommendations": []
        }

    def analyze_layer_results(self, layer_name: str, result_file: str) -> Dict[str, Any]:
        """分析单层测试结果"""
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list):
                # 处理多个结果文件的情况
                combined_result = self._combine_results(data)
            else:
                combined_result = data

            layer_analysis = {
                "layer_name": layer_name,
                "total_tests": combined_result.get("summary", {}).get("total_tests", 0),
                "passed": combined_result.get("summary", {}).get("passed", 0),
                "failed": combined_result.get("summary", {}).get("failed", 0),
                "skipped": combined_result.get("summary", {}).get("skipped", 0),
                "errors": combined_result.get("summary", {}).get("errors", 0),
                "success_rate": combined_result.get("summary", {}).get("success_rate", 0.0),
                "duration": combined_result.get("duration", 0),
                "coverage": combined_result.get("coverage", 0.0),
                "quality_score": 0.0,
                "issues": []
            }

            # 计算质量评分
            layer_analysis["quality_score"] = self._calculate_quality_score(layer_analysis)

            # 识别问题
            layer_analysis["issues"] = self._identify_issues(layer_analysis)

            return layer_analysis

        except Exception as e:
            print(f"分析 {layer_name} 结果失败: {e}")
            return {
                "layer_name": layer_name,
                "error": str(e),
                "quality_score": 0.0,
                "issues": ["分析失败"]
            }

    def _combine_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """合并多个测试结果"""
        if not results:
            return {}

        combined = {
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": 0,
                "success_rate": 0.0
            },
            "duration": 0,
            "coverage": 0.0
        }

        for result in results:
            summary = result.get("summary", {})
            combined["summary"]["total_tests"] += summary.get("total_tests", 0)
            combined["summary"]["passed"] += summary.get("passed", 0)
            combined["summary"]["failed"] += summary.get("failed", 0)
            combined["summary"]["skipped"] += summary.get("skipped", 0)
            combined["summary"]["errors"] += summary.get("errors", 0)
            combined["duration"] += result.get("duration", 0)
            combined["coverage"] = max(combined["coverage"], result.get("coverage", 0.0))

        total_tests = combined["summary"]["total_tests"]
        passed = combined["summary"]["passed"]
        combined["summary"]["success_rate"] = (passed / max(1, total_tests)) * 100

        return combined

    def _calculate_quality_score(self, layer_analysis: Dict[str, Any]) -> float:
        """计算质量评分 (0-100)"""
        score = 0.0

        # 成功率权重 40%
        success_rate = layer_analysis.get("success_rate", 0.0)
        score += (success_rate / 100) * 40

        # 覆盖率权重 30%
        coverage = layer_analysis.get("coverage", 0.0)
        score += (coverage / 100) * 30

        # 测试数量权重 20%
        test_count = layer_analysis.get("total_tests", 0)
        test_score = min(100, test_count * 2)  # 50个测试得满分
        score += (test_score / 100) * 20

        # 执行时间权重 10%
        duration = layer_analysis.get("duration", 0)
        if duration > 0:
            time_score = max(0, 100 - (duration / 60) * 10)  # 每分钟扣10分
            score += (time_score / 100) * 10

        return round(score, 2)

    def _identify_issues(self, layer_analysis: Dict[str, Any]) -> List[str]:
        """识别质量问题"""
        issues = []

        if layer_analysis.get("success_rate", 0.0) < 95.0:
            issues.append(".1f" if layer_analysis.get("failed", 0) > 0:
            issues.append(f"存在 {layer_analysis['failed']} 个测试失败")

        if layer_analysis.get("errors", 0) > 0:
            issues.append(f"存在 {layer_analysis['errors']} 个测试错误")

        if layer_analysis.get("coverage", 0.0) < 80.0:
            issues.append(".1f" if layer_analysis.get("total_tests", 0) < 10:
            issues.append("测试数量过少，建议增加更多测试用例")

        if layer_analysis.get("duration", 0) > 300:  # 5分钟
            issues.append("测试执行时间过长，可能存在性能问题")

        if layer_analysis.get("quality_score", 0.0) < 70.0:
            issues.append("整体质量评分较低，需要重点改进")

        return issues

    def analyze_all_layers(self, input_dir: str) -> Dict[str, Any]:
        """分析所有层的结果"""
        input_path=Path(input_dir)

        if not input_path.exists():
            print(f"输入目录不存在: {input_dir}")
            return self.analysis_results

        # 查找所有结果文件
        result_files=list(input_path.glob("*.json"))

        if not result_files:
            print(f"在 {input_dir} 中未找到测试结果文件")
            return self.analysis_results

        print(f"发现 {len(result_files)} 个测试结果文件")

        # 分析每个层
        for result_file in result_files:
            layer_name=result_file.stem
            layer_analysis=self.analyze_layer_results(layer_name, str(result_file))
            self.analysis_results["layers"][layer_name]=layer_analysis

        # 计算总体质量指标
        self._calculate_overall_metrics()

        # 生成改进建议
        self._generate_recommendations()

        return self.analysis_results

    def _calculate_overall_metrics(self):
        """计算总体质量指标"""
        layers=self.analysis_results["layers"]

        if not layers:
            return

        # 计算平均值
        quality_scores=[layer.get("quality_score", 0.0) for layer in layers.values()]
        success_rates=[layer.get("success_rate", 0.0) for layer in layers.values()]
        coverages=[layer.get("coverage", 0.0) for layer in layers.values()]

        self.analysis_results["quality_metrics"]={
            "average_quality_score": round(statistics.mean(quality_scores), 2) if quality_scores else 0.0,
            "average_success_rate": round(statistics.mean(success_rates), 2) if success_rates else 0.0,
            "average_coverage": round(statistics.mean(coverages), 2) if coverages else 0.0,
            "total_tests": sum(layer.get("total_tests", 0) for layer in layers.values()),
            "total_passed": sum(layer.get("passed", 0) for layer in layers.values()),
            "total_failed": sum(layer.get("failed", 0) for layer in layers.values()),
            "total_errors": sum(layer.get("errors", 0) for layer in layers.values()),
            "layers_analyzed": len(layers)
        }

    def _generate_recommendations(self):
        """生成改进建议"""
        metrics=self.analysis_results["quality_metrics"]
        layers=self.analysis_results["layers"]

        recommendations=[]

        # 基于整体质量的建议
        if metrics.get("average_quality_score", 0.0) < 70.0:
            recommendations.append("整体测试质量需要提升，建议重点关注失败率较高的模块")

        if metrics.get("average_success_rate", 0.0) < 95.0:
            recommendations.append("测试成功率偏低，建议修复失败的测试用例")

        if metrics.get("average_coverage", 0.0) < 80.0:
            recommendations.append("代码覆盖率不足，建议增加更多测试用例覆盖未测试的代码")

        # 基于各层的建议
        for layer_name, layer in layers.items():
            issues=layer.get("issues", [])
            if issues:
                recommendations.append(f"{layer_name}: {'; '.join(issues)}")

        # 基于测试数量的建议
        total_tests=metrics.get("total_tests", 0)
        if total_tests < 100:
            recommendations.append("整体测试数量偏少，建议大幅增加测试用例")
        elif total_tests < 500:
            recommendations.append("测试数量中等，可以进一步扩展测试覆盖范围")

        self.analysis_results["recommendations"]=recommendations

    def generate_markdown_report(self, output_file: str):
        """生成Markdown格式的报告"""
        report=[]

        # 标题
        report.append("# 测试覆盖率分析报告")
        report.append("")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # 总体概况
        report.append("## 📊 总体概况")
        report.append("")

        metrics=self.analysis_results["quality_metrics"]
        report.append(f"- **总测试数**: {metrics.get('total_tests', 0)}")
        report.append(f"- **通过测试**: {metrics.get('total_passed', 0)}")
        report.append(f"- **失败测试**: {metrics.get('total_failed', 0)}")
        report.append(f"- **错误测试**: {metrics.get('total_errors', 0)}")
        report.append(".1f"        report.append(".1f"        report.append(".1f"        report.append(f"- **分析层数**: {metrics.get('layers_analyzed', 0)}")
        report.append("")

        # 分层详情
        report.append("## 📋 分层详情")
        report.append("")

        for layer_name, layer in self.analysis_results["layers"].items():
            report.append(f"### {layer_name}")
            report.append("")
            report.append(f"- **测试总数**: {layer.get('total_tests', 0)}")
            report.append(f"- **通过**: {layer.get('passed', 0)}")
            report.append(f"- **失败**: {layer.get('failed', 0)}")
            report.append(f"- **跳过**: {layer.get('skipped', 0)}")
            report.append(f"- **错误**: {layer.get('errors', 0)}")
            report.append(".1f"            report.append(".1f"            report.append(".2f"            report.append(f"- **执行时间**: {layer.get('duration', 0):.2f}秒")
            report.append("")

            issues=layer.get("issues", [])
            if issues:
                report.append("**问题**:")
                for issue in issues:
                    report.append(f"- {issue}")
                report.append("")

        # 改进建议
        recommendations=self.analysis_results.get("recommendations", [])
        if recommendations:
            report.append("## 🎯 改进建议")
            report.append("")

            for i, rec in enumerate(recommendations, 1):
                report.append(f"{i}. {rec}")
            report.append("")

        # 保存报告
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

        print(f"✅ Markdown报告已生成: {output_file}")

    def print_summary(self):
        """打印分析摘要"""
        print("\n" + "="*60)
        print("🎯 测试覆盖率分析摘要")
        print("="*60)

        metrics=self.analysis_results["quality_metrics"]
        print(".1f" print(".1f" print(".1f" print(f"📈 总测试数: {metrics.get('total_tests', 0)}")
        print(f"✅ 通过: {metrics.get('total_passed', 0)}")
        print(f"❌ 失败: {metrics.get('total_failed', 0)}")
        print(f"🔥 错误: {metrics.get('total_errors', 0)}")
        print(f"📊 分析层数: {metrics.get('layers_analyzed', 0)}")

        print("\n分层质量评分:")
        for layer_name, layer in self.analysis_results["layers"].items():
            score=layer.get("quality_score", 0.0)
            status="🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
            print(".1f"
        recommendations=self.analysis_results.get("recommendations", [])
        if recommendations:
            print("
🎯 关键改进建议: ")
            for i, rec in enumerate(recommendations[:3], 1):  # 只显示前3条
                print(f"  {i}. {rec}")


def main():
    """主函数"""
    parser=argparse.ArgumentParser(description="测试覆盖率分析脚本")
    parser.add_argument("--input-dir", "-i", required=True, help="测试结果输入目录")
    parser.add_argument("--output", "-o", default="coverage_analysis.md", help="输出报告文件")

    args=parser.parse_args()

    analyzer=CoverageAnalyzer()
    analyzer.analyze_all_layers(args.input_dir)
    analyzer.generate_markdown_report(args.output)
    analyzer.print_summary()


if __name__ == "__main__":
    main()
