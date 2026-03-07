#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
质量门检查脚本
检查测试质量指标是否满足标准要求
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any


class QualityGateChecker:
    """质量门检查器"""

    def __init__(self):
        # 定义质量门标准
        self.quality_gates = {
            "minimum_success_rate": 90.0,  # 最低成功率 90%
            "minimum_coverage": 75.0,      # 最低覆盖率 75%
            "minimum_quality_score": 70.0,  # 最低质量评分 70
            "maximum_failure_rate": 5.0,   # 最高失败率 5%
            "minimum_test_count": 50,      # 最低测试数量 50
            "maximum_execution_time": 600,  # 最高执行时间 10分钟
        }

        self.results = {
            "gates_checked": [],
            "passed_gates": [],
            "failed_gates": [],
            "overall_pass": True,
            "details": {}
        }

    def check_quality_gates(self, analysis_file: str) -> Dict[str, Any]:
        """检查质量门"""
        try:
            # 读取分析文件
            with open(analysis_file, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)

            # 检查总体质量门
            self._check_overall_gates(analysis_data)

            # 检查各层质量门
            self._check_layer_gates(analysis_data)

            # 生成检查报告
            self._generate_report()

            return self.results

        except Exception as e:
            print(f"质量门检查失败: {e}")
            self.results["error"] = str(e)
            self.results["overall_pass"] = False
            return self.results

    def _check_overall_gates(self, analysis_data: Dict[str, Any]):
        """检查总体质量门"""
        metrics = analysis_data.get("quality_metrics", {})

        # 检查成功率
        success_rate = metrics.get("average_success_rate", 0.0)
        self._check_gate(
            "overall_success_rate",
            success_rate,
            self.quality_gates["minimum_success_rate"],
            f"整体成功率 {success_rate:.1f}% 低于最低要求 {self.quality_gates['minimum_success_rate']}%"
        )

        # 检查覆盖率
        coverage = metrics.get("average_coverage", 0.0)
        self._check_gate(
            "overall_coverage",
            coverage,
            self.quality_gates["minimum_coverage"],
            f"整体覆盖率 {coverage:.1f}% 低于最低要求 {self.quality_gates['minimum_coverage']}%"
        )

        # 检查质量评分
        quality_score = metrics.get("average_quality_score", 0.0)
        self._check_gate(
            "overall_quality_score",
            quality_score,
            self.quality_gates["minimum_quality_score"],
            f"整体质量评分 {quality_score:.1f} 低于最低要求 {self.quality_gates['minimum_quality_score']}"
        )

        # 检查失败率
        total_tests = metrics.get("total_tests", 0)
        total_failed = metrics.get("total_failed", 0) + metrics.get("total_errors", 0)
        failure_rate = (total_failed / max(1, total_tests)) * 100
        self._check_gate(
            "failure_rate",
            failure_rate,
            self.quality_gates["maximum_failure_rate"],
            f"失败率 {failure_rate:.1f}% 高于最高允许 {self.quality_gates['maximum_failure_rate']}%",
            higher_is_worse=True
        )

        # 检查测试数量
        test_count = metrics.get("total_tests", 0)
        self._check_gate(
            "total_test_count",
            test_count,
            self.quality_gates["minimum_test_count"],
            f"总测试数 {test_count} 低于最低要求 {self.quality_gates['minimum_test_count']}"
        )

    def _check_layer_gates(self, analysis_data: Dict[str, Any]):
        """检查各层质量门"""
        layers = analysis_data.get("layers", {})

        for layer_name, layer_data in layers.items():
            # 检查层级成功率
            success_rate = layer_data.get("success_rate", 0.0)
            self._check_gate(
                f"{layer_name}_success_rate",
                success_rate,
                self.quality_gates["minimum_success_rate"],
                f"{layer_name} 成功率 {success_rate:.1f}% 低于要求"
            )

            # 检查层级覆盖率
            coverage = layer_data.get("coverage", 0.0)
            self._check_gate(
                f"{layer_name}_coverage",
                coverage,
                self.quality_gates["minimum_coverage"],
                f"{layer_name} 覆盖率 {coverage:.1f}% 低于要求"
            )

            # 检查层级质量评分
            quality_score = layer_data.get("quality_score", 0.0)
            self._check_gate(
                f"{layer_name}_quality_score",
                quality_score,
                self.quality_gates["minimum_quality_score"],
                f"{layer_name} 质量评分 {quality_score:.1f} 低于要求"
            )

            # 检查层级执行时间
            duration = layer_data.get("duration", 0)
            self._check_gate(
                f"{layer_name}_execution_time",
                duration,
                self.quality_gates["maximum_execution_time"],
                f"{layer_name} 执行时间 {duration:.1f}秒 超过限制",
                higher_is_worse=True
            )

    def _check_gate(self, gate_name: str, actual_value: float, threshold: float,
                   failure_message: str, higher_is_worse: bool = False):
        """检查单个质量门"""
        self.results["gates_checked"].append(gate_name)

        gate_result = {
            "gate": gate_name,
            "threshold": threshold,
            "actual": actual_value,
            "passed": False,
            "message": ""
        }

        if higher_is_worse:
            passed = actual_value <= threshold
        else:
            passed = actual_value >= threshold

        gate_result["passed"] = passed

        if passed:
            self.results["passed_gates"].append(gate_name)
            gate_result["message"] = f"✅ 通过 (实际值: {actual_value:.2f}, 阈值: {threshold:.2f})"
        else:
            self.results["failed_gates"].append(gate_name)
            self.results["overall_pass"] = False
            gate_result["message"] = f"❌ 失败 - {failure_message}"

        self.results["details"][gate_name] = gate_result

    def _generate_report(self):
        """生成检查报告"""
        pass

    def print_results(self):
        """打印检查结果"""
        print("\n" + "="*60)
        print("🔍 质量门检查结果")
        print("="*60)

        total_gates = len(self.results["gates_checked"])
        passed_gates = len(self.results["passed_gates"])
        failed_gates = len(self.results["failed_gates"])

        print(f"📊 检查质量门: {total_gates}")
        print(f"✅ 通过: {passed_gates}")
        print(f"❌ 失败: {failed_gates}")
        print(".1f"
        if self.results["overall_pass"]:
            print("🎉 所有质量门检查通过!")
        else:
            print("⚠️  部分质量门检查失败")

        print("\n详细结果:")
        for gate_name, detail in self.results["details"].items():
            status="✅" if detail["passed"] else "❌"
            print(f"  {status} {gate_name}: {detail['message']}")

        if failed_gates > 0:
            print("
❌ 失败的质量门: ")
            for gate in self.results["failed_gates"]:
                detail=self.results["details"][gate]
                print(f"  • {gate}: {detail['message']}")

    def save_results(self, output_file: str):
        """保存检查结果"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"✅ 质量门检查结果已保存到: {output_file}")


def main():
    """主函数"""
    parser=argparse.ArgumentParser(description="质量门检查脚本")
    parser.add_argument("--analysis-file", "-a", required=True, help="覆盖率分析文件")
    parser.add_argument("--output", "-o", default="quality_gate_results.json", help="输出结果文件")

    args=parser.parse_args()

    checker=QualityGateChecker()
    checker.check_quality_gates(args.analysis_file)
    checker.print_results()
    checker.save_results(args.output)

    # 设置退出码
    sys.exit(0 if checker.results["overall_pass"] else 1)


if __name__ == "__main__":
    main()
