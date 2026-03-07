#!/usr/bin/env python3
"""
RQA2025 分层测试执行脚本
按架构层级顺序执行测试，确保100%通过率
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime

class LayerTestRunner:
    """分层测试执行器"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.results = {}

        # 按优先级排序的层级执行顺序
        self.execution_order = [
            # 核心支撑层级 - 优先修复
            "infrastructure",
            "core_service",
            # 核心业务层级
            "data",
            "features",
            "ml",
            "strategy",
            "trading",
            "streaming",
            "risk",
            # 辅助支撑层级
            "monitoring",
            "optimization",
            "gateway",
            "adapters",
            "automation",
            "resilience",
            "testing",
            "utils",
            "distributed",
            "async_processor",
            "mobile",
            "boundary"
        ]

    def run_layer_tests(self, layer_name, test_paths):
        """运行指定层级的测试"""
        print(f"\n{'='*60}")
        print(f"🧪 执行 {layer_name} 层级测试")
        print(f"{'='*60}")

        layer_results = {
            "layer": layer_name,
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "pass_rate": 0.0,
            "status": "unknown",
            "failed_tests": []
        }

        for test_path in test_paths:
            full_path = self.project_root / test_path
            if not full_path.exists():
                continue

            print(f"\n📁 测试路径: {test_path}")

            try:
                # 运行pytest
                cmd = [
                    sys.executable, "-m", "pytest",
                    str(full_path),
                    "-v", "--tb=short", "--maxfail=5",
                    "--disable-warnings",
                    "-q"
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=self.project_root,
                    timeout=600  # 10分钟超时
                )

                # 解析结果
                output = result.stdout + result.stderr

                # 提取测试统计
                import re

                # 查找 "passed, X failed, Y skipped" 格式
                match = re.search(r'(\d+)\s*passed.*?(\d+)\s*failed.*?(\d+)\s*skipped', output, re.IGNORECASE)
                if match:
                    passed = int(match.group(1))
                    failed = int(match.group(2))
                    skipped = int(match.group(3))

                    layer_results["passed"] += passed
                    layer_results["failed"] += failed
                    layer_results["skipped"] += skipped
                    layer_results["total_tests"] += passed + failed + skipped

                    print(f"   ✅ 通过: {passed}, ❌ 失败: {failed}, ⏭️  跳过: {skipped}")

                    # 收集失败的测试
                    if failed > 0:
                        failed_lines = [line for line in output.split('\n') if 'FAILED' in line or 'ERROR' in line]
                        layer_results["failed_tests"].extend(failed_lines[:10])  # 只保留前10个失败
                else:
                    print(f"   ⚠️  无法解析测试结果: {result.returncode}")
                    layer_results["errors"] += 1

            except subprocess.TimeoutExpired:
                print(f"   ⏰ 测试超时")
                layer_results["errors"] += 1
            except Exception as e:
                print(f"   ❌ 测试执行失败: {e}")
                layer_results["errors"] += 1

        # 计算通过率
        if layer_results["total_tests"] > 0:
            layer_results["pass_rate"] = (layer_results["passed"] / layer_results["total_tests"]) * 100

            if layer_results["pass_rate"] >= 95:
                layer_results["status"] = "优秀"
            elif layer_results["pass_rate"] >= 80:
                layer_results["status"] = "良好"
            elif layer_results["pass_rate"] >= 60:
                layer_results["status"] = "基本"
            else:
                layer_results["status"] = "需修复"
        else:
            layer_results["status"] = "无测试"

        print(f"   🎯 通过率: {layer_results['pass_rate']:.1f}% | 状态: {layer_results['status']}")
        return layer_results

    def run_all_layers(self):
        """运行所有层级的测试"""
        print("🎯 RQA2025 分层测试执行 - 目标: 100%通过率")
        print("="*80)

        # 层级配置
        layer_configs = {
            "infrastructure": ["tests/unit/infrastructure/", "tests/integration/infrastructure/", "tests/e2e/infrastructure/"],
            "core_service": ["tests/unit/core/", "tests/integration/core/", "tests/e2e/core/"],
            "data": ["tests/unit/data/", "tests/integration/data/", "tests/e2e/data/"],
            "features": ["tests/unit/features/", "tests/integration/features/", "tests/e2e/features/"],
            "ml": ["tests/unit/ml/", "tests/integration/ml/", "tests/e2e/ml/"],
            "strategy": ["tests/unit/strategy/", "tests/integration/strategy/", "tests/e2e/strategy/"],
            "trading": ["tests/unit/trading/", "tests/integration/trading/", "tests/e2e/trading/"],
            "streaming": ["tests/unit/streaming/", "tests/integration/streaming/", "tests/e2e/streaming/"],
            "risk": ["tests/unit/risk/", "tests/integration/risk/", "tests/e2e/risk/"],
            "monitoring": ["tests/unit/monitoring/", "tests/integration/monitoring/", "tests/e2e/monitoring/"],
            "optimization": ["tests/unit/optimization/", "tests/integration/optimization/", "tests/e2e/optimization/"],
            "gateway": ["tests/unit/gateway/", "tests/integration/gateway/", "tests/e2e/gateway/"],
            "adapters": ["tests/unit/adapters/", "tests/integration/adapters/", "tests/e2e/adapters/"],
            "automation": ["tests/unit/automation/", "tests/integration/automation/", "tests/e2e/automation/"],
            "resilience": ["tests/unit/resilience/", "tests/integration/resilience/", "tests/e2e/resilience/"],
            "testing": ["tests/unit/testing/", "tests/integration/testing/", "tests/e2e/testing/"],
            "utils": ["tests/unit/utils/", "tests/integration/utils/", "tests/e2e/utils/"],
            "distributed": ["tests/unit/distributed/", "tests/integration/distributed/", "tests/e2e/distributed/"],
            "async_processor": ["tests/unit/async_processor/", "tests/integration/async_processor/", "tests/e2e/async_processor/"],
            "mobile": ["tests/unit/mobile/", "tests/integration/mobile/", "tests/e2e/mobile/"],
            "boundary": ["tests/unit/boundary/", "tests/integration/boundary/", "tests/e2e/boundary/"]
        }

        total_passed = 0
        total_failed = 0
        total_tests = 0

        for layer_name in self.execution_order:
            if layer_name in layer_configs:
                result = self.run_layer_tests(layer_name, layer_configs[layer_name])
                self.results[layer_name] = result

                total_passed += result["passed"]
                total_failed += result["failed"]
                total_tests += result["total_tests"]

        # 生成汇总报告
        self.generate_summary_report(total_passed, total_failed, total_tests)

    def generate_summary_report(self, total_passed, total_failed, total_tests):
        """生成汇总报告"""
        print(f"\n{'='*80}")
        print("📊 测试执行汇总报告")
        print(f"{'='*80}")

        print(f"⏰ 执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📦 总测试数: {total_tests}")
        print(f"✅ 通过测试: {total_passed}")
        print(f"❌ 失败测试: {total_failed}")

        if total_tests > 0:
            pass_rate = (total_passed / total_tests) * 100
            print(f"📊 总体通过率: {pass_rate:.1f}%")
            if pass_rate >= 95:
                print("🎉 达到目标: 95%+ 通过率")
            elif pass_rate >= 80:
                print("⚠️  接近目标: 需要继续修复")
            else:
                print("❌ 未达目标: 需要重点修复")
        else:
            print("⚠️  无测试执行")

        print(f"\n🏆 各层级状态:")

        status_counts = {"优秀": 0, "良好": 0, "基本": 0, "需修复": 0, "无测试": 0}
        for layer_name, result in self.results.items():
            status = result["status"]
            status_counts[status] = status_counts.get(status, 0) + 1

            emoji = {"优秀": "✅", "良好": "✅", "基本": "⚠️", "需修复": "❌", "无测试": "❓"}.get(status, "❓")
            print(f"            {emoji} {status}")
        print(f"\n📈 状态分布: 优秀({status_counts['优秀']}) | 良好({status_counts['良好']}) | 基本({status_counts['基本']}) | 需修复({status_counts['需修复']}) | 无测试({status_counts['无测试']})")

        # 保存详细报告
        self.save_detailed_report()

    def save_detailed_report(self):
        """保存详细报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.project_root / "test_logs" / f"layer_test_execution_{timestamp}.json"

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        print(f"\n📄 详细报告已保存: {report_file}")

        # 生成Markdown格式的报告
        md_report_file = self.project_root / "test_logs" / f"layer_test_execution_{timestamp}.md"
        with open(md_report_file, 'w', encoding='utf-8') as f:
            f.write("# RQA2025 分层测试执行报告\n\n")
            f.write(f"**执行时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            for layer_name, result in self.results.items():
                f.write(f"## {layer_name.upper()}\n\n")
                f.write(f"- **总测试数**: {result['total_tests']}\n")
                f.write(f"- **通过**: {result['passed']}\n")
                f.write(f"- **失败**: {result['failed']}\n")
                f.write(f"- **跳过**: {result['skipped']}\n")
                f.write(f"- **错误**: {result['errors']}\n")
                f.write(f"- **通过率**: {result['pass_rate']:.1f}%\n")
                f.write(f"- **状态**: {result['status']}\n")

                if result['failed_tests']:
                    f.write("- **失败详情**:\n")
                    for failed_test in result['failed_tests'][:5]:  # 只显示前5个
                        f.write(f"  - {failed_test}\n")

                f.write("\n")

        print(f"📄 Markdown报告已保存: {md_report_file}")

def main():
    """主函数"""
    runner = LayerTestRunner()
    runner.run_all_layers()

if __name__ == "__main__":
    main()
