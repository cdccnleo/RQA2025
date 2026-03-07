#!/usr/bin/env python3
"""
测试覆盖率监控脚本
用于定期检查各层级的测试覆盖率状态
"""

import subprocess
import json
import time
from pathlib import Path
from typing import Dict
import argparse


class CoverageMonitor:
    """测试覆盖率监控器"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.report_dir = self.project_root / "reports" / "testing"
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # 监控的层级配置
        self.layers = [
            {
                "name": "utils",
                "path": "tests/unit/utils",
                "cov_path": "src/utils",
                "target": 25.0,
                "timeout": 30
            },
            {
                "name": "core",
                "path": "tests/unit/core",
                "cov_path": "src/core",
                "target": 25.0,
                "timeout": 30
            },
            {
                "name": "engine",
                "path": "tests/unit/engine",
                "cov_path": "src/engine",
                "target": 25.0,
                "timeout": 45
            },
            {
                "name": "infrastructure",
                "path": "tests/unit/infrastructure",
                "cov_path": "src/infrastructure",
                "target": 25.0,
                "timeout": 60
            }
        ]

    def run_layer_test(self, layer_config: Dict) -> Dict:
        """运行单个层级的测试"""
        layer_name = layer_config["name"]
        test_path = layer_config["path"]
        cov_path = layer_config["cov_path"]
        timeout = layer_config["timeout"]

        print(f"🔍 监控 {layer_name} 层覆盖率...")

        try:
            cmd = [
                "conda", "run", "-n", "test",
                "python", "-m", "pytest", test_path,
                f"--cov={cov_path}",
                "--cov-report=term-missing",
                "-v", "--tb=short",
                "--maxfail=3",
                f"--timeout={timeout}"
            ]

            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                timeout=timeout + 10,
                capture_output=True,
                text=True
            )

            # 解析覆盖率
            coverage = self._parse_coverage(result.stdout)

            return {
                "layer": layer_name,
                "success": result.returncode == 0,
                "coverage": coverage,
                "target": layer_config["target"],
                "timeout": timeout,
                "returncode": result.returncode,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

        except subprocess.TimeoutExpired:
            print(f"❌ {layer_name} 层测试超时")
            return {
                "layer": layer_name,
                "success": False,
                "coverage": 0.0,
                "target": layer_config["target"],
                "timeout": timeout,
                "error": "timeout",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            print(f"❌ {layer_name} 层测试异常: {e}")
            return {
                "layer": layer_name,
                "success": False,
                "coverage": 0.0,
                "target": layer_config["target"],
                "timeout": timeout,
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

    def _parse_coverage(self, output: str) -> float:
        """解析覆盖率数据"""
        try:
            lines = output.split('\n')
            for line in lines:
                if 'TOTAL' in line and '%' in line:
                    parts = line.split()
                    for part in parts:
                        if '%' in part:
                            return float(part.replace('%', ''))
            return 0.0
        except:
            return 0.0

    def monitor_all_layers(self) -> Dict:
        """监控所有层级"""
        print("🚀 开始监控测试覆盖率...")
        print("=" * 60)

        results = {}

        for layer_config in self.layers:
            result = self.run_layer_test(layer_config)
            results[layer_config['name']] = result

            # 显示结果
            if result["success"]:
                status = "✅" if result['coverage'] >= result['target'] else "⚠️"
                print(
                    f"{status} {layer_config['name']}: {result['coverage']:.2f}% (目标: {result['target']}%)")
            else:
                print(f"❌ {layer_config['name']}: 测试失败")

            # 短暂休息
            time.sleep(2)

        return results

    def generate_monitor_report(self, results: Dict) -> None:
        """生成监控报告"""
        print("\n📋 覆盖率监控报告")
        print("=" * 60)

        total_layers = len(results)
        successful_layers = sum(1 for r in results.values() if r["success"])
        total_coverage = sum(r["coverage"] for r in results.values() if r["success"])
        avg_coverage = total_coverage / successful_layers if successful_layers > 0 else 0

        print(f"监控层级数: {total_layers}")
        print(f"成功测试: {successful_layers}")
        print(f"平均覆盖率: {avg_coverage:.2f}%")

        print("\n详细状态:")
        for layer_name, result in results.items():
            if result["success"]:
                status = "✅" if result['coverage'] >= result['target'] else "⚠️"
                print(f"{status} {layer_name}: {result['coverage']:.2f}%")
            else:
                print(f"❌ {layer_name}: 测试失败")

        # 保存报告
        report_file = self.report_dir / f"coverage_monitor_{time.strftime('%Y%m%d_%H%M%S')}.json"

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "summary": {
                    "total_layers": total_layers,
                    "successful_layers": successful_layers,
                    "average_coverage": avg_coverage
                },
                "results": results
            }, f, indent=2, ensure_ascii=False)

        print(f"\n📄 监控报告已保存: {report_file}")

    def check_coverage_trend(self) -> None:
        """检查覆盖率趋势"""
        print("\n📈 覆盖率趋势分析")
        print("=" * 60)

        # 查找历史报告
        report_files = list(self.report_dir.glob("coverage_monitor_*.json"))
        report_files.sort()

        if len(report_files) < 2:
            print("⚠️ 历史报告不足，无法分析趋势")
            return

        # 读取最新的两个报告
        latest_report = json.loads(report_files[-1].read_text(encoding='utf-8'))
        previous_report = json.loads(report_files[-2].read_text(encoding='utf-8'))

        print(f"最新报告: {latest_report['timestamp']}")
        print(f"上次报告: {previous_report['timestamp']}")

        latest_avg = latest_report['summary']['average_coverage']
        previous_avg = previous_report['summary']['average_coverage']

        change = latest_avg - previous_avg
        print(f"平均覆盖率变化: {change:+.2f}%")

        if change > 0:
            print("📈 覆盖率有所提升")
        elif change < 0:
            print("📉 覆盖率有所下降")
        else:
            print("➡️ 覆盖率保持稳定")

    def run(self) -> None:
        """运行监控"""
        print("🎯 RQA2025 测试覆盖率监控")
        print("=" * 60)

        # 监控所有层级
        results = self.monitor_all_layers()

        # 生成报告
        self.generate_monitor_report(results)

        # 检查趋势
        self.check_coverage_trend()

        print("\n🎉 覆盖率监控完成！")
        print("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="测试覆盖率监控")
    parser.add_argument("--trend", action="store_true", help="仅检查覆盖率趋势")
    args = parser.parse_args()

    monitor = CoverageMonitor()

    if args.trend:
        monitor.check_coverage_trend()
    else:
        monitor.run()


if __name__ == "__main__":
    main()
