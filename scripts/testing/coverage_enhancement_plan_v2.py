#!/usr/bin/env python3
"""
RQA2025 测试覆盖率提升计划 V2
采用分步骤、短时间、高优先级的方式避免死锁问题
"""

import subprocess
import time
import json
import os
from pathlib import Path
from typing import Dict
import argparse


class CoverageEnhancementPlanV2:
    """测试覆盖率提升计划V2"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.test_timeout = 120  # 缩短超时时间到2分钟
        self.max_retries = 1     # 减少重试次数

        # 按优先级排序的测试层级
        self.priority_layers = [
            {
                "name": "data",
                "path": "tests/unit/data",
                "cov_path": "src/data",
                "target": 25.0,
                "priority": "high",
                "timeout": 60
            },
            {
                "name": "utils",
                "path": "tests/unit/utils",
                "cov_path": "src/utils",
                "target": 25.0,
                "priority": "high",
                "timeout": 60
            },
            {
                "name": "engine",
                "path": "tests/unit/engine",
                "cov_path": "src/engine",
                "target": 25.0,
                "priority": "medium",
                "timeout": 90
            },
            {
                "name": "infrastructure",
                "path": "tests/unit/infrastructure",
                "cov_path": "src/infrastructure",
                "target": 35.0,
                "priority": "medium",
                "timeout": 120
            },
            {
                "name": "trading",
                "path": "tests/unit/trading",
                "cov_path": "src/trading",
                "target": 55.0,
                "priority": "low",
                "timeout": 120
            },
            {
                "name": "features",
                "path": "tests/unit/features",
                "cov_path": "src/features",
                "target": 60.0,
                "priority": "low",
                "timeout": 120
            }
        ]

        # 需要修复依赖的层级（暂时跳过）
        self.dependency_issues = ["models", "backtest", "acceleration"]

    def run_quick_test(self, layer_config: Dict) -> Dict:
        """运行快速测试，避免死锁"""
        layer_name = layer_config["name"]
        test_path = layer_config["path"]
        cov_path = layer_config["cov_path"]
        timeout = layer_config["timeout"]

        print(f"\n🔍 测试 {layer_name} 层 (超时: {timeout}秒)")

        try:
            # 使用简化的测试命令
            cmd = [
                "conda", "run", "-n", "test",
                "python", "-m", "pytest", test_path,
                f"--cov={cov_path}",
                "--cov-report=term-missing",
                "-v", "--tb=short",
                f"--timeout={timeout}"
            ]

            print(f"执行命令: {' '.join(cmd)}")

            # 设置环境变量避免死锁
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.project_root / 'src')
            env['PYTEST_TIMEOUT'] = str(timeout)

            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                timeout=timeout + 30,  # 额外30秒缓冲
                capture_output=True,
                text=True,
                env=env
            )

            # 解析覆盖率
            coverage = self._parse_coverage(result.stdout)

            return {
                "layer": layer_name,
                "success": result.returncode == 0,
                "coverage": coverage,
                "target": layer_config["target"],
                "timeout": timeout,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }

        except subprocess.TimeoutExpired:
            print(f"❌ {layer_name} 层测试超时")
            return {
                "layer": layer_name,
                "success": False,
                "coverage": 0.0,
                "target": layer_config["target"],
                "timeout": timeout,
                "error": "timeout"
            }
        except Exception as e:
            print(f"❌ {layer_name} 层测试异常: {e}")
            return {
                "layer": layer_name,
                "success": False,
                "coverage": 0.0,
                "target": layer_config["target"],
                "timeout": timeout,
                "error": str(e)
            }

    def _parse_coverage(self, output: str) -> float:
        """解析覆盖率数据"""
        try:
            lines = output.split('\n')
            for line in lines:
                if 'TOTAL' in line and '%' in line:
                    # 提取覆盖率百分比
                    parts = line.split()
                    for part in parts:
                        if '%' in part:
                            return float(part.replace('%', ''))
            return 0.0
        except:
            return 0.0

    def run_priority_tests(self) -> Dict:
        """运行高优先级测试"""
        print("🚀 开始执行高优先级测试覆盖率提升计划")
        print("=" * 60)

        results = {}

        for layer_config in self.priority_layers:
            if layer_config["priority"] == "high":
                print(f"\n🎯 高优先级测试: {layer_config['name']}")
                result = self.run_quick_test(layer_config)
                results[layer_config['name']] = result

                # 显示结果
                if result["success"]:
                    print(
                        f"✅ {layer_config['name']}: {result['coverage']:.2f}% (目标: {result['target']}%)")
                else:
                    print(f"❌ {layer_config['name']}: 测试失败")

                # 短暂休息避免资源冲突
                time.sleep(2)

        return results

    def run_medium_priority_tests(self) -> Dict:
        """运行中优先级测试"""
        print("\n🔄 开始执行中优先级测试")
        print("=" * 60)

        results = {}

        for layer_config in self.priority_layers:
            if layer_config["priority"] == "medium":
                print(f"\n📊 中优先级测试: {layer_config['name']}")
                result = self.run_quick_test(layer_config)
                results[layer_config['name']] = result

                if result["success"]:
                    print(
                        f"✅ {layer_config['name']}: {result['coverage']:.2f}% (目标: {result['target']}%)")
                else:
                    print(f"❌ {layer_config['name']}: 测试失败")

                time.sleep(3)  # 稍长休息时间

        return results

    def generate_report(self, results: Dict) -> None:
        """生成测试报告"""
        print("\n📋 测试覆盖率提升报告")
        print("=" * 60)

        total_tests = len(results)
        successful_tests = sum(1 for r in results.values() if r["success"])
        total_coverage = sum(r["coverage"] for r in results.values() if r["success"])
        avg_coverage = total_coverage / successful_tests if successful_tests > 0 else 0

        print(f"总测试数: {total_tests}")
        print(f"成功测试: {successful_tests}")
        print(f"平均覆盖率: {avg_coverage:.2f}%")

        print("\n详细结果:")
        for layer_name, result in results.items():
            status = "✅" if result["success"] else "❌"
            coverage = f"{result['coverage']:.2f}%" if result["success"] else "N/A"
            print(f"{status} {layer_name}: {coverage}")

        # 保存报告
        report_file = self.project_root / "reports" / "testing" / "coverage_enhancement_v2_report.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "summary": {
                    "total_tests": total_tests,
                    "successful_tests": successful_tests,
                    "average_coverage": avg_coverage
                },
                "results": results
            }, f, indent=2, ensure_ascii=False)

        print(f"\n📄 详细报告已保存: {report_file}")

    def create_missing_test_dirs(self) -> None:
        """创建缺失的测试目录"""
        missing_dirs = [
            "tests/unit/services",
            "tests/unit/tuning"
        ]

        print("\n📁 创建缺失的测试目录")
        for dir_path in missing_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                full_path.mkdir(parents=True, exist_ok=True)
                print(f"✅ 创建目录: {dir_path}")

                # 创建__init__.py文件
                init_file = full_path / "__init__.py"
                if not init_file.exists():
                    init_file.touch()
                    print(f"✅ 创建文件: {dir_path}/__init__.py")

    def run(self) -> None:
        """执行完整的覆盖率提升计划"""
        print("🎯 RQA2025 测试覆盖率提升计划 V2")
        print("=" * 60)
        print("策略: 分步骤、短时间、高优先级")
        print("目标: 避免死锁，快速提升覆盖率")

        # 创建缺失的测试目录
        self.create_missing_test_dirs()

        # 运行高优先级测试
        high_priority_results = self.run_priority_tests()

        # 运行中优先级测试
        medium_priority_results = self.run_medium_priority_tests()

        # 合并结果
        all_results = {**high_priority_results, **medium_priority_results}

        # 生成报告
        self.generate_report(all_results)

        print("\n🎉 测试覆盖率提升计划执行完成！")
        print("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="测试覆盖率提升计划V2")
    parser.add_argument("--quick", action="store_true", help="仅运行高优先级测试")
    args = parser.parse_args()

    plan = CoverageEnhancementPlanV2()

    if args.quick:
        print("🚀 快速模式：仅运行高优先级测试")
        results = plan.run_priority_tests()
        plan.generate_report(results)
    else:
        plan.run()


if __name__ == "__main__":
    main()
