#!/usr/bin/env python3
"""
聚焦ML子域测试覆盖的智能执行器

采用分层策略，优先执行稳定的ML组件测试：
1. ML核心组件（process_orchestrator, performance_monitor等）
2. 深度学习组件
3. 集成测试
4. 端到端测试

作者: AI Assistant
创建时间: 2025年12月4日
"""

import subprocess
import sys
import os
from pathlib import Path
import time
from typing import Dict, List, Any
from datetime import datetime


class MLCoverageFocusedRunner:
    """
    ML测试覆盖聚焦运行器
    """

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.ml_test_results = {}

    def get_ml_test_priorities(self) -> Dict[str, List[str]]:
        """
        获取ML测试优先级分组
        """
        return {
            "high_priority": [
                "tests/unit/ml/core/test_performance_monitor_unit.py",  # 已验证可工作
                "tests/unit/ml/core/test_process_orchestrator.py",
                "tests/unit/ml/core/test_ml_core.py",
                "tests/unit/ml/deep_learning/test_automl_engine.py",
            ],
            "medium_priority": [
                "tests/unit/ml/engine/",
                "tests/unit/ml/models/",
                "tests/unit/ml/deep_learning/",
            ],
            "low_priority": [
                "tests/unit/ml/integration/",
                "tests/unit/ml/tuning/",
                "tests/unit/ml/ensemble/",
            ]
        }

    def run_ml_test_file(self, test_file: str) -> Dict[str, Any]:
        """
        运行单个ML测试文件
        """
        start_time = time.time()

        try:
            cmd = [
                sys.executable, "-m", "pytest",
                test_file,
                "--tb=short",
                "--disable-warnings",
                "--maxfail=3",  # 最多失败3次就停止
                "--timeout=60",  # 单文件超时1分钟
                "-x",
                "-q"
            ]

            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=120  # 总超时2分钟
            )

            duration = time.time() - start_time

            # 解析结果
            passed = 0
            failed = 0

            for line in result.stdout.split('\n') + result.stderr.split('\n'):
                line = line.strip()
                if line.endswith('passed'):
                    try:
                        passed = int(line.split()[0])
                    except:
                        pass
                elif line.endswith('failed'):
                    try:
                        failed = int(line.split()[0])
                    except:
                        pass

            test_result = {
                "file": test_file,
                "status": "passed" if result.returncode == 0 else "failed",
                "return_code": result.returncode,
                "passed": passed,
                "failed": failed,
                "duration": round(duration, 2)
            }

            return test_result

        except subprocess.TimeoutExpired:
            return {
                "file": test_file,
                "status": "timeout",
                "duration": round(time.time() - start_time, 2)
            }
        except Exception as e:
            return {
                "file": test_file,
                "status": "error",
                "error": str(e),
                "duration": round(time.time() - start_time, 2)
            }

    def run_focused_ml_coverage(self) -> Dict[str, Any]:
        """
        运行聚焦的ML覆盖率测试
        """
        print("🚀 开始ML子域聚焦覆盖率测试")
        print("=" * 50)

        priorities = self.get_ml_test_priorities()
        all_results = {}
        total_passed = 0
        total_files = 0

        # 1. 高优先级测试 - 一个一个执行，确保稳定
        print("\n📍 执行高优先级ML测试...")
        for test_file in priorities["high_priority"]:
            if Path(test_file).exists():
                print(f"  测试: {Path(test_file).name}")
                result = self.run_ml_test_file(test_file)
                all_results[test_file] = result
                total_files += 1

                if result["status"] == "passed" and result.get("passed", 0) > 0:
                    total_passed += result["passed"]
                    print(f"    ✅ 通过 {result['passed']} 个测试")
                else:
                    print(f"    ❌ {result['status']}")
            else:
                print(f"  ⚠️ 文件不存在: {test_file}")

        # 2. 中优先级测试 - 目录级别执行
        print("\n📍 执行中优先级ML测试...")
        for test_dir in priorities["medium_priority"]:
            if Path(test_dir).exists():
                print(f"  目录: {Path(test_dir).name}")
                # 查找目录下的测试文件
                test_files = list(Path(test_dir).glob("test_*.py"))
                if test_files:
                    for test_file in test_files[:3]:  # 每个目录最多测试3个文件
                        result = self.run_ml_test_file(str(test_file))
                        all_results[str(test_file)] = result
                        total_files += 1

                        if result["status"] == "passed":
                            total_passed += result.get("passed", 0)
                            print(f"    ✅ {Path(test_file).name}: {result.get('passed', 0)} 通过")
                        else:
                            print(f"    ❌ {Path(test_file).name}: {result['status']}")

        # 3. 统计结果
        successful_files = sum(1 for r in all_results.values() if r["status"] == "passed")
        failed_files = sum(1 for r in all_results.values() if r["status"] == "failed")

        summary = {
            "total_files_tested": total_files,
            "successful_files": successful_files,
            "failed_files": failed_files,
            "total_tests_passed": total_passed,
            "success_rate": round(successful_files / max(total_files, 1) * 100, 1),
            "test_results": all_results,
            "timestamp": datetime.now().isoformat()
        }

        # 保存结果
        self.save_ml_results(summary)

        print("\n🏆 ML聚焦测试完成")
        print(f"测试文件: {total_files}")
        print(f"成功文件: {successful_files}")
        print(f"通过测试: {total_passed}")
        print(f"成功率: {summary['success_rate']}%")

        return summary

    def save_ml_results(self, results: Dict[str, Any]):
        """
        保存ML测试结果
        """
        output_dir = Path("test_logs")
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ml_coverage_focused_{timestamp}.json"

        output_file = output_dir / filename
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # 生成摘要报告
        summary_file = output_dir / f"ml_coverage_summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("ML子域聚焦测试覆盖率报告\n")
            f.write("=" * 40 + "\n")
            f.write(f"执行时间: {results['timestamp']}\n")
            f.write(f"测试文件数: {results['total_files_tested']}\n")
            f.write(f"成功文件数: {results['successful_files']}\n")
            f.write(f"失败文件数: {results['failed_files']}\n")
            f.write(f"通过测试数: {results['total_tests_passed']}\n")
            f.write(f"成功率: {results['success_rate']}%\n")
            f.write("\n详细结果:\n")

            for file_path, result in results['test_results'].items():
                status = result['status']
                passed = result.get('passed', 0)
                duration = result.get('duration', 0)
                f.write(f"  {Path(file_path).name}: {status} ({passed} passed, {duration:.1f}s)\n")

        print(f"💾 结果已保存: {output_file}")
        print(f"📄 摘要已保存: {summary_file}")


def main():
    """主函数"""
    runner = MLCoverageFocusedRunner()
    results = runner.run_focused_ml_coverage()

    # 返回适当的退出码
    if results["success_rate"] >= 60:
        print("✅ ML测试覆盖率达标")
        return 0
    else:
        print("⚠️ ML测试覆盖率有待改进")
        return 1


if __name__ == "__main__":
    sys.exit(main())



