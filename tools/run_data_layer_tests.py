#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 数据层测试自动化脚本

自动执行数据层所有测试用例，生成覆盖率报告和性能基准。
"""

import sys
import time
import subprocess
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 日志降级处理


def get_data_logger(name: str):
    """获取数据层日志器，支持降级"""
    try:
        from src.infrastructure.logging.unified_logger import UnifiedLogger
        return UnifiedLogger(name)
    except ImportError:
        import logging
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


logger = get_data_logger('data_layer_test_runner')


class DataLayerTestRunner:
    """数据层测试运行器"""

    def __init__(self):
        self.project_root = project_root
        self.test_results = {}
        self.coverage_report = {}
        self.performance_results = {}

    def run_unit_tests(self) -> Dict[str, Any]:
        """运行单元测试"""
        logger.info("开始运行数据层单元测试...")

        test_commands = [
            ["python", "-m", "pytest", "tests/unit/data/test_standard_interfaces.py", "-v", "--tb=short"],
            ["python", "-m", "pytest", "tests/unit/data/test_smart_data_cache.py", "-v", "--tb=short"]
        ]

        results = {}
        for i, cmd in enumerate(test_commands):
            test_name = f"unit_test_{i+1}"
            logger.info(f"执行: {' '.join(cmd)}")

            start_time = time.time()
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
            end_time = time.time()

            results[test_name] = {
                "command": cmd,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": end_time - start_time,
                "success": result.returncode == 0
            }

            if result.returncode == 0:
                logger.info(f"✅ {test_name} 通过")
            else:
                logger.error(f"❌ {test_name} 失败")

        return results

    def run_integration_tests(self) -> Dict[str, Any]:
        """运行集成测试"""
        logger.info("开始运行数据层集成测试...")

        test_commands = [
            ["python", "-m", "pytest", "tests/integration/data/test_data_layer_integration.py", "-v", "--tb=short"]
        ]

        results = {}
        for i, cmd in enumerate(test_commands):
            test_name = f"integration_test_{i+1}"
            logger.info(f"执行: {' '.join(cmd)}")

            start_time = time.time()
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
            end_time = time.time()

            results[test_name] = {
                "command": cmd,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": end_time - start_time,
                "success": result.returncode == 0
            }

            if result.returncode == 0:
                logger.info(f"✅ {test_name} 通过")
            else:
                logger.error(f"❌ {test_name} 失败")

        return results

    def run_performance_tests(self) -> Dict[str, Any]:
        """运行性能测试"""
        logger.info("开始运行数据层性能测试...")

        test_commands = [
            ["python", "-m", "pytest", "tests/performance/data/test_data_layer_performance.py::TestDataLayerPerformance::test_cache_performance", "-v", "--tb=short", "-s"],
            ["python", "-m", "pytest", "tests/performance/data/test_data_layer_performance.py::TestDataLayerPerformance::test_quality_monitor_performance", "-v", "--tb=short", "-s"]
        ]

        results = {}
        for i, cmd in enumerate(test_commands):
            test_name = f"performance_test_{i+1}"
            logger.info(f"执行: {' '.join(cmd)}")

            start_time = time.time()
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
            end_time = time.time()

            results[test_name] = {
                "command": cmd,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": end_time - start_time,
                "success": result.returncode == 0
            }

            if result.returncode == 0:
                logger.info(f"✅ {test_name} 通过")
            else:
                logger.error(f"❌ {test_name} 失败")

        return results

    def generate_coverage_report(self) -> Dict[str, Any]:
        """生成覆盖率报告"""
        logger.info("生成测试覆盖率报告...")

        try:
            cmd = ["python", "-m", "pytest", "tests/unit/data/", "tests/integration/data/",
                   "--cov=src.data", "--cov-report=html", "--cov-report=term"]
            result = subprocess.run(cmd, cwd=self.project_root,
                                    capture_output=True, text=True, timeout=300)

            coverage_info = {
                "command": cmd,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0
            }

            if result.returncode == 0:
                logger.info("✅ 覆盖率报告生成成功")
            else:
                logger.error("❌ 覆盖率报告生成失败")

            return coverage_info

        except subprocess.TimeoutExpired:
            logger.error("覆盖率报告生成超时")
            return {"error": "timeout"}
        except Exception as e:
            logger.error(f"覆盖率报告生成出错: {e}")
            return {"error": str(e)}

    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        logger.info("🚀 开始执行数据层完整测试套件")
        start_time = time.time()

        # 运行各类型测试
        self.test_results["unit_tests"] = self.run_unit_tests()
        self.test_results["integration_tests"] = self.run_integration_tests()
        self.test_results["performance_tests"] = self.run_performance_tests()

        # 生成覆盖率报告
        self.coverage_report = self.generate_coverage_report()

        end_time = time.time()
        total_duration = end_time - start_time

        # 汇总结果
        summary = self._generate_summary(total_duration)

        logger.info(f"⏱️ 总执行时间: {total_duration:.2f}秒")
        logger.info(f"📊 测试通过率: {summary['overall_pass_rate']:.1f}%")

        return {
            "summary": summary,
            "results": self.test_results,
            "coverage": self.coverage_report,
            "duration": total_duration,
            "timestamp": datetime.now().isoformat()
        }

    def _generate_summary(self, total_duration: float) -> Dict[str, Any]:
        """生成测试摘要"""
        all_tests = []
        for test_type, results in self.test_results.items():
            for test_name, result in results.items():
                all_tests.append(result)

        total_tests = len(all_tests)
        passed_tests = sum(1 for test in all_tests if test["success"])
        failed_tests = total_tests - passed_tests

        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        # 按类型统计
        type_stats = {}
        for test_type, results in self.test_results.items():
            type_total = len(results)
            type_passed = sum(1 for result in results.values() if result["success"])
            type_stats[test_type] = {
                "total": type_total,
                "passed": type_passed,
                "failed": type_total - type_passed,
                "pass_rate": (type_passed / type_total * 100) if type_total > 0 else 0
            }

        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "overall_pass_rate": pass_rate,
            "total_duration": total_duration,
            "type_stats": type_stats,
            "coverage_success": self.coverage_report.get("success", False)
        }

    def save_results(self, results: Dict[str, Any], output_file: str = None):
        """保存测试结果"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"data_layer_test_results_{timestamp}.json"

        output_path = self.project_root / "reports" / output_file

        # 确保reports目录存在
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"测试结果已保存到: {output_path}")
        return output_path


def main():
    """主函数"""
    logger.info("🎯 RQA2025 数据层测试自动化开始")

    runner = DataLayerTestRunner()
    results = runner.run_all_tests()

    # 保存结果
    output_file = runner.save_results(results)

    # 打印最终摘要
    summary = results["summary"]
    print("\n" + "="*60)
    print("🎊 数据层测试执行完成")
    print("="*60)
    print(f"总测试数: {summary['total_tests']}")
    print(f"通过测试: {summary['passed_tests']}")
    print(f"失败测试: {summary['failed_tests']}")
    print(f"测试通过率: {summary['overall_pass_rate']:.1f}%")
    print(f"总执行时间: {summary['total_duration']:.2f}秒")
    print(f"覆盖率报告: {'✅ 成功' if summary['coverage_success'] else '❌ 失败'}")
    print(f"结果文件: {output_file}")
    print("="*60)

    # 返回适当的退出码
    return 0 if summary['overall_pass_rate'] >= 80 else 1


if __name__ == "__main__":
    sys.exit(main())
