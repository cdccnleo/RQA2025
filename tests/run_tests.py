#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RQA2025 自动化测试执行主程序
"""

import argparse
import concurrent.futures
from typing import List, Dict
import time
from datetime import datetime
from pathlib import Path
import sys

# 导入自定义模块
from utils.test_environment import TestEnvironmentManager
from utils.test_monitor import TestMonitor
from utils.data_visualizer import TestDataVisualizer
from utils.report_generator import HTMLReportGenerator

class TestRunner:
    def __init__(self, config_path="config/test_config.json"):
        """
        初始化测试运行器
        :param config_path: 测试配置文件路径
        """
        self.config = self._load_config(config_path)
        self.monitor = TestMonitor(self.config.get("alert"))
        self.visualizer = TestDataVisualizer(port=self.config.get("visualizer_port", 8050))
        self.report_generator = HTMLReportGenerator()

    def run_tests(self, test_types: List[str] = None, parallel: int = 1):
        """
        执行测试
        :param test_types: 要执行的测试类型列表(unit/integration/performance)
        :param parallel: 并行执行数
        """
        print("🚀 启动 RQA2025 自动化测试")
        start_time = datetime.now()

        # 启动可视化面板
        self.visualizer.start()

        # 初始化测试环境
        with TestEnvironmentManager(self.config["environment"]) as env:
            print("\n🔧 测试环境准备就绪")

            # 确定要执行的测试用例
            test_cases = self._select_test_cases(test_types)
            print(f"📋 共 {len(test_cases)} 个测试用例待执行")

            # 执行测试
            if parallel > 1:
                self._run_parallel_tests(test_cases, parallel)
            else:
                self._run_sequential_tests(test_cases)

        # 生成测试报告
        self._generate_reports(start_time)

        print(f"\n✅ 所有测试执行完成，总耗时: {datetime.now() - start_time}")

    def _run_sequential_tests(self, test_cases: List[Dict]):
        """顺序执行测试用例"""
        for case in test_cases:
            self._execute_test_case(case)

    def _run_parallel_tests(self, test_cases: List[Dict], max_workers: int):
        """并行执行测试用例"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for case in test_cases:
                futures.append(executor.submit(self._execute_test_case, case))

            # 等待所有测试完成
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"❌ 测试执行异常: {str(e)}")

    def _execute_test_case(self, test_case: Dict):
        """执行单个测试用例"""
        # 记录测试开始
        self.monitor.start_test_case(test_case["name"], test_case["type"])
        print(f"\n▶️ 开始执行: {test_case['name']} ({test_case['type']})")

        try:
            # 动态导入测试模块
            module = __import__(f"tests.{test_case['module']}", fromlist=[test_case["class"]])
            test_class = getattr(module, test_case["class"])
            test_method = getattr(test_class(), test_case["method"])

            # 执行测试
            start_time = time.time()
            test_method()
            duration = time.time() - start_time

            # 记录测试通过
            self.monitor.end_test_case(
                test_case["name"],
                "passed",
                f"执行成功，耗时: {duration:.2f}秒"
            )
            print(f"✅ 测试通过: {test_case['name']} (耗时: {duration:.2f}秒)")

            # 更新可视化数据
            self._update_visualizer(test_case, duration)

        except AssertionError as e:
            # 断言失败
            self.monitor.end_test_case(
                test_case["name"],
                "failed",
                f"断言失败: {str(e)}"
            )
            print(f"❌ 测试失败: {test_case['name']} - {str(e)}")

        except Exception as e:
            # 其他异常
            self.monitor.end_test_case(
                test_case["name"],
                "failed",
                f"执行异常: {str(e)}"
            )
            print(f"⚠️ 测试异常: {test_case['name']} - {str(e)}")

    def _update_visualizer(self, test_case: Dict, duration: float):
        """更新可视化数据"""
        if test_case["type"] == "performance":
            # 性能测试数据
            self.visualizer.add_performance_data(
                latency=duration,
                throughput=1000/duration if duration > 0 else 0
            )
        else:
            # 模拟市场数据
            symbol = test_case.get("symbol", "600519.SH")
            price = 1800 + (hash(test_case["name"]) % 200 - 100)
            volume = 10000 + (hash(test_case["name"]) % 5000)
            self.visualizer.add_market_data(symbol, price, volume)

            # 模拟订单数据
            self.visualizer.add_order_data(
                symbol=symbol,
                price=price,
                quantity=100,
                status="FILLED"
            )

    def _select_test_cases(self, test_types: List[str] = None) -> List[Dict]:
        """选择要执行的测试用例"""
        if test_types is None:
            return self.config["test_cases"]

        return [
            case for case in self.config["test_cases"]
            if case["type"] in test_types
        ]

    def _generate_reports(self, start_time):
        """生成测试报告"""
        print("\n📊 生成测试报告中...")

        # 准备报告数据
        test_results = {
            "test_cases": [],
            "performance": {
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat()
            }
        }

        # 添加测试用例结果
        for case in self.monitor.get_summary().to_dict("records"):
            test_results["test_cases"].append({
                "name": case["name"],
                "type": case["type"],
                "status": case["status"],
                "duration": case["duration"],
                "error": case["error"]
            })

        # 保存并生成报告
        self.report_generator.save_test_results(test_results)
        report_path = self.report_generator.generate_report(test_results)
        print(f"📄 测试报告已生成: file://{Path(report_path).absolute()}")

    def _load_config(self, config_path: str) -> Dict:
        """加载测试配置"""
        import json
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="RQA2025 自动化测试执行程序")
    parser.add_argument(
        "--types",
        nargs="+",
        choices=["unit", "integration", "performance"],
        help="指定要执行的测试类型"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="并行执行数"
    )
    parser.add_argument(
        "--config",
        default="config/test_config.json",
        help="测试配置文件路径"
    )

    args = parser.parse_args()

    try:
        runner = TestRunner(args.config)
        runner.run_tests(args.types, args.parallel)
    except Exception as e:
        print(f"❌ 测试执行失败: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
