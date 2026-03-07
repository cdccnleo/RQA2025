#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能测试执行脚本
针对关键业务模块进行性能基准测试和压力测试
"""

import os
import sys
import time
import json
import cProfile
import pstats
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from memory_profiler import profile as memory_profile


class PerformanceTestRunner:
    """性能测试运行器"""

    def __init__(self, workspace_path: str = None):
        self.workspace_path = workspace_path or os.getcwd()
        self.results = {
            "timestamp": time.time(),
            "performance_tests": {},
            "benchmark_results": {},
            "memory_analysis": {},
            "concurrency_tests": {},
            "stress_tests": {}
        }

    def run_trading_engine_performance_test(self) -> Dict[str, Any]:
        """运行交易引擎性能测试"""
        print("🚀 执行交易引擎性能测试")

        result = {
            "test_name": "trading_engine_performance",
            "execution_time": 0,
            "orders_processed": 0,
            "orders_per_second": 0,
            "memory_usage": {},
            "cpu_usage": {},
            "success": False
        }

        try:
            # 导入交易引擎
            sys.path.insert(0, self.workspace_path)
            from src.trading.trading_engine import TradingEngine

            # 初始化引擎
            engine = TradingEngine()

            # 准备测试数据
            test_orders = self._generate_test_orders(1000)

            # 记录开始时间和资源使用
            start_time = time.time()
            start_memory = psutil.virtual_memory().used
            start_cpu = psutil.cpu_percent(interval=None)

            # 执行性能测试
            processed_orders = []
            for order in test_orders:
                try:
                    result = engine.process_order(order)
                    processed_orders.append(result)
                except Exception as e:
                    print(f"处理订单失败: {e}")
                    continue

            # 记录结束时间和资源使用
            end_time = time.time()
            end_memory = psutil.virtual_memory().used
            end_cpu = psutil.cpu_percent(interval=None)

            # 计算结果
            execution_time = end_time - start_time
            result.update({
                "execution_time": execution_time,
                "orders_processed": len(processed_orders),
                "orders_per_second": len(processed_orders) / execution_time if execution_time > 0 else 0,
                "memory_usage": {
                    "start": start_memory,
                    "end": end_memory,
                    "delta": end_memory - start_memory
                },
                "cpu_usage": {
                    "start": start_cpu,
                    "end": end_cpu,
                    "average": (start_cpu + end_cpu) / 2
                },
                "success": True
            })

        except Exception as e:
            result["error"] = str(e)
            print(f"交易引擎性能测试失败: {e}")

        self.results["performance_tests"]["trading_engine"] = result
        return result

    def run_data_processing_performance_test(self) -> Dict[str, Any]:
        """运行数据处理性能测试"""
        print("🚀 执行数据处理性能测试")

        result = {
            "test_name": "data_processing_performance",
            "execution_time": 0,
            "data_records_processed": 0,
            "records_per_second": 0,
            "memory_usage": {},
            "success": False
        }

        try:
            # 准备大数据集
            test_data = self._generate_test_market_data(50000)

            # 记录开始时间
            start_time = time.time()
            start_memory = psutil.virtual_memory().used

            # 执行数据处理
            processed_data = []
            for record in test_data:
                # 模拟数据处理
                processed_record = {
                    "symbol": record["symbol"],
                    "price": record["price"] * 1.01,  # 模拟价格调整
                    "volume": record["volume"],
                    "timestamp": record["timestamp"]
                }
                processed_data.append(processed_record)

            # 记录结束时间
            end_time = time.time()
            end_memory = psutil.virtual_memory().used

            execution_time = end_time - start_time
            result.update({
                "execution_time": execution_time,
                "data_records_processed": len(processed_data),
                "records_per_second": len(processed_data) / execution_time if execution_time > 0 else 0,
                "memory_usage": {
                    "start": start_memory,
                    "end": end_memory,
                    "delta": end_memory - start_memory
                },
                "success": True
            })

        except Exception as e:
            result["error"] = str(e)
            print(f"数据处理性能测试失败: {e}")

        self.results["performance_tests"]["data_processing"] = result
        return result

    def run_concurrency_test(self, module_name: str, test_function: callable, num_threads: int = 10) -> Dict[str, Any]:
        """运行并发测试"""
        print(f"🚀 执行{module_name}并发测试 (线程数: {num_threads})")

        result = {
            "test_name": f"{module_name}_concurrency",
            "num_threads": num_threads,
            "total_operations": 0,
            "operations_per_second": 0,
            "execution_time": 0,
            "errors": 0,
            "success": False
        }

        def worker():
            nonlocal result
            try:
                start_time = time.time()
                operations = test_function()
                end_time = time.time()

                return {
                    "operations": operations,
                    "execution_time": end_time - start_time,
                    "success": True
                }
            except Exception as e:
                return {
                    "operations": 0,
                    "execution_time": 0,
                    "success": False,
                    "error": str(e)
                }

        try:
            start_time = time.time()

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(worker) for _ in range(num_threads)]
                results = []

                for future in as_completed(futures):
                    results.append(future.result())

            end_time = time.time()

            # 汇总结果
            total_operations = sum(r["operations"] for r in results)
            total_errors = sum(1 for r in results if not r["success"])
            avg_execution_time = sum(r["execution_time"] for r in results) / len(results)

            result.update({
                "total_operations": total_operations,
                "operations_per_second": total_operations / avg_execution_time if avg_execution_time > 0 else 0,
                "execution_time": end_time - start_time,
                "errors": total_errors,
                "success": total_errors == 0
            })

        except Exception as e:
            result["error"] = str(e)
            print(f"{module_name}并发测试失败: {e}")

        self.results["concurrency_tests"][module_name] = result
        return result

    def run_memory_profile_test(self, test_function: callable, test_name: str) -> Dict[str, Any]:
        """运行内存分析测试"""
        print(f"🚀 执行{test_name}内存分析测试")

        result = {
            "test_name": f"{test_name}_memory_profile",
            "peak_memory": 0,
            "memory_growth": 0,
            "memory_efficiency": 0,
            "success": False
        }

        try:
            # 使用memory_profiler进行内存分析
            @memory_profile
            def profiled_function():
                return test_function()

            # 执行测试并收集内存数据
            start_memory = psutil.virtual_memory().used
            test_result = profiled_function()
            end_memory = psutil.virtual_memory().used

            result.update({
                "peak_memory": end_memory,
                "memory_growth": end_memory - start_memory,
                "memory_efficiency": len(test_result) / (end_memory - start_memory) if end_memory > start_memory else 0,
                "success": True
            })

        except Exception as e:
            result["error"] = str(e)
            print(f"{test_name}内存分析测试失败: {e}")

        self.results["memory_analysis"][test_name] = result
        return result

    def run_stress_test(self, module_name: str, duration: int = 60) -> Dict[str, Any]:
        """运行压力测试"""
        print(f"🚀 执行{module_name}压力测试 (持续时间: {duration}秒)")

        result = {
            "test_name": f"{module_name}_stress_test",
            "duration": duration,
            "total_operations": 0,
            "operations_per_second": 0,
            "error_rate": 0.0,
            "resource_usage": {},
            "success": False
        }

        try:
            start_time = time.time()
            operations = 0
            errors = 0

            # 持续执行测试直到时间结束
            while time.time() - start_time < duration:
                try:
                    # 执行随机操作
                    self._perform_random_operation(module_name)
                    operations += 1
                except Exception as e:
                    errors += 1
                    continue

            end_time = time.time()
            actual_duration = end_time - start_time

            result.update({
                "duration": actual_duration,
                "total_operations": operations,
                "operations_per_second": operations / actual_duration if actual_duration > 0 else 0,
                "error_rate": errors / operations if operations > 0 else 0,
                "resource_usage": {
                    "cpu_percent": psutil.cpu_percent(interval=1),
                    "memory_percent": psutil.virtual_memory().percent
                },
                "success": True
            })

        except Exception as e:
            result["error"] = str(e)
            print(f"{module_name}压力测试失败: {e}")

        self.results["stress_tests"][module_name] = result
        return result

    def _generate_test_orders(self, count: int) -> List[Dict[str, Any]]:
        """生成测试订单数据"""
        orders = []
        for i in range(count):
            order = {
                "order_id": f"ORD-{i:06d}",
                "symbol": f"00000{i%1000:03d}.SZ",
                "side": "BUY" if i % 2 == 0 else "SELL",
                "quantity": (i % 1000) + 100,
                "price": 10.0 + (i % 50),
                "order_type": "LIMIT"
            }
            orders.append(order)
        return orders

    def _generate_test_market_data(self, count: int) -> List[Dict[str, Any]]:
        """生成测试市场数据"""
        data = []
        for i in range(count):
            record = {
                "symbol": f"00000{i%1000:03d}.SZ",
                "price": 10.0 + (i % 100),
                "volume": (i % 10000) + 1000,
                "timestamp": time.time() + i
            }
            data.append(record)
        return data

    def _perform_random_operation(self, module_name: str):
        """执行随机操作（用于压力测试）"""
        import random

        if module_name == "trading":
            # 模拟交易操作
            time.sleep(random.uniform(0.001, 0.01))
        elif module_name == "data":
            # 模拟数据处理操作
            data = [random.random() for _ in range(100)]
            sorted(data)
        elif module_name == "cache":
            # 模拟缓存操作
            cache_key = f"key_{random.randint(1, 1000)}"
            # 这里可以添加实际的缓存操作

    def run_full_performance_suite(self) -> Dict[str, Any]:
        """运行完整的性能测试套件"""
        print("🏁 开始执行完整性能测试套件")

        start_time = time.time()

        # 1. 交易引擎性能测试
        self.run_trading_engine_performance_test()

        # 2. 数据处理性能测试
        self.run_data_processing_performance_test()

        # 3. 并发测试
        def trading_operation():
            orders = self._generate_test_orders(10)
            return len(orders)  # 模拟处理订单数量

        self.run_concurrency_test("trading", trading_operation, num_threads=5)

        # 4. 内存分析测试
        def memory_test_function():
            data = []
            for i in range(10000):
                data.append({"id": i, "value": i * 2})
            return data

        self.run_memory_profile_test(memory_test_function, "data_processing")

        # 5. 压力测试
        self.run_stress_test("trading", duration=30)

        end_time = time.time()
        self.results["total_execution_time"] = end_time - start_time

        return self.results

    def generate_performance_report(self, output_path: str = None) -> str:
        """生成性能测试报告"""
        if output_path is None:
            output_path = f"performance_test_report_{int(time.time())}.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"✅ 性能测试报告已保存到: {output_path}")
        return output_path

    def print_summary(self):
        """打印性能测试摘要"""
        print("\n" + "="*60)
        print("⚡ 性能测试执行摘要")
        print("="*60)

        # 性能测试结果
        perf_tests = self.results.get("performance_tests", {})
        if perf_tests:
            print("\n🏃 性能测试结果:")
            for test_name, result in perf_tests.items():
                if result.get("success"):
                    ops_per_sec = result.get("orders_per_second", 0) or result.get(
                        "records_per_second", 0)
                    exec_time = result.get("execution_time", 0)
                    print(".2f" else:
                    print(f"  ❌ {test_name}: 失败 - {result.get('error', '未知错误')}")

        # 并发测试结果
        conc_tests=self.results.get("concurrency_tests", {})
        if conc_tests:
            print("\n🔄 并发测试结果:")
            for test_name, result in conc_tests.items():
                if result.get("success"):
                    ops_per_sec=result.get("operations_per_second", 0)
                    threads=result.get("num_threads", 0)
                    print(".1f" else:
                    print(f"  ❌ {test_name}: 失败 - {result.get('error', '未知错误')}")

        # 压力测试结果
        stress_tests=self.results.get("stress_tests", {})
        if stress_tests:
            print("\n💪 压力测试结果:")
            for test_name, result in stress_tests.items():
                if result.get("success"):
                    ops_per_sec=result.get("operations_per_second", 0)
                    error_rate=result.get("error_rate", 0)
                    print(".1f" else:
                    print(f"  ❌ {test_name}: 失败 - {result.get('error', '未知错误')}")

        total_time=self.results.get("total_execution_time", 0)
        print(".2f" print("\n✅ 性能测试执行完成")


def main():
    """主函数"""
    parser=argparse.ArgumentParser(description="性能测试执行脚本")
    parser.add_argument("--workspace", "-w", default=None, help="工作目录路径")
    parser.add_argument("--output", "-o", default=None, help="输出报告路径")
    parser.add_argument("--test-type", "-t", choices=["full", "trading", "data", "concurrency", "memory", "stress"],
                       default="full", help="测试类型")

    args=parser.parse_args()

    runner=PerformanceTestRunner(args.workspace)

    if args.test_type == "full":
        runner.run_full_performance_suite()
    elif args.test_type == "trading":
        runner.run_trading_engine_performance_test()
    elif args.test_type == "data":
        runner.run_data_processing_performance_test()
    elif args.test_type == "concurrency":
        def test_func():
            return 100
        runner.run_concurrency_test("test", test_func, num_threads=5)
    elif args.test_type == "memory":
        def mem_func():
            return [i for i in range(1000)]
        runner.run_memory_profile_test(mem_func, "memory_test")
    elif args.test_type == "stress":
        runner.run_stress_test("test", duration=30)

    # 生成报告
    report_path=runner.generate_performance_report(args.output)

    # 打印摘要
    runner.print_summary()


if __name__ == "__main__":
    main()
