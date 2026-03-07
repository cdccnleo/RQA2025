#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成测试框架
提供端到端测试和系统集成测试能力
"""

import os
import sys
import json
import time
import unittest
from pathlib import Path
from typing import Dict, List, Any, Callable
from abc import ABC, abstractmethod
import psutil


class IntegrationTestCase(unittest.TestCase):
    """集成测试基类"""

    def setUp(self):
        """测试前准备"""
        self.start_time = time.time()
        self.test_data = {}
        self.resources_used = []

    def tearDown(self):
        """测试后清理"""
        end_time = time.time()
        execution_time = end_time - self.start_time

        # 记录资源使用
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()

        self.resources_used.append({
            "execution_time": execution_time,
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "timestamp": time.time()
        })

        # 清理测试数据
        self._cleanup_test_data()

    def _cleanup_test_data(self):
        """清理测试数据"""
        # 子类可以重写此方法
        pass

    def assert_response_time(self, execution_time: float, max_time: float, operation: str = ""):
        """断言响应时间"""
        self.assertLess(
            execution_time,
            max_time,
            f"{operation} 执行时间 {execution_time:.2f}秒 超过最大允许时间 {max_time}秒"
        )

    def assert_resource_usage(self, cpu_limit: float = 80.0, memory_limit: float = 80.0):
        """断言资源使用"""
        if self.resources_used:
            latest = self.resources_used[-1]
            self.assertLess(
                latest["cpu_percent"],
                cpu_limit,
                ".1f")
            self.assertLess(
                latest["memory_percent"],
                memory_limit,
                ".1f")

    def measure_performance(self, operation: Callable, operation_name: str = "") -> Dict[str, Any]:
        """测量性能"""
        start_time = time.time()
        start_cpu = psutil.cpu_percent(interval=None)
        start_memory = psutil.virtual_memory().used

        try:
            result = operation()
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)

        end_time = time.time()
        end_cpu = psutil.cpu_percent(interval=None)
        end_memory = psutil.virtual_memory().used

        performance_data = {
            "operation": operation_name,
            "execution_time": end_time - start_time,
            "cpu_usage": {
                "start": start_cpu,
                "end": end_cpu,
                "delta": end_cpu - start_cpu
            },
            "memory_usage": {
                "start": start_memory,
                "end": end_memory,
                "delta": end_memory - start_memory
            },
            "success": success,
            "error": error,
            "result": result
        }

        return performance_data


class TradingSystemIntegrationTest(IntegrationTestCase):
    """交易系统集成测试"""

    def setUp(self):
        super().setUp()
        # 初始化交易系统组件
        self.test_orders = []
        self.test_positions = []

    def test_complete_trading_workflow(self):
        """测试完整的交易工作流"""
        # 1. 创建订单
        order_data = {
            "symbol": "000001.SZ",
            "side": "BUY",
            "quantity": 100,
            "price": 10.50,
            "order_type": "LIMIT"
        }

        # 2. 提交订单
        performance_data = self.measure_performance(
            lambda: self._submit_order(order_data),
            "订单提交"
        )

        self.assertTrue(performance_data["success"])
        self.assert_response_time(performance_data["execution_time"], 2.0, "订单提交")

        order_id = performance_data["result"]
        self.assertIsNotNone(order_id)

        # 3. 检查订单状态
        status_data = self.measure_performance(
            lambda: self._check_order_status(order_id),
            "订单状态查询"
        )

        self.assertTrue(status_data["success"])
        self.assertEqual(status_data["result"]["status"], "PENDING")

        # 4. 模拟成交
        fill_data = self.measure_performance(
            lambda: self._simulate_order_fill(order_id),
            "订单成交"
        )

        self.assertTrue(fill_data["success"])

        # 5. 验证持仓更新
        position_data = self.measure_performance(
            lambda: self._check_position("000001.SZ"),
            "持仓查询"
        )

        self.assertTrue(position_data["success"])
        position = position_data["result"]
        self.assertEqual(position["quantity"], 100)
        self.assertAlmostEqual(position["average_price"], 10.50, places=2)

    def test_concurrent_order_processing(self):
        """测试并发订单处理"""
        import threading

        def submit_order_worker(order_id):
            order_data = {
                "symbol": f"00000{order_id:03d}.SZ",
                "side": "BUY",
                "quantity": 100,
                "price": 10.0 + order_id * 0.1
            }

            try:
                result = self._submit_order(order_data)
                return {"order_id": order_id, "result": result, "success": True}
            except Exception as e:
                return {"order_id": order_id, "error": str(e), "success": False}

        # 启动多个线程提交订单
        threads = []
        results = []

        for i in range(10):
            thread = threading.Thread(target=lambda: results.append(submit_order_worker(i)))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证结果
        successful_orders = [r for r in results if r.get("success", False)]
        self.assertGreaterEqual(len(successful_orders), 8, "至少80%的订单应该成功提交")

    def test_risk_management_integration(self):
        """测试风险管理集成"""
        # 1. 设置风险限额
        risk_limits = {
            "max_position_size": 1000000,
            "max_daily_loss": 50000,
            "max_single_order_size": 10000
        }

        setup_data = self.measure_performance(
            lambda: self._setup_risk_limits(risk_limits),
            "风险限额设置"
        )

        self.assertTrue(setup_data["success"])

        # 2. 提交大额订单（应该被拒绝）
        large_order = {
            "symbol": "000001.SZ",
            "side": "BUY",
            "quantity": 10000,  # 超过单笔订单限额
            "price": 10.50
        }

        rejection_data = self.measure_performance(
            lambda: self._submit_order(large_order),
            "大额订单提交"
        )

        # 订单应该被风控拒绝
        self.assertFalse(rejection_data["success"])
        self.assertIn("risk", rejection_data.get("error", "").lower())

    def test_data_flow_integration(self):
        """测试数据流集成"""
        # 1. 市场数据流
        market_data_stream = self._start_market_data_stream()

        # 2. 策略计算
        strategy_result = self.measure_performance(
            lambda: self._run_trading_strategy(market_data_stream),
            "策略计算"
        )

        self.assertTrue(strategy_result["success"])

        # 3. 信号生成
        signals = strategy_result["result"]
        self.assertIsInstance(signals, list)
        self.assertGreater(len(signals), 0)

        # 4. 订单生成和执行
        for signal in signals[:3]:  # 只处理前3个信号
            order_result = self.measure_performance(
                lambda: self._generate_and_execute_order(signal),
                f"信号{signal['id']}订单执行"
            )

            self.assertTrue(order_result["success"])

    def _submit_order(self, order_data):
        """模拟订单提交"""
        # 这里应该调用实际的交易引擎
        order_id = f"ORD-{int(time.time() * 1000)}"
        self.test_orders.append({"id": order_id, **order_data})
        return order_id

    def _check_order_status(self, order_id):
        """模拟订单状态查询"""
        order = next((o for o in self.test_orders if o["id"] == order_id), None)
        if order:
            return {"status": "PENDING", "order": order}
        return {"status": "NOT_FOUND"}

    def _simulate_order_fill(self, order_id):
        """模拟订单成交"""
        order = next((o for o in self.test_orders if o["id"] == order_id), None)
        if order:
            # 创建持仓记录
            position = {
                "symbol": order["symbol"],
                "quantity": order["quantity"],
                "average_price": order["price"],
                "market_value": order["quantity"] * order["price"]
            }
            self.test_positions.append(position)
        return True

    def _check_position(self, symbol):
        """模拟持仓查询"""
        position = next((p for p in self.test_positions if p["symbol"] == symbol), None)
        return position or {"quantity": 0, "average_price": 0}

    def _setup_risk_limits(self, limits):
        """模拟风险限额设置"""
        # 这里应该调用实际的风险管理系统
        return True

    def _start_market_data_stream(self):
        """模拟市场数据流"""
        # 生成模拟市场数据
        return [
            {"symbol": "000001.SZ", "price": 10.50, "volume": 10000, "timestamp": time.time() + i}
            for i in range(100)
        ]

    def _run_trading_strategy(self, market_data):
        """模拟策略计算"""
        signals = []
        for i, data in enumerate(market_data):
            if data["price"] > 10.40:  # 简单的价格突破策略
                signals.append({
                    "id": i,
                    "symbol": data["symbol"],
                    "signal": "BUY",
                    "price": data["price"],
                    "timestamp": data["timestamp"]
                })
        return signals

    def _generate_and_execute_order(self, signal):
        """模拟订单生成和执行"""
        order_data = {
            "symbol": signal["symbol"],
            "side": signal["signal"],
            "quantity": 100,
            "price": signal["price"]
        }

        order_id = self._submit_order(order_data)
        self._simulate_order_fill(order_id)
        return order_id


class DataProcessingIntegrationTest(IntegrationTestCase):
    """数据处理集成测试"""

    def test_data_pipeline_integration(self):
        """测试数据管道集成"""
        # 1. 数据摄取
        raw_data = self._ingest_raw_data()

        # 2. 数据清洗
        clean_data = self.measure_performance(
            lambda: self._clean_data(raw_data),
            "数据清洗"
        )

        self.assertTrue(clean_data["success"])
        self.assertGreater(len(clean_data["result"]), 0)

        # 3. 数据转换
        transformed_data = self.measure_performance(
            lambda: self._transform_data(clean_data["result"]),
            "数据转换"
        )

        self.assertTrue(transformed_data["success"])

        # 4. 数据存储
        storage_result = self.measure_performance(
            lambda: self._store_data(transformed_data["result"]),
            "数据存储"
        )

        self.assertTrue(storage_result["success"])

        # 5. 数据检索
        retrieved_data = self.measure_performance(
            lambda: self._retrieve_data(),
            "数据检索"
        )

        self.assertTrue(retrieved_data["success"])
        self.assertEqual(len(retrieved_data["result"]), len(transformed_data["result"]))

    def test_real_time_data_processing(self):
        """测试实时数据处理"""
        import queue
        import threading

        # 创建数据队列
        data_queue = queue.Queue()
        processed_data = []

        # 启动数据处理线程
        def process_data_worker():
            while True:
                try:
                    data = data_queue.get(timeout=1)
                    processed = self._process_single_record(data)
                    processed_data.append(processed)
                    data_queue.task_done()
                except queue.Empty:
                    break

        # 启动处理线程
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=process_data_worker)
            thread.start()
            threads.append(thread)

        # 生成实时数据
        for i in range(100):
            data_record = {
                "id": i,
                "value": i * 1.5,
                "timestamp": time.time(),
                "quality": "good" if i % 10 != 0 else "poor"
            }
            data_queue.put(data_record)

        # 等待处理完成
        data_queue.join()

        # 停止处理线程
        for thread in threads:
            thread.join()

        # 验证结果
        self.assertEqual(len(processed_data), 100)
        good_quality_count = sum(1 for p in processed_data if p.get("quality_score", 0) > 0.8)
        self.assertGreater(good_quality_count, 80)  # 80%的数据质量应该良好

    def _ingest_raw_data(self):
        """模拟原始数据摄取"""
        return [
            {"id": i, "raw_value": i * 2.5, "status": "raw"}
            for i in range(1000)
        ]

    def _clean_data(self, raw_data):
        """模拟数据清洗"""
        cleaned = []
        for record in raw_data:
            if record.get("status") == "raw":
                cleaned_record = {
                    "id": record["id"],
                    "value": record["raw_value"],
                    "status": "cleaned",
                    "quality_score": 0.95
                }
                cleaned.append(cleaned_record)
        return cleaned

    def _transform_data(self, clean_data):
        """模拟数据转换"""
        transformed = []
        for record in clean_data:
            transformed_record = {
                "id": record["id"],
                "normalized_value": (record["value"] - 0) / 2500,  # 简单的归一化
                "category": "high" if record["value"] > 1250 else "low",
                "processed_at": time.time()
            }
            transformed.append(transformed_record)
        return transformed

    def _store_data(self, transformed_data):
        """模拟数据存储"""
        # 这里应该调用实际的数据存储系统
        return len(transformed_data)

    def _retrieve_data(self):
        """模拟数据检索"""
        # 这里应该从存储系统中检索数据
        return [{"id": i, "value": i * 0.001} for i in range(1000)]

    def _process_single_record(self, record):
        """处理单个数据记录"""
        processed = {
            "id": record["id"],
            "processed_value": record["value"] * 1.1,
            "quality_score": 0.9 if record["quality"] == "good" else 0.6,
            "processed_at": time.time()
        }
        return processed


class SystemIntegrationTestSuite(unittest.TestSuite):
    """系统集成测试套件"""

    def __init__(self):
        super().__init__()
        self.addTest(TradingSystemIntegrationTest('test_complete_trading_workflow'))
        self.addTest(TradingSystemIntegrationTest('test_concurrent_order_processing'))
        self.addTest(TradingSystemIntegrationTest('test_risk_management_integration'))
        self.addTest(TradingSystemIntegrationTest('test_data_flow_integration'))
        self.addTest(DataProcessingIntegrationTest('test_data_pipeline_integration'))
        self.addTest(DataProcessingIntegrationTest('test_real_time_data_processing'))


def run_integration_tests(output_file: str = None) -> Dict[str, Any]:
    """运行集成测试"""
    print("🚀 开始执行集成测试")

    # 创建测试套件
    suite = SystemIntegrationTestSuite()

    # 创建测试运行器
    runner = unittest.TextTestRunner(verbosity=2)

    # 运行测试
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()

    # 收集测试结果
    test_results = {
        "timestamp": time.time(),
        "execution_time": end_time - start_time,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100,
        "failures_details": [str(failure) for failure in result.failures],
        "errors_details": [str(error) for error in result.errors],
        "skipped_details": [str(skip) for skip in result.skipped]
    }

    # 保存结果
    if output_file is None:
        output_file = f"integration_test_results_{int(time.time())}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)

    print(f"✅ 集成测试结果已保存到: {output_file}")

    # 打印摘要
    print("📊 集成测试摘要: ")
    print(f"  总测试数: {test_results['tests_run']}")
    print(f"  成功: {test_results['tests_run'] - test_results['failures'] - test_results['errors']}")
    print(f"  失败: {test_results['failures']}")
    print(f"  错误: {test_results['errors']}")
    print(f"  跳过: {test_results['skipped']}")
    print(f"  成功率: {test_results['success_rate']:.1f}%")
    print(f"  执行时间: {test_results['execution_time']:.2f}s")
    if test_results["success_rate"] >= 90:
        print("🎉 集成测试通过!")
    else:
        print("⚠️  部分集成测试失败，需要修复")

    return test_results


if __name__ == "__main__":
    import argparse

    parser=argparse.ArgumentParser(description="集成测试框架")
    parser.add_argument("--output", "-o", default=None, help="输出结果文件")

    args=parser.parse_args()

    run_integration_tests(args.output)
