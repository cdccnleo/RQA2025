#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
实时引擎压力测试模块
用于验证系统在高负载情况下的稳定性和性能
"""

import time
import random
import threading
from typing import Dict, List, Optional
from queue import Queue
from src.utils.logger import get_logger
from src.engine.realtime_engine import RealTimeEngine
from src.data.china.china_data_adapter import ChinaDataAdapter

logger = get_logger(__name__)

class StressTester:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化压力测试器
        :param config: 配置参数
        """
        self.config = config
        self.engine = RealTimeEngine(config)
        self.data_adapter = ChinaDataAdapter(config)
        self.running = False
        self.threads = []
        self.test_scenarios = self._load_test_scenarios()

    def _load_test_scenarios(self) -> List[Dict[str, Any]]:
        """
        加载测试场景配置
        :return: 测试场景列表
        """
        return [
            {
                "name": "Level2行情风暴",
                "description": "模拟Level2行情数据高峰",
                "params": {
                    "tick_rate": 10000,  # 每秒消息数
                    "duration": 60,      # 测试持续时间(秒)
                    "symbols": ["600519.SH", "000001.SZ", "688981.SH"]
                }
            },
            {
                "name": "订单风暴",
                "description": "模拟高频订单场景",
                "params": {
                    "order_rate": 5000,  # 每秒订单数
                    "duration": 60,
                    "symbols": ["600519.SH", "000001.SZ"]
                }
            },
            {
                "name": "混合负载",
                "description": "行情和订单混合压力测试",
                "params": {
                    "tick_rate": 8000,
                    "order_rate": 3000,
                    "duration": 120,
                    "symbols": ["600519.SH", "000001.SZ", "688981.SH"]
                }
            }
        ]

    def run_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """
        运行指定测试场景
        :param scenario_name: 场景名称
        :return: 测试结果
        """
        scenario = next((s for s in self.test_scenarios if s['name'] == scenario_name), None)
        if not scenario:
            raise ValueError(f"未知测试场景: {scenario_name}")

        logger.info(f"开始压力测试场景: {scenario['name']}")
        logger.info(f"场景描述: {scenario['description']}")

        # 初始化测试结果
        result = {
            "scenario": scenario['name'],
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": {
                "message_processed": 0,
                "orders_executed": 0,
                "latency_stats": [],
                "error_count": 0
            }
        }

        # 启动测试
        self.running = True
        params = scenario['params']

        # 创建测试线程
        if 'tick_rate' in params:
            tick_thread = threading.Thread(
                target=self._generate_market_data,
                args=(params['tick_rate'], params['duration'], params['symbols'], result)
            )
            self.threads.append(tick_thread)
            tick_thread.start()

        if 'order_rate' in params:
            order_thread = threading.Thread(
                target=self._generate_orders,
                args=(params['order_rate'], params['duration'], params['symbols'], result)
            )
            self.threads.append(order_thread)
            order_thread.start()

        # 等待测试完成
        for t in self.threads:
            t.join()

        # 计算统计指标
        result['end_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
        if result['metrics']['latency_stats']:
            result['metrics']['avg_latency'] = sum(result['metrics']['latency_stats']) / len(result['metrics']['latency_stats'])
            result['metrics']['max_latency'] = max(result['metrics']['latency_stats'])
        else:
            result['metrics']['avg_latency'] = 0
            result['metrics']['max_latency'] = 0

        logger.info(f"压力测试场景 {scenario['name']} 完成")
        return result

    def _generate_market_data(self, rate: int, duration: int, symbols: List[str], result: Dict[str, Any]):
        """
        生成模拟市场数据
        :param rate: 每秒消息数
        :param duration: 持续时间(秒)
        :param symbols: 股票代码列表
        :param result: 测试结果字典
        """
        start_time = time.time()
        interval = 1.0 / rate

        while self.running and (time.time() - start_time) < duration:
            try:
                # 生成模拟Level2行情数据
                tick_data = self.data_adapter.generate_mock_level2(
                    symbol=random.choice(symbols),
                    price=random.uniform(100, 200),
                    volume=random.randint(100, 10000)
                )

                # 记录处理开始时间
                process_start = time.time()

                # 发送到实时引擎
                self.engine.process_market_data(tick_data)

                # 记录延迟
                latency = (time.time() - process_start) * 1000  # 转换为毫秒
                result['metrics']['latency_stats'].append(latency)
                result['metrics']['message_processed'] += 1

                # 控制发送速率
                time.sleep(interval)

            except Exception as e:
                logger.error(f"生成市场数据异常: {str(e)}")
                result['metrics']['error_count'] += 1

    def _generate_orders(self, rate: int, duration: int, symbols: List[str], result: Dict[str, Any]):
        """
        生成模拟订单
        :param rate: 每秒订单数
        :param duration: 持续时间(秒)
        :param symbols: 股票代码列表
        :param result: 测试结果字典
        """
        start_time = time.time()
        interval = 1.0 / rate

        while self.running and (time.time() - start_time) < duration:
            try:
                # 生成模拟订单
                order = {
                    "symbol": random.choice(symbols),
                    "side": random.choice(["buy", "sell"]),
                    "price": random.uniform(100, 200),
                    "quantity": random.randint(100, 1000),
                    "type": "LIMIT"
                }

                # 记录处理开始时间
                process_start = time.time()

                # 发送到交易引擎
                self.engine.process_order(order)

                # 记录延迟
                latency = (time.time() - process_start) * 1000  # 转换为毫秒
                result['metrics']['latency_stats'].append(latency)
                result['metrics']['orders_executed'] += 1

                # 控制发送速率
                time.sleep(interval)

            except Exception as e:
                logger.error(f"生成订单异常: {str(e)}")
                result['metrics']['error_count'] += 1

    def stop(self):
        """停止压力测试"""
        self.running = False
        for t in self.threads:
            t.join()
        logger.info("压力测试已停止")

    def run_all_scenarios(self) -> List[Dict[str, Any]]:
        """
        运行所有测试场景
        :return: 所有测试结果列表
        """
        results = []
        for scenario in self.test_scenarios:
            try:
                result = self.run_scenario(scenario['name'])
                results.append(result)
                # 场景之间暂停10秒
                time.sleep(10)
            except Exception as e:
                logger.error(f"运行场景 {scenario['name']} 失败: {str(e)}")
                continue
        return results
