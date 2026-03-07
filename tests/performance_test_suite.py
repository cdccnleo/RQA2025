#!/usr/bin/env python3
"""
性能测试脚本

测试RQA2025量化交易系统的性能指标和极限承载能力
包括：
1. 策略信号生成性能
2. 批量策略执行性能
3. 回测引擎性能
4. 数据处理性能
5. 系统并发处理能力
6. 系统极限承载能力
"""

import pytest
import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.strategy.core.strategy_service import UnifiedStrategyService as StrategyService
from src.data.core.data_loader import FileDataLoader as DataLoader
from src.strategy.backtest.backtest_engine import BacktestEngine
from src.features.core.feature_engineer import FeatureEngineer
from src.ml.core.ml_service import MLService


class TestPerformance:
    """
    性能测试类
    """
    
    @pytest.fixture(autouse=True)
    def setup_test(self):
        """
        测试环境设置
        """
        # 初始化各服务
        self.strategy_service = StrategyService()
        self.data_loader = DataLoader()
        self.backtest_engine = BacktestEngine()
        self.feature_engineer = FeatureEngineer()
        self.ml_service = MLService()
        
        # 测试数据路径
        self.test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        
        # 性能测试配置
        self.performance_config = {
            'signal_generation_target': 1,  # ms
            'batch_execution_target': 100,  # ms
            'backtest_speed_target': 10,  # seconds for 10M data
            'data_processing_target': 1000,  # records per second
            'concurrency_target': 100,  # concurrent requests
            'throughput_target': 1000  # TPS
        }
    
    def test_signal_generation_performance(self):
        """
        测试策略信号生成性能
        目标：单策略信号生成 < 1ms
        """
        # 创建测试策略
        from src.strategy.interfaces.strategy_interfaces import StrategyConfig, StrategyType
        import uuid
        
        strategy_id = str(uuid.uuid4())
        strategy_config = StrategyConfig(
            strategy_id=strategy_id,
            strategy_name='Test Performance Strategy',
            strategy_type=StrategyType.MOMENTUM,
            parameters={
                'lookback_period': 20,
                'momentum_threshold': 0.05
            },
            symbols=['AAPL'],
            risk_limits={}
        )
        
        self.strategy_service.create_strategy(strategy_config)
        
        # 准备测试数据
        market_data_file = os.path.join(self.test_data_dir, 'market_data', 'normal', 'AAPL_normal.csv')
        market_data = pd.read_csv(market_data_file, parse_dates=['date'])
        
        # 转换为适合execute_strategy的格式
        market_data_dict = {
            'AAPL': market_data.tail(20).to_dict('records')
        }
        
        # 测试信号生成性能
        start_time = time.perf_counter()
        
        # 执行多次以获得准确的性能数据
        num_trials = 1000
        for _ in range(num_trials):
            # 模拟信号生成
            result = self.strategy_service.execute_strategy(strategy_id, market_data_dict)
        
        end_time = time.perf_counter()
        
        # 计算平均执行时间
        avg_time_ms = (end_time - start_time) * 1000 / num_trials
        print(f"Signal generation average time: {avg_time_ms:.3f} ms")
        
        # 验证性能指标
        assert avg_time_ms < self.performance_config['signal_generation_target'], \
            f"Signal generation performance not met: {avg_time_ms:.3f} ms > {self.performance_config['signal_generation_target']} ms"
        
        # 清理
        self.strategy_service.delete_strategy(strategy_id)
    
    def test_batch_execution_performance(self):
        """
        测试批量策略执行性能
        目标：批量策略执行 < 100ms
        """
        # 创建多个测试策略
        from src.strategy.interfaces.strategy_interfaces import StrategyConfig, StrategyType
        import uuid
        
        strategies = []
        for i in range(10):
            strategy_id = str(uuid.uuid4())
            strategy_config = StrategyConfig(
                strategy_id=strategy_id,
                strategy_name=f'Test Strategy {i}',
                strategy_type=StrategyType.MOMENTUM,
                parameters={
                    'lookback_period': 20 + i,
                    'momentum_threshold': 0.05
                },
                symbols=['AAPL'],
                risk_limits={}
            )
            self.strategy_service.create_strategy(strategy_config)
            strategies.append(strategy_id)
        
        # 准备测试数据
        market_data_file = os.path.join(self.test_data_dir, 'market_data', 'normal', 'AAPL_normal.csv')
        market_data = pd.read_csv(market_data_file, parse_dates=['date'])
        
        # 转换为适合execute_strategy的格式
        market_data_dict = {
            'AAPL': market_data.tail(30).to_dict('records')
        }
        
        # 测试批量执行性能（使用循环执行多个策略）
        start_time = time.perf_counter()
        
        # 循环执行多个策略
        batch_results = []
        for strategy_id in strategies:
            result = self.strategy_service.execute_strategy(strategy_id, market_data_dict)
            batch_results.append(result)
        
        end_time = time.perf_counter()
        
        # 计算执行时间
        execution_time_ms = (end_time - start_time) * 1000
        print(f"Batch execution time: {execution_time_ms:.3f} ms")
        
        # 验证性能指标
        assert execution_time_ms < self.performance_config['batch_execution_target'], \
            f"Batch execution performance not met: {execution_time_ms:.3f} ms > {self.performance_config['batch_execution_target']} ms"
        
        # 清理
        for strategy_id in strategies:
            self.strategy_service.delete_strategy(strategy_id)
    
    def test_backtest_performance(self):
        """
        测试回测引擎性能
        目标：1000万条历史数据回测 < 10秒
        """
        # 生成大规模测试数据（模拟1000万条数据）
        def generate_large_test_data():
            dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(252 * 5)]  # 5年数据
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
            
            data = []
            for symbol in symbols:
                # 生成价格数据
                np.random.seed(42 + hash(symbol) % 1000)
                returns = np.random.normal(0.001, 0.02, len(dates))
                prices = 100 * np.exp(np.cumsum(returns))
                
                for i, date in enumerate(dates):
                    data.append({
                        'symbol': symbol,
                        'date': date.strftime('%Y-%m-%d'),
                        'open': prices[i] * 1.001,
                        'high': prices[i] * 1.02,
                        'low': prices[i] * 0.98,
                        'close': prices[i],
                        'volume': 1000000,
                        'returns': returns[i]
                    })
            
            return pd.DataFrame(data)
        
        large_data = generate_large_test_data()
        print(f"Generated test data with {len(large_data)} records")
        
        # 测试策略配置
        strategy_config = {
            'name': 'Test Backtest Strategy',
            'type': 'momentum',
            'parameters': {
                'lookback_period': 20,
                'threshold': 0.05
            }
        }
        
        # 测试回测性能
        start_time = time.perf_counter()
        
        # 使用正确的方法签名
        backtest_result = self.backtest_engine.run_backtest(
            strategy_config, large_data
        )
        
        end_time = time.perf_counter()
        
        # 计算执行时间
        execution_time_seconds = end_time - start_time
        print(f"Backtest execution time: {execution_time_seconds:.2f} seconds")
        
        # 估算1000万条数据的回测时间
        # 当前数据量: len(large_data) ~ 6300 records
        # 1000万条数据的估算时间 = execution_time_seconds * (10000000 / len(large_data))
        estimated_large_scale_time = execution_time_seconds * (10000000 / len(large_data))
        print(f"Estimated time for 10M records: {estimated_large_scale_time:.2f} seconds")
        
        # 验证性能指标
        assert estimated_large_scale_time < self.performance_config['backtest_speed_target'], \
            f"Backtest performance not met: {estimated_large_scale_time:.2f} seconds > {self.performance_config['backtest_speed_target']} seconds"
    
    def test_data_processing_performance(self):
        """
        测试数据处理性能
        目标：数据处理吞吐量 > 1000 条/秒
        """
        # 加载测试数据
        market_data_file = os.path.join(self.test_data_dir, 'market_data', 'normal', 'AAPL_normal.csv')
        market_data = pd.read_csv(market_data_file)
        
        # 复制数据以增加数据量
        large_data = pd.concat([market_data] * 100, ignore_index=True)
        print(f"Processing data with {len(large_data)} records")
        
        # 测试数据处理性能
        start_time = time.perf_counter()
        
        # 使用策略服务的数据准备方法
        processed_data = self.strategy_service.prepare_market_data(
            large_data, {'clean_data': True, 'transform_data': True}
        )
        
        end_time = time.perf_counter()
        
        # 计算处理速度
        processing_time_seconds = end_time - start_time
        records_per_second = len(large_data) / processing_time_seconds
        print(f"Data processing speed: {records_per_second:.2f} records per second")
        
        # 验证性能指标
        assert records_per_second > self.performance_config['data_processing_target'], \
            f"Data processing performance not met: {records_per_second:.2f} records/s < {self.performance_config['data_processing_target']} records/s"
    
    def test_concurrency_performance(self):
        """
        测试系统并发处理能力
        目标：支持 100 并发请求
        """
        # 创建测试策略
        from src.strategy.interfaces.strategy_interfaces import StrategyConfig, StrategyType
        import uuid
        
        strategy_id = str(uuid.uuid4())
        strategy_config = StrategyConfig(
            strategy_id=strategy_id,
            strategy_name='Test Concurrency Strategy',
            strategy_type=StrategyType.MOMENTUM,
            parameters={
                'lookback_period': 20,
                'momentum_threshold': 0.05
            },
            symbols=['AAPL'],
            risk_limits={}
        )
        
        self.strategy_service.create_strategy(strategy_config)
        
        # 准备测试数据
        market_data_file = os.path.join(self.test_data_dir, 'market_data', 'normal', 'AAPL_normal.csv')
        market_data = pd.read_csv(market_data_file, parse_dates=['date'])
        
        # 转换为适合execute_strategy的格式
        test_data = {
            'AAPL': market_data.tail(20).to_dict('records')
        }
        
        # 测试并发处理
        def execute_concurrent_request():
            # 模拟策略执行
            result = self.strategy_service.execute_strategy(strategy_id, test_data)
            return result
        
        # 测试不同并发级别
        concurrency_levels = [10, 50, 100, 200]
        
        for concurrency in concurrency_levels:
            print(f"Testing concurrency level: {concurrency}")
            
            start_time = time.perf_counter()
            
            # 执行并发请求
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(execute_concurrent_request) for _ in range(concurrency)]
                
                # 等待所有请求完成
                for future in as_completed(futures):
                    result = future.result()
            
            end_time = time.perf_counter()
            
            # 计算性能指标
            execution_time = end_time - start_time
            throughput = concurrency / execution_time  # TPS
            
            print(f"Concurrency: {concurrency}, Time: {execution_time:.2f}s, Throughput: {throughput:.2f} TPS")
            
            # 验证目标并发级别
            if concurrency <= self.performance_config['concurrency_target']:
                # 确保系统能够处理目标并发级别
                assert execution_time < concurrency * 0.1, "每个请求平均响应时间 < 100ms"
        
        # 清理
        self.strategy_service.delete_strategy(strategy_id)
    
    def test_stress_test(self):
        """
        测试系统极限承载能力
        目标：系统吞吐量 > 1000 TPS
        """
        # 创建测试策略
        from src.strategy.interfaces.strategy_interfaces import StrategyConfig, StrategyType
        import uuid
        
        strategy_id = str(uuid.uuid4())
        strategy_config = StrategyConfig(
            strategy_id=strategy_id,
            strategy_name='Test Stress Strategy',
            strategy_type=StrategyType.MOMENTUM,
            parameters={
                'lookback_period': 20,
                'momentum_threshold': 0.05
            },
            symbols=['AAPL'],
            risk_limits={}
        )
        
        self.strategy_service.create_strategy(strategy_config)
        
        # 准备测试数据
        market_data_file = os.path.join(self.test_data_dir, 'market_data', 'normal', 'AAPL_normal.csv')
        market_data = pd.read_csv(market_data_file, parse_dates=['date'])
        
        # 转换为适合execute_strategy的格式
        test_data = {
            'AAPL': market_data.tail(20).to_dict('records')
        }
        
        # 压力测试函数
        def stress_test_request():
            try:
                # 模拟策略执行
                result = self.strategy_service.execute_strategy(strategy_id, test_data)
                return True
            except Exception as e:
                print(f"Error during stress test: {e}")
                return False
        
        # 执行压力测试
        total_requests = 10000
        concurrency = 100
        
        print(f"Starting stress test: {total_requests} requests with {concurrency} concurrency")
        
        start_time = time.perf_counter()
        
        successful_requests = 0
        
        # 分批执行以避免系统过载
        batch_size = 1000
        for batch in range(0, total_requests, batch_size):
            current_batch_size = min(batch_size, total_requests - batch)
            
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(stress_test_request) for _ in range(current_batch_size)]
                
                for future in as_completed(futures):
                    if future.result():
                        successful_requests += 1
            
            print(f"Completed batch {batch//batch_size + 1}/{total_requests//batch_size}, Successful: {successful_requests}")
        
        end_time = time.perf_counter()
        
        # 计算性能指标
        execution_time = end_time - start_time
        throughput = total_requests / execution_time  # TPS
        success_rate = successful_requests / total_requests * 100
        
        print(f"Stress test results:")
        print(f"Total requests: {total_requests}")
        print(f"Successful requests: {successful_requests}")
        print(f"Success rate: {success_rate:.2f}%")
        print(f"Execution time: {execution_time:.2f}s")
        print(f"Throughput: {throughput:.2f} TPS")
        
        # 验证性能指标
        assert throughput > self.performance_config['throughput_target'], \
            f"Throughput performance not met: {throughput:.2f} TPS < {self.performance_config['throughput_target']} TPS"
        
        assert success_rate > 99.0, \
            f"Success rate too low: {success_rate:.2f}% < 99.0%"
        
        # 清理
        self.strategy_service.delete_strategy(strategy_id)


if __name__ == '__main__':
    # 运行性能测试
    pytest.main(['-v', __file__])
