#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略服务层PostgreSQL持久化测试

测试内容:
1. StrategyPersistence PostgreSQL存储测试
2. BacktestPersistence PostgreSQL存储测试
3. 降级机制测试
4. 性能测试
"""

import os
import sys
import json
import time
import tempfile
import shutil
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class MockPostgresConnection:
    """模拟PostgreSQL连接"""
    
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.closed = False
        self._cursor = None
        self._data_store = {}
        self._committed = False
    
    def cursor(self, cursor_factory=None):
        if self.should_fail:
            raise Exception("Connection failed")
        self._cursor = MockCursor(self._data_store)
        return self._cursor
    
    def commit(self):
        if self.should_fail:
            raise Exception("Commit failed")
        self._committed = True
    
    def rollback(self):
        self._committed = False
    
    def close(self):
        self.closed = True
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class MockCursor:
    """模拟PostgreSQL游标"""
    
    def __init__(self, data_store):
        self._data_store = data_store
        self._results = []
        self._rowcount = 0
        self._last_query = None
    
    def execute(self, query, params=None):
        self._last_query = query
        self._results = []
        
        if "SELECT 1" in query:
            self._results = [(1,)]
        elif "SELECT COUNT" in query:
            self._results = [(len(self._data_store),)]
        elif "SELECT" in query and "FROM strategies" in query:
            for key, value in self._data_store.items():
                self._results.append(value)
        elif "SELECT" in query and "FROM backtest" in query:
            for key, value in self._data_store.items():
                self._results.append(value)
        
        return self
    
    def fetchone(self):
        if self._results:
            return self._results[0]
        return None
    
    def fetchall(self):
        return self._results
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class TestStrategyPersistencePostgreSQL(unittest.TestCase):
    """StrategyPersistence PostgreSQL存储测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """清理测试环境"""
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except PermissionError:
                pass
    
    def test_init_with_postgresql_disabled(self):
        """测试PostgreSQL禁用时的初始化"""
        from src.strategy.persistence.strategy_persistence import StrategyPersistence
        
        persistence = StrategyPersistence(
            storage_path=self.temp_dir,
            enable_postgresql=False
        )
        
        self.assertFalse(persistence._pg_available)
    
    def test_save_and_load_strategy(self):
        """测试保存和加载策略"""
        from src.strategy.persistence.strategy_persistence import StrategyPersistence
        
        persistence = StrategyPersistence(
            storage_path=self.temp_dir,
            enable_postgresql=False
        )
        
        strategy_data = {
            'strategy_name': 'test_strategy',
            'strategy_type': 'trend_following',
            'version': '1.0.0',
            'status': 'active',
            'description': 'Test strategy',
            'config': {'param1': 10, 'param2': 20}
        }
        
        success = persistence.save_strategy('test_strategy_001', strategy_data)
        self.assertTrue(success)
        
        loaded = persistence.load_strategy('test_strategy_001')
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded['strategy_name'], 'test_strategy')
    
    def test_list_strategies(self):
        """测试列出策略"""
        from src.strategy.persistence.strategy_persistence import StrategyPersistence
        
        persistence = StrategyPersistence(
            storage_path=self.temp_dir,
            enable_postgresql=False
        )
        
        persistence.save_strategy('list_test_001', {
            'strategy_name': 'strategy_1',
            'strategy_type': 'trend_following'
        })
        
        persistence.save_strategy('list_test_002', {
            'strategy_name': 'strategy_2',
            'strategy_type': 'mean_reversion'
        })
        
        strategies = persistence.list_strategies()
        self.assertGreaterEqual(len(strategies), 2)
    
    def test_delete_strategy(self):
        """测试删除策略"""
        from src.strategy.persistence.strategy_persistence import StrategyPersistence
        
        persistence = StrategyPersistence(
            storage_path=self.temp_dir,
            enable_postgresql=False
        )
        
        persistence.save_strategy('delete_test_001', {
            'strategy_name': 'delete_test',
            'strategy_type': 'trend_following'
        })
        
        success = persistence.delete_strategy('delete_test_001')
        self.assertTrue(success)
        
        loaded = persistence.load_strategy('delete_test_001')
        self.assertIsNone(loaded)
    
    def test_save_strategy_config(self):
        """测试保存策略配置"""
        from src.strategy.persistence.strategy_persistence import StrategyPersistence
        
        persistence = StrategyPersistence(
            storage_path=self.temp_dir,
            enable_postgresql=False
        )
        
        config = {
            'capital': 100000,
            'risk_per_trade': 0.02,
            'max_positions': 5
        }
        
        success = persistence.save_strategy_config('config_test_001', config)
        self.assertTrue(success)
        
        loaded = persistence.load_strategy_config('config_test_001')
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded['capital'], 100000)
    
    def test_get_storage_stats(self):
        """测试获取存储统计"""
        from src.strategy.persistence.strategy_persistence import StrategyPersistence
        
        persistence = StrategyPersistence(
            storage_path=self.temp_dir,
            enable_postgresql=False
        )
        
        persistence.save_strategy('stats_test_001', {
            'strategy_name': 'stats_test',
            'strategy_type': 'trend_following'
        })
        
        stats = persistence.get_storage_stats()
        
        self.assertIn('postgresql_available', stats)
        self.assertIn('cache_size', stats)
        self.assertFalse(stats['postgresql_available'])


class TestBacktestPersistencePostgreSQL(unittest.TestCase):
    """BacktestPersistence PostgreSQL存储测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """清理测试环境"""
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except PermissionError:
                pass
    
    def test_init_with_postgresql_disabled(self):
        """测试PostgreSQL禁用时的初始化"""
        from src.strategy.persistence.backtest_persistence import BacktestPersistence
        
        persistence = BacktestPersistence(
            storage_path=self.temp_dir,
            enable_postgresql=False
        )
        
        self.assertFalse(persistence._pg_available)
    
    def test_save_and_load_backtest_config(self):
        """测试保存和加载回测配置"""
        from src.strategy.persistence.backtest_persistence import BacktestPersistence
        from src.strategy.interfaces.backtest_interfaces import BacktestConfig, BacktestMode
        
        persistence = BacktestPersistence(
            storage_path=self.temp_dir,
            enable_postgresql=False
        )
        
        config = BacktestConfig(
            backtest_id='backtest_config_001',
            strategy_id='strategy_001',
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=100000.0,
            commission=0.0003,
            slippage=0.0001,
            benchmark_symbol='000300.SH',
            data_frequency='1d',
            mode=BacktestMode.SINGLE,
            parameters={'window': 20},
            risk_limits={'max_drawdown': 0.2}
        )
        
        success = persistence.save_backtest_config(config)
        self.assertTrue(success)
        
        loaded = persistence.load_backtest_config('backtest_config_001')
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.strategy_id, 'strategy_001')
        self.assertEqual(loaded.initial_capital, 100000.0)
    
    def test_save_and_load_backtest_result(self):
        """测试保存和加载回测结果"""
        from src.strategy.persistence.backtest_persistence import BacktestPersistence
        from src.strategy.interfaces.backtest_interfaces import BacktestResult, BacktestStatus
        
        persistence = BacktestPersistence(
            storage_path=self.temp_dir,
            enable_postgresql=False
        )
        
        result = BacktestResult(
            backtest_id='backtest_result_001',
            strategy_id='strategy_001',
            status=BacktestStatus.COMPLETED,
            execution_time=5.5,
            start_time=datetime(2023, 1, 1),
            end_time=datetime(2023, 12, 31),
            returns=pd.Series([0.01, 0.02, -0.01, 0.03]),
            positions=pd.DataFrame({'symbol': ['AAPL', 'MSFT'], 'quantity': [100, 200]}),
            trades=pd.DataFrame({'symbol': ['AAPL'], 'side': ['buy'], 'price': [150.0]}),
            metrics={'total_return': 0.15, 'sharpe_ratio': 1.5},
            risk_metrics={'max_drawdown': 0.1}
        )
        
        success = persistence.save_backtest_result(result)
        self.assertTrue(success)
        
        loaded = persistence.load_backtest_result('backtest_result_001')
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.strategy_id, 'strategy_001')
        self.assertEqual(loaded.status, BacktestStatus.COMPLETED)
    
    def test_list_backtests(self):
        """测试列出回测"""
        from src.strategy.persistence.backtest_persistence import BacktestPersistence
        from src.strategy.interfaces.backtest_interfaces import BacktestConfig, BacktestMode
        
        persistence = BacktestPersistence(
            storage_path=self.temp_dir,
            enable_postgresql=False
        )
        
        config1 = BacktestConfig(
            backtest_id='list_backtest_001',
            strategy_id='strategy_001',
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=100000.0,
            mode=BacktestMode.SINGLE
        )
        
        config2 = BacktestConfig(
            backtest_id='list_backtest_002',
            strategy_id='strategy_002',
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=100000.0,
            mode=BacktestMode.SINGLE
        )
        
        persistence.save_backtest_config(config1)
        persistence.save_backtest_config(config2)
        
        backtests = persistence.list_backtests()
        self.assertGreaterEqual(len(backtests), 2)
    
    def test_delete_backtest(self):
        """测试删除回测"""
        from src.strategy.persistence.backtest_persistence import BacktestPersistence
        from src.strategy.interfaces.backtest_interfaces import BacktestConfig, BacktestMode
        
        persistence = BacktestPersistence(
            storage_path=self.temp_dir,
            enable_postgresql=False
        )
        
        config = BacktestConfig(
            backtest_id='delete_backtest_001',
            strategy_id='strategy_001',
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=100000.0,
            mode=BacktestMode.SINGLE
        )
        
        persistence.save_backtest_config(config)
        
        success = persistence.delete_backtest('delete_backtest_001')
        self.assertTrue(success)
        
        loaded = persistence.load_backtest_config('delete_backtest_001')
        self.assertIsNone(loaded)
    
    def test_get_storage_stats(self):
        """测试获取存储统计"""
        from src.strategy.persistence.backtest_persistence import BacktestPersistence
        from src.strategy.interfaces.backtest_interfaces import BacktestConfig, BacktestMode
        
        persistence = BacktestPersistence(
            storage_path=self.temp_dir,
            enable_postgresql=False
        )
        
        config = BacktestConfig(
            backtest_id='stats_backtest_001',
            strategy_id='strategy_001',
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=100000.0,
            mode=BacktestMode.SINGLE
        )
        
        persistence.save_backtest_config(config)
        
        stats = persistence.get_storage_stats()
        
        self.assertIn('postgresql_available', stats)
        self.assertIn('configs_cache_size', stats)
        self.assertFalse(stats['postgresql_available'])


class TestPostgreSQLFailover(unittest.TestCase):
    """PostgreSQL故障转移测试"""
    
    def test_strategy_persistence_fallback_on_pg_failure(self):
        """测试StrategyPersistence在PostgreSQL失败时降级"""
        from src.strategy.persistence.strategy_persistence import StrategyPersistence
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            with patch.object(StrategyPersistence, '_test_postgresql_connection', return_value=False):
                persistence = StrategyPersistence(
                    storage_path=temp_dir,
                    enable_postgresql=True
                )
                
                self.assertFalse(persistence._pg_available)
                
                success = persistence.save_strategy('fallback_test', {
                    'strategy_name': 'fallback_test',
                    'strategy_type': 'trend_following'
                })
                
                self.assertTrue(success)
        finally:
            try:
                shutil.rmtree(temp_dir)
            except PermissionError:
                pass
    
    def test_backtest_persistence_fallback_on_pg_failure(self):
        """测试BacktestPersistence在PostgreSQL失败时降级"""
        from src.strategy.persistence.backtest_persistence import BacktestPersistence
        from src.strategy.interfaces.backtest_interfaces import BacktestConfig, BacktestMode
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            with patch.object(BacktestPersistence, '_test_postgresql_connection', return_value=False):
                persistence = BacktestPersistence(
                    storage_path=temp_dir,
                    enable_postgresql=True
                )
                
                self.assertFalse(persistence._pg_available)
                
                config = BacktestConfig(
                    backtest_id='fallback_backtest',
                    strategy_id='strategy_001',
                    start_date=datetime(2023, 1, 1),
                    end_date=datetime(2023, 12, 31),
                    initial_capital=100000.0,
                    mode=BacktestMode.SINGLE
                )
                
                success = persistence.save_backtest_config(config)
                self.assertTrue(success)
        finally:
            try:
                shutil.rmtree(temp_dir)
            except PermissionError:
                pass


class TestPerformance(unittest.TestCase):
    """性能测试"""
    
    def test_strategy_save_load_performance(self):
        """测试策略保存加载性能"""
        from src.strategy.persistence.strategy_persistence import StrategyPersistence
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            persistence = StrategyPersistence(
                storage_path=temp_dir,
                enable_postgresql=False
            )
            
            strategy_data = {
                'strategy_name': 'performance_test',
                'strategy_type': 'trend_following',
                'config': {f'param_{i}': i for i in range(100)}
            }
            
            start = time.time()
            persistence.save_strategy('perf_test_strategy', strategy_data)
            save_time = time.time() - start
            
            start = time.time()
            loaded = persistence.load_strategy('perf_test_strategy')
            load_time = time.time() - start
            
            self.assertIsNotNone(loaded)
            self.assertLess(save_time, 1.0, "策略保存时间应小于1秒")
            self.assertLess(load_time, 0.5, "策略加载时间应小于0.5秒")
            
            print(f"\n策略保存时间: {save_time:.3f}s")
            print(f"策略加载时间: {load_time:.3f}s")
            
        finally:
            try:
                shutil.rmtree(temp_dir)
            except PermissionError:
                pass
    
    def test_backtest_result_save_load_performance(self):
        """测试回测结果保存加载性能"""
        from src.strategy.persistence.backtest_persistence import BacktestPersistence
        from src.strategy.interfaces.backtest_interfaces import BacktestResult, BacktestStatus
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            persistence = BacktestPersistence(
                storage_path=temp_dir,
                enable_postgresql=False
            )
            
            result = BacktestResult(
                backtest_id='perf_backtest_result',
                strategy_id='strategy_001',
                status=BacktestStatus.COMPLETED,
                execution_time=5.5,
                start_time=datetime(2023, 1, 1),
                end_time=datetime(2023, 12, 31),
                returns=pd.Series(np.random.randn(1000)),
                positions=pd.DataFrame(np.random.randn(100, 10)),
                trades=pd.DataFrame({
                    'symbol': ['AAPL'] * 500,
                    'side': ['buy'] * 500,
                    'price': np.random.randn(500) * 100 + 150
                }),
                metrics={'total_return': 0.15},
                risk_metrics={'max_drawdown': 0.1}
            )
            
            start = time.time()
            persistence.save_backtest_result(result)
            save_time = time.time() - start
            
            start = time.time()
            loaded = persistence.load_backtest_result('perf_backtest_result')
            load_time = time.time() - start
            
            self.assertIsNotNone(loaded)
            self.assertLess(save_time, 2.0, "回测结果保存时间应小于2秒")
            self.assertLess(load_time, 1.0, "回测结果加载时间应小于1秒")
            
            print(f"\n回测结果保存时间: {save_time:.3f}s")
            print(f"回测结果加载时间: {load_time:.3f}s")
            
        finally:
            try:
                shutil.rmtree(temp_dir)
            except PermissionError:
                pass


def run_tests():
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestStrategyPersistencePostgreSQL))
    suite.addTests(loader.loadTestsFromTestCase(TestBacktestPersistencePostgreSQL))
    suite.addTests(loader.loadTestsFromTestCase(TestPostgreSQLFailover))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    run_tests()
