# -*- coding: utf-8 -*-
"""
数据源管理器Mock测试
测试数据源连接、数据获取和管理功能
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
import asyncio


class MockDataSource:
    """模拟数据源"""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.is_available = False
        self.connection_count = 0
        self.fetch_count = 0
        self.error_count = 0
        self.logger = Mock()

    def check_availability(self) -> bool:
        """检查数据源可用性"""
        try:
            # 模拟连接检查
            self.connection_count += 1

            # 模拟某些数据源不可用
            if "unavailable" in self.name.lower():
                self.is_available = False
                return False

            # 模拟网络延迟
            import time
            time.sleep(0.001)  # 1ms延迟

            self.is_available = True
            return True
        except Exception as e:
            self.logger.error(f"Availability check failed for {self.name}: {e}")
            self.is_available = False
            return False

    def fetch_data(self, symbol: str, start_date: str, end_date: str,
                   interval: str = '1d') -> pd.DataFrame:
        """获取数据"""
        self.fetch_count += 1

        try:
            if not self.is_available:
                raise Exception(f"Data source {self.name} is not available")

            # 模拟数据获取
            return self._generate_mock_data(symbol, start_date, end_date, interval)
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Failed to fetch data from {self.name}: {e}")
            raise

    def _generate_mock_data(self, symbol: str, start_date: str, end_date: str,
                           interval: str = '1d') -> pd.DataFrame:
        """生成模拟数据"""
        # 解析日期
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        # 根据interval生成数据点
        if interval == '1d':
            # 日线数据
            dates = pd.date_range(start=start, end=end, freq='D')
        elif interval == '1h':
            # 小时数据
            dates = pd.date_range(start=start, end=end, freq='h')
        elif interval == '1m':
            # 分钟数据（限制数量）
            dates = pd.date_range(start=start, end=end, freq='T')[:100]  # 最多100条
        else:
            dates = pd.date_range(start=start, end=end, freq='D')

        # 生成模拟价格数据
        import numpy as np
        np.random.seed(abs(hash(symbol)) % (2**32))  # 保证可重复性，限制在有效范围内

        n_points = len(dates)
        if n_points == 0:
            return pd.DataFrame()

        # 基础价格（根据symbol调整）
        base_price = 100.0
        if "AAPL" in symbol:
            base_price = 150.0
        elif "GOOGL" in symbol:
            base_price = 2500.0
        elif "BTC" in symbol:
            base_price = 50000.0
        elif "ETH" in symbol:
            base_price = 3000.0

        # 生成价格序列
        price_changes = np.random.normal(0, 0.02, n_points)  # 正态分布波动
        prices = base_price * (1 + np.cumsum(price_changes))

        # 生成OHLCV数据
        high_multipliers = 1 + np.abs(np.random.normal(0, 0.01, n_points))
        low_multipliers = 1 - np.abs(np.random.normal(0, 0.01, n_points))
        volume_base = 1000000 if "STOCK" in symbol.upper() else 100000

        data = []
        for i, date in enumerate(dates):
            close_price = prices[i]
            high_price = close_price * high_multipliers[i]
            low_price = close_price * low_multipliers[i]
            open_price = prices[i-1] if i > 0 else close_price * (1 + np.random.normal(0, 0.005))
            volume = int(volume_base * (1 + np.random.normal(0, 0.5)))

            data.append({
                'timestamp': date,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': volume,
                'symbol': symbol,
                'source': self.name
            })

        return pd.DataFrame(data)

    def get_info(self) -> Dict[str, Any]:
        """获取数据源信息"""
        return {
            'name': self.name,
            'available': self.is_available,
            'config': self.config,
            'connection_count': self.connection_count,
            'fetch_count': self.fetch_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(1, self.fetch_count)
        }


class MockDatabaseDataSource(MockDataSource):
    """模拟数据库数据源"""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.connection_pool = Mock()
        self.query_count = 0

    def check_availability(self) -> bool:
        """检查数据库连接"""
        try:
            # 模拟数据库连接检查
            self.connection_count += 1

            # 检查配置
            required_config = ['host', 'port', 'database', 'username']
            if not all(key in self.config for key in required_config):
                self.is_available = False
                return False

            # 模拟连接池检查
            self.connection_pool.get_connection = Mock(return_value=Mock())
            self.is_available = True
            return True
        except Exception:
            self.is_available = False
            return False

    def fetch_data(self, symbol: str, start_date: str, end_date: str,
                   interval: str = '1d') -> pd.DataFrame:
        """从数据库获取数据"""
        self.query_count += 1

        if not self.is_available:
            raise Exception("Database connection not available")

        try:
            # 模拟SQL查询
            query = f"""
            SELECT timestamp, open, high, low, close, volume, symbol
            FROM market_data
            WHERE symbol = '{symbol}'
            AND timestamp BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY timestamp
            """

            self.logger.info(f"Executing query: {query}")

            # 生成模拟数据
            return self._generate_mock_data(symbol, start_date, end_date, interval)
        except Exception as e:
            self.error_count += 1
            raise


class MockAPIDataSource(MockDataSource):
    """模拟API数据源"""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.session = Mock()
        self.request_count = 0
        self.rate_limited = False

    def check_availability(self) -> bool:
        """检查API可用性"""
        try:
            self.connection_count += 1

            # 检查API配置
            if 'api_key' not in self.config:
                self.is_available = False
                return False

            # 模拟API健康检查请求
            self.session.get = Mock()
            self.session.get.return_value.status_code = 200

            self.is_available = True
            return True
        except Exception:
            self.is_available = False
            return False

    def fetch_data(self, symbol: str, start_date: str, end_date: str,
                   interval: str = '1d') -> pd.DataFrame:
        """从API获取数据"""
        self.request_count += 1

        if not self.is_available:
            raise Exception("API not available")

        if self.rate_limited:
            raise Exception("Rate limit exceeded")

        try:
            # 模拟API请求
            url = f"https://api.example.com/data/{symbol}"
            params = {
                'start_date': start_date,
                'end_date': end_date,
                'interval': interval
            }

            # 模拟API调用
            self.logger.info(f"Making API request to {url} with params {params}")

            # 模拟rate limiting（每10次请求限流一次）
            if self.request_count % 10 == 0:
                self.rate_limited = True
                raise Exception("Rate limit exceeded")

            return self._generate_mock_data(symbol, start_date, end_date, interval)
        except Exception as e:
            self.error_count += 1
            raise


class MockDataSourceManager:
    """模拟数据源管理器"""

    def __init__(self):
        self.sources: Dict[str, MockDataSource] = {}
        self.active_source: Optional[str] = None
        self.failover_enabled = True
        self.request_count = 0
        self.success_count = 0
        self.failover_count = 0
        self.logger = Mock()

    def add_source(self, source: MockDataSource) -> bool:
        """添加数据源"""
        if source.name in self.sources:
            return False

        self.sources[source.name] = source
        return True

    def remove_source(self, name: str) -> bool:
        """移除数据源"""
        if name not in self.sources:
            return False

        del self.sources[name]
        if self.active_source == name:
            self.active_source = None
        return True

    def get_available_sources(self) -> List[MockDataSource]:
        """获取可用数据源"""
        return [source for source in self.sources.values() if source.is_available]

    def get_source(self, name: str) -> Optional[MockDataSource]:
        """获取指定数据源"""
        return self.sources.get(name)

    def set_active_source(self, name: str) -> bool:
        """设置活动数据源"""
        if name not in self.sources:
            return False

        self.active_source = name
        return True

    def fetch_data_with_failover(self, symbol: str, start_date: str, end_date: str,
                                interval: str = '1d') -> pd.DataFrame:
        """带故障转移的数据获取"""
        self.request_count += 1

        # 首先尝试活动数据源
        if self.active_source and self.active_source in self.sources:
            try:
                source = self.sources[self.active_source]
                if source.check_availability():
                    data = source.fetch_data(symbol, start_date, end_date, interval)
                    self.success_count += 1
                    return data
            except Exception as e:
                self.logger.warning(f"Active source {self.active_source} failed: {e}")

        # 故障转移到其他可用数据源
        available_sources = self.get_available_sources()
        for source in available_sources:
            if source.name != self.active_source:
                try:
                    self.failover_count += 1
                    self.logger.info(f"Failover to source: {source.name}")
                    data = source.fetch_data(symbol, start_date, end_date, interval)
                    self.success_count += 1
                    # 设置为新的活动数据源
                    self.active_source = source.name
                    return data
                except Exception as e:
                    self.logger.warning(f"Source {source.name} also failed: {e}")
                    continue

        # 所有数据源都失败
        raise Exception("All data sources failed")

    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        available_count = len(self.get_available_sources())
        total_count = len(self.sources)

        return {
            'total_sources': total_count,
            'available_sources': available_count,
            'availability_rate': available_count / max(1, total_count),
            'active_source': self.active_source,
            'request_count': self.request_count,
            'success_count': self.success_count,
            'success_rate': self.success_count / max(1, self.request_count),
            'failover_count': self.failover_count
        }

    def refresh_source_availability(self) -> Dict[str, bool]:
        """刷新所有数据源的可用性"""
        results = {}
        for name, source in self.sources.items():
            results[name] = source.check_availability()

        return results


class TestMockDataSource:
    """模拟数据源测试"""

    def setup_method(self):
        """设置测试方法"""
        self.config = {
            'host': 'localhost',
            'port': 5432,
            'timeout': 30
        }
        self.source = MockDataSource("test_source", self.config)

    def test_source_initialization(self):
        """测试数据源初始化"""
        assert self.source.name == "test_source"
        assert self.source.config == self.config
        assert not self.source.is_available
        assert self.source.connection_count == 0

    def test_source_availability_check(self):
        """测试可用性检查"""
        assert self.source.check_availability()
        assert self.source.is_available
        assert self.source.connection_count == 1

        # 再次检查
        assert self.source.check_availability()
        assert self.source.connection_count == 2

    def test_unavailable_source(self):
        """测试不可用数据源"""
        unavailable_source = MockDataSource("unavailable_test_source")
        assert not unavailable_source.check_availability()
        assert not unavailable_source.is_available

    def test_fetch_data(self):
        """测试数据获取"""
        self.source.check_availability()

        data = self.source.fetch_data("AAPL", "2023-01-01", "2023-01-05", "1d")

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert 'timestamp' in data.columns
        assert 'open' in data.columns
        assert 'high' in data.columns
        assert 'low' in data.columns
        assert 'close' in data.columns
        assert 'volume' in data.columns
        assert 'symbol' in data.columns
        assert 'source' in data.columns

        # 检查symbol
        assert all(data['symbol'] == "AAPL")
        assert all(data['source'] == "test_source")

        assert self.source.fetch_count == 1

    def test_fetch_data_unavailable_source(self):
        """测试从不可用数据源获取数据"""
        with pytest.raises(Exception, match="not available"):
            self.source.fetch_data("AAPL", "2023-01-01", "2023-01-05")

    def test_get_info(self):
        """测试获取信息"""
        info = self.source.get_info()

        assert info['name'] == "test_source"
        assert info['available'] == self.source.is_available
        assert info['config'] == self.config
        assert info['connection_count'] == 0
        assert info['fetch_count'] == 0
        assert info['error_count'] == 0


class TestMockDatabaseDataSource:
    """模拟数据库数据源测试"""

    def setup_method(self):
        """设置测试方法"""
        self.config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'market_data',
            'username': 'user',
            'password': 'pass'
        }
        self.source = MockDatabaseDataSource("db_source", self.config)

    def test_db_source_initialization(self):
        """测试数据库数据源初始化"""
        assert self.source.name == "db_source"
        assert self.source.config == self.config

    def test_db_source_availability_valid_config(self):
        """测试有效配置的数据库可用性"""
        assert self.source.check_availability()
        assert self.source.is_available
        assert self.source.connection_count == 1

    def test_db_source_availability_invalid_config(self):
        """测试无效配置的数据库可用性"""
        invalid_source = MockDatabaseDataSource("invalid_db", {})
        assert not invalid_source.check_availability()
        assert not invalid_source.is_available

    def test_db_source_fetch_data(self):
        """测试数据库数据获取"""
        self.source.check_availability()

        data = self.source.fetch_data("GOOGL", "2023-01-01", "2023-01-03", "1d")

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert all(data['symbol'] == "GOOGL")
        assert all(data['source'] == "db_source")

        assert self.source.query_count == 1


class TestMockAPIDataSource:
    """模拟API数据源测试"""

    def setup_method(self):
        """设置测试方法"""
        self.config = {
            'api_key': 'test_key_123',
            'base_url': 'https://api.example.com',
            'timeout': 30
        }
        self.source = MockAPIDataSource("api_source", self.config)

    def test_api_source_initialization(self):
        """测试API数据源初始化"""
        assert self.source.name == "api_source"
        assert self.source.config == self.config

    def test_api_source_availability_valid_config(self):
        """测试有效配置的API可用性"""
        assert self.source.check_availability()
        assert self.source.is_available

    def test_api_source_availability_invalid_config(self):
        """测试无效配置的API可用性"""
        invalid_source = MockAPIDataSource("invalid_api", {})
        assert not invalid_source.check_availability()
        assert not invalid_source.is_available

    def test_api_source_fetch_data(self):
        """测试API数据获取"""
        self.source.check_availability()

        data = self.source.fetch_data("BTC", "2023-01-01", "2023-01-02", "1h")

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert all(data['symbol'] == "BTC")
        assert all(data['source'] == "api_source")

        assert self.source.request_count == 1

    def test_api_source_rate_limiting(self):
        """测试API限流"""
        self.source.check_availability()

        # 模拟多次请求以触发限流
        for i in range(12):  # 第10次应该触发限流
            if i == 9:  # 第10次请求（索引9）
                with pytest.raises(Exception, match="Rate limit exceeded"):
                    self.source.fetch_data("BTC", "2023-01-01", "2023-01-02")
                break
            else:
                self.source.fetch_data("BTC", "2023-01-01", "2023-01-02")

        assert self.source.rate_limited


class TestMockDataSourceManager:
    """模拟数据源管理器测试"""

    def setup_method(self):
        """设置测试方法"""
        self.manager = MockDataSourceManager()

        # 添加测试数据源
        self.db_source = MockDatabaseDataSource("db_source", {
            'host': 'localhost', 'port': 5432, 'database': 'test', 'username': 'user'
        })
        self.api_source = MockAPIDataSource("api_source", {'api_key': 'test_key'})
        self.backup_source = MockDataSource("backup_source")

        self.manager.add_source(self.db_source)
        self.manager.add_source(self.api_source)
        self.manager.add_source(self.backup_source)

    def test_manager_initialization(self):
        """测试管理器初始化"""
        assert len(self.manager.sources) == 3
        assert self.manager.active_source is None
        assert self.manager.failover_enabled is True

    def test_add_remove_source(self):
        """测试添加和移除数据源"""
        new_source = MockDataSource("new_source")
        assert self.manager.add_source(new_source)
        assert "new_source" in self.manager.sources

        assert self.manager.remove_source("new_source")
        assert "new_source" not in self.manager.sources

        # 移除不存在的数据源
        assert not self.manager.remove_source("nonexistent")

    def test_get_available_sources(self):
        """测试获取可用数据源"""
        # 初始状态下都不可用
        available = self.manager.get_available_sources()
        assert len(available) == 0

        # 使一些数据源可用
        self.db_source.check_availability()
        self.api_source.check_availability()

        available = self.manager.get_available_sources()
        assert len(available) == 2
        source_names = [s.name for s in available]
        assert "db_source" in source_names
        assert "api_source" in source_names

    def test_set_active_source(self):
        """测试设置活动数据源"""
        assert self.manager.set_active_source("db_source")
        assert self.manager.active_source == "db_source"

        # 设置不存在的数据源
        assert not self.manager.set_active_source("nonexistent")

    def test_fetch_data_with_failover(self):
        """测试带故障转移的数据获取"""
        # 使数据源可用
        assert self.db_source.check_availability()
        assert self.api_source.check_availability()
        assert self.backup_source.check_availability()

        # 验证数据源确实可用
        available_sources = self.manager.get_available_sources()
        assert len(available_sources) == 3

        # 设置活动数据源
        assert self.manager.set_active_source("db_source")
        assert self.manager.active_source == "db_source"

        # 正常获取
        data = self.manager.fetch_data_with_failover("AAPL", "2023-01-01", "2023-01-02")
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert self.manager.success_count == 1
        assert self.manager.failover_count == 0

    def test_fetch_data_failover(self):
        """测试故障转移"""
        # 只使backup_source可用，模拟其他数据源失败
        assert self.backup_source.check_availability()

        # 不设置活动数据源，强制故障转移
        data = self.manager.fetch_data_with_failover("AAPL", "2023-01-01", "2023-01-02")
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert self.manager.success_count == 1
        assert self.manager.failover_count >= 1  # 发生了故障转移

    def test_fetch_data_all_sources_fail(self):
        """测试所有数据源都失败的情况"""
        # 所有数据源都不可用
        with pytest.raises(Exception, match="All data sources failed"):
            self.manager.fetch_data_with_failover("AAPL", "2023-01-01", "2023-01-02")

    def test_get_health_status(self):
        """测试获取健康状态"""
        status = self.manager.get_health_status()

        assert status['total_sources'] == 3
        assert status['available_sources'] == 0  # 初始都不可用
        assert status['availability_rate'] == 0.0
        assert status['active_source'] is None
        assert status['request_count'] == 0

        # 使一些数据源可用并执行请求
        assert self.db_source.check_availability()
        assert self.api_source.check_availability()
        data = self.manager.fetch_data_with_failover("AAPL", "2023-01-01", "2023-01-02")
        assert isinstance(data, pd.DataFrame)

        status = self.manager.get_health_status()
        assert status['available_sources'] == 2
        assert status['availability_rate'] == 2/3
        assert status['request_count'] == 1
        assert status['success_count'] == 1
        assert status['success_rate'] == 1.0

    def test_refresh_source_availability(self):
        """测试刷新数据源可用性"""
        results = self.manager.refresh_source_availability()

        assert len(results) == 3
        assert all(isinstance(available, bool) for available in results.values())

        # db_source和api_source应该可用（配置完整）
        assert results["db_source"] is True
        assert results["api_source"] is True
        # backup_source应该可用（MockDataSource总是可用，除非名字包含"unavailable"）


class TestDataSourceIntegration:
    """数据源集成测试"""

    def test_multi_source_data_consistency(self):
        """测试多数据源数据一致性"""
        # 创建两个相同配置的数据源
        source1 = MockDataSource("source1")
        source2 = MockDataSource("source2")

        source1.check_availability()
        source2.check_availability()

        # 使数据源可用
        assert source1.check_availability()
        assert source2.check_availability()

        # 从两个数据源获取相同的数据
        data1 = source1.fetch_data("AAPL", "2023-01-01", "2023-01-03", "1d")
        data2 = source2.fetch_data("AAPL", "2023-01-01", "2023-01-03", "1d")

        # 数据应该具有相同的结构
        assert len(data1) == len(data2)
        assert list(data1.columns) == list(data2.columns)

        # 数据点时间戳应该相同
        assert all(data1['timestamp'] == data2['timestamp'])

    def test_different_intervals(self):
        """测试不同间隔的数据获取"""
        source = MockDataSource("interval_test")
        assert source.check_availability()

        # 测试日线数据
        daily_data = source.fetch_data("AAPL", "2023-01-01", "2023-01-05", "1d")
        assert len(daily_data) <= 5  # 最多5天

        # 测试小时数据
        hourly_data = source.fetch_data("BTC", "2023-01-01T00:00:00", "2023-01-01T23:00:00", "1h")
        assert len(hourly_data) <= 24  # 最多24小时

    def test_source_performance_monitoring(self):
        """测试数据源性能监控"""
        source = MockDataSource("perf_test")

        # 执行一系列操作
        assert source.check_availability()
        source.fetch_data("AAPL", "2023-01-01", "2023-01-02")
        source.fetch_data("GOOGL", "2023-01-01", "2023-01-02")

        # 模拟一次错误
        source.is_available = False
        try:
            source.fetch_data("ERROR", "2023-01-01", "2023-01-02")
        except:
            pass

        info = source.get_info()

        assert info['connection_count'] == 1
        assert info['fetch_count'] == 3  # 2次成功 + 1次失败
        assert info['error_count'] == 1
        assert info['error_rate'] == 1/3

    def test_manager_load_balancing_simulation(self):
        """测试管理器负载均衡模拟"""
        manager = MockDataSourceManager()

        # 添加多个数据源
        for i in range(5):
            source = MockDataSource(f"source_{i}")
            manager.add_source(source)

        # 使所有数据源可用
        results = manager.refresh_source_availability()
        assert all(results.values())  # 所有数据源都应该可用

        # 执行多次请求，观察故障转移行为
        success_count = 0
        for i in range(10):
            try:
                data = manager.fetch_data_with_failover(f"SYMBOL_{i}", "2023-01-01", "2023-01-02")
                assert isinstance(data, pd.DataFrame)
                success_count += 1
            except Exception as e:
                # 如果所有数据源都失败，记录但不中断测试
                print(f"Request {i} failed: {e}")

        # 验证统计信息
        status = manager.get_health_status()
        assert status['request_count'] == 10
        assert status['success_count'] == success_count

    def test_error_recovery_and_resilience(self):
        """测试错误恢复和弹性"""
        manager = MockDataSourceManager()

        # 添加一个总是失败的数据源和一个正常的数据源
        failing_source = MockDataSource("failing_source")
        # 修改failing_source使其总是失败
        failing_source.fetch_data = Mock(side_effect=Exception("Persistent failure"))

        good_source = MockDataSource("good_source")

        manager.add_source(failing_source)
        manager.add_source(good_source)

        # 刷新可用性 - failing_source应该可用（MockDataSource默认可用）
        results = manager.refresh_source_availability()
        assert results["failing_source"] is True
        assert results["good_source"] is True

        # 设置失败的数据源为活动数据源
        assert manager.set_active_source("failing_source")

        # 请求应该自动故障转移到好的数据源
        data = manager.fetch_data_with_failover("AAPL", "2023-01-01", "2023-01-02")

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0

        # 验证故障转移发生
        status = manager.get_health_status()
        assert status['failover_count'] > 0
        assert status['success_count'] == 1

        # 活动数据源应该切换到好的数据源
        assert status['active_source'] == "good_source"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
