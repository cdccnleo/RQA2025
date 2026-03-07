# -*- coding: utf-8 -*-
"""
股票数据加载器Mock测试
测试股票数据的加载、验证和处理功能
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, List, Any, Optional
from pathlib import Path


class MockStockDataLoader:
    """模拟股票数据加载器"""

    def __init__(self, save_path: str, max_retries: int = 3, cache_days: int = 30,
                 frequency: str = 'daily', adjust_type: str = 'none'):
        self.save_path = Path(save_path)
        self.max_retries = max_retries
        self.cache_days = cache_days
        self.frequency = frequency
        self.adjust_type = adjust_type

        # 状态跟踪
        self.load_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.retry_count = 0
        self.errors: List[str] = []

        # 模拟数据存储
        self.cached_data: Dict[str, Any] = {}

        # 日志模拟
        self.logger = Mock()
        self.logger.info = Mock()
        self.logger.warning = Mock()
        self.logger.error = Mock()

    def load(self, symbol: str, start_date: str, end_date: str, **kwargs) -> Optional[Dict[str, Any]]:
        """加载股票数据"""
        # 检查缓存
        cache_key = f"{symbol}_{start_date}_{end_date}_{self.frequency}_{self.adjust_type}"
        if cache_key in self.cached_data:
            cached_result = self.cached_data[cache_key]
            # 检查缓存是否过期
            if not self._is_cache_expired(cached_result):
                self.cache_hits += 1
                return cached_result

        # 缓存未命中，需要加载数据
        self.load_count += 1
        self.cache_misses += 1

        # 模拟数据加载（带重试机制）
        for attempt in range(self.max_retries + 1):
            try:
                data = self._fetch_stock_data(symbol, start_date, end_date, **kwargs)
                if data:
                    # 缓存结果
                    data['_cache_time'] = datetime.now()
                    self.cached_data[cache_key] = data
                    return data
            except Exception as e:
                self.retry_count += 1
                self.errors.append(str(e))
                if attempt < self.max_retries:
                    continue
                else:
                    self.logger.error(f"Failed to load {symbol} after {self.max_retries + 1} attempts: {e}")
                    return None

        return None

    def _fetch_stock_data(self, symbol: str, start_date: str, end_date: str, **kwargs) -> Optional[Dict[str, Any]]:
        """模拟获取股票数据"""
        # 模拟不同的股票数据
        stock_data = {
            "AAPL": {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "data": [
                    {"date": start_date, "open": 150.0, "high": 155.0, "low": 149.0, "close": 152.0, "volume": 1000000},
                    {"date": end_date, "open": 152.0, "high": 158.0, "low": 151.0, "close": 155.0, "volume": 1200000}
                ]
            },
            "GOOGL": {
                "symbol": "GOOGL",
                "name": "Alphabet Inc.",
                "data": [
                    {"date": start_date, "open": 2500.0, "high": 2550.0, "low": 2490.0, "close": 2520.0, "volume": 500000},
                    {"date": end_date, "open": 2520.0, "high": 2580.0, "low": 2510.0, "close": 2550.0, "volume": 600000}
                ]
            },
            "MSFT": {
                "symbol": "MSFT",
                "name": "Microsoft Corporation",
                "data": [
                    {"date": start_date, "open": 300.0, "high": 305.0, "low": 299.0, "close": 302.0, "volume": 800000},
                    {"date": end_date, "open": 302.0, "high": 308.0, "low": 301.0, "close": 305.0, "volume": 900000}
                ]
            }
        }

        # 模拟API错误
        if symbol == "ERROR":
            raise Exception("API rate limit exceeded")

        if symbol == "EMPTY":
            return None

        # 返回股票数据或默认数据
        if symbol in stock_data:
            result = stock_data[symbol].copy()
            result.update({
                "start_date": start_date,
                "end_date": end_date,
                "frequency": self.frequency,
                "adjust_type": self.adjust_type,
                "loaded_at": datetime.now().isoformat(),
                "metadata": {
                    "source": "mock_api",
                    "adjustments": self.adjust_type,
                    "frequency": self.frequency
                }
            })
            return result

        # 默认数据
        return {
            "symbol": symbol,
            "name": f"{symbol} Corporation",
            "data": [
                {"date": start_date, "open": 100.0, "high": 105.0, "low": 99.0, "close": 102.0, "volume": 500000},
                {"date": end_date, "open": 102.0, "high": 108.0, "low": 101.0, "close": 105.0, "volume": 600000}
            ],
            "start_date": start_date,
            "end_date": end_date,
            "frequency": self.frequency,
            "adjust_type": self.adjust_type,
            "loaded_at": datetime.now().isoformat(),
            "metadata": {
                "source": "mock_api",
                "adjustments": self.adjust_type,
                "frequency": self.frequency
            }
        }

    def _is_cache_expired(self, cached_data: Dict[str, Any]) -> bool:
        """检查缓存是否过期"""
        cache_time = cached_data.get('_cache_time')
        if not cache_time:
            return True

        age = datetime.now() - cache_time
        return age.days >= self.cache_days

    def validate(self, data: Any) -> bool:
        """验证股票数据"""
        if not isinstance(data, dict):
            return False

        required_fields = ['symbol', 'data', 'start_date', 'end_date']
        if not all(field in data for field in required_fields):
            return False

        # 检查数据数组
        if not isinstance(data['data'], list) or len(data['data']) == 0:
            return False

        # 检查每条记录的必需字段
        for record in data['data']:
            if not isinstance(record, dict):
                return False
            if not all(field in record for field in ['date', 'open', 'high', 'low', 'close', 'volume']):
                return False

        return True

    def get_metadata(self) -> Dict[str, Any]:
        """获取加载器元数据"""
        return {
            "loader_type": "StockDataLoader",
            "save_path": self.save_path.as_posix(),  # 使用POSIX格式统一路径
            "max_retries": self.max_retries,
            "cache_days": self.cache_days,
            "frequency": self.frequency,
            "adjust_type": self.adjust_type,
            "load_count": self.load_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "retry_count": self.retry_count,
            "error_count": len(self.errors),
            "cache_size": len(self.cached_data)
        }

    def clear_cache(self):
        """清空缓存"""
        self.cached_data.clear()
        # 不重置统计，只重置缓存相关状态

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            "cache_size": len(self.cached_data),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "oldest_cache_age": self._get_oldest_cache_age()
        }

    def _get_oldest_cache_age(self) -> Optional[float]:
        """获取最旧缓存的年龄（小时）"""
        if not self.cached_data:
            return None

        oldest_time = min(data.get('_cache_time') for data in self.cached_data.values()
                         if data.get('_cache_time'))
        if oldest_time:
            age = datetime.now() - oldest_time
            return age.total_seconds() / 3600
        return None


class TestMockStockDataLoader:
    """模拟股票数据加载器测试"""

    def setup_method(self):
        """设置测试方法"""
        self.save_path = "/tmp/stock_data"
        self.loader = MockStockDataLoader(
            save_path=self.save_path,
            max_retries=3,
            cache_days=30,
            frequency='daily',
            adjust_type='none'
        )

    def test_loader_initialization(self):
        """测试加载器初始化"""
        assert self.loader.save_path == Path(self.save_path)
        assert self.loader.max_retries == 3
        assert self.loader.cache_days == 30
        assert self.loader.frequency == 'daily'
        assert self.loader.adjust_type == 'none'
        assert self.loader.load_count == 0
        assert len(self.loader.cached_data) == 0

    def test_load_known_stocks(self):
        """测试加载知名股票"""
        # 加载AAPL
        aapl_data = self.loader.load("AAPL", "2023-01-01", "2023-01-02")
        assert aapl_data is not None
        assert aapl_data["symbol"] == "AAPL"
        assert aapl_data["name"] == "Apple Inc."
        assert len(aapl_data["data"]) == 2
        assert self.loader.load_count == 1

        # 验证数据结构
        first_record = aapl_data["data"][0]
        required_fields = ['date', 'open', 'high', 'low', 'close', 'volume']
        assert all(field in first_record for field in required_fields)

    def test_load_unknown_stock(self):
        """测试加载未知股票"""
        data = self.loader.load("UNKNOWN", "2023-01-01", "2023-01-02")
        assert data is not None
        assert data["symbol"] == "UNKNOWN"
        assert data["name"] == "UNKNOWN Corporation"
        assert len(data["data"]) == 2

    def test_load_with_cache(self):
        """测试缓存功能"""
        # 第一次加载
        data1 = self.loader.load("AAPL", "2023-01-01", "2023-01-02")
        assert self.loader.cache_misses == 1
        assert self.loader.cache_hits == 0

        # 第二次加载（应该从缓存获取）
        data2 = self.loader.load("AAPL", "2023-01-01", "2023-01-02")
        assert data1 == data2
        assert self.loader.cache_misses == 1
        assert self.loader.cache_hits == 1
        assert self.loader.load_count == 1  # 只加载了一次

    def test_cache_expiration(self):
        """测试缓存过期"""
        # 创建一个短期缓存的加载器
        short_cache_loader = MockStockDataLoader(
            save_path=self.save_path,
            cache_days=0  # 立即过期
        )

        # 第一次加载
        data1 = short_cache_loader.load("AAPL", "2023-01-01", "2023-01-02")

        # 第二次加载（缓存已过期）
        data2 = short_cache_loader.load("AAPL", "2023-01-01", "2023-01-02")

        # 应该加载了两次，因为缓存立即过期
        assert short_cache_loader.load_count == 2
        assert short_cache_loader.cache_misses == 2

    def test_error_handling(self):
        """测试错误处理"""
        # 加载会出错的股票
        result = self.loader.load("ERROR", "2023-01-01", "2023-01-02")
        assert result is None
        assert self.loader.retry_count == 4  # 初始尝试 + 3次重试
        assert len(self.loader.errors) > 0

    def test_empty_data_handling(self):
        """测试空数据处理"""
        result = self.loader.load("EMPTY", "2023-01-01", "2023-01-02")
        assert result is None

    def test_validate_valid_data(self):
        """测试验证有效数据"""
        valid_data = {
            "symbol": "AAPL",
            "data": [
                {"date": "2023-01-01", "open": 150.0, "high": 155.0, "low": 149.0, "close": 152.0, "volume": 1000000}
            ],
            "start_date": "2023-01-01",
            "end_date": "2023-01-02"
        }
        assert self.loader.validate(valid_data)

    def test_validate_invalid_data(self):
        """测试验证无效数据"""
        # 无效类型
        assert not self.loader.validate("invalid")

        # 缺少必需字段
        invalid_data = {"symbol": "AAPL"}  # 缺少data等字段
        assert not self.loader.validate(invalid_data)

        # 空数据数组
        empty_data = {
            "symbol": "AAPL",
            "data": [],
            "start_date": "2023-01-01",
            "end_date": "2023-01-02"
        }
        assert not self.loader.validate(empty_data)

        # 数据记录缺少字段
        bad_record_data = {
            "symbol": "AAPL",
            "data": [{"date": "2023-01-01"}],  # 缺少OHLCV字段
            "start_date": "2023-01-01",
            "end_date": "2023-01-02"
        }
        assert not self.loader.validate(bad_record_data)

    def test_get_metadata(self):
        """测试获取元数据"""
        metadata = self.loader.get_metadata()

        assert metadata["loader_type"] == "StockDataLoader"
        assert metadata["save_path"] == self.save_path
        assert metadata["max_retries"] == 3
        assert metadata["cache_days"] == 30
        assert metadata["frequency"] == 'daily'
        assert metadata["adjust_type"] == 'none'

    def test_clear_cache(self):
        """测试清空缓存"""
        # 先加载一些数据到缓存
        self.loader.load("AAPL", "2023-01-01", "2023-01-02")
        self.loader.load("GOOGL", "2023-01-01", "2023-01-02")

        assert len(self.loader.cached_data) > 0

        # 清空缓存
        self.loader.clear_cache()

        assert len(self.loader.cached_data) == 0
        # 统计信息保持累积，不会被清空
        assert self.loader.cache_hits == 0  # 没有命中
        assert self.loader.cache_misses == 2  # 前面有2次未命中

    def test_get_cache_stats(self):
        """测试获取缓存统计"""
        # 初始状态
        stats = self.loader.get_cache_stats()
        assert stats["cache_size"] == 0
        assert stats["total_requests"] == 0
        assert stats["hit_rate"] == 0.0

        # 加载数据
        self.loader.load("AAPL", "2023-01-01", "2023-01-02")  # miss
        self.loader.load("AAPL", "2023-01-01", "2023-01-02")  # hit

        stats = self.loader.get_cache_stats()
        assert stats["cache_size"] == 1
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 1
        assert stats["total_requests"] == 2
        assert stats["hit_rate"] == 0.5


class TestStockLoaderIntegration:
    """股票加载器集成测试"""

    def test_complete_loading_workflow(self):
        """测试完整的加载工作流"""
        loader = MockStockDataLoader(save_path="/tmp/test")

        # 1. 加载多个股票
        symbols = ["AAPL", "GOOGL", "MSFT"]
        loaded_data = {}

        for symbol in symbols:
            data = loader.load(symbol, "2023-01-01", "2023-01-02")
            loaded_data[symbol] = data

        # 2. 验证所有数据都加载成功
        assert len(loaded_data) == 3
        for symbol, data in loaded_data.items():
            assert data is not None
            assert data["symbol"] == symbol
            assert len(data["data"]) == 2
            assert loader.validate(data)

        # 3. 验证缓存效果
        # 重新加载，应该全部来自缓存
        cache_hits_before = loader.cache_hits
        for symbol in symbols:
            cached_data = loader.load(symbol, "2023-01-01", "2023-01-02")
            assert cached_data == loaded_data[symbol]

        assert loader.cache_hits == cache_hits_before + 3
        assert loader.load_count == 3  # 只加载了3次

        # 4. 验证统计信息
        metadata = loader.get_metadata()
        assert metadata["load_count"] == 3
        assert metadata["cache_hits"] == 3
        assert metadata["cache_misses"] == 3

        cache_stats = loader.get_cache_stats()
        assert cache_stats["cache_size"] == 3
        assert cache_stats["hit_rate"] == 0.5

    def test_different_frequencies(self):
        """测试不同频率的数据加载"""
        daily_loader = MockStockDataLoader(save_path="/tmp", frequency='daily')
        weekly_loader = MockStockDataLoader(save_path="/tmp", frequency='weekly')

        # 加载相同股票不同频率
        daily_data = daily_loader.load("AAPL", "2023-01-01", "2023-01-07")
        weekly_data = weekly_loader.load("AAPL", "2023-01-01", "2023-01-07")

        # 验证频率设置
        assert daily_data["frequency"] == 'daily'
        assert weekly_data["frequency"] == 'weekly'

        # 验证缓存分离（不同频率应该分开缓存）
        assert daily_loader.load_count == 1
        assert weekly_loader.load_count == 1

    def test_adjustment_types(self):
        """测试复权类型"""
        none_loader = MockStockDataLoader(save_path="/tmp", adjust_type='none')
        split_loader = MockStockDataLoader(save_path="/tmp", adjust_type='split')

        # 加载相同股票不同复权
        none_data = none_loader.load("AAPL", "2023-01-01", "2023-01-02")
        split_data = split_loader.load("AAPL", "2023-01-01", "2023-01-02")

        # 验证复权设置
        assert none_data["adjust_type"] == 'none'
        assert split_data["adjust_type"] == 'split'

        # 验证缓存分离
        assert none_loader.load_count == 1
        assert split_loader.load_count == 1

    def test_retry_mechanism(self):
        """测试重试机制"""
        loader = MockStockDataLoader(save_path="/tmp", max_retries=2)

        # 加载会失败的股票
        result = loader.load("ERROR", "2023-01-01", "2023-01-02")

        assert result is None
        assert loader.retry_count == 3  # 初始尝试 + 2次重试
        assert len(loader.errors) == 3

    def test_bulk_loading_pattern(self):
        """测试批量加载模式"""
        loader = MockStockDataLoader(save_path="/tmp")

        # 模拟批量加载模式
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
        results = {}

        # 依次加载（模拟批量处理）
        for symbol in symbols:
            data = loader.load(symbol, "2023-01-01", "2023-01-02")
            results[symbol] = data

        # 验证所有加载成功
        assert len(results) == 4
        assert all(data is not None for data in results.values())

        # 验证缓存状态
        cache_stats = loader.get_cache_stats()
        assert cache_stats["cache_size"] == 4
        assert cache_stats["total_requests"] == 4
        assert cache_stats["hit_rate"] == 0.0  # 首次加载都是miss

    def test_error_recovery(self):
        """测试错误恢复"""
        loader = MockStockDataLoader(save_path="/tmp", max_retries=1)

        # 先加载一个会失败的股票
        failed_result = loader.load("ERROR", "2023-01-01", "2023-01-02")
        assert failed_result is None

        # 再加载一个正常的股票，确保加载器仍然工作
        success_result = loader.load("AAPL", "2023-01-01", "2023-01-02")
        assert success_result is not None
        assert success_result["symbol"] == "AAPL"

        # 验证统计
        metadata = loader.get_metadata()
        assert metadata["load_count"] == 2  # 加载了两次（一次失败，一次成功）
        assert metadata["error_count"] == 2  # 有2次错误（重试机制）

    def test_cache_management(self):
        """测试缓存管理"""
        loader = MockStockDataLoader(save_path="/tmp", cache_days=1)

        # 加载数据
        loader.load("AAPL", "2023-01-01", "2023-01-02")
        loader.load("GOOGL", "2023-01-01", "2023-01-02")

        assert len(loader.cached_data) == 2

        # 模拟缓存过期检查
        # 注意：在我们的实现中，cache_days=1表示不会过期

        # 清空缓存
        loader.clear_cache()
        assert len(loader.cached_data) == 0

        # 再次加载应该重新获取数据
        loader.load("AAPL", "2023-01-01", "2023-01-02")
        assert len(loader.cached_data) == 1
        assert loader.cache_misses == 3  # 统计保持累积


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
