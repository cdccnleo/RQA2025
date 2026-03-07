# -*- coding: utf-8 -*-
"""
数据适配器组件Mock测试
测试数据适配器的核心功能和适配器模式
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, List, Any, Optional
from enum import Enum


class MockAdapterType(Enum):
    """模拟适配器类型枚举"""
    STOCK = "stock"
    CRYPTO = "crypto"
    FOREX = "forex"
    BOND = "bond"
    COMMODITY = "commodity"
    NEWS = "news"
    INDEX = "index"
    MACRO = "macro"
    OPTIONS = "options"


class MockAdapterConfig:
    """模拟适配器配置"""

    def __init__(self, name: str = "", adapter_type: str = "", timeout: int = 30,
                 max_retries: int = 3, connection_params: Optional[Dict[str, Any]] = None,
                 validation_rules: Optional[Dict[str, Any]] = None):
        self.name = name
        self.adapter_type = adapter_type
        self.timeout = timeout
        self.max_retries = max_retries
        self.connection_params = connection_params or {}
        self.validation_rules = validation_rules or {}

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "adapter_type": self.adapter_type,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "connection_params": self.connection_params,
            "validation_rules": self.validation_rules
        }


class MockDataRequest:
    """模拟数据请求"""

    def __init__(self, symbols: List[str], start_date: Optional[str] = None,
                 end_date: Optional[str] = None, params: Optional[Dict[str, Any]] = None):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.params = params or {}
        self.request_id = f"req_{hash(str(self.symbols) + str(datetime.now()))}"
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "request_id": self.request_id,
            "symbols": self.symbols,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "params": self.params,
            "timestamp": self.timestamp.isoformat()
        }


class MockDataResponse:
    """模拟数据响应"""

    def __init__(self, request: MockDataRequest, data: Optional[Dict[str, Any]] = None,
                 error: Optional[str] = None, success: bool = True):
        self.request = request
        self.data = data or {}
        self.error = error
        self.success = success and error is None
        self.response_id = f"resp_{request.request_id}"
        self.timestamp = datetime.now()
        self.processing_time = (self.timestamp - request.timestamp).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "response_id": self.response_id,
            "request_id": self.request.request_id,
            "success": self.success,
            "error": self.error,
            "data": self.data,
            "processing_time": self.processing_time,
            "timestamp": self.timestamp.isoformat()
        }


class MockBaseAdapter:
    """模拟基础数据适配器"""

    def __init__(self, config: MockAdapterConfig):
        self.config = config
        self.is_connected = False
        self.is_initialized = False
        self.request_count = 0
        self.error_count = 0
        self.last_request_time = None
        self.logger = Mock()
        self.logger.info = Mock()
        self.logger.warning = Mock()
        self.logger.error = Mock()

    def initialize(self) -> bool:
        """初始化适配器"""
        try:
            # 模拟初始化逻辑
            self.is_initialized = True
            self.logger.info(f"Adapter {self.config.name} initialized")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize adapter: {e}")
            return False

    def connect(self) -> bool:
        """连接数据源"""
        if not self.is_initialized:
            raise Exception("Adapter not initialized")

        try:
            # 模拟连接逻辑
            self.is_connected = True
            self.logger.info(f"Adapter {self.config.name} connected")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect adapter: {e}")
            return False

    def disconnect(self) -> bool:
        """断开连接"""
        try:
            self.is_connected = False
            self.logger.info(f"Adapter {self.config.name} disconnected")
            return True
        except Exception as e:
            self.logger.error(f"Failed to disconnect adapter: {e}")
            return False

    def validate_request(self, request: MockDataRequest) -> bool:
        """验证请求"""
        if not self.is_connected:
            return False

        # 基本验证
        if not request.symbols or len(request.symbols) == 0:
            return False

        # 日期验证
        if request.start_date and request.end_date:
            try:
                start = datetime.fromisoformat(request.start_date.replace('Z', '+00:00'))
                end = datetime.fromisoformat(request.end_date.replace('Z', '+00:00'))
                if start > end:
                    return False
            except ValueError:
                return False

        return True

    def get_metadata(self) -> Dict[str, Any]:
        """获取适配器元数据"""
        return {
            "adapter_type": self.config.adapter_type,
            "name": self.config.name,
            "is_connected": self.is_connected,
            "is_initialized": self.is_initialized,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "last_request_time": self.last_request_time.isoformat() if self.last_request_time else None,
            "config": self.config.to_dict()
        }

    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        return {
            "healthy": self.is_connected and self.is_initialized,
            "connection_status": "connected" if self.is_connected else "disconnected",
            "initialization_status": "initialized" if self.is_initialized else "uninitialized",
            "error_rate": self.error_count / max(1, self.request_count),
            "last_check": datetime.now().isoformat()
        }


class MockStockAdapter(MockBaseAdapter):
    """模拟股票数据适配器"""

    def __init__(self, config: MockAdapterConfig):
        super().__init__(config)
        self.stock_data_cache: Dict[str, Dict[str, Any]] = {}

    def fetch_data(self, request: MockDataRequest) -> MockDataResponse:
        """获取股票数据"""
        self.request_count += 1
        self.last_request_time = datetime.now()

        if not self.validate_request(request):
            self.error_count += 1
            return MockDataResponse(request, error="Invalid request", success=False)

        try:
            # 模拟股票数据获取
            data = {}
            for symbol in request.symbols:
                if symbol not in self.stock_data_cache:
                    # 生成模拟数据
                    self.stock_data_cache[symbol] = self._generate_stock_data(symbol, request)

                data[symbol] = self.stock_data_cache[symbol]

            return MockDataResponse(request, data=data)
        except Exception as e:
            self.error_count += 1
            return MockDataResponse(request, error=str(e), success=False)

    def _generate_stock_data(self, symbol: str, request: MockDataRequest) -> Dict[str, Any]:
        """生成模拟股票数据"""
        # 模拟不同股票的基本数据
        base_prices = {
            "AAPL": 150.0,
            "GOOGL": 2500.0,
            "MSFT": 300.0,
            "TSLA": 200.0
        }

        base_price = base_prices.get(symbol, 100.0)

        # 生成OHLCV数据
        import random
        random.seed(hash(symbol))  # 保证每次运行结果一致

        data_points = []
        current_date = datetime.fromisoformat(request.start_date.replace('Z', '+00:00')) if request.start_date else datetime.now() - timedelta(days=30)

        for i in range(30):  # 生成30天的数据
            # 价格波动
            change = random.uniform(-0.05, 0.05)  # -5% 到 +5%的波动
            open_price = base_price * (1 + change)
            high_price = open_price * random.uniform(1.0, 1.03)  # 最高价在开盘价的0-3%范围内
            low_price = open_price * random.uniform(0.97, 1.0)  # 最低价在开盘价的-3%-0范围内
            close_price = random.uniform(low_price, high_price)
            volume = random.randint(100000, 1000000)

            data_points.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": volume,
                "symbol": symbol
            })

            current_date += timedelta(days=1)
            base_price = close_price  # 下一天的基准价基于前一天的收盘价

        return {
            "symbol": symbol,
            "data": data_points,
            "metadata": {
                "source": "mock_stock_api",
                "data_points": len(data_points),
                "start_date": request.start_date,
                "end_date": request.end_date
            }
        }


class MockCryptoAdapter(MockBaseAdapter):
    """模拟加密货币数据适配器"""

    def __init__(self, config: MockAdapterConfig):
        super().__init__(config)
        self.crypto_data_cache: Dict[str, Dict[str, Any]] = {}

    def fetch_data(self, request: MockDataRequest) -> MockDataResponse:
        """获取加密货币数据"""
        self.request_count += 1
        self.last_request_time = datetime.now()

        if not self.validate_request(request):
            self.error_count += 1
            return MockDataResponse(request, error="Invalid request", success=False)

        try:
            data = {}
            for symbol in request.symbols:
                if symbol not in self.crypto_data_cache:
                    self.crypto_data_cache[symbol] = self._generate_crypto_data(symbol, request)

                data[symbol] = self.crypto_data_cache[symbol]

            return MockDataResponse(request, data=data)
        except Exception as e:
            self.error_count += 1
            return MockDataResponse(request, error=str(e), success=False)

    def _generate_crypto_data(self, symbol: str, request: MockDataRequest) -> Dict[str, Any]:
        """生成模拟加密货币数据"""
        base_prices = {
            "BTC": 50000.0,
            "ETH": 3000.0,
            "BNB": 400.0,
            "ADA": 1.5
        }

        base_price = base_prices.get(symbol, 100.0)
        import random
        random.seed(hash(symbol))

        data_points = []
        current_date = datetime.fromisoformat(request.start_date.replace('Z', '+00:00')) if request.start_date else datetime.now() - timedelta(days=30)

        for i in range(30):
            # 加密货币波动更大
            change = random.uniform(-0.1, 0.1)  # -10% 到 +10%的波动
            open_price = base_price * (1 + change)
            high_price = open_price * random.uniform(1.0, 1.05)
            low_price = open_price * random.uniform(0.95, 1.0)
            close_price = random.uniform(low_price, high_price)
            volume = random.randint(1000000, 10000000)  # 更大的交易量

            data_points.append({
                "timestamp": int(current_date.timestamp()),
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": volume,
                "symbol": symbol,
                "market_cap": round(close_price * 19000000, 2)  # 模拟市值
            })

            current_date += timedelta(hours=1)  # 加密货币数据更频繁
            base_price = close_price

        return {
            "symbol": symbol,
            "data": data_points,
            "metadata": {
                "source": "mock_crypto_api",
                "data_points": len(data_points),
                "frequency": "hourly",
                "start_date": request.start_date,
                "end_date": request.end_date
            }
        }


class MockAdapterRegistry:
    """模拟适配器注册表"""

    def __init__(self):
        self.adapters: Dict[str, MockBaseAdapter] = {}
        self.adapter_types: Dict[str, str] = {}

    def register_adapter(self, name: str, adapter: MockBaseAdapter) -> bool:
        """注册适配器"""
        if name in self.adapters:
            return False

        self.adapters[name] = adapter
        self.adapter_types[name] = adapter.config.adapter_type
        return True

    def unregister_adapter(self, name: str) -> bool:
        """注销适配器"""
        if name not in self.adapters:
            return False

        del self.adapters[name]
        del self.adapter_types[name]
        return True

    def get_adapter(self, name: str) -> Optional[MockBaseAdapter]:
        """获取适配器"""
        return self.adapters.get(name)

    def get_adapters_by_type(self, adapter_type: str) -> List[MockBaseAdapter]:
        """按类型获取适配器"""
        return [adapter for adapter in self.adapters.values()
                if adapter.config.adapter_type == adapter_type]

    def list_adapters(self) -> List[str]:
        """列出所有适配器"""
        return list(self.adapters.keys())

    def get_registry_stats(self) -> Dict[str, Any]:
        """获取注册表统计"""
        type_counts = {}
        for adapter_type in self.adapter_types.values():
            type_counts[adapter_type] = type_counts.get(adapter_type, 0) + 1

        return {
            "total_adapters": len(self.adapters),
            "adapter_types": type_counts,
            "adapter_names": list(self.adapters.keys())
        }


class TestMockAdapterConfig:
    """模拟适配器配置测试"""

    def test_config_creation(self):
        """测试配置创建"""
        config = MockAdapterConfig(
            name="test_adapter",
            adapter_type="stock",
            timeout=60,
            max_retries=5
        )

        assert config.name == "test_adapter"
        assert config.adapter_type == "stock"
        assert config.timeout == 60
        assert config.max_retries == 5
        assert config.connection_params == {}
        assert config.validation_rules == {}

    def test_config_with_params(self):
        """测试带参数的配置"""
        connection_params = {"host": "localhost", "port": 8080}
        validation_rules = {"max_symbols": 100}

        config = MockAdapterConfig(
            connection_params=connection_params,
            validation_rules=validation_rules
        )

        assert config.connection_params == connection_params
        assert config.validation_rules == validation_rules

    def test_config_to_dict(self):
        """测试配置序列化"""
        config = MockAdapterConfig(
            name="test",
            adapter_type="crypto",
            timeout=30
        )

        data = config.to_dict()
        assert data["name"] == "test"
        assert data["adapter_type"] == "crypto"
        assert data["timeout"] == 30


class TestMockDataRequest:
    """模拟数据请求测试"""

    def test_request_creation(self):
        """测试请求创建"""
        symbols = ["AAPL", "GOOGL"]
        start_date = "2023-01-01"
        end_date = "2023-01-31"

        request = MockDataRequest(symbols, start_date, end_date)

        assert request.symbols == symbols
        assert request.start_date == start_date
        assert request.end_date == end_date
        assert request.params == {}
        assert request.request_id.startswith("req_")
        assert isinstance(request.timestamp, datetime)

    def test_request_with_params(self):
        """测试带参数的请求"""
        params = {"frequency": "daily", "adjust": "split"}
        request = MockDataRequest(["BTC"], params=params)

        assert request.params == params

    def test_request_to_dict(self):
        """测试请求序列化"""
        request = MockDataRequest(["AAPL"], "2023-01-01", "2023-01-02")

        data = request.to_dict()
        assert data["symbols"] == ["AAPL"]
        assert data["start_date"] == "2023-01-01"
        assert data["end_date"] == "2023-01-02"
        assert "request_id" in data
        assert "timestamp" in data


class TestMockDataResponse:
    """模拟数据响应测试"""

    def test_success_response(self):
        """测试成功响应"""
        request = MockDataRequest(["AAPL"])
        data = {"AAPL": {"price": 150.0}}
        response = MockDataResponse(request, data=data)

        assert response.request == request
        assert response.data == data
        assert response.success is True
        assert response.error is None
        assert response.response_id.startswith("resp_")
        assert response.processing_time >= 0

    def test_error_response(self):
        """测试错误响应"""
        request = MockDataRequest(["INVALID"])
        response = MockDataResponse(request, error="Symbol not found", success=False)

        assert response.success is False
        assert response.error == "Symbol not found"
        assert response.data == {}

    def test_response_to_dict(self):
        """测试响应序列化"""
        request = MockDataRequest(["AAPL"])
        response = MockDataResponse(request, data={"test": "data"})

        data = response.to_dict()
        assert data["success"] is True
        assert data["data"] == {"test": "data"}
        assert "response_id" in data
        assert "processing_time" in data


class TestMockBaseAdapter:
    """模拟基础适配器测试"""

    def setup_method(self):
        """设置测试方法"""
        self.config = MockAdapterConfig(
            name="test_adapter",
            adapter_type="stock",
            timeout=30,
            max_retries=3
        )
        self.adapter = MockBaseAdapter(self.config)

    def test_adapter_initialization(self):
        """测试适配器初始化"""
        assert not self.adapter.is_initialized
        assert not self.adapter.is_connected

        assert self.adapter.initialize()
        assert self.adapter.is_initialized

    def test_adapter_connection(self):
        """测试适配器连接"""
        self.adapter.initialize()

        assert self.adapter.connect()
        assert self.adapter.is_connected

        assert self.adapter.disconnect()
        assert not self.adapter.is_connected

    def test_adapter_uninitialized_operations(self):
        """测试未初始化操作"""
        with pytest.raises(Exception, match="Adapter not initialized"):
            self.adapter.connect()

    def test_request_validation(self):
        """测试请求验证"""
        self.adapter.initialize()
        self.adapter.connect()

        # 有效请求
        valid_request = MockDataRequest(["AAPL"], "2023-01-01", "2023-01-02")
        assert self.adapter.validate_request(valid_request)

        # 无效请求 - 空符号
        invalid_request1 = MockDataRequest([])
        assert not self.adapter.validate_request(invalid_request1)

        # 无效请求 - 日期顺序错误
        invalid_request2 = MockDataRequest(["AAPL"], "2023-01-02", "2023-01-01")
        assert not self.adapter.validate_request(invalid_request2)

        # 未连接状态验证
        self.adapter.disconnect()
        assert not self.adapter.validate_request(valid_request)

    def test_get_metadata(self):
        """测试获取元数据"""
        metadata = self.adapter.get_metadata()

        assert metadata["name"] == "test_adapter"
        assert metadata["adapter_type"] == "stock"
        assert metadata["is_initialized"] is False
        assert metadata["is_connected"] is False
        assert metadata["request_count"] == 0

    def test_get_health_status(self):
        """测试获取健康状态"""
        # 未初始化状态
        status = self.adapter.get_health_status()
        assert status["healthy"] is False
        assert status["connection_status"] == "disconnected"
        assert status["initialization_status"] == "uninitialized"

        # 初始化后连接
        self.adapter.initialize()
        self.adapter.connect()

        status = self.adapter.get_health_status()
        assert status["healthy"] is True
        assert status["connection_status"] == "connected"
        assert status["initialization_status"] == "initialized"


class TestMockStockAdapter:
    """模拟股票适配器测试"""

    def setup_method(self):
        """设置测试方法"""
        self.config = MockAdapterConfig(
            name="stock_adapter",
            adapter_type="stock",
            timeout=30
        )
        self.adapter = MockStockAdapter(self.config)

    def test_stock_adapter_initialization(self):
        """测试股票适配器初始化"""
        assert self.adapter.config.adapter_type == "stock"
        assert len(self.adapter.stock_data_cache) == 0

    def test_stock_adapter_fetch_data(self):
        """测试股票数据获取"""
        self.adapter.initialize()
        self.adapter.connect()

        request = MockDataRequest(["AAPL"], "2023-01-01", "2023-01-02")
        response = self.adapter.fetch_data(request)

        assert response.success is True
        assert "AAPL" in response.data
        assert len(response.data["AAPL"]["data"]) > 0

        # 验证数据结构
        stock_data = response.data["AAPL"]
        assert stock_data["symbol"] == "AAPL"
        assert "data" in stock_data
        assert "metadata" in stock_data

        # 验证第一条数据
        first_record = stock_data["data"][0]
        required_fields = ["date", "open", "high", "low", "close", "volume", "symbol"]
        for field in required_fields:
            assert field in first_record

    def test_stock_adapter_multiple_symbols(self):
        """测试多符号数据获取"""
        self.adapter.initialize()
        self.adapter.connect()

        symbols = ["AAPL", "GOOGL", "MSFT"]
        request = MockDataRequest(symbols, "2023-01-01", "2023-01-02")
        response = self.adapter.fetch_data(request)

        assert response.success is True
        for symbol in symbols:
            assert symbol in response.data
            assert response.data[symbol]["symbol"] == symbol

    def test_stock_adapter_invalid_request(self):
        """测试无效请求"""
        self.adapter.initialize()
        self.adapter.connect()

        # 空符号请求
        request = MockDataRequest([])
        response = self.adapter.fetch_data(request)

        assert response.success is False
        assert "Invalid request" in response.error

    def test_stock_adapter_caching(self):
        """测试数据缓存"""
        self.adapter.initialize()
        self.adapter.connect()

        request = MockDataRequest(["AAPL"], "2023-01-01", "2023-01-02")

        # 第一次请求
        response1 = self.adapter.fetch_data(request)
        assert response1.success is True

        # 第二次请求（应该从缓存获取）
        response2 = self.adapter.fetch_data(request)
        assert response2.success is True

        # 数据应该相同
        assert response1.data == response2.data

        # 缓存中应该有数据
        assert "AAPL" in self.adapter.stock_data_cache


class TestMockCryptoAdapter:
    """模拟加密货币适配器测试"""

    def setup_method(self):
        """设置测试方法"""
        self.config = MockAdapterConfig(
            name="crypto_adapter",
            adapter_type="crypto",
            timeout=30
        )
        self.adapter = MockCryptoAdapter(self.config)

    def test_crypto_adapter_initialization(self):
        """测试加密货币适配器初始化"""
        assert self.adapter.config.adapter_type == "crypto"
        assert len(self.adapter.crypto_data_cache) == 0

    def test_crypto_adapter_fetch_data(self):
        """测试加密货币数据获取"""
        self.adapter.initialize()
        self.adapter.connect()

        request = MockDataRequest(["BTC"], "2023-01-01T00:00:00Z", "2023-01-02T00:00:00Z")
        response = self.adapter.fetch_data(request)

        assert response.success is True
        assert "BTC" in response.data

        crypto_data = response.data["BTC"]
        assert crypto_data["symbol"] == "BTC"
        assert len(crypto_data["data"]) > 0

        # 验证加密货币数据结构
        first_record = crypto_data["data"][0]
        required_fields = ["timestamp", "open", "high", "low", "close", "volume", "symbol", "market_cap"]
        for field in required_fields:
            assert field in first_record

    def test_crypto_adapter_high_volatility(self):
        """测试加密货币高波动性"""
        self.adapter.initialize()
        self.adapter.connect()

        request = MockDataRequest(["BTC"], "2023-01-01T00:00:00Z", "2023-01-02T00:00:00Z")
        response = self.adapter.fetch_data(request)

        prices = [record["close"] for record in response.data["BTC"]["data"]]

        # 计算价格变化
        price_changes = []
        for i in range(1, len(prices)):
            change = abs(prices[i] - prices[i-1]) / prices[i-1]
            price_changes.append(change)

        # 加密货币应该有较高的波动性
        avg_volatility = sum(price_changes) / len(price_changes)
        assert avg_volatility > 0.01  # 平均波动率应该大于1%

    def test_crypto_adapter_hourly_data(self):
        """测试小时级别数据"""
        self.adapter.initialize()
        self.adapter.connect()

        request = MockDataRequest(["ETH"], "2023-01-01T00:00:00Z", "2023-01-02T00:00:00Z")
        response = self.adapter.fetch_data(request)

        metadata = response.data["ETH"]["metadata"]
        assert metadata["frequency"] == "hourly"

        # 验证时间戳间隔（应该大约是1小时）
        timestamps = [record["timestamp"] for record in response.data["ETH"]["data"]]
        if len(timestamps) > 1:
            time_diff = timestamps[1] - timestamps[0]
            assert abs(time_diff - 3600) < 60  # 允许60秒的误差


class TestMockAdapterRegistry:
    """模拟适配器注册表测试"""

    def setup_method(self):
        """设置测试方法"""
        self.registry = MockAdapterRegistry()

    def test_registry_initialization(self):
        """测试注册表初始化"""
        assert len(self.registry.adapters) == 0
        assert len(self.registry.adapter_types) == 0

    def test_register_adapter(self):
        """测试注册适配器"""
        config = MockAdapterConfig(name="stock_adapter", adapter_type="stock")
        adapter = MockStockAdapter(config)

        assert self.registry.register_adapter("stock_adapter", adapter)
        assert "stock_adapter" in self.registry.adapters
        assert self.registry.adapters["stock_adapter"] == adapter

    def test_register_duplicate_adapter(self):
        """测试注册重复适配器"""
        config = MockAdapterConfig(name="adapter1", adapter_type="stock")
        adapter = MockStockAdapter(config)

        # 第一次注册成功
        assert self.registry.register_adapter("adapter1", adapter)

        # 第二次注册失败
        assert not self.registry.register_adapter("adapter1", adapter)

    def test_unregister_adapter(self):
        """测试注销适配器"""
        config = MockAdapterConfig(name="adapter1", adapter_type="stock")
        adapter = MockStockAdapter(config)

        self.registry.register_adapter("adapter1", adapter)
        assert "adapter1" in self.registry.adapters

        assert self.registry.unregister_adapter("adapter1")
        assert "adapter1" not in self.registry.adapters

    def test_get_adapter(self):
        """测试获取适配器"""
        config = MockAdapterConfig(name="adapter1", adapter_type="stock")
        adapter = MockStockAdapter(config)

        self.registry.register_adapter("adapter1", adapter)

        retrieved = self.registry.get_adapter("adapter1")
        assert retrieved == adapter

        # 获取不存在的适配器
        assert self.registry.get_adapter("nonexistent") is None

    def test_get_adapters_by_type(self):
        """测试按类型获取适配器"""
        # 注册不同类型的适配器
        stock_config = MockAdapterConfig(name="stock1", adapter_type="stock")
        stock_adapter = MockStockAdapter(stock_config)

        crypto_config = MockAdapterConfig(name="crypto1", adapter_type="crypto")
        crypto_adapter = MockCryptoAdapter(crypto_config)

        stock_config2 = MockAdapterConfig(name="stock2", adapter_type="stock")
        stock_adapter2 = MockStockAdapter(stock_config2)

        self.registry.register_adapter("stock1", stock_adapter)
        self.registry.register_adapter("crypto1", crypto_adapter)
        self.registry.register_adapter("stock2", stock_adapter2)

        # 获取股票适配器
        stock_adapters = self.registry.get_adapters_by_type("stock")
        assert len(stock_adapters) == 2
        assert stock_adapter in stock_adapters
        assert stock_adapter2 in stock_adapters

        # 获取加密货币适配器
        crypto_adapters = self.registry.get_adapters_by_type("crypto")
        assert len(crypto_adapters) == 1
        assert crypto_adapter in crypto_adapters

    def test_list_adapters(self):
        """测试列出适配器"""
        adapters = ["adapter1", "adapter2", "adapter3"]

        for name in adapters:
            config = MockAdapterConfig(name=name, adapter_type="stock")
            adapter = MockStockAdapter(config)
            self.registry.register_adapter(name, adapter)

        adapter_list = self.registry.list_adapters()
        assert len(adapter_list) == 3
        for name in adapters:
            assert name in adapter_list

    def test_get_registry_stats(self):
        """测试获取注册表统计"""
        # 注册不同类型的适配器
        types_and_counts = {
            "stock": 3,
            "crypto": 2,
            "forex": 1
        }

        adapter_counter = 0
        for adapter_type, count in types_and_counts.items():
            for i in range(count):
                name = f"{adapter_type}_{i}"
                config = MockAdapterConfig(name=name, adapter_type=adapter_type)
                if adapter_type == "crypto":
                    adapter = MockCryptoAdapter(config)
                else:
                    adapter = MockStockAdapter(config)
                self.registry.register_adapter(name, adapter)
                adapter_counter += 1

        stats = self.registry.get_registry_stats()

        assert stats["total_adapters"] == adapter_counter
        assert stats["adapter_types"] == types_and_counts
        assert len(stats["adapter_names"]) == adapter_counter


class TestAdapterIntegration:
    """适配器集成测试"""

    def test_complete_adapter_workflow(self):
        """测试完整的适配器工作流"""
        # 创建注册表
        registry = MockAdapterRegistry()

        # 注册适配器
        stock_config = MockAdapterConfig(name="stock_adapter", adapter_type="stock")
        stock_adapter = MockStockAdapter(stock_config)

        crypto_config = MockAdapterConfig(name="crypto_adapter", adapter_type="crypto")
        crypto_adapter = MockCryptoAdapter(crypto_config)

        registry.register_adapter("stock", stock_adapter)
        registry.register_adapter("crypto", crypto_adapter)

        # 初始化和连接适配器
        for adapter in [stock_adapter, crypto_adapter]:
            assert adapter.initialize()
            assert adapter.connect()

        # 创建请求
        stock_request = MockDataRequest(["AAPL", "GOOGL"], "2023-01-01", "2023-01-02")
        crypto_request = MockDataRequest(["BTC", "ETH"], "2023-01-01T00:00:00Z", "2023-01-02T00:00:00Z")

        # 获取数据
        stock_response = stock_adapter.fetch_data(stock_request)
        crypto_response = crypto_adapter.fetch_data(crypto_request)

        # 验证响应
        assert stock_response.success is True
        assert len(stock_response.data) == 2
        assert all(symbol in stock_response.data for symbol in ["AAPL", "GOOGL"])

        assert crypto_response.success is True
        assert len(crypto_response.data) == 2
        assert all(symbol in crypto_response.data for symbol in ["BTC", "ETH"])

        # 验证统计信息
        stock_stats = stock_adapter.get_health_status()
        crypto_stats = crypto_adapter.get_health_status()

        assert stock_stats["healthy"] is True
        assert crypto_stats["healthy"] is True

        # 验证注册表统计
        registry_stats = registry.get_registry_stats()
        assert registry_stats["total_adapters"] == 2
        assert registry_stats["adapter_types"]["stock"] == 1
        assert registry_stats["adapter_types"]["crypto"] == 1

    def test_adapter_error_handling(self):
        """测试适配器错误处理"""
        config = MockAdapterConfig(name="faulty_adapter", adapter_type="stock")
        adapter = MockStockAdapter(config)

        adapter.initialize()
        adapter.connect()

        # 创建无效请求
        invalid_request = MockDataRequest([])  # 空符号列表
        response = adapter.fetch_data(invalid_request)

        assert response.success is False
        assert response.error is not None

        # 验证错误统计
        health_status = adapter.get_health_status()
        assert health_status["error_rate"] > 0

        metadata = adapter.get_metadata()
        assert metadata["error_count"] == 1
        assert metadata["request_count"] == 1

    def test_adapter_performance_monitoring(self):
        """测试适配器性能监控"""
        config = MockAdapterConfig(name="perf_adapter", adapter_type="stock")
        adapter = MockStockAdapter(config)

        adapter.initialize()
        adapter.connect()

        # 执行多个请求
        symbols_list = [["AAPL"], ["GOOGL"], ["MSFT"], ["AAPL", "GOOGL"]]  # 最后一个是批量请求

        responses = []
        for symbols in symbols_list:
            request = MockDataRequest(symbols, "2023-01-01", "2023-01-02")
            response = adapter.fetch_data(request)
            responses.append(response)

        # 验证所有请求成功
        assert all(resp.success for resp in responses)

        # 验证统计信息
        metadata = adapter.get_metadata()
        assert metadata["request_count"] == len(symbols_list)
        assert metadata["error_count"] == 0

        health_status = adapter.get_health_status()
        assert health_status["error_rate"] == 0.0

        # 验证批量请求的数据量
        batch_response = responses[-1]
        assert len(batch_response.data) == 2

    def test_adapter_type_specialization(self):
        """测试适配器类型专业化"""
        # 股票适配器
        stock_config = MockAdapterConfig(name="stock_specialized", adapter_type="stock")
        stock_adapter = MockStockAdapter(stock_config)

        # 加密货币适配器
        crypto_config = MockAdapterConfig(name="crypto_specialized", adapter_type="crypto")
        crypto_adapter = MockCryptoAdapter(crypto_config)

        # 初始化和连接
        for adapter in [stock_adapter, crypto_adapter]:
            adapter.initialize()
            adapter.connect()

        # 测试股票数据结构
        stock_request = MockDataRequest(["AAPL"], "2023-01-01", "2023-01-02")
        stock_response = stock_adapter.fetch_data(stock_request)

        stock_record = stock_response.data["AAPL"]["data"][0]
        # 股票数据应该有日期字段
        assert "date" in stock_record
        assert "symbol" in stock_record

        # 测试加密货币数据结构
        crypto_request = MockDataRequest(["BTC"], "2023-01-01T00:00:00Z", "2023-01-02T00:00:00Z")
        crypto_response = crypto_adapter.fetch_data(crypto_request)

        crypto_record = crypto_response.data["BTC"]["data"][0]
        # 加密货币数据应该有时间戳字段
        assert "timestamp" in crypto_record
        assert "market_cap" in crypto_record

        # 验证元数据差异
        stock_metadata = stock_response.data["AAPL"]["metadata"]
        crypto_metadata = crypto_response.data["BTC"]["metadata"]

        assert stock_metadata["source"] == "mock_stock_api"
        assert crypto_metadata["source"] == "mock_crypto_api"
        assert stock_metadata.get("frequency") != crypto_metadata.get("frequency")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
