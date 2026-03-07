# -*- coding: utf-8 -*-
"""
数据流Mock测试
测试数据流处理、ETL管道、多市场同步和错误恢复功能
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass
import asyncio


class MockDataType(Enum):
    """模拟数据类型枚举"""
    TICK = "tick"
    OHLC = "ohlc"
    BUSINESS = "business"
    QUOTE = "quote"
    ORDERBOOK = "orderbook"


class MockSyncType(Enum):
    """模拟同步类型枚举"""
    REAL_TIME = "real_time"
    BATCH = "batch"
    HISTORICAL = "historical"


@dataclass
class MockDataFlowConfig:
    """模拟数据流配置"""

    def __init__(self, buffer_size: int = 1000, max_workers: int = 4,
                 retry_attempts: int = 3, timeout: float = 30.0,
                 enable_compression: bool = True, sync_mode: str = "real_time"):
        self.buffer_size = buffer_size
        self.max_workers = max_workers
        self.retry_attempts = retry_attempts
        self.timeout = timeout
        self.enable_compression = enable_compression
        self.sync_mode = sync_mode

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "buffer_size": self.buffer_size,
            "max_workers": self.max_workers,
            "retry_attempts": self.retry_attempts,
            "timeout": self.timeout,
            "enable_compression": self.enable_compression,
            "sync_mode": self.sync_mode
        }


class MockDataFlow:
    """模拟数据流"""

    def __init__(self, flow_id: str, config: MockDataFlowConfig):
        self.flow_id = flow_id
        self.config = config
        self.is_active = False
        self.buffer = []
        self.processed_count = 0
        self.error_count = 0
        self.retry_count = 0
        self.compression_ratio = 0.0
        self.throughput = 0.0
        self.logger = Mock()
        self.logger.info = Mock()
        self.logger.warning = Mock()
        self.logger.error = Mock()

    def start_flow(self) -> bool:
        """启动数据流"""
        try:
            self.is_active = True
            self.logger.info(f"Data flow {self.flow_id} started")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start data flow: {e}")
            return False

    def stop_flow(self) -> bool:
        """停止数据流"""
        try:
            self.is_active = False
            self.logger.info(f"Data flow {self.flow_id} stopped")
            return True
        except Exception as e:
            self.logger.error(f"Failed to stop data flow: {e}")
            return False

    def add_data(self, data: Any, priority: int = 1) -> bool:
        """添加数据到流"""
        if not self.is_active:
            return False

        if len(self.buffer) >= self.config.buffer_size:
            self.logger.warning("Buffer full, dropping data")
            return False

        self.buffer.append({"data": data, "priority": priority, "timestamp": datetime.now()})
        return True

    def process_batch(self, batch_size: int = 10) -> int:
        """处理一批数据"""
        if not self.is_active:
            return 0

        processed = 0
        remaining_items = []

        for item in self.buffer[:batch_size]:
            try:
                self._process_item(item["data"])
                self.processed_count += 1
                processed += 1
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Failed to process item: {e}")
                # 失败的项目保留在缓冲区中用于重试
                remaining_items.append(item)

        # 重构缓冲区：先是未处理的失败项目，然后是剩余的未处理项目
        self.buffer = remaining_items + self.buffer[batch_size:]
        return processed

    def _process_item(self, data: Any) -> None:
        """处理单个数据项"""
        # 模拟处理逻辑
        if isinstance(data, dict):
            # 验证必需字段
            if "symbol" not in data:
                raise ValueError("Missing symbol field")
        elif isinstance(data, pd.DataFrame):
            if data.empty:
                raise ValueError("Empty DataFrame")
        elif isinstance(data, str):
            if data == "invalid":
                raise ValueError("Invalid data")

        # 模拟压缩
        if self.config.enable_compression:
            self.compression_ratio = 0.7  # 模拟70%的压缩率

    def get_flow_stats(self) -> Dict[str, Any]:
        """获取流统计"""
        return {
            "flow_id": self.flow_id,
            "is_active": self.is_active,
            "buffer_size": len(self.buffer),
            "processed_count": self.processed_count,
            "error_count": self.error_count,
            "retry_count": self.retry_count,
            "compression_ratio": self.compression_ratio,
            "throughput": self.throughput,
            "error_rate": self.error_count / max(1, self.processed_count + self.error_count)
        }


class MockETLPipeline:
    """模拟ETL管道"""

    def __init__(self, pipeline_id: str, config: MockDataFlowConfig):
        self.pipeline_id = pipeline_id
        self.config = config
        self.is_running = False
        self.stages = {
            "extract": Mock(),
            "transform": Mock(),
            "load": Mock()
        }
        self.stage_stats = {
            "extract": {"processed": 0, "errors": 0},
            "transform": {"processed": 0, "errors": 0},
            "load": {"processed": 0, "errors": 0}
        }
        self.logger = Mock()

    def initialize_pipeline(self) -> bool:
        """初始化管道"""
        try:
            self.is_running = False
            self.logger.info(f"ETL Pipeline {self.pipeline_id} initialized")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline: {e}")
            return False

    def run_pipeline(self, data_source: Any) -> bool:
        """运行ETL管道"""
        if not self.initialize_pipeline():
            return False

        try:
            self.is_running = True

            # Extract阶段
            extracted_data = self._run_extract(data_source)

            # Transform阶段
            transformed_data = self._run_transform(extracted_data)

            # Load阶段
            success = self._run_load(transformed_data)

            self.is_running = False
            return success

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            self.is_running = False
            return False

    def _run_extract(self, data_source: Any) -> Any:
        """运行提取阶段"""
        try:
            # 模拟数据提取
            if isinstance(data_source, str):
                # 从文件路径提取
                return pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
            elif isinstance(data_source, dict):
                # 从配置提取
                return data_source
            elif isinstance(data_source, pd.DataFrame):
                # 直接返回DataFrame
                return data_source
            else:
                return data_source
        except Exception as e:
            self.stage_stats["extract"]["errors"] += 1
            raise
        finally:
            self.stage_stats["extract"]["processed"] += 1

    def _run_transform(self, data: Any) -> Any:
        """运行转换阶段"""
        try:
            # 模拟数据转换
            if isinstance(data, pd.DataFrame):
                # 数据转换逻辑 - 使用第一列进行转换
                if not data.empty and len(data.columns) > 0:
                    first_col = data.columns[0]
                    data["transformed"] = data[first_col] * 2
                return data
            else:
                return data
        except Exception as e:
            self.stage_stats["transform"]["errors"] += 1
            raise
        finally:
            self.stage_stats["transform"]["processed"] += 1

    def _run_load(self, data: Any) -> bool:
        """运行加载阶段"""
        try:
            # 模拟数据加载
            if isinstance(data, pd.DataFrame):
                # 验证数据完整性
                if data.empty:
                    raise ValueError("Cannot load empty data")

            self.logger.info("Data loaded successfully")
            return True
        except Exception as e:
            self.stage_stats["load"]["errors"] += 1
            raise
        finally:
            self.stage_stats["load"]["processed"] += 1

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """获取管道统计"""
        return {
            "pipeline_id": self.pipeline_id,
            "is_running": self.is_running,
            "stage_stats": self.stage_stats,
            "total_processed": sum(s["processed"] for s in self.stage_stats.values()),
            "total_errors": sum(s["errors"] for s in self.stage_stats.values())
        }


class MockMultiMarketSync:
    """模拟多市场同步"""

    def __init__(self, sync_id: str, config: MockDataFlowConfig):
        self.sync_id = sync_id
        self.config = config
        self.is_syncing = False
        self.markets = {}
        self.sync_stats = {
            "total_synced": 0,
            "markets_active": 0,
            "sync_errors": 0,
            "last_sync_time": None
        }
        self.logger = Mock()

    def add_market(self, market_id: str, market_config: Dict[str, Any]) -> bool:
        """添加市场"""
        try:
            self.markets[market_id] = {
                "config": market_config,
                "is_active": False,
                "last_sync": None,
                "sync_count": 0
            }
            return True
        except Exception:
            return False

    def start_sync(self) -> bool:
        """启动同步"""
        try:
            self.is_syncing = True
            for market_id, market_info in self.markets.items():
                market_info["is_active"] = True
                market_info["last_sync"] = datetime.now()

            self.sync_stats["markets_active"] = len([m for m in self.markets.values() if m["is_active"]])
            self.logger.info(f"Multi-market sync {self.sync_id} started")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start sync: {e}")
            return False

    def stop_sync(self) -> bool:
        """停止同步"""
        try:
            self.is_syncing = False
            for market_info in self.markets.values():
                market_info["is_active"] = False

            self.sync_stats["markets_active"] = 0
            self.logger.info(f"Multi-market sync {self.sync_id} stopped")
            return True
        except Exception as e:
            self.logger.error(f"Failed to stop sync: {e}")
            return False

    def sync_market_data(self, market_id: str, data: Any) -> bool:
        """同步市场数据"""
        if not self.is_syncing:
            return False

        if market_id not in self.markets:
            self.logger.error(f"Market {market_id} not found")
            return False

        try:
            # 模拟数据同步
            self.markets[market_id]["sync_count"] += 1
            self.markets[market_id]["last_sync"] = datetime.now()
            self.sync_stats["total_synced"] += 1
            self.sync_stats["last_sync_time"] = datetime.now()

            return True
        except Exception as e:
            self.sync_stats["sync_errors"] += 1
            self.logger.error(f"Failed to sync market {market_id}: {e}")
            return False

    def get_sync_stats(self) -> Dict[str, Any]:
        """获取同步统计"""
        return {
            "sync_id": self.sync_id,
            "is_syncing": self.is_syncing,
            "markets": self.markets,
            "sync_stats": self.sync_stats
        }


class MockErrorRecovery:
    """模拟错误恢复"""

    def __init__(self, recovery_id: str, config: MockDataFlowConfig):
        self.recovery_id = recovery_id
        self.config = config
        self.recovery_attempts = 0
        self.successful_recoveries = 0
        self.failed_recoveries = 0
        self.recovery_strategies = ["retry", "fallback", "circuit_breaker"]
        self.logger = Mock()

    def recover_from_error(self, error_context: Dict[str, Any]) -> bool:
        """从错误中恢复"""
        self.recovery_attempts += 1

        try:
            error_type = error_context.get("error_type", "unknown")
            strategy = self._select_recovery_strategy(error_type)

            if strategy == "retry":
                success = self._retry_operation(error_context)
            elif strategy == "fallback":
                success = self._fallback_operation(error_context)
            elif strategy == "circuit_breaker":
                success = self._circuit_breaker_recovery(error_context)
            else:
                # 未知策略，直接失败
                success = False

            if success:
                self.successful_recoveries += 1
            else:
                self.failed_recoveries += 1

            return success

        except Exception as e:
            self.failed_recoveries += 1
            self.logger.error(f"Recovery failed: {e}")
            return False

    def _select_recovery_strategy(self, error_type: str) -> str:
        """选择恢复策略"""
        if error_type == "network":
            return "retry"
        elif error_type == "data_corruption":
            return "fallback"
        elif error_type == "service_unavailable":
            return "circuit_breaker"
        else:
            return "unknown"  # 未知错误类型

    def _retry_operation(self, error_context: Dict[str, Any]) -> bool:
        """重试操作"""
        max_retries = self.config.retry_attempts
        for attempt in range(max_retries):
            try:
                # 模拟重试逻辑
                return True
            except Exception:
                if attempt == max_retries - 1:
                    return False
        return False

    def _fallback_operation(self, error_context: Dict[str, Any]) -> bool:
        """后备操作"""
        try:
            # 模拟后备逻辑
            return True
        except Exception:
            return False

    def _circuit_breaker_recovery(self, error_context: Dict[str, Any]) -> bool:
        """断路器恢复"""
        try:
            # 模拟断路器逻辑
            return True
        except Exception:
            return False

    def get_recovery_stats(self) -> Dict[str, Any]:
        """获取恢复统计"""
        return {
            "recovery_id": self.recovery_id,
            "recovery_attempts": self.recovery_attempts,
            "successful_recoveries": self.successful_recoveries,
            "failed_recoveries": self.failed_recoveries,
            "success_rate": self.successful_recoveries / max(1, self.recovery_attempts)
        }


class TestMockDataFlowConfig:
    """模拟数据流配置测试"""

    def test_config_creation(self):
        """测试配置创建"""
        config = MockDataFlowConfig(
            buffer_size=2000,
            max_workers=8,
            retry_attempts=5
        )

        assert config.buffer_size == 2000
        assert config.max_workers == 8
        assert config.retry_attempts == 5
        assert config.enable_compression is True

    def test_config_to_dict(self):
        """测试配置序列化"""
        config = MockDataFlowConfig(sync_mode="batch", timeout=60.0)
        data = config.to_dict()

        assert data["sync_mode"] == "batch"
        assert data["timeout"] == 60.0


class TestMockDataFlow:
    """模拟数据流测试"""

    def setup_method(self):
        """设置测试方法"""
        self.config = MockDataFlowConfig(buffer_size=100, max_workers=4)
        self.flow = MockDataFlow("test_flow", self.config)

    def test_flow_initialization(self):
        """测试流初始化"""
        assert self.flow.flow_id == "test_flow"
        assert not self.flow.is_active
        assert len(self.flow.buffer) == 0

    def test_flow_start_stop(self):
        """测试流启动和停止"""
        assert self.flow.start_flow()
        assert self.flow.is_active

        assert self.flow.stop_flow()
        assert not self.flow.is_active

    def test_add_data_to_flow(self):
        """测试向流添加数据"""
        self.flow.start_flow()

        # 添加数据
        assert self.flow.add_data({"symbol": "AAPL", "price": 150.0})
        assert len(self.flow.buffer) == 1

        # 添加到满的缓冲区
        for i in range(99):
            self.flow.add_data(f"data_{i}")

        assert len(self.flow.buffer) == 100

        # 缓冲区满时应该拒绝新数据
        assert not self.flow.add_data("overflow_data")
        assert len(self.flow.buffer) == 100

    def test_process_batch(self):
        """测试批处理"""
        self.flow.start_flow()

        # 添加测试数据
        test_data = [
            {"symbol": "AAPL", "price": 150.0},  # 成功
            {"invalid": "data"},  # 无效数据 - 失败
            pd.DataFrame(),  # 空DataFrame - 失败
            {"symbol": "GOOGL", "price": 2500.0},  # 成功
            {"symbol": "MSFT", "price": 300.0}  # 成功
        ]

        for data in test_data:
            self.flow.add_data(data)

        # 处理一批
        processed = self.flow.process_batch(3)
        assert processed == 1  # 1个成功处理，2个失败

        # 检查统计
        stats = self.flow.get_flow_stats()
        assert stats["processed_count"] == 1
        assert stats["error_count"] == 2
        assert stats["buffer_size"] == 4  # 剩余2个失败的项目 + 2个未处理的项目

    def test_flow_stats(self):
        """测试流统计"""
        self.flow.start_flow()

        # 进行一些操作
        self.flow.add_data("test1")
        self.flow.add_data("test2")
        self.flow.process_batch(2)

        stats = self.flow.get_flow_stats()
        assert stats["flow_id"] == "test_flow"
        assert stats["is_active"] is True
        assert stats["processed_count"] == 2
        assert "error_rate" in stats


class TestMockETLPipeline:
    """模拟ETL管道测试"""

    def setup_method(self):
        """设置测试方法"""
        self.config = MockDataFlowConfig()
        self.pipeline = MockETLPipeline("test_pipeline", self.config)

    def test_pipeline_initialization(self):
        """测试管道初始化"""
        assert self.pipeline.pipeline_id == "test_pipeline"
        assert not self.pipeline.is_running
        assert len(self.pipeline.stages) == 3

    def test_run_complete_pipeline(self):
        """测试运行完整管道"""
        # 从文件路径运行
        result = self.pipeline.run_pipeline("/path/to/data.csv")
        assert result is True
        assert not self.pipeline.is_running

        # 检查统计
        stats = self.pipeline.get_pipeline_stats()
        assert stats["total_processed"] == 3  # 三个阶段
        assert stats["total_errors"] == 0

    def test_pipeline_with_invalid_data(self):
        """测试管道处理无效数据"""
        # 运行空数据
        result = self.pipeline.run_pipeline(pd.DataFrame())
        assert result is False  # Load阶段应该失败

        stats = self.pipeline.get_pipeline_stats()
        assert stats["total_errors"] > 0

    def test_pipeline_stages(self):
        """测试管道各个阶段"""
        # 测试提取阶段
        data = self.pipeline._run_extract("test_path")
        assert isinstance(data, pd.DataFrame)

        # 测试转换阶段
        transformed = self.pipeline._run_transform(data)
        assert "transformed" in transformed.columns

        # 测试加载阶段
        success = self.pipeline._run_load(transformed)
        assert success is True


class TestMockMultiMarketSync:
    """模拟多市场同步测试"""

    def setup_method(self):
        """设置测试方法"""
        self.config = MockDataFlowConfig()
        self.sync = MockMultiMarketSync("test_sync", self.config)

    def test_sync_initialization(self):
        """测试同步初始化"""
        assert self.sync.sync_id == "test_sync"
        assert not self.sync.is_syncing
        assert len(self.sync.markets) == 0

    def test_add_market(self):
        """测试添加市场"""
        market_config = {"exchange": "NYSE", "timezone": "America/New_York"}

        assert self.sync.add_market("NYSE", market_config)
        assert "NYSE" in self.sync.markets
        assert self.sync.markets["NYSE"]["config"] == market_config

    def test_start_stop_sync(self):
        """测试启动和停止同步"""
        # 添加市场
        self.sync.add_market("NYSE", {})
        self.sync.add_market("LSE", {})

        # 启动同步
        assert self.sync.start_sync()
        assert self.sync.is_syncing

        stats = self.sync.get_sync_stats()
        assert stats["sync_stats"]["markets_active"] == 2

        # 停止同步
        assert self.sync.stop_sync()
        assert not self.sync.is_syncing
        assert stats["sync_stats"]["markets_active"] == 0

    def test_sync_market_data(self):
        """测试同步市场数据"""
        self.sync.add_market("NYSE", {})
        self.sync.start_sync()

        # 同步数据
        test_data = {"symbol": "AAPL", "price": 150.0}
        assert self.sync.sync_market_data("NYSE", test_data)

        # 检查统计
        stats = self.sync.get_sync_stats()
        assert stats["sync_stats"]["total_synced"] == 1
        assert self.sync.markets["NYSE"]["sync_count"] == 1

        # 同步不存在的市场
        assert not self.sync.sync_market_data("INVALID", test_data)

    def test_sync_stats(self):
        """测试同步统计"""
        self.sync.add_market("NYSE", {})
        self.sync.start_sync()

        # 进行同步
        self.sync.sync_market_data("NYSE", "data1")
        self.sync.sync_market_data("NYSE", "data2")

        stats = self.sync.get_sync_stats()
        assert stats["is_syncing"] is True
        assert stats["sync_stats"]["total_synced"] == 2
        assert stats["markets"]["NYSE"]["sync_count"] == 2


class TestMockErrorRecovery:
    """模拟错误恢复测试"""

    def setup_method(self):
        """设置测试方法"""
        self.config = MockDataFlowConfig(retry_attempts=3)
        self.recovery = MockErrorRecovery("test_recovery", self.config)

    def test_recovery_initialization(self):
        """测试恢复初始化"""
        assert self.recovery.recovery_id == "test_recovery"
        assert self.recovery.recovery_attempts == 0
        assert len(self.recovery.recovery_strategies) == 3

    def test_recover_from_network_error(self):
        """测试从网络错误恢复"""
        error_context = {"error_type": "network", "operation": "fetch_data"}

        result = self.recovery.recover_from_error(error_context)
        assert result is True

        stats = self.recovery.get_recovery_stats()
        assert stats["recovery_attempts"] == 1
        assert stats["successful_recoveries"] == 1

    def test_recover_from_service_error(self):
        """测试从服务错误恢复"""
        error_context = {"error_type": "service_unavailable", "service": "api"}

        result = self.recovery.recover_from_error(error_context)
        assert result is True

        stats = self.recovery.get_recovery_stats()
        assert stats["successful_recoveries"] == 1

    def test_recovery_failure(self):
        """测试恢复失败"""
        error_context = {"error_type": "unknown", "operation": "invalid"}

        result = self.recovery.recover_from_error(error_context)
        assert result is False

        stats = self.recovery.get_recovery_stats()
        assert stats["failed_recoveries"] == 1

    def test_recovery_stats(self):
        """测试恢复统计"""
        # 模拟多次恢复
        for i in range(5):
            error_type = "network" if i < 3 else "unknown"
            error_context = {"error_type": error_type}
            self.recovery.recover_from_error(error_context)

        stats = self.recovery.get_recovery_stats()
        assert stats["recovery_attempts"] == 5
        assert stats["successful_recoveries"] == 3  # 3个网络错误恢复成功
        assert stats["failed_recoveries"] == 2  # 2个未知错误恢复失败
        assert abs(stats["success_rate"] - 0.6) < 0.001


class TestDataFlowIntegration:
    """数据流集成测试"""

    def test_complete_data_flow_pipeline(self):
        """测试完整数据流管道"""
        # 创建数据流
        flow_config = MockDataFlowConfig(buffer_size=50, enable_compression=True)
        data_flow = MockDataFlow("integration_flow", flow_config)

        # 创建ETL管道
        etl_pipeline = MockETLPipeline("integration_pipeline", flow_config)

        # 创建多市场同步
        market_sync = MockMultiMarketSync("integration_sync", flow_config)

        # 创建错误恢复
        error_recovery = MockErrorRecovery("integration_recovery", flow_config)

        # 初始化所有组件
        assert data_flow.start_flow()
        assert etl_pipeline.initialize_pipeline()
        assert market_sync.start_sync()

        # 添加市场
        market_sync.add_market("NYSE", {"exchange": "NYSE"})
        market_sync.add_market("LSE", {"exchange": "LSE"})

        try:
            # 1. ETL管道处理数据
            source_data = pd.DataFrame({
                "symbol": ["AAPL", "GOOGL", "MSFT"],
                "price": [150.0, 2500.0, 300.0],
                "volume": [1000, 2000, 1500]
            })

            etl_success = etl_pipeline.run_pipeline(source_data)
            assert etl_success

            # 2. 将处理后的数据添加到数据流
            processed_data = source_data.copy()
            processed_data["processed"] = True

            for _, row in processed_data.iterrows():
                data_flow.add_data(row.to_dict())

            # 3. 处理数据流
            processed_count = data_flow.process_batch(10)
            assert processed_count == 3

            # 4. 同步到多个市场
            for market_id in ["NYSE", "LSE"]:
                for _, row in processed_data.iterrows():
                    market_sync.sync_market_data(market_id, row.to_dict())

            # 验证统计
            flow_stats = data_flow.get_flow_stats()
            etl_stats = etl_pipeline.get_pipeline_stats()
            sync_stats = market_sync.get_sync_stats()
            recovery_stats = error_recovery.get_recovery_stats()

            assert flow_stats["processed_count"] == 3
            assert etl_stats["total_processed"] == 3
            assert sync_stats["sync_stats"]["total_synced"] == 6  # 2个市场 * 3条数据

        finally:
            # 清理资源
            data_flow.stop_flow()
            market_sync.stop_sync()

    def test_error_handling_in_data_flow(self):
        """测试数据流中的错误处理"""
        flow_config = MockDataFlowConfig(retry_attempts=2)
        data_flow = MockDataFlow("error_flow", flow_config)
        error_recovery = MockErrorRecovery("error_recovery", flow_config)

        data_flow.start_flow()

        # 添加正常数据和错误数据
        data_flow.add_data({"symbol": "AAPL", "price": 150.0})  # 正常
        data_flow.add_data({"invalid": "data"})  # 错误数据
        data_flow.add_data(pd.DataFrame())  # 空DataFrame

        # 处理数据
        processed = data_flow.process_batch(10)

        # 应该只有1个成功处理，2个错误
        assert processed == 1

        flow_stats = data_flow.get_flow_stats()
        assert flow_stats["processed_count"] == 1
        assert flow_stats["error_count"] == 2

        # 测试错误恢复
        error_context = {"error_type": "network", "component": "data_flow"}
        recovery_success = error_recovery.recover_from_error(error_context)
        assert recovery_success

        recovery_stats = error_recovery.get_recovery_stats()
        assert recovery_stats["successful_recoveries"] == 1

    def test_performance_monitoring(self):
        """测试性能监控"""
        flow_config = MockDataFlowConfig(buffer_size=1000, max_workers=8)
        data_flow = MockDataFlow("perf_flow", flow_config)

        data_flow.start_flow()

        # 模拟高负载数据处理
        import time
        start_time = time.time()

        for i in range(100):
            data_flow.add_data({"symbol": f"SYMBOL_{i}", "price": 100.0 + i})

        processed = data_flow.process_batch(100)
        end_time = time.time()

        processing_time = end_time - start_time
        throughput = processed / processing_time if processing_time > 0 else 100  # 避免除零

        # 更新流的吞吐量
        data_flow.throughput = throughput

        stats = data_flow.get_flow_stats()
        assert stats["processed_count"] == 100
        assert stats["throughput"] > 0
        assert stats["buffer_size"] == 0  # 所有数据都被处理

    def test_concurrent_data_flow_processing(self):
        """测试并发数据流处理"""
        import threading

        flow_config = MockDataFlowConfig(buffer_size=1000)
        data_flow = MockDataFlow("concurrent_flow", flow_config)
        data_flow.start_flow()

        results = []
        errors = []

        def producer():
            """生产者线程"""
            for i in range(50):
                success = data_flow.add_data(f"data_{i}")
                if not success:
                    errors.append(f"Failed to add data_{i}")

        def consumer():
            """消费者线程"""
            processed = data_flow.process_batch(25)
            results.append(processed)

        # 创建线程
        producer_thread = threading.Thread(target=producer)
        consumer_thread = threading.Thread(target=consumer)

        # 启动线程
        producer_thread.start()
        consumer_thread.start()

        # 等待完成
        producer_thread.join()
        consumer_thread.join()

        # 验证结果
        assert len(errors) == 0  # 没有添加错误
        assert len(results) == 1
        assert results[0] == 25  # 消费了25个项目

        final_stats = data_flow.get_flow_stats()
        assert final_stats["processed_count"] == 25
        assert final_stats["buffer_size"] == 25  # 剩余25个未处理的项目


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
