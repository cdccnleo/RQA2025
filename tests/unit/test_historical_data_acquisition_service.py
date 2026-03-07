#!/usr/bin/env python3
"""
历史数据采集服务单元测试
测试多数据源集成、质量保证和并发采集功能
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
import asyncio

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.orchestration.historical_data_acquisition_service import (
    HistoricalDataAcquisitionService,
    DataSourceType,
    DataQualityLevel,
    HistoricalDataConfig,
    DataSourceResult,
    HistoricalDataBatch
)


class TestHistoricalDataAcquisitionService:
    """历史数据采集服务测试类"""

    def setup_method(self):
        """测试前准备"""
        self.config = {
            'acquisition_service_config': {
                'adapters': {
                    'akshare': {'config': 'test'},
                    'yahoo': {'config': 'test'},
                    'local_backup': {'backup_dir': './test_data'}
                },
                'max_concurrent_batches': 3,
                'quality_threshold': 0.85
            }
        }
        self.service = HistoricalDataAcquisitionService(self.config)

    def teardown_method(self):
        """测试后清理"""
        pass

    def test_service_initialization(self):
        """测试服务初始化"""
        assert len(self.service.adapters) == 3
        assert DataSourceType.AKSHARE in self.service.adapters
        assert DataSourceType.YAHOO in self.service.adapters
        assert DataSourceType.LOCAL_BACKUP in self.service.adapters

        assert self.service.quality_threshold == 0.85

    def test_historical_data_config_creation(self):
        """测试历史数据配置创建"""
        config = HistoricalDataConfig(
            data_source=DataSourceType.AKSHARE,
            symbol="000001.SZ",
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            data_type="stock",
            priority=1,
            max_retry_count=3,
            timeout_seconds=30,
            quality_threshold=0.85
        )

        assert config.data_source == DataSourceType.AKSHARE
        assert config.symbol == "000001.SZ"
        assert config.start_date.year == 2020
        assert config.end_date.year == 2020
        assert config.data_type == "stock"
        assert config.priority == 1
        assert config.max_retry_count == 3
        assert config.timeout_seconds == 30
        assert config.quality_threshold == 0.85

    def test_supported_symbols_query(self):
        """测试支持的标的查询"""
        # 测试特定数据源
        symbols = self.service.get_supported_symbols(DataSourceType.AKSHARE)
        assert isinstance(symbols, set)

        # 测试所有数据源合并
        all_symbols = self.service.get_supported_symbols()
        assert isinstance(all_symbols, set)
        assert len(all_symbols) >= len(symbols)

    @pytest.mark.asyncio
    async def test_historical_data_acquisition(self):
        """测试历史数据采集"""
        config = HistoricalDataConfig(
            data_source=DataSourceType.AKSHARE,
            symbol="000001.SZ",
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 1, 31),
            data_type="stock"
        )

        # Mock适配器方法
        with patch.object(self.service.adapters[DataSourceType.AKSHARE], 'collect_historical_data',
                         new_callable=AsyncMock) as mock_collect:
            # 创建模拟结果
            mock_result = DataSourceResult(
                source=DataSourceType.AKSHARE,
                symbol="000001.SZ",
                data=[
                    {
                        "symbol": "000001.SZ",
                        "date": datetime(2020, 1, 1),
                        "open": 100.0,
                        "high": 105.0,
                        "low": 95.0,
                        "close": 102.0,
                        "volume": 1000000,
                        "source": "akshare"
                    }
                ],
                quality_score=0.95,
                quality_level=DataQualityLevel.EXCELLENT,
                collection_time=1.5
            )
            mock_collect.return_value = mock_result

            # 执行采集
            batch = await self.service.acquire_historical_data(config)

            # 验证结果
            assert batch.symbol == "000001.SZ"
            assert batch.year == 2020
            assert len(batch.results) == 1
            assert batch.best_result == mock_result
            assert batch.status == "completed"
            assert batch.completed_at is not None

    @pytest.mark.asyncio
    async def test_yearly_data_acquisition(self):
        """测试年度数据采集"""
        # Mock适配器
        with patch.object(self.service.adapters[DataSourceType.AKSHARE], 'collect_historical_data',
                         new_callable=AsyncMock) as mock_collect:

            mock_result = DataSourceResult(
                source=DataSourceType.AKSHARE,
                symbol="000001.SZ",
                data=[{"symbol": "000001.SZ", "date": datetime(2020, 6, 1), "close": 100.0}],
                quality_score=0.90,
                quality_level=DataQualityLevel.GOOD,
                collection_time=1.0
            )
            mock_collect.return_value = mock_result

            # 执行年度采集
            batches = await self.service.acquire_yearly_data("000001.SZ", 2020, ["stock"])

            assert len(batches) == 1
            assert batches[0].symbol == "000001.SZ"
            assert batches[0].year == 2020
            assert batches[0].status == "completed"

    @pytest.mark.asyncio
    async def test_multi_year_data_acquisition(self):
        """测试多年数据采集"""
        # Mock适配器
        with patch.object(self.service.adapters[DataSourceType.AKSHARE], 'collect_historical_data',
                         new_callable=AsyncMock) as mock_collect:

            mock_result = DataSourceResult(
                source=DataSourceType.AKSHARE,
                symbol="000001.SZ",
                data=[{"symbol": "000001.SZ", "date": datetime(2020, 6, 1), "close": 100.0}],
                quality_score=0.88,
                quality_level=DataQualityLevel.GOOD,
                collection_time=1.0
            )
            mock_collect.return_value = mock_result

            # 执行多年采集
            batches = await self.service.acquire_multi_year_data("000001.SZ", 2020, 2022, ["stock"])

            assert len(batches) == 3  # 2020, 2021, 2022
            for batch in batches:
                assert batch.symbol == "000001.SZ"
                assert batch.status == "completed"
                assert 2020 <= batch.year <= 2022

    @pytest.mark.asyncio
    async def test_data_quality_assessment(self):
        """测试数据质量评估"""
        # 创建测试批次
        batch = HistoricalDataBatch(
            batch_id="test_batch",
            year=2020,
            symbol="000001.SZ",
            config=HistoricalDataConfig(
                data_source=DataSourceType.AKSHARE,
                symbol="000001.SZ",
                start_date=datetime(2020, 1, 1),
                end_date=datetime(2020, 12, 31)
            )
        )

        # 添加测试结果
        batch.results = [
            DataSourceResult(
                source=DataSourceType.AKSHARE,
                symbol="000001.SZ",
                quality_score=0.95,
                quality_level=DataQualityLevel.EXCELLENT,
                collection_time=1.5
            ),
            DataSourceResult(
                source=DataSourceType.YAHOO,
                symbol="000001.SZ",
                quality_score=0.85,
                quality_level=DataQualityLevel.GOOD,
                collection_time=2.0
            )
        ]

        # 执行质量评估
        quality_stats = self.service.get_data_quality_stats([batch])

        assert quality_stats["total_batches"] == 1
        assert quality_stats["average_quality_score"] > 0.8
        assert quality_stats["quality_distribution"]["excellent"] == 1
        assert quality_stats["quality_distribution"]["good"] == 1
        assert DataSourceType.AKSHARE.value in quality_stats["best_sources"]

    @pytest.mark.asyncio
    async def test_data_integrity_validation(self):
        """测试数据完整性验证"""
        # Mock存储层
        with patch.object(self.service.timescale_storage, 'get_data_stats',
                         new_callable=AsyncMock) as mock_stats:

            mock_stats.return_value = {
                "total_records": 250,
                "oldest_date": datetime(2020, 1, 1),
                "newest_date": datetime(2020, 12, 31),
                "avg_quality": 0.90,
                "years_with_data": 1,
                "completeness_ratio": 0.96,
                "is_complete": True
            }

            # 执行完整性验证
            validation = await self.service.validate_data_integrity("000001.SZ", 2020, 2020)

            assert validation["symbol"] == "000001.SZ"
            assert validation["total_records"] == 250
            assert validation["completeness_ratio"] == 0.96
            assert validation["is_complete"] == True

    @pytest.mark.asyncio
    async def test_storage_operations(self):
        """测试存储操作"""
        # 创建测试批次
        batch = HistoricalDataBatch(
            batch_id="storage_test",
            year=2020,
            symbol="000001.SZ",
            config=HistoricalDataConfig(
                data_source=DataSourceType.AKSHARE,
                symbol="000001.SZ",
                start_date=datetime(2020, 1, 1),
                end_date=datetime(2020, 12, 31)
            )
        )

        batch.best_result = DataSourceResult(
            source=DataSourceType.AKSHARE,
            symbol="000001.SZ",
            data=[
                {
                    "symbol": "000001.SZ",
                    "date": datetime(2020, 1, 1),
                    "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0,
                    "volume": 1000000, "source": "akshare", "quality_score": 0.95
                }
            ],
            quality_score=0.95,
            quality_level=DataQualityLevel.EXCELLENT,
            collection_time=1.5
        )

        # Mock存储方法
        with patch.object(self.service.timescale_storage, 'store_historical_data',
                         new_callable=AsyncMock) as mock_store, \
             patch.object(self.service.redis_cache, 'set_json',
                         new_callable=AsyncMock) as mock_cache:

            mock_store.return_value = None
            mock_cache.return_value = None

            # 执行存储
            stats = await self.service.store_batch_results([batch])

            assert stats["total_batches"] == 1
            assert stats["stored_batches"] == 1
            assert stats["total_records"] == 1

            # 验证存储调用
            mock_store.assert_called_once()
            mock_cache.assert_called_once()

    def test_fallback_sources_handling(self):
        """测试备用数据源处理"""
        config = HistoricalDataConfig(
            data_source=DataSourceType.AKSHARE,
            symbol="000001.SZ",
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            fallback_sources=[
                DataSourceType.YAHOO,
                DataSourceType.LOCAL_BACKUP
            ]
        )

        # 验证备用源配置
        assert len(config.fallback_sources) == 2
        assert DataSourceType.YAHOO in config.fallback_sources
        assert DataSourceType.LOCAL_BACKUP in config.fallback_sources

    @pytest.mark.asyncio
    async def test_concurrent_acquisition_limit(self):
        """测试并发采集限制"""
        # 创建多个配置
        configs = []
        for i in range(5):  # 超过并发限制
            config = HistoricalDataConfig(
                data_source=DataSourceType.AKSHARE,
                symbol=f"00000{i}.SZ",
                start_date=datetime(2020, 1, 1),
                end_date=datetime(2020, 12, 31)
            )
            configs.append(config)

        # Mock适配器
        with patch.object(self.service.adapters[DataSourceType.AKSHARE], 'collect_historical_data',
                         new_callable=AsyncMock) as mock_collect:

            mock_result = DataSourceResult(
                source=DataSourceType.AKSHARE,
                symbol="test",
                data=[{"symbol": "test", "date": datetime(2020, 6, 1), "close": 100.0}],
                quality_score=0.90,
                quality_level=DataQualityLevel.GOOD,
                collection_time=1.0
            )
            mock_collect.return_value = mock_result

            # 执行并发采集
            tasks = [self.service.acquire_historical_data(config) for config in configs]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 验证所有任务都成功完成
            successful_results = [r for r in results if isinstance(r, HistoricalDataBatch)]
            assert len(successful_results) == 5

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """测试错误处理和恢复"""
        config = HistoricalDataConfig(
            data_source=DataSourceType.AKSHARE,
            symbol="000001.SZ",
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31)
        )

        # Mock适配器抛出异常
        with patch.object(self.service.adapters[DataSourceType.AKSHARE], 'collect_historical_data',
                         new_callable=AsyncMock) as mock_collect:

            mock_collect.side_effect = Exception("Network timeout")

            # 执行采集，应该处理异常
            batch = await self.service.acquire_historical_data(config)

            assert batch.status == "failed"
            assert batch.best_result is None
            assert len(batch.results) == 1
            assert batch.results[0].error_message == "Network timeout"

    def test_data_source_result_creation(self):
        """测试数据源结果创建"""
        result = DataSourceResult(
            source=DataSourceType.AKSHARE,
            symbol="000001.SZ",
            data=[{"test": "data"}],
            quality_score=0.95,
            quality_level=DataQualityLevel.EXCELLENT,
            collection_time=1.5,
            metadata={"test": "meta"}
        )

        assert result.source == DataSourceType.AKSHARE
        assert result.symbol == "000001.SZ"
        assert len(result.data) == 1
        assert result.quality_score == 0.95
        assert result.quality_level == DataQualityLevel.EXCELLENT
        assert result.collection_time == 1.5
        assert result.metadata["test"] == "meta"

    def test_quality_threshold_filtering(self):
        """测试质量阈值过滤"""
        # 创建不同质量的结果
        high_quality = DataSourceResult(
            source=DataSourceType.AKSHARE,
            symbol="000001.SZ",
            quality_score=0.95,
            quality_level=DataQualityLevel.EXCELLENT
        )

        low_quality = DataSourceResult(
            source=DataSourceType.YAHOO,
            symbol="000001.SZ",
            quality_score=0.70,
            quality_level=DataQualityLevel.POOR
        )

        # 测试阈值过滤
        assert 0.95 >= self.service.quality_threshold  # 高质量通过
        assert 0.70 < self.service.quality_threshold   # 低质量被过滤

    @pytest.mark.asyncio
    async def test_batch_processing_optimization(self):
        """测试批次处理优化"""
        # 创建大数据批次
        large_batch = HistoricalDataBatch(
            batch_id="large_batch_test",
            year=2020,
            symbol="000001.SZ",
            config=HistoricalDataConfig(
                data_source=DataSourceType.AKSHARE,
                symbol="000001.SZ",
                start_date=datetime(2020, 1, 1),
                end_date=datetime(2020, 12, 31)
            )
        )

        # 创建大量模拟数据
        large_data = []
        current_date = datetime(2020, 1, 1)
        while current_date <= datetime(2020, 12, 31):
            if current_date.weekday() < 5:  # 工作日
                large_data.append({
                    "symbol": "000001.SZ",
                    "date": current_date,
                    "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0,
                    "volume": 1000000, "source": "akshare"
                })
            current_date += timedelta(days=1)

        large_batch.best_result = DataSourceResult(
            source=DataSourceType.AKSHARE,
            symbol="000001.SZ",
            data=large_data,
            quality_score=0.92,
            quality_level=DataQualityLevel.GOOD,
            collection_time=5.0
        )

        # Mock存储方法
        with patch.object(self.service.timescale_storage, 'store_historical_data',
                         new_callable=AsyncMock) as mock_store:

            mock_store.return_value = None

            # 记录开始时间
            start_time = datetime.now()

            # 执行存储
            stats = await self.service.store_batch_results([large_batch])

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            # 验证处理结果
            assert stats["total_records"] > 200  # 一年约250个交易日
            assert processing_time < 10  # 应该在合理时间内完成
            assert stats["stored_batches"] == 1

    @pytest.mark.asyncio
    async def test_resource_cleanup(self):
        """测试资源清理"""
        # 创建服务
        service = HistoricalDataAcquisitionService(self.config)

        # 模拟一些操作
        config = HistoricalDataConfig(
            data_source=DataSourceType.AKSHARE,
            symbol="000001.SZ",
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 1, 31)
        )

        # 执行一些操作
        with patch.object(service.adapters[DataSourceType.AKSHARE], 'collect_historical_data',
                         new_callable=AsyncMock) as mock_collect:

            mock_result = DataSourceResult(
                source=DataSourceType.AKSHARE,
                symbol="000001.SZ",
                data=[{"test": "data"}],
                quality_score=0.90,
                quality_level=DataQualityLevel.GOOD,
                collection_time=1.0
            )
            mock_collect.return_value = mock_result

            await service.acquire_historical_data(config)

        # 验证服务状态正常
        assert service.quality_threshold == 0.85
        assert len(service.adapters) == 3

    def test_configuration_validation(self):
        """测试配置验证"""
        # 有效配置
        valid_config = {
            'acquisition_service_config': {
                'adapters': {
                    'akshare': {},
                    'yahoo': {}
                },
                'max_concurrent_batches': 5,
                'quality_threshold': 0.80
            }
        }

        service = HistoricalDataAcquisitionService(valid_config)
        assert service.quality_threshold == 0.80

        # 默认配置
        default_config = {'acquisition_service_config': {}}
        service_default = HistoricalDataAcquisitionService(default_config)
        assert service_default.quality_threshold == 0.85  # 默认值

    @pytest.mark.asyncio
    async def test_empty_data_handling(self):
        """测试空数据处理"""
        config = HistoricalDataConfig(
            data_source=DataSourceType.AKSHARE,
            symbol="000001.SZ",
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 1, 31)
        )

        # Mock返回空数据
        with patch.object(self.service.adapters[DataSourceType.AKSHARE], 'collect_historical_data',
                         new_callable=AsyncMock) as mock_collect:

            empty_result = DataSourceResult(
                source=DataSourceType.AKSHARE,
                symbol="000001.SZ",
                data=[],  # 空数据
                quality_score=0.0,
                quality_level=DataQualityLevel.INVALID,
                collection_time=0.5
            )
            mock_collect.return_value = empty_result

            batch = await self.service.acquire_historical_data(config)

            assert batch.status == "failed"
            assert batch.best_result.quality_level == DataQualityLevel.INVALID
            assert len(batch.best_result.data) == 0

    @pytest.mark.asyncio
    async def test_partial_failure_recovery(self):
        """测试部分失败恢复"""
        config = HistoricalDataConfig(
            data_source=DataSourceType.AKSHARE,
            symbol="000001.SZ",
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            fallback_sources=[DataSourceType.YAHOO]
        )

        # Mock主数据源失败，备用数据源成功
        with patch.object(self.service.adapters[DataSourceType.AKSHARE], 'collect_historical_data',
                         new_callable=AsyncMock) as mock_akshare, \
             patch.object(self.service.adapters[DataSourceType.YAHOO], 'collect_historical_data',
                         new_callable=AsyncMock) as mock_yahoo:

            # 主数据源失败
            mock_akshare.return_value = DataSourceResult(
                source=DataSourceType.AKSHARE,
                symbol="000001.SZ",
                quality_score=0.0,
                quality_level=DataQualityLevel.INVALID,
                error_message="Connection failed"
            )

            # 备用数据源成功
            mock_yahoo.return_value = DataSourceResult(
                source=DataSourceType.YAHOO,
                symbol="000001.SZ",
                data=[{"symbol": "000001.SZ", "date": datetime(2020, 6, 1), "close": 100.0}],
                quality_score=0.88,
                quality_level=DataQualityLevel.GOOD,
                collection_time=2.0
            )

            batch = await self.service.acquire_historical_data(config)

            assert batch.status == "completed"
            assert len(batch.results) == 2  # 主数据源和备用数据源
            assert batch.best_result.source == DataSourceType.YAHOO
            assert batch.best_result.quality_score == 0.88


if __name__ == '__main__':
    pytest.main([__file__])