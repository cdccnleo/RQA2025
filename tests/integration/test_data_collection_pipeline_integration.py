#!/usr/bin/env python3
"""
数据采集管道集成测试
测试完整的数据采集、质量保证、存储流程
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
import asyncio
import tempfile
import os

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.orchestration.historical_data_acquisition_service import (
    HistoricalDataAcquisitionService,
    HistoricalDataConfig,
    DataSourceType
)
from src.core.orchestration.strategy_backtest_data_workflow import (
    StrategyBacktestDataWorkflow,
    WorkflowConfig
)
from src.core.orchestration.data_quality_manager import (
    DataQualityManager,
    QualityCheckLevel
)
from src.core.persistence.timescale_storage import TimescaleStorage
from src.core.orchestration.performance_optimizer import PerformanceOptimizer


class TestDataCollectionPipelineIntegration:
    """数据采集管道集成测试类"""

    def setup_method(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()

        # 配置各个组件
        self.acquisition_config = {
            'acquisition_service_config': {
                'adapters': {
                    'akshare': {'config': 'test'},
                    'yahoo': {'config': 'test'},
                    'local_backup': {'backup_dir': self.temp_dir}
                },
                'max_concurrent_batches': 2,
                'quality_threshold': 0.80
            }
        }

        self.workflow_config = {
            'acquisition_service_config': self.acquisition_config['acquisition_service_config'],
            'timescale_config': {
                'host': 'localhost',
                'port': 5432,
                'database': 'test_db',
                'user': 'test_user',
                'password': 'test_pass'
            },
            'redis_config': {
                'host': 'localhost',
                'port': 6379,
                'db': 0
            },
            'monitor_config': {},
            'max_concurrent_workflows': 1
        }

        self.quality_config = {
            'stock_checker': {
                'thresholds': {
                    'min_completeness': 0.90,
                    'max_missing_rate': 0.10,
                    'outlier_threshold': 3.0
                }
            }
        }

        self.performance_config = {
            'performance': {
                'max_concurrent_downloads': 3,
                'max_concurrent_parsing': 2,
                'batch_size': 500,
                'memory_limit_mb': 256,
                'cpu_limit_percent': 70.0
            }
        }

    def teardown_method(self):
        """测试后清理"""
        # 清理临时文件
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_complete_data_collection_pipeline(self):
        """测试完整数据采集管道"""
        # 初始化组件
        acquisition_service = HistoricalDataAcquisitionService(self.acquisition_config)
        quality_manager = DataQualityManager(self.quality_config)
        performance_optimizer = PerformanceOptimizer(self.performance_config)

        await performance_optimizer.initialize()

        # 创建测试配置
        config = HistoricalDataConfig(
            data_source=DataSourceType.AKSHARE,
            symbol="000001.SZ",
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 1, 31),
            data_type="stock"
        )

        # Mock数据源适配器
        mock_data = [
            {
                "symbol": "000001.SZ",
                "date": datetime(2020, 1, 1),
                "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0,
                "volume": 1000000, "source": "akshare"
            },
            {
                "symbol": "000001.SZ",
                "date": datetime(2020, 1, 2),
                "open": 102.0, "high": 107.0, "low": 98.0, "close": 105.0,
                "volume": 1100000, "source": "akshare"
            }
        ]

        with patch.object(acquisition_service.adapters[DataSourceType.AKSHARE], 'collect_historical_data',
                         new_callable=AsyncMock) as mock_collect:

            from src.core.orchestration.historical_data_acquisition_service import DataSourceResult, DataQualityLevel

            mock_result = DataSourceResult(
                source=DataSourceType.AKSHARE,
                symbol="000001.SZ",
                data=mock_data,
                quality_score=0.95,
                quality_level=DataQualityLevel.EXCELLENT,
                collection_time=1.0
            )
            mock_collect.return_value = mock_result

            # 1. 数据采集阶段
            batch = await acquisition_service.acquire_historical_data(config)
            assert batch.status == "completed"
            assert batch.best_result.quality_score >= 0.8

            # 2. 质量检查阶段
            quality_result = await quality_manager.check_data_quality(
                batch.best_result.data, "stock", QualityCheckLevel.STANDARD
            )
            assert quality_result.overall_score > 0.8

            # 3. 质量修复阶段（如果需要）
            if quality_result.overall_score < 0.9:
                repaired_data, repair_logs = await quality_manager.repair_data_quality(
                    batch.best_result.data, quality_result.issues, repair_level="conservative"
                )
                assert len(repaired_data) == len(batch.best_result.data)

            # 4. 性能优化验证
            optimized_result = await performance_optimizer.optimize_data_collection(
                lambda: asyncio.sleep(0.1) or batch.best_result.data
            )
            assert len(optimized_result) == len(mock_data)

        await performance_optimizer.cleanup()

    @pytest.mark.asyncio
    async def test_workflow_orchestration_integration(self):
        """测试工作流编排集成"""
        # 初始化工作流
        workflow = StrategyBacktestDataWorkflow(self.workflow_config)

        # 创建工作流配置
        workflow_config = WorkflowConfig(
            name="integration_test_workflow",
            symbol="000001.SZ",
            start_year=2020,
            end_year=2020,
            data_types=["stock"],
            max_concurrent_years=1,
            quality_threshold=0.85,
            enable_progress_tracking=True
        )

        # Mock各个组件
        with patch.object(workflow.acquisition_service.adapters[DataSourceType.AKSHARE], 'collect_historical_data',
                         new_callable=AsyncMock) as mock_collect, \
             patch.object(workflow.timescale_storage, 'store_historical_data',
                         new_callable=AsyncMock) as mock_store, \
             patch.object(workflow.timescale_storage, 'get_data_stats',
                         new_callable=AsyncMock) as mock_stats:

            from src.core.orchestration.historical_data_acquisition_service import DataSourceResult, DataQualityLevel

            # Mock采集结果
            mock_data = [
                {
                    "symbol": "000001.SZ",
                    "date": datetime(2020, 6, 1),
                    "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0,
                    "volume": 1000000, "source": "akshare"
                }
            ]

            mock_result = DataSourceResult(
                source=DataSourceType.AKSHARE,
                symbol="000001.SZ",
                data=mock_data,
                quality_score=0.92,
                quality_level=DataQualityLevel.GOOD,
                collection_time=1.0
            )
            mock_collect.return_value = mock_result

            # Mock存储
            mock_store.return_value = None

            # Mock统计
            mock_stats.return_value = {
                "total_records": 1,
                "oldest_date": datetime(2020, 6, 1),
                "newest_date": datetime(2020, 6, 1),
                "avg_quality": 0.92,
                "years_with_data": 1,
                "completeness_ratio": 1.0,
                "is_complete": True
            }

            # 启动工作流
            workflow_id = await workflow.start_workflow(workflow_config)
            assert workflow_id.startswith("workflow_")

            # 等待工作流完成（简化的等待）
            await asyncio.sleep(0.5)

            # 获取工作流状态
            status = workflow.get_workflow_status(workflow_id)
            assert status is not None
            assert status.status in ["collecting", "validating", "storing", "completed"]

            # 如果工作流已完成，验证结果
            if status.status == "completed":
                assert len(status.batches) > 0
                assert status.progress.total_records > 0
                assert status.storage_stats["stored_batches"] >= 0

    @pytest.mark.asyncio
    async def test_multi_component_data_flow(self):
        """测试多组件数据流"""
        # 初始化所有组件
        acquisition_service = HistoricalDataAcquisitionService(self.acquisition_config)
        quality_manager = DataQualityManager(self.quality_config)

        # 创建测试数据流
        test_data = [
            {
                "symbol": "000001.SZ",
                "date": datetime(2020, 1, 1),
                "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0,
                "volume": 1000000, "source": "akshare"
            },
            # 添加一些质量问题
            {
                "symbol": "000001.SZ",
                "date": datetime(2020, 1, 2),
                "open": None, "high": 106.0, "low": 96.0, "close": 103.0,
                "volume": 1100000, "source": "akshare"
            }
        ]

        # 1. 质量检查
        quality_result = await quality_manager.check_data_quality(
            test_data, "stock", QualityCheckLevel.COMPREHENSIVE
        )

        # 2. 质量修复
        repaired_data, repair_logs = await quality_manager.repair_data_quality(
            test_data, quality_result.issues, repair_level="conservative"
        )

        # 3. 验证修复效果
        validation = await quality_manager.validate_repair_effectiveness(
            test_data, repaired_data, "stock"
        )

        # 验证整个流程
        assert validation['original_score'] == quality_result.overall_score
        assert validation['repaired_score'] >= validation['original_score']  # 修复后应该更好或相等
        assert 'improvement' in validation
        assert len(repair_logs) >= 0

    @pytest.mark.asyncio
    async def test_error_recovery_and_resilience(self):
        """测试错误恢复和弹性"""
        acquisition_service = HistoricalDataAcquisitionService(self.acquisition_config)

        config = HistoricalDataConfig(
            data_source=DataSourceType.AKSHARE,
            symbol="000001.SZ",
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 1, 31)
        )

        # Mock连续失败的情况
        with patch.object(acquisition_service.adapters[DataSourceType.AKSHARE], 'collect_historical_data',
                         new_callable=AsyncMock) as mock_collect:

            # 第一次调用失败
            mock_collect.side_effect = [
                Exception("Network timeout"),
                Exception("Server error"),
                # 第三次成功
                DataSourceResult(
                    source=DataSourceType.AKSHARE,
                    symbol="000001.SZ",
                    data=[{"symbol": "000001.SZ", "date": datetime(2020, 1, 1), "close": 100.0}],
                    quality_score=0.85,
                    quality_level=DataQualityLevel.GOOD,
                    collection_time=1.0
                )
            ]

            # 执行采集，应该能够从错误中恢复
            batch = await acquisition_service.acquire_historical_data(config)

            # 验证最终成功
            assert batch.status == "completed"
            assert batch.best_result is not None
            assert batch.best_result.quality_score >= 0.8

    @pytest.mark.asyncio
    async def test_performance_optimization_integration(self):
        """测试性能优化集成"""
        performance_optimizer = PerformanceOptimizer(self.performance_config)
        await performance_optimizer.initialize()

        # 创建一个模拟的性能密集型任务
        async def intensive_task():
            # 模拟数据处理
            data = []
            for i in range(1000):
                data.append({
                    "symbol": f"TEST{i:04d}.SZ",
                    "date": datetime(2020, 1, 1) + timedelta(days=i % 365),
                    "close": 100.0 + (i % 100),
                    "volume": 1000000 + (i % 500000)
                })
            return data

        # 执行性能优化
        start_time = datetime.now()
        result = await performance_optimizer.optimize_data_collection(intensive_task)
        end_time = datetime.now()

        # 验证结果
        assert len(result) == 1000

        # 验证性能监控
        assert len(performance_optimizer.metrics_history) == 1
        metrics = performance_optimizer.metrics_history[0]

        assert metrics.duration_seconds > 0
        assert metrics.throughput_records_per_second > 0
        assert metrics.end_time > metrics.start_time

        # 验证资源使用监控
        report = performance_optimizer.get_performance_report()
        assert "resource_usage" in report
        assert "cache_stats" in report

        await performance_optimizer.cleanup()

    @pytest.mark.asyncio
    async def test_storage_integration_with_quality(self):
        """测试存储与质量保证的集成"""
        # Mock TimescaleDB存储
        mock_storage = AsyncMock()
        mock_storage.store_historical_data.return_value = None
        mock_storage.get_data_stats.return_value = {
            "total_records": 100,
            "oldest_date": datetime(2020, 1, 1),
            "newest_date": datetime(2020, 12, 31),
            "avg_quality": 0.88,
            "completeness_ratio": 0.95,
            "is_complete": True
        }

        # 创建测试数据
        test_data = [
            {
                "symbol": "000001.SZ",
                "date": datetime(2020, i, 1),
                "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0,
                "volume": 1000000, "source": "akshare"
            } for i in range(1, 13)  # 12个月数据
        ]

        with patch('src.core.orchestration.historical_data_acquisition_service.TimescaleStorage',
                   return_value=mock_storage):

            acquisition_service = HistoricalDataAcquisitionService(self.acquisition_config)

            # 执行存储
            config = HistoricalDataConfig(
                data_source=DataSourceType.AKSHARE,
                symbol="000001.SZ",
                start_date=datetime(2020, 1, 1),
                end_date=datetime(2020, 12, 31)
            )

            # Mock采集
            with patch.object(acquisition_service.adapters[DataSourceType.AKSHARE], 'collect_historical_data',
                             new_callable=AsyncMock) as mock_collect:

                from src.core.orchestration.historical_data_acquisition_service import DataSourceResult, DataQualityLevel

                mock_result = DataSourceResult(
                    source=DataSourceType.AKSHARE,
                    symbol="000001.SZ",
                    data=test_data,
                    quality_score=0.90,
                    quality_level=DataQualityLevel.GOOD,
                    collection_time=2.0
                )
                mock_collect.return_value = mock_result

                batch = await acquisition_service.acquire_historical_data(config)
                stats = await acquisition_service.store_batch_results([batch])

                # 验证存储调用
                mock_storage.store_historical_data.assert_called_once()
                assert stats["stored_batches"] == 1
                assert stats["total_records"] == len(test_data)

    def test_configuration_consistency(self):
        """测试配置一致性"""
        # 验证各个组件的配置兼容性
        acquisition_service = HistoricalDataAcquisitionService(self.acquisition_config)

        # 验证适配器配置
        assert DataSourceType.AKSHARE in acquisition_service.adapters
        assert DataSourceType.YAHOO in acquisition_service.adapters
        assert DataSourceType.LOCAL_BACKUP in acquisition_service.adapters

        # 验证质量配置
        quality_manager = DataQualityManager(self.quality_config)
        assert hasattr(quality_manager, 'checkers')
        assert 'stock' in quality_manager.checkers

        # 验证性能配置
        performance_optimizer = PerformanceOptimizer(self.performance_config)
        assert performance_optimizer.config.max_concurrent_downloads > 0
        assert performance_optimizer.config.memory_limit_mb > 0

    @pytest.mark.asyncio
    async def test_end_to_end_workflow_simulation(self):
        """测试端到端工作流模拟"""
        # 创建完整的模拟工作流
        workflow_config = WorkflowConfig(
            name="e2e_test_workflow",
            symbol="000001.SZ",
            start_year=2020,
            end_year=2020,
            data_types=["stock"],
            max_concurrent_years=1,
            quality_threshold=0.80
        )

        # 使用完整的mock设置
        with patch('src.core.orchestration.strategy_backtest_data_workflow.HistoricalDataAcquisitionService') as mock_acquisition_cls, \
             patch('src.core.orchestration.strategy_backtest_data_workflow.TimescaleStorage') as mock_storage_cls, \
             patch('src.core.orchestration.strategy_backtest_data_workflow.RedisCache') as mock_cache_cls, \
             patch('src.core.orchestration.strategy_backtest_data_workflow.DataCollectionMonitor') as mock_monitor_cls:

            # 创建mock实例
            mock_acquisition = AsyncMock()
            mock_storage = AsyncMock()
            mock_cache = AsyncMock()
            mock_monitor = AsyncMock()

            mock_acquisition_cls.return_value = mock_acquisition
            mock_storage_cls.return_value = mock_storage
            mock_cache_cls.return_value = mock_cache
            mock_monitor_cls.return_value = mock_monitor

            # Mock数据采集
            mock_batch = Mock()
            mock_batch.status = "completed"
            mock_batch.best_result = Mock()
            mock_batch.best_result.data = [{"test": "data"}]
            mock_acquisition.acquire_yearly_data.return_value = [mock_batch]

            # Mock存储
            mock_storage.store_historical_data.return_value = None
            mock_storage.get_data_stats.return_value = {
                "total_records": 1, "completeness_ratio": 1.0, "is_complete": True
            }

            # 创建工作流并执行
            workflow = StrategyBacktestDataWorkflow(self.workflow_config)
            workflow_id = await workflow.start_workflow(workflow_config)

            # 等待执行完成
            await asyncio.sleep(0.2)

            # 验证工作流状态
            status = workflow.get_workflow_status(workflow_id)
            assert status is not None

            # 验证组件调用
            mock_acquisition.acquire_yearly_data.assert_called()
            mock_storage.store_historical_data.assert_called()
            mock_monitor.record_workflow_start.assert_called()
            mock_monitor.record_workflow_completion.assert_called()

    @pytest.mark.asyncio
    async def test_scalability_under_load(self):
        """测试负载下的可扩展性"""
        acquisition_service = HistoricalDataAcquisitionService(self.acquisition_config)

        # 创建多个并发请求
        configs = []
        for i in range(5):
            config = HistoricalDataConfig(
                data_source=DataSourceType.AKSHARE,
                symbol=f"TEST{i:04d}.SZ",
                start_date=datetime(2020, 1, 1),
                end_date=datetime(2020, 1, 31)
            )
            configs.append(config)

        # Mock所有请求
        with patch.object(acquisition_service.adapters[DataSourceType.AKSHARE], 'collect_historical_data',
                         new_callable=AsyncMock) as mock_collect:

            from src.core.orchestration.historical_data_acquisition_service import DataSourceResult, DataQualityLevel

            mock_result = DataSourceResult(
                source=DataSourceType.AKSHARE,
                symbol="test",
                data=[{"symbol": "test", "date": datetime(2020, 1, 1), "close": 100.0}],
                quality_score=0.88,
                quality_level=DataQualityLevel.GOOD,
                collection_time=1.0
            )
            mock_collect.return_value = mock_result

            # 并发执行
            start_time = datetime.now()
            tasks = [acquisition_service.acquire_historical_data(config) for config in configs]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = datetime.now()

            # 验证并发执行
            successful_results = [r for r in results if isinstance(r, object) and hasattr(r, 'status')]
            assert len(successful_results) == 5

            # 验证总执行时间合理（并发应该比串行快）
            total_time = (end_time - start_time).total_seconds()
            assert total_time < 10  # 5个并发任务应该在合理时间内完成

    @pytest.mark.asyncio
    async def test_data_consistency_across_components(self):
        """测试跨组件的数据一致性"""
        acquisition_service = HistoricalDataAcquisitionService(self.acquisition_config)
        quality_manager = DataQualityManager(self.quality_config)

        # 创建一致的测试数据
        symbol = "000001.SZ"
        test_data = [
            {
                "symbol": symbol,
                "date": datetime(2020, 1, 1),
                "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0,
                "volume": 1000000, "source": "akshare"
            }
        ]

        # 在不同组件间传递数据
        # 1. 采集服务处理
        config = HistoricalDataConfig(
            data_source=DataSourceType.AKSHARE,
            symbol=symbol,
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 1, 31)
        )

        with patch.object(acquisition_service.adapters[DataSourceType.AKSHARE], 'collect_historical_data',
                         new_callable=AsyncMock) as mock_collect:

            from src.core.orchestration.historical_data_acquisition_service import DataSourceResult, DataQualityLevel

            mock_result = DataSourceResult(
                source=DataSourceType.AKSHARE,
                symbol=symbol,
                data=test_data,
                quality_score=0.95,
                quality_level=DataQualityLevel.EXCELLENT,
                collection_time=1.0
            )
            mock_collect.return_value = mock_result

            batch = await acquisition_service.acquire_historical_data(config)

            # 2. 质量管理器处理相同数据
            quality_result = await quality_manager.check_data_quality(
                batch.best_result.data, "stock", QualityCheckLevel.BASIC
            )

            # 验证数据一致性
            assert len(batch.best_result.data) == len(test_data)
            assert batch.best_result.data[0]["symbol"] == test_data[0]["symbol"]
            assert batch.best_result.data[0]["date"] == test_data[0]["date"]

            # 验证质量评估一致性
            assert quality_result.total_records == len(test_data)
            assert quality_result.overall_score > 0


if __name__ == '__main__':
    pytest.main([__file__])