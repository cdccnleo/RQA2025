#!/usr/bin/env python3
"""
数据采集管道集成测试
测试P0-P2阶段组件间的集成
"""

import pytest
import sys
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestDataCollectionPipeline:
    """数据采集管道集成测试"""

    def setup_method(self):
        """测试前准备"""
        pass

    def teardown_method(self):
        """测试后清理"""
        pass

    @pytest.mark.asyncio
    async def test_full_collection_pipeline(self):
        """测试完整采集管道"""
        # 这个测试模拟完整的从市场监控到数据持久化的流程

        # 1. 市场状态监控
        from src.core.orchestration.market_adaptive_monitor import get_market_adaptive_monitor

        monitor = get_market_adaptive_monitor()
        with patch.object(monitor, '_fetch_market_data_from_infrastructure', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = {
                'timestamp': datetime.now(),
                'indices': {
                    'sh000001': {'price': 3200.0, 'change_pct': 0.02, 'volume': 200000000, 'volatility': 0.025}
                },
                'market_breadth': 0.55,
                'total_volume': 350000000,
                'sentiment_score': 0.6
            }

            regime_analysis = await monitor.get_current_regime()
            assert regime_analysis.current_regime.name in ['HIGH_VOLATILITY', 'BULL', 'BEAR', 'SIDEWAYS', 'LOW_LIQUIDITY']

        # 2. 数据优先级管理
        from src.core.orchestration.data_priority_manager import get_data_priority_manager

        priority_manager = get_data_priority_manager()
        priority = priority_manager.get_data_priority('000001')
        assert priority.priority_level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']

        # 3. 增量采集策略
        from src.core.orchestration.incremental_collection_strategy import get_incremental_collection_strategy

        strategy = get_incremental_collection_strategy()
        window = strategy.determine_collection_strategy('000001', 'stock')
        assert window.mode in ['incremental', 'complement', 'full']
        assert window.priority in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']

        # 4. 智能调度器
        from src.core.orchestration.business_process.service_scheduler import get_data_collection_scheduler

        scheduler = get_data_collection_scheduler()
        status = await scheduler.get_intelligent_scheduling_status()
        assert 'market_regime' in status
        assert 'scheduler_adjustments' in status

    @pytest.mark.asyncio
    async def test_market_driven_scheduling(self):
        """测试市场驱动的调度调整"""
        # 模拟高波动市场状态
        from src.core.orchestration.market_adaptive_monitor import get_market_adaptive_monitor
        from src.core.orchestration.business_process.service_scheduler import get_data_collection_scheduler

        monitor = get_market_adaptive_monitor()
        scheduler = get_data_collection_scheduler()

        # Mock高波动市场数据
        high_volatility_data = {
            'timestamp': datetime.now(),
            'indices': {
                'sh000001': {'price': 3200.0, 'change_pct': 0.05, 'volume': 150000000, 'volatility': 0.08}
            },
            'market_breadth': 0.45,
            'total_volume': 280000000,
            'sentiment_score': 0.4
        }

        with patch.object(monitor, '_fetch_market_data_from_infrastructure', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = high_volatility_data

            # 获取市场状态
            regime_analysis = await monitor.get_current_regime()
            assert regime_analysis.current_regime.name == 'HIGH_VOLATILITY'

            # 检查调度器是否相应调整
            status = await scheduler.get_intelligent_scheduling_status()

            # 在高波动市场下，调度器应该减少采集频率
            market_regime = status.get('market_regime', {})
            assert market_regime.get('current') == 'HIGH_VOLATILITY'

    def test_data_persistence_pipeline(self):
        """测试数据持久化管道"""
        # 模拟从采集到持久化的完整流程

        # 1. 生成测试数据
        test_stock_data = [
            {
                'symbol': '000001.SZ',
                'date': '2023-01-01',
                'open': 10.5,
                'high': 11.0,
                'low': 10.2,
                'close': 10.8,
                'volume': 1000000,
                'amount': 10800000.0,
                'source': 'akshare'
            },
            {
                'symbol': '000002.SZ',
                'date': '2023-01-01',
                'open': 20.5,
                'high': 21.0,
                'low': 20.2,
                'close': 20.8,
                'volume': 2000000,
                'amount': 41600000.0,
                'source': 'akshare'
            }
        ]

        # 2. 数据去重处理
        from src.gateway.web.postgresql_deduplication import DataDeduplicationManager

        deduplication_manager = DataDeduplicationManager()
        deduplicated_data = deduplication_manager.deduplicate_data(test_stock_data, 'stock')
        assert len(deduplicated_data) <= len(test_stock_data)

        # 3. 数据质量验证
        from src.gateway.web.postgresql_deduplication import DataQualityValidator

        quality_validator = DataQualityValidator()
        quality_results = quality_validator.validate_batch(deduplicated_data, 'stock')

        # 验证质量检查结果
        assert 'valid_records' in quality_results
        assert 'invalid_records' in quality_results
        assert 'quality_score' in quality_results

        # 4. 数据合并优化
        from src.gateway.web.data_merge_optimizer import DataMergeOptimizer

        merge_optimizer = DataMergeOptimizer()
        optimized_data = merge_optimizer.optimize_incremental_merge(
            deduplicated_data, [], 'stock', 'akshare'
        )

        assert 'merged_data' in optimized_data
        assert 'quality_report' in optimized_data
        assert 'performance_stats' in optimized_data

    def test_complement_scheduler_integration(self):
        """测试补全调度器集成"""
        from src.core.orchestration.data_complement_scheduler import DataComplementScheduler
        from src.core.orchestration.complement_priority_manager import get_complement_priority_manager

        # 创建补全任务
        scheduler = DataComplementScheduler()
        priority_manager = get_complement_priority_manager()

        # 生成补全任务
        tasks = scheduler.generate_complement_tasks('000001', 'stock', 'MONTHLY')

        if tasks:
            # 添加到优先级队列
            for task in tasks:
                success = priority_manager.enqueue_task(task)
                assert success

            # 获取下一个任务
            next_task = priority_manager.dequeue_task()
            if next_task:
                assert next_task.task_id is not None
                assert next_task.source_id == '000001'

                # 完成任务
                priority_manager.complete_task(next_task.task_id, success=True)

        # 验证队列统计
        stats = priority_manager.get_queue_statistics()
        assert 'queue_size' in stats
        assert 'active_tasks' in stats
        assert 'completed_tasks' in stats

    @pytest.mark.asyncio
    async def test_batch_complement_processor(self):
        """测试批处理补全处理器"""
        from src.core.orchestration.batch_complement_processor import BatchComplementProcessor
        from src.core.orchestration.data_complement_scheduler import ComplementTask, ComplementMode, ComplementPriority

        processor = BatchComplementProcessor()

        # 创建测试任务
        task = ComplementTask(
            task_id='test_task_001',
            source_id='000001',
            data_type='stock',
            mode=ComplementMode.MONTHLY,
            priority=ComplementPriority.HIGH,
            created_at=datetime.now()
        )

        # 执行批处理
        result = await processor.process_complement_task(task)

        assert 'task_id' in result
        assert 'success' in result
        assert 'processed_records' in result
        assert 'processing_time' in result

        # 验证结果结构
        assert result['task_id'] == 'test_task_001'
        assert isinstance(result['success'], bool)
        assert isinstance(result['processed_records'], int)
        assert isinstance(result['processing_time'], (int, float))

    def test_incremental_persistence_tracking(self):
        """测试增量持久化跟踪"""
        from src.core.orchestration.incremental_collection_persistence import IncrementalCollectionPersistence

        persistence = IncrementalCollectionPersistence()

        # 记录采集统计
        stats = {
            'source_id': '000001',
            'data_type': 'stock',
            'collection_mode': 'incremental',
            'records_collected': 1000,
            'start_time': datetime.now() - timedelta(minutes=5),
            'end_time': datetime.now(),
            'quality_score': 0.95,
            'errors_count': 2
        }

        success = persistence.save_collection_statistics(stats)
        assert success

        # 查询统计历史
        history = persistence.get_collection_history('000001', 'stock', limit=5)
        assert isinstance(history, list)

        if history:
            latest = history[0]
            assert 'source_id' in latest
            assert 'records_collected' in latest
            assert 'quality_score' in latest

    def test_error_handling_and_recovery(self):
        """测试错误处理和恢复"""
        # 测试各个组件的错误处理能力

        # 1. 市场监控错误处理
        from src.core.orchestration.market_adaptive_monitor import get_market_adaptive_monitor

        monitor = get_market_adaptive_monitor()
        with patch.object(monitor, '_fetch_market_data_from_infrastructure', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.side_effect = Exception("Network timeout")

            # 应该返回默认状态
            regime_analysis = await monitor.get_current_regime()
            assert regime_analysis.current_regime.name == 'SIDEWAYS'

        # 2. 数据持久化错误处理
        from src.gateway.web.postgresql_persistence_batch import PostgreSQLBatchInserter

        config = {
            'host': 'invalid_host',
            'port': 5432,
            'database': 'test_db',
            'user': 'test_user',
            'password': 'test_pass',
            'pool_size': 1,
            'max_overflow': 0,
            'batch_size': 100,
            'max_retries': 2,
            'retry_delay': 0.1
        }

        with patch('psycopg2.pool.SimpleConnectionPool') as mock_pool:
            mock_pool.side_effect = Exception("Connection failed")

            inserter = PostgreSQLBatchInserter(config)

            test_data = [{'symbol': '000001.SZ', 'date': '2023-01-01', 'open': 10.5, 'close': 10.8}]

            # 应该优雅处理连接失败
            result = inserter.batch_insert_stock_data(test_data, 'akshare')
            assert result['success'] is False
            assert 'error' in result or result['failed_count'] > 0

    def test_performance_monitoring_integration(self):
        """测试性能监控集成"""
        # 运行性能监控脚本
        import subprocess
        import sys

        try:
            result = subprocess.run([
                sys.executable, 'scripts/monitor_system_performance.py'
            ], capture_output=True, text=True, timeout=30)

            # 检查执行结果
            assert result.returncode == 0 or result.returncode == 1  # 0表示通过，1表示需要优化

            # 检查输出内容
            output = result.stdout + result.stderr
            assert '系统性能监控检查完成' in output
            assert '综合评分' in output

        except subprocess.TimeoutExpired:
            pytest.fail("性能监控脚本执行超时")
        except Exception as e:
            pytest.fail(f"性能监控脚本执行失败: {e}")

    def test_end_to_end_data_flow(self):
        """测试端到端数据流"""
        # 模拟从数据源到最终持久化的完整流程

        # 1. 数据源配置
        source_config = {
            'source_id': 'akshare_stock',
            'data_type': 'stock',
            'collection_params': {
                'symbols': ['000001.SZ', '000002.SZ'],
                'start_date': '2023-01-01',
                'end_date': '2023-01-05'
            }
        }

        # 2. 采集策略确定
        from src.core.orchestration.incremental_collection_strategy import get_incremental_collection_strategy

        strategy = get_incremental_collection_strategy()
        window = strategy.determine_collection_strategy('000001', 'stock')

        assert window.source_id == '000001'
        assert window.data_type == 'stock'

        # 3. 优先级评估
        from src.core.orchestration.data_priority_manager import get_data_priority_manager

        priority_manager = get_data_priority_manager()
        priority = priority_manager.get_data_priority('000001')

        assert priority.priority_level == 'CRITICAL'

        # 4. 质量验证
        from src.gateway.web.postgresql_deduplication import DataQualityValidator

        validator = DataQualityValidator()
        test_records = [
            {'symbol': '000001.SZ', 'date': '2023-01-01', 'open': 10.5, 'close': 10.8},
            {'symbol': '000002.SZ', 'date': '2023-01-01', 'open': 20.5, 'close': 20.8}
        ]

        quality_result = validator.validate_batch(test_records, 'stock')
        assert 'quality_score' in quality_result
        assert quality_result['quality_score'] >= 0.8  # 假设数据质量良好


if __name__ == '__main__':
    pytest.main([__file__])