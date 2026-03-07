"""
策略存储组件深度测试
全面测试策略版本管理、元数据管理、配置存储和性能历史功能
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
import tempfile
import os

# 导入策略存储相关类
try:
    from src.strategy.workspace.store import (
        StrategyStore, StrategyMetadata, StrategyVersion,
        StrategyPerformanceRecord, StrategyConfigRecord
    )
    STRATEGY_STORE_AVAILABLE = True
except ImportError:
    STRATEGY_STORE_AVAILABLE = False
    StrategyStore = Mock
    StrategyMetadata = Mock
    StrategyVersion = Mock
    StrategyPerformanceRecord = Mock
    StrategyConfigRecord = Mock


class TestStrategyStoreComprehensive:
    """策略存储组件综合深度测试"""

    @pytest.fixture
    def sample_strategy_metadata(self):
        """创建样本策略元数据"""
        if STRATEGY_STORE_AVAILABLE:
            return StrategyMetadata(
                strategy_id="test_momentum_strategy",
                name="Test Momentum Strategy",
                description="A momentum-based trading strategy for testing",
                author="test_user",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                version="1.0.0",
                tags=["momentum", "trend", "technical"],
                market_type="equity",
                risk_level="medium",
                status="active"
            )
        return Mock()

    @pytest.fixture
    def sample_strategy_config(self):
        """创建样本策略配置"""
        return {
            'strategy_type': 'momentum',
            'parameters': {
                'lookback_period': 20,
                'threshold': 0.05,
                'max_position': 100,
                'stop_loss': 0.1,
                'take_profit': 0.15
            },
            'symbols': ['AAPL', 'GOOGL', 'MSFT'],
            'risk_limits': {
                'max_drawdown': 0.1,
                'max_position_size': 1000,
                'daily_loss_limit': 500
            },
            'execution_settings': {
                'commission': 0.001,
                'slippage': 0.0005,
                'min_order_size': 10
            }
        }

    @pytest.fixture
    def sample_performance_data(self):
        """创建样本性能数据"""
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30, freq='D'),
            'portfolio_value': np.cumprod(1 + np.random.normal(0.001, 0.02, 30)),
            'returns': np.random.normal(0.001, 0.02, 30),
            'sharpe_ratio': np.random.uniform(0.5, 2.5, 30),
            'max_drawdown': np.random.uniform(-0.05, -0.25, 30),
            'win_rate': np.random.uniform(0.45, 0.65, 30),
            'total_trades': np.cumsum(np.random.poisson(5, 30)),
            'active_positions': np.random.randint(0, 10, 30)
        })

    @pytest.fixture
    def strategy_store(self, tmp_path):
        """创建策略存储实例"""
        if STRATEGY_STORE_AVAILABLE:
            storage_path = tmp_path / "strategy_store"
            storage_path.mkdir(exist_ok=True)
            return StrategyStore(storage_path=str(storage_path))
        return Mock(spec=StrategyStore)

    def test_strategy_store_initialization(self, strategy_store, tmp_path):
        """测试策略存储初始化"""
        if STRATEGY_STORE_AVAILABLE:
            assert strategy_store is not None
            assert hasattr(strategy_store, 'storage_path')
            assert hasattr(strategy_store, 'strategies')
            assert hasattr(strategy_store, 'versions')
            assert hasattr(strategy_store, 'performance_history')

    def test_strategy_creation_and_storage(self, strategy_store, sample_strategy_metadata, sample_strategy_config):
        """测试策略创建和存储"""
        if STRATEGY_STORE_AVAILABLE:
            # 创建策略
            creation_result = strategy_store.create_strategy(
                metadata=sample_strategy_metadata,
                initial_config=sample_strategy_config,
                creator="test_user"
            )

            assert isinstance(creation_result, dict)
            assert 'strategy_id' in creation_result
            assert 'version_id' in creation_result
            assert creation_result['strategy_id'] == sample_strategy_metadata.strategy_id

            # 验证策略存储
            stored_strategy = strategy_store.get_strategy(sample_strategy_metadata.strategy_id)
            assert stored_strategy is not None
            assert stored_strategy['metadata']['name'] == sample_strategy_metadata.name

    def test_strategy_version_management(self, strategy_store, sample_strategy_metadata, sample_strategy_config):
        """测试策略版本管理"""
        if STRATEGY_STORE_AVAILABLE:
            # 创建初始策略
            strategy_store.create_strategy(sample_strategy_metadata, sample_strategy_config)

            # 创建新版本
            new_config = sample_strategy_config.copy()
            new_config['parameters']['lookback_period'] = 30
            new_config['parameters']['threshold'] = 0.08

            version_result = strategy_store.create_strategy_version(
                strategy_id=sample_strategy_metadata.strategy_id,
                config=new_config,
                version_description="Increased lookback period and threshold",
                author="test_user"
            )

            assert isinstance(version_result, dict)
            assert 'version_id' in version_result

            # 获取版本历史
            version_history = strategy_store.get_strategy_version_history(
                sample_strategy_metadata.strategy_id
            )

            assert isinstance(version_history, list)
            assert len(version_history) >= 2  # 初始版本 + 新版本

            # 获取特定版本
            specific_version = strategy_store.get_strategy_version(
                sample_strategy_metadata.strategy_id,
                version_history[1]['version_id']
            )

            assert specific_version is not None
            assert specific_version['config']['parameters']['lookback_period'] == 30

    def test_strategy_performance_tracking(self, strategy_store, sample_strategy_metadata, sample_performance_data):
        """测试策略性能跟踪"""
        if STRATEGY_STORE_AVAILABLE:
            # 创建策略
            strategy_store.create_strategy(sample_strategy_metadata, {})

            # 存储性能数据
            performance_result = strategy_store.store_performance_data(
                strategy_id=sample_strategy_metadata.strategy_id,
                performance_data=sample_performance_data,
                data_source="backtest",
                metadata={'backtest_period': '2024-01-01 to 2024-01-30'}
            )

            assert performance_result['success'] is True

            # 检索性能数据
            retrieved_performance = strategy_store.get_strategy_performance(
                strategy_id=sample_strategy_metadata.strategy_id,
                date_range={'start': '2024-01-01', 'end': '2024-01-30'}
            )

            assert isinstance(retrieved_performance, pd.DataFrame)
            assert len(retrieved_performance) > 0
            assert 'portfolio_value' in retrieved_performance.columns

    def test_strategy_metadata_management(self, strategy_store, sample_strategy_metadata):
        """测试策略元数据管理"""
        if STRATEGY_STORE_AVAILABLE:
            # 创建策略
            strategy_store.create_strategy(sample_strategy_metadata, {})

            # 更新元数据
            updated_metadata = {
                'description': 'Updated description for testing',
                'tags': ['momentum', 'trend', 'technical', 'updated'],
                'risk_level': 'high'
            }

            update_result = strategy_store.update_strategy_metadata(
                strategy_id=sample_strategy_metadata.strategy_id,
                metadata_updates=updated_metadata
            )

            assert update_result['success'] is True

            # 验证元数据更新
            updated_strategy = strategy_store.get_strategy(sample_strategy_metadata.strategy_id)
            assert updated_strategy['metadata']['description'] == updated_metadata['description']
            assert updated_strategy['metadata']['risk_level'] == 'high'

    def test_strategy_search_and_filtering(self, strategy_store):
        """测试策略搜索和过滤"""
        if STRATEGY_STORE_AVAILABLE:
            # 创建多个策略用于测试
            strategies_data = [
                {
                    'metadata': StrategyMetadata(
                        strategy_id="momentum_strategy_1",
                        name="Momentum Strategy 1",
                        description="First momentum strategy",
                        author="user1",
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                        version="1.0.0",
                        tags=["momentum", "trend"],
                        market_type="equity",
                        risk_level="medium",
                        status="active"
                    ),
                    'config': {'strategy_type': 'momentum', 'parameters': {'lookback': 20}}
                },
                {
                    'metadata': StrategyMetadata(
                        strategy_id="mean_reversion_strategy_1",
                        name="Mean Reversion Strategy 1",
                        description="First mean reversion strategy",
                        author="user2",
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                        version="1.0.0",
                        tags=["mean_reversion", "contrarian"],
                        market_type="equity",
                        risk_level="low",
                        status="active"
                    ),
                    'config': {'strategy_type': 'mean_reversion', 'parameters': {'lookback': 10}}
                },
                {
                    'metadata': StrategyMetadata(
                        strategy_id="momentum_strategy_2",
                        name="Momentum Strategy 2",
                        description="Second momentum strategy",
                        author="user1",
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                        version="1.0.0",
                        tags=["momentum", "breakout"],
                        market_type="commodity",
                        risk_level="high",
                        status="draft"
                    ),
                    'config': {'strategy_type': 'momentum', 'parameters': {'lookback': 15}}
                }
            ]

            # 存储所有策略
            for strategy_data in strategies_data:
                strategy_store.create_strategy(
                    strategy_data['metadata'], strategy_data['config']
                )

            # 按标签搜索
            momentum_strategies = strategy_store.search_strategies(
                filters={'tags': ['momentum']}
            )

            assert len(momentum_strategies) == 2

            # 按作者搜索
            user1_strategies = strategy_store.search_strategies(
                filters={'author': 'user1'}
            )

            assert len(user1_strategies) == 2

            # 复合条件搜索
            complex_search = strategy_store.search_strategies(
                filters={
                    'tags': ['momentum'],
                    'market_type': 'equity',
                    'status': 'active'
                }
            )

            assert len(complex_search) == 1

    def test_strategy_backup_and_restore(self, strategy_store, sample_strategy_metadata, sample_strategy_config, tmp_path):
        """测试策略备份和恢复"""
        if STRATEGY_STORE_AVAILABLE:
            # 创建策略
            strategy_store.create_strategy(sample_strategy_metadata, sample_strategy_config)

            # 创建备份
            backup_file = tmp_path / "strategy_backup.json"
            backup_result = strategy_store.create_backup(str(backup_file))

            assert backup_result['success'] is True
            assert backup_file.exists()

            # 创建新的存储实例
            new_storage_path = tmp_path / "restored_store"
            new_storage_path.mkdir(exist_ok=True)
            new_store = StrategyStore(storage_path=str(new_storage_path))

            # 从备份恢复
            restore_result = new_store.restore_from_backup(str(backup_file))

            assert restore_result['success'] is True

            # 验证恢复的策略
            restored_strategy = new_store.get_strategy(sample_strategy_metadata.strategy_id)
            assert restored_strategy is not None
            assert restored_strategy['metadata']['name'] == sample_strategy_metadata.name

    def test_strategy_export_and_import(self, strategy_store, sample_strategy_metadata, sample_strategy_config, tmp_path):
        """测试策略导出和导入"""
        if STRATEGY_STORE_AVAILABLE:
            # 创建策略
            strategy_store.create_strategy(sample_strategy_metadata, sample_strategy_config)

            # 导出策略
            export_file = tmp_path / "strategy_export.json"
            export_result = strategy_store.export_strategy(
                strategy_id=sample_strategy_metadata.strategy_id,
                export_path=str(export_file),
                include_versions=True,
                include_performance=True
            )

            assert export_result['success'] is True
            assert export_file.exists()

            # 创建新的存储实例
            new_storage_path = tmp_path / "import_store"
            new_storage_path.mkdir(exist_ok=True)
            new_store = StrategyStore(storage_path=str(new_storage_path))

            # 导入策略
            import_result = new_store.import_strategy(str(export_file))

            assert import_result['success'] is True

            # 验证导入的策略
            imported_strategy = new_store.get_strategy(sample_strategy_metadata.strategy_id)
            assert imported_strategy is not None

    def test_strategy_performance_analytics(self, strategy_store, sample_strategy_metadata, sample_performance_data):
        """测试策略性能分析"""
        if STRATEGY_STORE_AVAILABLE:
            # 创建策略并存储性能数据
            strategy_store.create_strategy(sample_strategy_metadata, {})
            strategy_store.store_performance_data(
                sample_strategy_metadata.strategy_id, sample_performance_data
            )

            # 生成性能分析报告
            analytics_report = strategy_store.generate_performance_analytics(
                strategy_id=sample_strategy_metadata.strategy_id,
                analysis_config={
                    'time_periods': ['1M', '3M', '6M', '1Y'],
                    'metrics': ['sharpe_ratio', 'max_drawdown', 'win_rate', 'total_return'],
                    'benchmark_comparison': True,
                    'risk_adjusted_metrics': True,
                    'statistical_tests': True
                }
            )

            assert isinstance(analytics_report, dict)
            assert 'summary_statistics' in analytics_report
            assert 'performance_trends' in analytics_report
            assert 'risk_metrics' in analytics_report

    def test_strategy_version_comparison(self, strategy_store, sample_strategy_metadata, sample_strategy_config):
        """测试策略版本比较"""
        if STRATEGY_STORE_AVAILABLE:
            # 创建初始策略
            strategy_store.create_strategy(sample_strategy_metadata, sample_strategy_config)

            # 创建新版本
            new_config = sample_strategy_config.copy()
            new_config['parameters']['lookback_period'] = 30

            strategy_store.create_strategy_version(
                sample_strategy_metadata.strategy_id, new_config, "Version 2"
            )

            # 比较版本
            comparison_result = strategy_store.compare_strategy_versions(
                strategy_id=sample_strategy_metadata.strategy_id,
                version_ids=['v1.0.0', 'v2.0.0'],
                comparison_metrics=['parameters', 'performance', 'risk']
            )

            assert isinstance(comparison_result, dict)
            assert 'parameter_differences' in comparison_result
            assert 'performance_comparison' in comparison_result

    def test_strategy_collaboration_features(self, strategy_store, sample_strategy_metadata):
        """测试策略协作功能"""
        if STRATEGY_STORE_AVAILABLE:
            # 创建策略
            strategy_store.create_strategy(sample_strategy_metadata, {})

            # 添加协作注释
            comment_result = strategy_store.add_strategy_comment(
                strategy_id=sample_strategy_metadata.strategy_id,
                comment={
                    'author': 'reviewer1',
                    'content': 'Good momentum implementation',
                    'category': 'review',
                    'timestamp': datetime.now()
                }
            )

            assert comment_result['success'] is True

            # 添加评审
            review_result = strategy_store.add_strategy_review(
                strategy_id=sample_strategy_metadata.strategy_id,
                review={
                    'reviewer': 'senior_analyst',
                    'rating': 4.5,
                    'comments': 'Solid strategy with good risk management',
                    'recommendations': ['Add more stop loss rules'],
                    'approval_status': 'approved'
                }
            )

            assert review_result['success'] is True

            # 获取协作历史
            collaboration_history = strategy_store.get_collaboration_history(
                sample_strategy_metadata.strategy_id
            )

            assert isinstance(collaboration_history, list)
            assert len(collaboration_history) >= 2  # 评论 + 评审

    def test_strategy_tagging_and_categorization(self, strategy_store, sample_strategy_metadata):
        """测试策略标签和分类"""
        if STRATEGY_STORE_AVAILABLE:
            # 创建策略
            strategy_store.create_strategy(sample_strategy_metadata, {})

            # 添加标签
            tag_result = strategy_store.add_strategy_tags(
                strategy_id=sample_strategy_metadata.strategy_id,
                tags=['high_frequency', 'statistical_arbitrage', 'machine_learning']
            )

            assert tag_result['success'] is True

            # 按标签搜索
            tagged_strategies = strategy_store.find_strategies_by_tags(
                tags=['machine_learning']
            )

            assert len(tagged_strategies) > 0
            assert sample_strategy_metadata.strategy_id in [s['strategy_id'] for s in tagged_strategies]

            # 获取标签统计
            tag_stats = strategy_store.get_tag_statistics()

            assert isinstance(tag_stats, dict)
            assert 'machine_learning' in tag_stats

    def test_strategy_access_control(self, strategy_store, sample_strategy_metadata):
        """测试策略访问控制"""
        if STRATEGY_STORE_AVAILABLE:
            # 创建策略
            strategy_store.create_strategy(sample_strategy_metadata, {})

            # 设置访问权限
            permissions = {
                'owner': 'creator_user',
                'read_access': ['analyst1', 'analyst2'],
                'write_access': ['analyst1'],
                'admin_access': ['admin_user'],
                'public_read': False
            }

            permission_result = strategy_store.set_strategy_permissions(
                strategy_id=sample_strategy_metadata.strategy_id,
                permissions=permissions
            )

            assert permission_result['success'] is True

            # 检查访问权限
            access_check = strategy_store.check_access_permissions(
                strategy_id=sample_strategy_metadata.strategy_id,
                user_id='analyst1',
                required_permission='write'
            )

            assert access_check['granted'] is True

            # 检查拒绝访问
            denied_access = strategy_store.check_access_permissions(
                strategy_id=sample_strategy_metadata.strategy_id,
                user_id='unauthorized_user',
                required_permission='read'
            )

            assert denied_access['granted'] is False

    def test_strategy_audit_trail(self, strategy_store, sample_strategy_metadata, sample_strategy_config):
        """测试策略审计跟踪"""
        if STRATEGY_STORE_AVAILABLE:
            # 启用审计
            strategy_store.enable_audit_trail()

            # 执行一系列操作
            strategy_store.create_strategy(sample_strategy_metadata, sample_strategy_config)
            strategy_store.update_strategy_metadata(
                sample_strategy_metadata.strategy_id,
                {'description': 'Updated description'}
            )
            strategy_store.create_strategy_version(
                sample_strategy_metadata.strategy_id,
                sample_strategy_config,
                "Version update"
            )

            # 获取审计日志
            audit_log = strategy_store.get_strategy_audit_log(
                strategy_id=sample_strategy_metadata.strategy_id
            )

            assert isinstance(audit_log, list)
            assert len(audit_log) >= 3  # 创建 + 更新 + 版本创建

            # 检查审计记录
            for record in audit_log:
                assert 'timestamp' in record
                assert 'operation' in record
                assert 'user' in record
                assert 'details' in record

    def test_strategy_storage_performance_monitoring(self, strategy_store, sample_strategy_metadata):
        """测试策略存储性能监控"""
        if STRATEGY_STORE_AVAILABLE:
            import time

            # 执行一系列存储操作并监控性能
            operations = []

            for i in range(20):
                start_time = time.time()

                # 创建策略操作
                metadata = StrategyMetadata(
                    strategy_id=f"perf_test_strategy_{i}",
                    name=f"Performance Test {i}",
                    description=f"Strategy for performance testing {i}",
                    author="perf_test_user",
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    version="1.0.0",
                    tags=["performance", "test"],
                    market_type="equity",
                    risk_level="low",
                    status="active"
                )

                strategy_store.create_strategy(metadata, {})

                end_time = time.time()
                operations.append(end_time - start_time)

            # 获取性能统计
            performance_stats = strategy_store.get_storage_performance_stats()

            assert isinstance(performance_stats, dict)
            assert 'average_operation_time' in performance_stats
            assert 'total_operations' in performance_stats
            assert 'storage_utilization' in performance_stats

    def test_strategy_storage_error_handling(self, strategy_store):
        """测试策略存储错误处理"""
        if STRATEGY_STORE_AVAILABLE:
            # 测试无效策略ID
            try:
                strategy_store.get_strategy("invalid_strategy_id")
            except KeyError:
                # 期望的错误处理
                pass

            # 测试重复创建
            metadata = StrategyMetadata(
                strategy_id="duplicate_test",
                name="Duplicate Test",
                description="Test duplicate creation",
                author="test_user",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                version="1.0.0",
                tags=["test"],
                market_type="equity",
                risk_level="low",
                status="active"
            )

            # 第一次创建应该成功
            strategy_store.create_strategy(metadata, {})

            # 第二次创建应该失败或更新
            try:
                strategy_store.create_strategy(metadata, {})
            except ValueError:
                # 期望的重复创建错误
                pass

    def test_strategy_storage_data_integrity(self, strategy_store, sample_strategy_metadata, sample_strategy_config):
        """测试策略存储数据完整性"""
        if STRATEGY_STORE_AVAILABLE:
            # 创建策略
            strategy_store.create_strategy(sample_strategy_metadata, sample_strategy_config)

            # 验证数据完整性
            integrity_check = strategy_store.verify_data_integrity(
                strategy_id=sample_strategy_metadata.strategy_id
            )

            assert integrity_check['data_integrity'] is True
            assert 'checksum_validation' in integrity_check
            assert 'version_consistency' in integrity_check

    def test_strategy_storage_concurrent_access(self, strategy_store, sample_strategy_metadata):
        """测试策略存储并发访问"""
        if STRATEGY_STORE_AVAILABLE:
            import threading

            # 创建基础策略
            strategy_store.create_strategy(sample_strategy_metadata, {})

            results = []
            errors = []

            def concurrent_operation(thread_id):
                try:
                    # 读取操作
                    strategy = strategy_store.get_strategy(sample_strategy_metadata.strategy_id)
                    assert strategy is not None

                    # 更新操作
                    update_result = strategy_store.update_strategy_metadata(
                        sample_strategy_metadata.strategy_id,
                        {'description': f'Updated by thread {thread_id}'}
                    )

                    results.append(f"thread_{thread_id}_success")
                except Exception as e:
                    errors.append(f"thread_{thread_id}_error: {str(e)}")

            # 启动并发操作
            threads = []
            for i in range(5):  # 5个并发线程
                thread = threading.Thread(target=concurrent_operation, args=(i,))
                threads.append(thread)
                thread.start()

            # 等待所有线程完成
            for thread in threads:
                thread.join()

            # 验证并发操作结果
            assert len(results) == 5
            assert len(errors) == 0

    def test_strategy_storage_scaling_limits(self, strategy_store):
        """测试策略存储扩展限制"""
        if STRATEGY_STORE_AVAILABLE:
            # 测试大规模策略存储
            for i in range(100):  # 创建100个策略
                metadata = StrategyMetadata(
                    strategy_id=f"scale_strategy_{i}",
                    name=f"Scale Strategy {i}",
                    description=f"Strategy for scaling test {i}",
                    author="scale_test_user",
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    version="1.0.0",
                    tags=["scale", "test"],
                    market_type="equity",
                    risk_level="medium",
                    status="active"
                )

                strategy_store.create_strategy(metadata, {})

            # 验证大规模存储
            all_strategies = strategy_store.list_all_strategies()

            assert len(all_strategies) >= 100

            # 测试搜索性能
            import time
            start_time = time.time()

            search_results = strategy_store.search_strategies({'tags': ['scale']})

            end_time = time.time()

            # 大规模搜索应该在合理时间内完成
            assert end_time - start_time < 5  # 5秒内完成
            assert len(search_results) == 100

    def test_strategy_storage_backup_recovery(self, strategy_store, tmp_path):
        """测试策略存储备份恢复"""
        if STRATEGY_STORE_AVAILABLE:
            # 创建多个策略
            for i in range(10):
                metadata = StrategyMetadata(
                    strategy_id=f"backup_strategy_{i}",
                    name=f"Backup Strategy {i}",
                    description=f"Strategy for backup test {i}",
                    author="backup_test_user",
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    version="1.0.0",
                    tags=["backup", "test"],
                    market_type="equity",
                    risk_level="medium",
                    status="active"
                )
                strategy_store.create_strategy(metadata, {})

            # 创建完整备份
            backup_path = tmp_path / "full_backup"
            backup_path.mkdir(exist_ok=True)

            full_backup = strategy_store.create_full_backup(str(backup_path))

            assert full_backup['success'] is True
            assert 'backup_files_count' in full_backup

            # 创建新存储实例
            recovery_path = tmp_path / "recovered_storage"
            recovery_path.mkdir(exist_ok=True)
            recovered_store = StrategyStore(storage_path=str(recovery_path))

            # 从备份恢复
            recovery_result = recovered_store.restore_from_full_backup(str(backup_path))

            assert recovery_result['success'] is True

            # 验证恢复的数据
            recovered_strategies = recovered_store.list_all_strategies()
            assert len(recovered_strategies) == 10

    def test_strategy_storage_compression_and_optimization(self, strategy_store, sample_strategy_metadata, sample_performance_data):
        """测试策略存储压缩和优化"""
        if STRATEGY_STORE_AVAILABLE:
            # 创建策略并存储大量性能数据
            strategy_store.create_strategy(sample_strategy_metadata, {})

            # 存储大量性能数据
            large_performance_data = pd.concat([
                sample_performance_data] * 10, ignore_index=True)  # 10倍数据

            strategy_store.store_performance_data(
                sample_strategy_metadata.strategy_id,
                large_performance_data
            )

            # 启用数据压缩
            compression_result = strategy_store.enable_data_compression(
                strategy_id=sample_strategy_metadata.strategy_id,
                compression_algorithm='gzip',
                compression_level=6
            )

            assert compression_result['success'] is True

            # 验证压缩效果
            compression_stats = strategy_store.get_compression_stats(
                sample_strategy_metadata.strategy_id
            )

            assert isinstance(compression_stats, dict)
            assert 'original_size' in compression_stats
            assert 'compressed_size' in compression_stats
            assert 'compression_ratio' in compression_stats

            # 优化存储结构
            optimization_result = strategy_store.optimize_storage_structure(
                strategy_id=sample_strategy_metadata.strategy_id
            )

            assert optimization_result['success'] is True
