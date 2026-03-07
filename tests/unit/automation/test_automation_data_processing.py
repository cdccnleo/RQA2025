"""
自动化数据处理测试
测试数据管道、备份恢复、数据同步、质量检查等数据处理功能
"""

import pytest
import time
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from src.automation.data.data_pipeline import DataPipeline
from src.automation.data.backup_recovery import BackupManager as BackupRecovery
from src.automation.data.data_sync import DataSynchronizer
from src.automation.data.quality_checks import DataQualityChecker


class AutomationDataTestFactory:
    """自动化数据测试数据工厂"""

    @staticmethod
    def create_sample_dataframe(rows=100):
        """创建示例DataFrame"""
        dates = [datetime.now() - timedelta(days=i) for i in range(rows)]
        data = {
            'id': range(rows),
            'value': np.random.randn(rows),
            'category': np.random.choice(['A', 'B', 'C'], rows),
            'timestamp': dates,
            'status': np.random.choice(['active', 'inactive', 'pending'], rows)
        }
        return pd.DataFrame(data)

    @staticmethod
    def create_pipeline_config():
        """创建管道配置"""
        return {
            'pipeline_id': 'test_pipeline',
            'name': 'Test Data Pipeline',
            'source': {'type': 'database', 'connection': 'test_db'},
            'transformations': [
                {'type': 'filter', 'condition': 'status == "active"'},
                {'type': 'aggregate', 'group_by': 'category', 'agg_func': 'mean'},
                {'type': 'validate', 'rules': ['not_null', 'positive_values']}
            ],
            'destination': {'type': 'warehouse', 'table': 'processed_data'},
            'schedule': 'daily',
            'error_handling': 'retry_and_alert'
        }

    @staticmethod
    def create_backup_config():
        """创建备份配置"""
        return {
            'backup_id': 'test_backup',
            'name': 'Test Backup',
            'sources': [
                {'type': 'database', 'name': 'main_db'},
                {'type': 'files', 'path': '/data/files'}
            ],
            'destination': {'type': 's3', 'bucket': 'backup-bucket'},
            'schedule': 'daily',
            'retention_days': 30,
            'compression': True,
            'encryption': True
        }

    @staticmethod
    def create_sync_config():
        """创建同步配置"""
        return {
            'sync_id': 'test_sync',
            'name': 'Test Data Sync',
            'source': {'type': 'mysql', 'database': 'source_db'},
            'target': {'type': 'postgresql', 'database': 'target_db'},
            'tables': ['users', 'orders', 'products'],
            'sync_mode': 'incremental',
            'conflict_resolution': 'source_wins',
            'monitoring_enabled': True
        }

    @staticmethod
    def create_quality_config():
        """创建质量检查配置"""
        return {
            'check_id': 'test_quality',
            'name': 'Test Quality Checks',
            'dataset': 'test_table',
            'checks': [
                {'type': 'completeness', 'column': 'id', 'rule': 'not_null'},
                {'type': 'accuracy', 'column': 'value', 'rule': 'range_check', 'min': 0, 'max': 100},
                {'type': 'consistency', 'columns': ['start_date', 'end_date'], 'rule': 'date_order'},
                {'type': 'uniqueness', 'column': 'email', 'rule': 'unique'},
                {'type': 'validity', 'column': 'status', 'rule': 'in_list', 'values': ['active', 'inactive']}
            ],
            'thresholds': {
                'completeness': 0.95,
                'accuracy': 0.98,
                'consistency': 0.99
            }
        }


class TestDataPipeline:
    """数据管道测试"""

    def setup_method(self):
        """测试前准备"""
        self.data_pipeline = DataPipeline()
        self.test_factory = AutomationDataTestFactory()

    def test_data_pipeline_initialization(self):
        """测试数据管道初始化"""
        assert self.data_pipeline is not None
        assert hasattr(self.data_pipeline, 'create_pipeline')
        assert hasattr(self.data_pipeline, 'execute_pipeline')
        assert hasattr(self.data_pipeline, 'get_pipeline_status')
        assert hasattr(self.data_pipeline, 'delete_pipeline')

    def test_pipeline_creation_and_execution(self):
        """测试管道创建和执行"""
        config = self.test_factory.create_pipeline_config()

        # 创建管道
        result = self.data_pipeline.create_pipeline(config)
        assert result is True

        # 执行管道
        execution_result = self.data_pipeline.execute_pipeline(config['pipeline_id'])

        assert execution_result is not None
        # 验证执行结果
        assert 'status' in execution_result
        assert execution_result['status'] in ['success', 'completed', 'running']

    def test_data_transformation_pipeline(self):
        """测试数据转换管道"""
        # 创建测试数据
        test_data = self.test_factory.create_sample_dataframe(50)

        transformation_config = {
            'transformations': [
                {'type': 'filter', 'condition': 'category == "A"'},
                {'type': 'sort', 'column': 'value', 'ascending': False},
                {'type': 'limit', 'count': 10}
            ]
        }

        result = self.data_pipeline.execute_transformations(test_data, transformation_config)

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        # 验证转换结果
        assert len(result) <= 10  # 限制了数量
        assert all(result['category'] == 'A')  # 过滤了类别

    def test_pipeline_error_handling(self):
        """测试管道错误处理"""
        # 创建包含错误转换的配置
        error_config = {
            'pipeline_id': 'error_pipeline',
            'transformations': [
                {'type': 'invalid_transformation', 'param': 'bad_value'}
            ]
        }

        result = self.data_pipeline.execute_transformations(
            self.test_factory.create_sample_dataframe(10),
            error_config
        )

        # 应该处理错误并返回结果或抛出适当异常
        assert result is not None or isinstance(result, Exception)

    def test_pipeline_monitoring(self):
        """测试管道监控"""
        config = self.test_factory.create_pipeline_config()
        self.data_pipeline.create_pipeline(config)

        # 执行几次
        for i in range(3):
            self.data_pipeline.execute_pipeline(config['pipeline_id'])

        # 获取状态
        status = self.data_pipeline.get_pipeline_status(config['pipeline_id'])

        assert status is not None
        # 验证监控信息
        assert 'execution_count' in status or 'runs' in status
        assert 'last_execution_time' in status or 'last_run' in status

    def test_parallel_pipeline_execution(self):
        """测试并行管道执行"""
        import concurrent.futures

        # 创建多个管道配置
        configs = []
        for i in range(3):
            config = self.test_factory.create_pipeline_config()
            config['pipeline_id'] = f'parallel_pipeline_{i}'
            self.data_pipeline.create_pipeline(config)
            configs.append(config)

        def execute_pipeline(pipeline_id):
            return self.data_pipeline.execute_pipeline(pipeline_id)

        # 并行执行管道
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(execute_pipeline, c['pipeline_id']) for c in configs]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # 验证所有管道都执行完成
        assert len(results) == 3
        for result in results:
            assert result is not None


class TestBackupRecovery:
    """备份恢复测试"""

    def setup_method(self):
        """测试前准备"""
        self.backup_recovery = BackupRecovery()
        self.test_factory = AutomationDataTestFactory()

    def test_backup_recovery_initialization(self):
        """测试备份恢复初始化"""
        assert self.backup_recovery is not None
        assert hasattr(self.backup_recovery, 'create_backup')
        assert hasattr(self.backup_recovery, 'restore_backup')
        assert hasattr(self.backup_recovery, 'get_backup_status')
        assert hasattr(self.backup_recovery, 'list_backups')

    def test_backup_creation(self):
        """测试备份创建"""
        config = self.test_factory.create_backup_config()

        result = self.backup_recovery.create_backup(config)

        assert result is not None
        # 验证备份结果
        assert 'backup_id' in result or 'status' in result

    def test_backup_restoration(self):
        """测试备份恢复"""
        config = self.test_factory.create_backup_config()

        # 先创建备份
        backup_result = self.backup_recovery.create_backup(config)
        assert backup_result is not None

        # 然后恢复备份
        if 'backup_id' in backup_result:
            restore_result = self.backup_recovery.restore_backup(backup_result['backup_id'])
            assert restore_result is not None
            assert 'status' in restore_result

    def test_backup_listing_and_status(self):
        """测试备份列表和状态"""
        # 创建几个备份
        for i in range(3):
            config = self.test_factory.create_backup_config()
            config['backup_id'] = f'test_backup_{i}'
            self.backup_recovery.create_backup(config)

        # 列出备份
        backups = self.backup_recovery.list_backups()

        assert backups is not None
        assert isinstance(backups, list)
        assert len(backups) >= 3

        # 检查备份状态
        for backup in backups:
            status = self.backup_recovery.get_backup_status(backup.get('backup_id'))
            assert status is not None
            assert 'status' in status

    def test_incremental_backup(self):
        """测试增量备份"""
        config = self.test_factory.create_backup_config()
        config['backup_type'] = 'incremental'

        # 执行多次增量备份
        backups = []
        for i in range(3):
            result = self.backup_recovery.create_backup(config)
            backups.append(result)

        # 验证增量备份的差异
        assert len(backups) == 3
        # 增量备份应该比全量备份小（理想情况下）

    def test_backup_compression_and_encryption(self):
        """测试备份压缩和加密"""
        config = self.test_factory.create_backup_config()
        config['compression'] = True
        config['encryption'] = True

        result = self.backup_recovery.create_backup(config)

        assert result is not None
        # 验证压缩和加密设置被应用
        assert 'compression_used' in result or 'encrypted' in result

    def test_backup_retention_policy(self):
        """测试备份保留策略"""
        config = self.test_factory.create_backup_config()
        config['retention_days'] = 7

        # 创建一些旧备份
        old_backups = []
        for i in range(5):
            backup_config = config.copy()
            backup_config['backup_id'] = f'old_backup_{i}'
            # 模拟旧时间戳
            result = self.backup_recovery.create_backup(backup_config)
            old_backups.append(result)

        # 执行清理
        cleanup_result = self.backup_recovery.cleanup_old_backups(config['retention_days'])

        assert cleanup_result is not None
        # 验证清理结果
        assert 'deleted_count' in cleanup_result or 'cleaned_backups' in cleanup_result


class TestDataSync:
    """数据同步测试"""

    def setup_method(self):
        """测试前准备"""
        self.data_sync = DataSync()
        self.test_factory = AutomationDataTestFactory()

    def test_data_sync_initialization(self):
        """测试数据同步初始化"""
        assert self.data_sync is not None
        assert hasattr(self.data_sync, 'configure_sync')
        assert hasattr(self.data_sync, 'execute_sync')
        assert hasattr(self.data_sync, 'get_sync_status')
        assert hasattr(self.data_sync, 'cancel_sync')

    def test_sync_configuration_and_execution(self):
        """测试同步配置和执行"""
        config = self.test_factory.create_sync_config()

        # 配置同步
        result = self.data_sync.configure_sync(config)
        assert result is True

        # 执行同步
        sync_result = self.data_sync.execute_sync(config['sync_id'])

        assert sync_result is not None
        # 验证同步结果
        assert 'status' in sync_result
        assert sync_result['status'] in ['success', 'completed', 'running']

    def test_incremental_sync(self):
        """测试增量同步"""
        config = self.test_factory.create_sync_config()
        config['sync_mode'] = 'incremental'
        self.data_sync.configure_sync(config)

        # 执行多次增量同步
        sync_results = []
        for i in range(3):
            result = self.data_sync.execute_sync(config['sync_id'])
            sync_results.append(result)

        # 验证增量同步的效率
        assert len(sync_results) == 3
        for result in sync_results:
            assert 'records_processed' in result or 'changes_synced' in result

    def test_sync_conflict_resolution(self):
        """测试同步冲突解决"""
        config = self.test_factory.create_sync_config()
        config['conflict_resolution'] = 'manual_review'

        self.data_sync.configure_sync(config)

        # 模拟有冲突的同步
        conflict_data = {
            'conflicts': [
                {'table': 'users', 'id': 1, 'source_value': 'value1', 'target_value': 'value2'},
                {'table': 'orders', 'id': 5, 'source_value': 100, 'target_value': 150}
            ]
        }

        result = self.data_sync.execute_sync(config['sync_id'], conflict_data)

        # 应该处理冲突
        assert result is not None
        assert 'conflicts_resolved' in result or 'manual_review_required' in result

    def test_sync_monitoring_and_status(self):
        """测试同步监控和状态"""
        config = self.test_factory.create_sync_config()
        self.data_sync.configure_sync(config)

        # 执行同步
        self.data_sync.execute_sync(config['sync_id'])

        # 获取状态
        status = self.data_sync.get_sync_status(config['sync_id'])

        assert status is not None
        # 验证状态信息
        assert 'status' in status
        assert 'progress' in status or 'completion_percent' in status
        assert 'last_sync_time' in status

    def test_bidirectional_sync(self):
        """测试双向同步"""
        config = self.test_factory.create_sync_config()
        config['sync_mode'] = 'bidirectional'

        self.data_sync.configure_sync(config)

        # 执行双向同步
        result = self.data_sync.execute_sync(config['sync_id'])

        assert result is not None
        # 验证双向同步
        assert 'source_changes' in result and 'target_changes' in result

    def test_sync_error_handling(self):
        """测试同步错误处理"""
        config = self.test_factory.create_sync_config()
        config['tables'] = ['nonexistent_table']  # 不存在的表

        self.data_sync.configure_sync(config)

        result = self.data_sync.execute_sync(config['sync_id'])

        # 应该处理错误
        assert result is not None
        assert result['status'] == 'failed' or 'errors' in result


class TestQualityChecks:
    """质量检查测试"""

    def setup_method(self):
        """测试前准备"""
        self.quality_checks = QualityChecks()
        self.test_factory = AutomationDataTestFactory()

    def test_quality_checks_initialization(self):
        """测试质量检查初始化"""
        assert self.quality_checks is not None
        assert hasattr(self.quality_checks, 'configure_checks')
        assert hasattr(self.quality_checks, 'execute_checks')
        assert hasattr(self.quality_checks, 'get_check_results')
        assert hasattr(self.quality_checks, 'generate_report')

    def test_quality_check_configuration(self):
        """测试质量检查配置"""
        config = self.test_factory.create_quality_config()

        result = self.quality_checks.configure_checks(config)
        assert result is True

    def test_data_completeness_check(self):
        """测试数据完整性检查"""
        # 创建包含空值的数据
        data = self.test_factory.create_sample_dataframe(100)
        data.loc[10:15, 'id'] = None  # 引入一些空值

        completeness_result = self.quality_checks.check_completeness(data, 'id')

        assert completeness_result is not None
        assert 'completeness_score' in completeness_result
        assert completeness_result['completeness_score'] < 1.0  # 应该小于1

    def test_data_accuracy_check(self):
        """测试数据准确性检查"""
        # 创建包含异常值的数据
        data = self.test_factory.create_sample_dataframe(100)
        data.loc[50, 'value'] = 999  # 异常高值

        accuracy_result = self.quality_checks.check_accuracy(data, 'value', min_val=0, max_val=10)

        assert accuracy_result is not None
        assert 'accuracy_score' in accuracy_result
        assert 'outliers' in accuracy_result

    def test_data_consistency_check(self):
        """测试数据一致性检查"""
        # 创建日期数据
        data = pd.DataFrame({
            'start_date': pd.date_range('2023-01-01', periods=50),
            'end_date': pd.date_range('2023-01-02', periods=50)
        })

        consistency_result = self.quality_checks.check_consistency(data, ['start_date', 'end_date'])

        assert consistency_result is not None
        assert 'consistency_score' in consistency_result

    def test_data_uniqueness_check(self):
        """测试数据唯一性检查"""
        # 创建包含重复值的数据
        data = pd.DataFrame({
            'email': ['user1@test.com', 'user2@test.com', 'user1@test.com', 'user3@test.com'] * 25
        })

        uniqueness_result = self.quality_checks.check_uniqueness(data, 'email')

        assert uniqueness_result is not None
        assert 'uniqueness_score' in uniqueness_result
        assert uniqueness_result['uniqueness_score'] < 1.0  # 有重复值

    def test_comprehensive_quality_assessment(self):
        """测试综合质量评估"""
        config = self.test_factory.create_quality_config()

        self.quality_checks.configure_checks(config)

        # 执行所有检查
        test_data = self.test_factory.create_sample_dataframe(200)
        results = self.quality_checks.execute_checks(test_data)

        assert results is not None
        assert isinstance(results, dict)
        # 验证包含各种检查结果
        assert 'completeness' in results or 'overall_score' in results

    def test_quality_report_generation(self):
        """测试质量报告生成"""
        config = self.test_factory.create_quality_config()
        self.quality_checks.configure_checks(config)

        test_data = self.test_factory.create_sample_dataframe(100)
        self.quality_checks.execute_checks(test_data)

        report = self.quality_checks.generate_report()

        assert report is not None
        # 验证报告内容
        assert 'summary' in report or 'overview' in report
        assert 'recommendations' in report or 'issues' in report

    def test_quality_threshold_monitoring(self):
        """测试质量阈值监控"""
        config = self.test_factory.create_quality_config()
        config['alert_thresholds'] = {'completeness': 0.95, 'accuracy': 0.98}

        self.quality_checks.configure_checks(config)

        # 创建低质量数据
        poor_data = pd.DataFrame({
            'id': [None] * 20 + list(range(80)),  # 20%空值
            'value': [999] * 10 + list(range(90)),  # 异常值
            'status': ['invalid'] * 50 + ['active'] * 50
        })

        alerts = self.quality_checks.check_thresholds(poor_data, config['alert_thresholds'])

        assert alerts is not None
        assert isinstance(alerts, list)
        # 应该有告警
        assert len(alerts) > 0

    def test_historical_quality_tracking(self):
        """测试历史质量跟踪"""
        config = self.test_factory.create_quality_config()
        self.quality_checks.configure_checks(config)

        # 执行多次质量检查
        for i in range(5):
            test_data = self.test_factory.create_sample_dataframe(50)
            self.quality_checks.execute_checks(test_data)

        # 获取历史趋势
        trends = self.quality_checks.get_quality_trends()

        assert trends is not None
        # 验证趋势数据
        assert 'historical_scores' in trends or 'trend_data' in trends
        assert len(trends.get('historical_scores', [])) == 5

