"""
数据质量深度测试
测试数据质量检查、验证和处理的各种场景
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.data.quality.unified_quality_monitor import UnifiedDataValidator, UnifiedQualityMonitor
from src.data.quality.data_quality_monitor import DataQualityMonitor
from src.data.quality.advanced_quality_monitor import AdvancedQualityMonitor
from src.data.validation.validator import DataValidator
from src.data.loader.stock_loader import StockDataLoader


class TestDataQualityComprehensive:
    """数据质量综合测试"""

    @pytest.fixture
    def quality_checker(self):
        """创建数据质量检查器实例"""
        return UnifiedDataValidator()

    @pytest.fixture
    def data_validator(self):
        """创建数据验证器实例"""
        return DataValidator()

    @pytest.fixture
    def stock_loader(self):
        """创建股票数据加载器实例"""
        return StockDataLoader()

    @pytest.fixture
    def advanced_quality_monitor(self):
        """创建高级质量监控器实例"""
        return AdvancedQualityMonitor()

    def test_data_quality_checker_initialization(self, quality_checker):
        """测试数据质量检查器初始化"""
        assert quality_checker is not None
        assert hasattr(quality_checker, 'config')
        assert hasattr(quality_checker, 'validation_rules')

    def test_comprehensive_data_quality_check(self, quality_checker):
        """测试综合数据质量检查"""
        # 创建测试数据
        data = pd.DataFrame({
            'symbol': ['000001', '000002', '000003', '000001', '000002'],
            'price': [10.5, 20.3, None, 10.8, 20.1],  # 包含缺失值
            'volume': [1000, 2000, 1500, 1200, -500],  # 包含负值
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='D')
        })

        quality_report = quality_checker.perform_comprehensive_check(data)

        assert isinstance(quality_report, dict)
        assert 'overall_quality_score' in quality_report
        assert 'quality_issues' in quality_report
        assert 'recommendations' in quality_report

        # 应该检测到缺失值和负值问题
        assert len(quality_report['quality_issues']) > 0

    def test_missing_data_handling(self, quality_checker):
        """测试缺失数据处理"""
        # 创建包含多种缺失模式的测试数据
        data = pd.DataFrame({
            'A': [1, 2, None, 4, 5],
            'B': [None, None, None, None, None],  # 完全缺失
            'C': [1, 2, 3, 4, 5],  # 无缺失
            'D': [1, None, None, None, 5]  # 部分缺失
        })

        missing_analysis = quality_checker.analyze_missing_data(data)

        assert isinstance(missing_analysis, dict)
        assert 'missing_summary' in missing_analysis
        assert 'missing_patterns' in missing_analysis
        assert 'imputation_suggestions' in missing_analysis

        # 应该识别出B列完全缺失
        assert missing_analysis['missing_summary']['B'] == 1.0  # 100%缺失

    def test_outlier_detection_algorithms(self, quality_checker):
        """测试异常值检测算法"""
        # 创建包含异常值的数据
        np.random.seed(42)
        normal_data = np.random.normal(100, 10, 100)
        outliers = np.array([500, -200, 1000])  # 明显的异常值
        data = np.concatenate([normal_data, outliers])

        outlier_results = quality_checker.detect_outliers_multimethod(data)

        assert isinstance(outlier_results, dict)
        assert 'outlier_indices' in outlier_results
        assert 'outlier_scores' in outlier_results
        assert 'detection_methods' in outlier_results

        # 应该检测到异常值
        assert len(outlier_results['outlier_indices']) >= 3

    def test_data_consistency_validation(self, quality_checker):
        """测试数据一致性验证"""
        # 创建包含不一致关系的数据
        data = pd.DataFrame({
            'open': [10.0, 11.0, 12.0, 9.0],
            'high': [10.5, 11.5, 12.5, 9.5],
            'low': [9.5, 10.5, 11.5, 8.5],
            'close': [10.2, 11.2, 12.2, 9.2],
            'volume': [1000, 1100, 1200, 900]
        })

        # 添加不一致的数据（high < low）
        inconsistent_data = data.copy()
        inconsistent_data.loc[0, 'high'] = 9.0  # high < low，不一致

        consistency_check = quality_checker.validate_data_consistency(inconsistent_data)

        assert isinstance(consistency_check, dict)
        assert 'consistency_score' in consistency_check
        assert 'inconsistencies_found' in consistency_check
        assert consistency_check['inconsistencies_found'] > 0

    def test_temporal_data_quality(self, quality_checker):
        """测试时序数据质量"""
        # 创建时序数据
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'timestamp': dates,
            'price': [100, 101, 102, 103, 102, 101, 100, 99, 98, 200]  # 最后价格异常
        })

        # 模拟缺失的时间点
        incomplete_data = data.drop([2, 5])  # 删除一些时间点

        temporal_quality = quality_checker.assess_temporal_quality(incomplete_data)

        assert isinstance(temporal_quality, dict)
        assert 'temporal_completeness' in temporal_quality
        assert 'missing_periods' in temporal_quality
        assert 'anomaly_detection' in temporal_quality

    def test_cross_field_validation(self, quality_checker):
        """测试跨字段验证"""
        # 创建包含字段间关系的财务数据
        financial_data = pd.DataFrame({
            'total_assets': [1000, 1100, 1200, 1300],
            'current_assets': [400, 450, 500, 550],
            'total_liabilities': [600, 650, 700, 750],
            'current_liabilities': [200, 220, 240, 260],
            'equity': [400, 450, 500, 550]
        })

        # 添加不一致的数据（资产不等于负债+权益）
        inconsistent_data = financial_data.copy()
        inconsistent_data.loc[0, 'equity'] = 300  # 打破平衡关系

        cross_validation = quality_checker.validate_cross_fields(inconsistent_data)

        assert isinstance(cross_validation, dict)
        assert 'validation_rules' in cross_validation
        assert 'violations' in cross_validation
        assert len(cross_validation['violations']) > 0

    def test_data_freshness_validation(self, quality_checker):
        """测试数据新鲜度验证"""
        # 创建包含时间戳的数据
        current_time = datetime.now()

        fresh_data = pd.DataFrame({
            'symbol': ['000001', '000002'],
            'price': [100.0, 200.0],
            'timestamp': [current_time, current_time - timedelta(minutes=5)]
        })

        stale_data = pd.DataFrame({
            'symbol': ['000003', '000004'],
            'price': [300.0, 400.0],
            'timestamp': [current_time - timedelta(days=10), current_time - timedelta(hours=25)]
        })

        freshness_check = quality_checker.validate_data_freshness(
            pd.concat([fresh_data, stale_data]), max_age_hours=24
        )

        assert isinstance(freshness_check, dict)
        assert 'freshness_score' in freshness_check
        assert 'stale_records' in freshness_check
        assert len(freshness_check['stale_records']) > 0

    def test_data_accuracy_assessment(self, quality_checker):
        """测试数据准确性评估"""
        # 模拟基准数据和测试数据
        benchmark_data = pd.DataFrame({
            'symbol': ['000001', '000002', '000003'],
            'price': [100.0, 200.0, 300.0]
        })

        # 测试数据包含一些误差
        test_data = pd.DataFrame({
            'symbol': ['000001', '000002', '000003'],
            'price': [100.5, 198.0, 310.0]  # 有些许差异
        })

        accuracy_assessment = quality_checker.assess_data_accuracy(
            test_data, benchmark_data, 'symbol'
        )

        assert isinstance(accuracy_assessment, dict)
        assert 'accuracy_score' in accuracy_assessment
        assert 'error_distribution' in accuracy_assessment
        assert 'accuracy_by_field' in accuracy_assessment

    def test_data_format_validation(self, quality_checker):
        """测试数据格式验证"""
        # 创建包含各种格式问题的测试数据
        mixed_format_data = pd.DataFrame({
            'date_str': ['2024-01-01', '2024/01/02', '01-01-2024', 'invalid'],
            'number_str': ['100.5', '200.0', 'abc', '300.5'],
            'boolean_mixed': [True, False, 'true', 'false', 1, 0],
            'currency': ['$100', '¥200', '€300', '100USD']
        })

        format_validation = quality_checker.validate_data_formats(mixed_format_data)

        assert isinstance(format_validation, dict)
        assert 'format_issues' in format_validation
        assert 'standardization_suggestions' in format_validation
        assert len(format_validation['format_issues']) > 0

    def test_duplicate_detection_and_handling(self, quality_checker):
        """测试重复数据检测和处理"""
        # 创建包含重复的数据
        data_with_duplicates = pd.DataFrame({
            'symbol': ['000001', '000002', '000001', '000003', '000002', '000001'],
            'price': [100.0, 200.0, 100.0, 300.0, 200.0, 105.0],  # 有些价格不同
            'volume': [1000, 2000, 1000, 3000, 2000, 1500]
        })

        duplicate_analysis = quality_checker.analyze_and_handle_duplicates(
            data_with_duplicates, subset=['symbol']
        )

        assert isinstance(duplicate_analysis, dict)
        assert 'duplicate_summary' in duplicate_analysis
        assert 'duplicate_groups' in duplicate_analysis
        assert 'deduplication_suggestions' in duplicate_analysis

        # 应该检测到重复
        assert duplicate_analysis['duplicate_summary']['total_duplicates'] > 0

    def test_data_distribution_analysis(self, quality_checker):
        """测试数据分布分析"""
        # 创建不同分布特征的数据
        np.random.seed(42)
        data = pd.DataFrame({
            'normal_feature': np.random.normal(100, 10, 1000),
            'skewed_feature': np.random.exponential(2, 1000),
            'uniform_feature': np.random.uniform(0, 100, 1000),
            'outlier_feature': np.concatenate([
                np.random.normal(50, 5, 990),
                np.array([200, -50, 300])  # 异常值
            ])
        })

        distribution_analysis = quality_checker.analyze_data_distributions(data)

        assert isinstance(distribution_analysis, dict)
        assert 'distribution_stats' in distribution_analysis
        assert 'normality_tests' in distribution_analysis
        assert 'outlier_analysis' in distribution_analysis

    def test_data_quality_monitoring(self, quality_checker):
        """测试数据质量监控"""
        # 模拟时间序列质量指标
        quality_history = [
            {'date': '2024-01-01', 'completeness': 0.95, 'accuracy': 0.98, 'timeliness': 0.92},
            {'date': '2024-01-02', 'completeness': 0.93, 'accuracy': 0.97, 'timeliness': 0.88},
            {'date': '2024-01-03', 'completeness': 0.90, 'accuracy': 0.95, 'timeliness': 0.85},
            {'date': '2024-01-04', 'completeness': 0.85, 'accuracy': 0.92, 'timeliness': 0.80}
        ]

        quality_monitoring = quality_checker.monitor_quality_trends(quality_history)

        assert isinstance(quality_monitoring, dict)
        assert 'trend_analysis' in quality_monitoring
        assert 'quality_alerts' in quality_monitoring
        assert 'predictive_insights' in quality_monitoring

    def test_data_lineage_tracking(self, quality_checker):
        """测试数据血缘追踪"""
        # 模拟数据处理流程
        data_lineage = {
            'source': 'raw_market_data.csv',
            'transformations': [
                {'step': 'load', 'timestamp': '2024-01-01T10:00:00', 'rows': 10000},
                {'step': 'clean', 'timestamp': '2024-01-01T10:05:00', 'rows': 9800, 'removed': 200},
                {'step': 'validate', 'timestamp': '2024-01-01T10:10:00', 'rows': 9700, 'invalid': 100},
                {'step': 'transform', 'timestamp': '2024-01-01T10:15:00', 'rows': 9700, 'new_features': 5}
            ],
            'quality_checks': [
                {'check': 'completeness', 'timestamp': '2024-01-01T10:12:00', 'score': 0.97},
                {'check': 'accuracy', 'timestamp': '2024-01-01T10:17:00', 'score': 0.95}
            ]
        }

        lineage_analysis = quality_checker.track_data_lineage(data_lineage)

        assert isinstance(lineage_analysis, dict)
        assert 'lineage_integrity' in lineage_analysis
        assert 'transformation_chain' in lineage_analysis
        assert 'quality_evolution' in lineage_analysis

    def test_data_security_validation(self, quality_checker):
        """测试数据安全验证"""
        # 包含敏感信息的数据
        sensitive_data = pd.DataFrame({
            'user_id': ['user123', 'user456', 'user789'],
            'account_number': ['123456789', '987654321', '456789123'],
            'balance': [10000.50, 25000.75, 5000.25],
            'ssn': ['123-45-6789', '987-65-4321', '456-78-9123'],  # 敏感信息
            'email': ['user1@email.com', 'user2@email.com', 'user3@email.com']
        })

        security_validation = quality_checker.validate_data_security(sensitive_data)

        assert isinstance(security_validation, dict)
        assert 'security_score' in security_validation
        assert 'sensitive_fields_detected' in security_validation
        assert 'anonymization_required' in security_validation
        assert len(security_validation['sensitive_fields_detected']) > 0

    def test_performance_data_quality(self, quality_checker):
        """测试性能数据质量"""
        # 模拟系统性能指标数据
        performance_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'cpu_usage': np.random.uniform(10, 90, 100),
            'memory_usage': np.random.uniform(20, 95, 100),
            'response_time': np.random.exponential(100, 100),  # 指数分布
            'error_rate': np.random.beta(1, 99, 100)  # 低错误率
        })

        # 引入一些异常值
        performance_data.loc[50, 'response_time'] = 5000  # 异常响应时间
        performance_data.loc[75, 'error_rate'] = 0.5  # 高错误率

        performance_quality = quality_checker.assess_performance_data_quality(performance_data)

        assert isinstance(performance_quality, dict)
        assert 'performance_metrics_quality' in performance_quality
        assert 'anomaly_detection' in performance_quality
        assert 'trend_analysis' in performance_quality

    def test_compliance_data_quality(self, quality_checker):
        """测试合规数据质量"""
        # 模拟合规相关数据
        compliance_data = pd.DataFrame({
            'trade_id': range(1, 101),
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'user_id': ['user_' + str(i % 10) for i in range(100)],
            'amount': np.random.uniform(100, 10000, 100),
            'compliance_flags': [''] * 95 + ['suspicious'] * 5,  # 5个可疑交易
            'geolocation': ['CN'] * 80 + ['US'] * 15 + ['Unknown'] * 5
        })

        compliance_quality = quality_checker.validate_compliance_data_quality(compliance_data)

        assert isinstance(compliance_quality, dict)
        assert 'regulatory_compliance_score' in compliance_quality
        assert 'data_integrity_checks' in compliance_quality
        assert 'audit_trail_completeness' in compliance_quality

    def test_real_time_data_quality_monitoring(self, quality_checker):
        """测试实时数据质量监控"""
        # 模拟实时数据流
        real_time_data = [
            {'timestamp': datetime.now() - timedelta(seconds=10), 'value': 100.5, 'quality_score': 0.95},
            {'timestamp': datetime.now() - timedelta(seconds=5), 'value': None, 'quality_score': 0.80},  # 缺失值
            {'timestamp': datetime.now(), 'value': 9999.99, 'quality_score': 0.60}  # 异常值
        ]

        real_time_monitoring = quality_checker.monitor_real_time_data_quality(real_time_data)

        assert isinstance(real_time_monitoring, dict)
        assert 'current_quality_status' in real_time_monitoring
        assert 'quality_trends' in real_time_monitoring
        assert 'alerts_triggered' in real_time_monitoring

    def test_data_quality_benchmarking(self, quality_checker):
        """测试数据质量基准测试"""
        # 模拟多个数据源的质量比较
        data_sources = {
            'source_a': {
                'completeness': 0.95,
                'accuracy': 0.98,
                'timeliness': 0.92,
                'consistency': 0.96
            },
            'source_b': {
                'completeness': 0.88,
                'accuracy': 0.95,
                'timeliness': 0.85,
                'consistency': 0.90
            },
            'source_c': {
                'completeness': 0.97,
                'accuracy': 0.99,
                'timeliness': 0.98,
                'consistency': 0.97
            }
        }

        industry_benchmarks = {
            'completeness': 0.90,
            'accuracy': 0.95,
            'timeliness': 0.85,
            'consistency': 0.92
        }

        benchmarking_results = quality_checker.benchmark_data_quality(
            data_sources, industry_benchmarks
        )

        assert isinstance(benchmarking_results, dict)
        assert 'benchmark_comparison' in benchmarking_results
        assert 'best_performing_source' in benchmarking_results
        assert 'improvement_opportunities' in benchmarking_results

    def test_automated_data_cleansing(self, quality_checker):
        """测试自动化数据清洗"""
        # 创建需要清洗的脏数据
        dirty_data = pd.DataFrame({
            'name': ['John', 'Jane', 'Bob', 'Alice', None],  # 缺失值
            'age': [25, 30, 'thirty', 40, 35],  # 混合类型
            'salary': ['$50,000', '60000', '70k', '80000', '90000'],  # 不一致格式
            'department': ['HR', 'hr', 'IT', 'it', 'Finance']  # 不一致大小写
        })

        cleaning_pipeline = quality_checker.create_automated_cleaning_pipeline(dirty_data)

        assert isinstance(cleaning_pipeline, dict)
        assert 'cleaning_steps' in cleaning_pipeline
        assert 'expected_improvements' in cleaning_pipeline
        assert 'validation_checks' in cleaning_pipeline

    def test_advanced_quality_monitor_initialization(self, advanced_quality_monitor):
        """测试高级质量监控器初始化"""
        assert advanced_quality_monitor is not None
        assert hasattr(advanced_quality_monitor, 'config')
        assert hasattr(advanced_quality_monitor, 'quality_metrics')

    def test_advanced_quality_monitor_comprehensive_check(self, advanced_quality_monitor, sample_stock_data):
        """测试高级质量监控器综合检查"""
        # 执行综合质量检查
        quality_report = advanced_quality_monitor.monitor_comprehensive_quality(sample_stock_data)

        # 检查报告结构
        assert isinstance(quality_report, dict)
        assert 'overall_score' in quality_report
        assert 'dimensions' in quality_report
        assert 'issues' in quality_report

        # 检查分数在合理范围内
        assert 0 <= quality_report['overall_score'] <= 100

    def test_advanced_quality_monitor_cross_source_consistency(self, advanced_quality_monitor):
        """测试高级质量监控器跨数据源一致性"""
        # 创建多个数据源的数据
        source1_data = pd.DataFrame({
            'symbol': ['AAPL'] * 10,
            'price': np.random.uniform(100, 200, 10),
            'volume': np.random.randint(1000, 10000, 10)
        })

        source2_data = pd.DataFrame({
            'symbol': ['AAPL'] * 10,
            'price': source1_data['price'] * 1.01,  # 轻微差异
            'volume': source1_data['volume']
        })

        consistency_report = advanced_quality_monitor.check_cross_source_consistency(
            [source1_data, source2_data]
        )

        assert isinstance(consistency_report, dict)
        assert 'consistency_score' in consistency_report
        assert 'discrepancies' in consistency_report

    def test_advanced_quality_monitor_timeliness(self, advanced_quality_monitor):
        """测试高级质量监控器时效性"""
        # 创建带时间戳的数据
        timely_data = pd.DataFrame({
            'symbol': ['AAPL'] * 10,
            'price': np.random.uniform(100, 200, 10),
            'timestamp': [datetime.now() - timedelta(minutes=i) for i in range(10)]
        })

        timeliness_report = advanced_quality_monitor.monitor_timeliness(timely_data, 'timestamp')

        assert isinstance(timeliness_report, dict)
        assert 'timeliness_score' in timeliness_report
        assert 'max_age' in timeliness_report
        assert 'avg_age' in timeliness_report

    def test_advanced_quality_monitor_data_integrity(self, advanced_quality_monitor, sample_stock_data):
        """测试高级质量监控器数据完整性"""
        integrity_report = advanced_quality_monitor.check_data_integrity(sample_stock_data)

        assert isinstance(integrity_report, dict)
        assert 'integrity_score' in integrity_report
        assert 'corruption_indicators' in integrity_report

    def test_advanced_quality_monitor_accuracy_assessment(self, advanced_quality_monitor, sample_stock_data):
        """测试高级质量监控器准确性评估"""
        # 添加一些可能的异常值
        accuracy_data = sample_stock_data.copy()
        accuracy_data.loc[0, 'close'] = 10000  # 明显的异常值

        accuracy_report = advanced_quality_monitor.assess_accuracy(accuracy_data)

        assert isinstance(accuracy_report, dict)
        assert 'accuracy_score' in accuracy_report
        assert 'outliers_detected' in accuracy_report
        assert accuracy_report['outliers_detected'] > 0

    def test_advanced_quality_monitor_reliability_evaluation(self, advanced_quality_monitor, sample_stock_data):
        """测试高级质量监控器可靠性评估"""
        reliability_report = advanced_quality_monitor.evaluate_reliability(sample_stock_data)

        assert isinstance(reliability_report, dict)
        assert 'reliability_score' in reliability_report
        assert 'confidence_intervals' in reliability_report

    def test_advanced_quality_monitor_real_time_monitoring(self, advanced_quality_monitor, sample_stock_data):
        """测试高级质量监控器实时监控"""
        # 启用实时监控
        advanced_quality_monitor.enable_real_time_monitoring()

        # 处理数据流
        quality_scores = []
        for i in range(5):
            chunk = sample_stock_data.head(5)
            score = advanced_quality_monitor.monitor_real_time(chunk)
            quality_scores.append(score)

        # 检查实时监控结果
        assert len(quality_scores) == 5
        for score in quality_scores:
            assert isinstance(score, (int, float))
            assert 0 <= score <= 100

    def test_advanced_quality_monitor_automatic_repair(self, advanced_quality_monitor):
        """测试高级质量监控器自动修复"""
        # 创建包含质量问题的数据
        problematic_data = pd.DataFrame({
            'symbol': ['AAPL', 'GOOGL', None, 'AAPL'],
            'price': [100, 200, 150, None],  # 包含缺失值
            'volume': [1000, 2000, 1500, 1200]
        })

        # 执行自动修复
        repaired_data = advanced_quality_monitor.auto_repair_data(problematic_data)

        # 检查修复结果
        assert repaired_data is not None
        assert isinstance(repaired_data, pd.DataFrame)

        # 检查缺失值是否被处理
        assert not repaired_data['symbol'].isna().any()

    def test_advanced_quality_monitor_alert_generation(self, advanced_quality_monitor, sample_stock_data):
        """测试高级质量监控器告警生成"""
        # 配置告警阈值
        advanced_quality_monitor.set_alert_thresholds({
            'quality_drop': 0.8,
            'missing_data': 0.1
        })

        # 创建有质量问题的数据
        bad_quality_data = sample_stock_data.copy()
        bad_quality_data.loc[:5, 'close'] = None  # 引入缺失值

        # 检查告警生成
        alerts = advanced_quality_monitor.generate_alerts(bad_quality_data)

        assert isinstance(alerts, list)
        # 应该至少生成一个告警（由于缺失值）

    def test_advanced_quality_monitor_historical_trends(self, advanced_quality_monitor):
        """测试高级质量监控器历史趋势"""
        # 创建历史质量数据
        historical_scores = [85, 87, 82, 89, 91, 88, 86, 84, 90, 92]

        trend_analysis = advanced_quality_monitor.analyze_historical_trends(historical_scores)

        assert isinstance(trend_analysis, dict)
        assert 'trend_direction' in trend_analysis
        assert 'volatility' in trend_analysis
        assert 'stability_score' in trend_analysis

    def test_data_quality_reporting(self, quality_checker):
        """测试数据质量报告生成"""
        # 综合质量检查结果
        quality_results = {
            'completeness': {'score': 0.92, 'issues': 15},
            'accuracy': {'score': 0.95, 'errors': 8},
            'consistency': {'score': 0.88, 'violations': 25},
            'timeliness': {'score': 0.90, 'delays': 12},
            'validity': {'score': 0.93, 'invalid_records': 18}
        }

        quality_report = quality_checker.generate_comprehensive_quality_report(quality_results)

        assert isinstance(quality_report, dict)
        assert 'executive_summary' in quality_report
        assert 'detailed_findings' in quality_report
        assert 'action_items' in quality_report
        assert 'quality_trends' in quality_report
        assert 'compliance_status' in quality_report
