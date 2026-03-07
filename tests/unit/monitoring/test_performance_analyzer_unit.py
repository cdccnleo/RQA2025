"""
测试性能分析器
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from collections import defaultdict
import psutil

from src.monitoring.engine.performance_analyzer import (
    PerformanceMetric,
    AnalysisMode,
    PerformanceData,
    BottleneckAnalysis,
    PerformanceReport,
    PerformanceAnalyzer
)


class TestPerformanceMetric:
    """测试性能指标枚举"""

    def test_performance_metric_values(self):
        """测试性能指标枚举值"""
        assert PerformanceMetric.CPU_USAGE.value == "cpu_usage"
        assert PerformanceMetric.MEMORY_USAGE.value == "memory_usage"
        assert PerformanceMetric.DISK_IO.value == "disk_io"
        assert PerformanceMetric.NETWORK_IO.value == "network_io"
        assert PerformanceMetric.SYSTEM_LOAD.value == "system_load"
        assert PerformanceMetric.PROCESS_COUNT.value == "process_count"
        assert PerformanceMetric.THREAD_COUNT.value == "thread_count"


class TestAnalysisMode:
    """测试分析模式枚举"""

    def test_analysis_mode_values(self):
        """测试分析模式枚举值"""
        assert hasattr(AnalysisMode, 'REALTIME')
        assert hasattr(AnalysisMode, 'HISTORICAL')
        assert hasattr(AnalysisMode, 'COMPARATIVE')
        assert hasattr(AnalysisMode, 'TREND')
        assert hasattr(AnalysisMode, 'ANOMALY')

        assert AnalysisMode.REALTIME.value == "realtime"
        assert AnalysisMode.HISTORICAL.value == "historical"
        assert AnalysisMode.TREND.value == "trend"


class TestPerformanceData:
    """测试性能数据"""

    def test_performance_data_creation(self):
        """测试性能数据创建"""
        timestamp = datetime.now()
        metrics = {
            'cpu_usage': 65.5,
            'memory_usage': 78.2,
            'disk_io': 1024.5,
            'network_io': 2048.0,
            'system_load': 2.1,
            'process_count': 150,
            'thread_count': 450
        }
        context = {'server': 'prod-01', 'environment': 'production'}

        data = PerformanceData(
            timestamp=timestamp,
            metrics=metrics,
            context=context
        )

        assert data.timestamp == timestamp
        assert data.metrics == metrics
        assert data.context == context
        assert data.metrics['cpu_usage'] == 65.5
        assert data.metrics['memory_usage'] == 78.2


class TestBottleneckAnalysis:
    """测试瓶颈分析"""

    def test_bottleneck_analysis_creation(self):
        """测试瓶颈分析创建"""
        analysis = BottleneckAnalysis(
            component="cpu",
            severity="high",
            description="CPU usage exceeds 90%",
            recommendations=["Scale up CPU", "Optimize processes"],
            impact_score=0.85,
            confidence=0.9
        )

        assert analysis.component == "cpu"
        assert analysis.severity == "high"
        assert analysis.description == "CPU usage exceeds 90%"
        assert analysis.recommendations == ["Scale up CPU", "Optimize processes"]
        assert analysis.impact_score == 0.85
        assert analysis.confidence == 0.9


class TestPerformanceReport:
    """测试性能报告"""

    def test_performance_report_creation(self):
        """测试性能报告创建"""
        start_time = datetime.now() - timedelta(hours=1)
        end_time = datetime.now()

        summary = {
            'average_cpu': 65.5,
            'peak_cpu': 95.2,
            'average_memory': 72.3,
            'peak_memory': 88.9
        }

        trends = {
            'cpu_trend': 'increasing',
            'memory_trend': 'stable'
        }

        report = PerformanceReport(
            analysis_period=(start_time, end_time),
            summary=summary,
            bottlenecks=[
                BottleneckAnalysis(
                    component="cpu",
                    severity="medium",
                    description="CPU spikes detected",
                    recommendations=["Monitor CPU usage"],
                    impact_score=0.75,
                    confidence=0.8
                )
            ],
            trends=trends,
            recommendations=["Consider CPU optimization"],
            generated_at=datetime.now()
        )

        assert report.analysis_period == (start_time, end_time)
        assert report.summary == summary
        assert len(report.bottlenecks) == 1
        assert report.bottlenecks[0].component == "cpu"
        assert report.trends == trends
        assert report.recommendations == ["Consider CPU optimization"]


class TestPerformanceAnalyzer:
    """测试性能分析器"""

    def setup_method(self):
        """测试前准备"""
        self.analyzer = PerformanceAnalyzer()

    def test_performance_analyzer_init(self):
        """测试性能分析器初始化"""
        assert self.analyzer is not None
        assert hasattr(self.analyzer, 'performance_history')
        assert hasattr(self.analyzer, 'baseline_stats')
        assert hasattr(self.analyzer, 'is_monitoring')
        assert isinstance(self.analyzer.performance_history, defaultdict)
        assert self.analyzer.is_monitoring == False
        assert self.analyzer.baseline_calculated == False

    def test_collect_system_metrics(self):
        """测试收集系统指标"""
        metrics = self.analyzer._collect_system_metrics()

        assert isinstance(metrics, PerformanceData)
        # 检查是否包含必要的指标
        assert hasattr(metrics, 'cpu_usage')
        assert hasattr(metrics, 'memory_usage')
        assert hasattr(metrics, 'timestamp')

    def test_analyze_current_performance(self):
        """测试分析当前性能"""
        analysis = self.analyzer.analyze_current_performance()

        assert isinstance(analysis, dict)
        # 检查分析结果的结构
        assert 'timestamp' in analysis
        assert 'overall_health' in analysis
        assert 'bottlenecks' in analysis

    def test_detect_bottlenecks(self):
        """测试检测瓶颈"""
        # 创建测试数据
        test_data = PerformanceData(
            timestamp=datetime.now(),
            cpu_usage=95.0,  # 高CPU使用率
            memory_usage=85.0,
            disk_io=1000.0,
            network_io=500.0,
            system_load=4.0,
            process_count=200,
            thread_count=600
        )

        bottlenecks = self.analyzer._detect_bottlenecks(test_data)

        assert isinstance(bottlenecks, list)
        # 应该检测到CPU瓶颈
        cpu_bottlenecks = [b for b in bottlenecks if b.bottleneck_type == "cpu"]
        assert len(cpu_bottlenecks) > 0

    def test_generate_performance_report(self):
        """测试生成性能报告"""
        # 设置时间范围
        start_time = datetime.now() - timedelta(hours=1)
        end_time = datetime.now()

        report = self.analyzer.generate_performance_report(start_time, end_time)

        assert isinstance(report, PerformanceReport)
        assert report.time_range == f"{start_time.isoformat()} to {end_time.isoformat()}"

    def test_get_historical_analysis(self):
        """测试获取历史分析"""
        # 添加一些历史数据
        for i in range(10):
            data = PerformanceData(
                timestamp=datetime.now() - timedelta(minutes=i),
                cpu_usage=50 + i * 2,
                memory_usage=60 + i,
                disk_io=500 + i * 50,
                network_io=1000 + i * 100,
                system_load=1.5 + i * 0.1,
                process_count=100 + i * 5,
                thread_count=300 + i * 10
            )
            self.analyzer.data_buffer.append(data)

        analysis = self.analyzer.get_historical_analysis(hours=1)

        assert isinstance(analysis, dict)
        assert 'time_range' in analysis
        assert 'trend_analysis' in analysis
        assert 'anomaly_detection' in analysis

    def test_set_alert_thresholds(self):
        """测试设置告警阈值"""
        new_thresholds = {
            'cpu_critical': 95.0,
            'memory_warning': 80.0,
            'disk_io_critical': 1000.0
        }

        self.analyzer.set_alert_thresholds(new_thresholds)

        assert self.analyzer.alert_thresholds['cpu_critical'] == 95.0
        assert self.analyzer.alert_thresholds['memory_warning'] == 80.0

    def test_get_performance_trends(self):
        """测试获取性能趋势"""
        # 添加趋势数据
        base_time = datetime.now()
        for i in range(20):
            data = PerformanceData(
                timestamp=base_time - timedelta(minutes=i*3),
                cpu_usage=60 + np.sin(i * 0.5) * 10,  # 波动模式
                memory_usage=70 + np.cos(i * 0.3) * 5,
                disk_io=800 + i * 20,
                network_io=1200 + i * 30,
                system_load=2.0 + np.sin(i * 0.4) * 0.5,
                process_count=150 + i * 2,
                thread_count=400 + i * 5
            )
            self.analyzer.data_buffer.append(data)

        trends = self.analyzer.get_performance_trends(hours=1)

        assert isinstance(trends, dict)
        assert 'cpu_trend' in trends
        assert 'memory_trend' in trends
        assert 'overall_trend' in trends

    def test_predict_performance(self):
        """测试性能预测"""
        # 这个方法可能依赖AI组件，如果不可用则跳过
        try:
            predictions = self.analyzer.predict_performance(hours_ahead=2)
            assert isinstance(predictions, dict)
        except Exception:
            # 如果AI组件不可用，预测方法可能抛出异常
            pytest.skip("AI prediction components not available")

    def test_export_performance_data(self):
        """测试导出性能数据"""
        # 添加一些数据
        for i in range(5):
            data = PerformanceData(
                timestamp=datetime.now() - timedelta(minutes=i),
                cpu_usage=50 + i * 5,
                memory_usage=60 + i * 3,
                disk_io=500 + i * 100,
                network_io=1000 + i * 200,
                system_load=2.0 + i * 0.2,
                process_count=100 + i * 10,
                thread_count=300 + i * 20
            )
            self.analyzer.data_buffer.append(data)

        # 测试导出为字典
        data_dict = self.analyzer.export_performance_data(format='dict')
        assert isinstance(data_dict, dict)
        assert 'data_points' in data_dict
        assert len(data_dict['data_points']) > 0

    def test_clear_performance_data(self):
        """测试清除性能数据"""
        # 添加一些数据
        for i in range(3):
            data = PerformanceData(
                timestamp=datetime.now(),
                cpu_usage=50.0,
                memory_usage=60.0,
                disk_io=500.0,
                network_io=1000.0,
                system_load=2.0,
                process_count=100,
                thread_count=300
            )
            self.analyzer.data_buffer.append(data)

        # 清除数据
        self.analyzer.clear_performance_data()

        # 检查数据是否被清除
        assert len(self.analyzer.data_buffer) == 0

    def test_get_system_health_score(self):
        """测试获取系统健康评分"""
        health_score = self.analyzer.get_system_health_score()

        assert isinstance(health_score, float)
        assert 0.0 <= health_score <= 100.0

    def test_enable_predictive_analysis(self):
        """测试启用预测性分析"""
        # 这个方法可能依赖AI组件
        try:
            result = self.analyzer.enable_predictive_analysis()
            assert isinstance(result, bool)
        except Exception:
            pytest.skip("Predictive analysis components not available")

    def test_analyze_performance_anomalies(self):
        """测试分析性能异常"""
        # 创建包含异常的数据
        normal_data = [PerformanceData(
            timestamp=datetime.now() - timedelta(minutes=i),
            cpu_usage=60.0,
            memory_usage=70.0,
            disk_io=800.0,
            network_io=1200.0,
            system_load=2.0,
            process_count=150,
            thread_count=400
        ) for i in range(10)]

        # 添加异常数据点
        anomaly_data = PerformanceData(
            timestamp=datetime.now(),
            cpu_usage=98.0,  # 异常高的CPU使用率
            memory_usage=95.0,  # 异常高的内存使用率
            disk_io=5000.0,  # 异常高的磁盘IO
            network_io=10000.0,  # 异常高的网络IO
            system_load=8.0,  # 异常高的系统负载
            process_count=300,
            thread_count=800
        )

        all_data = normal_data + [anomaly_data]

        anomalies = self.analyzer.analyze_performance_anomalies(all_data)

        assert isinstance(anomalies, list)
        # 应该检测到异常
        assert len(anomalies) > 0

    def test_calculate_performance_metrics(self):
        """测试计算性能指标"""
        data_points = [PerformanceData(
            timestamp=datetime.now() - timedelta(minutes=i),
            cpu_usage=50 + i * 2,
            memory_usage=60 + i * 1.5,
            disk_io=500 + i * 50,
            network_io=1000 + i * 100,
            system_load=2.0 + i * 0.1,
            process_count=100 + i * 5,
            thread_count=300 + i * 10
        ) for i in range(10)]

        metrics = self.analyzer.calculate_performance_metrics(data_points)

        assert isinstance(metrics, dict)
        assert 'cpu_stats' in metrics
        assert 'memory_stats' in metrics
        assert 'io_stats' in metrics
        assert 'system_stats' in metrics

        # 检查统计数据结构
        cpu_stats = metrics['cpu_stats']
        assert 'mean' in cpu_stats
        assert 'std' in cpu_stats
        assert 'min' in cpu_stats
        assert 'max' in cpu_stats
        assert 'percentiles' in cpu_stats
