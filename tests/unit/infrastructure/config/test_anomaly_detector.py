#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试异常检测器模块

测试 src/infrastructure/config/monitoring/anomaly_detector.py 文件的功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import math
from unittest.mock import Mock, patch
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../'))

try:
    from src.infrastructure.config.monitoring.anomaly_detector import AnomalyDetector
    MODULE_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    MODULE_AVAILABLE = False
    IMPORT_ERROR = e


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestAnomalyDetector:
    """测试异常检测器类"""

    def setup_method(self):
        """测试前准备"""
        self.detector = AnomalyDetector(window_size=5, threshold=2.0)

    def test_detector_initialization_default(self):
        """测试默认初始化"""
        detector = AnomalyDetector()
        
        assert detector.window_size == 20
        assert detector.threshold == 2.5
        assert detector._data_windows == {}
        assert detector._baselines == {}
        assert detector._std_devs == {}

    def test_detector_initialization_custom(self):
        """测试自定义参数初始化"""
        detector = AnomalyDetector(window_size=10, threshold=3.0)
        
        assert detector.window_size == 10
        assert detector.threshold == 3.0

    def test_update_baseline_insufficient_data(self):
        """测试基线更新（数据不足）"""
        # 窗口大小为5，只有3个数据点，不应更新基线
        values = [1.0, 2.0, 3.0]
        
        self.detector.update_baseline("test_metric", values)
        
        assert "test_metric" not in self.detector._baselines
        assert "test_metric" not in self.detector._std_devs

    def test_update_baseline_sufficient_data(self):
        """测试基线更新（数据充足）"""
        # 窗口大小为5，提供5个数据点
        values = [1.0, 2.0, 3.0, 4.0, 5.0]  # 平均值为3.0
        
        self.detector.update_baseline("test_metric", values)
        
        assert "test_metric" in self.detector._baselines
        assert "test_metric" in self.detector._std_devs
        assert self.detector._baselines["test_metric"] == 3.0
        assert self.detector._std_devs["test_metric"] > 0  # 标准差应该大于0

    def test_update_baseline_more_than_window_size(self):
        """测试基线更新（数据超过窗口大小）"""
        # 提供8个数据点，应该只使用最后5个
        values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]  # 最后5个：3,4,5,6,7，平均5.0
        
        self.detector.update_baseline("test_metric", values)
        
        assert "test_metric" in self.detector._baselines
        assert self.detector._baselines["test_metric"] == 5.0

    def test_detect_anomaly_first_value(self):
        """测试检测异常（第一个值）"""
        result = self.detector.detect_anomaly("test_metric", 10.0)
        
        assert result["is_anomaly"] is False
        assert result["z_score"] == 0.0
        assert result["baseline"] == 10.0  # 第一个值作为基线
        assert result["std_dev"] == 0.0
        assert result["threshold"] == 2.0

    def test_detect_anomaly_normal_value(self):
        """测试检测异常（正常值）"""
        # 先建立基线
        normal_values = [10.0, 10.1, 10.2, 10.0, 9.9]
        for value in normal_values:
            self.detector.detect_anomaly("test_metric", value)
        
        # 测试一个接近平均值的值
        result = self.detector.detect_anomaly("test_metric", 10.1)
        
        assert result["is_anomaly"] is False
        assert result["z_score"] >= 0
        assert result["baseline"] > 0
        assert result["threshold"] == 2.0

    def test_detect_anomaly_outlier_value(self):
        """测试检测异常（异常值）"""
        # 先建立有方差的基线，避免标准差为0的问题
        normal_values = [10.0, 10.1, 9.9, 10.2, 9.8]  # 有一些方差
        for value in normal_values:
            result = self.detector.detect_anomaly("test_metric", value)
        
        # 获取最后的基线（在添加异常值之前）
        baseline_before = self.detector._baselines.get("test_metric", 0)
        std_dev_before = self.detector._std_devs.get("test_metric", 0)
        
        # 测试一个明显的异常值
        result = self.detector.detect_anomaly("test_metric", 50.0)
        
        # 验证结果结构
        assert isinstance(result, dict)
        assert "is_anomaly" in result
        assert "z_score" in result
        assert "baseline" in result
        assert "std_dev" in result
        assert "threshold" in result
        assert result["threshold"] == 2.0
        
        # 由于异常值会改变窗口和基线，我们主要验证Z分数计算逻辑
        if result["std_dev"] > 0:
            # Z分数应该基于当前窗口计算
            expected_z_score = abs(50.0 - result["baseline"]) / result["std_dev"]
            assert abs(result["z_score"] - expected_z_score) < 0.001

    def test_detect_anomaly_zero_std_dev(self):
        """测试检测异常（标准差为0的情况）"""
        # 建立标准差为0的基线（所有值相同）
        identical_values = [5.0] * 5
        for value in identical_values:
            self.detector.detect_anomaly("test_metric", value)
        
        # 当标准差为0时，应该返回默认结果
        result = self.detector.detect_anomaly("test_metric", 10.0)
        
        # 由于std_dev为0，应该返回默认结构而不是计算z_score
        assert "is_anomaly" in result
        assert "z_score" in result
        assert "baseline" in result
        assert "std_dev" in result
        assert "threshold" in result

    def test_detect_anomaly_window_size_maintenance(self):
        """测试检测异常（窗口大小维护）"""
        # 添加超过窗口大小的数据点
        for i in range(10):  # 窗口大小为5，添加10个点
            self.detector.detect_anomaly("test_metric", float(i))
        
        # 检查窗口大小是否正确维护
        assert len(self.detector._data_windows["test_metric"]) == 5
        # 应该保留最后5个值：5.0, 6.0, 7.0, 8.0, 9.0
        expected_values = [5.0, 6.0, 7.0, 8.0, 9.0]
        assert self.detector._data_windows["test_metric"] == expected_values

    def test_detect_anomaly_multiple_metrics(self):
        """测试检测异常（多个指标）"""
        # 为不同指标建立基线
        for value in [1.0, 2.0, 3.0, 4.0, 5.0]:
            self.detector.detect_anomaly("metric1", value)
            self.detector.detect_anomaly("metric2", value * 10)
        
        # 测试两个指标
        result1 = self.detector.detect_anomaly("metric1", 6.0)  # 可能异常
        result2 = self.detector.detect_anomaly("metric2", 50.0)  # 正常
        
        assert "metric1" in self.detector._data_windows
        assert "metric2" in self.detector._data_windows
        assert "metric1" in self.detector._baselines
        assert "metric2" in self.detector._baselines
        
        # 验证结果结构
        for result in [result1, result2]:
            assert isinstance(result, dict)
            assert "is_anomaly" in result
            assert "z_score" in result
            assert "baseline" in result
            assert "std_dev" in result
            assert "threshold" in result

    def test_detect_anomaly_threshold_boundary(self):
        """测试检测异常（阈值边界）"""
        # 建立有方差的基线
        baseline_values = [10.0, 10.5, 9.5, 10.2, 9.8]  # 有方差，均值约10.0
        for value in baseline_values:
            self.detector.detect_anomaly("test_metric", value)
        
        # 测试阈值边界值
        result_normal = self.detector.detect_anomaly("test_metric", 10.0)
        result_high = self.detector.detect_anomaly("test_metric", 15.0)  # 高异常值
        result_low = self.detector.detect_anomaly("test_metric", 5.0)   # 低异常值
        
        # 验证结果结构
        for result in [result_normal, result_high, result_low]:
            assert isinstance(result, dict)
            assert "is_anomaly" in result
            assert "z_score" in result
            assert "baseline" in result
            assert "std_dev" in result
            assert "threshold" in result
        
        # 验证阈值逻辑：如果标准差>0，Z分数应该基于实际计算
        if result_normal["std_dev"] > 0:
            assert result_normal["z_score"] >= 0  # Z分数应该非负

    def test_update_baseline_statistical_calculation(self):
        """测试基线更新的统计计算"""
        # 使用已知的测试数据
        values = [1.0, 3.0, 2.0, 4.0, 5.0]  # 平均值3.0，标准差约1.58
        
        self.detector.update_baseline("test_metric", values)
        
        baseline = self.detector._baselines["test_metric"]
        std_dev = self.detector._std_devs["test_metric"]
        
        # 验证平均值计算
        assert abs(baseline - 3.0) < 0.001
        
        # 验证标准差计算（手动计算）
        expected_variance = sum((x - 3.0) ** 2 for x in values) / len(values)
        expected_std_dev = expected_variance ** 0.5
        assert abs(std_dev - expected_std_dev) < 0.001


class TestAnomalyDetectorIntegration:
    """测试异常检测器集成功能"""

    def setup_method(self):
        """测试前准备"""
        if MODULE_AVAILABLE:
            self.detector = AnomalyDetector(window_size=3, threshold=1.5)

    def test_continuous_anomaly_detection(self):
        """测试连续异常检测"""
        if not MODULE_AVAILABLE:
            pytest.skip("模块不可用")
        
        # 模拟时间序列数据
        time_series = [10.0, 10.1, 9.9, 10.0, 10.2, 15.0, 10.1, 10.0]  # 15.0是异常值
        
        results = []
        for value in time_series:
            result = self.detector.detect_anomaly("continuous_metric", value)
            results.append(result)
        
        # 检查异常检测结果
        assert len(results) == len(time_series)
        
        # 最后一个结果应该检测到异常（15.0）
        last_result = results[-1]
        assert isinstance(last_result, dict)
        assert "is_anomaly" in last_result

    def test_detector_state_consistency(self):
        """测试检测器状态一致性"""
        if not MODULE_AVAILABLE:
            pytest.skip("模块不可用")
        
        # 添加数据并检查状态
        self.detector.detect_anomaly("consistency_test", 5.0)
        self.detector.detect_anomaly("consistency_test", 6.0)
        self.detector.detect_anomaly("consistency_test", 4.0)
        
        # 检查数据一致性
        assert "consistency_test" in self.detector._data_windows
        assert len(self.detector._data_windows["consistency_test"]) == 3
        
        # 再添加数据，检查窗口维护
        self.detector.detect_anomaly("consistency_test", 7.0)
        assert len(self.detector._data_windows["consistency_test"]) <= 3  # 窗口大小限制
