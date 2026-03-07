"""
测试阈值相关常量定义

覆盖 ThresholdConstants 类的所有常量值
"""

import pytest
from src.infrastructure.constants.threshold_constants import ThresholdConstants


class TestThresholdConstants:
    """ThresholdConstants 单元测试"""

    def test_cpu_usage_thresholds(self):
        """测试CPU使用率阈值"""
        assert ThresholdConstants.CPU_USAGE_INFO == 50.0
        assert ThresholdConstants.CPU_USAGE_WARNING == 70.0
        assert ThresholdConstants.CPU_USAGE_CRITICAL == 80.0
        assert ThresholdConstants.CPU_USAGE_EMERGENCY == 90.0

    def test_memory_usage_thresholds(self):
        """测试内存使用率阈值"""
        assert ThresholdConstants.MEMORY_USAGE_INFO == 60.0
        assert ThresholdConstants.MEMORY_USAGE_WARNING == 75.0
        assert ThresholdConstants.MEMORY_USAGE_CRITICAL == 85.0
        assert ThresholdConstants.MEMORY_USAGE_EMERGENCY == 90.0

    def test_disk_usage_thresholds(self):
        """测试磁盘使用率阈值"""
        assert ThresholdConstants.DISK_USAGE_INFO == 70.0
        assert ThresholdConstants.DISK_USAGE_WARNING == 80.0
        assert ThresholdConstants.DISK_USAGE_CRITICAL == 90.0
        assert ThresholdConstants.DISK_USAGE_EMERGENCY == 95.0

    def test_network_thresholds(self):
        """测试网络阈值"""
        assert ThresholdConstants.NETWORK_LATENCY_WARNING == 100
        assert ThresholdConstants.NETWORK_LATENCY_CRITICAL == 500
        assert ThresholdConstants.NETWORK_PACKET_LOSS_WARNING == 1.0
        assert ThresholdConstants.NETWORK_PACKET_LOSS_CRITICAL == 5.0

    def test_database_pool_thresholds(self):
        """测试数据库连接池阈值"""
        assert ThresholdConstants.DB_POOL_WARNING == 80
        assert ThresholdConstants.DB_POOL_CRITICAL == 90

    def test_cache_hit_rate_thresholds(self):
        """测试缓存命中率阈值"""
        assert ThresholdConstants.CACHE_HIT_RATE_WARNING == 60.0
        assert ThresholdConstants.CACHE_HIT_RATE_CRITICAL == 40.0

    def test_response_time_thresholds(self):
        """测试响应时间阈值"""
        assert ThresholdConstants.RESPONSE_TIME_FAST == 100
        assert ThresholdConstants.RESPONSE_TIME_NORMAL == 500
        assert ThresholdConstants.RESPONSE_TIME_SLOW == 1000
        assert ThresholdConstants.RESPONSE_TIME_CRITICAL == 3000

    def test_error_rate_thresholds(self):
        """测试错误率阈值"""
        assert ThresholdConstants.ERROR_RATE_INFO == 0.1
        assert ThresholdConstants.ERROR_RATE_WARNING == 1.0
        assert ThresholdConstants.ERROR_RATE_CRITICAL == 5.0
        assert ThresholdConstants.ERROR_RATE_EMERGENCY == 10.0

    def test_health_score_thresholds(self):
        """测试健康评分阈值"""
        assert ThresholdConstants.HEALTH_SCORE_EXCELLENT == 90
        assert ThresholdConstants.HEALTH_SCORE_GOOD == 80
        assert ThresholdConstants.HEALTH_SCORE_WARNING == 60
        assert ThresholdConstants.HEALTH_SCORE_CRITICAL == 40

    def test_health_deduction_values(self):
        """测试健康评分扣减值"""
        assert ThresholdConstants.HEALTH_DEDUCTION_MINOR == 10
        assert ThresholdConstants.HEALTH_DEDUCTION_MEDIUM == 15
        assert ThresholdConstants.HEALTH_DEDUCTION_MAJOR == 20
        assert ThresholdConstants.HEALTH_DEDUCTION_CRITICAL == 25

    def test_cpu_threshold_progression(self):
        """测试CPU阈值递增规律"""
        cpu_thresholds = [
            ThresholdConstants.CPU_USAGE_INFO,
            ThresholdConstants.CPU_USAGE_WARNING,
            ThresholdConstants.CPU_USAGE_CRITICAL,
            ThresholdConstants.CPU_USAGE_EMERGENCY
        ]

        # 验证递增顺序
        for i in range(len(cpu_thresholds) - 1):
            assert cpu_thresholds[i] < cpu_thresholds[i + 1]

    def test_memory_threshold_progression(self):
        """测试内存阈值递增规律"""
        memory_thresholds = [
            ThresholdConstants.MEMORY_USAGE_INFO,
            ThresholdConstants.MEMORY_USAGE_WARNING,
            ThresholdConstants.MEMORY_USAGE_CRITICAL,
            ThresholdConstants.MEMORY_USAGE_EMERGENCY
        ]

        # 验证递增顺序
        for i in range(len(memory_thresholds) - 1):
            assert memory_thresholds[i] < memory_thresholds[i + 1]

    def test_disk_threshold_progression(self):
        """测试磁盘阈值递增规律"""
        disk_thresholds = [
            ThresholdConstants.DISK_USAGE_INFO,
            ThresholdConstants.DISK_USAGE_WARNING,
            ThresholdConstants.DISK_USAGE_CRITICAL,
            ThresholdConstants.DISK_USAGE_EMERGENCY
        ]

        # 验证递增顺序
        for i in range(len(disk_thresholds) - 1):
            assert disk_thresholds[i] < disk_thresholds[i + 1]

    def test_response_time_progression(self):
        """测试响应时间递增规律"""
        response_times = [
            ThresholdConstants.RESPONSE_TIME_FAST,
            ThresholdConstants.RESPONSE_TIME_NORMAL,
            ThresholdConstants.RESPONSE_TIME_SLOW,
            ThresholdConstants.RESPONSE_TIME_CRITICAL
        ]

        # 验证递增顺序
        for i in range(len(response_times) - 1):
            assert response_times[i] < response_times[i + 1]

    def test_error_rate_progression(self):
        """测试错误率递增规律"""
        error_rates = [
            ThresholdConstants.ERROR_RATE_INFO,
            ThresholdConstants.ERROR_RATE_WARNING,
            ThresholdConstants.ERROR_RATE_CRITICAL,
            ThresholdConstants.ERROR_RATE_EMERGENCY
        ]

        # 验证递增顺序
        for i in range(len(error_rates) - 1):
            assert error_rates[i] < error_rates[i + 1]

    def test_health_score_progression(self):
        """测试健康评分递减规律"""
        health_scores = [
            ThresholdConstants.HEALTH_SCORE_EXCELLENT,
            ThresholdConstants.HEALTH_SCORE_GOOD,
            ThresholdConstants.HEALTH_SCORE_WARNING,
            ThresholdConstants.HEALTH_SCORE_CRITICAL
        ]

        # 验证递减顺序（评分越高越好）
        for i in range(len(health_scores) - 1):
            assert health_scores[i] > health_scores[i + 1]

    def test_health_deduction_progression(self):
        """测试健康扣减递增规律"""
        deductions = [
            ThresholdConstants.HEALTH_DEDUCTION_MINOR,
            ThresholdConstants.HEALTH_DEDUCTION_MEDIUM,
            ThresholdConstants.HEALTH_DEDUCTION_MAJOR,
            ThresholdConstants.HEALTH_DEDUCTION_CRITICAL
        ]

        # 验证递增顺序
        for i in range(len(deductions) - 1):
            assert deductions[i] < deductions[i + 1]

    def test_network_latency_relationships(self):
        """测试网络延迟关系"""
        assert ThresholdConstants.NETWORK_LATENCY_WARNING < ThresholdConstants.NETWORK_LATENCY_CRITICAL

    def test_network_packet_loss_relationships(self):
        """测试网络丢包率关系"""
        assert ThresholdConstants.NETWORK_PACKET_LOSS_WARNING < ThresholdConstants.NETWORK_PACKET_LOSS_CRITICAL

    def test_database_pool_relationships(self):
        """测试数据库连接池关系"""
        assert ThresholdConstants.DB_POOL_WARNING < ThresholdConstants.DB_POOL_CRITICAL

    def test_cache_hit_rate_relationships(self):
        """测试缓存命中率关系"""
        # 警告阈值应该高于临界阈值（因为命中率越高越好）
        assert ThresholdConstants.CACHE_HIT_RATE_WARNING > ThresholdConstants.CACHE_HIT_RATE_CRITICAL

    def test_percentage_values(self):
        """测试百分比值在合理范围内"""
        percentage_values = [
            ThresholdConstants.CPU_USAGE_INFO,
            ThresholdConstants.CPU_USAGE_WARNING,
            ThresholdConstants.CPU_USAGE_CRITICAL,
            ThresholdConstants.CPU_USAGE_EMERGENCY,
            ThresholdConstants.MEMORY_USAGE_INFO,
            ThresholdConstants.MEMORY_USAGE_WARNING,
            ThresholdConstants.MEMORY_USAGE_CRITICAL,
            ThresholdConstants.MEMORY_USAGE_EMERGENCY,
            ThresholdConstants.DISK_USAGE_INFO,
            ThresholdConstants.DISK_USAGE_WARNING,
            ThresholdConstants.DISK_USAGE_CRITICAL,
            ThresholdConstants.DISK_USAGE_EMERGENCY,
            ThresholdConstants.NETWORK_PACKET_LOSS_WARNING,
            ThresholdConstants.NETWORK_PACKET_LOSS_CRITICAL,
            ThresholdConstants.DB_POOL_WARNING,
            ThresholdConstants.DB_POOL_CRITICAL,
            ThresholdConstants.CACHE_HIT_RATE_WARNING,
            ThresholdConstants.CACHE_HIT_RATE_CRITICAL,
            ThresholdConstants.ERROR_RATE_INFO,
            ThresholdConstants.ERROR_RATE_WARNING,
            ThresholdConstants.ERROR_RATE_CRITICAL,
            ThresholdConstants.ERROR_RATE_EMERGENCY
        ]

        for value in percentage_values:
            assert 0 <= value <= 100, f"Percentage value {value} should be between 0 and 100"

    def test_health_score_values(self):
        """测试健康评分值在合理范围内"""
        health_scores = [
            ThresholdConstants.HEALTH_SCORE_EXCELLENT,
            ThresholdConstants.HEALTH_SCORE_GOOD,
            ThresholdConstants.HEALTH_SCORE_WARNING,
            ThresholdConstants.HEALTH_SCORE_CRITICAL
        ]

        for score in health_scores:
            assert 0 <= score <= 100, f"Health score {score} should be between 0 and 100"

    def test_positive_values(self):
        """测试所有数值常量都是正值"""
        numeric_constants = [
            ThresholdConstants.CPU_USAGE_INFO,
            ThresholdConstants.CPU_USAGE_WARNING,
            ThresholdConstants.CPU_USAGE_CRITICAL,
            ThresholdConstants.CPU_USAGE_EMERGENCY,
            ThresholdConstants.MEMORY_USAGE_INFO,
            ThresholdConstants.MEMORY_USAGE_WARNING,
            ThresholdConstants.MEMORY_USAGE_CRITICAL,
            ThresholdConstants.MEMORY_USAGE_EMERGENCY,
            ThresholdConstants.DISK_USAGE_INFO,
            ThresholdConstants.DISK_USAGE_WARNING,
            ThresholdConstants.DISK_USAGE_CRITICAL,
            ThresholdConstants.DISK_USAGE_EMERGENCY,
            ThresholdConstants.NETWORK_LATENCY_WARNING,
            ThresholdConstants.NETWORK_LATENCY_CRITICAL,
            ThresholdConstants.NETWORK_PACKET_LOSS_WARNING,
            ThresholdConstants.NETWORK_PACKET_LOSS_CRITICAL,
            ThresholdConstants.DB_POOL_WARNING,
            ThresholdConstants.DB_POOL_CRITICAL,
            ThresholdConstants.CACHE_HIT_RATE_WARNING,
            ThresholdConstants.CACHE_HIT_RATE_CRITICAL,
            ThresholdConstants.RESPONSE_TIME_FAST,
            ThresholdConstants.RESPONSE_TIME_NORMAL,
            ThresholdConstants.RESPONSE_TIME_SLOW,
            ThresholdConstants.RESPONSE_TIME_CRITICAL,
            ThresholdConstants.ERROR_RATE_INFO,
            ThresholdConstants.ERROR_RATE_WARNING,
            ThresholdConstants.ERROR_RATE_CRITICAL,
            ThresholdConstants.ERROR_RATE_EMERGENCY,
            ThresholdConstants.HEALTH_SCORE_EXCELLENT,
            ThresholdConstants.HEALTH_SCORE_GOOD,
            ThresholdConstants.HEALTH_SCORE_WARNING,
            ThresholdConstants.HEALTH_SCORE_CRITICAL,
            ThresholdConstants.HEALTH_DEDUCTION_MINOR,
            ThresholdConstants.HEALTH_DEDUCTION_MEDIUM,
            ThresholdConstants.HEALTH_DEDUCTION_MAJOR,
            ThresholdConstants.HEALTH_DEDUCTION_CRITICAL
        ]

        for constant in numeric_constants:
            assert constant > 0, f"Constant {constant} should be positive"

    def test_reasonable_thresholds(self):
        """测试合理的阈值设置"""
        # CPU使用率阈值应该在合理范围内
        assert 40 <= ThresholdConstants.CPU_USAGE_INFO <= 60
        assert 65 <= ThresholdConstants.CPU_USAGE_WARNING <= 75
        assert 75 <= ThresholdConstants.CPU_USAGE_CRITICAL <= 85
        assert 85 <= ThresholdConstants.CPU_USAGE_EMERGENCY <= 95

        # 内存使用率阈值应该在合理范围内
        assert 50 <= ThresholdConstants.MEMORY_USAGE_INFO <= 70
        assert 70 <= ThresholdConstants.MEMORY_USAGE_WARNING <= 80
        assert 80 <= ThresholdConstants.MEMORY_USAGE_CRITICAL <= 90
        assert 85 <= ThresholdConstants.MEMORY_USAGE_EMERGENCY <= 95

        # 响应时间阈值应该在合理范围内
        assert 50 <= ThresholdConstants.RESPONSE_TIME_FAST <= 200
        assert 200 <= ThresholdConstants.RESPONSE_TIME_NORMAL <= 800
        assert 500 <= ThresholdConstants.RESPONSE_TIME_SLOW <= 1500
        assert 1000 <= ThresholdConstants.RESPONSE_TIME_CRITICAL <= 5000