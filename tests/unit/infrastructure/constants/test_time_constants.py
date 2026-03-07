"""
测试时间相关常量定义

覆盖 TimeConstants 类的所有常量值
"""

import pytest
from src.infrastructure.constants.time_constants import TimeConstants


class TestTimeConstants:
    """TimeConstants 单元测试"""

    def test_basic_time_units(self):
        """测试基础时间单位"""
        assert TimeConstants.SECOND == 1
        assert TimeConstants.MINUTE == 60
        assert TimeConstants.HOUR == 3600
        assert TimeConstants.DAY == 86400
        assert TimeConstants.WEEK == 604800

    def test_monitor_intervals(self):
        """测试监控间隔常量"""
        assert TimeConstants.MONITOR_INTERVAL_FAST == 5
        assert TimeConstants.MONITOR_INTERVAL_NORMAL == 30
        assert TimeConstants.MONITOR_INTERVAL_SLOW == 60
        assert TimeConstants.MONITOR_INTERVAL_VERY_SLOW == 300

    def test_health_check_intervals(self):
        """测试健康检查间隔常量"""
        assert TimeConstants.HEALTH_CHECK_INTERVAL_FAST == 10
        assert TimeConstants.HEALTH_CHECK_INTERVAL_NORMAL == 30
        assert TimeConstants.HEALTH_CHECK_INTERVAL_SLOW == 60

    def test_timeout_settings(self):
        """测试超时设置常量"""
        assert TimeConstants.TIMEOUT_SHORT == 5
        assert TimeConstants.TIMEOUT_NORMAL == 30
        assert TimeConstants.TIMEOUT_LONG == 60
        assert TimeConstants.TIMEOUT_VERY_LONG == 300

    def test_retry_delays(self):
        """测试重试延迟常量"""
        assert TimeConstants.RETRY_DELAY_FAST == 1
        assert TimeConstants.RETRY_DELAY_NORMAL == 5
        assert TimeConstants.RETRY_DELAY_SLOW == 10

    def test_lock_timeouts(self):
        """测试锁超时常量"""
        assert TimeConstants.LOCK_TIMEOUT_SHORT == 10
        assert TimeConstants.LOCK_TIMEOUT_NORMAL == 30
        assert TimeConstants.LOCK_TIMEOUT_LONG == 60

    def test_data_retention_periods(self):
        """测试数据保留期常量"""
        assert TimeConstants.RETENTION_METRICS == 30
        assert TimeConstants.RETENTION_LOGS == 90
        assert TimeConstants.RETENTION_CACHE == 7
        assert TimeConstants.RETENTION_BACKUPS == 30
        assert TimeConstants.RETENTION_VERSIONS == 30

    def test_refresh_intervals(self):
        """测试刷新间隔常量"""
        assert TimeConstants.REFRESH_INTERVAL_FAST == 5
        assert TimeConstants.REFRESH_INTERVAL_NORMAL == 30
        assert TimeConstants.REFRESH_INTERVAL_SLOW == 60

    def test_alert_cooldown_times(self):
        """测试告警冷却时间常量"""
        assert TimeConstants.ALERT_COOLDOWN_SHORT == 60
        assert TimeConstants.ALERT_COOLDOWN_NORMAL == 300
        assert TimeConstants.ALERT_COOLDOWN_LONG == 3600

    def test_session_timeouts(self):
        """测试会话超时常量"""
        assert TimeConstants.SESSION_TIMEOUT == 1800
        assert TimeConstants.TOKEN_EXPIRY == 3600
        assert TimeConstants.REFRESH_TOKEN_EXPIRY == 604800

    def test_time_unit_progression(self):
        """测试时间单位递增规律"""
        # 验证分钟 = 秒 * 60
        assert TimeConstants.MINUTE == TimeConstants.SECOND * 60
        # 验证小时 = 分钟 * 60
        assert TimeConstants.HOUR == TimeConstants.MINUTE * 60
        # 验证天 = 小时 * 24
        assert TimeConstants.DAY == TimeConstants.HOUR * 24
        # 验证周 = 天 * 7
        assert TimeConstants.WEEK == TimeConstants.DAY * 7

    def test_monitor_interval_progression(self):
        """测试监控间隔递增规律"""
        intervals = [
            TimeConstants.MONITOR_INTERVAL_FAST,
            TimeConstants.MONITOR_INTERVAL_NORMAL,
            TimeConstants.MONITOR_INTERVAL_SLOW,
            TimeConstants.MONITOR_INTERVAL_VERY_SLOW
        ]

        # 验证递增顺序
        for i in range(len(intervals) - 1):
            assert intervals[i] < intervals[i + 1]

    def test_health_check_interval_progression(self):
        """测试健康检查间隔递增规律"""
        intervals = [
            TimeConstants.HEALTH_CHECK_INTERVAL_FAST,
            TimeConstants.HEALTH_CHECK_INTERVAL_NORMAL,
            TimeConstants.HEALTH_CHECK_INTERVAL_SLOW
        ]

        # 验证递增顺序
        for i in range(len(intervals) - 1):
            assert intervals[i] < intervals[i + 1]

    def test_timeout_progression(self):
        """测试超时设置递增规律"""
        timeouts = [
            TimeConstants.TIMEOUT_SHORT,
            TimeConstants.TIMEOUT_NORMAL,
            TimeConstants.TIMEOUT_LONG,
            TimeConstants.TIMEOUT_VERY_LONG
        ]

        # 验证递增顺序
        for i in range(len(timeouts) - 1):
            assert timeouts[i] < timeouts[i + 1]

    def test_retry_delay_progression(self):
        """测试重试延迟递增规律"""
        delays = [
            TimeConstants.RETRY_DELAY_FAST,
            TimeConstants.RETRY_DELAY_NORMAL,
            TimeConstants.RETRY_DELAY_SLOW
        ]

        # 验证递增顺序
        for i in range(len(delays) - 1):
            assert delays[i] < delays[i + 1]

    def test_lock_timeout_progression(self):
        """测试锁超时递增规律"""
        timeouts = [
            TimeConstants.LOCK_TIMEOUT_SHORT,
            TimeConstants.LOCK_TIMEOUT_NORMAL,
            TimeConstants.LOCK_TIMEOUT_LONG
        ]

        # 验证递增顺序
        for i in range(len(timeouts) - 1):
            assert timeouts[i] < timeouts[i + 1]

    def test_refresh_interval_progression(self):
        """测试刷新间隔递增规律"""
        intervals = [
            TimeConstants.REFRESH_INTERVAL_FAST,
            TimeConstants.REFRESH_INTERVAL_NORMAL,
            TimeConstants.REFRESH_INTERVAL_SLOW
        ]

        # 验证递增顺序
        for i in range(len(intervals) - 1):
            assert intervals[i] < intervals[i + 1]

    def test_alert_cooldown_progression(self):
        """测试告警冷却时间递增规律"""
        cooldowns = [
            TimeConstants.ALERT_COOLDOWN_SHORT,
            TimeConstants.ALERT_COOLDOWN_NORMAL,
            TimeConstants.ALERT_COOLDOWN_LONG
        ]

        # 验证递增顺序
        for i in range(len(cooldowns) - 1):
            assert cooldowns[i] < cooldowns[i + 1]

    def test_session_timeout_relationships(self):
        """测试会话超时关系"""
        # 会话超时应该小于等于Token过期时间
        assert TimeConstants.SESSION_TIMEOUT <= TimeConstants.TOKEN_EXPIRY
        # Token过期时间应该小于等于刷新Token过期时间
        assert TimeConstants.TOKEN_EXPIRY <= TimeConstants.REFRESH_TOKEN_EXPIRY

    def test_monitor_health_consistency(self):
        """测试监控和健康检查的一致性"""
        # 健康检查间隔应该与监控间隔有一定关系
        assert TimeConstants.HEALTH_CHECK_INTERVAL_FAST >= TimeConstants.MONITOR_INTERVAL_FAST
        assert TimeConstants.HEALTH_CHECK_INTERVAL_NORMAL >= TimeConstants.MONITOR_INTERVAL_NORMAL

    def test_timeout_retry_consistency(self):
        """测试超时和重试的一致性"""
        # 重试延迟应该小于超时时间
        assert TimeConstants.RETRY_DELAY_NORMAL < TimeConstants.TIMEOUT_NORMAL
        assert TimeConstants.RETRY_DELAY_SLOW < TimeConstants.TIMEOUT_LONG

    def test_positive_values(self):
        """测试所有常量都是正值"""
        numeric_constants = [
            TimeConstants.SECOND,
            TimeConstants.MINUTE,
            TimeConstants.HOUR,
            TimeConstants.DAY,
            TimeConstants.WEEK,
            TimeConstants.MONITOR_INTERVAL_FAST,
            TimeConstants.MONITOR_INTERVAL_NORMAL,
            TimeConstants.MONITOR_INTERVAL_SLOW,
            TimeConstants.MONITOR_INTERVAL_VERY_SLOW,
            TimeConstants.HEALTH_CHECK_INTERVAL_FAST,
            TimeConstants.HEALTH_CHECK_INTERVAL_NORMAL,
            TimeConstants.HEALTH_CHECK_INTERVAL_SLOW,
            TimeConstants.TIMEOUT_SHORT,
            TimeConstants.TIMEOUT_NORMAL,
            TimeConstants.TIMEOUT_LONG,
            TimeConstants.TIMEOUT_VERY_LONG,
            TimeConstants.RETRY_DELAY_FAST,
            TimeConstants.RETRY_DELAY_NORMAL,
            TimeConstants.RETRY_DELAY_SLOW,
            TimeConstants.LOCK_TIMEOUT_SHORT,
            TimeConstants.LOCK_TIMEOUT_NORMAL,
            TimeConstants.LOCK_TIMEOUT_LONG,
            TimeConstants.RETENTION_METRICS,
            TimeConstants.RETENTION_LOGS,
            TimeConstants.RETENTION_CACHE,
            TimeConstants.RETENTION_BACKUPS,
            TimeConstants.RETENTION_VERSIONS,
            TimeConstants.REFRESH_INTERVAL_FAST,
            TimeConstants.REFRESH_INTERVAL_NORMAL,
            TimeConstants.REFRESH_INTERVAL_SLOW,
            TimeConstants.ALERT_COOLDOWN_SHORT,
            TimeConstants.ALERT_COOLDOWN_NORMAL,
            TimeConstants.ALERT_COOLDOWN_LONG,
            TimeConstants.SESSION_TIMEOUT,
            TimeConstants.TOKEN_EXPIRY,
            TimeConstants.REFRESH_TOKEN_EXPIRY
        ]

        for constant in numeric_constants:
            assert constant > 0, f"Constant {constant} should be positive"

    def test_reasonable_time_values(self):
        """测试合理的时间值"""
        # 监控间隔应该在合理范围内
        assert 1 <= TimeConstants.MONITOR_INTERVAL_FAST <= 10
        assert 20 <= TimeConstants.MONITOR_INTERVAL_NORMAL <= 60
        assert 30 <= TimeConstants.MONITOR_INTERVAL_SLOW <= 120

        # 超时设置应该在合理范围内
        assert 1 <= TimeConstants.TIMEOUT_SHORT <= 10
        assert 20 <= TimeConstants.TIMEOUT_NORMAL <= 60
        assert 30 <= TimeConstants.TIMEOUT_LONG <= 120

        # 数据保留期应该在合理范围内
        assert 1 <= TimeConstants.RETENTION_CACHE <= 30  # 缓存数据保留不应太长
        assert 7 <= TimeConstants.RETENTION_METRICS <= 90  # 指标数据保留适中
        assert 30 <= TimeConstants.RETENTION_LOGS <= 365  # 日志保留时间较长

        # 会话超时应该在合理范围内
        assert 600 <= TimeConstants.SESSION_TIMEOUT <= 3600  # 10分钟到1小时
        assert 1800 <= TimeConstants.TOKEN_EXPIRY <= 7200  # 30分钟到2小时
        assert 86400 <= TimeConstants.REFRESH_TOKEN_EXPIRY <= 2592000  # 1天到30天