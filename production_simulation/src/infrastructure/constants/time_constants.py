"""
时间相关常量定义
"""


class TimeConstants:
    """时间相关常量（单位：秒）"""
    
    # 基础时间单位
    SECOND = 1
    MINUTE = 60
    HOUR = 3600
    DAY = 86400
    WEEK = 604800
    
    # 监控间隔
    MONITOR_INTERVAL_FAST = 5  # 5秒
    MONITOR_INTERVAL_NORMAL = 30  # 30秒
    MONITOR_INTERVAL_SLOW = 60  # 1分钟
    MONITOR_INTERVAL_VERY_SLOW = 300  # 5分钟
    
    # 健康检查间隔
    HEALTH_CHECK_INTERVAL_FAST = 10  # 10秒
    HEALTH_CHECK_INTERVAL_NORMAL = 30  # 30秒
    HEALTH_CHECK_INTERVAL_SLOW = 60  # 1分钟
    
    # 超时设置
    TIMEOUT_SHORT = 5  # 5秒
    TIMEOUT_NORMAL = 30  # 30秒
    TIMEOUT_LONG = 60  # 1分钟
    TIMEOUT_VERY_LONG = 300  # 5分钟
    
    # 重试延迟
    RETRY_DELAY_FAST = 1  # 1秒
    RETRY_DELAY_NORMAL = 5  # 5秒
    RETRY_DELAY_SLOW = 10  # 10秒
    
    # 锁超时
    LOCK_TIMEOUT_SHORT = 10  # 10秒
    LOCK_TIMEOUT_NORMAL = 30  # 30秒
    LOCK_TIMEOUT_LONG = 60  # 1分钟
    
    # 数据保留期（天）
    RETENTION_METRICS = 30  # 指标数据保留30天
    RETENTION_LOGS = 90  # 日志保留90天
    RETENTION_CACHE = 7  # 缓存数据保留7天
    RETENTION_BACKUPS = 30  # 备份保留30天
    RETENTION_VERSIONS = 30  # 版本保留30天
    
    # 刷新间隔
    REFRESH_INTERVAL_FAST = 5  # 5秒
    REFRESH_INTERVAL_NORMAL = 30  # 30秒
    REFRESH_INTERVAL_SLOW = 60  # 1分钟
    
    # 告警冷却时间
    ALERT_COOLDOWN_SHORT = 60  # 1分钟
    ALERT_COOLDOWN_NORMAL = 300  # 5分钟
    ALERT_COOLDOWN_LONG = 3600  # 1小时
    
    # 会话超时
    SESSION_TIMEOUT = 1800  # 30分钟
    TOKEN_EXPIRY = 3600  # 1小时
    REFRESH_TOKEN_EXPIRY = 604800  # 7天

