#!/usr/bin/env python3
"""
生产环境配置文件

定义生产环境的各项配置参数
"""

import os
from pathlib import Path


class ProductionConfig:
    """生产环境配置"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent

        # 基础配置
        self.DEBUG = False
        self.TESTING = False
        self.SECRET_KEY = os.getenv('SECRET_KEY', 'production-secret-key-change-in-production')

        # 服务器配置
        self.HOST = os.getenv('HOST', '0.0.0.0')
        self.PORT = int(os.getenv('PORT', '8080'))
        self.WORKERS = int(os.getenv('WORKERS', '4'))

        # 数据库配置
        self.DATABASE_URL = os.getenv(
            'DATABASE_URL', 'postgresql://user:password@localhost:5432/rqa_prod')
        self.DB_POOL_SIZE = int(os.getenv('DB_POOL_SIZE', '20'))
        self.DB_MAX_OVERFLOW = int(os.getenv('DB_MAX_OVERFLOW', '30'))

        # Redis配置
        self.REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
        self.REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
        self.REDIS_DB = int(os.getenv('REDIS_DB', '0'))
        self.REDIS_PASSWORD = os.getenv('REDIS_PASSWORD')
        self.REDIS_MAX_CONNECTIONS = int(os.getenv('REDIS_MAX_CONNECTIONS', '20'))

        # 缓存配置
        self.CACHE_TYPE = os.getenv('CACHE_TYPE', 'redis')
        self.CACHE_DEFAULT_TIMEOUT = int(os.getenv('CACHE_DEFAULT_TIMEOUT', '3600'))
        self.CACHE_KEY_PREFIX = os.getenv('CACHE_KEY_PREFIX', 'rqa_prod')

        # 日志配置
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
        self.LOG_FILE = os.getenv('LOG_FILE', '/var/log/rqa/production.log')
        self.LOG_MAX_SIZE = int(os.getenv('LOG_MAX_SIZE', '10485760'))  # 10MB
        self.LOG_BACKUP_COUNT = int(os.getenv('LOG_BACKUP_COUNT', '5'))

        # 安全配置
        self.JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'jwt-secret-key-change-in-production')
        self.JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(
            os.getenv('JWT_ACCESS_TOKEN_EXPIRE_MINUTES', '30'))
        self.JWT_REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv('JWT_REFRESH_TOKEN_EXPIRE_DAYS', '7'))

        # API配置
        self.API_TITLE = "RQA2025 Production API"
        self.API_VERSION = "1.0.0"
        self.API_DESCRIPTION = "Production API for RQA2025 system"
        self.API_PREFIX = "/api/v1"

        # 速率限制
        self.RATE_LIMIT_REQUESTS = int(os.getenv('RATE_LIMIT_REQUESTS', '100'))
        self.RATE_LIMIT_WINDOW = int(os.getenv('RATE_LIMIT_WINDOW', '60'))  # 60秒

        # 监控配置
        self.MONITORING_ENABLED = os.getenv('MONITORING_ENABLED', 'true').lower() == 'true'
        self.METRICS_COLLECTION_INTERVAL = int(os.getenv('METRICS_COLLECTION_INTERVAL', '30'))
        self.HEALTH_CHECK_INTERVAL = int(os.getenv('HEALTH_CHECK_INTERVAL', '60'))

        # 告警配置
        self.ALERT_EMAIL_ENABLED = os.getenv('ALERT_EMAIL_ENABLED', 'false').lower() == 'true'
        self.ALERT_EMAIL_RECIPIENTS = os.getenv('ALERT_EMAIL_RECIPIENTS', '').split(',')
        self.ALERT_SLACK_WEBHOOK = os.getenv('ALERT_SLACK_WEBHOOK', '')

        # 第三方服务配置
        self.EXTERNAL_API_TIMEOUT = int(os.getenv('EXTERNAL_API_TIMEOUT', '10'))
        self.EXTERNAL_API_RETRIES = int(os.getenv('EXTERNAL_API_RETRIES', '3'))

        # 功能开关
        self.FEATURE_CACHE_ENABLED = os.getenv('FEATURE_CACHE_ENABLED', 'true').lower() == 'true'
        self.FEATURE_ASYNC_PROCESSING = os.getenv(
            'FEATURE_ASYNC_PROCESSING', 'true').lower() == 'true'
        self.FEATURE_METRICS_ENABLED = os.getenv(
            'FEATURE_METRICS_ENABLED', 'true').lower() == 'true'

        # 性能配置
        self.MAX_REQUEST_SIZE = int(os.getenv('MAX_REQUEST_SIZE', '1048576'))  # 1MB
        self.REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '30'))
        self.CONNECTION_POOL_SIZE = int(os.getenv('CONNECTION_POOL_SIZE', '20'))

        # 备份配置
        self.BACKUP_ENABLED = os.getenv('BACKUP_ENABLED', 'true').lower() == 'true'
        self.BACKUP_INTERVAL_HOURS = int(os.getenv('BACKUP_INTERVAL_HOURS', '24'))
        self.BACKUP_RETENTION_DAYS = int(os.getenv('BACKUP_RETENTION_DAYS', '30'))

        # 环境特定配置
        self.ENVIRONMENT = os.getenv('ENVIRONMENT', 'production')
        self.REGION = os.getenv('REGION', 'us-east-1')
        self.CLUSTER_NAME = os.getenv('CLUSTER_NAME', 'rqa-prod-cluster')

    def to_dict(self):
        """转换为字典格式"""
        return {key: value for key, value in self.__dict__.items()
                if not key.startswith('_') and not callable(value)}

    def validate_config(self):
        """验证配置的完整性"""
        required_configs = [
            'SECRET_KEY', 'DATABASE_URL', 'REDIS_HOST',
            'JWT_SECRET_KEY', 'LOG_FILE'
        ]

        missing_configs = []
        for config in required_configs:
            value = getattr(self, config, None)
            if not value or str(value).startswith('change-in-production'):
                missing_configs.append(config)

        if missing_configs:
            raise ValueError(f"Missing required production configurations: {missing_configs}")

        return True

    def get_database_config(self):
        """获取数据库配置"""
        return {
            'url': self.DATABASE_URL,
            'pool_size': self.DB_POOL_SIZE,
            'max_overflow': self.DB_MAX_OVERFLOW,
            'pool_pre_ping': True,
            'pool_recycle': 3600
        }

    def get_cache_config(self):
        """获取缓存配置"""
        return {
            'type': self.CACHE_TYPE,
            'redis_host': self.REDIS_HOST,
            'redis_port': self.REDIS_PORT,
            'redis_db': self.REDIS_DB,
            'redis_password': self.REDIS_PASSWORD,
            'max_connections': self.REDIS_MAX_CONNECTIONS,
            'default_timeout': self.CACHE_DEFAULT_TIMEOUT,
            'key_prefix': self.CACHE_KEY_PREFIX
        }

    def get_logging_config(self):
        """获取日志配置"""
        return {
            'level': self.LOG_LEVEL,
            'file': self.LOG_FILE,
            'max_size': self.LOG_MAX_SIZE,
            'backup_count': self.LOG_BACKUP_COUNT,
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }

    def get_monitoring_config(self):
        """获取监控配置"""
        return {
            'enabled': self.MONITORING_ENABLED,
            'collection_interval': self.METRICS_COLLECTION_INTERVAL,
            'health_check_interval': self.HEALTH_CHECK_INTERVAL,
            'alert_email_enabled': self.ALERT_EMAIL_ENABLED,
            'alert_email_recipients': self.ALERT_EMAIL_RECIPIENTS,
            'alert_slack_webhook': self.ALERT_SLACK_WEBHOOK
        }

    def get_security_config(self):
        """获取安全配置"""
        return {
            'jwt_secret_key': self.JWT_SECRET_KEY,
            'jwt_access_token_expire_minutes': self.JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
            'jwt_refresh_token_expire_days': self.JWT_REFRESH_TOKEN_EXPIRE_DAYS,
            'rate_limit_requests': self.RATE_LIMIT_REQUESTS,
            'rate_limit_window': self.RATE_LIMIT_WINDOW
        }


# 全局配置实例
_config = None


def get_config():
    """获取全局配置实例"""
    global _config
    if _config is None:
        _config = ProductionConfig()
    return _config


def load_config_from_env():
    """从环境变量加载配置"""
    config = get_config()
    # 环境变量已在__init__中通过os.getenv()加载
    return config


def validate_production_config():
    """验证生产配置"""
    config = get_config()
    return config.validate_config()


def get_config_summary():
    """获取配置摘要"""
    config = get_config()

    return {
        'environment': config.ENVIRONMENT,
        'database': 'configured' if config.DATABASE_URL else 'missing',
        'redis': 'configured' if config.REDIS_HOST else 'missing',
        'logging': 'configured' if config.LOG_FILE else 'missing',
        'monitoring': 'enabled' if config.MONITORING_ENABLED else 'disabled',
        'security': 'configured' if config.JWT_SECRET_KEY else 'missing',
        'features': {
            'cache': 'enabled' if config.FEATURE_CACHE_ENABLED else 'disabled',
            'async': 'enabled' if config.FEATURE_ASYNC_PROCESSING else 'disabled',
            'metrics': 'enabled' if config.FEATURE_METRICS_ENABLED else 'disabled'
        }
    }


# 便捷函数
def init_app_config():
    """初始化应用配置"""
    config = load_config_from_env()
    config.validate_config()
    return config


def get_database_url():
    """获取数据库URL"""
    return get_config().DATABASE_URL


def is_production():
    """检查是否为生产环境"""
    return get_config().ENVIRONMENT == 'production'


def get_log_level():
    """获取日志级别"""
    return get_config().LOG_LEVEL


if __name__ == "__main__":
    # 配置验证示例
    try:
        config = init_app_config()
        summary = get_config_summary()

        print("=== 生产环境配置验证 ===\n")

        print("✅ 配置加载成功!")
        print(f"环境: {summary['environment']}")
        print(f"数据库: {summary['database']}")
        print(f"Redis: {summary['redis']}")
        print(f"日志: {summary['logging']}")
        print(f"监控: {summary['monitoring']}")
        print(f"安全: {summary['security']}")

        print("\n功能开关:")
        for feature, status in summary['features'].items():
            print(f"  • {feature}: {status}")

        print("\n✅ 生产配置验证通过!")

    except ValueError as e:
        print(f"❌ 配置验证失败: {e}")
        print("\n请设置以下环境变量:")
        print("  - SECRET_KEY")
        print("  - DATABASE_URL")
        print("  - REDIS_HOST")
        print("  - JWT_SECRET_KEY")
        print("  - LOG_FILE")
