#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试验证结果
"""

import tempfile
import shutil
from src.infrastructure.core.config.environment_manager import EnvironmentConfigManager, ProductionConfigValidator


def debug_validation():
    """调试验证结果"""
    temp_dir = tempfile.mkdtemp()
    try:
        config_manager = EnvironmentConfigManager(temp_dir, "production")
        validator = ProductionConfigValidator(config_manager)

        # 设置生产环境配置
        config_manager.set_config("database.host", "prod-db.example.com")
        config_manager.set_config("database.port", 5432)
        config_manager.set_config("redis.host", "prod-redis.example.com")
        config_manager.set_config("redis.port", 6379)
        config_manager.set_config("security.encryption_enabled", True)
        config_manager.set_config("security.session_timeout", 1800)
        config_manager.set_config("security.max_login_attempts", 5)
        config_manager.set_config("performance.cache_ttl", 3600)
        config_manager.set_config("performance.max_connections", 100)
        config_manager.set_config("performance.timeout", 30)
        config_manager.set_config("monitoring.enabled", True)
        config_manager.set_config("monitoring.metrics_interval", 60)
        config_manager.set_config("monitoring.alert_threshold", 0.8)
        config_manager.set_config("backup.enabled", True)

        # 创建备份文件
        backup_path = config_manager.backup_config("test_backup")
        print(f"备份文件路径: {backup_path}")

        # 验证生产环境就绪
        result = validator.validate_production_ready()
        print("验证结果:")
        print(f"整体结果: {result['overall']}")
        print(f"安全验证: {result['security']}")
        print(f"性能验证: {result['performance']}")
        print(f"监控验证: {result['monitoring']}")
        print(f"备份验证: {result['backup']}")

        # 单独测试备份验证
        backup_result = validator._validate_backup()
        print(f"备份验证结果: {backup_result}")

        # 检查备份目录
        backup_dir = config_manager.base_config_path / "backups"
        print(f"备份目录: {backup_dir}")
        print(f"备份目录存在: {backup_dir.exists()}")
        if backup_dir.exists():
            backup_files = list(backup_dir.glob("*.yaml"))
            print(f"备份文件: {backup_files}")

    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    debug_validation()
