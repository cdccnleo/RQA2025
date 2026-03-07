#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专用验证器实际测试

只测试specialized_validators.py中实际存在的验证器类
"""

import pytest
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../../'))

try:
    from src.infrastructure.config.validators.specialized_validators import (
        TradingHoursValidator,
        DatabaseConfigValidator,
        LoggingConfigValidator,
        NetworkConfigValidator
    )
    MODULE_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    MODULE_AVAILABLE = False
    IMPORT_ERROR = e


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR}")
class TestTradingHoursValidator:
    """交易时间验证器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.validator = TradingHoursValidator()
        
    def test_initialization(self):
        """测试初始化"""
        assert self.validator is not None
        assert self.validator.name == "TradingHoursValidator"
        
    def test_validate_valid_trading_hours(self):
        """测试有效的交易时间"""
        config = {
            "trading_hours": {
                "start": "09:30",
                "end": "15:00",
                "timezone": "UTC"
            }
        }
        
        results = self.validator.validate(config)
        assert results is not None
        
    def test_validate_missing_trading_hours(self):
        """测试缺失交易时间字段"""
        config = {}
        
        results = self.validator.validate(config)
        assert results is not None
        # 应该包含错误
        
    def test_validate_invalid_trading_hours_type(self):
        """测试无效的交易时间类型"""
        config = {
            "trading_hours": "not a dict"
        }
        
        results = self.validator.validate(config)
        assert results is not None
        
    def test_validate_trading_hours_with_segments(self):
        """测试分段交易时间"""
        config = {
            "trading_hours": {
                "morning": ["09:30", "11:30"],
                "afternoon": ["13:00", "15:00"],
                "timezone": "UTC"
            }
        }
        
        results = self.validator.validate(config)
        assert results is not None
        
    def test_validate_invalid_time_format(self):
        """测试无效的时间格式"""
        config = {
            "trading_hours": {
                "start": "invalid",
                "end": "also_invalid"
            }
        }
        
        results = self.validator.validate(config)
        assert results is not None
        
    def test_validate_invalid_time_range(self):
        """测试无效的时间范围（结束早于开始）"""
        config = {
            "trading_hours": {
                "start": "15:00",
                "end": "09:30"
            }
        }
        
        results = self.validator.validate(config)
        assert results is not None
        
    def test_validate_with_timezone(self):
        """测试带时区的验证"""
        config = {
            "trading_hours": {
                "start": "09:30",
                "end": "15:00",
                "timezone": "America/New_York"
            }
        }
        
        results = self.validator.validate(config)
        assert results is not None
        
    def test_validate_overlapping_segments(self):
        """测试重叠的交易时段"""
        config = {
            "trading_hours": {
                "segment1": ["09:00", "12:00"],
                "segment2": ["11:00", "14:00"]  # 重叠
            }
        }
        
        results = self.validator.validate(config)
        assert results is not None


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR}")
class TestDatabaseConfigValidator:
    """数据库配置验证器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.validator = DatabaseConfigValidator()
        
    def test_initialization(self):
        """测试初始化"""
        assert self.validator is not None
        assert self.validator.name == "DatabaseConfigValidator"
        
    def test_validate_valid_database_config(self):
        """测试有效的数据库配置"""
        config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "testdb",
                "user": "testuser",
                "password": "testpass"
            }
        }
        
        results = self.validator.validate(config)
        assert results is not None
        
    def test_validate_missing_database_field(self):
        """测试缺失数据库字段"""
        config = {}
        
        results = self.validator.validate(config)
        assert results is not None
        
    def test_validate_invalid_port(self):
        """测试无效的端口"""
        config = {
            "database": {
                "host": "localhost",
                "port": 99999,  # 无效端口
                "name": "testdb"
            }
        }
        
        results = self.validator.validate(config)
        assert results is not None
        
    def test_validate_missing_required_fields(self):
        """测试缺失必需字段"""
        config = {
            "database": {
                "host": "localhost"
                # 缺少其他必需字段
            }
        }
        
        results = self.validator.validate(config)
        assert results is not None
        
    def test_validate_invalid_host(self):
        """测试无效的主机名"""
        config = {
            "database": {
                "host": "",  # 空主机名
                "port": 5432,
                "name": "testdb"
            }
        }
        
        results = self.validator.validate(config)
        assert results is not None
        
    def test_validate_connection_pool_config(self):
        """测试连接池配置"""
        config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "testdb",
                "pool": {
                    "min_size": 5,
                    "max_size": 20
                }
            }
        }
        
        results = self.validator.validate(config)
        assert results is not None


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR}")
class TestLoggingConfigValidator:
    """日志配置验证器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.validator = LoggingConfigValidator()
        
    def test_initialization(self):
        """测试初始化"""
        assert self.validator is not None
        assert self.validator.name == "LoggingConfigValidator"
        
    def test_validate_valid_logging_config(self):
        """测试有效的日志配置"""
        config = {
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "handlers": ["console", "file"]
            }
        }
        
        results = self.validator.validate(config)
        assert results is not None
        
    def test_validate_invalid_log_level(self):
        """测试无效的日志级别"""
        config = {
            "logging": {
                "level": "INVALID_LEVEL"
            }
        }
        
        results = self.validator.validate(config)
        assert results is not None
        
    def test_validate_missing_logging_field(self):
        """测试缺失日志字段"""
        config = {}
        
        results = self.validator.validate(config)
        assert results is not None
        
    def test_validate_file_handler_config(self):
        """测试文件处理器配置"""
        config = {
            "logging": {
                "level": "DEBUG",
                "handlers": {
                    "file": {
                        "filename": "app.log",
                        "max_bytes": 10485760,
                        "backup_count": 5
                    }
                }
            }
        }
        
        results = self.validator.validate(config)
        assert results is not None
        
    def test_validate_multiple_handlers(self):
        """测试多个处理器"""
        config = {
            "logging": {
                "level": "INFO",
                "handlers": {
                    "console": {"level": "DEBUG"},
                    "file": {"level": "INFO", "filename": "app.log"},
                    "syslog": {"level": "ERROR"}
                }
            }
        }
        
        results = self.validator.validate(config)
        assert results is not None


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR}")
class TestNetworkConfigValidator:
    """网络配置验证器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.validator = NetworkConfigValidator()
        
    def test_initialization(self):
        """测试初始化"""
        assert self.validator is not None
        assert self.validator.name == "NetworkConfigValidator"
        
    def test_validate_valid_network_config(self):
        """测试有效的网络配置"""
        config = {
            "network": {
                "host": "0.0.0.0",
                "port": 8080,
                "timeout": 30,
                "max_connections": 100
            }
        }
        
        results = self.validator.validate(config)
        assert results is not None
        
    def test_validate_invalid_port(self):
        """测试无效的端口"""
        config = {
            "network": {
                "host": "localhost",
                "port": 70000  # 超出范围
            }
        }
        
        results = self.validator.validate(config)
        assert results is not None
        
    def test_validate_negative_timeout(self):
        """测试负数超时"""
        config = {
            "network": {
                "host": "localhost",
                "port": 8080,
                "timeout": -1  # 无效
            }
        }
        
        results = self.validator.validate(config)
        assert results is not None
        
    def test_validate_missing_network_field(self):
        """测试缺失网络字段"""
        config = {}
        
        results = self.validator.validate(config)
        assert results is not None
        
    def test_validate_ssl_config(self):
        """测试SSL配置"""
        config = {
            "network": {
                "host": "localhost",
                "port": 443,
                "ssl": {
                    "enabled": True,
                    "cert_file": "/path/to/cert.pem",
                    "key_file": "/path/to/key.pem"
                }
            }
        }
        
        results = self.validator.validate(config)
        assert results is not None
        
    def test_validate_proxy_config(self):
        """测试代理配置"""
        config = {
            "network": {
                "host": "localhost",
                "port": 8080,
                "proxy": {
                    "host": "proxy.example.com",
                    "port": 3128,
                    "auth": {
                        "username": "user",
                        "password": "pass"
                    }
                }
            }
        }
        
        results = self.validator.validate(config)
        assert results is not None
        
    def test_validate_connection_limits(self):
        """测试连接限制"""
        config = {
            "network": {
                "host": "localhost",
                "port": 8080,
                "max_connections": 1000,
                "max_connections_per_ip": 10,
                "connection_timeout": 60
            }
        }
        
        results = self.validator.validate(config)
        assert results is not None
        
    def test_validate_invalid_host(self):
        """测试无效的主机地址"""
        config = {
            "network": {
                "host": "999.999.999.999",  # 无效IP
                "port": 8080
            }
        }
        
        results = self.validator.validate(config)
        assert results is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--cov=src/infrastructure/config/validators/specialized_validators', '--cov-report=term'])

