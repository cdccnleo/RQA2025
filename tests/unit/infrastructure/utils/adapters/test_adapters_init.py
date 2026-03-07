#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层utils/adapters/__init__.py模块测试

测试目标：提升utils/adapters/__init__.py的真实覆盖率
实际导入和使用src.infrastructure.utils.adapters模块
"""

import pytest


class TestAdaptersInit:
    """测试adapters模块初始化"""
    
    def test_database_adapter_import(self):
        """测试DatabaseAdapter导入"""
        from src.infrastructure.utils.adapters import DatabaseAdapter
        
        assert DatabaseAdapter is not None
    
    def test_postgresql_adapter_import(self):
        """测试PostgreSQLAdapter导入"""
        from src.infrastructure.utils.adapters import PostgreSQLAdapter
        
        assert PostgreSQLAdapter is not None
    
    def test_redis_adapter_import(self):
        """测试RedisAdapter导入"""
        from src.infrastructure.utils.adapters import RedisAdapter
        
        assert RedisAdapter is not None
    
    def test_sqlite_adapter_import(self):
        """测试SQLiteAdapter导入"""
        from src.infrastructure.utils.adapters import SQLiteAdapter
        
        assert SQLiteAdapter is not None
    
    def test_influxdb_adapter_import(self):
        """测试InfluxDBAdapter导入"""
        from src.infrastructure.utils.adapters import InfluxDBAdapter
        
        assert InfluxDBAdapter is not None
    
    def test_data_api_import(self):
        """测试DataAPI导入"""
        from src.infrastructure.utils.adapters import DataAPI
        
        assert DataAPI is not None
    
    def test_idatabase_adapter_import(self):
        """测试IDatabaseAdapter接口导入"""
        from src.infrastructure.utils.adapters import IDatabaseAdapter
        
        assert IDatabaseAdapter is not None
    
    def test_query_result_import(self):
        """测试QueryResult导入"""
        from src.infrastructure.utils.adapters import QueryResult
        
        assert QueryResult is not None
    
    def test_write_result_import(self):
        """测试WriteResult导入"""
        from src.infrastructure.utils.adapters import WriteResult
        
        assert WriteResult is not None
    
    def test_health_check_result_import(self):
        """测试HealthCheckResult导入"""
        from src.infrastructure.utils.adapters import HealthCheckResult
        
        assert HealthCheckResult is not None
    
    def test_module_all(self):
        """测试模块__all__导出"""
        from src.infrastructure.utils.adapters import __all__
        
        assert isinstance(__all__, list)
        assert len(__all__) > 0
        assert "DatabaseAdapter" in __all__
        assert "PostgreSQLAdapter" in __all__
        assert "RedisAdapter" in __all__
        assert "SQLiteAdapter" in __all__
        assert "InfluxDBAdapter" in __all__

