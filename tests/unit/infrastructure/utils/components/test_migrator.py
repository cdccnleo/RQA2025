#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层数据库迁移器组件测试

测试目标：提升utils/components/migrator.py的真实覆盖率
实际导入和使用src.infrastructure.utils.components.migrator模块
"""

import pytest
from unittest.mock import MagicMock, Mock
from src.infrastructure.utils.core.interfaces import QueryResult


class TestMigrationConstants:
    """测试迁移器常量"""
    
    def test_constants(self):
        """测试常量值"""
        from src.infrastructure.utils.components.migrator import MigrationConstants
        
        assert MigrationConstants.DEFAULT_BATCH_SIZE == 1000
        assert MigrationConstants.DEFAULT_RETRY_COUNT == 3
        assert MigrationConstants.DEFAULT_RETRY_DELAY == 5
        assert MigrationConstants.DEFAULT_MIGRATED_COUNT == 0
        assert MigrationConstants.DEFAULT_FAILED_COUNT == 0
        assert MigrationConstants.EMPTY_DATA_LENGTH == 0
        assert MigrationConstants.FINAL_RETRY_ATTEMPT == 1
        assert MigrationConstants.SAMPLE_SIZE == 10
        assert MigrationConstants.PROGRESS_UPDATE_INTERVAL == 1.0
        assert MigrationConstants.MAX_WORKERS == 4
        assert MigrationConstants.LOG_FILE_SUFFIX == ".log"
        assert MigrationConstants.BACKUP_SUFFIX == ".backup"


class TestDatabaseMigrator:
    """测试数据库迁移器"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.components.migrator import DatabaseMigrator
        from src.infrastructure.utils.core.interfaces import IDatabaseAdapter
        
        class MockAdapter(IDatabaseAdapter):
            def connect(self):
                pass
            
            def disconnect(self):
                pass
            
            def execute_query(self, query: str):
                return QueryResult(success=True, data=[], row_count=0, execution_time=0.0)
            
            def execute_write(self, query: str, data: list):
                return QueryResult(success=True, data=[], row_count=0, execution_time=0.0)
            
            def batch_write(self, data: list):
                return QueryResult(success=True, data=[], row_count=0, execution_time=0.0)
            
            def connection_status(self):
                return "connected"
            
            def get_connection_info(self):
                return {}
            
            def health_check(self):
                return True
        
        source_adapter = MockAdapter()
        target_adapter = MockAdapter()
        
        migrator = DatabaseMigrator(source_adapter, target_adapter)
        
        assert migrator.source_adapter == source_adapter
        assert migrator.target_adapter == target_adapter
        assert migrator.batch_size == 1000
        assert migrator.retry_count == 3
        assert migrator.retry_delay == 5
    
    def test_build_batch_query(self):
        """测试构建批查询"""
        from src.infrastructure.utils.components.migrator import DatabaseMigrator
        from src.infrastructure.utils.core.interfaces import IDatabaseAdapter
        
        class MockAdapter(IDatabaseAdapter):
            def connect(self):
                pass
            
            def disconnect(self):
                pass
            
            def execute_query(self, query: str):
                return QueryResult(success=True, data=[], row_count=0, execution_time=0.0)
            
            def execute_write(self, query: str, data: list):
                return QueryResult(success=True, data=[], row_count=0, execution_time=0.0)
            
            def batch_write(self, data: list):
                return QueryResult(success=True, data=[], row_count=0, execution_time=0.0)
            
            def connection_status(self):
                return "connected"
            
            def get_connection_info(self):
                return {}
            
            def health_check(self):
                return True
        
        source_adapter = MockAdapter()
        target_adapter = MockAdapter()
        migrator = DatabaseMigrator(source_adapter, target_adapter)
        
        query = migrator._build_batch_query("test_table", "", 0)
        assert "SELECT * FROM test_table" in query
        assert "LIMIT 1000" in query
        assert "OFFSET 0" in query
    
    def test_build_batch_query_with_condition(self):
        """测试构建带条件的批查询"""
        from src.infrastructure.utils.components.migrator import DatabaseMigrator
        from src.infrastructure.utils.core.interfaces import IDatabaseAdapter
        
        class MockAdapter(IDatabaseAdapter):
            def connect(self):
                pass
            
            def disconnect(self):
                pass
            
            def execute_query(self, query: str):
                return QueryResult(success=True, data=[], row_count=0, execution_time=0.0)
            
            def execute_write(self, query: str, data: list):
                return QueryResult(success=True, data=[], row_count=0, execution_time=0.0)
            
            def batch_write(self, data: list):
                return QueryResult(success=True, data=[], row_count=0, execution_time=0.0)
            
            def connection_status(self):
                return "connected"
            
            def get_connection_info(self):
                return {}
            
            def health_check(self):
                return True
        
        source_adapter = MockAdapter()
        target_adapter = MockAdapter()
        migrator = DatabaseMigrator(source_adapter, target_adapter)
        
        query = migrator._build_batch_query("test_table", "id > 100", 100)
        assert "SELECT * FROM test_table" in query
        assert "WHERE id > 100" in query
        assert "LIMIT 1000" in query
        assert "OFFSET 100" in query
    
    def test_initialize_migration(self):
        """测试初始化迁移状态"""
        from src.infrastructure.utils.components.migrator import DatabaseMigrator
        from src.infrastructure.utils.core.interfaces import IDatabaseAdapter
        
        class MockAdapter(IDatabaseAdapter):
            def __init__(self, count=0):
                self.count = count
            
            def connect(self):
                pass
            
            def disconnect(self):
                pass
            
            def execute_query(self, query: str):
                if "COUNT" in query.upper():
                    return QueryResult(success=True, data=[{"count": self.count}], row_count=1, execution_time=0.0)
                return QueryResult(success=True, data=[], row_count=0, execution_time=0.0)
            
            def execute_write(self, query: str, data: list):
                return QueryResult(success=True, data=[])
            
            def batch_write(self, data: list):
                return QueryResult(success=True, data=[], row_count=0, execution_time=0.0)
            
            def connection_status(self):
                return "connected"
            
            def get_connection_info(self):
                return {}
            
            def health_check(self):
                return True
        
        source_adapter = MockAdapter(count=100)
        target_adapter = MockAdapter()
        migrator = DatabaseMigrator(source_adapter, target_adapter)
        
        state = migrator._initialize_migration("test_table", "")
        assert "total_count" in state
        assert "migrated" in state
        assert "failed" in state
        assert "start_time" in state
        assert state["migrated"] == 0
        assert state["failed"] == 0

