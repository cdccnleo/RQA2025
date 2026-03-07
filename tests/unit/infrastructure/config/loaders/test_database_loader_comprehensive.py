#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库配置加载器全面测试

提升database_loader.py的测试覆盖率到80%+
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../../'))

try:
    from src.infrastructure.config.loaders.database_loader import DatabaseConfigLoader
    MODULE_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    MODULE_AVAILABLE = False
    IMPORT_ERROR = e


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR}")
class TestDatabaseConfigLoaderComprehensive:
    """数据库配置加载器全面测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.loader = DatabaseConfigLoader()
    
    def test_initialization_default(self):
        """测试默认初始化"""
        loader = DatabaseConfigLoader()
        assert loader is not None
        assert hasattr(loader, 'load')
        
    def test_initialization_with_connection(self):
        """测试带连接参数的初始化"""
        mock_connection = Mock()
        loader = DatabaseConfigLoader(connection=mock_connection)
        assert loader is not None
        
    @patch('src.infrastructure.config.loaders.database_loader.create_engine')
    def test_load_from_postgresql(self, mock_create_engine):
        """测试从PostgreSQL加载配置"""
        # Mock数据库连接和查询结果
        mock_engine = Mock()
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            ('key1', 'value1'),
            ('key2', 'value2')
        ]
        mock_connection.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        # 加载配置
        config = self.loader.load('postgresql://localhost/config')
        
        # 验证结果
        assert config is not None
        assert isinstance(config, dict)
        
    @patch('src.infrastructure.config.loaders.database_loader.create_engine')
    def test_load_from_mysql(self, mock_create_engine):
        """测试从MySQL加载配置"""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            ('database.host', 'localhost'),
            ('database.port', '3306')
        ]
        mock_connection.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        config = self.loader.load('mysql://localhost/config')
        
        assert config is not None
        assert isinstance(config, dict)
        
    @patch('src.infrastructure.config.loaders.database_loader.create_engine')
    def test_load_from_sqlite(self, mock_create_engine):
        """测试从SQLite加载配置"""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            ('app.name', 'TestApp'),
            ('app.version', '1.0.0')
        ]
        mock_connection.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        config = self.loader.load('sqlite:///config.db')
        
        assert config is not None
        assert isinstance(config, dict)
        
    @patch('src.infrastructure.config.loaders.database_loader.create_engine')
    def test_load_with_table_name(self, mock_create_engine):
        """测试指定表名加载配置"""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            ('setting1', 'value1')
        ]
        mock_connection.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        loader = DatabaseConfigLoader(table_name='custom_config')
        config = loader.load('postgresql://localhost/db')
        
        assert config is not None
        
    @patch('src.infrastructure.config.loaders.database_loader.create_engine')
    def test_load_connection_error(self, mock_create_engine):
        """测试连接错误处理"""
        mock_create_engine.side_effect = Exception("Connection failed")
        
        with pytest.raises(Exception, match="Connection failed"):
            self.loader.load('postgresql://invalid/db')
            
    @patch('src.infrastructure.config.loaders.database_loader.create_engine')
    def test_load_query_error(self, mock_create_engine):
        """测试查询错误处理"""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_connection.execute.side_effect = Exception("Query failed")
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        with pytest.raises(Exception):
            self.loader.load('postgresql://localhost/db')
            
    @patch('src.infrastructure.config.loaders.database_loader.create_engine')
    def test_load_empty_result(self, mock_create_engine):
        """测试空结果集"""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.fetchall.return_value = []
        mock_connection.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        config = self.loader.load('postgresql://localhost/db')
        
        assert config is not None
        assert isinstance(config, dict)
        assert len(config) == 0
        
    @patch('src.infrastructure.config.loaders.database_loader.create_engine')
    def test_load_with_nested_keys(self, mock_create_engine):
        """测试嵌套键名处理"""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            ('app.database.host', 'localhost'),
            ('app.database.port', '5432'),
            ('app.cache.enabled', 'true')
        ]
        mock_connection.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        config = self.loader.load('postgresql://localhost/db')
        
        assert config is not None
        assert isinstance(config, dict)
        
    @patch('src.infrastructure.config.loaders.database_loader.create_engine')
    def test_load_with_type_conversion(self, mock_create_engine):
        """测试类型转换"""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            ('int_value', 42),
            ('float_value', 3.14),
            ('bool_value', True),
            ('str_value', 'test')
        ]
        mock_connection.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        config = self.loader.load('postgresql://localhost/db')
        
        assert config is not None
        
    def test_supports_format(self):
        """测试支持的格式"""
        assert self.loader.supports_format('database')
        assert self.loader.supports_format('db')
        assert self.loader.supports_format('sql')
        assert not self.loader.supports_format('json')
        
    @patch('src.infrastructure.config.loaders.database_loader.create_engine')
    def test_load_with_custom_query(self, mock_create_engine):
        """测试自定义查询"""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.fetchall.return_value = [('key', 'value')]
        mock_connection.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        loader = DatabaseConfigLoader(query="SELECT * FROM custom_table")
        config = loader.load('postgresql://localhost/db')
        
        assert config is not None
        
    @patch('src.infrastructure.config.loaders.database_loader.create_engine')
    def test_load_with_timeout(self, mock_create_engine):
        """测试超时设置"""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.fetchall.return_value = []
        mock_connection.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        loader = DatabaseConfigLoader(timeout=10)
        config = loader.load('postgresql://localhost/db')
        
        assert config is not None
        
    @patch('src.infrastructure.config.loaders.database_loader.create_engine')
    def test_load_with_pool_settings(self, mock_create_engine):
        """测试连接池设置"""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.fetchall.return_value = []
        mock_connection.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        loader = DatabaseConfigLoader(pool_size=5, max_overflow=10)
        config = loader.load('postgresql://localhost/db')
        
        assert config is not None
        
    def test_close_connection(self):
        """测试关闭连接"""
        mock_connection = Mock()
        loader = DatabaseConfigLoader(connection=mock_connection)
        
        loader.close()
        
        mock_connection.close.assert_called_once()
        
    @patch('src.infrastructure.config.loaders.database_loader.create_engine')
    def test_load_with_retry(self, mock_create_engine):
        """测试重试机制"""
        mock_engine = Mock()
        
        # 第一次失败，第二次成功
        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Temporary failure")
            mock_connection = Mock()
            mock_result = Mock()
            mock_result.fetchall.return_value = []
            mock_connection.execute.return_value = mock_result
            return mock_connection
            
        mock_engine.connect.return_value.__enter__.side_effect = side_effect
        mock_create_engine.return_value = mock_engine
        
        loader = DatabaseConfigLoader(retry_count=3, retry_delay=0.1)
        config = loader.load('postgresql://localhost/db')
        
        assert config is not None
        
    @patch('src.infrastructure.config.loaders.database_loader.create_engine')
    def test_load_mongodb(self, mock_create_engine):
        """测试MongoDB加载器"""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            ('mongodb.host', 'localhost'),
            ('mongodb.port', '27017')
        ]
        mock_connection.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        config = self.loader.load('mongodb://localhost/config')
        
        assert config is not None


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR}")
class TestDatabaseLoaderEdgeCases:
    """数据库加载器边界情况测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.loader = DatabaseConfigLoader()
        
    @patch('src.infrastructure.config.loaders.database_loader.create_engine')
    def test_load_with_special_characters(self, mock_create_engine):
        """测试包含特殊字符的配置"""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            ('key.with.dots', 'value'),
            ('key-with-dashes', 'value'),
            ('key_with_underscores', 'value')
        ]
        mock_connection.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        config = self.loader.load('postgresql://localhost/db')
        
        assert config is not None
        
    @patch('src.infrastructure.config.loaders.database_loader.create_engine')
    def test_load_with_null_values(self, mock_create_engine):
        """测试NULL值处理"""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            ('key1', None),
            ('key2', 'value2')
        ]
        mock_connection.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        config = self.loader.load('postgresql://localhost/db')
        
        assert config is not None
        
    @patch('src.infrastructure.config.loaders.database_loader.create_engine')
    def test_load_with_large_dataset(self, mock_create_engine):
        """测试大数据集处理"""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_result = Mock()
        # 生成1000条记录
        mock_result.fetchall.return_value = [
            (f'key{i}', f'value{i}') for i in range(1000)
        ]
        mock_connection.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        config = self.loader.load('postgresql://localhost/db')
        
        assert config is not None
        assert len(config) > 0
        
    @patch('src.infrastructure.config.loaders.database_loader.create_engine')
    def test_load_with_unicode(self, mock_create_engine):
        """测试Unicode字符处理"""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            ('中文键', '中文值'),
            ('日本語キー', '日本語の値')
        ]
        mock_connection.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        config = self.loader.load('postgresql://localhost/db')
        
        assert config is not None
        
    def test_invalid_connection_string(self):
        """测试无效的连接字符串"""
        with pytest.raises(Exception):
            self.loader.load('invalid_connection_string')
            
    @patch('src.infrastructure.config.loaders.database_loader.create_engine')
    def test_load_with_authentication(self, mock_create_engine):
        """测试带认证的加载"""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.fetchall.return_value = []
        mock_connection.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        config = self.loader.load('postgresql://user:pass@localhost/db')
        
        assert config is not None
        mock_create_engine.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--cov=src/infrastructure/config/loaders/database_loader', '--cov-report=term'])

