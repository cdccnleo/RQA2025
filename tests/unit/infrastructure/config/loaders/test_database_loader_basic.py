#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""database_loader基础测试"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import Mock, patch, MagicMock


def test_database_loader_import():
    """测试database_loader可以导入"""
    from src.infrastructure.config.loaders.database_loader import DatabaseLoader
    assert DatabaseLoader is not None


def test_database_loader_supported_databases():
    """测试支持的数据库类型"""
    from src.infrastructure.config.loaders.database_loader import DatabaseLoader
    assert 'postgresql' in DatabaseLoader.SUPPORTED_DATABASES
    assert 'mysql' in DatabaseLoader.SUPPORTED_DATABASES
    assert 'sqlite' in DatabaseLoader.SUPPORTED_DATABASES
    assert 'mongodb' in DatabaseLoader.SUPPORTED_DATABASES
    assert 'redis' in DatabaseLoader.SUPPORTED_DATABASES


def test_database_loader_init_postgresql():
    """测试PostgreSQL loader初始化"""
    from src.infrastructure.config.loaders.database_loader import DatabaseLoader
    params = {'database': 'test_db', 'app': 'test_app'}
    loader = DatabaseLoader(db_type='postgresql', connection_params=params)
    assert loader.db_type == 'postgresql'
    assert loader.db_name == 'test_db'
    assert loader.app_name == 'test_app'


def test_database_loader_init_mysql():
    """测试MySQL loader初始化"""
    from src.infrastructure.config.loaders.database_loader import DatabaseLoader
    params = {'database': 'test_db'}
    loader = DatabaseLoader(db_type='mysql', connection_params=params)
    assert loader.db_type == 'mysql'


def test_database_loader_init_sqlite():
    """测试SQLite loader初始化"""
    from src.infrastructure.config.loaders.database_loader import DatabaseLoader
    params = {'database': ':memory:'}
    loader = DatabaseLoader(db_type='sqlite', connection_params=params)
    assert loader.db_type == 'sqlite'


def test_database_loader_unsupported_type():
    """测试不支持的数据库类型"""
    from src.infrastructure.config.loaders.database_loader import DatabaseLoader
    with pytest.raises(ValueError, match="Unsupported database type"):
        DatabaseLoader(db_type='oracle')


def test_database_loader_format():
    """测试format属性"""
    from src.infrastructure.config.loaders.database_loader import DatabaseLoader
    from src.infrastructure.config.interfaces.unified_interface import ConfigFormat
    loader = DatabaseLoader(db_type='postgresql')
    assert loader.format == ConfigFormat.DATABASE


@patch('psycopg2.connect')
def test_database_loader_connect_postgresql(mock_connect):
    """测试PostgreSQL连接"""
    from src.infrastructure.config.loaders.database_loader import DatabaseLoader
    mock_conn = Mock()
    mock_connect.return_value = mock_conn
    
    params = {
        'host': 'localhost',
        'port': 5432,
        'database': 'test_db',
        'user': 'test_user',
        'password': 'test_pass'
    }
    loader = DatabaseLoader(db_type='postgresql', connection_params=params)
    loader.connect()
    
    mock_connect.assert_called_once()
    assert loader._connection is not None


@patch('pymysql.connect')
def test_database_loader_connect_mysql(mock_connect):
    """测试MySQL连接"""
    from src.infrastructure.config.loaders.database_loader import DatabaseLoader
    mock_conn = Mock()
    mock_connect.return_value = mock_conn
    
    params = {
        'host': 'localhost',
        'database': 'test_db',
        'user': 'test_user',
        'password': 'test_pass'
    }
    loader = DatabaseLoader(db_type='mysql', connection_params=params)
    loader.connect()
    
    assert loader._connection is not None

