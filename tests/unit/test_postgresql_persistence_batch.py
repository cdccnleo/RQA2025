#!/usr/bin/env python3
"""
PostgreSQL批量持久化单元测试
"""

import asyncio
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.gateway.web.postgresql_persistence_batch import PostgreSQLBatchInserter


class TestPostgreSQLBatchInserter:
    """PostgreSQL批量持久化测试类"""

    def setup_method(self):
        """测试前准备"""
        self.config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'test_db',
            'user': 'test_user',
            'password': 'test_pass',
            'pool_size': 5,
            'max_overflow': 10,
            'batch_size': 1000,
            'max_retries': 3,
            'retry_delay': 1.0
        }

    def teardown_method(self):
        """测试后清理"""
        pass

    @patch('psycopg2.pool.SimpleConnectionPool')
    def test_initialization(self, mock_pool):
        """测试初始化"""
        # 创建模拟连接池
        mock_connection_pool = Mock()
        mock_pool.return_value = mock_connection_pool

        # 创建实例
        inserter = PostgreSQLBatchInserter(self.config)

        # 验证初始化
        assert inserter.config == self.config
        assert inserter.batch_size == self.config['batch_size']
        assert inserter.max_retries == self.config['max_retries']
        assert inserter.connection_pool == mock_connection_pool

        # 验证连接池创建参数
        mock_pool.assert_called_once_with(
            minconn=1,
            maxconn=self.config['pool_size'] + self.config['max_overflow'],
            host=self.config['host'],
            port=self.config['port'],
            database=self.config['database'],
            user=self.config['user'],
            password=self.config['password']
        )

    @patch('psycopg2.pool.SimpleConnectionPool')
    def test_stock_data_preprocessing(self, mock_pool):
        """测试股票数据预处理"""
        mock_pool.return_value = Mock()
        inserter = PostgreSQLBatchInserter(self.config)

        # 测试数据
        test_data = [
            {
                'symbol': '000001.SZ',
                'date': '2023-01-01',
                'open': 10.5,
                'high': 11.0,
                'low': 10.2,
                'close': 10.8,
                'volume': 1000000,
                'amount': 10800000.0,
                'source': 'akshare'
            },
            {
                'symbol': '000002.SZ',
                'date': '2023-01-02',
                'open': 20.5,
                'high': 21.0,
                'low': 20.2,
                'close': 20.8,
                'volume': 2000000,
                'amount': 41600000.0,
                'source': 'akshare'
            }
        ]

        # 调用预处理方法
        processed_data = inserter._preprocess_stock_data(test_data)

        # 验证预处理结果
        assert len(processed_data) == 2
        assert processed_data[0]['symbol'] == '000001.SZ'
        assert processed_data[0]['date'] == '2023-01-01'
        assert processed_data[0]['open'] == 10.5
        assert processed_data[0]['close'] == 10.8
        assert processed_data[1]['symbol'] == '000002.SZ'

    @patch('psycopg2.pool.SimpleConnectionPool')
    def test_data_validation(self, mock_pool):
        """测试数据验证"""
        mock_pool.return_value = Mock()
        inserter = PostgreSQLBatchInserter(self.config)

        # 有效数据
        valid_record = {
            'symbol': '000001.SZ',
            'date': '2023-01-01',
            'open': 10.5,
            'high': 11.0,
            'low': 10.2,
            'close': 10.8,
            'volume': 1000000,
            'amount': 10800000.0
        }

        # 验证有效数据
        is_valid, validated_record = inserter._validate_and_convert_stock_record(valid_record)
        assert is_valid is True
        assert validated_record['symbol'] == '000001.SZ'

        # 无效数据：缺少必需字段
        invalid_record = {
            'symbol': '000001.SZ',
            'date': '2023-01-01'
            # 缺少价格和成交量数据
        }

        is_valid, error_msg = inserter._validate_and_convert_stock_record(invalid_record)
        assert is_valid is False
        assert '缺少必需字段' in error_msg

        # 无效数据：价格为负数
        negative_price_record = valid_record.copy()
        negative_price_record['open'] = -10.5

        is_valid, error_msg = inserter._validate_and_convert_stock_record(negative_price_record)
        assert is_valid is False
        assert '价格不能为负数' in error_msg

    @patch('psycopg2.pool.SimpleConnectionPool')
    @patch('psycopg2.extras.execute_values')
    def test_batch_insert_success(self, mock_execute_values, mock_pool):
        """测试批量插入成功情况"""
        # 模拟连接池和连接
        mock_connection = Mock()
        mock_connection.cursor.return_value.__enter__.return_value = Mock()
        mock_connection.cursor.return_value.__exit__ = Mock(return_value=None)

        mock_connection_pool = Mock()
        mock_connection_pool.getconn.return_value = mock_connection
        mock_connection_pool.putconn = Mock()
        mock_pool.return_value = mock_connection_pool

        # 创建实例
        inserter = PostgreSQLBatchInserter(self.config)

        # 测试数据
        test_data = [
            {
                'symbol': '000001.SZ',
                'date': '2023-01-01',
                'open': 10.5,
                'high': 11.0,
                'low': 10.2,
                'close': 10.8,
                'volume': 1000000,
                'amount': 10800000.0
            }
        ]

        # 执行批量插入
        result = inserter.batch_insert_stock_data(test_data, 'akshare')

        # 验证结果
        assert result['success'] is True
        assert result['inserted_count'] == 1
        assert result['failed_count'] == 0
        assert 'total_time' in result
        assert 'avg_time_per_record' in result

        # 验证execute_values被调用
        mock_execute_values.assert_called_once()

    @patch('psycopg2.pool.SimpleConnectionPool')
    def test_batch_insert_failure_retry(self, mock_pool):
        """测试批量插入失败重试"""
        # 模拟连接池
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=None)
        mock_connection.cursor.return_value = mock_cursor

        # 模拟execute_values失败两次，然后成功
        mock_connection.cursor.return_value.execute.side_effect = [
            Exception("Connection failed"),  # 第一次失败
            Exception("Connection failed"),  # 第二次失败
            None  # 第三次成功
        ]

        mock_connection_pool = Mock()
        mock_connection_pool.getconn.return_value = mock_connection
        mock_connection_pool.putconn = Mock()
        mock_pool.return_value = mock_connection_pool

        inserter = PostgreSQLBatchInserter(self.config)

        test_data = [{'symbol': '000001.SZ', 'date': '2023-01-01', 'open': 10.5, 'close': 10.8}]

        with patch('psycopg2.extras.execute_values') as mock_execute_values:
            mock_execute_values.side_effect = [
                Exception("Insert failed"),
                Exception("Insert failed"),
                None  # 第三次成功
            ]

            result = inserter.batch_insert_stock_data(test_data, 'akshare')

            # 验证重试了3次
            assert mock_execute_values.call_count == 3
            assert result['success'] is True
            assert result['retry_count'] == 2

    @patch('psycopg2.pool.SimpleConnectionPool')
    def test_performance_statistics(self, mock_pool):
        """测试性能统计"""
        mock_connection_pool = Mock()
        mock_pool.return_value = mock_connection_pool

        inserter = PostgreSQLBatchInserter(self.config)

        # 模拟性能数据
        inserter.performance_stats = {
            'total_operations': 10,
            'successful_operations': 9,
            'failed_operations': 1,
            'total_records_processed': 5000,
            'total_processing_time': 25.0,
            'average_batch_size': 500
        }

        stats = inserter.get_performance_statistics()

        assert stats['total_operations'] == 10
        assert stats['success_rate'] == 0.9
        assert stats['failure_rate'] == 0.1
        assert stats['avg_processing_time'] == 2.5
        assert stats['avg_records_per_operation'] == 500
        assert stats['throughput_records_per_second'] == 200.0

    @patch('psycopg2.pool.SimpleConnectionPool')
    def test_connection_pool_management(self, mock_pool):
        """测试连接池管理"""
        mock_connection_pool = Mock()
        mock_pool.return_value = mock_connection_pool

        inserter = PostgreSQLBatchInserter(self.config)

        # 测试连接获取和释放
        mock_connection = Mock()
        mock_connection_pool.getconn.return_value = mock_connection

        # 模拟使用连接
        with inserter._get_connection() as conn:
            assert conn == mock_connection

        # 验证连接被正确释放
        mock_connection_pool.putconn.assert_called_once_with(mock_connection)

    def test_batch_splitting(self):
        """测试批量分割"""
        config = self.config.copy()
        config['batch_size'] = 3

        with patch('psycopg2.pool.SimpleConnectionPool'):
            inserter = PostgreSQLBatchInserter(config)

            # 创建5条测试数据
            test_data = [{'id': i} for i in range(5)]

            # 分割批次
            batches = inserter._split_into_batches(test_data)

            # 验证分割结果
            assert len(batches) == 2
            assert len(batches[0]) == 3
            assert len(batches[1]) == 2
            assert batches[0][0]['id'] == 0
            assert batches[0][2]['id'] == 2
            assert batches[1][0]['id'] == 3
            assert batches[1][1]['id'] == 4


if __name__ == '__main__':
    pytest.main([__file__])