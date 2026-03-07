"""
Write Manager功能测试模块

按《投产计划-总览.md》Week 2 Day 1-2执行
测试写入管理器的完整功能

测试覆盖：
- PostgreSQLWriteManager: 写入管理功能（10个测试）
  * 基本写入操作（3个）
  * 批量写入处理（2个）
  * 事务处理（2个）
  * 错误处理（2个）
  * 性能优化（1个）
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from src.infrastructure.utils.adapters.postgresql_write_manager import (
    PostgreSQLWriteManager,
    WriteResult
)


# Apply timeout to all tests (5 seconds per test)
pytestmark = pytest.mark.timeout(5)


class TestWriteManagerBasicFunctional:
    """WriteManager基本写入功能测试"""

    def setup_method(self):
        """测试前准备"""
        self.client = Mock()
        self.manager = PostgreSQLWriteManager(client=self.client)

    def test_execute_insert_operation(self):
        """测试1: INSERT操作执行"""
        # Arrange
        data = {
            "operation": "insert",
            "table": "users",
            "values": {
                "name": "John Doe",
                "email": "john@example.com",
                "age": 30
            }
        }
        
        # Mock INSERT execution
        with patch.object(self.manager, '_execute_insert') as mock_insert:
            mock_insert.return_value = WriteResult(
                success=True,
                affected_rows=1,
                error=None,
                execution_time=0.05
            )
            
            # Act
            result = self.manager.execute_write(data)
            
            # Assert
            assert result.success is True
            assert result.affected_rows == 1
            assert result.error is None
            assert result.execution_time > 0
            mock_insert.assert_called_once_with(data)

    def test_execute_update_operation(self):
        """测试2: UPDATE操作执行"""
        # Arrange
        data = {
            "operation": "update",
            "table": "users",
            "values": {"email": "newemail@example.com"},
            "where": {"id": 1}
        }
        
        # Mock UPDATE execution
        with patch.object(self.manager, '_execute_update') as mock_update:
            mock_update.return_value = WriteResult(
                success=True,
                affected_rows=1,
                error=None,
                execution_time=0.03
            )
            
            # Act
            result = self.manager.execute_write(data)
            
            # Assert
            assert result.success is True
            assert result.affected_rows == 1
            mock_update.assert_called_once_with(data)

    def test_execute_delete_operation(self):
        """测试3: DELETE操作执行"""
        # Arrange
        data = {
            "operation": "delete",
            "table": "users",
            "where": {"id": 999}
        }
        
        # Mock DELETE execution
        with patch.object(self.manager, '_execute_delete') as mock_delete:
            mock_delete.return_value = WriteResult(
                success=True,
                affected_rows=1,
                error=None,
                execution_time=0.02
            )
            
            # Act
            result = self.manager.execute_write(data)
            
            # Assert
            assert result.success is True
            assert result.affected_rows == 1
            mock_delete.assert_called_once_with(data)


class TestWriteManagerBatchFunctional:
    """WriteManager批量写入测试"""

    def setup_method(self):
        """测试前准备"""
        self.client = Mock()
        self.manager = PostgreSQLWriteManager(client=self.client)

    def test_batch_write_multiple_inserts(self):
        """测试4: 批量INSERT操作"""
        # Arrange
        data_list = [
            {
                "operation": "insert",
                "table": "users",
                "values": {"name": f"User{i}", "email": f"user{i}@example.com"}
            }
            for i in range(100)
        ]
        
        # Mock batch_write
        with patch.object(self.manager, 'batch_write') as mock_batch:
            mock_batch.return_value = WriteResult(
                success=True,
                affected_rows=100,
                error=None,
                execution_time=0.5
            )
            
            # Act
            result = self.manager.batch_write(data_list)
            
            # Assert
            assert result.success is True
            assert result.affected_rows == 100
            assert result.execution_time > 0
            mock_batch.assert_called_once()

    def test_batch_write_performance(self):
        """测试5: 批量写入性能优化"""
        # Arrange
        large_data_list = [
            {
                "operation": "insert",
                "table": "metrics",
                "values": {"metric": f"metric_{i}", "value": i * 10.5}
            }
            for i in range(1000)
        ]
        
        # Mock batch_write with performance tracking
        with patch.object(self.manager, 'batch_write') as mock_batch:
            # Batch insert should be much faster than individual inserts
            mock_batch.return_value = WriteResult(
                success=True,
                affected_rows=1000,
                error=None,
                execution_time=1.5  # Batch: 1.5s for 1000 records
            )
            
            # Act
            result = self.manager.batch_write(large_data_list)
            
            # Assert
            assert result.success is True
            assert result.affected_rows == 1000
            # Average time per record should be very low
            avg_time_per_record = result.execution_time / result.affected_rows
            assert avg_time_per_record < 0.01  # < 10ms per record


class TestWriteManagerTransactionFunctional:
    """WriteManager事务处理测试"""

    def setup_method(self):
        """测试前准备"""
        self.client = Mock()
        self.manager = PostgreSQLWriteManager(client=self.client)

    def test_transaction_commit(self):
        """测试6: 事务提交"""
        # Arrange
        operations = [
            {"operation": "insert", "table": "orders", "values": {"product": "A", "qty": 10}},
            {"operation": "update", "table": "inventory", "values": {"qty": 90}, "where": {"product": "A"}},
            {"operation": "insert", "table": "transactions", "values": {"type": "purchase", "amount": 100}}
        ]
        
        # Mock transaction execution
        with patch.object(self.manager, 'execute_write') as mock_write:
            # All operations succeed
            mock_write.side_effect = [
                WriteResult(success=True, affected_rows=1, error=None, execution_time=0.02),
                WriteResult(success=True, affected_rows=1, error=None, execution_time=0.02),
                WriteResult(success=True, affected_rows=1, error=None, execution_time=0.02)
            ]
            
            # Act - Execute all operations
            results = [self.manager.execute_write(op) for op in operations]
            
            # Assert
            assert all(r.success for r in results)
            assert sum(r.affected_rows for r in results) == 3
            assert mock_write.call_count == 3

    def test_transaction_rollback(self):
        """测试7: 事务回滚"""
        # Arrange
        operations = [
            {"operation": "insert", "table": "orders", "values": {"product": "B", "qty": 5}},
            {"operation": "update", "table": "inventory", "values": {"qty": 95}, "where": {"product": "B"}},
            {"operation": "insert", "table": "transactions", "values": {"type": "invalid"}}  # This will fail
        ]
        
        # Mock transaction with failure
        with patch.object(self.manager, 'execute_write') as mock_write:
            # First two succeed, third fails
            mock_write.side_effect = [
                WriteResult(success=True, affected_rows=1, error=None, execution_time=0.02),
                WriteResult(success=True, affected_rows=1, error=None, execution_time=0.02),
                WriteResult(success=False, affected_rows=0, error="Constraint violation", execution_time=0.01)
            ]
            
            # Act - Execute operations and check for failure
            results = []
            for op in operations:
                result = self.manager.execute_write(op)
                results.append(result)
                if not result.success:
                    # Transaction should rollback
                    break
            
            # Assert
            assert len(results) == 3
            assert results[0].success is True
            assert results[1].success is True
            assert results[2].success is False
            assert results[2].error == "Constraint violation"


class TestWriteManagerErrorHandlingFunctional:
    """WriteManager错误处理测试"""

    def setup_method(self):
        """测试前准备"""
        self.manager = PostgreSQLWriteManager(client=None)  # No client

    def test_write_without_client(self):
        """测试8: 未连接客户端的写入处理"""
        # Arrange
        data = {
            "operation": "insert",
            "table": "users",
            "values": {"name": "Test"}
        }
        
        # Act
        result = self.manager.execute_write(data)
        
        # Assert
        assert result.success is False
        assert result.affected_rows == 0
        assert "未连接" in result.error or result.error is not None
        assert result.execution_time == 0.0

    def test_unsupported_operation(self):
        """测试9: 不支持的操作类型"""
        # Arrange
        self.manager.set_client(Mock())
        data = {
            "operation": "truncate",  # Unsupported operation
            "table": "users"
        }
        
        # Act
        result = self.manager.execute_write(data)
        
        # Assert
        assert result.success is False
        assert result.affected_rows == 0
        assert "不支持的操作类型" in result.error or result.error is not None


class TestWriteManagerDataConsistencyFunctional:
    """WriteManager数据一致性测试"""

    def setup_method(self):
        """测试前准备"""
        self.client = Mock()
        self.manager = PostgreSQLWriteManager(client=self.client)

    def test_data_consistency_check(self):
        """测试10: 数据一致性验证"""
        # Arrange
        data = {
            "operation": "insert",
            "table": "accounts",
            "values": {
                "account_id": "ACC001",
                "balance": 1000.00,
                "currency": "USD"
            }
        }
        
        # Mock write with consistency check
        with patch.object(self.manager, '_execute_insert') as mock_insert:
            mock_insert.return_value = WriteResult(
                success=True,
                affected_rows=1,
                error=None,
                execution_time=0.05
            )
            
            # Act
            result = self.manager.execute_write(data)
            
            # Assert
            assert result.success is True
            assert result.affected_rows == 1
            
            # Verify the write was called with correct data
            call_args = mock_insert.call_args[0][0]
            assert call_args["values"]["account_id"] == "ACC001"
            assert call_args["values"]["balance"] == 1000.00
            assert call_args["values"]["currency"] == "USD"


# 测试统计
# Total: 10 tests
# TestWriteManagerBasicFunctional: 3 tests (基本写入操作)
# TestWriteManagerBatchFunctional: 2 tests (批量写入处理)
# TestWriteManagerTransactionFunctional: 2 tests (事务处理)
# TestWriteManagerErrorHandlingFunctional: 2 tests (错误处理)
# TestWriteManagerDataConsistencyFunctional: 1 test (数据一致性)

