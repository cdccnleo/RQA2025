"""
基础设施层 - 异常类单元测试

测试统一异常体系的核心功能，包括异常定义、错误码、异常层次结构等。
覆盖率目标: 85%+
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
from unittest.mock import Mock, patch, MagicMock

from src.infrastructure.error.exceptions.unified_exceptions import (
    InfrastructureError,
    NetworkError,
    DatabaseError,
    CacheError,
    ConfigurationError,
    SecurityError,
    SystemError,
    DataLoaderError,
    ErrorCode
)


class TestUnifiedExceptions(unittest.TestCase):
    """统一异常体系单元测试"""

    def test_error_code_enum_values(self):
        """测试错误码枚举值"""
        # 验证错误码定义完整性
        error_codes = [
            ErrorCode.NETWORK_CONNECTION_FAILED,
            ErrorCode.NETWORK_TIMEOUT,
            ErrorCode.DATABASE_CONNECTION_FAILED,
            ErrorCode.DATABASE_QUERY_FAILED,
            ErrorCode.CACHE_CONNECTION_FAILED,
            ErrorCode.CACHE_OPERATION_FAILED,
            ErrorCode.CONFIG_FILE_NOT_FOUND,
            ErrorCode.CONFIG_INVALID_FORMAT,
            ErrorCode.SECURITY_ACCESS_DENIED,
            ErrorCode.SECURITY_INVALID_CREDENTIALS,
            ErrorCode.SYSTEM_RESOURCE_EXHAUSTED,
            ErrorCode.SYSTEM_OPERATION_FAILED,
            ErrorCode.DATALOADER_FILE_NOT_FOUND,
            ErrorCode.DATALOADER_INVALID_FORMAT,
            ErrorCode.UNKNOWN_ERROR
        ]

        # 验证所有错误码都存在
        for error_code in error_codes:
            self.assertIsInstance(error_code, ErrorCode)
            self.assertIsNotNone(error_code.value)

    def test_infrastructure_error_base_class(self):
        """测试基础设施错误基类"""
        error = InfrastructureError("基础错误信息")

        self.assertIsInstance(error, InfrastructureError)
        self.assertIsInstance(error, Exception)
        self.assertEqual(str(error), "基础错误信息")
        self.assertIsNone(error.error_code)
        self.assertIsNone(error.details)
        self.assertIsNone(error.context)

    def test_infrastructure_error_with_details(self):
        """测试带详细信息的InfrastructureError"""
        details = {"component": "database", "operation": "connect"}
        context = {"host": "localhost", "port": 5432}

        error = InfrastructureError(
            "数据库连接失败",
            error_code=ErrorCode.DATABASE_CONNECTION_FAILED,
            details=details,
            context=context
        )

        self.assertEqual(error.error_code, ErrorCode.DATABASE_CONNECTION_FAILED)
        self.assertEqual(error.details, details)
        self.assertEqual(error.context, context)

    def test_network_error(self):
        """测试网络错误"""
        error = NetworkError("网络连接失败", error_code=ErrorCode.NETWORK_CONNECTION_FAILED)

        self.assertIsInstance(error, NetworkError)
        self.assertIsInstance(error, InfrastructureError)
        self.assertEqual(error.error_code, ErrorCode.NETWORK_CONNECTION_FAILED)

        # 测试默认错误码
        default_error = NetworkError("默认网络错误")
        self.assertIsNone(default_error.error_code)

    def test_network_error_subclasses(self):
        """测试网络错误子类"""
        # 测试连接错误
        connection_error = NetworkError.connection_error("连接被拒绝")
        self.assertEqual(connection_error.error_code, ErrorCode.NETWORK_CONNECTION_FAILED)

        # 测试超时错误
        timeout_error = NetworkError.timeout_error("请求超时")
        self.assertEqual(timeout_error.error_code, ErrorCode.NETWORK_TIMEOUT)

        # 测试DNS错误
        dns_error = NetworkError.dns_error("DNS解析失败")
        self.assertIsNotNone(dns_error.error_code)

    def test_database_error(self):
        """测试数据库错误"""
        error = DatabaseError("数据库操作失败", error_code=ErrorCode.DATABASE_QUERY_FAILED)

        self.assertIsInstance(error, DatabaseError)
        self.assertIsInstance(error, InfrastructureError)
        self.assertEqual(error.error_code, ErrorCode.DATABASE_QUERY_FAILED)

    def test_database_error_subclasses(self):
        """测试数据库错误子类"""
        # 测试连接错误
        connection_error = DatabaseError.connection_error("无法连接到数据库")
        self.assertEqual(connection_error.error_code, ErrorCode.DATABASE_CONNECTION_FAILED)

        # 测试查询错误
        query_error = DatabaseError.query_error("SQL语法错误")
        self.assertEqual(query_error.error_code, ErrorCode.DATABASE_QUERY_FAILED)

        # 测试事务错误
        transaction_error = DatabaseError.transaction_error("事务回滚")
        self.assertIsNotNone(transaction_error.error_code)

    def test_cache_error(self):
        """测试缓存错误"""
        error = CacheError("缓存操作失败", error_code=ErrorCode.CACHE_OPERATION_FAILED)

        self.assertIsInstance(error, CacheError)
        self.assertIsInstance(error, InfrastructureError)
        self.assertEqual(error.error_code, ErrorCode.CACHE_OPERATION_FAILED)

    def test_cache_error_subclasses(self):
        """测试缓存错误子类"""
        # 测试连接错误
        connection_error = CacheError.connection_error("无法连接到缓存服务器")
        self.assertEqual(connection_error.error_code, ErrorCode.CACHE_CONNECTION_FAILED)

        # 测试操作错误
        operation_error = CacheError.operation_error("缓存写入失败")
        self.assertEqual(operation_error.error_code, ErrorCode.CACHE_OPERATION_FAILED)

        # 测试序列化错误
        serialization_error = CacheError.serialization_error("对象序列化失败")
        self.assertIsNotNone(serialization_error.error_code)

    def test_configuration_error(self):
        """测试配置错误"""
        error = ConfigurationError("配置加载失败", error_code=ErrorCode.CONFIG_FILE_NOT_FOUND)

        self.assertIsInstance(error, ConfigurationError)
        self.assertIsInstance(error, InfrastructureError)
        self.assertEqual(error.error_code, ErrorCode.CONFIG_FILE_NOT_FOUND)

    def test_configuration_error_subclasses(self):
        """测试配置错误子类"""
        # 测试文件未找到错误
        file_error = ConfigurationError.file_not_found("配置文件不存在")
        self.assertEqual(file_error.error_code, ErrorCode.CONFIG_FILE_NOT_FOUND)

        # 测试格式错误
        format_error = ConfigurationError.invalid_format("配置文件格式错误")
        self.assertEqual(format_error.error_code, ErrorCode.CONFIG_INVALID_FORMAT)

        # 测试缺失配置错误
        missing_error = ConfigurationError.missing_config("缺少必要配置项")
        self.assertIsNotNone(missing_error.error_code)

    def test_security_error(self):
        """测试安全错误"""
        error = SecurityError("安全验证失败", error_code=ErrorCode.SECURITY_ACCESS_DENIED)

        self.assertIsInstance(error, SecurityError)
        self.assertIsInstance(error, InfrastructureError)
        self.assertEqual(error.error_code, ErrorCode.SECURITY_ACCESS_DENIED)

    def test_security_error_subclasses(self):
        """测试安全错误子类"""
        # 测试访问拒绝错误
        access_error = SecurityError.access_denied("访问被拒绝")
        self.assertEqual(access_error.error_code, ErrorCode.SECURITY_ACCESS_DENIED)

        # 测试凭据错误
        credential_error = SecurityError.invalid_credentials("用户名或密码错误")
        self.assertEqual(credential_error.error_code, ErrorCode.SECURITY_INVALID_CREDENTIALS)

        # 测试权限错误
        permission_error = SecurityError.insufficient_permissions("权限不足")
        self.assertIsNotNone(permission_error.error_code)

    def test_system_error(self):
        """测试系统错误"""
        error = SystemError("系统资源不足", error_code=ErrorCode.SYSTEM_RESOURCE_EXHAUSTED)

        self.assertIsInstance(error, SystemError)
        self.assertIsInstance(error, InfrastructureError)
        self.assertEqual(error.error_code, ErrorCode.SYSTEM_RESOURCE_EXHAUSTED)

    def test_system_error_subclasses(self):
        """测试系统错误子类"""
        # 测试资源耗尽错误
        resource_error = SystemError.resource_exhausted("内存不足")
        self.assertEqual(resource_error.error_code, ErrorCode.SYSTEM_RESOURCE_EXHAUSTED)

        # 测试操作失败错误
        operation_error = SystemError.operation_failed("系统调用失败")
        self.assertEqual(operation_error.error_code, ErrorCode.SYSTEM_OPERATION_FAILED)

        # 测试硬件错误
        hardware_error = SystemError.hardware_failure("硬件故障")
        self.assertIsNotNone(hardware_error.error_code)

    def test_data_loader_error(self):
        """测试数据加载错误"""
        error = DataLoaderError("数据加载失败", error_code=ErrorCode.DATALOADER_FILE_NOT_FOUND)

        self.assertIsInstance(error, DataLoaderError)
        self.assertIsInstance(error, InfrastructureError)
        self.assertEqual(error.error_code, ErrorCode.DATALOADER_FILE_NOT_FOUND)

    def test_data_loader_error_subclasses(self):
        """测试数据加载错误子类"""
        # 测试文件未找到错误
        file_error = DataLoaderError.file_not_found("数据文件不存在")
        self.assertEqual(file_error.error_code, ErrorCode.DATALOADER_FILE_NOT_FOUND)

        # 测试格式错误
        format_error = DataLoaderError.invalid_format("数据格式错误")
        self.assertEqual(format_error.error_code, ErrorCode.DATALOADER_INVALID_FORMAT)

        # 测试编码错误
        encoding_error = DataLoaderError.encoding_error("文件编码错误")
        self.assertIsNotNone(encoding_error.error_code)

    def test_error_hierarchy(self):
        """测试异常层次结构"""
        # 验证继承关系
        network_error = NetworkError("测试")
        self.assertIsInstance(network_error, InfrastructureError)
        self.assertIsInstance(network_error, Exception)

        database_error = DatabaseError("测试")
        self.assertIsInstance(database_error, InfrastructureError)
        self.assertIsInstance(database_error, Exception)

        # 验证不同类型错误不相互继承
        self.assertNotIsInstance(network_error, DatabaseError)
        self.assertNotIsInstance(database_error, NetworkError)

    def test_error_details_and_context(self):
        """测试错误详细信息和上下文"""
        details = {
            "component": "database",
            "operation": "query",
            "query": "SELECT * FROM users"
        }
        context = {
            "user_id": 123,
            "session_id": "abc123",
            "timestamp": "2024-01-01T10:00:00Z"
        }

        error = DatabaseError(
            "查询执行失败",
            error_code=ErrorCode.DATABASE_QUERY_FAILED,
            details=details,
            context=context
        )

        self.assertEqual(error.details, details)
        self.assertEqual(error.context, context)
        self.assertEqual(error.details["component"], "database")
        self.assertEqual(error.context["user_id"], 123)

    def test_error_string_representation(self):
        """测试错误字符串表示"""
        error = InfrastructureError("简单错误信息")
        self.assertEqual(str(error), "简单错误信息")

        # 带错误码的错误
        error_with_code = NetworkError("网络错误", error_code=ErrorCode.NETWORK_CONNECTION_FAILED)
        error_str = str(error_with_code)
        self.assertIn("网络错误", error_str)
        # 注意：错误码的字符串表示可能因实现而异

    def test_error_code_uniqueness(self):
        """测试错误码唯一性"""
        error_codes = [code for code in ErrorCode]
        unique_codes = set(code.value for code in error_codes)

        # 验证所有错误码值都是唯一的
        self.assertEqual(len(error_codes), len(unique_codes))

    def test_error_code_categories(self):
        """测试错误码分类"""
        # 网络错误码
        network_codes = [
            ErrorCode.NETWORK_CONNECTION_FAILED,
            ErrorCode.NETWORK_TIMEOUT,
        ]

        # 数据库错误码
        database_codes = [
            ErrorCode.DATABASE_CONNECTION_FAILED,
            ErrorCode.DATABASE_QUERY_FAILED,
        ]

        # 验证不同类别的错误码值不同
        all_codes = network_codes + database_codes
        code_values = [code.value for code in all_codes]
        unique_values = set(code_values)

        self.assertEqual(len(all_codes), len(unique_values))

    def test_error_factory_methods(self):
        """测试错误工厂方法"""
        # 测试各种工厂方法的返回类型
        errors = [
            NetworkError.connection_error("连接错误"),
            DatabaseError.query_error("查询错误"),
            CacheError.operation_error("缓存错误"),
            ConfigurationError.file_not_found("配置错误"),
            SecurityError.access_denied("安全错误"),
            SystemError.resource_exhausted("系统错误"),
            DataLoaderError.invalid_format("数据错误")
        ]

        for error in errors:
            self.assertIsInstance(error, InfrastructureError)
            self.assertIsNotNone(error.error_code)

    def test_error_context_preservation(self):
        """测试错误上下文保持"""
        original_context = {
            "request_id": "req-123",
            "user_agent": "test-client/1.0",
            "endpoint": "/api/data"
        }

        error = InfrastructureError(
            "处理请求失败",
            context=original_context
        )

        # 验证上下文完整保持
        self.assertEqual(error.context, original_context)
        self.assertEqual(error.context["request_id"], "req-123")
        self.assertEqual(error.context["user_agent"], "test-client/1.0")
        self.assertEqual(error.context["endpoint"], "/api/data")

    def test_error_details_immutability(self):
        """测试错误详情的不可变性"""
        details = {"key": "value", "list": [1, 2, 3]}
        error = InfrastructureError("测试错误", details=details)

        # 修改原始字典
        details["new_key"] = "new_value"
        details["list"].append(4)

        # 验证错误对象的details不受影响
        self.assertNotIn("new_key", error.details)
        self.assertEqual(len(error.details["list"]), 3)
        self.assertEqual(error.details["list"], [1, 2, 3])

    def test_error_inheritance_consistency(self):
        """测试错误继承的一致性"""
        # 创建各种类型的错误
        errors = [
            InfrastructureError("基础错误"),
            NetworkError("网络错误"),
            DatabaseError("数据库错误"),
            CacheError("缓存错误"),
            ConfigurationError("配置错误"),
            SecurityError("安全错误"),
            SystemError("系统错误"),
            DataLoaderError("数据加载错误")
        ]

        # 验证所有错误都是InfrastructureError的实例
        for error in errors:
            self.assertIsInstance(error, InfrastructureError)

        # 验证具体类型错误有正确的类型
        self.assertIsInstance(errors[1], NetworkError)
        self.assertIsInstance(errors[2], DatabaseError)
        self.assertIsInstance(errors[3], CacheError)
        self.assertIsInstance(errors[4], ConfigurationError)
        self.assertIsInstance(errors[5], SecurityError)
        self.assertIsInstance(errors[6], SystemError)
        self.assertIsInstance(errors[7], DataLoaderError)

    def test_error_code_enum_completeness(self):
        """测试错误码枚举的完整性"""
        # 确保所有预期的错误类型都有对应的错误码
        required_categories = {
            'NETWORK': ['CONNECTION_FAILED', 'TIMEOUT'],
            'DATABASE': ['CONNECTION_FAILED', 'QUERY_FAILED'],
            'CACHE': ['CONNECTION_FAILED', 'OPERATION_FAILED'],
            'CONFIG': ['FILE_NOT_FOUND', 'INVALID_FORMAT'],
            'SECURITY': ['ACCESS_DENIED', 'INVALID_CREDENTIALS'],
            'SYSTEM': ['RESOURCE_EXHAUSTED', 'OPERATION_FAILED'],
            'DATALOADER': ['FILE_NOT_FOUND', 'INVALID_FORMAT'],
            'UNKNOWN': ['ERROR']
        }

        code_names = [code.name for code in ErrorCode]

        for category, error_types in required_categories.items():
            for error_type in error_types:
                expected_name = f"{category}_{error_type}"
                self.assertIn(expected_name, code_names,
                            f"缺少错误码: {expected_name}")

    def test_error_with_none_values(self):
        """测试包含None值的错误"""
        error = InfrastructureError(
            "测试错误",
            error_code=None,
            details=None,
            context=None
        )

        self.assertIsNone(error.error_code)
        self.assertIsNone(error.details)
        self.assertIsNone(error.context)
        self.assertEqual(str(error), "测试错误")

    def test_error_chaining(self):
        """测试错误链"""
        try:
            try:
                raise ValueError("原始错误")
            except ValueError as e:
                raise NetworkError("网络层错误") from e
        except NetworkError as network_error:
            # 验证错误链信息
            self.assertIsInstance(network_error, NetworkError)
            self.assertIsNotNone(network_error.__cause__)
            self.assertIsInstance(network_error.__cause__, ValueError)


if __name__ == '__main__':
    unittest.main()
