"""
数据管理层异常处理
Data Management Layer Exception Handling

定义数据管理相关的异常类和错误处理机制
"""


class DataManagementException(Exception):
    """数据管理基础异常类"""

    def __init__(self, message: str, error_code: int = -1):
        super().__init__(message)
        self.error_code = error_code
        self.message = message


class DataLoadError(DataManagementException):
    """数据加载异常"""

    def __init__(self, message: str, source_name: str = None):
        super().__init__(f"数据加载失败 - {source_name}: {message}")
        self.source_name = source_name


class DataValidationError(DataManagementException):
    """数据验证异常"""

    def __init__(self, message: str, field_name: str = None):
        super().__init__(f"数据验证失败 - {field_name}: {message}")
        self.field_name = field_name


class DataTransformationError(DataManagementException):
    """数据转换异常"""

    def __init__(self, message: str, transform_type: str = None):
        super().__init__(f"数据转换失败 - {transform_type}: {message}")
        self.transform_type = transform_type


class DatabaseError(DataManagementException):
    """数据库异常"""

    def __init__(self, message: str, table_name: str = None):
        super().__init__(f"数据库操作失败 - {table_name}: {message}")
        self.table_name = table_name


class CacheError(DataManagementException):
    """缓存异常"""

    def __init__(self, message: str, cache_key: str = None):
        super().__init__(f"缓存操作失败 - {cache_key}: {message}")
        self.cache_key = cache_key


class FileOperationError(DataManagementException):
    """文件操作异常"""

    def __init__(self, message: str, file_path: str = None):
        super().__init__(f"文件操作失败 - {file_path}: {message}")
        self.file_path = file_path


class DataQualityError(DataManagementException):
    """数据质量异常"""

    def __init__(self, message: str, quality_metric: str = None):
        super().__init__(f"数据质量问题 - {quality_metric}: {message}")
        self.quality_metric = quality_metric


class SynchronizationError(DataManagementException):
    """同步异常"""

    def __init__(self, message: str, sync_target: str = None):
        super().__init__(f"数据同步失败 - {sync_target}: {message}")
        self.sync_target = sync_target


class BackupError(DataManagementException):
    """备份异常"""

    def __init__(self, message: str, backup_type: str = None):
        super().__init__(f"数据备份失败 - {backup_type}: {message}")
        self.backup_type = backup_type


class VersionControlError(DataManagementException):
    """版本控制异常"""

    def __init__(self, message: str, version: str = None):
        super().__init__(f"版本控制失败 - {version}: {message}")
        self.version = version


def handle_data_exception(func):
    """
    装饰器：统一处理数据管理异常

    Args:
        func: 被装饰的函数

    Returns:
        包装后的函数
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except DataManagementException:
            # 重新抛出数据管理异常
            raise
        except Exception as e:
            # 将其他异常包装为数据管理异常
            raise DataManagementException(f"意外数据管理错误: {str(e)}") from e

    return wrapper


def validate_data_format(data, expected_format: str = None):
    """
    验证数据格式

    Args:
        data: 数据对象
        expected_format: 期望格式

    Raises:
        DataValidationError: 数据格式验证失败
    """
    if data is None:
        raise DataValidationError("数据不能为空")

    if expected_format == "dataframe" and not hasattr(data, 'columns'):
        raise DataValidationError("期望DataFrame格式数据")

    if expected_format == "series" and not hasattr(data, 'index'):
        raise DataValidationError("期望Series格式数据")


def validate_data_quality(data, quality_checks: dict = None):
    """
    验证数据质量

    Args:
        data: 数据对象
        quality_checks: 质量检查配置

    Raises:
        DataQualityError: 数据质量验证失败
    """
    if quality_checks is None:
        quality_checks = {}

    # 检查空值比例
    null_threshold = quality_checks.get('max_null_ratio', 0.1)
    if hasattr(data, 'isnull'):
        null_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if null_ratio > null_threshold:
            raise DataQualityError(f"空值比例过高: {null_ratio:.1%}", "null_ratio")

    # 检查重复数据
    duplicate_threshold = quality_checks.get('max_duplicate_ratio', 0.05)
    if hasattr(data, 'duplicated'):
        duplicate_ratio = data.duplicated().sum() / len(data)
        if duplicate_ratio > duplicate_threshold:
            raise DataQualityError(f"重复数据比例过高: {duplicate_ratio:.1%}", "duplicate_ratio")


def validate_database_connection(connection_config: dict):
    """
    验证数据库连接配置

    Args:
        connection_config: 连接配置字典

    Raises:
        DatabaseError: 数据库连接验证失败
    """
    required_fields = ['host', 'port', 'database']
    missing_fields = [field for field in required_fields if field not in connection_config]

    if missing_fields:
        raise DatabaseError(f"数据库连接配置缺少必需字段: {missing_fields}")

    # 验证端口范围
    port = connection_config.get('port')
    if not isinstance(port, int) or port < 1 or port > 65535:
        raise DatabaseError(f"无效的端口号: {port}")


def validate_file_access(file_path: str, operation: str = "read"):
    """
    验证文件访问权限

    Args:
        file_path: 文件路径
        operation: 操作类型

    Raises:
        FileOperationError: 文件访问验证失败
    """
    import os

    if not file_path:
        raise FileOperationError("文件路径不能为空")

    if operation == "read" and not os.path.exists(file_path):
        raise FileOperationError(f"文件不存在: {file_path}", file_path)

    if operation == "write":
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            raise FileOperationError(f"目录不存在: {directory}", file_path)


def check_data_consistency(primary_data, secondary_data, key_columns: list):
    """
    检查数据一致性

    Args:
        primary_data: 主要数据
        secondary_data: 次要数据
        key_columns: 关键列列表

    Returns:
        一致性检查结果

    Raises:
        DataValidationError: 数据一致性检查失败
    """
    consistency_result = {
        'is_consistent': True,
        'inconsistencies': [],
        'checked_records': 0
    }

    try:
        # 检查记录数量
        if len(primary_data) != len(secondary_data):
            consistency_result['is_consistent'] = False
            consistency_result['inconsistencies'].append(
                f"记录数量不匹配: {len(primary_data)} vs {len(secondary_data)}"
            )

        # 检查关键字段一致性
        for col in key_columns:
            if col in primary_data.columns and col in secondary_data.columns:
                primary_values = set(primary_data[col].dropna())
                secondary_values = set(secondary_data[col].dropna())

                if primary_values != secondary_values:
                    consistency_result['is_consistent'] = False
                    diff_count = len(primary_values.symmetric_difference(secondary_values))
                    consistency_result['inconsistencies'].append(
                        f"字段 {col} 存在 {diff_count} 个不一致的值"
                    )

        consistency_result['checked_records'] = len(primary_data)

    except Exception as e:
        raise DataValidationError(f"数据一致性检查失败: {str(e)}")

    return consistency_result
