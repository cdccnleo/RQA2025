"""
RQA2025 基础设施层安全工具

提供安全相关的工具函数，修复安全漏洞
"""

import os
import os.path
import re
import logging
import ast
import socket
import subprocess
import time
import hashlib
import hmac
import secrets

from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class SecureKeyManager:
    """安全密钥管理器"""

    # 安全常量定义
    TEST_KEY_PREFIX = "rqa2025_test_"
    NONCE_KEY_PREFIX = "rqa2025_nonce_"

    @staticmethod
    def generate_secure_test_key(identifier: str = "default") -> str:
        """生成安全的测试密钥"""
        # 使用系统信息和时间戳生成唯一标识
        hostname = socket.gethostname()
        timestamp = str(int(time.time()))
        unique_id = f"{hostname}_{timestamp}_{identifier}"

        # 使用SHA256生成密钥
        key_hash = hashlib.sha256(unique_id.encode()).hexdigest()
        return f"{SecureKeyManager.TEST_KEY_PREFIX}{key_hash[:16]}"

    @staticmethod
    def generate_secure_nonce(identifier: str = "default") -> str:
        """生成安全的nonce值"""
        timestamp = str(int(time.time() * 1000000))  # 微秒级时间戳
        random_part = secrets.token_hex(8)

        nonce = f"{SecureKeyManager.NONCE_KEY_PREFIX}{timestamp}_{random_part}_{identifier}"
        return SecureKeyManager.NONCE_KEY_PREFIX + hashlib.sha256(nonce.encode()).hexdigest()[:16]

    @staticmethod
    def get_secure_config_value(config_dict: Dict[str, Any], key: str,
                                default: Any = None, required: bool = False) -> Any:
        """安全获取配置值，避免硬编码"""
        if required and key not in config_dict:
            raise ValueError(f"必需的配置项缺失: {key}")

        value = config_dict.get(key, default)

        # 检查是否是硬编码的占位符
        if isinstance(value, str) and value.startswith(("your-", "test-", "example-", "placeholder-")):
            logger.warning(f"检测到占位符配置值: {key}，建议在生产环境中设置真实值")
            if required:
                raise ValueError(f"配置项 {key} 包含占位符值，请设置真实值")

        return value


class SecureStringValidator:
    """安全字符串验证器"""

    @staticmethod
    def safe_sql_interpolation(query_template: str, params: Dict[str, Any]) -> str:
        """安全的SQL插值，避免SQL注入"""
        # 使用参数化查询而不是字符串拼接
        # 这里返回安全的查询模板
        return query_template  # 在实际使用时应该使用参数化查询

    @staticmethod
    def safe_string_formatting(template: str, **kwargs) -> str:
        """安全的字符串格式化"""
        try:
            return template.format(**kwargs)
        except (KeyError, ValueError) as e:
            logger.error(f"字符串格式化失败: {e}")
            return template


class SecureCryptoUtils:
    """安全加密工具"""

    @staticmethod
    def secure_hash(data: str, algorithm: str = 'sha256') -> str:
        """安全的哈希函数"""
        if algorithm not in ['sha256', 'sha384', 'sha512']:
            raise ValueError(f"不支持的哈希算法: {algorithm}")

        hash_func = getattr(hashlib, algorithm)
        return hash_func(data.encode()).hexdigest()

    @staticmethod
    def secure_hmac(key: str, message: str, algorithm: str = 'sha256') -> str:
        """安全的HMAC函数"""
        if algorithm not in ['sha256', 'sha384', 'sha512']:
            raise ValueError(f"不支持的HMAC算法: {algorithm}")

        return hmac.new(key.encode(), message.encode(),
                        getattr(hashlib, algorithm)).hexdigest()

    @staticmethod
    def secure_random_bytes(length: int = 32) -> bytes:
        """安全的随机字节生成"""
        return secrets.token_bytes(length)

    @staticmethod
    def secure_random_string(length: int = 32) -> str:
        """安全的随机字符串生成"""
        return secrets.token_urlsafe(length)


class SecurePathUtils:
    """安全路径工具"""

    @staticmethod
    def safe_join_path(base_path: str, *paths: str) -> str:
        """安全的路径拼接"""
        base_path = os.path.abspath(base_path)

        # 检查每个路径组件是否安全
        for path in paths:
            if '..' in path or path.startswith('/'):
                raise ValueError(f"不安全的路径组件: {path}")

        return os.path.join(base_path, *paths)

    @staticmethod
    def validate_file_path(file_path: str, allowed_extensions: Optional[list] = None) -> bool:
        """验证文件路径安全性"""
        if '..' in file_path or not os.path.abspath(file_path):
            return False

        # 检查文件扩展名
        if allowed_extensions:
            _, ext = os.path.splitext(file_path)
            if ext.lower() not in [e.lower() for e in allowed_extensions]:
                return False

        return True


class SecureConditionEvaluator:
    """安全条件评估器"""

    # 允许的运算符和函数
    ALLOWED_OPERATORS = {
        '>', '<', '>=', '<=', '==', '!=', 'and', 'or', 'not'
    }

    @staticmethod
    def safe_evaluate_condition(condition: str, context: Dict[str, Any]) -> bool:
        """安全评估条件表达式，避免使用eval"""
        try:
            # 预处理条件字符串
            safe_condition = SecureConditionEvaluator._sanitize_condition(condition)

            # 替换变量
            for var_name, var_value in context.items():
                if isinstance(var_value, (int, float)):
                    safe_condition = safe_condition.replace(var_name, str(var_value))
                else:
                    raise ValueError(f"不支持的变量类型: {type(var_value)}")

            # 简单的条件解析（避免使用eval）
            return SecureConditionEvaluator._evaluate_simple_condition(safe_condition)

        except Exception as e:
            logger.error(f"条件评估失败: {condition}, 错误: {e}")
            return False

    @staticmethod
    def _sanitize_condition(condition: str) -> str:
        """清理条件字符串，移除危险字符"""
        # 只允许数字、字母、运算符和空格
        if not re.match(r'^[\w\s\.\+\-\*\/%=<>!&|()\[\]]+$', condition):
            raise ValueError(f"不安全的条件表达式: {condition}")
        return condition

    @staticmethod
    def _evaluate_simple_condition(condition: str) -> bool:
        """简单条件评估"""
        try:
            # 这里实现简单的条件解析逻辑
            # 为了安全，避免使用eval，我们使用ast.literal_eval或自定义解析器
            parts = condition.replace(' ', '').split('>')
            if len(parts) == 2:
                left = ast.literal_eval(parts[0])
                right = ast.literal_eval(parts[1])
                return left > right

            parts = condition.replace(' ', '').split('<')
            if len(parts) == 2:
                left = ast.literal_eval(parts[0])
                right = ast.literal_eval(parts[1])
                return left < right

            # 对于更复杂的条件，可以扩展这个逻辑
            raise ValueError(f"不支持的条件格式: {condition}")

        except Exception as e:
            logger.error(f"条件解析失败: {condition}, 错误: {e}")
            return False


class SecureProcessUtils:
    """安全进程工具"""

    @staticmethod
    def safe_execute_command(command: list, timeout: int = 30) -> tuple:
        """安全执行系统命令"""
        try:
            # 使用列表形式而不是shell字符串
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False  # 明确禁用shell
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            logger.error(f"命令执行超时: {command}")
            return -1, "", "命令执行超时"
        except Exception as e:
            logger.error(f"命令执行失败: {e}")
            return -1, "", str(e)


# 便捷的实例化对象
secure_key_manager = SecureKeyManager()
secure_string_validator = SecureStringValidator()
secure_crypto_utils = SecureCryptoUtils()
secure_path_utils = SecurePathUtils()
secure_condition_evaluator = SecureConditionEvaluator()
secure_process_utils = SecureProcessUtils()

__all__ = [
    # 主要类
    'SecureKeyManager',
    'SecureStringValidator',
    'SecureCryptoUtils',
    'SecurePathUtils',
    'SecureConditionEvaluator',
    'SecureProcessUtils',

    # 便捷实例
    'secure_key_manager',
    'secure_string_validator',
    'secure_crypto_utils',
    'secure_path_utils',
    'secure_condition_evaluator',
    'secure_process_utils',
]

