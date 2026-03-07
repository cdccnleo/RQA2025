"""
config_validators 模块

提供 config_validators 相关功能和接口。
"""


from typing import Optional, Tuple
"""
配置验证器 - 提取自UnifiedConfigManager.set方法的验证逻辑

Phase 6.0复杂方法治理: 将UnifiedConfigManager.set的40复杂度降低至<10
"""


class ConfigKeyValidator:
    """配置键验证器 - 负责所有键相关的验证逻辑"""

    # 危险字符集合
    DANGEROUS_CHARS = {'<', '>', ';', ' ', '/', '\\', ':', '@', '#'}

    # 最大键长度
    MAX_KEY_LENGTH = 100

    @staticmethod
    def validate_key(key: str) -> Tuple[bool, Optional[str]]:
        """
        验证配置键

        Args:
            key: 要验证的键

        Returns:
            Tuple[bool, Optional[str]]: (是否有效, 错误信息)
        """
        # 基础类型检查
        if key is None:
            return False, "Key cannot be None"

        if not isinstance(key, str) or not key:
            return False, "Key must be a non-empty string"

        # 长度检查
        if len(key) > ConfigKeyValidator.MAX_KEY_LENGTH:
            return False, f"Key length exceeds maximum {ConfigKeyValidator.MAX_KEY_LENGTH} characters"

        # 格式检查
        if key == "." or key.startswith(".") or key.endswith(".") or ".." in key:
            return False, "Invalid key format: cannot start/end with '.' or contain '..'"

        # 危险字符检查
        if any(char in key for char in ConfigKeyValidator.DANGEROUS_CHARS):
            return False, f"Key contains dangerous characters: {ConfigKeyValidator.DANGEROUS_CHARS}"

        return True, None

    @staticmethod
    def parse_key_structure(key: str) -> Tuple[bool, list, Optional[str]]:
        """
        解析键的结构

        Args:
            key: 要解析的键

        Returns:
            Tuple[bool, list, Optional[str]]: (是否有效, 解析后的部分, 错误信息)
        """
        parts = key.split('.')
        if len(parts) < 1:
            return False, [], "Key must have at least one part"

        return True, parts, None


class ConfigValueValidator:
    """配置值验证器 - 负责值相关的验证逻辑"""

    @staticmethod
    def validate_value(value) -> bool:
        """
        验证配置值的基本有效性

        Args:
            value: 要验证的值

        Returns:
            bool: 值是否有效
        """
        # 这里可以添加更复杂的验证逻辑
        # 目前只做基本检查
        return value is not None




