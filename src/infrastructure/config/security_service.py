"""配置安全服务

提供配置签名验证和敏感操作保护功能
"""
import hmac
import hashlib
from typing import Dict, Any, Optional
from dataclasses import dataclass
from src.infrastructure.config.interfaces.security_service import ISecurityService

@dataclass
class SecurityConfig:
    """安全配置"""
    secret_key: bytes  # HMAC签名密钥
    sensitive_keys: list[str]  # 敏感配置键列表
    require_2fa: bool = True  # 是否要求双因素认证
    audit_level: str = 'standard'  # 审计级别: standard/strict/none
    validation_level: str = 'basic'  # 验证级别: basic/full/none

class SecurityService(ISecurityService):
    """配置安全服务实现

    功能:
    1. 配置签名验证
    2. 敏感操作检查
    3. 双因素认证集成
    """

    def __init__(self, config: Optional[SecurityConfig] = None):
        """初始化安全服务

        Args:
            config: 安全配置，None表示使用默认值
        """
        self._config = config or SecurityConfig(
            secret_key=b'default-secret-key-change-me',
            sensitive_keys=[
                'password',
                'api_keys',
                'encryption_keys',
                'database.password'
            ]
        )

    def verify_signature(self, signed_data: Dict[str, Any]) -> bool:
        """验证配置签名

        Args:
            signed_data: 包含config和signature的字典

        Returns:
            bool: 签名是否有效
        """
        if "config" not in signed_data or "signature" not in signed_data:
            return False
            
        config = signed_data["config"]
        signature = signed_data["signature"]
        
        # 序列化配置为可签名字符串
        serialized = self._serialize_config(config)
        # 计算HMAC-SHA256签名
        expected = hmac.new(
            self._config.secret_key,
            serialized.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        # 安全比较签名
        return hmac.compare_digest(expected, signature)

    def is_sensitive_operation(self, key: str) -> bool:
        """检查是否为敏感操作

        Args:
            key: 配置键

        Returns:
            bool: 是否为敏感操作
        """
        return any(
            key.startswith(sensitive)
            for sensitive in self._config.sensitive_keys
        )

    def require_2fa(self, key: str) -> bool:
        """检查操作是否需要双因素认证

        Args:
            key: 配置键

        Returns:
            bool: 是否需要2FA
        """
        return self._config.require_2fa and self.is_sensitive_operation(key)

    @property
    def audit_level(self) -> str:
        """获取当前审计级别
        
        Returns:
            str: 审计级别 (standard/strict/none)
        """
        return self._config.audit_level

    @property
    def validation_level(self) -> str:
        """获取当前验证级别
        
        Returns:
            str: 验证级别 (basic/full/none)
        """
        return self._config.validation_level

    def sign_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """生成配置签名
        
        Args:
            config: 原始配置
            
        Returns:
            包含签名和配置的字典
        """
        serialized = self._serialize_config(config)
        signature = hmac.new(
            self._config.secret_key,
            serialized.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return {
            "config": config,
            "signature": signature
        }

    def filter_sensitive_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """过滤敏感数据
        
        Args:
            config: 原始配置
            
        Returns:
            过滤后的配置
        """
        def _filter(data, path=""):
            if isinstance(data, dict):
                filtered = {}
                for k, v in data.items():
                    current_path = f"{path}.{k}" if path else k
                    if isinstance(v, dict):
                        filtered[k] = _filter(v, current_path)
                    elif isinstance(v, list):
                        # 如果键名在敏感键列表中，替换整个数组为["***", ...]
                        if self.is_sensitive_operation(k):
                            filtered[k] = ["***"] * len(v)
                        else:
                            filtered[k] = [_filter(item, f"{current_path}[{i}]") 
                                         for i, item in enumerate(v)]
                    else:
                        # 检查完整路径或键名是否敏感
                        if self.is_sensitive_operation(k) or self.is_sensitive_operation(current_path):
                            filtered[k] = "***"
                        else:
                            filtered[k] = v
                return filtered
            return data
            
        return _filter(config)

    def detect_malicious_input(self, input_data: Dict[str, Any]) -> bool:
        """检测恶意输入
        
        Args:
            input_data: 输入数据
            
        Returns:
            bool: 是否检测到恶意输入
        """
        patterns = [
            ";", "--", "/*", "*/", "xp_", 
            "../", "\\", "|", "&", "$",
            "<", ">", "'", "\""
        ]
        
        def _check(value):
            if isinstance(value, str):
                return any(p in value for p in patterns)
            elif isinstance(value, (dict, list)):
                return any(_check(v) for v in (value.values() if isinstance(value, dict) else value))
            return False
            
        return _check(input_data)

    def _serialize_config(self, config: Dict[str, Any]) -> str:
        """序列化配置为可签名字符串

        Args:
            config: 配置字典

        Returns:
            str: 序列化后的字符串
        """
        def _sort_and_serialize(data):
            if isinstance(data, dict):
                return '{' + ','.join(
                    f'"{k}":{_sort_and_serialize(v)}'
                    for k, v in sorted(data.items())
                ) + '}'
            elif isinstance(data, list):
                return '[' + ','.join(
                    _sort_and_serialize(item) for item in data
                ) + ']'
            else:
                return str(data)

        return _sort_and_serialize(config)
