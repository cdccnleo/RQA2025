"""
数据脱敏模块

本模块实现敏感数据脱敏功能，满足量化交易系统合规要求：
- QTS-018: 数据脱敏

功能特性：
- 敏感字段自动识别与脱敏
- 多种脱敏策略（掩码、哈希、截断）
- 配置化脱敏规则
- API响应自动脱敏
- 日志脱敏
"""

import hashlib
import json
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Callable
from dataclasses import dataclass, field
import copy


class MaskingStrategy(Enum):
    """脱敏策略"""
    MASK = "mask"                    # 掩码 (如: ****1234)
    HASH = "hash"                    # 哈希 (不可逆)
    TRUNCATE = "truncate"            # 截断 (保留前后部分)
    REMOVE = "remove"                # 移除 (设为None)
    REPLACE = "replace"              # 替换为固定值
    PARTIAL = "partial"              # 部分显示


@dataclass
class MaskingRule:
    """脱敏规则"""
    field_pattern: str               # 字段匹配模式 (支持正则)
    strategy: MaskingStrategy
    params: Dict[str, Any] = field(default_factory=dict)
    
    def matches(self, field_name: str) -> bool:
        """检查字段名是否匹配规则"""
        return bool(re.match(self.field_pattern, field_name, re.IGNORECASE))


class DataMasker:
    """
    数据脱敏器
    
    提供敏感数据脱敏功能
    """
    
    # 默认脱敏规则
    DEFAULT_RULES: List[MaskingRule] = [
        # API密钥
        MaskingRule(
            field_pattern=r".*api[_-]?key.*",
            strategy=MaskingStrategy.MASK,
            params={'show_last': 4}
        ),
        # 密码
        MaskingRule(
            field_pattern=r".*password.*",
            strategy=MaskingStrategy.HASH
        ),
        # 密钥
        MaskingRule(
            field_pattern=r".*secret.*",
            strategy=MaskingStrategy.MASK,
            params={'show_last': 4}
        ),
        # Token
        MaskingRule(
            field_pattern=r".*token.*",
            strategy=MaskingStrategy.MASK,
            params={'show_last': 4}
        ),
        # 身份证号
        MaskingRule(
            field_pattern=r".*id[_-]?card.*",
            strategy=MaskingStrategy.PARTIAL,
            params={'show_first': 3, 'show_last': 4}
        ),
        # 手机号
        MaskingRule(
            field_pattern=r".*phone.*",
            strategy=MaskingStrategy.PARTIAL,
            params={'show_first': 3, 'show_last': 4}
        ),
        # 邮箱
        MaskingRule(
            field_pattern=r".*email.*",
            strategy=MaskingStrategy.PARTIAL,
            params={'show_first': 2, 'show_last': 4}
        ),
        # IP地址
        MaskingRule(
            field_pattern=r".*ip[_-]?address.*",
            strategy=MaskingStrategy.PARTIAL,
            params={'show_last_segment': True}
        ),
        # 账户余额
        MaskingRule(
            field_pattern=r".*balance.*",
            strategy=MaskingStrategy.MASK,
            params={'mask_char': '*', 'show_last': 2}
        ),
        # 持仓数量
        MaskingRule(
            field_pattern=r".*position.*",
            strategy=MaskingStrategy.PARTIAL,
            params={'precision': 2}
        ),
    ]
    
    def __init__(self, rules: Optional[List[MaskingRule]] = None):
        """
        初始化脱敏器
        
        Args:
            rules: 自定义脱敏规则，None则使用默认规则
        """
        self.rules = rules or self.DEFAULT_RULES.copy()
        self._custom_handlers: Dict[str, Callable[[Any], Any]] = {}
    
    def add_rule(self, rule: MaskingRule):
        """添加脱敏规则"""
        self.rules.append(rule)
    
    def remove_rule(self, pattern: str):
        """移除脱敏规则"""
        self.rules = [r for r in self.rules if r.field_pattern != pattern]
    
    def register_handler(self, field_name: str, handler: Callable[[Any], Any]):
        """注册自定义处理器"""
        self._custom_handlers[field_name] = handler
    
    def mask_value(self, value: Any, rule: MaskingRule) -> Any:
        """
        对单个值进行脱敏
        
        Args:
            value: 原始值
            rule: 脱敏规则
        
        Returns:
            脱敏后的值
        """
        if value is None:
            return None
        
        strategy = rule.strategy
        params = rule.params
        
        if strategy == MaskingStrategy.REMOVE:
            return None
        
        elif strategy == MaskingStrategy.REPLACE:
            return params.get('replacement', '[REDACTED]')
        
        elif strategy == MaskingStrategy.HASH:
            # 使用SHA256哈希
            value_str = str(value)
            return hashlib.sha256(value_str.encode()).hexdigest()[:16]
        
        elif strategy == MaskingStrategy.MASK:
            value_str = str(value)
            mask_char = params.get('mask_char', '*')
            show_first = params.get('show_first', 0)
            show_last = params.get('show_last', 0)
            
            if len(value_str) <= show_first + show_last:
                return mask_char * len(value_str)
            
            masked_part = mask_char * (len(value_str) - show_first - show_last)
            return value_str[:show_first] + masked_part + value_str[-show_last:]
        
        elif strategy == MaskingStrategy.TRUNCATE:
            value_str = str(value)
            max_length = params.get('max_length', 10)
            
            if len(value_str) <= max_length:
                return value_str
            
            show_start = params.get('show_start', max_length // 2)
            show_end = params.get('show_end', max_length // 2)
            
            return value_str[:show_start] + '...' + value_str[-show_end:]
        
        elif strategy == MaskingStrategy.PARTIAL:
            value_str = str(value)
            show_first = params.get('show_first', 0)
            show_last = params.get('show_last', 0)
            
            # 特殊处理IP地址
            if params.get('show_last_segment') and '.' in value_str:
                parts = value_str.split('.')
                return '*.*.*.' + parts[-1]
            
            # 特殊处理数字精度
            if params.get('precision') and isinstance(value, (int, float)):
                precision = params['precision']
                return round(float(value), precision)
            
            if len(value_str) <= show_first + show_last:
                return value_str
            
            masked_part = '*' * (len(value_str) - show_first - show_last)
            return value_str[:show_first] + masked_part + value_str[-show_last:]
        
        return value
    
    def mask_dict(
        self,
        data: Dict[str, Any],
        depth: int = 0,
        max_depth: int = 10
    ) -> Dict[str, Any]:
        """
        对字典进行脱敏
        
        Args:
            data: 原始数据
            depth: 当前深度
            max_depth: 最大深度
        
        Returns:
            脱敏后的数据
        """
        if depth > max_depth:
            return data
        
        result = {}
        
        for key, value in data.items():
            # 检查是否有自定义处理器
            if key in self._custom_handlers:
                result[key] = self._custom_handlers[key](value)
                continue
            
            # 查找匹配的规则
            matching_rule = None
            for rule in self.rules:
                if rule.matches(key):
                    matching_rule = rule
                    break
            
            if matching_rule:
                # 应用脱敏规则
                if isinstance(value, dict):
                    result[key] = self.mask_dict(value, depth + 1, max_depth)
                elif isinstance(value, list):
                    result[key] = self.mask_list(value, depth + 1, max_depth)
                else:
                    result[key] = self.mask_value(value, matching_rule)
            else:
                # 递归处理嵌套结构
                if isinstance(value, dict):
                    result[key] = self.mask_dict(value, depth + 1, max_depth)
                elif isinstance(value, list):
                    result[key] = self.mask_list(value, depth + 1, max_depth)
                else:
                    result[key] = value
        
        return result
    
    def mask_list(
        self,
        data: List[Any],
        depth: int = 0,
        max_depth: int = 10
    ) -> List[Any]:
        """对列表进行脱敏"""
        if depth > max_depth:
            return data
        
        result = []
        
        for item in data:
            if isinstance(item, dict):
                result.append(self.mask_dict(item, depth + 1, max_depth))
            elif isinstance(item, list):
                result.append(self.mask_list(item, depth + 1, max_depth))
            else:
                result.append(item)
        
        return result
    
    def mask(self, data: Any) -> Any:
        """
        对数据进行脱敏
        
        Args:
            data: 任意类型的数据
        
        Returns:
            脱敏后的数据
        """
        if isinstance(data, dict):
            return self.mask_dict(data)
        elif isinstance(data, list):
            return self.mask_list(data)
        else:
            return data
    
    def mask_json(self, json_str: str) -> str:
        """
        对JSON字符串进行脱敏
        
        Args:
            json_str: JSON字符串
        
        Returns:
            脱敏后的JSON字符串
        """
        try:
            data = json.loads(json_str)
            masked_data = self.mask(data)
            return json.dumps(masked_data, ensure_ascii=False)
        except json.JSONDecodeError:
            return json_str


class APIMaskingMiddleware:
    """
    API响应脱敏中间件
    
    自动对API响应进行脱敏处理
    """
    
    def __init__(
        self,
        masker: Optional[DataMasker] = None,
        exempt_paths: Optional[List[str]] = None
    ):
        self.masker = masker or DataMasker()
        self.exempt_paths = exempt_paths or []
    
    def should_mask(self, path: str) -> bool:
        """检查路径是否需要脱敏"""
        for exempt in self.exempt_paths:
            if path.startswith(exempt):
                return False
        return True
    
    def process_response(self, response_data: Any, path: str) -> Any:
        """
        处理API响应
        
        Args:
            response_data: 响应数据
            path: API路径
        
        Returns:
            脱敏后的响应数据
        """
        if not self.should_mask(path):
            return response_data
        
        return self.masker.mask(response_data)


class LogMasker:
    """
    日志脱敏器
    
    对日志内容进行脱敏
    """
    
    # 敏感信息正则模式
    SENSITIVE_PATTERNS = [
        (r'([\"\']?password[\"\']?\s*[:=]\s*)["\']?([^"\'\s,}]+)', r'\1[MASKED]'),
        (r'([\"\']?api[_-]?key[\"\']?\s*[:=]\s*)["\']?([^"\'\s,}]+)', r'\1[MASKED]'),
        (r'([\"\']?secret[\"\']?\s*[:=]\s*)["\']?([^"\'\s,}]+)', r'\1[MASKED]'),
        (r'([\"\']?token[\"\']?\s*[:=]\s*)["\']?([^"\'\s,}]+)', r'\1[MASKED]'),
        (r'(Bearer\s+)([a-zA-Z0-9_\-\.]+)', r'\1[MASKED]'),
    ]
    
    @classmethod
    def mask_log(cls, message: str) -> str:
        """
        对日志消息进行脱敏
        
        Args:
            message: 原始日志消息
        
        Returns:
            脱敏后的消息
        """
        masked = message
        
        for pattern, replacement in cls.SENSITIVE_PATTERNS:
            masked = re.sub(pattern, replacement, masked, flags=re.IGNORECASE)
        
        return masked


# 便捷函数

def mask_sensitive_data(data: Any, rules: Optional[List[MaskingRule]] = None) -> Any:
    """脱敏敏感数据"""
    masker = DataMasker(rules)
    return masker.mask(data)


def mask_json_string(json_str: str, rules: Optional[List[MaskingRule]] = None) -> str:
    """脱敏JSON字符串"""
    masker = DataMasker(rules)
    return masker.mask_json(json_str)


def mask_log_message(message: str) -> str:
    """脱敏日志消息"""
    return LogMasker.mask_log(message)


# 常用脱敏规则

PII_RULES = [
    # 个人身份信息
    MaskingRule(r".*name.*", MaskingStrategy.PARTIAL, {'show_first': 1, 'show_last': 1}),
    MaskingRule(r".*phone.*", MaskingStrategy.PARTIAL, {'show_first': 3, 'show_last': 4}),
    MaskingRule(r".*email.*", MaskingStrategy.PARTIAL, {'show_first': 2, 'show_last': 4}),
    MaskingRule(r".*address.*", MaskingStrategy.TRUNCATE, {'max_length': 10}),
    MaskingRule(r".*id[_-]?card.*", MaskingStrategy.PARTIAL, {'show_first': 3, 'show_last': 4}),
]

FINANCIAL_RULES = [
    # 金融敏感信息
    MaskingRule(r".*balance.*", MaskingStrategy.MASK, {'mask_char': '*', 'show_last': 2}),
    MaskingRule(r".*amount.*", MaskingStrategy.PARTIAL, {'precision': 2}),
    MaskingRule(r".*price.*", MaskingStrategy.PARTIAL, {'precision': 2}),
    MaskingRule(r".*account.*", MaskingStrategy.MASK, {'show_last': 4}),
    MaskingRule(r".*card[_-]?number.*", MaskingStrategy.MASK, {'show_last': 4}),
]

SECURITY_RULES = [
    # 安全敏感信息
    MaskingRule(r".*password.*", MaskingStrategy.HASH),
    MaskingRule(r".*secret.*", MaskingStrategy.HASH),
    MaskingRule(r".*api[_-]?key.*", MaskingStrategy.MASK, {'show_last': 4}),
    MaskingRule(r".*token.*", MaskingStrategy.MASK, {'show_last': 4}),
    MaskingRule(r".*credential.*", MaskingStrategy.HASH),
    MaskingRule(r".*private[_-]?key.*", MaskingStrategy.HASH),
]


# 全局脱敏器实例
_default_masker: Optional[DataMasker] = None


def get_default_masker() -> DataMasker:
    """获取默认脱敏器"""
    global _default_masker
    if _default_masker is None:
        _default_masker = DataMasker()
    return _default_masker
