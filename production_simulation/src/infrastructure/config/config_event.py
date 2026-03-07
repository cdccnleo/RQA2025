
import uuid
import copy

from typing import Dict, Any, Optional
from datetime import datetime
import logging
import time
#!/usr/bin/env python3
"""
配置事件管理
处理配置相关的各种事件
"""

logger = logging.getLogger(__name__)


class ConfigEvent:
    """配置事件基类"""

    def __init__(self, event_type: str = "generic", data: Dict[str, Any] = None, source: str = None):
        self.event_type = event_type
        self.data = copy.deepcopy(data) if data else {}
        self.source = source or "config_system"
        self.timestamp = time.time()
        self.event_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'data': self.data,
            'source': self.source,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat()
        }


class ConfigChangeEvent(ConfigEvent):
    """配置变更事件"""

    def __init__(self, key: str, old_value: Any, new_value: Any, timestamp_or_source=None):
        # 处理参数：可能是时间戳或source
        if isinstance(timestamp_or_source, (int, float)):
            # 如果是数字，当作时间戳
            timestamp = timestamp_or_source
            source = None
        else:
            # 否则当作source
            timestamp = None
            source = timestamp_or_source

        super().__init__('config_changed', {
            'key': key,
            'old_value': old_value,
            'new_value': new_value,
            'change_type': self._determine_change_type(old_value, new_value)
        }, source)

        # 如果提供了自定义时间戳，覆盖默认时间戳
        if timestamp is not None:
            self.timestamp = timestamp

        # 添加便捷属性访问
        self.key = key
        self.old_value = old_value
        self.new_value = new_value
        self.change_type = self._determine_change_type(old_value, new_value)

    def _determine_change_type(self, old_value: Any, new_value: Any) -> str:
        """确定变更类型"""
        if old_value is None and new_value is not None:
            return 'added'
        elif old_value is not None and new_value is None:
            return 'deleted'
        elif old_value == new_value:
            return 'unchanged'
        else:
            return 'modified'


class ConfigLoadEvent(ConfigEvent):
    """配置加载事件"""

    def __init__(
        self,
        *args,
        source: str = None,
        file_path: str = None,
        success: bool = True,
        error_message: str = None,
        data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """创建配置加载事件，兼容多种历史调用方式"""

        payload: Dict[str, Any] = dict(data or {})

        positional = list(args)

        # 兼容旧签名：ConfigLoadEvent(source_type, file_path, success, error_message, data)
        if positional:
            if len(positional) == 1:
                first = positional[0]
                if isinstance(first, str):
                    file_path = file_path or first
                elif isinstance(first, dict):
                    payload.update(first)
            elif len(positional) >= 2:
                first, second = positional[0], positional[1]
                if isinstance(first, str) and isinstance(second, dict):
                    file_path = file_path or first
                    payload.update(second)
                    if len(positional) >= 3 and isinstance(positional[2], bool):
                        success = positional[2]
                    if len(positional) >= 4 and error_message is None and isinstance(positional[3], str):
                        error_message = positional[3]
                elif isinstance(first, str) and isinstance(second, str):
                    source = source or first
                    file_path = file_path or second
                    if len(positional) >= 3 and isinstance(positional[2], bool):
                        success = positional[2]
                    if len(positional) >= 4 and error_message is None and isinstance(positional[3], str):
                        error_message = positional[3]
                    if len(positional) >= 5 and isinstance(positional[4], dict):
                        payload.update(positional[4])
                else:
                    # 回退：将第一个参数视为文件路径，第二个若为dict则并入数据
                    if isinstance(first, str):
                        file_path = file_path or first
                    if isinstance(second, dict):
                        payload.update(second)
                    if len(positional) >= 3 and isinstance(positional[2], bool):
                        success = positional[2]
                    if len(positional) >= 4 and error_message is None and isinstance(positional[3], str):
                        error_message = positional[3]

        # 处理通过关键字提供的可选字段
        if 'data' in kwargs and isinstance(kwargs['data'], dict):
            payload.update(kwargs['data'])

        resolved_file_path = file_path or kwargs.get('path') or ""
        resolved_source = source or kwargs.get('source_type')

        if not resolved_source:
            resolved_source = self._determine_format(resolved_file_path) if resolved_file_path else "unknown"

        # 默认错误消息为空字符串以匹配测试期望
        error_message = "" if error_message is None else error_message

        event_payload = {
            'source_type': resolved_source,
            'source_path': resolved_file_path,
            'file_path': resolved_file_path,
            'success': success,
            'error_message': error_message,
            'data': payload,
        }

        super().__init__('config_loaded', event_payload)

        self.file_path = resolved_file_path
        self.format = resolved_source
        self.success = success
        self.error_message = error_message

    @staticmethod
    def _determine_format(file_path: str) -> str:
        """根据文件路径确定格式"""
        if '.' not in file_path:
            return 'unknown'

        extension = file_path.split('.')[-1].lower()
        format_map = {
            'json': 'json',
            'yaml': 'yaml',
            'yml': 'yaml',
            'toml': 'toml',
            'ini': 'ini',
            'cfg': 'ini',
            'conf': 'ini'
        }
        return format_map.get(extension, 'unknown')


class ConfigValidationEvent(ConfigEvent):
    """配置验证事件"""

    def __init__(self, config_key: str, passed: bool, errors: list = None):
        super().__init__('config_validated', {
            'config_key': config_key,
            'is_valid': passed,  # 为了兼容测试
            'passed': passed,
            'validation_errors': errors or [],  # 为了兼容测试
            'errors': errors or []
        })

        self.config_key = config_key
        self.passed = passed
        self.errors = errors or []


class ConfigReloadEvent(ConfigEvent):
    """配置重载事件"""

    def __init__(self, trigger: str, success: bool, changed_keys: list = None):
        super().__init__('config_reloaded', {
            'trigger': trigger,
            'success': success,
            'changed_keys': changed_keys or []
        })


class ConfigBackupEvent(ConfigEvent):
    """配置备份事件"""

    def __init__(self, backup_path: str, success: bool, backup_size: int = None):
        super().__init__('config_backed_up', {
            'backup_path': backup_path,
            'success': success,
            'backup_size': backup_size
        })


class ConfigErrorEvent(ConfigEvent):
    """配置错误事件"""

    def __init__(self, error_type_or_operation: str, error_message_or_exception, context: Dict[str, Any] = None):
        # 支持两种调用方式：
        # 1. ConfigErrorEvent("validation_error", "Invalid host", context) - error_type, error_message, context
        # 2. ConfigErrorEvent("save", Exception("error"), context) - operation, exception, context

        if isinstance(error_message_or_exception, str):
            # 方式1: error_type, error_message, context
            error_type = error_type_or_operation
            error_message = error_message_or_exception
            operation = context.get('operation', 'unknown') if context else 'unknown'
        else:
            # 方式2: operation, exception, context
            operation = error_type_or_operation
            exception = error_message_or_exception
            if exception is None:
                error_type = "Unknown"
                error_message = "Unknown error"
            else:
                error_type = type(exception).__name__
                error_message = f"{error_type}: {str(exception)}"

        super().__init__('config_error', {
            'operation': operation,
            'error_type': error_type,
            'error_message': error_message,
            'context': context or {}
        })

        self.operation = operation
        self.error_message = error_message
        self.error_type = error_type
        self.context = context or {}

# 注意: ConfigEventBus 已统一到 services/event_service.py 中




