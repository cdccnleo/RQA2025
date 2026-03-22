# -*- coding: utf-8 -*-
"""
事件格式验证模块

提供事件格式验证功能，确保事件符合系统规范。

函数级注释:
- 所有公开函数均包含详细的中文注释
- 支持事件格式版本管理
- 提供详细的验证错误信息

作者: AI系统集成助手
日期: 2026-03-22
版本: 1.0.0
"""

import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum


class ValidationError:
    """验证错误类"""
    
    def __init__(self, field: str, message: str, severity: str = "error"):
        """
        初始化验证错误
        
        Args:
            field: 错误字段
            message: 错误信息
            severity: 严重程度(error/warning)
        """
        self.field = field
        self.message = message
        self.severity = severity
    
    def __str__(self):
        return f"[{self.severity.upper()}] {self.field}: {self.message}"


class EventType(Enum):
    """标准事件类型枚举"""
    # 交易层事件
    ORDER_CREATED = "ORDER_CREATED"
    ORDER_EXECUTED = "ORDER_EXECUTED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    ORDER_REJECTED = "ORDER_REJECTED"
    
    # 风控层事件
    RISK_INTERCEPTED = "RISK_INTERCEPTED"
    RISK_ALERT = "RISK_ALERT"
    
    # 数据层事件
    DATA_COLLECTED = "DATA_COLLECTED"
    DATA_QUALITY_CHECKED = "DATA_QUALITY_CHECKED"
    
    # 策略层事件
    SIGNAL_GENERATED = "SIGNAL_GENERATED"
    STRATEGY_DECISION_READY = "STRATEGY_DECISION_READY"
    
    # 调度器事件
    TASK_COMPLETED = "TASK_COMPLETED"
    TASK_FAILED = "TASK_FAILED"
    
    # 风险拦截处理事件
    RISK_INTERCEPT_HANDLED = "RISK_INTERCEPT_HANDLED"


class EventValidator:
    """
    事件验证器类
    
    提供事件格式验证功能，确保事件符合RQA2025系统规范。
    
    使用示例:
        validator = EventValidator()
        is_valid, errors = validator.validate(event_data)
        
        if not is_valid:
            for error in errors:
                print(error)
    """
    
    # 必填字段列表
    REQUIRED_FIELDS = [
        "event_id",
        "event_type",
        "timestamp",
        "correlation_id",
        "source",
        "version",
        "payload"
    ]
    
    # 有效的事件类型
    VALID_EVENT_TYPES = {e.value for e in EventType}
    
    # 版本号正则
    VERSION_PATTERN = re.compile(r'^(\d+)\.(\d+)(?:\.(\d+))?$')
    
    def __init__(self, strict_mode: bool = True):
        """
        初始化验证器
        
        Args:
            strict_mode: 是否启用严格模式，严格模式下所有警告视为错误
        """
        self.strict_mode = strict_mode
        self.errors: List[ValidationError] = []
    
    def validate(self, event: Dict[str, Any]) -> Tuple[bool, List[ValidationError]]:
        """
        验证事件格式
        
        Args:
            event: 事件数据字典
            
        Returns:
            Tuple[bool, List[ValidationError]]: (是否有效, 错误列表)
        """
        self.errors = []
        
        # 检查是否为字典
        if not isinstance(event, dict):
            self.errors.append(ValidationError("root", "事件必须是字典类型"))
            return False, self.errors
        
        # 验证必填字段
        self._validate_required_fields(event)
        
        # 验证各字段格式
        if "event_id" in event:
            self._validate_uuid(event["event_id"], "event_id")
        
        if "event_type" in event:
            self._validate_event_type(event["event_type"])
        
        if "timestamp" in event:
            self._validate_timestamp(event["timestamp"])
        
        if "correlation_id" in event:
            self._validate_uuid(event["correlation_id"], "correlation_id")
        
        if "source" in event:
            self._validate_source(event["source"])
        
        if "version" in event:
            self._validate_version(event["version"])
        
        if "payload" in event:
            self._validate_payload(event["payload"])
        
        if "metadata" in event:
            self._validate_metadata(event["metadata"])
        
        # 检查是否有错误
        has_errors = any(e.severity == "error" for e in self.errors)
        has_warnings = any(e.severity == "warning" for e in self.errors)
        
        if self.strict_mode and has_warnings:
            has_errors = True
        
        return not has_errors, self.errors
    
    def _validate_required_fields(self, event: Dict[str, Any]):
        """验证必填字段"""
        for field in self.REQUIRED_FIELDS:
            if field not in event:
                self.errors.append(ValidationError(field, f"缺少必填字段: {field}"))
    
    def _validate_uuid(self, value: Any, field_name: str):
        """验证UUID格式"""
        if not isinstance(value, str):
            self.errors.append(ValidationError(field_name, f"{field_name}必须是字符串"))
            return
        
        try:
            uuid.UUID(value)
        except ValueError:
            self.errors.append(ValidationError(field_name, f"{field_name}必须是有效的UUID格式"))
    
    def _validate_event_type(self, value: Any):
        """验证事件类型"""
        if not isinstance(value, str):
            self.errors.append(ValidationError("event_type", "event_type必须是字符串"))
            return
        
        # 检查是否为大写字母+下划线格式
        if not re.match(r'^[A-Z][A-Z_]*$', value):
            self.errors.append(
                ValidationError("event_type", "event_type必须使用大写字母+下划线命名", "warning")
            )
        
        # 检查是否为已知事件类型
        if value not in self.VALID_EVENT_TYPES:
            self.errors.append(
                ValidationError("event_type", f"未知的事件类型: {value}", "warning")
            )
    
    def _validate_timestamp(self, value: Any):
        """验证时间戳格式"""
        if not isinstance(value, str):
            self.errors.append(ValidationError("timestamp", "timestamp必须是字符串"))
            return
        
        # 尝试解析ISO 8601格式
        try:
            # 支持带时区的格式
            if '+' in value or 'Z' in value:
                # 尝试解析带时区的格式
                try:
                    datetime.fromisoformat(value.replace('Z', '+00:00'))
                except ValueError:
                    self.errors.append(ValidationError("timestamp", "timestamp必须是有效的ISO 8601格式(带时区)"))
            else:
                self.errors.append(
                    ValidationError("timestamp", "timestamp建议包含时区信息", "warning")
                )
                try:
                    datetime.fromisoformat(value)
                except ValueError:
                    self.errors.append(ValidationError("timestamp", "timestamp必须是有效的ISO 8601格式"))
        except Exception:
            self.errors.append(ValidationError("timestamp", "timestamp必须是有效的ISO 8601格式"))
    
    def _validate_source(self, value: Any):
        """验证来源字段"""
        if not isinstance(value, str):
            self.errors.append(ValidationError("source", "source必须是字符串"))
            return
        
        if not value:
            self.errors.append(ValidationError("source", "source不能为空"))
            return
        
        # 建议使用小写字母+下划线
        if not re.match(r'^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$', value):
            self.errors.append(
                ValidationError("source", "source建议使用小写字母+下划线命名(如: trading.execution)", "warning")
            )
    
    def _validate_version(self, value: Any):
        """验证版本号"""
        if not isinstance(value, str):
            self.errors.append(ValidationError("version", "version必须是字符串"))
            return
        
        if not self.VERSION_PATTERN.match(value):
            self.errors.append(ValidationError("version", "version必须符合语义化版本规范(如: 1.0.0)"))
    
    def _validate_payload(self, value: Any):
        """验证payload"""
        if not isinstance(value, dict):
            self.errors.append(ValidationError("payload", "payload必须是字典类型"))
            return
        
        if not value:
            self.errors.append(ValidationError("payload", "payload不能为空对象"))
    
    def _validate_metadata(self, value: Any):
        """验证metadata"""
        if value is None:
            return
        
        if not isinstance(value, dict):
            self.errors.append(ValidationError("metadata", "metadata必须是字典类型"))
            return


class EventFormatConverter:
    """
    事件格式转换器
    
    提供旧格式事件到新格式的转换功能。
    """
    
    @staticmethod
    def convert_legacy_event(legacy_event: Dict[str, Any], source: str = "unknown") -> Dict[str, Any]:
        """
        将旧格式事件转换为新格式
        
        Args:
            legacy_event: 旧格式事件
            source: 事件来源
            
        Returns:
            Dict[str, Any]: 新格式事件
        """
        # 生成新的标准格式事件
        new_event = {
            "event_id": legacy_event.get("id") or str(uuid.uuid4()),
            "event_type": legacy_event.get("type", "UNKNOWN").upper(),
            "timestamp": legacy_event.get("timestamp") or datetime.now().isoformat(),
            "correlation_id": legacy_event.get("correlation_id") or str(uuid.uuid4()),
            "source": source,
            "version": "1.0",
            "payload": {},
            "metadata": {}
        }
        
        # 将旧事件的其他字段移到payload中
        for key, value in legacy_event.items():
            if key not in ["id", "type", "timestamp", "correlation_id"]:
                if key in ["user_id", "session_id", "trace_id"]:
                    new_event["metadata"][key] = value
                else:
                    new_event["payload"][key] = value
        
        return new_event


# 便捷函数
def validate_event(event: Dict[str, Any], strict: bool = True) -> Tuple[bool, List[str]]:
    """
    便捷验证函数
    
    Args:
        event: 事件数据
        strict: 是否严格模式
        
    Returns:
        Tuple[bool, List[str]]: (是否有效, 错误信息列表)
    """
    validator = EventValidator(strict_mode=strict)
    is_valid, errors = validator.validate(event)
    
    error_messages = [str(e) for e in errors]
    return is_valid, error_messages


def is_valid_event(event: Dict[str, Any]) -> bool:
    """
    快速验证事件是否有效
    
    Args:
        event: 事件数据
        
    Returns:
        bool: 是否有效
    """
    validator = EventValidator(strict_mode=False)
    is_valid, _ = validator.validate(event)
    return is_valid
