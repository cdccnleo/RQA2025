#!/usr/bin/env python3
"""
基础重构执行器

定义重构执行器的接口和基础功能。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from tools.smart_code_analyzer import RefactoringSuggestion


@dataclass
class RefactorResult:
    """重构执行结果"""

    success: bool
    changes: List[Dict[str, Any]] = None
    errors: List[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.changes is None:
            self.changes = []
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class BaseRefactorExecutor(ABC):
    """基础重构执行器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @property
    @abstractmethod
    def refactor_type(self) -> str:
        """重构类型"""

    @abstractmethod
    def can_execute(self, suggestion: RefactoringSuggestion) -> bool:
        """
        检查是否可以执行此重构

        Args:
            suggestion: 重构建议

        Returns:
            是否可以执行
        """

    @abstractmethod
    def execute(self, suggestion: RefactoringSuggestion, context: Optional[Dict[str, Any]] = None) -> RefactorResult:
        """
        执行重构

        Args:
            suggestion: 重构建议
            context: 执行上下文

        Returns:
            执行结果
        """

    def validate_preconditions(self, suggestion: RefactoringSuggestion) -> List[str]:
        """
        验证执行前置条件

        Args:
            suggestion: 重构建议

        Returns:
            错误消息列表，如果为空表示条件满足
        """
        errors = []

        # 检查文件是否存在
        import os
        if not os.path.exists(suggestion.file_path):
            errors.append(f"文件不存在: {suggestion.file_path}")

        # 检查文件是否可写
        if not os.access(suggestion.file_path, os.W_OK):
            errors.append(f"文件不可写: {suggestion.file_path}")

        return errors

    def create_change_record(self, change_type: str, description: str, **kwargs) -> Dict[str, Any]:
        """
        创建变更记录

        Args:
            change_type: 变更类型
            description: 描述
            **kwargs: 额外信息

        Returns:
            变更记录字典
        """
        import time
        return {
            'type': change_type,
            'description': description,
            'timestamp': time.time(),
            **kwargs
        }

    def read_file_content(self, file_path: str) -> str:
        """读取文件内容"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def write_file_content(self, file_path: str, content: str):
        """写入文件内容"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

    def backup_file(self, file_path: str) -> str:
        """备份文件（返回备份路径）"""
        import shutil
        from pathlib import Path
        import tempfile

        backup_dir = Path(tempfile.gettempdir()) / "auto_refactor_temp"
        backup_dir.mkdir(exist_ok=True)

        source_path = Path(file_path)
        backup_name = f"{source_path.stem}_refactor_backup{source_path.suffix}"
        backup_path = backup_dir / backup_name

        shutil.copy2(source_path, backup_path)
        return str(backup_path)
