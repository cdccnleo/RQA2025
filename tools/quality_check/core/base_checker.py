"""
质量检查器基类

定义所有质量检查器的通用接口和基础功能。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

from .check_result import CheckResult, Issue, IssueSeverity


class BaseChecker(ABC):
    """
    质量检查器基类

    所有具体的质量检查器都应该继承此类。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化检查器

        Args:
            config: 检查器配置
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

        # 设置默认配置
        self._setup_default_config()

    def _setup_default_config(self) -> None:
        """设置默认配置"""
        # 子类可以重写此方法来设置自己的默认配置

    @property
    @abstractmethod
    def checker_name(self) -> str:
        """检查器名称"""

    @property
    @abstractmethod
    def checker_description(self) -> str:
        """检查器描述"""

    @abstractmethod
    def check(self, target_path: str) -> CheckResult:
        """
        执行检查

        Args:
            target_path: 检查目标路径

        Returns:
            CheckResult: 检查结果
        """

    def _create_result(self) -> CheckResult:
        """创建检查结果对象"""
        return CheckResult(self.checker_name)

    def _create_issue(self,
                      file_path: str,
                      message: str,
                      severity: IssueSeverity,
                      rule_id: str,
                      line_number: Optional[int] = None,
                      details: Optional[Dict[str, Any]] = None) -> Issue:
        """
        创建问题对象

        Args:
            file_path: 文件路径
            message: 问题描述
            severity: 严重程度
            rule_id: 规则ID
            line_number: 行号
            details: 详细信息

        Returns:
            Issue: 问题对象
        """
        return Issue(
            file_path=file_path,
            line_number=line_number,
            message=message,
            severity=severity,
            rule_id=rule_id,
            details=details
        )

    def _get_config_value(self, key: str, default: Any = None) -> Any:
        """
        获取配置值

        Args:
            key: 配置键
            default: 默认值

        Returns:
            配置值
        """
        return self.config.get(key, default)

    def _is_python_file(self, file_path: str) -> bool:
        """
        检查是否为Python文件

        Args:
            file_path: 文件路径

        Returns:
            bool: 是否为Python文件
        """
        return file_path.endswith('.py')

    def _collect_python_files(self, target_path: str) -> List[str]:
        """
        收集Python文件

        Args:
            target_path: 目标路径

        Returns:
            List[str]: Python文件列表
        """
        python_files = []
        path = Path(target_path)

        if path.is_file():
            if self._is_python_file(str(path)):
                python_files.append(str(path))
        elif path.is_dir():
            for py_file in path.rglob('*.py'):
                # 排除__pycache__目录和隐藏文件
                if '__pycache__' not in str(py_file) and not py_file.name.startswith('.'):
                    python_files.append(str(py_file))

        return python_files

    def _read_file_content(self, file_path: str) -> Optional[str]:
        """
        读取文件内容

        Args:
            file_path: 文件路径

        Returns:
            Optional[str]: 文件内容，读取失败返回None
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            self.logger.warning(f"无法读取文件 {file_path}: {e}")
            return None
