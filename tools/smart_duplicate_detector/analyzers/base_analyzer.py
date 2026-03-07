"""
基础分析器

提供代码分析的基础设施和通用功能。
"""

import ast
import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pathlib import Path
import fnmatch

from ..core.config import SmartDuplicateConfig


class BaseAnalyzer(ABC):
    """
    基础代码分析器

    提供文件处理、AST解析等通用功能。
    """

    def __init__(self, config: SmartDuplicateConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # 缓存
        self._file_cache: Dict[str, str] = {}
        self._ast_cache: Dict[str, ast.AST] = {}

    @abstractmethod
    def analyze(self, target_path: str) -> Any:
        """
        执行分析

        Args:
            target_path: 分析目标路径

        Returns:
            Any: 分析结果
        """

    def get_python_files(self, target_path: str) -> List[str]:
        """
        获取所有Python文件

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
        else:
            for root, dirs, files in os.walk(target_path):
                # 排除目录
                dirs[:] = [d for d in dirs if d not in self.config.exclude_dirs]

                for file in files:
                    file_path = os.path.join(root, file)
                    if self._is_python_file(file_path):
                        python_files.append(file_path)

        # 限制文件数量
        if len(python_files) > self.config.performance.max_files_to_analyze:
            self.logger.warning(f"文件数量过多({len(python_files)})，"
                                f"限制为{self.config.performance.max_files_to_analyze}个")
            python_files = python_files[:self.config.performance.max_files_to_analyze]

        return python_files

    def _is_python_file(self, file_path: str) -> bool:
        """
        判断是否为Python文件

        Args:
            file_path: 文件路径

        Returns:
            bool: 是否为Python文件
        """
        # 检查文件扩展名
        if not any(fnmatch.fnmatch(file_path, pattern) for pattern in self.config.file_patterns):
            return False

        # 检查排除模式
        if any(fnmatch.fnmatch(os.path.basename(file_path), pattern)
               for pattern in self.config.exclude_patterns):
            return False

        return True

    def read_file_content(self, file_path: str) -> Optional[str]:
        """
        读取文件内容（带缓存）

        Args:
            file_path: 文件路径

        Returns:
            Optional[str]: 文件内容
        """
        if file_path in self._file_cache:
            return self._file_cache[file_path]

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self._file_cache[file_path] = content
            return content
        except Exception as e:
            self.logger.error(f"读取文件失败 {file_path}: {e}")
            return None

    def parse_ast(self, file_path: str) -> Optional[ast.AST]:
        """
        解析AST（带缓存）

        Args:
            file_path: 文件路径

        Returns:
            Optional[ast.AST]: AST根节点
        """
        if file_path in self._ast_cache:
            return self._ast_cache[file_path]

        content = self.read_file_content(file_path)
        if content is None:
            return None

        try:
            tree = ast.parse(content, filename=file_path)
            self._ast_cache[file_path] = tree
            return tree
        except SyntaxError as e:
            self.logger.error(f"AST解析失败 {file_path}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"解析文件失败 {file_path}: {e}")
            return None

    def clear_cache(self) -> None:
        """清除缓存"""
        self._file_cache.clear()
        self._ast_cache.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计"""
        return {
            'file_cache_size': len(self._file_cache),
            'ast_cache_size': len(self._ast_cache)
        }
