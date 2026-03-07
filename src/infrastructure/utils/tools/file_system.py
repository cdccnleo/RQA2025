"""
file_system 模块

提供 file_system 相关功能和接口。
"""

import json
import os

# 文件系统常量
import glob

from src.infrastructure.utils.core import StorageAdapter
from pathlib import Path
from typing import Dict, Optional, Union, List

"""
基础设施层 - 工具组件组件

file_system 模块

通用工具组件
提供工具组件相关的功能实现。
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class FileSystem:
    """文件系统工具类"""
    
    def __init__(self):
        """初始化文件系统工具"""
        pass
    
    def create_directory(self, path: Union[str, Path]) -> bool:
        """创建目录"""
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            return True
        except:
            return False
    
    def list_directory(self, path: Union[str, Path]) -> list:
        """列出目录内容"""
        try:
            return [str(p) for p in Path(path).iterdir()]
        except:
            return []
    
    def join_path(self, *parts) -> str:
        """连接路径"""
        return str(Path(*parts))


class FileSystemConstants:
    """文件系统相关常量"""

    # 默认存储路径
    DEFAULT_BASE_PATH = "data/storage"

    # 文件格式配置
    JSON_FILE_SUFFIX = ".json"
    JSON_INDENT_LEVEL = 2

    # 编码配置
    DEFAULT_ENCODING = "utf-8"


class FileSystemAdapter(StorageAdapter):
    """本地文件系统存储适配器"""

    def __init__(self, base_path: str = FileSystemConstants.DEFAULT_BASE_PATH):
        self.base_path = Path(base_path)
        os.makedirs(self.base_path, exist_ok=True)

    def _build_path(self, path: str) -> Path:
        return (self.base_path / path).with_suffix(FileSystemConstants.JSON_FILE_SUFFIX)

    def write(self, path: str, data: Dict) -> bool:
        """写入数据到文件系统"""
        full_path = self._build_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(
                str(full_path), "w", encoding=FileSystemConstants.DEFAULT_ENCODING
            ) as f:
                json.dump(
                    data,
                    f,
                    indent=FileSystemConstants.JSON_INDENT_LEVEL,
                    ensure_ascii=False,
                )
            return True
        except (IOError, TypeError):
            return False

    def read(self, path: str) -> Optional[Dict]:
        """从文件系统读取数据"""
        full_path = self._build_path(path)
        if not full_path.exists():
            return None

        try:
            with open(
                str(full_path), "r", encoding=FileSystemConstants.DEFAULT_ENCODING
            ) as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError):
            return None

    def delete(self, path: str) -> bool:
        """删除文件"""
        full_path = self._build_path(path)
        try:
            os.remove(str(full_path))
            return True
        except FileNotFoundError:
            return False
        except OSError:
            return False

    # StorageAdapter 接口兼容实现
    def save(self, key: str, data: Dict) -> bool:
        return self.write(key, data)

    def load(self, key: str) -> Optional[Dict]:
        return self.read(key)

    def exists(self, key: str) -> bool:
        return self._build_path(key).exists()

    def list_keys(self) -> List[str]:
        pattern = str(self.base_path / f"**/*{FileSystemConstants.JSON_FILE_SUFFIX}")
        return [Path(p).relative_to(self.base_path).with_suffix("").as_posix() for p in glob.glob(pattern, recursive=True)]

class AShareFileSystemAdapter(FileSystemAdapter):
    """A股专用文件存储适配器"""

    def __init__(self, base_path: str = "data/storage"):
        super().__init__(base_path)

    def _build_path(self, path: str) -> Path:  # type: ignore[override]
        return self.base_path / path

    def format_path(self, symbol: str, date: str) -> str:
        """格式化文件路径"""
        return f"stock/{symbol}/{date}.parquet"


    def batch_write(self, batch_data: Dict[str, Dict[str, Dict]]) -> bool:
        """批量写入数据"""
        for symbol, date_dict in batch_data.items():
            for date, data in date_dict.items():
                path = self.format_path(symbol, date)
                if not self.write(path, data):
                    return False
        return True

    def get_latest_data(self, symbol: str) -> Optional[str]:
        """获取最新数据文件路径"""
        pattern = Path(self.base_path) / f"stock/{symbol}/*.parquet"
        files = list(glob.glob(str(pattern)))
        if not files:
            return None
        return max(files)
