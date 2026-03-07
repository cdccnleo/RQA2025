"""
file 模块

提供 file 相关功能和接口。
"""

import logging

import gzip

from .base import BaseHandler
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from ..core.exceptions import LogHandlerError as HandlerError
"""
基础设施层 - 文件日志处理器

实现文件日志输出功能，支持轮转和压缩。
"""


class FileHandler(BaseHandler):
    """文件日志处理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化文件处理器

        Args:
            config: 处理器配置
        """
        super().__init__(config)

        self.file_path = Path(self.config.get('file_path', 'logs/app.log'))
        self.max_bytes = self.config.get('max_bytes', 10 * 1024 * 1024)  # 10MB
        self.backup_count = self.config.get('backup_count', 5)
        self.encoding = self.config.get('encoding', 'utf-8')
        self.compress = self.config.get('compress', False)

        self._file = None
        self._current_size = 0
        self._formatter = None

        # 确保目录存在
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def _emit(self, record: logging.LogRecord) -> None:
        """发出日志记录到文件"""
        try:
            # 检查是否需要轮转
            if self._should_rotate():
                self._rotate()

            # 确保文件已打开
            if self._file is None:
                self._open_file()

            # 格式化并写入
            message = self._format_record(record)
            self._file.write(message + '\n')
            self._file.flush()

            # 更新文件大小
            self._current_size += len(message.encode(self.encoding))

        except Exception as e:
            raise HandlerError(f"Failed to write to file {self.file_path}: {e}")

    def _format_record(self, record: logging.LogRecord) -> str:
        """格式化日志记录"""
        if self._formatter:
            return self._formatter.format(record)
        else:
            # 默认格式
            timestamp = datetime.fromtimestamp(record.created).isoformat()
            return f"{timestamp} {record.levelname} {record.name}: {record.getMessage()}"

    def _should_rotate(self) -> bool:
        """检查是否需要轮转"""
        if self._file is None:
            return False

        return self._current_size >= self.max_bytes

    def _rotate(self) -> None:
        """
        执行文件轮转

        关闭当前文件，重命名备份文件，压缩文件，重置计数器
        """
        self._close_current_file()
        self._rotate_backup_files()
        self._rotate_current_file()
        self._reset_size_counter()

    def _close_current_file(self) -> None:
        """关闭当前文件"""
        if self._file:
            self._file.close()
            self._file = None

    def _rotate_backup_files(self) -> None:
        """轮转备份文件"""
        for i in range(self.backup_count - 1, 0, -1):
            src = self._get_backup_path(i - 1)
            dst = self._get_backup_path(i)
            if src.exists():
                src.replace(dst)

    def _rotate_current_file(self) -> None:
        """轮转当前文件"""
        if self.file_path.exists():
            backup_path = self._get_backup_path(0)
            self.file_path.replace(backup_path)

            # 压缩旧文件
            if self.compress:
                self._compress_file(backup_path)

    def _reset_size_counter(self) -> None:
        """重置大小计数器"""
        self._current_size = 0

    def _get_backup_path(self, index: int) -> Path:
        """获取备份文件路径"""
        stem = self.file_path.stem
        suffix = self.file_path.suffix
        return self.file_path.parent / f"{stem}.{index}{suffix}"

    def _compress_file(self, file_path: Path) -> None:
        """压缩文件"""
        try:
            compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
            with file_path.open('rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    f_out.writelines(f_in)
            file_path.unlink()  # 删除原文件
        except Exception:
            # 压缩失败，保留原文件
            pass

    def _open_file(self) -> None:
        """打开日志文件"""
        try:
            self._file = self.file_path.open('a', encoding=self.encoding)
            # 获取当前文件大小
            if self.file_path.exists():
                self._current_size = self.file_path.stat().st_size
            else:
                self._current_size = 0
        except Exception as e:
            raise HandlerError(f"Failed to open log file {self.file_path}: {e}")

    def _close(self) -> None:
        """关闭文件处理器"""
        if self._file:
            self._file.close()
            self._file = None

    def set_formatter(self, formatter: logging.Formatter) -> None:
        """设置格式化器"""
        self._formatter = formatter

    def get_status(self) -> Dict[str, Any]:
        """获取处理器状态"""
        status = super().get_status()
        status.update({
            'file_path': str(self.file_path),
            'current_size': self._current_size,
            'max_bytes': self.max_bytes,
            'backup_count': self.backup_count,
            'compress': self.compress,
            'file_exists': self.file_path.exists(),
            'has_formatter': self._formatter is not None
        })
        return status
