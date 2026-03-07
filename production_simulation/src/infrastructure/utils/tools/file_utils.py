"""
file_utils 模块

提供 file_utils 相关功能和接口。
"""

import json
import logging

import pickle

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

"""
RQA2025 文件工具模块

提供文件操作相关的工具函数
"""

logger = logging.getLogger(__name__)


def _safe_logger_log(level: int, message: str) -> None:
    """在单测环境下安全输出日志，兼容被 mock 的 handler.level / parent Mock。"""
    seen_handlers = set()
    visited_loggers = set()
    current_logger = logger
    depth = 0

    while current_logger and id(current_logger) not in visited_loggers and depth < 10:
        visited_loggers.add(id(current_logger))
        depth += 1

        handlers_attr = getattr(current_logger, "handlers", None)
        if isinstance(handlers_attr, (list, tuple, set)):
            handlers = list(handlers_attr)
        elif isinstance(handlers_attr, logging.Handler):
            handlers = [handlers_attr]
        else:
            handlers = []

        for handler in handlers:
            if id(handler) in seen_handlers:
                continue
            seen_handlers.add(id(handler))
            level_value = getattr(handler, "level", logging.NOTSET)
            if not isinstance(level_value, int):
                try:
                    handler.setLevel(logging.NOTSET)  # type: ignore[attr-defined]
                except Exception:
                    try:
                        handler.level = logging.NOTSET  # type: ignore[attr-defined]
                    except Exception:
                        try:
                            handler.__dict__["level"] = logging.NOTSET  # type: ignore[attr-defined]
                        except Exception:
                            pass
            if not isinstance(getattr(handler, "level", None), int):
                try:
                    object.__setattr__(handler, "level", logging.NOTSET)  # type: ignore[attr-defined]
                except Exception:
                    pass

        if not getattr(current_logger, "propagate", True):
            break

        parent_logger = getattr(current_logger, "parent", None)
        if parent_logger is None or parent_logger is current_logger:
            break
        if not isinstance(parent_logger, logging.Logger):
            break
        current_logger = parent_logger

    try:
        if level == logging.INFO and hasattr(logger, "info"):
            logger.info(message)
        elif level == logging.ERROR and hasattr(logger, "error"):
            logger.error(message)
        elif level == logging.WARNING and hasattr(logger, "warning"):
            logger.warning(message)
        elif level == logging.DEBUG and hasattr(logger, "debug"):
            logger.debug(message)
        else:
            logger.log(level, message)
    except TypeError:
        logging.getLogger(logger.name).log(level, message)


class FileUtils:
    """文件工具类"""
    
    def __init__(self):
        """初始化文件工具"""
        pass
    
    def read_file(self, file_path: Union[str, Path], encoding: str = 'utf-8') -> str:
        """读取文件"""
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except:
            return ""
    
    def write_file(self, file_path: Union[str, Path], content: str, encoding: str = 'utf-8') -> bool:
        """写入文件"""
        try:
            with open(file_path, 'w', encoding=encoding) as f:
                f.write(content)
            return True
        except:
            return False
    
    def copy_file(self, src: Union[str, Path], dst: Union[str, Path]) -> bool:
        """复制文件"""
        try:
            import shutil
            shutil.copy2(src, dst)
            return True
        except:
            return False
    
    def delete_file(self, file_path: Union[str, Path]) -> bool:
        """删除文件"""
        try:
            Path(file_path).unlink()
            return True
        except:
            return False


def safe_file_write(
    file_path: Union[str, Path],
    content: Union[str, bytes, Dict, List],
    mode: str = "w",
    encoding: str = "utf - 8",
) -> bool:
    """
    安全地写入文件

    Args:
        file_path: 文件路径
        content: 文件内容
        mode: 写入模式
        encoding: 编码方式

    Returns:
        是否写入成功
    """
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(content, (dict, list)):
            with open(file_path, "w", encoding=encoding) as f:
                json.dump(content, f, ensure_ascii=False, indent=2)
        elif isinstance(content, str):
            with open(file_path, mode, encoding=encoding) as f:
                f.write(content)
        elif isinstance(content, bytes):
            with open(file_path, "wb") as f:
                f.write(content)
        else:
            # 尝试序列化对象
            with open(file_path, "wb") as f:
                pickle.dump(content, f)

        _safe_logger_log(logging.INFO, f"文件写入成功: {file_path}")
        return True

    except Exception as e:
        _safe_logger_log(logging.ERROR, f"文件写入失败 {file_path}: {e}")
        return False


def safe_file_read(
    file_path: Union[str, Path], encoding: str = "utf - 8"
) -> Optional[Any]:
    """
    安全地读取文件

    Args:
        file_path: 文件路径
        encoding: 编码方式

    Returns:
        文件内容或None
    """
    try:
        file_path = Path(file_path)
        normalized_encoding = encoding.replace(" ", "") if isinstance(encoding, str) else encoding

        if not file_path.exists():
            _safe_logger_log(logging.WARNING, f"文件不存在: {file_path}")
            return None

        # 尝试JSON格式
        try:
            with open(file_path, "r", encoding=normalized_encoding) as f:
                return json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            pass

        # 尝试普通文本
        try:
            with open(file_path, "r", encoding=normalized_encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            pass

        # 尝试二进制格式
        try:
            with open(file_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            pass

        _safe_logger_log(logging.ERROR, f"无法读取文件: {file_path}")
        return None

    except Exception as e:
        _safe_logger_log(logging.ERROR, f"文件读取失败 {file_path}: {e}")
        return None


def ensure_directory(dir_path: Union[str, Path]) -> bool:
    """
    确保目录存在

    Args:
        dir_path: 目录路径

    Returns:
        是否创建成功
    """
    try:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        _safe_logger_log(logging.ERROR, f"创建目录失败 {dir_path}: {e}")
        return False


def list_files(dir_path: Union[str, Path], pattern: str = "*") -> List[Path]:
    """
    列出目录中的文件

    Args:
        dir_path: 目录路径
        pattern: 文件模式

    Returns:
        文件列表
    """
    try:
        dir_path = Path(dir_path)
        if not dir_path.exists():
            return []

        return list(dir_path.glob(pattern))
    except Exception as e:
        _safe_logger_log(logging.ERROR, f"列出文件失败 {dir_path}: {e}")
        return []


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    获取文件大小

    Args:
        file_path: 文件路径

    Returns:
        文件大小（字节）
    """
    try:
        return Path(file_path).stat().st_size
    except Exception as e:
        _safe_logger_log(logging.ERROR, f"获取文件大小失败 {file_path}: {e}")
        return 0


def delete_file(file_path: Union[str, Path]) -> bool:
    """
    删除文件

    Args:
        file_path: 文件路径

    Returns:
        是否删除成功
    """
    try:
        Path(file_path).unlink(missing_ok=True)
        return True
    except Exception as e:
        _safe_logger_log(logging.ERROR, f"删除文件失败 {file_path}: {e}")
        return False
