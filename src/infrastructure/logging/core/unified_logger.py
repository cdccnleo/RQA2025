"""
统一日志器模块

提供统一的日志接口实现，以及测试使用的兼容日志器。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from .interfaces import LogLevel


def get_unified_logger(name: str = "unified") -> logging.Logger:
    """获取统一日志器"""
    return logging.getLogger(name)


class LogRecorder:
    """基础日志记录器包装"""

    def __init__(self, name: str = "unified") -> None:
        self.logger = logging.getLogger(name)

    def log(self, level: int, message: str) -> None:
        self.logger.log(level, message)


class UnifiedLogger:
    """统一日志器实现"""

    def __init__(
        self,
        name: str = "unified",
        level: Union[LogLevel, int, str] = LogLevel.INFO,
        category: Optional[str] = None,
    ) -> None:
        self.name = name or "unified"
        self.level = self._normalize_level(level)
        self.category = category or "general"
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self._resolve_numeric_level(self.level))
        self._recorder = LogRecorder(name)
        self._log_dir = Path("logs")
        self._log_dir.mkdir(exist_ok=True)
        self._business_logger = BusinessLoggerAdapter(name, self.level, self.category)

        self._stats: Dict[str, Any] = {
            "total": 0,
            "counts": {
                "DEBUG": 0,
                "INFO": 0,
                "WARNING": 0,
                "ERROR": 0,
                "CRITICAL": 0,
            },
        }
        self._log_history: List[Dict[str, Any]] = []
        self._performance_stats: Dict[str, Any] = {
            "total_logs": 0,
            "processing_times": [],
            "avg_processing_time": 0.0,
        }
        self._custom_handlers: List[Any] = []
        self._custom_filters: List[Any] = []

    # ------------------------------------------------------------------ #
    # 基础工具方法
    # ------------------------------------------------------------------ #
    def _normalize_level(self, level: Union[LogLevel, int, str]) -> LogLevel:
        if isinstance(level, LogLevel):
            return level
        if isinstance(level, int):
            for member in LogLevel:
                if getattr(logging, member.value) == level:
                    return member
            return LogLevel.INFO
        level_str = str(level).upper()
        return LogLevel[level_str] if level_str in LogLevel.__members__ else LogLevel.INFO

    def _resolve_numeric_level(self, level: Union[LogLevel, int, str]) -> int:
        if isinstance(level, LogLevel):
            return getattr(logging, level.value)
        if isinstance(level, int):
            return level
        return getattr(logging, str(level).upper(), logging.INFO)

    def _convert_level(self, level: Union[LogLevel, str, int]) -> int:
        return self._resolve_numeric_level(level)

    # ------------------------------------------------------------------ #
    # 结构化日志接口
    # ------------------------------------------------------------------ #
    def log_structured(self, level: Union[LogLevel, str], message: str, **kwargs) -> None:
        self._business_logger.log_structured(str(level), message, **kwargs)

    def log_performance(self, operation: str, duration: float, **kwargs) -> None:
        self._business_logger.log_performance(operation, duration, **kwargs)

    def log_error_with_context(self, error: Exception, context: Dict[str, Any]) -> None:
        self._business_logger.log_error_with_context(error, context)

    def log_business_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        self._business_logger.log_business_event(event_type, event_data)

    # ------------------------------------------------------------------ #
    # Handler / Filter 管理
    # ------------------------------------------------------------------ #
    def add_handler(self, handler: Any) -> None:
        if handler not in self._custom_handlers:
            self._custom_handlers.append(handler)
            self.logger.addHandler(handler)

    def remove_handler(self, handler: Any) -> None:
        if handler in self._custom_handlers:
            self._custom_handlers.remove(handler)
        if handler in self.logger.handlers:
            self.logger.removeHandler(handler)

    def clear_handlers(self) -> None:
        for handler in self._custom_handlers:
            if handler in self.logger.handlers:
                self.logger.removeHandler(handler)
        self._custom_handlers.clear()

    def get_handlers(self) -> List[Any]:
        return list(self._custom_handlers)

    def addFilter(self, filter_obj: Any) -> None:  # pragma: no cover - 兼容旧接口
        self.add_filter(filter_obj)

    def add_filter(self, filter_obj: Any) -> None:
        if filter_obj not in self._custom_filters:
            self._custom_filters.append(filter_obj)
            self.logger.addFilter(filter_obj)

    def removeFilter(self, filter_obj: Any) -> None:  # pragma: no cover - 兼容旧接口
        self.remove_filter(filter_obj)

    def remove_filter(self, filter_obj: Any) -> None:
        if filter_obj in self._custom_filters:
            self._custom_filters.remove(filter_obj)
        try:
            self.logger.removeFilter(filter_obj)
        except ValueError:  # pragma: no cover
            pass

    def addHandler(self, handler: Any) -> None:  # pragma: no cover - 兼容旧接口
        self.add_handler(handler)

    def removeHandler(self, handler: Any) -> None:  # pragma: no cover - 兼容旧接口
        self.remove_handler(handler)

    # ------------------------------------------------------------------ #
    # Level & Stats
    # ------------------------------------------------------------------ #
    def setLevel(self, level: Union[LogLevel, int, str]) -> None:
        self.set_level(level)

    def set_level(self, level: Union[LogLevel, int, str]) -> None:
        self.level = self._normalize_level(level)
        numeric = self._resolve_numeric_level(self.level)
        self.logger.setLevel(numeric)

    def get_level(self) -> LogLevel:
        return self.level

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total": self._stats["total"],
            "counts": self._stats["counts"].copy(),
        }

    def get_log_stats(self) -> Dict[str, Any]:
        if self._performance_stats["processing_times"]:
            avg = sum(self._performance_stats["processing_times"]) / len(self._performance_stats["processing_times"])
            self._performance_stats["avg_processing_time"] = avg
        return {
            "logger_name": self.name,
            "level": self.level.value,
            "handlers_count": len(self._custom_handlers),
            "performance": {
                "total_logs": self._performance_stats["total_logs"],
                "avg_processing_time": self._performance_stats["avg_processing_time"],
            },
        }

    # ------------------------------------------------------------------ #
    # Logging 接口
    # ------------------------------------------------------------------ #
    def _update_stats(self, level: str, message: str = "", **kwargs) -> None:
        start = time.time()
        self._stats["total"] += 1
        if level in self._stats["counts"]:
            self._stats["counts"][level] += 1

        self._performance_stats.setdefault("processing_times", [])
        entry = {"level": level, "message": message, "timestamp": datetime.now(), **kwargs}
        self._log_history.append(entry)

        duration = time.time() - start
        self._performance_stats["total_logs"] += 1
        self._performance_stats["processing_times"].append(duration)
        if len(self._performance_stats["processing_times"]) > 1000:
            self._performance_stats["processing_times"] = self._performance_stats["processing_times"][-1000:]

    def debug(self, message: str, **kwargs) -> None:
        self._update_stats("DEBUG", message, **kwargs)
        self._recorder.log(logging.DEBUG, message)
        self.logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        self._update_stats("INFO", message, **kwargs)
        self._recorder.log(logging.INFO, message)
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        self._update_stats("WARNING", message, **kwargs)
        self._recorder.log(logging.WARNING, message)
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        self._update_stats("ERROR", message, **kwargs)
        self._recorder.log(logging.ERROR, message)
        self.logger.error(message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        self._update_stats("CRITICAL", message, **kwargs)
        self._recorder.log(logging.CRITICAL, message)
        self.logger.critical(message, **kwargs)

    def log(self, level: str, message: str, **kwargs) -> None:
        level_str = str(level).upper()
        numeric = getattr(logging, level_str, logging.INFO)
        self._update_stats(level_str, message, **kwargs)
        self._recorder.log(numeric, message)
        self.logger.log(numeric, message, **kwargs)

    def get_log_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if limit is None:
            return self._log_history.copy()
        return self._log_history[-limit:] if limit > 0 else []

    def shutdown(self) -> None:
        self.clear_handlers()
        self._custom_filters.clear()


class BusinessLoggerAdapter:
    """业务日志器适配器"""

    def __init__(self, name: str, level: LogLevel, category: Optional[str] = None) -> None:
        self.name = name
        self.level = level
        self.logger = logging.getLogger(f"{name}.business")
        self.category = category or "business"

    def log_structured(self, level: str, message: str, **kwargs) -> None:
        structured = {
            "level": level,
            "message": message,
            "timestamp": time.time(),
            **kwargs,
        }
        structured.setdefault("category", self.category)
        self.logger.info(f"Structured: {structured}")

    def log_performance(self, operation: str, duration: float, **kwargs) -> None:
        perf = {
            "operation": operation,
            "duration": duration,
            "timestamp": time.time(),
            **kwargs,
        }
        self.logger.info(f"Performance: {perf}")

    def log_error_with_context(self, error: Exception, context: Dict[str, Any]) -> None:
        payload = {"error": str(error), "context": context, "timestamp": time.time()}
        self.logger.error(f"Error with context: {payload}")

    def log_business_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        payload = {"event_type": event_type, "event_data": event_data, "timestamp": time.time()}
        self.logger.info(f"Business event: {payload}")


class TestableUnifiedLogger(UnifiedLogger):
    """
    历史测试使用的可观测统一日志器，实现附加统计、过滤、历史记录等功能。
    """

    def __init__(self, name: str = "test_logger", level: Union[LogLevel, int, str] = LogLevel.INFO) -> None:
        super().__init__(name, level)
        self.config = {
            "max_history_size": 1000,
            "buffer_size": 1024,
            "flush_interval": 5.0,
            "async_logging": False,
        }
        self.log_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {
            "total_logs": 0,
            "total": 0,
            "counts": {
                "DEBUG": 0,
                "INFO": 0,
                "WARNING": 0,
                "ERROR": 0,
                "CRITICAL": 0,
            },
        }
        self.handlers: List[Any] = []
        self.filters: List[Any] = []
        self.disabled = False
        self.current_level = logging.DEBUG

    # ------------------------------------------------------------------ #
    # 重写日志方法，提供测试统计
    # ------------------------------------------------------------------ #
    def _log_with_level(self, level_name: str, message: str, **kwargs) -> None:
        if self.disabled or not self._should_log(level_name):
            return
        self.performance_metrics["total"] += 1
        self.performance_metrics["counts"][level_name] += 1
        self.performance_metrics["total_logs"] += 1
        entry = {
            "level": level_name,
            "message": message,
            "timestamp": datetime.now(),
            "extra": kwargs or {},
        }
        self._add_to_history(entry)
        self._stats["total"] = self.performance_metrics["total"]
        if level_name not in self._stats["counts"]:
            self._stats["counts"][level_name] = 0
        self._stats["counts"][level_name] = self.performance_metrics["counts"].get(level_name, 0)
        self._call_handlers(level_name, message, **kwargs)
        self._performance_stats["total_logs"] = self.performance_metrics["total_logs"]

    def debug(self, message: str, **kwargs) -> None:
        self._log_with_level("DEBUG", message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        self._log_with_level("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        self._log_with_level("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        self._log_with_level("ERROR", message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        self._log_with_level("CRITICAL", message, **kwargs)

    def exception(self, message: str, **kwargs) -> None:
        self._log_with_level("ERROR", message, **kwargs)

    # ------------------------------------------------------------------ #
    # 处理器 / 过滤器管理
    # ------------------------------------------------------------------ #
    def addHandler(self, handler: Any) -> None:
        if handler not in self.handlers:
            self.handlers.append(handler)

    def removeHandler(self, handler: Any) -> None:
        if handler in self.handlers:
            self.handlers.remove(handler)

    def addFilter(self, filter_obj: Any) -> None:
        if filter_obj not in self.filters:
            self.filters.append(filter_obj)
        super().add_filter(filter_obj)

    def removeFilter(self, filter_obj: Any) -> None:
        if filter_obj in self.filters:
            self.filters.remove(filter_obj)
        super().remove_filter(filter_obj)

    # ------------------------------------------------------------------ #
    # 历史 / 配置
    # ------------------------------------------------------------------ #
    def _add_to_history(self, entry: Dict[str, Any]) -> None:
        self.log_history.append(entry)
        max_size = self.config.get("max_history_size")
        if max_size and len(self.log_history) > max_size:
            self.log_history = self.log_history[-max_size:]

    def get_log_history(self, limit: Optional[int] = None, level: Optional[str] = None, **_) -> List[Dict[str, Any]]:
        history = self.log_history
        if level:
            history = [item for item in history if item.get("level") == level]
        if limit is not None and limit > 0:
            history = history[-limit:]
        return history.copy()

    def clear_history(self) -> None:
        self.log_history.clear()
        self._log_history.clear()
        self.performance_metrics["total"] = 0
        self.performance_metrics["total_logs"] = 0
        for key in self.performance_metrics["counts"]:
            self.performance_metrics["counts"][key] = 0

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total": self.performance_metrics["total"],
            "counts": self.performance_metrics["counts"].copy(),
        }

    def get_log_stats(self) -> Dict[str, Any]:
        return {
            "total": self.performance_metrics["total"],
            "counts": self.performance_metrics["counts"].copy(),
            "performance": {
                "total_logs": self.performance_metrics["total_logs"],
                "avg_processing_time": 0.0,
            },
        }

    def setLevel(self, level: Union[LogLevel, int, str]) -> None:
        self.set_level(level)

    def set_level(self, level: Union[LogLevel, int, str]) -> None:
        super().set_level(level)
        self.current_level = self._resolve_numeric_level(self.level)

    def _should_log(self, level_name: str) -> bool:
        level_value = getattr(logging, level_name, logging.INFO)
        return level_value >= self.current_level

    def _call_handlers(self, _level: str, message: str, **_kwargs) -> None:
        """调用注册的处理器（兼容旧测试）"""
        for handler in list(self.handlers):
            if hasattr(handler, "handle"):
                try:
                    handler.handle(message)
                except Exception:  # pragma: no cover
                    continue

    def _log(self, level: int, message: str, args: tuple, extra: Optional[Dict[str, Any]] = None) -> None:
        """兼容logging模块内部调用"""
        level_name = logging.getLevelName(level)
        if not isinstance(level_name, str):
            level_name = "INFO"
        level_name = level_name.upper()
        payload = extra or {}
        self._log_with_level(level_name, message, **payload)

    def get_logs(self, level: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取记录的日志列表（测试兼容性方法）

        Args:
            level: 可选的日志级别过滤器

        Returns:
            日志记录列表
        """
        if level is None:
            return self.log_history.copy()
        else:
            return [log for log in self.log_history if log.get('level') == level.upper()]


# 注册全局兼容别名
try:  # pragma: no cover - 兼容运行环境
    import builtins as _builtins

    if not hasattr(_builtins, "TestableUnifiedLogger"):
        setattr(_builtins, "TestableUnifiedLogger", TestableUnifiedLogger)
except Exception:
    pass

