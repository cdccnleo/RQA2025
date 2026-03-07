"""
TradingLogger 兼容实现

历史版本的 TradingLogger 提供较重的异步写入与安全签名能力。
为了配合当前测试与业务场景，我们保留相同接口，但以轻量级同步实现
来保持兼容：支持以字符串或配置字典初始化，并允许多次创建实例。
"""

from __future__ import annotations

import datetime
import hashlib
import hmac
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union


def _ensure_level(level: Any) -> int:
    """解析日志级别为 logging 所需的整数值。"""
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        return getattr(logging, level.upper(), logging.INFO)
    return logging.INFO


@dataclass
class SecureLogger:
    """简化的日志签名器，用于兼容旧接口。"""

    secret_key: str = "default_secret"

    def sign_log(self, log_entry: Dict[str, Any]) -> str:
        payload = f"{log_entry.get('timestamp')}{log_entry.get('message')}".encode()
        return hmac.new(self.secret_key.encode(), payload, hashlib.sha256).hexdigest()


class _DummyAsyncWriter:
    """占位异步写入器，仅保留旧属性。"""

    async def start(self) -> None:  # pragma: no cover - 占位实现
        return None

    async def stop(self) -> None:  # pragma: no cover - 占位实现
        return None


class TradingLogger:
    """轻量级交易日志器，实现向后兼容 API。"""

    DEFAULT_CONFIG: Dict[str, Any] = {
        "name": "trading",
        "base_level": logging.INFO,
        "log_dir": "logs",
        "rotation": "midnight",
        "backup_count": 7,
        "max_workers": 4,
        "security": {"secret_key": "default_secret"},
    }

    def __init__(self, config: Union[str, Dict[str, Any], None] = None) -> None:
        self.config = self._merge_config(config)
        self.name = self.config["name"]

        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(_ensure_level(self.config["base_level"]))
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
            )
            self.logger.addHandler(handler)

        self.async_writer = _DummyAsyncWriter()
        self.secure_logger = SecureLogger(self.config["security"]["secret_key"])

    # ------------------------------------------------------------------
    # 兼容性 API
    # ------------------------------------------------------------------
    def log(self, level: str, message: str, **kwargs) -> None:
        record = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "message": message,
        }
        signature = self.secure_logger.sign_log(record)
        extra = {"signature": signature, **kwargs}
        self.logger.log(_ensure_level(level), message, extra=extra)

    def info(self, message: str, **kwargs) -> None:
        self.log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        self.log("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        self.log("ERROR", message, **kwargs)

    def debug(self, message: str, **kwargs) -> None:
        self.log("DEBUG", message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        self.log("CRITICAL", message, **kwargs)

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    @classmethod
    async def initialize(cls, config: Optional[Dict[str, Any]] = None) -> "TradingLogger":
        instance = cls(config)
        await instance.async_writer.start()
        return instance

    async def shutdown(self) -> None:
        await self.async_writer.stop()

    @classmethod
    def _merge_config(cls, config: Union[str, Dict[str, Any], None]) -> Dict[str, Any]:
        if config is None:
            return cls.DEFAULT_CONFIG.copy()
        if isinstance(config, str):
            merged = cls.DEFAULT_CONFIG.copy()
            merged["name"] = config
            return merged

        merged = cls.DEFAULT_CONFIG.copy()
        merged.update({k: v for k, v in config.items() if k != "security"})
        security = cls.DEFAULT_CONFIG["security"].copy()
        security.update(config.get("security", {}))
        merged["security"] = security
        return merged

