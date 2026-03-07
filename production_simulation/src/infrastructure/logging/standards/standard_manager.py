"""
standard_manager 模块

提供 standard_manager 相关功能和接口。
"""

import json
import os

import asyncio

from ..core.interfaces import LogLevel, LogCategory
from .base_standard import StandardFormatType, StandardLogEntry
from .standard_formatter import StandardFormatter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
"""
RQA2025 基础设施层 - 标准格式管理器

管理多种日志分析平台的标准格式转换和输出。
"""


@dataclass
class StandardOutputConfig:
    """标准输出配置"""
    format_type: StandardFormatType
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    batch_size: int = 100
    timeout: float = 30.0
    retry_attempts: int = 3
    compression: bool = False
    async_mode: bool = False


class StandardFormatManager:
    """标准格式管理器"""

    def __init__(self):
        self.formatter = StandardFormatter()
        self.configs: Dict[str, StandardOutputConfig] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)

    def register_config(self, name: str, config: StandardOutputConfig):
        """
        注册输出配置

        Args:
            name: 配置名称
            config: 输出配置
        """
        self.configs[name] = config

    def unregister_config(self, name: str):
        """
        注销输出配置

        Args:
            name: 配置名称
        """
        self.configs.pop(name, None)

    def get_config(self, name: str) -> Optional[StandardOutputConfig]:
        """
        获取输出配置

        Args:
            name: 配置名称

        Returns:
            输出配置或None
        """
        return self.configs.get(name)

    def format_for_target(
        self,
        entry: StandardLogEntry,
        target: str
    ) -> Union[str, Dict[str, Any]]:
        """
        为指定目标格式化日志条目

        Args:
            entry: 标准日志条目
            target: 目标配置名称

        Returns:
            格式化后的数据
        """
        config = self.get_config(target)
        if not config:
            raise ValueError(f"未找到目标配置: {target}")

        return self.formatter.format_log_entry(entry, config.format_type)

    def format_batch_for_target(
        self,
        entries: List[StandardLogEntry],
        target: str
    ) -> Union[str, List[Dict[str, Any]]]:
        """
        为指定目标批量格式化日志条目

        Args:
            entries: 日志条目列表
            target: 目标配置名称

        Returns:
            批量格式化结果
        """
        config = self.get_config(target)
        if not config:
            raise ValueError(f"未找到目标配置: {target}")

        return self.formatter.format_batch(entries, config.format_type)

    async def send_to_target(self, entries: List[StandardLogEntry], target: str) -> Dict[str, Any]:
        """
        发送日志条目到指定目标

        Args:
            entries: 日志条目列表
            target: 目标配置名称

        Returns:
            发送结果
        """
        config = self.get_config(target)
        if not config:
            raise ValueError(f"未找到目标配置: {target}")

        if not config.endpoint:
            raise ValueError(f"目标 {target} 没有配置端点")

        formatted_data = self.format_batch_for_target(entries, target)

        # 这里应该实现实际的HTTP发送逻辑
        # 为演示目的，我们只返回模拟结果
        return await self._mock_send_to_endpoint(config, formatted_data)

    def send_batch_sync(self, entries: List[StandardLogEntry], target: str) -> Dict[str, Any]:
        """
        同步发送日志批次

        Args:
            entries: 日志条目列表
            target: 目标配置名称

        Returns:
            发送结果
        """
        config = self.get_config(target)
        if not config:
            raise ValueError(f"未找到目标配置: {target}")

        formatted_data = self.format_batch_for_target(entries, target)

        # 模拟同步发送
        return self._mock_send_sync(config, formatted_data)

    async def send_batch_async(self, entries: List[StandardLogEntry], target: str) -> Dict[str, Any]:
        """
        异步发送日志批次

        Args:
            entries: 日志条目列表
            target: 目标配置名称

        Returns:
            发送结果
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.send_batch_sync,
            entries,
            target
        )

    def create_batch_processor(self, target: str, batch_size: int = 100):
        """
        创建批量处理器

        Args:
            target: 目标配置名称
            batch_size: 批大小

        Returns:
            批量处理器函数
        """
        config = self.get_config(target)
        if not config:
            raise ValueError(f"未找到目标配置: {target}")

        def process_batch(entries: List[StandardLogEntry]) -> Dict[str, Any]:
            if len(entries) >= batch_size:
                return self.send_batch_sync(entries, target)
            return {"status": "pending", "count": len(entries)}

        return process_batch

    def get_supported_targets(self) -> List[str]:
        """
        获取所有支持的目标

        Returns:
            目标配置名称列表
        """
        return list(self.configs.keys())

    def get_target_info(self, target: str) -> Dict[str, Any]:
        """
        获取目标信息

        Args:
            target: 目标配置名称

        Returns:
            目标信息字典
        """
        config = self.get_config(target)
        if not config:
            raise ValueError(f"未找到目标配置: {target}")

        return {
            "format_type": config.format_type.value,
            "endpoint": config.endpoint,
            "batch_size": config.batch_size,
            "supports_batch": self.formatter.supports_batch(config.format_type),
            "content_type": self.formatter.get_content_type(config.format_type),
            "timeout": config.timeout,
            "compression": config.compression,
            "async_mode": config.async_mode
        }

    async def _mock_send_to_endpoint(self, config: StandardOutputConfig, data: Any) -> Dict[str, Any]:
        """
        模拟发送到端点（实际实现中应该使用aiohttp等）

        Args:
            config: 输出配置
            data: 要发送的数据

        Returns:
            发送结果
        """
        # 模拟网络延迟
        await asyncio.sleep(0.1)

        return {
            "status": "success",
            "target": config.format_type.value,
            "endpoint": config.endpoint,
            "data_size": len(str(data)) if isinstance(data, (str, dict)) else len(data),
            "timestamp": datetime.now().isoformat(),
            "response": {"status_code": 200, "message": "OK"}
        }

    def _mock_send_sync(self, config: StandardOutputConfig, data: Any) -> Dict[str, Any]:
        """
        模拟同步发送

        Args:
            config: 输出配置
            data: 要发送的数据

        Returns:
            发送结果
        """
        return {
            "status": "success",
            "target": config.format_type.value,
            "endpoint": config.endpoint,
            "data_size": len(str(data)) if isinstance(data, (str, dict)) else len(data),
            "timestamp": datetime.now().isoformat(),
            "response": {"status_code": 200, "message": "OK"}
        }

    def create_sample_configs(self) -> Dict[str, StandardOutputConfig]:
        """
        创建示例配置

        Returns:
            示例配置字典
        """
        return {
            "elk-dev": StandardOutputConfig(
                format_type=StandardFormatType.ELK,
                endpoint="http://localhost:9200/_bulk",
                batch_size=50
            ),
            "splunk-prod": StandardOutputConfig(
                format_type=StandardFormatType.SPLUNK,
                endpoint="https://splunk-hec.example.com:8088/services/collector",
                api_key=os.getenv("API_KEY", ""),
                batch_size=100
            ),
            "datadog-staging": StandardOutputConfig(
                format_type=StandardFormatType.DATADOG,
                endpoint=os.getenv("DATADOG_ENDPOINT", ""),
                api_key=os.getenv("DATADOG_API_KEY", ""),
                batch_size=200
            ),
            "loki-monitoring": StandardOutputConfig(
                format_type=StandardFormatType.LOKI,
                endpoint="http://loki.example.com:3100/loki/api/v1/push",
                batch_size=150
            ),
            "graylog-logging": StandardOutputConfig(
                format_type=StandardFormatType.GRAYLOG,
                endpoint="http://graylog.example.com:12201/gelf",
                batch_size=75
            )
        }
