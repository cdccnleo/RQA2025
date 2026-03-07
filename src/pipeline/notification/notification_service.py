"""
通知服务基类模块

定义通知系统的统一接口和基础功能，支持多渠道同时发送
"""

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed


class NotificationLevel(Enum):
    """通知级别枚举"""
    DEBUG = auto()       # 调试信息
    INFO = auto()        # 一般信息
    WARNING = auto()     # 警告信息
    ERROR = auto()       # 错误信息
    CRITICAL = auto()    # 严重错误


@dataclass
class NotificationResult:
    """
    通知发送结果

    Attributes:
        channel_name: 通道名称
        success: 是否发送成功
        timestamp: 发送时间戳
        message_id: 消息ID
        error: 错误信息
        response_data: 响应数据
    """
    channel_name: str
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: Optional[str] = None
    error: Optional[str] = None
    response_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "channel_name": self.channel_name,
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
            "message_id": self.message_id,
            "error": self.error,
            "response_data": self.response_data
        }


class NotificationChannel(ABC):
    """
    通知通道抽象基类

    所有通知通道必须继承此类并实现send方法

    Attributes:
        name: 通道名称
        enabled: 是否启用
        logger: 日志记录器
    """

    def __init__(self, name: str, enabled: bool = True):
        """
        初始化通知通道

        Args:
            name: 通道名称
            enabled: 是否启用
        """
        self.name = name
        self.enabled = enabled
        self.logger = logging.getLogger(f"notification.channel.{name}")

    @abstractmethod
    def send(
        self,
        message: str,
        level: NotificationLevel,
        **kwargs: Any
    ) -> NotificationResult:
        """
        发送通知

        Args:
            message: 通知消息内容
            level: 通知级别
            **kwargs: 额外参数

        Returns:
            通知发送结果
        """
        pass

    def is_enabled_for_level(self, level: NotificationLevel) -> bool:
        """
        检查通道是否对指定级别启用

        Args:
            level: 通知级别

        Returns:
            是否启用
        """
        return self.enabled

    def validate_message(self, message: str) -> bool:
        """
        验证消息内容

        Args:
            message: 消息内容

        Returns:
            验证是否通过
        """
        if not message or not isinstance(message, str):
            self.logger.warning("消息内容不能为空且必须是字符串")
            return False
        if len(message) > 10000:  # 限制消息长度
            self.logger.warning("消息内容超过最大长度限制(10000字符)")
            return False
        return True

    def __str__(self) -> str:
        """字符串表示"""
        return f"NotificationChannel(name={self.name}, enabled={self.enabled})"

    def __repr__(self) -> str:
        """详细字符串表示"""
        return self.__str__()


class NotificationService:
    """
    通知服务类

    管理多个通知通道，支持多渠道同时发送通知

    Attributes:
        channels: 通知通道字典
        default_channels: 默认通道列表
        logger: 日志记录器
    """

    def __init__(self):
        """初始化通知服务"""
        self._channels: Dict[str, NotificationChannel] = {}
        self._default_channels: Set[str] = set()
        self.logger = logging.getLogger("notification.service")
        self._executor = ThreadPoolExecutor(max_workers=10)

    def register_channel(
        self,
        channel: NotificationChannel,
        is_default: bool = False
    ) -> None:
        """
        注册通知通道

        Args:
            channel: 通知通道实例
            is_default: 是否设为默认通道
        """
        self._channels[channel.name] = channel
        if is_default:
            self._default_channels.add(channel.name)
        self.logger.info(f"注册通知通道: {channel.name}")

    def unregister_channel(self, channel_name: str) -> bool:
        """
        注销通知通道

        Args:
            channel_name: 通道名称

        Returns:
            是否成功注销
        """
        if channel_name in self._channels:
            del self._channels[channel_name]
            self._default_channels.discard(channel_name)
            self.logger.info(f"注销通知通道: {channel_name}")
            return True
        return False

    def get_channel(self, channel_name: str) -> Optional[NotificationChannel]:
        """
        获取指定通道

        Args:
            channel_name: 通道名称

        Returns:
            通知通道实例或None
        """
        return self._channels.get(channel_name)

    def get_all_channels(self) -> List[NotificationChannel]:
        """
        获取所有已注册通道

        Returns:
            通道列表
        """
        return list(self._channels.values())

    def send(
        self,
        message: str,
        level: NotificationLevel = NotificationLevel.INFO,
        channels: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Dict[str, NotificationResult]:
        """
        发送通知到指定通道

        Args:
            message: 通知消息内容
            level: 通知级别
            channels: 通道名称列表，None则使用默认通道
            **kwargs: 额外参数传递给通道

        Returns:
            各通道发送结果字典
        """
        # 确定目标通道
        target_channels = channels if channels else list(self._default_channels)
        if not target_channels:
            self.logger.warning("没有指定通知通道")
            return {}

        results: Dict[str, NotificationResult] = {}

        for channel_name in target_channels:
            channel = self._channels.get(channel_name)
            if not channel:
                self.logger.warning(f"通道不存在: {channel_name}")
                results[channel_name] = NotificationResult(
                    channel_name=channel_name,
                    success=False,
                    error="通道未注册"
                )
                continue

            if not channel.enabled:
                self.logger.debug(f"通道已禁用: {channel_name}")
                results[channel_name] = NotificationResult(
                    channel_name=channel_name,
                    success=False,
                    error="通道已禁用"
                )
                continue

            if not channel.is_enabled_for_level(level):
                self.logger.debug(f"通道 {channel_name} 不处理级别 {level.name}")
                continue

            try:
                result = channel.send(message, level, **kwargs)
                results[channel_name] = result
                if result.success:
                    self.logger.info(f"通知已通过 {channel_name} 发送成功")
                else:
                    self.logger.warning(f"通知通过 {channel_name} 发送失败: {result.error}")
            except Exception as e:
                self.logger.error(f"通知通过 {channel_name} 发送异常: {e}")
                results[channel_name] = NotificationResult(
                    channel_name=channel_name,
                    success=False,
                    error=str(e)
                )

        return results

    def send_async(
        self,
        message: str,
        level: NotificationLevel = NotificationLevel.INFO,
        channels: Optional[List[str]] = None,
        **kwargs: Any
    ) -> asyncio.Future:
        """
        异步发送通知

        Args:
            message: 通知消息内容
            level: 通知级别
            channels: 通道名称列表
            **kwargs: 额外参数

        Returns:
            Future对象
        """
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(
            self._executor,
            self.send,
            message,
            level,
            channels,
            **kwargs
        )

    def send_parallel(
        self,
        message: str,
        level: NotificationLevel = NotificationLevel.INFO,
        channels: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Dict[str, NotificationResult]:
        """
        并行发送通知到多个通道

        Args:
            message: 通知消息内容
            level: 通知级别
            channels: 通道名称列表
            **kwargs: 额外参数

        Returns:
            各通道发送结果字典
        """
        target_channels = channels if channels else list(self._default_channels)
        if not target_channels:
            self.logger.warning("没有指定通知通道")
            return {}

        results: Dict[str, NotificationResult] = {}
        futures = {}

        with ThreadPoolExecutor(max_workers=len(target_channels)) as executor:
            for channel_name in target_channels:
                channel = self._channels.get(channel_name)
                if not channel or not channel.enabled:
                    continue

                future = executor.submit(
                    self._send_to_channel,
                    channel,
                    message,
                    level,
                    **kwargs
                )
                futures[future] = channel_name

            for future in as_completed(futures):
                channel_name = futures[future]
                try:
                    result = future.result()
                    results[channel_name] = result
                except Exception as e:
                    self.logger.error(f"并行发送通知到 {channel_name} 失败: {e}")
                    results[channel_name] = NotificationResult(
                        channel_name=channel_name,
                        success=False,
                        error=str(e)
                    )

        return results

    def _send_to_channel(
        self,
        channel: NotificationChannel,
        message: str,
        level: NotificationLevel,
        **kwargs: Any
    ) -> NotificationResult:
        """
        内部方法：发送通知到单个通道

        Args:
            channel: 通知通道
            message: 消息内容
            level: 通知级别
            **kwargs: 额外参数

        Returns:
            发送结果
        """
        try:
            return channel.send(message, level, **kwargs)
        except Exception as e:
            self.logger.error(f"发送通知到 {channel.name} 失败: {e}")
            return NotificationResult(
                channel_name=channel.name,
                success=False,
                error=str(e)
            )

    def broadcast(
        self,
        message: str,
        level: NotificationLevel = NotificationLevel.INFO,
        **kwargs: Any
    ) -> Dict[str, NotificationResult]:
        """
        广播通知到所有启用的通道

        Args:
            message: 通知消息内容
            level: 通知级别
            **kwargs: 额外参数

        Returns:
            各通道发送结果字典
        """
        all_channels = [
            name for name, ch in self._channels.items()
            if ch.enabled
        ]
        return self.send(message, level, all_channels, **kwargs)

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取通知服务统计信息

        Returns:
            统计信息字典
        """
        return {
            "total_channels": len(self._channels),
            "enabled_channels": sum(1 for ch in self._channels.values() if ch.enabled),
            "default_channels": list(self._default_channels),
            "channel_names": list(self._channels.keys())
        }

    def shutdown(self) -> None:
        """关闭通知服务，释放资源"""
        self.logger.info("关闭通知服务")
        self._executor.shutdown(wait=True)
