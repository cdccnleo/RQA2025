
import threading

from .shared_interfaces import ILogger, IErrorHandler, StandardLogger, BaseErrorHandler
from .unified_resource_interfaces import IResourceConsumer
from typing import Dict, List, Optional, Any
"""
资源消费者注册表

Phase 3: 质量提升 - 文件拆分优化

管理资源消费者的注册、注销和查询功能。
"""


class ResourceConsumerRegistry:
    """资源消费者注册表"""

    def __init__(self, logger: Optional[ILogger] = None,
                 error_handler: Optional[IErrorHandler] = None):

        self.logger = logger or StandardLogger(self.__class__.__name__)
        self.error_handler = error_handler or BaseErrorHandler()

        # 消费者存储
        self._consumers: Dict[str, IResourceConsumer] = {}
        self._lock = threading.RLock()

    def register_consumer(self, consumer: IResourceConsumer) -> bool:
        """注册资源消费者"""
        try:
            # 检查consumer是否为None
            if consumer is None:
                self.logger.log_error("无法注册None消费者")
                return False
            
            consumer_id = f"{type(consumer).__name__}_{id(consumer)}"

            with self._lock:
                if consumer_id in self._consumers:
                    raise ValueError(f"消费者 '{consumer_id}' 已存在")

                self._consumers[consumer_id] = consumer

                # 注册到依赖注入容器（如果有的话）
                # 这里可以扩展依赖注入逻辑

            self.logger.log_info(f"资源消费者已注册: {consumer_id}")
            return True

        except Exception as e:
            consumer_type = type(consumer).__name__ if consumer is not None else "None"
            self.error_handler.handle_error(e, {
                "context": "注册资源消费者失败",
                "consumer_type": consumer_type
            })
            return False

    def unregister_consumer(self, consumer_id: str) -> bool:
        """注销资源消费者"""
        with self._lock:
            try:
                if consumer_id not in self._consumers:
                    return False

                consumer = self._consumers[consumer_id]
                del self._consumers[consumer_id]

                # 从依赖注入容器中移除（如果有的话）
                # 这里可以扩展依赖注入逻辑

                self.logger.log_info(f"资源消费者已注销: {consumer_id}")
                return True

            except Exception as e:
                self.error_handler.handle_error(e, {
                    "context": "注销资源消费者失败",
                    "consumer_id": consumer_id
                })
                return False

    def get_consumer(self, consumer_id: str) -> Optional[IResourceConsumer]:
        """获取指定消费者"""
        with self._lock:
            return self._consumers.get(consumer_id)

    def get_consumers(self) -> List[IResourceConsumer]:
        """获取所有消费者"""
        with self._lock:
            return list(self._consumers.values())

    def get_consumer_ids(self) -> List[str]:
        """获取所有消费者ID"""
        with self._lock:
            return list(self._consumers.keys())

    def has_consumer(self, consumer_id: str) -> bool:
        """检查消费者是否存在"""
        with self._lock:
            return consumer_id in self._consumers

    def get_consumer_count(self) -> int:
        """获取消费者数量"""
        with self._lock:
            return len(self._consumers)

    def get_consumer_info(self, consumer_id: str) -> Optional[Dict[str, Any]]:
        """获取消费者信息"""
        consumer = self.get_consumer(consumer_id)
        if not consumer:
            return None

        try:
            consumed_resources = consumer.get_consumed_resources()
            usage = consumer.get_resource_usage()

            return {
                "consumer_id": consumer_id,
                "consumer_type": type(consumer).__name__,
                "consumed_resources_count": len(consumed_resources),
                "resource_usage": usage,
                "status": "active"
            }
        except Exception as e:
            self.error_handler.handle_error(e, {
                "context": "获取消费者信息失败",
                "consumer_id": consumer_id
            })
            return {
                "consumer_id": consumer_id,
                "status": "error",
                "error": str(e)
            }

    def get_all_consumer_info(self) -> Dict[str, Dict[str, Any]]:
        """获取所有消费者信息"""
        info = {}
        for consumer_id in self.get_consumer_ids():
            consumer_info = self.get_consumer_info(consumer_id)
            if consumer_info:
                info[consumer_id] = consumer_info
        return info

    def clear(self):
        """清空所有消费者"""
        with self._lock:
            consumer_ids = list(self._consumers.keys())
            self._consumers.clear()

            self.logger.log_info(f"已清空所有资源消费者，共 {len(consumer_ids)} 个")
