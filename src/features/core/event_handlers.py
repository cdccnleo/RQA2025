import logging
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
事件处理器

提供完善的事件处理逻辑，包括事件过滤、任务优先级设置等功能。
"""

import time
from typing import Dict, Any, Callable, Optional
from datetime import datetime


logger = logging.getLogger(__name__)


class FeatureEventHandler:
    """
    特征层事件处理器
    """

    def __init__(self):
        """
        初始化事件处理器
        """
        self.event_filters = {}
        self.priority_rules = {}
        self.event_history = []
        self.max_history_size = 1000

    def register_event_filter(self, event_type: str, filter_func: Callable) -> bool:
        """
        注册事件过滤器

        Args:
            event_type: 事件类型
            filter_func: 过滤函数，返回True表示事件应该被处理，False表示应该被忽略

        Returns:
            是否注册成功
        """
        try:
            self.event_filters[event_type] = filter_func
            logger.info(f"已注册事件过滤器: {event_type}")
            return True
        except Exception as e:
            logger.error(f"注册事件过滤器失败: {e}")
            return False

    def register_priority_rule(self, event_type: str, priority_func: Callable) -> bool:
        """
        注册事件优先级规则

        Args:
            event_type: 事件类型
            priority_func: 优先级计算函数，返回优先级值

        Returns:
            是否注册成功
        """
        try:
            self.priority_rules[event_type] = priority_func
            logger.info(f"已注册事件优先级规则: {event_type}")
            return True
        except Exception as e:
            logger.error(f"注册事件优先级规则失败: {e}")
            return False

    def should_handle_event(self, event_type: str, event_data: Dict[str, Any]) -> bool:
        """
        判断是否应该处理事件

        Args:
            event_type: 事件类型
            event_data: 事件数据

        Returns:
            是否应该处理
        """
        # 检查是否有过滤器
        if event_type in self.event_filters:
            try:
                return self.event_filters[event_type](event_data)
            except Exception as e:
                logger.error(f"执行事件过滤器失败: {e}")
                return True  # 过滤器失败时默认处理事件

        # 默认处理所有事件
        return True

    def get_event_priority(self, event_type: str, event_data: Dict[str, Any]) -> int:
        """
        获取事件优先级

        Args:
            event_type: 事件类型
            event_data: 事件数据

        Returns:
            优先级值，数值越小优先级越高
        """
        # 检查是否有优先级规则
        if event_type in self.priority_rules:
            try:
                return self.priority_rules[event_type](event_data)
            except Exception as e:
                logger.error(f"执行事件优先级规则失败: {e}")

        # 默认优先级
        default_priorities = {
            "DATA_COLLECTION_COMPLETED": 1,
            "DATA_COLLECTED": 1,
            "FEATURE_EXTRACTION_STARTED": 2,
            "FEATURES_EXTRACTED": 2,
            "DEFAULT": 3
        }

        return default_priorities.get(event_type, default_priorities["DEFAULT"])

    def process_event(self, event_type: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理事件

        Args:
            event_type: 事件类型
            event_data: 事件数据

        Returns:
            处理结果
        """
        # 记录事件
        self._record_event(event_type, event_data)

        # 检查是否应该处理
        if not self.should_handle_event(event_type, event_data):
            logger.info(f"事件被过滤: {event_type}")
            return {
                "status": "filtered",
                "event_type": event_type
            }

        # 获取优先级
        priority = self.get_event_priority(event_type, event_data)
        logger.info(f"处理事件: {event_type}, 优先级: {priority}")

        # 根据事件类型处理
        if event_type in ["DATA_COLLECTION_COMPLETED", "DATA_COLLECTED"]:
            return self._handle_data_collection_event(event_type, event_data)
        elif event_type in ["FEATURE_EXTRACTION_STARTED", "FEATURES_EXTRACTED"]:
            return self._handle_feature_extraction_event(event_type, event_data)
        else:
            return self._handle_generic_event(event_type, event_data)

    def _handle_data_collection_event(self, event_type: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理数据采集相关事件

        Args:
            event_type: 事件类型
            event_data: 事件数据

        Returns:
            处理结果
        """
        try:
            source_id = event_data.get("source_id")
            source_config = event_data.get("source_config")

            logger.info(f"处理数据采集事件: {event_type}, 数据源: {source_id}")

            # 构建任务配置
            task_config = {
                "data_source": source_id,
                "source_config": source_config,
                "collection_time": event_data.get("timestamp", time.time()),
                "event_type": event_type
            }

            # 确定任务类型和优先级
            task_type, task_priority = self._determine_task_info(source_config)

            result = {
                "status": "processed",
                "event_type": event_type,
                "source_id": source_id,
                "task_type": task_type,
                "task_priority": task_priority,
                "task_config": task_config,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"数据采集事件处理完成: {source_id}, 任务类型: {task_type}")
            return result

        except Exception as e:
            logger.error(f"处理数据采集事件失败: {e}")
            return {
                "status": "error",
                "event_type": event_type,
                "error": str(e)
            }

    def _handle_feature_extraction_event(self, event_type: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理特征提取相关事件

        Args:
            event_type: 事件类型
            event_data: 事件数据

        Returns:
            处理结果
        """
        try:
            task_id = event_data.get("task_id")
            feature_count = event_data.get("feature_count", 0)

            logger.info(f"处理特征提取事件: {event_type}, 任务ID: {task_id}")

            result = {
                "status": "processed",
                "event_type": event_type,
                "task_id": task_id,
                "feature_count": feature_count,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"特征提取事件处理完成: {task_id}, 提取特征数: {feature_count}")
            return result

        except Exception as e:
            logger.error(f"处理特征提取事件失败: {e}")
            return {
                "status": "error",
                "event_type": event_type,
                "error": str(e)
            }

    def _handle_generic_event(self, event_type: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理通用事件

        Args:
            event_type: 事件类型
            event_data: 事件数据

        Returns:
            处理结果
        """
        try:
            logger.info(f"处理通用事件: {event_type}")

            result = {
                "status": "processed",
                "event_type": event_type,
                "event_data": event_data,
                "timestamp": datetime.now().isoformat()
            }

            return result

        except Exception as e:
            logger.error(f"处理通用事件失败: {e}")
            return {
                "status": "error",
                "event_type": event_type,
                "error": str(e)
            }

    def _determine_task_info(self, source_config: Optional[Dict[str, Any]]) -> tuple:
        """
        确定任务类型和优先级

        Args:
            source_config: 数据源配置

        Returns:
            (task_type, priority)
        """
        # 默认值
        task_type = "技术指标"
        priority = 2  # 中优先级

        # 根据数据源配置确定
        if source_config:
            data_type = source_config.get("data_type", "")
            if "sentiment" in data_type.lower():
                task_type = "情感特征"
                priority = 1  # 高优先级
            elif "statistical" in data_type.lower():
                task_type = "统计特征"
                priority = 2  # 中优先级
            elif "technical" in data_type.lower():
                task_type = "技术指标"
                priority = 2  # 中优先级
            elif "custom" in data_type.lower():
                task_type = "自定义特征"
                priority = 3  # 低优先级

        return task_type, priority

    def _record_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        记录事件到历史

        Args:
            event_type: 事件类型
            event_data: 事件数据
        """
        event_record = {
            "event_type": event_type,
            "event_data": event_data,
            "timestamp": datetime.now().isoformat()
        }

        self.event_history.append(event_record)

        # 限制历史大小
        if len(self.event_history) > self.max_history_size:
            self.event_history = self.event_history[-self.max_history_size:]

    def get_event_history(self, limit: int = 100) -> list:
        """
        获取事件历史

        Args:
            limit: 返回的历史记录数量

        Returns:
            事件历史记录
        """
        return self.event_history[-limit:]

    def clear_event_history(self) -> bool:
        """
        清空事件历史

        Returns:
            是否清空成功
        """
        try:
            self.event_history.clear()
            logger.info("事件历史已清空")
            return True
        except Exception as e:
            logger.error(f"清空事件历史失败: {e}")
            return False


# 全局事件处理器实例
_feature_event_handler = None


def get_feature_event_handler() -> FeatureEventHandler:
    """
    获取全局事件处理器实例

    Returns:
        FeatureEventHandler实例
    """
    global _feature_event_handler
    if _feature_event_handler is None:
        _feature_event_handler = FeatureEventHandler()
    return _feature_event_handler


def create_default_event_handler() -> FeatureEventHandler:
    """
    创建默认事件处理器

    Returns:
        配置好默认规则的FeatureEventHandler实例
    """
    handler = get_feature_event_handler()

    # 注册默认事件过滤器
    def data_collection_filter(event_data):
        """数据采集事件过滤器"""
        # 只处理成功的采集事件
        return event_data.get("status", "success") == "success"

    handler.register_event_filter("DATA_COLLECTION_COMPLETED", data_collection_filter)
    handler.register_event_filter("DATA_COLLECTED", data_collection_filter)

    # 注册默认优先级规则
    def data_collection_priority(event_data):
        """数据采集事件优先级规则"""
        # 基于数据源重要性设置优先级
        source_importance = event_data.get("source_importance", 3)
        return source_importance

    handler.register_priority_rule("DATA_COLLECTION_COMPLETED", data_collection_priority)
    handler.register_priority_rule("DATA_COLLECTED", data_collection_priority)

    logger.info("默认事件处理器已创建并配置")
    return handler
