"""
数据采集完成事件处理器
订阅DATA_COLLECTION_COMPLETED事件，触发数据质量检查和后续处理
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from src.core.event_bus import get_event_bus
from src.core.event_bus.types import EventType

logger = logging.getLogger(__name__)


class DataCollectionEventHandler:
    """
    数据采集完成事件处理器
    
    订阅数据采集完成事件，触发：
    1. 数据质量检查
    2. 数据验证
    3. 质量报告生成
    """
    
    def __init__(self):
        self.event_bus = get_event_bus()
        self._subscribed = False
        
    def subscribe(self):
        """订阅数据采集完成事件"""
        if self._subscribed:
            return
            
        try:
            # 订阅数据采集完成事件
            self.event_bus.subscribe(
                EventType.DATA_COLLECTION_COMPLETED,
                self._on_data_collection_completed
            )
            self._subscribed = True
            logger.info("✅ 数据质量监控已订阅数据采集完成事件")
        except Exception as e:
            logger.error(f"❌ 订阅数据采集完成事件失败: {e}")
            
    def _on_data_collection_completed(self, event: Dict[str, Any]):
        """
        处理数据采集完成事件
        
        Args:
            event: 事件数据，包含source_id, task_id, status, result等
        """
        try:
            source_id = event.get("source_id")
            task_id = event.get("task_id")
            result = event.get("result", {})
            
            logger.info(f"🎯 数据质量监控收到数据采集完成事件: {source_id}, task_id={task_id}")
            
            # 执行数据质量检查
            self._perform_quality_check(source_id, result)
            
            # 执行数据验证
            self._perform_data_validation(source_id, result)
            
            # 生成质量报告
            self._generate_quality_report(source_id, result)
            
            logger.info(f"✅ 数据质量处理完成: {source_id}")
            
        except Exception as e:
            logger.error(f"❌ 处理数据采集完成事件失败: {e}", exc_info=True)
            
    def _perform_quality_check(self, source_id: str, result: Dict[str, Any]):
        """执行数据质量检查"""
        try:
            logger.info(f"🔍 执行数据质量检查: {source_id}")
            
            # 获取采集的数据统计
            records_collected = result.get("records_collected", 0)
            symbols_processed = result.get("symbols_processed", 0)
            
            # 检查数据完整性
            if records_collected == 0:
                logger.warning(f"⚠️ 数据源 {source_id} 采集记录数为0")
            else:
                logger.info(f"✅ 数据源 {source_id} 质量检查通过: {records_collected} 条记录")
                
        except Exception as e:
            logger.error(f"❌ 数据质量检查失败: {source_id}, 错误={e}")
            
    def _perform_data_validation(self, source_id: str, result: Dict[str, Any]):
        """执行数据验证"""
        try:
            logger.info(f"🔍 执行数据验证: {source_id}")
            # TODO: 实现具体的数据验证逻辑
            # 可以调用 validator.py 中的验证函数
            
        except Exception as e:
            logger.error(f"❌ 数据验证失败: {source_id}, 错误={e}")
            
    def _generate_quality_report(self, source_id: str, result: Dict[str, Any]):
        """生成质量报告"""
        try:
            logger.info(f"📝 生成数据质量报告: {source_id}")
            
            report = {
                "source_id": source_id,
                "timestamp": datetime.now().isoformat(),
                "records_collected": result.get("records_collected", 0),
                "symbols_processed": result.get("symbols_processed", 0),
                "status": result.get("status", "unknown"),
                "quality_score": 100 if result.get("records_collected", 0) > 0 else 0
            }
            
            logger.info(f"📊 质量报告: {report}")
            
        except Exception as e:
            logger.error(f"❌ 生成质量报告失败: {source_id}, 错误={e}")


# 全局事件处理器实例
_data_collection_event_handler: Optional[DataCollectionEventHandler] = None


def get_data_collection_event_handler() -> DataCollectionEventHandler:
    """获取数据采集完成事件处理器实例（单例模式）"""
    global _data_collection_event_handler
    if _data_collection_event_handler is None:
        _data_collection_event_handler = DataCollectionEventHandler()
        _data_collection_event_handler.subscribe()
    return _data_collection_event_handler


def init_data_collection_event_handler():
    """初始化数据采集完成事件处理器"""
    handler = get_data_collection_event_handler()
    logger.info("✅ 数据采集完成事件处理器已初始化")
    return handler
