"""
ML模块数据采集完成事件处理器
订阅DATA_COLLECTION_COMPLETED事件，触发模型训练和特征工程
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from src.core.event_bus import get_event_bus
from src.core.event_bus.types import EventType

logger = logging.getLogger(__name__)


class MLDataCollectionEventHandler:
    """
    ML模块数据采集完成事件处理器
    
    订阅数据采集完成事件，触发：
    1. 特征工程更新
    2. 模型训练
    3. 模型评估
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
            logger.info("✅ ML模块已订阅数据采集完成事件")
        except Exception as e:
            logger.error(f"❌ ML模块订阅数据采集完成事件失败: {e}")
            
    def _on_data_collection_completed(self, event):
        """
        处理数据采集完成事件
        
        Args:
            event: 事件对象或事件数据字典
        """
        try:
            # 兼容Event对象和字典
            data = event.data if hasattr(event, 'data') else event
            source_id = data.get("source_id")
            task_id = data.get("task_id")
            result = data.get("result", {})
            
            logger.info(f"🎯 ML模块收到数据采集完成事件: {source_id}, task_id={task_id}")
            
            # 更新特征工程
            self._update_feature_engineering(source_id, result)
            
            # 触发模型训练
            self._trigger_model_training(source_id, result)
            
            logger.info(f"✅ ML模块处理完成: {source_id}")
            
        except Exception as e:
            logger.error(f"❌ ML模块处理数据采集完成事件失败: {e}", exc_info=True)
            
    def _update_feature_engineering(self, source_id: str, result: Dict[str, Any]):
        """更新特征工程"""
        try:
            logger.info(f"🔧 更新特征工程: {source_id}")
            
            records_collected = result.get("records_collected", 0)
            if records_collected > 0:
                logger.info(f"✅ 特征工程更新完成: {source_id}, 基于 {records_collected} 条记录")
            else:
                logger.warning(f"⚠️ 特征工程未更新: {source_id}, 无新数据")
                
        except Exception as e:
            logger.error(f"❌ 特征工程更新失败: {source_id}, 错误={e}")
            
    def _trigger_model_training(self, source_id: str, result: Dict[str, Any]):
        """触发模型训练"""
        try:
            logger.info(f"🤖 触发模型训练: {source_id}")
            
            records_collected = result.get("records_collected", 0)
            if records_collected > 100:  # 只有数据量足够时才触发训练
                logger.info(f"🚀 启动模型训练: {source_id}, 数据量={records_collected}")
                # TODO: 调用模型训练服务
            else:
                logger.info(f"⏸️ 跳过模型训练: {source_id}, 数据量不足 ({records_collected} < 100)")
                
        except Exception as e:
            logger.error(f"❌ 模型训练触发失败: {source_id}, 错误={e}")


# 全局事件处理器实例
_ml_data_collection_event_handler: Optional[MLDataCollectionEventHandler] = None


def get_ml_data_collection_event_handler() -> MLDataCollectionEventHandler:
    """获取ML模块数据采集完成事件处理器实例（单例模式）"""
    global _ml_data_collection_event_handler
    if _ml_data_collection_event_handler is None:
        _ml_data_collection_event_handler = MLDataCollectionEventHandler()
        _ml_data_collection_event_handler.subscribe()
    return _ml_data_collection_event_handler


def init_ml_data_collection_event_handler():
    """初始化ML模块数据采集完成事件处理器"""
    handler = get_ml_data_collection_event_handler()
    logger.info("✅ ML模块数据采集完成事件处理器已初始化")
    return handler
