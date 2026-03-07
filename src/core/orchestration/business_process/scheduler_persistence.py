#!/usr/bin/env python3
"""
调度器持久化模块

负责数据采集调度器状态的持久化，符合核心服务层架构设计：
- 使用基础设施层 UnifiedConfigManager 进行配置持久化
- 支持文件系统和PostgreSQL双重存储
- 确保调度器重启后能恢复采集时间状态
"""

import json
import time
from typing import Dict, Any, Optional
from datetime import datetime

from src.infrastructure.config.core.unified_manager_enhanced import UnifiedConfigManager
from src.infrastructure.logging.core.unified_logger import get_unified_logger

logger = get_unified_logger(__name__)


class SchedulerPersistence:
    """
    调度器持久化管理器
    
    负责调度器状态的持久化和恢复，符合基础设施层架构设计：
    - 使用 UnifiedConfigManager 进行配置管理
    - 支持双重存储（文件系统 + PostgreSQL）
    - 提供状态加载和保存接口
    """
    
    def __init__(self, config_manager: Optional[UnifiedConfigManager] = None):
        """
        初始化持久化管理器
        
        Args:
            config_manager: 配置管理器实例，如果为None则创建新实例
        """
        self.config_manager = config_manager or UnifiedConfigManager()
        self.config_key = "data_collection_scheduler.last_collection_times"
        self.metadata_key = "data_collection_scheduler.metadata"
        
    def load_last_collection_times(self) -> Dict[str, float]:
        """
        加载上次采集时间
        
        Returns:
            Dict[str, float]: 数据源ID到时间戳的映射
        """
        try:
            # 从配置管理器加载数据
            data = self.config_manager.get(self.config_key, default={})
            
            if isinstance(data, dict):
                # 确保所有值都是float类型
                result = {}
                for source_id, timestamp in data.items():
                    try:
                        result[source_id] = float(timestamp)
                    except (ValueError, TypeError):
                        logger.warning(f"无效的时间戳格式: {source_id}={timestamp}")
                        continue
                
                if result:
                    logger.info(f"成功加载 {len(result)} 个数据源的采集时间")
                else:
                    logger.debug("未找到历史采集时间数据")
                
                return result
            else:
                logger.warning(f"配置数据格式错误，期望dict，实际: {type(data)}")
                return {}
                
        except Exception as e:
            logger.warning(f"加载采集时间失败: {e}，使用空字典")
            return {}
    
    def save_last_collection_times(self, last_collection_times: Dict[str, float]) -> bool:
        """
        保存采集时间
        
        Args:
            last_collection_times: 数据源ID到时间戳的映射
            
        Returns:
            bool: 是否保存成功
        """
        try:
            if not last_collection_times:
                logger.debug("采集时间字典为空，跳过保存")
                return True
            
            # 保存采集时间数据
            self.config_manager.set(self.config_key, last_collection_times)
            
            # 更新元数据
            metadata = {
                "last_saved": datetime.now().isoformat(),
                "version": "1.0",
                "source_count": len(last_collection_times)
            }
            self.config_manager.set(self.metadata_key, metadata)
            
            logger.debug(f"成功保存 {len(last_collection_times)} 个数据源的采集时间")
            return True
            
        except Exception as e:
            logger.error(f"保存采集时间失败: {e}", exc_info=True)
            return False
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        获取持久化元数据
        
        Returns:
            Dict[str, Any]: 元数据信息
        """
        try:
            metadata = self.config_manager.get(self.metadata_key, default={})
            return metadata if isinstance(metadata, dict) else {}
        except Exception as e:
            logger.warning(f"获取元数据失败: {e}")
            return {}
    
    def clear_all(self) -> bool:
        """
        清除所有持久化数据（用于测试或重置）
        
        Returns:
            bool: 是否清除成功
        """
        try:
            self.config_manager.set(self.config_key, {})
            self.config_manager.set(self.metadata_key, {})
            logger.info("已清除所有调度器持久化数据")
            return True
        except Exception as e:
            logger.error(f"清除持久化数据失败: {e}")
            return False
