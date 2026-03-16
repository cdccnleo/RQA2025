import logging
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
事件监听器

提供事件监听功能，用于监听数据采集完成事件并自动触发特征提取任务。
"""

import time
from typing import Dict, Any


logger = logging.getLogger(__name__)


class FeatureEventListeners:
    """
    特征层事件监听器
    """

    def __init__(self):
        """
        初始化事件监听器
        """
        self.event_bus = None
        self.scheduler = None

    def initialize(self, event_bus, scheduler):
        """
        初始化事件监听器

        Args:
            event_bus: 事件总线实例
            scheduler: 任务调度器实例
        """
        self.event_bus = event_bus
        self.scheduler = scheduler

        # 注册事件监听器
        self._register_event_listeners()
        logger.info("特征层事件监听器已初始化")

    def _register_event_listeners(self):
        """
        注册事件监听器
        """
        logger.info(f"🔧 开始注册特征工程事件监听器... EventBus: {self.event_bus is not None}")
        
        if not self.event_bus:
            logger.warning("事件总线未初始化，无法注册监听器")
            return

        # 导入事件类型
        try:
            from src.core.event_bus.types import EventType
            logger.info(f"✅ 事件类型导入成功，DATA_COLLECTION_COMPLETED: {hasattr(EventType, 'DATA_COLLECTION_COMPLETED')}")
        except Exception as e:
            logger.error(f"❌ 导入事件类型失败: {e}")
            return

        # 注册数据采集完成事件监听器
        try:
            def handle_data_collection_completed(event):
                logger.info(f"📥 特征工程收到数据采集完成事件: {event}")
                self._handle_data_collection_completed(event)

            # 优先使用DATA_COLLECTION_COMPLETED事件，如果不存在则使用DATA_COLLECTED
            if hasattr(EventType, 'DATA_COLLECTION_COMPLETED'):
                self.event_bus.subscribe(
                    EventType.DATA_COLLECTION_COMPLETED,
                    handle_data_collection_completed
                )
                logger.info("✅ 已注册 DATA_COLLECTION_COMPLETED 事件监听器")
            else:
                # 使用现有的DATA_COLLECTED事件
                self.event_bus.subscribe(
                    EventType.DATA_COLLECTED,
                    handle_data_collection_completed
                )
                logger.info("✅ 已注册 DATA_COLLECTED 事件监听器")
        except Exception as e:
            logger.error(f"❌ 注册数据采集完成事件监听器失败: {e}", exc_info=True)

        # 注册数据采集开始事件监听器
        try:
            def handle_data_collection_started(event):
                self._handle_data_collection_started(event)

            self.event_bus.subscribe(
                EventType.DATA_COLLECTION_STARTED,
                handle_data_collection_started
            )
            logger.info("已注册数据采集开始事件监听器")
        except Exception as e:
            logger.error(f"注册数据采集开始事件监听器失败: {e}")

    def _handle_data_collection_completed(self, event):
        """
        处理数据采集完成事件

        Args:
            event: 事件对象
        """
        logger.info(f"🎯 开始处理数据采集完成事件: {event}")
        try:
            data = event.data if hasattr(event, 'data') else event
            source_id = data.get("source_id")
            source_config = data.get("source_config")

            logger.info(f"📊 收到数据采集完成事件，数据源: {source_id}, 配置: {source_config is not None}")

            if not source_id:
                logger.warning("⚠️ 事件数据中缺少 source_id")
                return

            # 自动创建特征提取任务
            logger.info(f"🔧 准备为数据源 {source_id} 创建特征提取任务...")
            self._create_feature_task(source_id, source_config)
            logger.info(f"✅ 数据源 {source_id} 的特征提取任务处理完成")

        except Exception as e:
            logger.error(f"❌ 处理数据采集完成事件失败: {e}", exc_info=True)

    def _handle_data_collection_started(self, event):
        """
        处理数据采集开始事件

        Args:
            event: 事件对象
        """
        try:
            data = event.data if hasattr(event, 'data') else event
            source_id = data.get("source_id")

            logger.info(f"收到数据采集开始事件，数据源: {source_id}")

        except Exception as e:
            logger.error(f"处理数据采集开始事件失败: {e}")

    def _create_feature_task(self, source_id: str, source_config: Dict[str, Any]):
        """
        创建特征提取任务 - 为每只股票创建独立的任务
        
        包含去重逻辑：如果相同股票已有未完成的任务，则跳过创建

        Args:
            source_id: 数据源ID
            source_config: 数据源配置
        """
        try:
            # 确定任务类型 - 使用统一的task_type名称
            task_type = "feature_extraction"  # 默认任务类型

            # 根据数据源类型确定任务类型
            if source_config:
                data_type = source_config.get("data_type", "")
                if "sentiment" in data_type.lower():
                    task_type = "feature_extraction"  # 情感特征也使用feature_extraction
                elif "statistical" in data_type.lower():
                    task_type = "feature_extraction"  # 统计特征也使用feature_extraction

            # 获取股票列表 - 从嵌套的config字段中获取
            config = source_config.get("config", {}) if source_config else {}
            custom_stocks = config.get("custom_stocks", []) if config else []
            
            if not custom_stocks:
                logger.warning(f"数据源 {source_id} 没有配置股票列表，无法创建特征提取任务")
                return

            # 去重检查：查询数据库中是否已有相同股票的未完成任务
            try:
                from src.gateway.web.feature_task_persistence import list_feature_tasks
                existing_tasks = list_feature_tasks(status='submitted', limit=1000) + \
                                list_feature_tasks(status='running', limit=1000)
                
                # 构建已存在任务的股票代码集合
                existing_stocks = set()
                for task in existing_tasks:
                    task_config = task.get('config', {})
                    stock_code = task_config.get('stock_code') or task_config.get('symbol')
                    if stock_code:
                        existing_stocks.add(stock_code)
                
                logger.info(f"数据源 {source_id}: 发现 {len(existing_stocks)} 只股票已有未完成的特征提取任务")
            except Exception as e:
                logger.warning(f"查询现有任务失败，继续创建新任务: {e}")
                existing_stocks = set()

            logger.info(f"为数据源 {source_id} 的 {len(custom_stocks)} 只股票创建特征提取任务，类型: {task_type}")

            # 获取日期范围 - 增加数据量以确保技术指标计算有足够的数据点
            # SMA/BOLL等指标需要20个数据点作为窗口期，考虑到周末和节假日，至少需要60天的数据
            # 强制使用90天，不使用config中的default_days（避免配置中的30天限制）
            default_days = 90  # 强制使用90天
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=default_days)).strftime("%Y-%m-%d")

            # 为每只股票创建独立的特征提取任务
            created_tasks = []
            skipped_tasks = []
            for stock in custom_stocks:
                stock_code = stock.get("code")
                stock_name = stock.get("name", "")
                
                if not stock_code:
                    logger.warning(f"股票数据缺少代码，跳过: {stock}")
                    continue
                
                # 去重检查：如果该股票已有未完成的任务，则跳过
                if stock_code in existing_stocks:
                    logger.info(f"股票 {stock_code} ({stock_name}) 已有未完成的特征提取任务，跳过创建")
                    skipped_tasks.append(stock_code)
                    continue

                # 构建单个股票的任务配置
                task_config = {
                    "symbol": stock_code,
                    "stock_code": stock_code,
                    "stock_name": stock_name,
                    "start_date": start_date,
                    "end_date": end_date,
                    "indicators": ["SMA", "EMA", "RSI", "MACD", "KDJ", "BOLL"],
                    "data_source": source_id,
                    "collection_time": time.time()
                }

                try:
                    from src.gateway.web.feature_engineering_service import create_feature_task
                    task = create_feature_task(task_type, task_config)
                    task_id = task.get('task_id')
                    
                    if task_id:
                        created_tasks.append({
                            'task_id': task_id,
                            'stock_code': stock_code,
                            'stock_name': stock_name
                        })
                        logger.info(f"✅ 股票 {stock_code} ({stock_name}) 的特征提取任务已创建，任务ID: {task_id}")
                        
                        # 如果调度器可用，同时提交到调度器进行调度执行
                        if self.scheduler:
                            try:
                                import asyncio
                                payload = {
                                    "task_config": task_config,
                                    "metadata": {
                                        "source_id": source_id,
                                        "stock_code": stock_code,
                                        "stock_name": stock_name,
                                        "created_from_event": True,
                                        "feature_task_id": task_id
                                    }
                                }
                                scheduler_task_id = asyncio.run(self.scheduler.submit_task(
                                    task_type=task_type,
                                    payload=payload,
                                    priority=5
                                ))
                                logger.debug(f"股票 {stock_code} 的任务已提交到调度器，调度器任务ID: {scheduler_task_id}")
                            except Exception as e:
                                logger.warning(f"⚠️ 股票 {stock_code} 提交到调度器失败（不影响数据库持久化）: {e}")
                except Exception as e:
                    logger.error(f"❌ 为股票 {stock_code} 创建特征提取任务失败: {e}")

            logger.info(f"✅ 数据源 {source_id} 的特征提取任务处理完成: "
                       f"创建 {len(created_tasks)} 个新任务, "
                       f"跳过 {len(skipped_tasks)} 个已有未完成任务")

        except Exception as e:
            logger.error(f"创建特征提取任务失败: {e}", exc_info=True)


# 全局事件监听器实例
_feature_event_listeners = None


def get_feature_event_listeners() -> FeatureEventListeners:
    """
    获取全局事件监听器实例

    Returns:
        FeatureEventListeners实例
    """
    global _feature_event_listeners
    if _feature_event_listeners is None:
        _feature_event_listeners = FeatureEventListeners()
    return _feature_event_listeners


def initialize_event_listeners(event_bus, scheduler):
    """
    初始化事件监听器

    Args:
        event_bus: 事件总线实例
        scheduler: 任务调度器实例
    """
    listeners = get_feature_event_listeners()
    listeners.initialize(event_bus, scheduler)
    return listeners
