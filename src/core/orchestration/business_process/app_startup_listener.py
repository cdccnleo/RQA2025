#!/usr/bin/env python3
"""
应用启动监听器

符合核心服务层架构设计，通过事件驱动方式启动后台服务：
- 监听应用启动事件
- 自动启动数据采集调度器等后台服务
- 确保后台服务启动失败不影响API服务

职责：
- 统一管理后台服务启动
- 提供优雅的错误处理
- 符合事件驱动架构设计
"""

import asyncio
import logging
from typing import Optional

from src.core.event_bus.core import EventBus
from src.core.event_bus.types import EventType
from src.core.event_bus.models import Event
from src.infrastructure.logging.core.unified_logger import get_unified_logger

logger = get_unified_logger(__name__)

# 全局监听器实例
_app_startup_listener: Optional['AppStartupListener'] = None


class AppStartupListener:
    """
    应用启动监听器
    
    监听应用启动事件，自动启动后台服务（如数据采集调度器）
    符合核心服务层架构设计：事件驱动 + 服务治理
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        初始化应用启动监听器
        
        Args:
            event_bus: 事件总线实例，如果为None则创建新实例
        """
        self.event_bus = event_bus
        self._registered = False
        self._scheduler_started = False
        self._historical_scheduler_started = False
        self._fallback_task = None  # 降级启动任务
        self._startup_timeout = 10.0  # 启动超时时间（秒）
        
    def register(self, event_bus: Optional[EventBus] = None):
        """
        注册监听器到事件总线
        
        Args:
            event_bus: 事件总线实例，如果为None则使用self.event_bus
        """
        if self._registered:
            logger.warning("应用启动监听器已注册，跳过重复注册")
            return
            
        if event_bus:
            self.event_bus = event_bus
        elif not self.event_bus:
            # 优先使用全局事件总线实例（确保实例一致性）
            try:
                from src.core.event_bus import get_event_bus
                self.event_bus = get_event_bus()
                logger.info(f"使用全局事件总线实例（ID: {id(self.event_bus)}）")
            except (ImportError, AttributeError) as e:
                # 降级：创建新的事件总线实例
                logger.warning(f"无法获取全局事件总线，使用降级方案: {e}")
                self.event_bus = EventBus()
                self.event_bus.initialize()
                logger.info(f"创建新的事件总线实例用于应用启动监听器（ID: {id(self.event_bus)}）")
        
        # 订阅应用启动完成事件（主要事件）
        # 使用 APPLICATION_STARTUP_COMPLETE 事件
        logger.info(f"订阅 APPLICATION_STARTUP_COMPLETE 事件到事件总线（ID: {id(self.event_bus)}）")
        self.event_bus.subscribe_async(
            EventType.APPLICATION_STARTUP_COMPLETE,
            self._handle_application_startup
        )
        
        # 同时订阅 SERVICE_STARTED 事件作为备选（向后兼容）
        # 当服务名为 "api_server" 时触发
        logger.info(f"订阅 SERVICE_STARTED 事件到事件总线（ID: {id(self.event_bus)}）")
        self.event_bus.subscribe_async(EventType.SERVICE_STARTED, self._handle_service_started)
        
        self._registered = True
        logger.info(f"应用启动监听器已注册到事件总线（ID: {id(self.event_bus)}）")
    
    async def _handle_service_started(self, event):
        """
        处理服务启动事件
        
        Args:
            event: Event 对象或事件数据字典
        """
        try:
            # 处理不同的事件格式
            if hasattr(event, 'data'):
                # Event 对象
                event_data = event.data
                service_name = event_data.get('service_name', '') if isinstance(event_data, dict) else ''
                logger.debug(f"处理服务启动事件（Event对象，服务名: {service_name}）")
            elif isinstance(event, dict):
                # 事件字典格式
                event_data = event.get('data', {})
                service_name = event_data.get('service_name', '') if isinstance(event_data, dict) else ''
                logger.debug(f"处理服务启动事件（字典格式，服务名: {service_name}）")
            else:
                service_name = ''
                logger.debug("处理服务启动事件（未知格式）")
            
            # 只处理 API 服务器启动事件
            if service_name in ['api_server', 'fastapi', 'gateway', 'web'] or 'api' in service_name.lower():
                logger.info(f"检测到API服务启动事件: {service_name}")
                await self._start_background_services()
        except Exception as e:
            logger.error(f"处理服务启动事件失败: {e}", exc_info=True)
    
    async def _handle_application_startup(self, event):
        """
        处理应用启动完成事件
        
        Args:
            event: 事件对象（EventBus 传递的是 Event 对象）
        """
        try:
            logger.info(f"收到应用启动完成事件（事件类型: {type(event)}）")
            
            # 取消降级任务（因为已经收到事件）
            self._cancel_fallback_task()
            
            # EventBus 的异步处理器接收的是 Event 对象
            if hasattr(event, 'data'):
                # Event 对象
                event_data = event.data
                event_id = getattr(event, 'event_id', 'unknown')
                logger.info(f"检测到应用启动完成事件（事件ID: {event_id}，事件数据: {event_data}）")
            elif isinstance(event, dict):
                # 事件字典格式（向后兼容）
                event_data = event.get('data', {})
                logger.info(f"检测到应用启动完成事件（字典格式，数据: {event_data}）")
            else:
                event_data = {}
                logger.info(f"检测到应用启动完成事件（未知格式: {type(event)}，内容: {event}）")
            
            logger.info("开始启动后台服务（事件驱动方式）...")
            await self._start_background_services()
        except Exception as e:
            logger.error(f"处理应用启动完成事件失败: {e}", exc_info=True)
            import traceback
            logger.error(f"错误详情: {traceback.format_exc()}")
            # 如果事件驱动失败，尝试降级启动
            await self._fallback_start_scheduler()
    
    async def _start_background_services(self):
        """
        启动后台服务
        
        包括：
        - 统一调度器 (UnifiedScheduler)
        - 数据采集调度器
        - 其他后台服务（可扩展）
        """
        if self._scheduler_started:
            logger.debug("后台服务已启动，跳过重复启动")
            return
        
        try:
            logger.info("开始启动后台服务（符合核心服务层架构设计）")
            
            # P0: 启动统一调度器（必须先启动，其他调度器依赖它）
            await self._start_unified_scheduler()
            
            # 启动数据采集调度器
            await self._start_data_collection_scheduler()

            # 启动历史数据采集调度器
            await self._start_historical_data_scheduler()

            # 启动模型训练任务执行器
            await self._start_model_training_executor()

            self._scheduler_started = True
            logger.info("后台服务启动完成（包括统一调度器、历史数据采集调度器和模型训练执行器）")
            
        except Exception as e:
            logger.error(f"启动后台服务失败: {e}", exc_info=True)
            # 注意：不抛出异常，确保不影响API服务
    
    async def _start_data_collection_scheduler(self):
        """
        启动数据采集调度器
        
        符合核心服务层架构设计：业务流程编排
        """
        try:
            from src.core.orchestration.business_process.service_scheduler import start_data_collection_scheduler
            
            logger.info("准备启动数据采集调度器...")
            logger.info(f"启动路径: app_startup_listener")
            try:
                # 传递启动路径，便于追踪和调试
                success = await start_data_collection_scheduler(startup_path="app_startup_listener")
                
                if success:
                    logger.info("✅ 数据采集调度器启动成功（符合核心服务层架构设计：在后端服务启动之后）")
                    self._scheduler_started = True
                else:
                    logger.warning("⚠️ 数据采集调度器启动失败（可选功能，不影响API服务）")
            except Exception as scheduler_error:
                logger.error(f"调用数据采集调度器启动函数失败: {scheduler_error}", exc_info=True)
                
        except ImportError as e:
            logger.warning(f"无法导入数据采集调度器（可选功能）: {e}")
        except Exception as e:
            logger.error(f"启动数据采集调度器失败（可选功能）: {e}", exc_info=True)
            import traceback
            logger.error(f"错误详情: {traceback.format_exc()}")
            # 不抛出异常，确保不影响API服务

    async def _start_unified_scheduler(self):
        """
        启动统一调度器 (UnifiedScheduler)
        
        这是P0级后台服务，必须先启动，其他调度器依赖它。
        符合核心服务层架构设计。
        """
        try:
            from src.core.orchestration.scheduler import get_unified_scheduler
            
            logger.info("准备启动统一调度器 (UnifiedScheduler)...")
            logger.info("启动路径: app_startup_listener")
            
            try:
                scheduler = get_unified_scheduler()
                
                # 启动调度器（ UnifiedScheduler.start() 是异步方法）
                await scheduler.start()
                
                # 验证调度器是否成功启动
                if scheduler._running:
                    logger.info("✅ 统一调度器启动成功（符合分布式协调器架构设计）")
                    logger.info(f"   - 工作节点类型: DATA_COLLECTOR, FEATURE_WORKER, TRAINING_EXECUTOR, INFERENCE_WORKER")
                    logger.info(f"   - 任务类型: DATA_COLLECTION, FEATURE_EXTRACTION, MODEL_TRAINING, MODEL_INFERENCE")
                else:
                    logger.warning("⚠️ 统一调度器启动后未运行（可选功能，不影响API服务）")
                    
            except Exception as scheduler_error:
                logger.error(f"调用统一调度器启动函数失败: {scheduler_error}", exc_info=True)
                
        except ImportError as e:
            logger.warning(f"无法导入统一调度器（可选功能）: {e}")
        except Exception as e:
            logger.error(f"启动统一调度器失败（可选功能）: {e}", exc_info=True)
            import traceback
            logger.error(f"错误详情: {traceback.format_exc()}")
            # 不抛出异常，确保不影响API服务

    async def _start_historical_data_scheduler(self):
        """启动历史数据采集调度器

        符合核心服务层架构设计：业务流程编排 + 定期任务调度
        """
        try:
            from src.core.orchestration.historical_data_scheduler import get_historical_data_scheduler

            logger.info("准备启动历史数据采集调度器...")
            logger.info(f"启动路径: app_startup_listener")

            try:
                scheduler = get_historical_data_scheduler()
                success = await scheduler.start()

                if success:
                    logger.info("✅ 历史数据采集调度器启动成功（定期采集将在配置的时间窗口内自动执行）")
                    self._historical_scheduler_started = True
                else:
                    logger.warning("⚠️ 历史数据采集调度器启动失败（可选功能，不影响API服务）")
            except Exception as scheduler_error:
                logger.error(f"调用历史数据采集调度器启动函数失败: {scheduler_error}", exc_info=True)

        except ImportError as e:
            logger.warning(f"无法导入历史数据采集调度器（可选功能）: {e}")
        except Exception as e:
            logger.error(f"启动历史数据采集调度器失败（可选功能）: {e}")
            import traceback
            logger.error(f"错误详情: {traceback.format_exc()}")
            # 不抛出异常，确保不影响API服务

    async def _start_model_training_executor(self):
        """启动模型训练任务执行器

        符合核心服务层架构设计：业务流程编排 + 任务执行
        """
        try:
            from src.gateway.web.training_job_executor import start_training_job_executor

            logger.info("准备启动模型训练任务执行器...")
            logger.info(f"启动路径: app_startup_listener")

            try:
                executor = await start_training_job_executor()

                if executor:
                    logger.info("✅ 模型训练任务执行器启动成功")
                else:
                    logger.warning("⚠️ 模型训练任务执行器启动失败（可选功能，不影响API服务）")
            except Exception as executor_error:
                logger.error(f"调用模型训练任务执行器启动函数失败: {executor_error}", exc_info=True)

        except ImportError as e:
            logger.warning(f"无法导入模型训练任务执行器（可选功能）: {e}")
        except Exception as e:
            logger.error(f"启动模型训练任务执行器失败（可选功能）: {e}")
            import traceback
            logger.error(f"错误详情: {traceback.format_exc()}")
            # 不抛出异常，确保不影响API服务
    
    async def stop_background_services(self):
        """
        停止后台服务
        
        在应用关闭时调用
        """
        try:
            logger.info("开始停止后台服务...")
            
            # 停止数据采集调度器
            try:
                from src.core.orchestration.business_process.service_scheduler import stop_data_collection_scheduler
                await stop_data_collection_scheduler()
                logger.info("数据采集调度器已停止")
            except Exception as e:
                logger.warning(f"停止数据采集调度器失败: {e}")

            # 停止历史数据采集调度器
            try:
                from src.core.orchestration.historical_data_scheduler import get_historical_data_scheduler
                scheduler = get_historical_data_scheduler()
                await scheduler.stop()
                logger.info("历史数据采集调度器已停止")
            except Exception as e:
                logger.warning(f"停止历史数据采集调度器失败: {e}")

            self._scheduler_started = False
            self._historical_scheduler_started = False
            logger.info("后台服务停止完成（包括历史数据采集调度器）")
            
        except Exception as e:
            logger.error(f"停止后台服务失败: {e}", exc_info=True)
    
    def _start_fallback_task(self):
        """启动降级任务：如果一定时间内未收到事件，直接启动调度器"""
        async def fallback_task():
            try:
                await asyncio.sleep(self._startup_timeout)
                if not self._scheduler_started:
                    logger.warning(f"在 {self._startup_timeout} 秒内未收到应用启动完成事件，使用降级方案直接启动调度器")
                    await self._fallback_start_scheduler()
            except Exception as e:
                logger.error(f"降级任务执行失败: {e}", exc_info=True)
        
        # 创建降级任务
        try:
            self._fallback_task = asyncio.create_task(fallback_task())
            logger.debug(f"降级启动任务已启动（超时时间: {self._startup_timeout}秒）")
        except Exception as e:
            logger.warning(f"无法创建降级任务: {e}")
    
    def _cancel_fallback_task(self):
        """取消降级任务"""
        if self._fallback_task and not self._fallback_task.done():
            self._fallback_task.cancel()
            logger.debug("降级启动任务已取消（已收到事件）")
    
    async def _fallback_start_scheduler(self):
        """降级启动调度器（直接启动，不通过事件）"""
        if self._scheduler_started:
            logger.debug("调度器已启动，跳过降级启动")
            return
        
        try:
            logger.info("使用降级方案直接启动数据采集调度器...")
            await self._start_data_collection_scheduler()
            if self._scheduler_started:
                logger.info("降级启动成功：数据采集调度器已启动（直接启动方式）")
            else:
                logger.warning("降级启动失败：数据采集调度器未启动")
        except Exception as e:
            logger.error(f"降级启动调度器失败: {e}", exc_info=True)


def get_app_startup_listener(event_bus: Optional[EventBus] = None) -> AppStartupListener:
    """
    获取应用启动监听器实例（单例模式）
    
    Args:
        event_bus: 事件总线实例
    
    Returns:
        AppStartupListener: 监听器实例
    """
    global _app_startup_listener
    if _app_startup_listener is None:
        _app_startup_listener = AppStartupListener(event_bus)
    return _app_startup_listener


def register_app_startup_listener(event_bus: Optional[EventBus] = None):
    """
    注册应用启动监听器（便捷函数）
    
    Args:
        event_bus: 事件总线实例，如果为None则创建新实例
    """
    listener = get_app_startup_listener(event_bus)
    listener.register(event_bus)
