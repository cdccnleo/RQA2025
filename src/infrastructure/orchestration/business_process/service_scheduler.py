#!/usr/bin/env python3
"""
⚠️ 废弃文件 ⚠️

此调度器实现已被统一调度器取代。
请使用 src.core.orchestration.scheduler.unified_scheduler

保留此文件仅作为历史参考，将在未来版本中删除。

新的统一调度器位置：
- src/core/orchestration/scheduler/unified_scheduler.py
- src/core/orchestration/scheduler/task_manager.py
- src/core/orchestration/scheduler/worker_manager.py

迁移说明：
- 所有调度功能已迁移到统一调度器
- API端点已更新为 /api/v1/data/scheduler/*
- 统一调度器支持数据采集、特征工程、模型训练等多种任务类型
"""

import warnings
warnings.warn(
    "service_scheduler is deprecated. Use unified_scheduler instead.",
    DeprecationWarning,
    stacklevel=2
)

"""
后台服务调度器（已废弃）

基于核心服务层架构设计，管理后台服务的启动和停止：
1. 数据采集调度器
2. 其他后台服务（可扩展）

职责：
- 统一管理后台服务生命周期
- 确保服务启动顺序正确
- 提供优雅的启动和停止机制

⚠️ 注意：此文件已废弃，请使用统一调度器
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, time as dttime

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False
    psutil = None

from src.infrastructure.orchestration.business_process.data_collection_orchestrator import DataCollectionWorkflow
from src.infrastructure.orchestration.business_process.scheduler_persistence import SchedulerPersistence
from src.gateway.web.data_source_config_manager import get_data_source_config_manager
from src.gateway.web.data_collectors import parse_rate_limit
from src.core.event_bus.core import EventBus
from src.core.event_bus.types import EventType
from src.infrastructure.logging.core.unified_logger import get_unified_logger

# P1阶段：集成智能调度组件
from src.infrastructure.orchestration.market_adaptive_monitor import get_market_adaptive_monitor
from src.infrastructure.orchestration.data_priority_manager import get_data_priority_manager
from src.infrastructure.orchestration.incremental_collection_strategy import get_incremental_collection_strategy

logger = get_unified_logger(__name__)

# 可选导入：数据质量验证器（如果模块不存在则跳过）
try:
    from src.infrastructure.monitoring.data_collection_quality_monitor import get_quality_validator
    QUALITY_VALIDATOR_AVAILABLE = True
except ImportError:
    logger.warning("数据质量验证器模块不存在，跳过质量验证功能")
    QUALITY_VALIDATOR_AVAILABLE = False
    get_quality_validator = None


class DataCollectionServiceScheduler:
    """
    数据采集服务调度器
    
    负责数据采集任务的调度和管理，符合核心服务层架构设计：
    - 使用业务流程编排器（DataCollectionWorkflow）
    - 事件驱动通信（EventBus）
    - 按照数据源配置的rate_limit进行自动调度
    """

    # 采集间隔上下限（避免对外部数据源请求过于频繁导致限流/封禁）
    MIN_COLLECTION_INTERVAL_SEC = 30
    MAX_COLLECTION_INTERVAL_SEC = 86400
    # 时间感知乘数：交易时段降频、非交易升频、周末/节假日进一步降频
    TRADING_HOURS_MULT = 1.3
    OFF_HOURS_MULT = 0.75
    WEEKEND_MULT = 1.5

    def __init__(self):
        """初始化数据采集服务调度器"""
        self.running = False
        self.scheduler_task: Optional[asyncio.Task] = None
        self.data_source_manager = None
        self.orchestrator: Optional[DataCollectionWorkflow] = None
        self.event_bus: Optional[EventBus] = None
        self.last_collection_times: Dict[str, float] = {}
        self.check_interval = 120  # 120秒检查一次，进一步降低检查频率以减少系统负载
        self._startup_path: Optional[str] = None  # 启动路径追踪
        self._startup_time: Optional[float] = None  # 启动时间

        # 并发控制 - 降低并发度以保护系统健康
        self.max_concurrent_tasks = 1  # 最多同时运行1个采集任务（降低并发以保护健康检查）
        self.active_tasks: Set[str] = set()  # 当前活跃的任务
        self.pending_sources: List[Dict[str, Any]] = []  # 待处理的数据源

        # 启动延迟控制（避免应用刚启动就大量采集）
        self.startup_delay = 60  # 启动后1分钟才开始正常采集，避免影响应用启动稳定性
        self.startup_time = None
        self.application_startup_time = time.time()  # 记录调度器创建时间，作为应用启动的参考时间

        # 负载保护机制 - 放宽限制
        self.high_load_count = 0  # 连续高负载次数
        self.max_high_load_count = 10  # 最多允许10次连续高负载（增加容忍度）
        self.load_check_enabled = _PSUTIL_AVAILABLE  # 是否启用负载检查

        # 初始化持久化管理器（符合基础设施层架构设计）
        self.persistence = SchedulerPersistence()
        self._last_save_time: float = 0.0
        self._save_interval = 300  # 每5分钟保存一次

        # P1阶段：集成智能调度组件
        self.market_monitor = get_market_adaptive_monitor()  # 市场状态感知器
        self.priority_manager = get_data_priority_manager()  # 数据优先级管理器
        self.incremental_strategy = get_incremental_collection_strategy()  # 增量采集策略

        # 可选初始化：数据质量验证器
        if QUALITY_VALIDATOR_AVAILABLE and get_quality_validator:
            try:
                self.quality_validator = get_quality_validator()  # 数据质量验证器
                logger.info("数据质量验证器初始化成功")
            except Exception as e:
                logger.warning(f"数据质量验证器初始化失败: {e}")
                self.quality_validator = None
        else:
            self.quality_validator = None
            logger.info("数据质量验证器不可用，跳过初始化")

        # 适应性参数（支持市场适应性调整）
        self.default_batch_size = 20
        self.default_interval = 1800.0  # 调试阶段调整为1800秒
        self.priority_multipliers = {
            'high': 1.0,
            'medium': 1.0,
            'low': 1.0
        }
        self.current_batch_size = self.default_batch_size
        self.current_interval = self.default_interval

        # 智能调度统计
        self.market_regime = None  # 当前市场状态
        self.last_regime_check = 0  # 上次检查市场状态的时间

        # 时间感知：交易时段判断（可选依赖 MarketAwareRetryHandler，失败则用 fallback）
        self._market_aware_handler = None
        try:
            from src.infrastructure.utils.tools.market_aware_retry import (
                MarketAwareRetryHandler,
                MarketPhase,
            )
            self._market_aware_handler = MarketAwareRetryHandler()
            self._market_phase_enum = MarketPhase
            logger.info("时间感知调度：已加载 MarketAwareRetryHandler（含节假日）")
        except ImportError as e:
            logger.debug(f"时间感知调度：MarketAwareRetryHandler 不可用 ({e})，使用 weekday+时段 fallback")

    async def start(self, startup_path: Optional[str] = None) -> bool:
        """
        启动数据采集调度器

        Args:
            startup_path: 启动路径，用于追踪调度器是从哪里启动的

        Returns:
            bool: 是否启动成功
        """
        logger.info(f"尝试启动数据采集调度器，当前运行状态: {self.running}, 实例ID: {id(self)}")

        # 检查全局单例是否已经在运行
        global _data_collection_scheduler
        if _data_collection_scheduler is not None and _data_collection_scheduler is not self:
            if _data_collection_scheduler.running:
                logger.warning(f"全局调度器实例已在运行中（实例ID: {id(_data_collection_scheduler)}），使用现有实例")
                return True

        if self.running:
            logger.warning(f"当前调度器实例已在运行中，跳过重复启动")
            return True
            
        try:
            # 记录启动路径和时间
            self._startup_path = startup_path or "unknown"
            self._startup_time = time.time()
            
            logger.info(f"启动数据采集调度器（符合核心服务层架构设计，启动路径: {self._startup_path}）")
            
            # 初始化组件（符合核心服务层架构设计）
            self.data_source_manager = get_data_source_config_manager()
            
            # 初始化业务流程编排器（符合核心服务层架构设计：业务流程编排）
            self.orchestrator = DataCollectionWorkflow()
            
            # 初始化事件总线（符合核心服务层架构设计：事件驱动通信）
            self.event_bus = EventBus()
            self.event_bus.initialize()
            
            # 加载历史采集时间（符合基础设施层架构设计：状态持久化）
            try:
                loaded_times = self.persistence.load_last_collection_times()
                if loaded_times:
                    self.last_collection_times.update(loaded_times)
                    logger.info(f"成功加载 {len(loaded_times)} 个数据源的历史采集时间")
                else:
                    logger.debug("未找到历史采集时间，将使用默认值")
            except Exception as e:
                logger.warning(f"加载历史采集时间失败: {e}，将使用默认值")
            
            # 启动调度循环
            self.running = True
            self.scheduler_task = asyncio.create_task(self._scheduler_loop())
            
            logger.info(f"数据采集调度器已启动（符合核心服务层架构设计：在后端服务启动之后，启动路径: {self._startup_path}）")
            return True
            
        except Exception as e:
            logger.error(f'数据采集调度器启动失败: {e}', exc_info=True)
            self.running = False
            self._startup_path = None
            self._startup_time = None
            return False
    
    async def stop(self) -> bool:
        """
        停止数据采集调度器
        
        Returns:
            bool: 是否停止成功
        """
        if not self.running:
            return True
            
        try:
            logger.info("停止数据采集调度器...")
            self.running = False
            
            # 取消调度任务
            if self.scheduler_task:
                self.scheduler_task.cancel()
                try:
                    await self.scheduler_task
                except asyncio.CancelledError:
                    pass
                self.scheduler_task = None
            
            # 保存当前采集时间状态（符合基础设施层架构设计：状态持久化）
            try:
                if self.last_collection_times:
                    self.persistence.save_last_collection_times(self.last_collection_times)
                    logger.info("调度器状态已保存")
            except Exception as e:
                logger.warning(f"保存调度器状态失败: {e}")
            
            logger.info("数据采集调度器已停止")
            return True
            
        except Exception as e:
            logger.error(f'停止数据采集调度器失败: {e}')
            return False

    def reload_data_sources(self) -> bool:
        """
        重新加载数据源配置

        Returns:
            bool: 是否重新加载成功
        """
        try:
            logger.info("重新加载数据源配置...")
            if self.data_source_manager:
                # 强制重新加载配置
                sources = self.data_source_manager.get_data_sources(force_reload=True)
                logger.info(f"成功重新加载 {len(sources)} 个数据源配置")
                return True
            else:
                logger.warning("数据源管理器未初始化，无法重新加载配置")
                return False
        except Exception as e:
            logger.error(f"重新加载数据源配置失败: {e}")
            return False

    def adjust_parameters(self, batch_size: Optional[int] = None,
                         interval_seconds: Optional[float] = None,
                         priority_multipliers: Optional[Dict[str, float]] = None):
        """
        适应性参数调整（由市场适应性监控器调用）

        Args:
            batch_size: 新的批次大小
            interval_seconds: 新的采集间隔（秒）
            priority_multipliers: 新的优先级倍数
        """
        try:
            changed = False

            if batch_size is not None and batch_size != self.current_batch_size:
                old_size = self.current_batch_size
                self.current_batch_size = max(5, min(batch_size, 200))  # 限制在合理范围内
                logger.info(f"适应性调整：批次大小 {old_size} -> {self.current_batch_size}")
                changed = True

            if interval_seconds is not None and abs(interval_seconds - self.current_interval) > 0.1:
                old_interval = self.current_interval
                self.current_interval = max(5.0, min(interval_seconds, 300.0))  # 限制在合理范围内
                logger.info(f"适应性调整：采集间隔 {old_interval:.1f} -> {self.current_interval:.1f}秒")
                changed = True

            if priority_multipliers is not None:
                old_multipliers = self.priority_multipliers.copy()
                self.priority_multipliers.update(priority_multipliers)

                # 验证优先级倍数合理性
                for level in ['high', 'medium', 'low']:
                    if level in self.priority_multipliers:
                        self.priority_multipliers[level] = max(0.1, min(self.priority_multipliers[level], 5.0))

                if old_multipliers != self.priority_multipliers:
                    logger.info(f"适应性调整：优先级倍数 {old_multipliers} -> {self.priority_multipliers}")
                    changed = True

            if changed:
                logger.info("✅ 适应性参数调整完成")

        except Exception as e:
            logger.error(f"适应性参数调整失败: {e}")

    def reset_to_defaults(self):
        """重置为默认参数"""
        try:
            self.current_batch_size = self.default_batch_size
            self.current_interval = self.default_interval
            self.priority_multipliers = {
                'high': 1.0,
                'medium': 1.0,
                'low': 1.0
            }
            logger.info("✅ 参数已重置为默认值")

        except Exception as e:
            logger.error(f"重置默认参数失败: {e}")

    def get_current_parameters(self) -> Dict[str, Any]:
        """获取当前参数配置"""
        return {
            'batch_size': self.current_batch_size,
            'interval_seconds': self.current_interval,
            'priority_multipliers': self.priority_multipliers.copy(),
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'check_interval': self.check_interval,
            'startup_delay': self.startup_delay
        }
    
    async def _scheduler_loop(self):
        """
        调度循环
        
        按照数据源配置的rate_limit进行自动调度
        """
        while self.running:
            try:
                # 获取启用的数据源
                sources = self.data_source_manager.get_data_sources()
                enabled_sources = [s for s in sources if s.get('enabled', False)]
                
                if not enabled_sources:
                    await asyncio.sleep(self.check_interval)
                    continue
                
                current_time = time.time()
                
                # 定期保存采集时间（符合基础设施层架构设计：状态持久化）
                if current_time - self._last_save_time >= self._save_interval:
                    try:
                        if self.last_collection_times:
                            self.persistence.save_last_collection_times(self.last_collection_times)
                            self._last_save_time = current_time
                            logger.debug("已定期保存采集时间状态")
                    except Exception as e:
                        logger.warning(f"定期保存采集时间失败: {e}")
                
                # 检查启动延迟 - 使用应用启动时间
                time_since_app_startup = current_time - self.application_startup_time
                if time_since_app_startup < self.startup_delay:
                    logger.info(f"应用启动延迟中，还需等待 {self.startup_delay - time_since_app_startup:.1f} 秒（启动路径: {self._startup_path}）")
                    await asyncio.sleep(min(self.check_interval, self.startup_delay - time_since_app_startup))
                    continue

                # 检查系统资源使用情况，避免在高负载时启动新任务
                if self._should_throttle_due_to_high_load():
                    self.high_load_count += 1
                    logger.warning(f"系统负载过高，暂停数据采集任务（活跃任务: {len(self.active_tasks)}，连续高负载: {self.high_load_count}/{self.max_high_load_count}）")

                    # 如果连续高负载次数过多，停止调度器
                    if self.high_load_count >= self.max_high_load_count:
                        logger.error(f"系统连续高负载 {self.high_load_count} 次，停止调度器以保护应用稳定性")
                        await self.stop()
                        continue
                        self.running = False
                        break

                    await asyncio.sleep(self.check_interval)
                    continue
                else:
                    # 重置高负载计数
                    if self.high_load_count > 0:
                        logger.info(f"系统负载恢复正常，重置高负载计数（之前: {self.high_load_count}）")
                        self.high_load_count = 0

                # P1阶段：智能调度 - 检查市场状态
                await self._check_market_regime()

                # 记录调度检查日志
                current_time = time.time()
                regime_info = f"市场状态: {self.market_regime.current_regime.value if self.market_regime else 'unknown'}"
                logger.info(f"智能调度检查周期开始：当前有 {len(enabled_sources)} 个启用的数据源，活跃任务: {len(self.active_tasks)}，待处理队列: {len(self.pending_sources)}，{regime_info}，时间: {datetime.fromtimestamp(current_time).isoformat()}")

                # 检查循环执行频率（调试用）
                if hasattr(self, '_last_cycle_time'):
                    cycle_interval = current_time - self._last_cycle_time
                    if cycle_interval < 50:  # 如果循环间隔小于50秒，记录警告
                        logger.warning(f"调度器循环执行过于频繁: {cycle_interval:.1f}秒，上次执行时间: {datetime.fromtimestamp(self._last_cycle_time).isoformat()}")
                self._last_cycle_time = current_time

                # 处理待处理的数据源（如果有空闲slot）
                while self.pending_sources and len(self.active_tasks) < self.max_concurrent_tasks:
                    pending_source = self.pending_sources.pop(0)
                    await self._start_collection_task(pending_source)

                # P1阶段：智能调度 - 按优先级和市场状态对数据源进行排序
                prioritized_sources = self._prioritize_sources_intelligent(enabled_sources)
                logger.debug(f"本次智能调度检查到 {len(prioritized_sources)} 个优先级排序的数据源")

                for source in prioritized_sources:
                    source_id = source['id']
                    rate_limit = source.get('rate_limit', '60次/分钟')

                    # 先解析基础频率
                    base_interval = parse_rate_limit(rate_limit)

                    # P1阶段：智能调度 - 根据市场状态和数据优先级调整采集频率
                    pool_priority = source.get('config', {}).get('pool_priority', 'medium')
                    adjusted_interval = self._adjust_interval_intelligent(base_interval, pool_priority, source_id)

                    # 使用调整后的间隔
                    interval_seconds = adjusted_interval
                    
                    # 检查是否到了采集时间
                    last_time = self.last_collection_times.get(source_id, 0)
                    
                    # 修复：如果是首次采集（last_time=0），应该等待一个完整的间隔周期
                    if last_time == 0:
                        logger.info(f"数据源 {source_id} 首次启动，将在下一个采集周期执行（间隔: {interval_seconds:.1f}秒）")
                        # 设置 last_time 为当前时间，下次检查时会等待完整间隔
                        self.last_collection_times[source_id] = current_time
                        continue
                    
                    time_since_last = current_time - last_time
                    
                    logger.info(
                        f"数据源 {source_id} 调度检查: 上次采集 {datetime.fromtimestamp(last_time).isoformat() if last_time > 0 else '从未采集'}, "
                        f"距离上次 {time_since_last:.1f}秒, 需要间隔 {interval_seconds:.1f}秒, "
                        f"状态: {'需要采集' if time_since_last >= interval_seconds else '等待中'}"
                    )
                    
                    if time_since_last >= interval_seconds:
                        # P1阶段：质量检查 - 在调度前检查数据质量
                        if not await self._should_schedule_source(source_id, source):
                            logger.debug(f"数据源 {source_id} 质量检查未通过，跳过本次调度")
                            continue

                        # 额外的并发检查：确保同一个数据源不会被同时多次启动
                        if source_id in self.active_tasks:
                            logger.debug(f"数据源 {source_id} 已有活跃任务，跳过本次调度")
                            continue

                        logger.debug(f"数据源 {source_id} 达到采集时间间隔且质量检查通过（{time_since_last:.1f}s >= {interval_seconds:.1f}s），准备调度")

                        # 同步启动采集任务（确保顺序执行，避免并发问题）
                        task_info = {
                            "source_id": source_id,
                            "source": source,
                            "rate_limit": rate_limit,
                            "interval_seconds": interval_seconds,
                            "current_time": current_time,
                            "time_since_last": time_since_last
                        }

                        # 创建任务但不等待完成，让调度循环继续
                        asyncio.create_task(self._start_collection_task(task_info))

                        # 检查并发限制
                        if len(self.active_tasks) >= self.max_concurrent_tasks:
                            # 添加到待处理队列
                            self.pending_sources.append({
                                "source_id": source_id,
                                "source": source,
                                "rate_limit": rate_limit,
                                "interval_seconds": interval_seconds,
                                "current_time": current_time,
                                "time_since_last": time_since_last
                            })
                            logger.debug(f"数据源 {source_id} 加入待处理队列（当前活跃任务: {len(self.active_tasks)}）")
                        else:
                            # 最后一次资源检查，确保启动任务前系统负载正常
                            if self._should_throttle_due_to_high_load():
                                logger.warning(f"数据源 {source_id} 最后资源检查失败，跳过本次采集")
                                continue

                            # 直接启动任务
                            await self._start_collection_task({
                                "source_id": source_id,
                                "source": source,
                                "rate_limit": rate_limit,
                                "interval_seconds": interval_seconds,
                                "current_time": current_time,
                                "time_since_last": time_since_last
                            })
                
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                logger.info("数据采集调度器循环被取消")
                break
            except Exception as e:
                logger.error(f'数据采集调度循环异常: {e}')
                await asyncio.sleep(5)  # 出错后等待5秒再试

    async def _start_collection_task(self, task_info: Dict[str, Any]):
        """启动采集任务（带并发控制）"""
        source_id = task_info["source_id"]
        source = task_info["source"]
        rate_limit = task_info["rate_limit"]
        interval_seconds = task_info["interval_seconds"]
        current_time = task_info["current_time"]
        time_since_last = task_info["time_since_last"]

        # 添加到活跃任务
        self.active_tasks.add(source_id)

        try:
            logger.info(
                f"数据源 {source_id} 启动采集任务（间隔: {interval_seconds:.1f}秒, "
                f"距离上次: {time_since_last:.1f}秒，活跃任务: {len(self.active_tasks)}）"
            )

            # 发布数据采集开始事件（符合核心服务层架构设计：事件驱动通信）
            self.event_bus.publish(
                EventType.DATA_COLLECTION_STARTED,
                {
                    "source_id": source_id,
                    "source_config": source,
                    "rate_limit": rate_limit,
                    "interval_seconds": interval_seconds,
                    "timestamp": current_time,
                    "time_since_last": time_since_last
                },
                source="data_collection_scheduler"
            )

            # 使用业务流程编排器启动采集流程（符合核心服务层架构设计）
            collection_result = await self.orchestrator.start_collection_process(source_id, source)

            if collection_result.get('success', False):
                # 检查是否完成了所有批次的采集
                completed_all_batches = collection_result.get('completed_all_batches', True)
                batches_info = collection_result.get('batches_info', {})

                if completed_all_batches:
                    # 完成了所有批次，更新采集时间戳
                    self.last_collection_times[source_id] = current_time
                    logger.info(
                        f"数据源 {source_id} 采集任务完成（间隔: {interval_seconds:.1f}秒，"
                        f"批次: {batches_info.get('completed', 0)}/{batches_info.get('total', 0)}，"
                        f"下次检查时间: {datetime.fromtimestamp(current_time + interval_seconds).isoformat()}）"
                    )
                else:
                    # 分批次采集未完成，不更新时间戳，允许快速继续下一批次
                    logger.info(
                        f"数据源 {source_id} 批次采集完成（批次: {batches_info.get('completed', 0)}/{batches_info.get('total', 0)}），"
                        f"将继续下一批次采集"
                    )
            else:
                # 采集失败时也更新时间戳，避免立即重新触发
                self.last_collection_times[source_id] = current_time
                logger.warning(f"数据源 {source_id} 采集任务失败（间隔: {interval_seconds:.1f}秒），已更新时间戳避免频繁重试")

        except Exception as e:
            logger.error(f"数据源 {source_id} 采集任务异常: {e}", exc_info=True)
        finally:
            # 从活跃任务中移除
            self.active_tasks.discard(source_id)
            logger.debug(f"数据源 {source_id} 任务结束，剩余活跃任务: {len(self.active_tasks)}")

    def _prioritize_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        按优先级对数据源进行排序

        优先级顺序：high -> medium -> low
        """
        def get_priority_order(source):
            priority = source.get('config', {}).get('pool_priority', 'medium')
            priority_order = {'high': 0, 'medium': 1, 'low': 2}
            return priority_order.get(priority, 1)

        return sorted(sources, key=get_priority_order)

    def _adjust_interval_by_priority(self, base_interval: float, priority: str) -> float:
        """
        根据池优先级调整采集间隔

        Args:
            base_interval: 基础采集间隔（秒）
            priority: 池优先级 (high/medium/low)

        Returns:
            调整后的采集间隔（秒）
        """
        # 优先级调整因子：高优先级采集更频繁，低优先级采集间隔更长
        priority_multipliers = {
            'high': 0.5,    # 高优先级：2倍频率（间隔减半）
            'medium': 1.0,  # 中优先级：正常频率
            'low': 2.0      # 低优先级：1/2频率（间隔翻倍）
        }

        multiplier = priority_multipliers.get(priority, 1.0)

        # 确保间隔不低于最小值（避免过于频繁采集）
        min_interval = 30  # 最短30秒间隔
        adjusted_interval = max(base_interval * multiplier, min_interval)

        return adjusted_interval

    def _should_throttle_due_to_high_load(self) -> bool:
        """检查是否因系统负载过高而应该限制任务启动"""
        if not _PSUTIL_AVAILABLE:
            # 如果没有psutil，只检查活跃任务数量
            return len(self.active_tasks) >= self.max_concurrent_tasks

        try:
            # 检查CPU使用率 - 降低阈值以保护健康检查
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 60:  # 从80%降低到60%
                logger.warning(f"CPU使用率过高: {cpu_percent}%")
                return True

            # 检查内存使用率 - 降低阈值以保护健康检查
            memory = psutil.virtual_memory()
            if memory.percent > 70:  # 从85%降低到70%
                logger.warning(f"内存使用率过高: {memory.percent}%")
                return True

            # 检查活跃任务数量
            if len(self.active_tasks) >= self.max_concurrent_tasks:
                return True

            return False

        except Exception as e:
            logger.warning(f"检查系统负载失败: {e}")
            # 如果无法检查负载，只检查活跃任务数量
            return len(self.active_tasks) >= self.max_concurrent_tasks
    
    def is_running(self) -> bool:
        """检查调度器是否正在运行"""
        return self.running
    
    def get_status(self) -> Dict[str, Any]:
        """获取调度器状态"""
        base_status = {
            'running': self.running,
            'startup_path': self._startup_path,
            'startup_time': datetime.fromtimestamp(self._startup_time).isoformat() if self._startup_time else None,
            'enabled_sources_count': len([s for s in (self.data_source_manager.get_data_sources() if self.data_source_manager else []) if s.get('enabled', False)]),
            'last_collection_times': {k: datetime.fromtimestamp(v).isoformat() for k, v in self.last_collection_times.items()},
            'check_interval': self.check_interval
        }

        # P1阶段：添加智能调度状态信息
        time_mult, time_phase = self._get_trading_hours_interval_multiplier()
        intelligent_status = {
            'market_regime': self.market_regime.current_regime.value if self.market_regime else 'unknown',
            'regime_confidence': self.market_regime.confidence if self.market_regime else 0.0,
            'last_regime_check': datetime.fromtimestamp(self.last_regime_check).isoformat() if self.last_regime_check else None,
            'intelligent_scheduling': True,
            'time_phase': time_phase,
            'time_phase_multiplier': time_mult,
        }

        return {**base_status, **intelligent_status}

    # ============================================================================
    # P1阶段：智能调度方法
    # ============================================================================

    async def _check_market_regime(self):
        """检查市场状态（智能调度核心）"""
        current_time = time.time()

        # 每5分钟检查一次市场状态
        if current_time - self.last_regime_check > 300:
            try:
                self.market_regime = await self.market_monitor.get_current_regime()
                self.last_regime_check = current_time

                logger.info(f"市场状态更新: {self.market_regime.current_regime.value}, "
                           f"置信度: {self.market_regime.confidence:.2f}")

                # 根据市场状态调整调度参数
                self._adjust_scheduler_for_market_regime()

            except Exception as e:
                logger.warning(f"检查市场状态失败: {e}")
                # 使用默认状态
                from src.infrastructure.orchestration.market_adaptive_monitor import MarketRegime, MarketRegimeAnalysis, MarketMetrics
                self.market_regime = MarketRegimeAnalysis(
                    current_regime=MarketRegime.SIDEWAYS,
                    confidence=0.5,
                    metrics=MarketMetrics(timestamp=datetime.now()),
                    indicators={},
                    recommended_actions=["使用标准调度策略"],
                    analysis_timestamp=datetime.now()
                )

    def _adjust_scheduler_for_market_regime(self):
        """根据市场状态调整调度器参数"""
        if not self.market_regime:
            return

        regime = self.market_regime.current_regime
        actions = self.market_regime.recommended_actions

        # 根据市场状态调整参数
        if regime.name == 'HIGH_VOLATILITY':
            # 高波动：减少并发，增加检查间隔
            self.max_concurrent_tasks = max(1, self.max_concurrent_tasks // 2)
            self.check_interval = min(300, self.check_interval * 1.5)
            logger.info("高波动市场：调整为保守调度模式")

        elif regime.name == 'LOW_LIQUIDITY':
            # 低流动性：减少采集频率
            self.check_interval = min(600, self.check_interval * 2)
            logger.info("低流动性市场：降低采集频率")

        elif regime.name == 'BULL':
            # 牛市：可以适当增加采集频率
            self.check_interval = max(60, self.check_interval * 0.8)
            self.max_concurrent_tasks = min(3, self.max_concurrent_tasks + 1)
            logger.info("牛市行情：优化为积极采集模式")

        else:
            # 默认状态：使用标准参数
            self.max_concurrent_tasks = 1
            self.check_interval = 120
            logger.info("标准市场状态：使用均衡调度模式")

    def _prioritize_sources_intelligent(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        智能优先级排序（集成市场状态和数据优先级）

        排序逻辑：
        1. 首先按数据优先级排序（核心股票 > 指数 > 全市场 > 宏观）
        2. 然后按最后采集时间排序（最久未采集的优先）
        3. 考虑市场状态的影响
        """
        if not sources:
            return sources

        # 为每个数据源计算优先级得分
        source_scores = []
        for source in sources:
            source_id = source['id']
            data_type = source.get('type', 'stock')

            # 获取数据优先级配置
            priority_config = self.priority_manager.get_data_priority(source_id, data_type)
            priority_score = self.priority_manager.calculate_task_priority_score(source_id, data_type)

            # 获取最后采集时间
            last_collection = self.last_collection_times.get(source_id, 0)
            time_since_last = time.time() - last_collection

            # 计算时间权重（越久未采集权重越高）
            time_weight = min(time_since_last / 3600, 24)  # 最多24小时权重

            # 市场状态调整因子
            market_factor = self._calculate_market_adjustment_factor(priority_config.priority_level)

            # 计算综合得分（优先级 + 时间权重 + 市场调整）
            total_score = priority_score - time_weight * market_factor

            source_scores.append((source, total_score))

        # 按得分排序（得分低的优先级高）
        source_scores.sort(key=lambda x: x[1])

        sorted_sources = [source for source, _ in source_scores]

        logger.debug(f"智能优先级排序完成: {len(sorted_sources)} 个数据源")
        return sorted_sources

    def _calculate_market_adjustment_factor(self, priority_level: str) -> float:
        """计算市场状态调整因子"""
        if not self.market_regime:
            return 1.0

        regime = self.market_regime.current_regime

        # 不同市场状态对不同优先级的影响
        adjustment_matrix = {
            'HIGH_VOLATILITY': {'critical': 0.5, 'high': 0.7, 'medium': 1.0, 'low': 1.5},
            'LOW_LIQUIDITY': {'critical': 0.8, 'high': 1.0, 'medium': 1.2, 'low': 1.5},
            'BULL': {'critical': 1.5, 'high': 1.2, 'medium': 1.0, 'low': 0.8},
            'BEAR': {'critical': 1.2, 'high': 1.0, 'medium': 0.9, 'low': 0.7},
            'SIDEWAYS': {'critical': 1.0, 'high': 1.0, 'medium': 1.0, 'low': 1.0}
        }

        regime_name = regime.name if hasattr(regime, 'name') else str(regime).split('.')[-1]
        return adjustment_matrix.get(regime_name, {}).get(priority_level, 1.0)

    def _adjust_interval_intelligent(self, base_interval: float, pool_priority: str, source_id: str) -> float:
        """
        智能采集间隔调整（集成市场状态和数据优先级）

        Args:
            base_interval: 基础采集间隔（秒）
            pool_priority: 池优先级
            source_id: 数据源ID

        Returns:
            调整后的采集间隔（秒）
        """
        # 获取数据优先级配置
        priority_config = self.priority_manager.get_data_priority(source_id)

        # 基础优先级调整
        priority_multiplier = self._get_priority_multiplier(pool_priority, priority_config.priority_level)

        # 市场状态调整
        market_multiplier = self._get_market_interval_multiplier()

        # 时间感知调整（交易时段降频、非交易升频、周末/节假日进一步降频）
        time_mult, _ = self._get_trading_hours_interval_multiplier()

        # 计算调整后的间隔
        adjusted_interval = base_interval * priority_multiplier * market_multiplier * time_mult

        # 确保间隔在合理范围内（避免对外部数据源请求过于频繁导致限流/封禁）
        adjusted_interval = max(
            self.MIN_COLLECTION_INTERVAL_SEC,
            min(adjusted_interval, self.MAX_COLLECTION_INTERVAL_SEC),
        )

        logger.debug(f"数据源 {source_id} 采集间隔调整: {base_interval:.1f}s -> {adjusted_interval:.1f}s "
                    f"(优先级: {pool_priority}->{priority_config.priority_level}, 市场: {market_multiplier:.2f}, 时段: {time_mult:.2f})")

        return adjusted_interval

    def _get_priority_multiplier(self, pool_priority: str, data_priority: str) -> float:
        """获取优先级调整倍数"""
        # 将pool_priority映射到data_priority
        priority_mapping = {
            'high': 'critical',
            'medium': 'medium',
            'low': 'low'
        }

        mapped_priority = priority_mapping.get(pool_priority, data_priority)

        # 优先级调整倍数
        multipliers = {
            'critical': 0.5,  # 高优先级：2倍频率
            'high': 0.7,
            'medium': 1.0,    # 中优先级：正常频率
            'low': 2.0        # 低优先级：1/2频率
        }

        return multipliers.get(mapped_priority, 1.0)

    def _get_market_interval_multiplier(self) -> float:
        """获取市场状态对采集间隔的影响倍数"""
        if not self.market_regime:
            return 1.0

        regime = self.market_regime.current_regime

        # 市场状态对采集间隔的影响
        interval_multipliers = {
            'HIGH_VOLATILITY': 1.5,  # 高波动：增加间隔
            'LOW_LIQUIDITY': 2.0,   # 低流动性：大幅增加间隔
            'BULL': 0.8,            # 牛市：减少间隔
            'BEAR': 1.0,            # 熊市：正常间隔
            'SIDEWAYS': 1.0         # 横盘：正常间隔
        }

        regime_name = regime.name if hasattr(regime, 'name') else str(regime).split('.')[-1]
        return interval_multipliers.get(regime_name, 1.0)

    def _get_trading_hours_interval_multiplier(self, dt: Optional[datetime] = None) -> tuple:
        """
        获取交易时段对采集间隔的乘数。乘数>1 表示拉长间隔（降频），<1 表示缩短间隔（升频）。

        Returns:
            (multiplier, phase): 乘数 float，及 "trading"|"off_hours"|"weekend"
        """
        dt = dt or datetime.now()
        if self._market_aware_handler is not None:
            phase = self._market_aware_handler.get_market_phase(dt)
            if phase.name in ('MORNING', 'AFTERNOON'):
                return (self.TRADING_HOURS_MULT, 'trading')
            if phase.name in ('PRE_OPEN', 'LUNCH_BREAK'):
                return (self.OFF_HOURS_MULT, 'off_hours')
            if phase.name == 'CLOSED':
                if dt.weekday() >= 5 or (hasattr(self._market_aware_handler, 'holidays') and dt.date() in self._market_aware_handler.holidays):
                    return (self.WEEKEND_MULT, 'weekend')
                return (self.OFF_HOURS_MULT, 'off_hours')
            return (self.OFF_HOURS_MULT, 'off_hours')
        # Fallback：仅按 weekday + 时间范围，不依赖 holidays/pytz
        if dt.weekday() >= 5:
            return (self.WEEKEND_MULT, 'weekend')
        t = dt.time()
        if (dttime(9, 30) <= t < dttime(11, 30)) or (dttime(13, 0) <= t < dttime(15, 0)):
            return (self.TRADING_HOURS_MULT, 'trading')
        return (self.OFF_HOURS_MULT, 'off_hours')

    async def _should_schedule_source(self, source_id: str, source_config: Dict[str, Any]) -> bool:
        """
        判断是否应该调度数据源（集成质量监控）

        Args:
            source_id: 数据源ID
            source_config: 数据源配置

        Returns:
            是否应该调度
        """
        try:
            data_type = source_config.get('type', 'stock')

            # 获取数据优先级配置
            priority_config = self.priority_manager.get_data_priority(source_id, data_type)

            # 构建质量要求
            quality_requirements = self._build_quality_requirements(priority_config)

            # 执行质量检查（暂时使用模拟数据，实际应该从数据库获取最近数据）
            quality_result = await self._perform_quality_check(source_id, data_type, quality_requirements)

            if not quality_result['is_valid']:
                logger.debug(f"数据源 {source_id} 质量检查失败: {quality_result.get('issues', [])}")
                return False

            # 检查质量分数是否达到最低要求
            min_quality_score = quality_requirements.get('min_quality_score', 0.8)
            if quality_result.get('quality_score', 0) < min_quality_score:
                logger.debug(f"数据源 {source_id} 质量分数不足: {quality_result.get('quality_score', 0):.2f} < {min_quality_score}")
                return False

            logger.debug(f"数据源 {source_id} 质量检查通过 (分数: {quality_result.get('quality_score', 0):.2f})")
            return True

        except Exception as e:
            logger.warning(f"数据源 {source_id} 质量检查异常: {e}，允许调度以避免阻塞")
            return True  # 出错时允许调度，避免完全阻塞采集

    def _build_quality_requirements(self, priority_config) -> Dict[str, Any]:
        """构建质量要求"""
        # 根据优先级设置不同的质量要求
        quality_requirements = {
            'critical': {
                'min_quality_score': 0.95,
                'min_completeness': 0.98,
                'min_accuracy': 0.99,
                'max_age_days': 1
            },
            'high': {
                'min_quality_score': 0.90,
                'min_completeness': 0.95,
                'min_accuracy': 0.98,
                'max_age_days': 2
            },
            'medium': {
                'min_quality_score': 0.85,
                'min_completeness': 0.90,
                'min_accuracy': 0.95,
                'max_age_days': 7
            },
            'low': {
                'min_quality_score': 0.80,
                'min_completeness': 0.85,
                'min_accuracy': 0.90,
                'max_age_days': 30
            }
        }

        return quality_requirements.get(priority_config.priority_level, quality_requirements['medium'])

    async def _perform_quality_check(self, source_id: str, data_type: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行质量检查

        暂时使用模拟检查，实际应该：
        1. 从数据库获取该数据源的最近数据样本
        2. 调用质量验证器进行验证
        3. 返回详细的质量报告
        """
        try:
            # 模拟质量检查结果（基于数据源ID生成伪随机分数）
            # 实际实现应该从数据库获取真实数据进行质量分析

            # 使用数据源ID的哈希值生成伪随机质量分数
            import hashlib
            hash_value = int(hashlib.md5(source_id.encode()).hexdigest(), 16)
            base_score = 0.7 + (hash_value % 30) / 100  # 0.7-1.0之间的分数

            # 根据优先级调整分数（高优先级数据源通常有更好的质量）
            if 'core' in source_id.lower() or 'critical' in source_id.lower():
                quality_score = min(0.98, base_score + 0.1)
            elif 'index' in source_id.lower() or 'high' in source_id.lower():
                quality_score = min(0.95, base_score + 0.05)
            else:
                quality_score = base_score

            # 模拟质量验证结果
            is_valid = quality_score >= requirements.get('min_quality_score', 0.8)

            return {
                'is_valid': is_valid,
                'quality_score': quality_score,
                'issues': [] if is_valid else ['质量分数低于要求'],
                'warnings': [],
                'data_type': data_type,
                'requirements': requirements
            }

        except Exception as e:
            logger.warning(f"质量检查执行失败: {e}")
            return {
                'is_valid': True,  # 出错时允许调度
                'quality_score': 0.8,
                'issues': [],
                'warnings': [f'质量检查异常: {e}'],
                'error': str(e)
            }

    async def get_intelligent_scheduling_status(self) -> Dict[str, Any]:
        """获取智能调度状态"""
        return {
            'market_regime': {
                'current': self.market_regime.current_regime.value if self.market_regime else 'unknown',
                'confidence': self.market_regime.confidence if self.market_regime else 0.0,
                'last_check': datetime.fromtimestamp(self.last_regime_check).isoformat() if self.last_regime_check else None
            },
            'scheduler_adjustments': {
                'max_concurrent_tasks': self.max_concurrent_tasks,
                'check_interval': self.check_interval,
                'market_adaptive': True
            },
            'priority_manager': {
                'active': True,
                'priority_levels': self.priority_manager.get_priority_order()
            },
            'quality_monitoring': {
                'active': True,
                'validator_available': self.quality_validator is not None
            }
        }


# 全局调度器实例
_data_collection_scheduler: Optional[DataCollectionServiceScheduler] = None


def get_data_collection_scheduler() -> DataCollectionServiceScheduler:
    """
    获取数据采集调度器实例（单例模式）
    
    Returns:
        DataCollectionServiceScheduler: 调度器实例
    """
    global _data_collection_scheduler
    if _data_collection_scheduler is None:
        _data_collection_scheduler = DataCollectionServiceScheduler()
    return _data_collection_scheduler


async def start_data_collection_scheduler(startup_path: str = "unknown") -> bool:
    """
    启动数据采集调度器（便捷函数，向后兼容）
    
    Args:
        startup_path: 启动路径，用于追踪调度器是从哪里启动的
    
    Returns:
        bool: 是否启动成功
    """
    scheduler = get_data_collection_scheduler()
    return await scheduler.start(startup_path)


async def stop_data_collection_scheduler() -> bool:
    """
    停止数据采集调度器（便捷函数）
    
    Returns:
        bool: 是否停止成功
    """
    scheduler = get_data_collection_scheduler()
    return await scheduler.stop()
