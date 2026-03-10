"""
数据采集调度管理器

根据数据源的采集频率配置，定时检查并自动生成采集任务。
"""

import threading
import logging
import time
from typing import Dict, Any, Optional, Set
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from src.gateway.web.rate_limit_parser import should_collect, parse_rate_limit
from src.core.orchestration.scheduler import (
    get_unified_scheduler, TaskType, TaskPriority
)

logger = logging.getLogger(__name__)


class DataCollectionSchedulerManager:
    """
    数据采集调度管理器
    
    定时检查已启用的数据源，根据采集频率自动生成采集任务。
    """
    
    def __init__(self, check_interval: int = 10):
        """
        初始化调度管理器
        
        Args:
            check_interval: 检查间隔（秒），默认10秒（缩短检查间隔以提高响应性）
        """
        self._running = False
        self._check_interval = check_interval
        self._scheduler_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # 跟踪已提交的任务，避免重复
        self._submitted_tasks: Set[str] = set()
        
        # 统计信息
        self._stats = {
            "total_checks": 0,
            "tasks_submitted": 0,
            "sources_checked": 0,
            "last_check_time": None,
            "next_check_time": None
        }
        
        logger.info(f"数据采集调度管理器初始化完成，检查间隔: {check_interval}秒")
    
    def start(self) -> bool:
        """
        启动调度管理器
        
        Returns:
            bool: 是否成功启动
        """
        with self._lock:
            if self._running:
                logger.debug("调度管理器已在运行中")
                return True
            
            self._running = True
            self._scheduler_thread = threading.Thread(
                target=self._scheduler_loop,
                name="DataCollectionScheduler",
                daemon=True
            )
            self._scheduler_thread.start()
            
            logger.info("✅ 数据采集调度管理器已启动")
            return True
    
    def stop(self) -> bool:
        """
        停止调度管理器
        
        Returns:
            bool: 是否成功停止
        """
        with self._lock:
            if not self._running:
                logger.debug("调度管理器未在运行")
                return True
            
            self._running = False
            
            if self._scheduler_thread:
                self._scheduler_thread.join(timeout=5)
            
            logger.info("🛑 数据采集调度管理器已停止")
            return True
    
    def _scheduler_loop(self):
        """调度器主循环"""
        logger.info("🔄 数据采集调度主循环已启动")
        
        while self._running:
            try:
                # 记录本次检查时间
                self._stats["last_check_time"] = datetime.now().isoformat()
                self._stats["next_check_time"] = None
                
                # 执行检查和调度
                self._check_and_schedule()
                
                # 计算下次检查时间
                next_check = datetime.now().timestamp() + self._check_interval
                self._stats["next_check_time"] = datetime.fromtimestamp(next_check).isoformat()
                
                # 休眠直到下次检查
                for _ in range(self._check_interval):
                    if not self._running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"❌ 调度循环错误: {e}")
                time.sleep(5)
        
        logger.info("🛑 数据采集调度主循环已停止")
    
    def _check_and_schedule(self):
        """
        检查数据源并生成采集任务
        """
        logger.debug("🔍 开始检查数据源...")
        
        try:
            # 获取所有已启用的数据源
            from src.gateway.web.data_source_config_manager import get_data_source_config_manager
            
            config_manager = get_data_source_config_manager()
            sources = config_manager.get_data_sources()
            
            enabled_sources = [s for s in sources if s.get("enabled", False)]
            
            self._stats["sources_checked"] = len(enabled_sources)
            self._stats["total_checks"] += 1
            
            logger.info(f"📊 检查 {len(enabled_sources)} 个已启用的数据源")
            
            # 检查每个数据源
            for source in enabled_sources:
                try:
                    self._check_source(source)
                except Exception as e:
                    logger.error(f"检查数据源失败 {source.get('id')}: {e}")
            
            # 清理已完成的任务记录
            self._cleanup_completed_tasks()
            
        except Exception as e:
            logger.error(f"检查数据源时出错: {e}")
    
    def _check_source(self, source: Dict[str, Any]):
        """
        检查单个数据源是否需要采集
        
        Args:
            source: 数据源配置
        """
        source_id = source.get("id")
        rate_limit = source.get("rate_limit", "")
        last_test = source.get("last_test")
        
        if not rate_limit:
            logger.debug(f"数据源 {source_id} 没有配置采集频率，跳过")
            return
        
        # 检查是否应该采集
        if should_collect(last_test, rate_limit):
            logger.info(f"🎯 数据源 {source_id} 到达采集时间，准备提交任务")
            
            # 检查是否已有待处理的任务
            if self._has_pending_task(source_id):
                logger.debug(f"数据源 {source_id} 已有待处理任务，跳过")
                return
            
            # 提交采集任务
            self._submit_collection_task(source_id, source)
        else:
            logger.debug(f"数据源 {source_id} 未到达采集时间")
    
    def _has_pending_task(self, source_id: str) -> bool:
        """
        检查是否已有待处理的任务
        
        Args:
            source_id: 数据源ID
            
        Returns:
            bool: 是否有待处理任务
        """
        # 生成任务标识
        task_key = f"{source_id}:{datetime.now().strftime('%Y%m%d')}"
        
        # 检查今天是否已提交过任务
        if task_key in self._submitted_tasks:
            return True
        
        # 检查统一调度器的任务队列
        try:
            scheduler = get_unified_scheduler()
            # 这里可以添加更复杂的检查逻辑
            # 例如检查调度器中是否有相同数据源的任务
        except Exception as e:
            logger.debug(f"检查任务队列失败: {e}")
        
        return False
    
    def _submit_collection_task(self, source_id: str, source_config: Dict[str, Any]):
        """
        提交采集任务到统一调度器
        
        Args:
            source_id: 数据源ID
            source_config: 数据源配置
        """
        import asyncio
        
        try:
            scheduler = get_unified_scheduler()
            
            # 准备任务数据
            task_data = {
                "source_id": source_id,
                "source_config": source_config,
                "collection_type": "scheduled",
                "submitted_at": datetime.now().isoformat()
            }
            
            # 定义任务完成回调
            def on_task_completed(task_id: str, status: str, result: Any, error: str):
                """任务完成回调"""
                logger.info(f"🎯 任务完成回调被调用: task_id={task_id}, source_id={source_id}, 状态={status}")
                
                if status == "completed":
                    logger.info(f"✅ 数据采集任务完成: {source_id}, 结果={result}")
                    # 更新数据源最后采集时间
                    try:
                        self._update_source_collection_time(source_id)
                        logger.info(f"✅ 已调用_update_source_collection_time: {source_id}")
                    except Exception as update_err:
                        logger.error(f"❌ 调用_update_source_collection_time失败: {source_id}, 错误={update_err}", exc_info=True)
                    
                    # 发布数据采集完成事件，触发后续业务流程
                    try:
                        from src.core.event_bus import get_event_bus
                        from src.core.event_bus.types import EventType
                        
                        event_bus = get_event_bus()
                        
                        # 获取数据源配置（用于特征工程模块）
                        source_config_for_event = source_config.copy() if source_config else {}
                        
                        event_bus.publish(
                            EventType.DATA_COLLECTION_COMPLETED,
                            {
                                "source_id": source_id,
                                "task_id": task_id,
                                "status": status,
                                "result": result,
                                "source_config": source_config_for_event,  # 添加source_config以兼容特征工程模块
                                "timestamp": datetime.now().isoformat(),
                                "collection_type": "scheduled"
                            },
                            source="data_collection_scheduler_manager"
                        )
                        logger.info(f"📢 数据采集完成事件已发布: {source_id}")
                    except Exception as event_err:
                        logger.warning(f"⚠️ 发布数据采集完成事件失败（非关键）: {source_id}, 错误={event_err}")
                    
                elif status == "failed":
                    logger.error(f"❌ 数据采集任务失败: {source_id}, 错误={error}")
                    
                    # 发布数据采集失败事件
                    try:
                        from src.core.event_bus import get_event_bus
                        from src.core.event_bus.types import EventType
                        
                        event_bus = get_event_bus()
                        event_bus.publish(
                            EventType.DATA_COLLECTION_FAILED,
                            {
                                "source_id": source_id,
                                "task_id": task_id,
                                "status": status,
                                "error": error,
                                "timestamp": datetime.now().isoformat()
                            },
                            source="data_collection_scheduler_manager"
                        )
                        logger.info(f"📢 数据采集失败事件已发布: {source_id}")
                    except Exception as event_err:
                        logger.warning(f"⚠️ 发布数据采集失败事件失败（非关键）: {source_id}, 错误={event_err}")
                
                # 从已提交任务集合中移除
                task_key = f"{source_id}:{datetime.now().strftime('%Y%m%d')}"
                if task_key in self._submitted_tasks:
                    self._submitted_tasks.discard(task_key)
                    logger.info(f"🗑️ 已从提交任务集合移除: {task_key}")
            
            # 提交任务（异步方法）
            async def submit_task_async():
                # 提交任务（使用枚举的值，而不是枚举本身）
                task_id = await scheduler.submit_task(
                    task_type=TaskType.DATA_COLLECTION.value,
                    payload=task_data,
                    priority=TaskPriority.NORMAL
                )
                
                # 注册任务完成回调
                worker_manager = scheduler._worker_manager
                worker_manager.register_task_callback(task_id, on_task_completed)
                
                return task_id
            
            # 在同步上下文中运行异步任务
            task_id = None
            try:
                # 尝试获取当前事件循环
                loop = asyncio.get_running_loop()
                # 如果已经有事件循环，使用run_coroutine_threadsafe
                future = asyncio.run_coroutine_threadsafe(submit_task_async(), loop)
                task_id = future.result(timeout=30)
            except RuntimeError:
                # 没有事件循环，创建新的事件循环
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    task_id = loop.run_until_complete(submit_task_async())
                finally:
                    loop.close()
            
            if task_id:
                # 记录已提交的任务
                task_key = f"{source_id}:{datetime.now().strftime('%Y%m%d')}"
                self._submitted_tasks.add(task_key)
                
                self._stats["tasks_submitted"] += 1
                
                logger.info(f"✅ 采集任务已提交: {task_id} (数据源: {source_id})")
                
                # 验证任务是否被记录
                try:
                    stats = scheduler.get_statistics()
                    logger.info(f"📊 提交后调度器任务统计: 总任务={stats.get('total_tasks', 0)}, 待处理={stats.get('pending_tasks', 0)}")
                except Exception as stats_err:
                    logger.debug(f"获取调度器统计失败: {stats_err}")
            else:
                logger.error(f"❌ 任务提交失败，未获取到任务ID: {source_id}")
            
        except Exception as e:
            logger.error(f"❌ 提交采集任务失败 {source_id}: {e}", exc_info=True)
    
    def _update_source_collection_time(self, source_id: str):
        """
        更新数据源最后采集时间
        
        Args:
            source_id: 数据源ID
        """
        try:
            from src.gateway.web.data_source_config_manager import get_data_source_config_manager
            
            logger.info(f"🔄 开始更新数据源最后采集时间: {source_id}")
            
            config_manager = get_data_source_config_manager()
            
            # 更新数据源配置
            source_config = config_manager.get_data_source(source_id)
            if source_config:
                now = datetime.now()
                now_iso = now.isoformat()
                now_str = now.strftime('%Y-%m-%d %H:%M:%S')
                
                old_collection_time = source_config.get("last_collection_time")
                old_test_time = source_config.get("last_test")
                
                # 同时更新 last_collection_time 和 last_test
                source_config["last_collection_time"] = now_iso
                source_config["last_test"] = now_str
                
                logger.info(f"📝 准备保存配置 - 数据源: {source_id}")
                logger.info(f"   last_collection_time: {old_collection_time} -> {now_iso}")
                logger.info(f"   last_test: {old_test_time} -> {now_str}")
                
                success = config_manager.update_data_source(source_id, source_config)
                
                if success:
                    logger.info(f"✅ 更新数据源时间成功: {source_id}")
                else:
                    logger.error(f"❌ 更新数据源时间失败: {source_id} - update_data_source返回False")
            else:
                logger.warning(f"⚠️ 未找到数据源配置: {source_id}")
            
        except Exception as e:
            logger.error(f"❌ 更新数据源采集时间失败 {source_id}: {e}", exc_info=True)
    
    def _cleanup_completed_tasks(self):
        """清理已完成的任务记录"""
        # 只保留最近3天的任务记录
        current_date = datetime.now().strftime('%Y%m%d')
        tasks_to_remove = []
        
        for task_key in self._submitted_tasks:
            # task_key 格式: source_id:YYYYMMDD
            if ':' in task_key:
                date_part = task_key.split(':')[-1]
                if date_part != current_date:
                    tasks_to_remove.append(task_key)
        
        for task_key in tasks_to_remove:
            self._submitted_tasks.discard(task_key)
        
        if tasks_to_remove:
            logger.debug(f"清理了 {len(tasks_to_remove)} 个历史任务记录")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取调度管理器统计信息
        
        Returns:
            Dict: 统计信息
        """
        return {
            "running": self._running,
            "check_interval": self._check_interval,
            "total_checks": self._stats["total_checks"],
            "tasks_submitted": self._stats["tasks_submitted"],
            "sources_checked": self._stats["sources_checked"],
            "last_check_time": self._stats["last_check_time"],
            "next_check_time": self._stats["next_check_time"],
            "pending_tasks_count": len(self._submitted_tasks)
        }
    
    def force_check(self):
        """
        强制立即执行一次检查
        """
        logger.info("🚀 强制执行数据源检查")
        self._check_and_schedule()


# 全局实例
_scheduler_manager: Optional[DataCollectionSchedulerManager] = None


def get_scheduler_manager() -> DataCollectionSchedulerManager:
    """
    获取全局调度管理器实例（单例模式）
    
    Returns:
        DataCollectionSchedulerManager: 调度管理器实例
    """
    global _scheduler_manager
    if _scheduler_manager is None:
        _scheduler_manager = DataCollectionSchedulerManager()
    return _scheduler_manager


def start_auto_collection() -> bool:
    """
    启动自动采集
    
    Returns:
        bool: 是否成功启动
    """
    manager = get_scheduler_manager()
    return manager.start()


def stop_auto_collection() -> bool:
    """
    停止自动采集
    
    Returns:
        bool: 是否成功停止
    """
    manager = get_scheduler_manager()
    return manager.stop()


def get_auto_collection_status() -> Dict[str, Any]:
    """
    获取自动采集状态
    
    Returns:
        Dict: 状态信息
    """
    manager = get_scheduler_manager()
    return manager.get_stats()


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("数据采集调度管理器测试")
    print("=" * 60)
    
    # 获取管理器实例
    manager = get_scheduler_manager()
    
    # 测试启动
    print("\n1. 测试启动调度管理器")
    result = manager.start()
    print(f"✅ 启动结果: {result}")
    
    # 获取状态
    print("\n2. 获取状态")
    stats = manager.get_stats()
    print(f"运行状态: {stats['running']}")
    print(f"检查间隔: {stats['check_interval']}秒")
    
    # 等待几秒
    print("\n3. 等待5秒...")
    time.sleep(5)
    
    # 再次获取状态
    stats = manager.get_stats()
    print(f"检查次数: {stats['total_checks']}")
    print(f"最后检查: {stats['last_check_time']}")
    
    # 测试停止
    print("\n4. 测试停止调度管理器")
    result = manager.stop()
    print(f"✅ 停止结果: {result}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
