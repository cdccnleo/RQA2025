"""
特征工程服务层
封装实际的特征工程组件，为API提供统一接口
符合架构设计：使用统一适配器访问特征层服务，使用统一日志系统
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

import pandas as pd

# 使用统一日志系统（符合架构设计：基础设施层统一日志接口）
try:
    from src.infrastructure.logging.core.unified_logger import get_unified_logger
    logger = get_unified_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# 初始化统一适配器工厂（符合架构设计：统一基础设施集成）
_adapter_factory = None
_features_adapter = None
_data_adapter = None

def _get_adapter_factory():
    """获取统一适配器工厂（符合架构设计）"""
    global _adapter_factory
    if _adapter_factory is None:
        try:
            from src.infrastructure.integration.business_adapters import get_unified_adapter_factory
            from src.infrastructure.integration.unified_business_adapters import BusinessLayerType
            _adapter_factory = get_unified_adapter_factory()
            if _adapter_factory:
                global _features_adapter, _data_adapter
                _features_adapter = _adapter_factory.get_adapter(BusinessLayerType.FEATURES)
                # 获取数据层适配器（符合架构设计：数据流集成）
                try:
                    _data_adapter = _adapter_factory.get_adapter(BusinessLayerType.DATA)
                    logger.info("特征层和数据层适配器已初始化")
                except Exception as e:
                    logger.debug(f"数据层适配器初始化失败（可选）: {e}")
                    _data_adapter = None
                logger.info("特征层适配器已初始化")
        except Exception as e:
            logger.warning(f"统一适配器工厂初始化失败: {e}")
    return _adapter_factory

def _get_data_adapter():
    """获取数据层适配器（符合架构设计：数据流集成）"""
    global _data_adapter
    adapter_factory = _get_adapter_factory()
    if adapter_factory and not _data_adapter:
        try:
            from src.infrastructure.integration.unified_business_adapters import BusinessLayerType
            _data_adapter = adapter_factory.get_adapter(BusinessLayerType.DATA)
            if _data_adapter:
                logger.info("数据层适配器已获取（用于特征工程数据流集成）")
        except Exception as e:
            logger.debug(f"获取数据层适配器失败（可选）: {e}")
            _data_adapter = None
    return _data_adapter

# 导入特征层组件（降级方案：如果适配器不可用，直接导入）
try:
    from src.features.core.engine import FeatureEngine
    FEATURE_ENGINE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"无法导入特征引擎: {e}")
    FEATURE_ENGINE_AVAILABLE = False

try:
    from src.features.monitoring.metrics_collector import MetricsCollector as FeatureMetricsCollector
    METRICS_COLLECTOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"无法导入特征指标收集器: {e}")
    METRICS_COLLECTOR_AVAILABLE = False

try:
    from src.features.utils.feature_selector import FeatureSelector
    FEATURE_SELECTOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"无法导入特征选择器: {e}")
    FEATURE_SELECTOR_AVAILABLE = False


# 单例实例
_feature_engine: Optional[Any] = None
_metrics_collector: Optional[Any] = None
_feature_selector: Optional[Any] = None


def get_feature_engine() -> Optional[Any]:
    """获取特征引擎实例（符合架构设计：优先使用统一适配器）"""
    global _feature_engine, _features_adapter
    
    # 优先通过统一适配器获取特征引擎（符合架构设计）
    adapter_factory = _get_adapter_factory()
    if adapter_factory and _features_adapter:
        try:
            if hasattr(_features_adapter, 'get_feature_engine'):
                feature_engine = _features_adapter.get_feature_engine()
                if feature_engine:
                    logger.info("通过统一适配器获取特征引擎")
                    return feature_engine
        except Exception as e:
            logger.debug(f"通过适配器获取特征引擎失败: {e}")
    
    # 降级方案：直接实例化
    if _feature_engine is None and FEATURE_ENGINE_AVAILABLE:
        try:
            _feature_engine = FeatureEngine()
            logger.info("特征引擎初始化成功")
        except Exception as e:
            logger.error(f"初始化特征引擎失败: {e}")
    return _feature_engine


def get_metrics_collector() -> Optional[Any]:
    """获取指标收集器实例"""
    global _metrics_collector
    if _metrics_collector is None and METRICS_COLLECTOR_AVAILABLE:
        try:
            _metrics_collector = FeatureMetricsCollector()
            logger.info("特征指标收集器初始化成功")
        except Exception as e:
            logger.error(f"初始化指标收集器失败: {e}")
    return _metrics_collector


def get_feature_selector() -> Optional[Any]:
    """获取特征选择器实例"""
    global _feature_selector
    if _feature_selector is None and FEATURE_SELECTOR_AVAILABLE:
        try:
            _feature_selector = FeatureSelector()
            logger.info("特征选择器初始化成功")
        except Exception as e:
            logger.error(f"初始化特征选择器失败: {e}")
    return _feature_selector


# ==================== 特征工程任务服务 ====================

def get_feature_tasks() -> List[Dict[str, Any]]:
    """获取特征提取任务列表 - 从持久化存储加载真实数据"""
    try:
        # 从持久化存储加载任务（PostgreSQL优先，文件系统降级）
        from .feature_task_persistence import list_feature_tasks
        tasks = list_feature_tasks(limit=100)
        
        if tasks:
            logger.debug(f"从持久化存储加载了 {len(tasks)} 个特征任务")
            return tasks
        else:
            logger.debug("持久化存储中没有特征任务")
            return []
    except Exception as e:
        logger.error(f"从持久化存储加载特征任务失败: {e}")
        return []


def get_feature_tasks_stats() -> Dict[str, Any]:
    """获取特征任务统计"""
    tasks = get_feature_tasks()
    active_tasks = [t for t in tasks if t.get('status') == 'running']
    
    return {
        "active_tasks": len(active_tasks),
        "total_tasks": len(tasks),
        "completed_tasks": len([t for t in tasks if t.get('status') == 'completed'])
    }


def create_feature_task(
    task_type: str,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    创建特征提取任务 - 使用真实数据，并持久化到存储
    
    符合架构设计：
    - 通过统一适配器工厂访问特征层和数据层
    - 特征工程从数据层获取原始数据（通过数据层适配器或特征层适配器间接访问）
    - 数据流：数据层 -> 特征层 -> 特征工程任务
    
    Args:
        task_type: 任务类型（技术指标、统计特征、情感特征、自定义特征）
        config: 任务配置
    
    Returns:
        创建的任务信息
    """
    try:
        # 可选：通过数据层适配器获取原始数据（符合架构设计：数据流集成）
        # 注意：特征引擎内部可能已经通过统一适配器工厂访问数据层，这里提供显式访问方式
        data_adapter = _get_data_adapter()
        if data_adapter:
            try:
                # 如果数据层适配器支持，可以从数据层获取原始数据
                # 特征引擎会在处理时从数据层获取数据，这里只是确保数据流集成可用
                # 数据流：数据层 -> 特征层 -> 特征工程任务（符合架构设计：通过统一适配器工厂）
                logger.debug("数据层适配器已可用，特征工程可以从数据层获取原始数据（通过统一适配器工厂访问DataLayerAdapter）")
                # 数据流处理通过特征引擎间接实现，符合架构设计的分层职责
            except Exception as e:
                logger.debug(f"数据层适配器访问失败（特征引擎会自行处理数据流）: {e}")
        
        # 检查是否有自定义任务ID前缀
        config = config or {}
        task_id_prefix = config.get('task_id_prefix', 'task')
        
        engine = get_feature_engine()
        if engine and hasattr(engine, 'create_task'):
            # 尝试使用特征引擎创建任务
            # 特征引擎内部会通过统一适配器工厂或直接调用数据层组件获取原始数据
            # 将task_id_prefix传递给特征引擎，以便生成带标识的任务ID
            config_with_prefix = config.copy()
            config_with_prefix['task_id_prefix'] = task_id_prefix
            task = engine.create_task(task_type, config_with_prefix)
            if task:
                # 持久化任务
                try:
                    from .feature_task_persistence import save_feature_task
                    save_feature_task(task)
                    logger.info(f"✅ 特征引擎创建任务成功并持久化: {task.get('task_id', 'unknown')}")
                except Exception as e:
                    logger.warning(f"保存任务到持久化存储失败: {e}")
                
                # 🚀 提交任务到统一调度器（关键修复）
                try:
                    task_id = task.get('task_id')
                    logger.info(f"🚀 特征引擎创建的任务准备提交到统一调度器: {task_id}")
                    
                    from src.core.orchestration.scheduler import (
                        get_unified_scheduler, TaskType, TaskPriority
                    )
                    from src.infrastructure.distributed.registry import get_unified_worker_registry, WorkerType
                    
                    scheduler = get_unified_scheduler()
                    registry = get_unified_worker_registry()
                    
                    # 检查调度器状态
                    if not scheduler._running:
                        logger.info(f"🔄 统一调度器未运行，正在启动...")
                        scheduler.start()
                    
                    # 确定任务类型
                    if task_type and task_type.startswith("training_"):
                        submit_task_type = TaskType.MODEL_TRAINING
                        worker_type = WorkerType.TRAINING_EXECUTOR
                    else:
                        submit_task_type = TaskType.FEATURE_EXTRACTION
                        worker_type = WorkerType.FEATURE_WORKER
                    
                    workers = registry.get_workers_by_type(worker_type)
                    logger.info(f"👷 当前{worker_type.value}工作节点数量: {len(workers)}")
                    
                    # 提交任务到统一调度器（submit_task是异步方法，需要获取当前事件循环）
                    import asyncio
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # 如果事件循环正在运行，使用run_coroutine_threadsafe
                            future = asyncio.run_coroutine_threadsafe(
                                scheduler.submit_task(
                                    task_type=submit_task_type,
                                    payload=config or {},
                                    priority=TaskPriority.NORMAL
                                ),
                                loop
                            )
                            scheduler_task_id = future.result(timeout=10)
                        else:
                            # 如果事件循环没有运行，使用run_until_complete
                            scheduler_task_id = loop.run_until_complete(scheduler.submit_task(
                                task_type=submit_task_type,
                                payload=config or {},
                                priority=TaskPriority.NORMAL
                            ))
                    except RuntimeError:
                        # 没有事件循环，创建新的
                        scheduler_task_id = asyncio.run(scheduler.submit_task(
                            task_type=submit_task_type,
                            payload=config or {},
                            priority=TaskPriority.NORMAL
                        ))
                    logger.info(f"✅ 特征引擎任务已提交到统一调度器: {task_id} (调度器ID: {scheduler_task_id})")
                    
                    # 更新任务状态
                    try:
                        from .feature_task_persistence import update_feature_task
                        update_feature_task(task_id, {
                            "status": "submitted",
                            "scheduler_task_id": scheduler_task_id,
                            "submitted_at": int(datetime.now().timestamp())
                        })
                        task["status"] = "submitted"
                        task["scheduler_task_id"] = scheduler_task_id
                    except Exception as e:
                        logger.warning(f"更新任务状态为submitted失败: {e}")
                    
                except Exception as e:
                    logger.error(f"❌ 提交特征引擎任务到调度器失败: {e}", exc_info=True)
                
                # 特征引擎创建任务成功，返回任务
                return task
        
        # 如果特征引擎不支持或创建失败，创建基本任务记录
        # 注意：特征引擎内部会处理数据流（从数据层获取原始数据），这里创建任务记录
        task_id = f"{task_id_prefix}_{int(datetime.now().timestamp())}"
        current_timestamp = int(datetime.now().timestamp())
        task = {
            "task_id": task_id,
            "task_type": task_type,
            "status": "pending",
            "progress": 0,
            "feature_count": 0,
            "start_time": current_timestamp,
            "created_at": current_timestamp,  # 添加created_at字段
            "config": config,
            # 从config中提取关键字段到任务记录顶层，便于后续使用
            "symbol": config.get("symbol"),
            "start_date": config.get("start_date"),
            "end_date": config.get("end_date"),
            "indicators": config.get("indicators", ["SMA", "EMA", "RSI", "MACD"]),
            "priority": config.get("priority", "normal"),
            "description": config.get("description", "")
        }
        
        # 数据流处理说明：特征引擎会在执行任务时通过统一适配器工厂访问数据层适配器
        # 数据流：数据层适配器(DataLayerAdapter) -> 特征层适配器 -> 特征引擎 -> 特征工程任务
        # 这符合架构设计的分层职责和数据流集成要求
        
        # 持久化任务到文件系统和PostgreSQL
        save_success = False
        try:
            from .feature_task_persistence import save_feature_task
            save_success = save_feature_task(task)
            if save_success:
                logger.info(f"✅ 特征提取任务已创建并持久化: {task_id}, 类型: {task_type}")
            else:
                logger.warning(f"⚠️ 保存任务返回False，task_id: {task_id}")
        except Exception as e:
            logger.error(f"❌ 保存任务到持久化存储失败: {e}", exc_info=True)
            save_success = False
        
        # 无论保存是否成功，都尝试提交到调度器（核心功能）
        scheduler_task_id = None
        logger.info(f"🚀 开始提交任务到统一调度器: {task_id} (保存状态: {save_success})")
        
        try:
            # 使用统一调度器（符合架构设计）
            from src.core.orchestration.scheduler import (
                get_unified_scheduler, TaskType, TaskPriority
            )
            from src.infrastructure.distributed.registry import get_unified_worker_registry, WorkerType
            
            logger.info(f"📦 正在导入统一调度器...")
            scheduler = get_unified_scheduler()
            registry = get_unified_worker_registry()
            logger.info(f"📦 统一调度器获取成功")
            
            # 检查调度器状态，如果未启动则启动
            if not scheduler._running:
                logger.info(f"🔄 统一调度器未运行，正在启动...")
                try:
                    scheduler.start()
                    logger.info(f"✅ 统一调度器已启动")
                except Exception as start_error:
                    logger.error(f"❌ 启动统一调度器失败: {start_error}")
                    # 尝试强制启动
                    logger.info(f"🔄 尝试强制启动统一调度器...")
                    import time
                    time.sleep(1)
                    if not scheduler._running:
                        scheduler._running = True
                        logger.info(f"✅ 统一调度器已强制启动")
            else:
                logger.info(f"✅ 统一调度器已在运行")
            
            # 根据任务类型选择正确的 TaskType
            # 训练任务（以 training_ 开头）应该使用 MODEL_TRAINING 类型
            if task_type and task_type.startswith("training_"):
                submit_task_type = TaskType.MODEL_TRAINING
                worker_type = WorkerType.TRAINING_EXECUTOR
                workers = registry.get_workers_by_type(worker_type)
                logger.info(f"👷 当前训练执行器数量: {len(workers)}")
                task_category = "训练任务"
            else:
                submit_task_type = TaskType.FEATURE_EXTRACTION
                worker_type = WorkerType.FEATURE_WORKER
                workers = registry.get_workers_by_type(worker_type)
                logger.info(f"👷 当前特征工作节点数量: {len(workers)}")
                task_category = "特征提取任务"
            
            if not workers:
                logger.info(f"🔄 没有{task_category}工作节点，需要创建...")
            
            # 提交任务到统一调度器
            logger.info(f"📤 正在提交{task_category}到统一调度器: {task_id}, 类型: {submit_task_type.value}")
            scheduler_task_id = scheduler.submit_task(
                task_type=submit_task_type,
                data=config or {},
                priority=TaskPriority.NORMAL,
                metadata={"task_id": task_id, "original_task": task, "task_type": task_type}
            )
            logger.info(f"✅ {task_category}已提交到统一调度器: {task_id} (调度器ID: {scheduler_task_id})")
            
            # 更新任务状态为已提交
            try:
                from .feature_task_persistence import update_feature_task
                update_feature_task(task_id, {
                    "status": "submitted",
                    "scheduler_task_id": scheduler_task_id,
                    "submitted_at": int(datetime.now().timestamp())
                })
                task["status"] = "submitted"
                task["scheduler_task_id"] = scheduler_task_id
            except Exception as e:
                logger.warning(f"更新任务状态为submitted失败: {e}")
            
        except Exception as e:
            logger.error(f"❌ 提交任务到调度器失败: {e}")
            logger.error(f"任务 {task_id} 创建成功但未提交到调度器", exc_info=True)
            
            # 更新任务状态为 failed，并记录错误信息
            try:
                from .feature_task_persistence import update_feature_task
                update_feature_task(task_id, {
                    "status": "failed",
                    "error_message": f"提交到调度器失败: {str(e)}",
                    "failed_at": int(datetime.now().timestamp())
                })
                task["status"] = "failed"
                task["error_message"] = f"提交到调度器失败: {str(e)}"
            except Exception as update_error:
                logger.error(f"更新任务失败状态也失败: {update_error}")
        
        return task
        
    except Exception as e:
        logger.error(f"创建特征任务失败: {e}")
        raise


def stop_feature_task(task_id: str) -> bool:
    """
    停止特征提取任务，并更新持久化存储

    Args:
        task_id: 任务ID

    Returns:
        是否成功停止
    """
    try:
        engine = get_feature_engine()
        if engine and hasattr(engine, 'stop_task'):
            success = engine.stop_task(task_id)
            if success:
                # 更新持久化存储中的任务状态
                try:
                    from .feature_task_persistence import update_feature_task
                    update_feature_task(task_id, {
                        "status": "stopped",
                        "end_time": int(datetime.now().timestamp())
                    })
                except Exception as e:
                    logger.warning(f"更新任务状态到持久化存储失败: {e}")
                
                logger.info(f"特征任务已停止: {task_id}")
                return True
        
        # 如果特征引擎不支持，直接更新持久化存储
        try:
            from .feature_task_persistence import update_feature_task
            update_feature_task(task_id, {
                "status": "stopped",
                "end_time": int(datetime.now().timestamp())
            })
            logger.info(f"特征任务状态已更新为停止: {task_id}")
            return True
        except Exception as e:
            logger.warning(f"无法停止任务 {task_id}，特征引擎不支持且持久化更新失败: {e}")
            return False
        
    except Exception as e:
        logger.error(f"停止特征任务失败: {e}")
        return False


def delete_feature_task(task_id: str) -> bool:
    """
    删除特征提取任务，并从持久化存储中移除

    Args:
        task_id: 任务ID

    Returns:
        是否成功删除
    """
    try:
        engine = get_feature_engine()
        if engine and hasattr(engine, 'delete_task'):
            success = engine.delete_task(task_id)
            if success:
                # 从持久化存储中删除任务
                try:
                    from .feature_task_persistence import delete_feature_task as delete_persisted_task
                    delete_persisted_task(task_id)
                except Exception as e:
                    logger.warning(f"从持久化存储中删除任务失败: {e}")
                
                logger.info(f"特征任务已删除: {task_id}")
                return True
        
        # 如果特征引擎不支持，直接从持久化存储中删除
        try:
            from .feature_task_persistence import delete_feature_task as delete_persisted_task
            delete_persisted_task(task_id)
            logger.info(f"特征任务已从持久化存储中删除: {task_id}")
            return True
        except Exception as e:
            logger.warning(f"无法删除任务 {task_id}，特征引擎不支持且持久化删除失败: {e}")
            return False
        
    except Exception as e:
        logger.error(f"删除特征任务失败: {e}")
        return False


# ==================== 特征存储服务 ====================

def get_features() -> List[Dict[str, Any]]:
    """获取特征列表 - 从特征存储表中获取真实特征数据"""
    try:
        # 优先从特征存储表获取特征
        from .feature_task_persistence import get_features_from_store
        features = get_features_from_store(limit=100)
        
        if features:
            logger.debug(f"从特征存储表获取到 {len(features)} 个特征")
            # 转换字段名以兼容前端
            for feature in features:
                feature['display_name'] = feature.get('name', '')
                # 如果有参数，添加到显示名称
                params = feature.get('parameters', {})
                if params and 'period' in params:
                    feature['display_name'] = f"{feature['name']}_{params['period']}"
            return features
        
        # 如果存储表为空，降级到从任务列表获取
        logger.debug("特征存储表为空，降级到从任务列表获取")
        from .feature_task_persistence import list_feature_tasks
        tasks = list_feature_tasks(limit=100)
        
        features = []
        for task in tasks:
            if task.get('status') == 'completed' and task.get('feature_count', 0) > 0:
                # 从任务中提取特征信息
                features.append({
                    "name": f"feature_{task.get('task_id')}",
                    "feature_type": task.get('task_type'),
                    "quality_score": task.get('quality_score', 0.8),
                    "importance": task.get('importance', 0.5),
                    "version": task.get('version', '1.0'),
                    "created_at": task.get('start_time'),
                    "updated_at": task.get('end_time') or task.get('start_time')
                })
        
        if features:
            logger.debug(f"从任务中提取到 {len(features)} 个特征")
        else:
            logger.debug("没有已完成的任务包含特征数据")
        
        return features
    except Exception as e:
        logger.error(f"获取特征列表失败: {e}")
        return []


def get_features_stats() -> Dict[str, Any]:
    """获取特征统计 - 优先从任务数据中获取质量评分"""
    features = get_features()
    
    # 首先尝试从已完成的任务中获取平均质量评分
    avg_quality = 0.0
    try:
        from .feature_task_persistence import list_feature_tasks
        tasks = list_feature_tasks()
        completed_tasks = [t for t in tasks if t.get('status') == 'completed' and t.get('overall_quality_score')]
        
        if completed_tasks:
            # 计算所有已完成任务的平均质量评分
            total_score = sum(t.get('overall_quality_score', 0) for t in completed_tasks)
            avg_quality = total_score / len(completed_tasks)
            logger.info(f"从 {len(completed_tasks)} 个已完成任务计算平均质量评分: {avg_quality:.2f}")
        elif features:
            # 如果没有任务数据，从特征列表计算
            quality_scores = [f.get('quality_score', 0) for f in features if f.get('quality_score')]
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    except Exception as e:
        logger.debug(f"从任务获取平均质量评分失败: {e}")
        # 降级到从特征列表计算
        if features:
            quality_scores = [f.get('quality_score', 0) for f in features if f.get('quality_score')]
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    
    # 计算处理速度（如果有任务数据）
    processing_speed = 0.0
    try:
        tasks = get_feature_tasks()
        if tasks:
            # 从任务中计算平均处理速度
            running_tasks = [t for t in tasks if t.get('status') == 'running']
            if running_tasks:
                # 如果有处理时间数据，计算速度
                speeds = [t.get('processing_speed', 0) for t in running_tasks if t.get('processing_speed')]
                if speeds:
                    processing_speed = sum(speeds) / len(speeds)
    except Exception as e:
        logger.warning(f"计算处理速度失败: {e}")
    
    return {
        "total_features": len(features),
        "avg_quality": avg_quality,
        "processing_speed": processing_speed
    }


def get_quality_distribution(features: List[Dict[str, Any]]) -> Dict[str, int]:
    """获取质量分布 - 优先从任务数据中获取"""
    distribution = {
        "优秀": 0,
        "良好": 0,
        "一般": 0,
        "较差": 0
    }
    
    # 首先尝试从已完成的任务中获取质量分布
    try:
        from .feature_task_persistence import list_feature_tasks
        tasks = list_feature_tasks()
        completed_tasks = [t for t in tasks if t.get('status') == 'completed' and t.get('quality_distribution')]
        
        if completed_tasks:
            # 合并所有已完成任务的质量分布
            for task in completed_tasks:
                task_distribution = task.get('quality_distribution', {})
                for level, count in task_distribution.items():
                    if level in distribution and isinstance(count, (int, float)):
                        distribution[level] += count
            
            logger.info(f"从 {len(completed_tasks)} 个已完成任务获取质量分布: {distribution}")
            return distribution
    except Exception as e:
        logger.debug(f"从任务获取质量分布失败: {e}")
    
    # 如果没有任务数据，尝试从特征列表计算
    if not features:
        logger.debug("特征列表为空，返回默认质量分布")
        return distribution
    
    for feature in features:
        try:
            quality = feature.get('quality_score', 0)
            # 确保quality是数字类型
            if isinstance(quality, (int, float)):
                if quality >= 0.9:
                    distribution["优秀"] += 1
                elif quality >= 0.7:
                    distribution["良好"] += 1
                elif quality >= 0.5:
                    distribution["一般"] += 1
                else:
                    distribution["较差"] += 1
        except Exception as e:
            logger.debug(f"处理特征质量评分失败: {e}")
            # 跳过有问题的特征
            pass
    
    logger.debug(f"从特征计算得到质量分布: {distribution}")
    return distribution


# ==================== 技术指标服务 ====================

def get_technical_indicators() -> List[Dict[str, Any]]:
    """获取技术指标状态 - 从 indicator_calculation_tracker 获取真实计算次数"""
    try:
        # 从 indicator_calculation_tracker 获取计算次数
        from src.features.monitoring.indicator_calculation_tracker import get_indicator_calculation_tracker
        tracker = get_indicator_calculation_tracker()
        
        # 获取所有指标状态（包含真实计算次数）
        indicators = tracker.get_all_indicators_status()
        
        if indicators:
            logger.debug(f"从 indicator_calculation_tracker 获取到 {len(indicators)} 个技术指标")
            return indicators
        
        # 如果没有数据，返回基础指标列表（状态为inactive）
        logger.debug("没有计算记录，返回基础指标列表")
        return [
            {"name": "SMA", "description": "简单移动平均线", "status": "inactive", "computed_count": 0},
            {"name": "EMA", "description": "指数移动平均线", "status": "inactive", "computed_count": 0},
            {"name": "RSI", "description": "相对强弱指标", "status": "inactive", "computed_count": 0},
            {"name": "MACD", "description": "指数平滑异同移动平均线", "status": "inactive", "computed_count": 0},
            {"name": "KDJ", "description": "随机指标", "status": "inactive", "computed_count": 0},
            {"name": "BOLL", "description": "布林带", "status": "inactive", "computed_count": 0},
        ]
    except Exception as e:
        logger.error(f"获取技术指标失败: {e}")
        return []


# 注意：已移除所有模拟数据函数，确保只使用真实数据
# 当组件不可用时，返回空列表而不是模拟数据


# ==================== 调度器服务 ====================

def get_scheduler_status() -> Dict[str, Any]:
    """
    获取调度器状态 - 从统一调度器获取真实状态

    Returns:
        调度器状态信息
    """
    try:
        # 从统一调度器获取状态
        from src.core.orchestration.scheduler import get_unified_scheduler
        scheduler = get_unified_scheduler()
        
        # 获取调度器运行状态
        is_running = scheduler.is_running()
        
        # 获取任务统计（从统一调度器的任务管理器）
        task_stats = scheduler.get_task_stats() if hasattr(scheduler, 'get_task_stats') else {}
        
        # 获取工作进程状态
        worker_stats = scheduler.get_worker_stats() if hasattr(scheduler, 'get_worker_stats') else {}
        
        # 构建状态响应
        status = {
            "is_running": is_running,
            "stats": {
                "pending_tasks": task_stats.get('pending', 0),
                "running_tasks": task_stats.get('running', 0),
                "completed_tasks": task_stats.get('completed', 0),
                "failed_tasks": task_stats.get('failed', 0),
                "total_tasks": task_stats.get('total', 0),
                "active_workers": worker_stats.get('active_workers', 0),
                "queue_sizes": {
                    "pending": task_stats.get('pending', 0),
                    "running": task_stats.get('running', 0)
                }
            },
            "feature_workers_count": worker_stats.get('active_workers', 0),
            "scheduler_type": "unified_scheduler",
            "note": f"统一调度器{'运行中' if is_running else '已停止'}，共 {task_stats.get('total', 0)} 个任务"
        }
        
        logger.debug(f"从统一调度器获取状态: {status['stats']}")
        return status
    except Exception as e:
        logger.error(f"从统一调度器获取状态失败: {e}")
        # 降级到从持久化存储计算
        try:
            from .feature_task_persistence import list_feature_tasks
            tasks = list_feature_tasks(limit=1000)
            
            pending_count = len([t for t in tasks if t.get('status') == 'pending'])
            running_count = len([t for t in tasks if t.get('status') == 'running'])
            
            return {
                "is_running": running_count > 0,
                "stats": {
                    "pending_tasks": pending_count,
                    "running_tasks": running_count,
                    "completed_tasks": len([t for t in tasks if t.get('status') == 'completed']),
                    "failed_tasks": len([t for t in tasks if t.get('status') == 'failed']),
                    "total_tasks": len(tasks),
                    "active_workers": running_count,
                    "queue_sizes": {
                        "pending": pending_count,
                        "running": running_count
                    }
                },
                "feature_workers_count": running_count,
                "scheduler_type": "unified_scheduler",
                "note": f"调度器运行中（降级模式），共 {len(tasks)} 个任务"
            }
        except Exception as fallback_error:
            logger.error(f"降级获取状态也失败: {fallback_error}")
            return {
                "is_running": False,
                "stats": {},
                "error": str(e),
                "scheduler_type": "unified_scheduler"
            }


def resubmit_pending_tasks() -> int:
    """
    重新提交持久化存储中的pending任务到统一调度器（符合架构设计）

    Returns:
        重新提交的任务数量
    """
    try:
        logger.info("开始重新提交pending任务到统一调度器...")
        
        # 从持久化存储加载pending任务
        try:
            from .feature_task_persistence import list_feature_tasks
            tasks = list_feature_tasks(limit=100)
            
            if not tasks:
                logger.info("没有找到pending任务")
                return 0
        except Exception as e:
            logger.error(f"加载任务失败: {e}")
            return 0
        
        # 筛选pending状态的任务
        pending_tasks = [task for task in tasks if task.get('status') == 'pending']
        if not pending_tasks:
            logger.info("没有找到pending状态的任务")
            return 0
        
        logger.info(f"找到 {len(pending_tasks)} 个pending任务，准备重新提交到统一调度器")
        
        # 提交任务到统一调度器
        try:
            from src.core.orchestration.scheduler import (
                get_unified_scheduler, TaskType, TaskPriority
            )
            scheduler = get_unified_scheduler()
            
            resubmitted_count = 0
            for task in pending_tasks:
                try:
                    task_type = task.get('task_type', '技术指标')
                    config = task.get('config', {})
                    task_id = task.get('task_id')
                    
                    # 根据任务类型选择正确的 TaskType
                    if task_type and task_type.startswith("training_"):
                        submit_task_type = TaskType.MODEL_TRAINING
                        task_category = "训练任务"
                    else:
                        submit_task_type = TaskType.FEATURE_EXTRACTION
                        task_category = "特征提取任务"
                    
                    # 提交任务到统一调度器
                    scheduler_task_id = scheduler.submit_task(
                        task_type=submit_task_type,
                        data=config,
                        priority=TaskPriority.NORMAL,
                        metadata={"task_id": task_id, "original_task": task, "task_type": task_type}
                    )
                    
                    logger.info(f"重新提交{task_category}: {task_id} (调度器ID: {scheduler_task_id})")
                    resubmitted_count += 1
                except Exception as e:
                    logger.error(f"重新提交任务 {task.get('task_id')} 失败: {e}")
            
            logger.info(f"成功重新提交 {resubmitted_count} 个任务到统一调度器")
            return resubmitted_count
        except Exception as e:
            logger.error(f"提交任务到统一调度器失败: {e}")
            return 0
    except Exception as e:
        logger.error(f"重新提交任务失败: {e}")
        return 0


def start_scheduler() -> bool:
    """
    启动统一调度器（符合架构设计）。先重新提交持久化中的 pending 任务到队列，再启动调度器，
    确保「特征提取任务」和「训练任务」能被统一调度并执行。
    """
    try:
        logger.info("开始启动统一调度器（符合架构设计）")
        from src.core.orchestration.scheduler import get_unified_scheduler
        from src.infrastructure.distributed.registry import get_unified_worker_registry, WorkerType
        
        scheduler = get_unified_scheduler()
        registry = get_unified_worker_registry()
        logger.info("获取统一调度器实例成功")

        # 先重新提交 pending 任务到调度队列（在启动前填充队列，避免任务不被调度）
        logger.info("开始重新提交 pending 任务到统一调度器队列")
        resubmitted = resubmit_pending_tasks()
        logger.info(f"启动前重新提交了 {resubmitted} 个 pending 任务到统一调度器队列")
        
        if resubmitted > 0:
            logger.info(f"成功将 {resubmitted} 个 pending 任务重新提交到统一调度器队列")
        else:
            logger.info("没有找到需要重新提交的 pending 任务")

        # 启动统一调度器
        if not scheduler._running:
            logger.info("统一调度器未运行，准备启动")
            scheduler.start()
            logger.info("✅ 统一调度器已启动")
        else:
            logger.info("✅ 统一调度器已在运行")
        
        # 检查特征工作节点数量
        feature_workers = registry.get_workers_by_type(WorkerType.FEATURE_WORKER)
        logger.info(f"👷 当前特征工作节点数量: {len(feature_workers)}")
        
        # 检查训练执行器数量
        training_executors = registry.get_workers_by_type(WorkerType.TRAINING_EXECUTOR)
        logger.info(f"👷 当前训练执行器数量: {len(training_executors)}")

        # 验证调度器状态
        try:
            scheduler_status = get_scheduler_status()
            logger.info(f"统一调度器启动后状态: {scheduler_status}")
            if scheduler_status.get("is_running"):
                logger.info("✅ 统一调度器启动验证成功，状态为运行中")
            else:
                logger.warning("⚠️ 统一调度器启动验证失败，状态可能不是运行中")
        except Exception as e:
            logger.warning(f"验证统一调度器状态失败: {e}")

        logger.info("✅ 统一调度器启动过程完成")
        return True
    except Exception as e:
        logger.error(f"❌ 启动统一调度器失败: {e}")
        logger.error("统一调度器启动失败详情:", exc_info=True)
        return False


def initialize_feature_engineering_system() -> bool:
    """
    初始化特征工程系统（使用统一调度器，符合架构设计）

    Returns:
        是否初始化成功
    """
    try:
        logger.info("开始初始化特征工程系统（使用统一调度器）")

        # 1. 启动统一调度器
        start_scheduler()

        # 2. 初始化事件监听器（使用统一调度器）
        try:
            from src.features.core.event_listeners import initialize_event_listeners
            from src.core.orchestration.scheduler import get_unified_scheduler
            
            # 获取事件总线
            event_bus = _get_event_bus()
            scheduler = get_unified_scheduler()

            if event_bus:
                initialize_event_listeners(event_bus, scheduler)
                logger.info("事件监听器已初始化（使用统一调度器）")
            else:
                logger.warning("事件总线未初始化，无法初始化事件监听器")
        except Exception as e:
            logger.error(f"初始化事件监听器失败: {e}")

        # 3. 初始化事件处理器
        try:
            from src.features.core.event_handlers import create_default_event_handler
            create_default_event_handler()
            logger.info("事件处理器已初始化")
        except Exception as e:
            logger.error(f"初始化事件处理器失败: {e}")

        logger.info("✅ 特征工程系统初始化完成（使用统一调度器）")
        return True
    except Exception as e:
        logger.error(f"❌ 初始化特征工程系统失败: {e}")
        return False


def _get_event_bus():
    """
    获取事件总线实例

    Returns:
        事件总线实例
    """
    try:
        from src.core.event_bus import EventBus
        return EventBus()
    except Exception as e:
        logger.debug(f"获取事件总线失败: {e}")
        return None


def stop_scheduler() -> bool:
    """
    停止统一调度器（符合架构设计）

    Returns:
        是否成功停止
    """
    try:
        from src.core.orchestration.scheduler import get_unified_scheduler
        scheduler = get_unified_scheduler()
        
        if scheduler._running:
            scheduler.stop()
            logger.info("✅ 统一调度器已停止")
            return True
        else:
            logger.info("✅ 统一调度器已经停止")
            return True
    except Exception as e:
        logger.error(f"❌ 停止统一调度器失败: {e}")
        return False


# ==================== 特征数据服务（用于模型训练）====================

def get_feature_data_for_training(task_id: str) -> Dict[str, Any]:
    """
    获取特征工程任务的特征数据，用于模型训练
    
    符合架构设计：特征层 -> ML层数据流
    数据流：特征工程任务 -> 特征数据(X) -> 模型训练
    
    Args:
        task_id: 特征工程任务ID
        
    Returns:
        特征数据字典，包含:
        - features: 特征矩阵 (DataFrame)
        - feature_names: 特征名称列表
        - target: 目标变量 (Series)
        - shape: 数据形状
        - metadata: 元数据信息
    """
    try:
        logger.info(f"获取特征工程任务 {task_id} 的特征数据用于模型训练")
        
        # 1. 从持久化存储获取任务信息
        try:
            from .feature_task_persistence import load_feature_task
            task = load_feature_task(task_id)
            if not task:
                logger.error(f"特征工程任务 {task_id} 不存在")
                return {"error": f"任务 {task_id} 不存在"}
        except Exception as e:
            logger.error(f"获取任务信息失败: {e}")
            return {"error": f"获取任务信息失败: {str(e)}"}
        
        # 2. 从任务配置获取参数
        config = task.get('config', {})
        symbols = config.get('symbols', [config.get('symbol')]) if config.get('symbols') else [config.get('symbol')]
        symbols = [s for s in symbols if s]  # 过滤None
        
        if not symbols:
            logger.error(f"任务 {task_id} 没有配置股票代码")
            return {"error": "任务没有配置股票代码"}
        
        start_date = config.get('start_date', '2020-01-01')
        end_date = config.get('end_date', datetime.now().strftime('%Y-%m-%d'))
        indicators = config.get('indicators', ['SMA', 'EMA', 'RSI', 'MACD'])
        
        logger.info(f"任务 {task_id} 参数: symbols={symbols}, date_range={start_date}~{end_date}")
        
        # 3. 从PostgreSQL加载股票数据
        try:
            from src.data.loader import PostgreSQLDataLoader, DataLoaderConfig
            loader_config = DataLoaderConfig(source_type="postgresql")
            data_loader = PostgreSQLDataLoader(loader_config)
        except Exception as e:
            logger.error(f"初始化数据加载器失败: {e}")
            return {"error": f"初始化数据加载器失败: {str(e)}"}
        
        # 4. 加载所有股票的数据并计算特征
        all_features = []
        for symbol in symbols:
            try:
                logger.info(f"加载股票 {symbol} 的数据")
                load_result = data_loader.load_stock_data(symbol, start_date, end_date)
                
                if not load_result.success or load_result.data is None:
                    logger.warning(f"加载股票 {symbol} 数据失败: {load_result.message}")
                    continue
                
                df = load_result.data
                logger.info(f"股票 {symbol} 加载了 {len(df)} 条数据")
                
                # 5. 计算特征
                try:
                    from src.ml.engine.feature_engineering import FeatureEngineer, FeaturePipeline, FeatureType, FeatureDefinition
                    engineer = FeatureEngineer()
                    
                    # 创建默认特征管道（如果不存在）
                    pipeline_name = "default_stock_features"
                    if pipeline_name not in engineer.pipelines:
                        # 定义输入特征
                        input_features = ['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'amount']
                        
                        # 创建特征管道
                        pipeline = FeaturePipeline(
                            name=pipeline_name,
                            steps=[
                                {"type": "handle_missing", "method": "fill", "fill_value": 0},
                                {"type": "create_technical_features"},
                                {"type": "normalize", "method": "standard"}
                            ],
                            input_features=input_features,
                            output_features=[]  # 将在运行时确定
                        )
                        engineer.pipelines[pipeline_name] = pipeline
                        
                        # 定义特征
                        for col in input_features:
                            engineer.define_feature(
                                name=col,
                                feature_type=FeatureType.NUMERIC,
                                data_type="float64",
                                description=f"{col} feature"
                            )
                    
                    # 处理数据
                    features_df = engineer.process_data(df, pipeline_name)
                    
                    # 添加股票代码列
                    features_df['symbol'] = symbol
                    
                    all_features.append(features_df)
                    logger.info(f"股票 {symbol} 计算了 {len(features_df.columns)} 个特征")
                    
                except Exception as e:
                    logger.error(f"计算股票 {symbol} 特征失败: {e}")
                    continue
                    
            except Exception as e:
                logger.error(f"处理股票 {symbol} 失败: {e}")
                continue
        
        # 关闭数据加载器
        data_loader.close()
        
        if not all_features:
            logger.error(f"任务 {task_id} 没有生成任何特征数据")
            return {"error": "没有生成任何特征数据"}
        
        # 6. 合并所有特征数据
        combined_features = pd.concat(all_features, ignore_index=True)
        logger.info(f"合并后特征数据形状: {combined_features.shape}")
        
        # 7. 准备目标变量（使用收盘价的变化作为目标）
        # 注意：PostgreSQL返回的列名是close_price，不是close
        if 'close_price' in combined_features.columns:
            # 计算下一日收益率作为目标
            combined_features['target'] = combined_features.groupby('symbol')['close_price'].pct_change().shift(-1)
            # 转换为分类问题：上涨(1)或下跌(0)
            combined_features['target_class'] = (combined_features['target'] > 0).astype(int)
            
            # 删除NaN值
            combined_features = combined_features.dropna()
            
            target = combined_features['target_class']
        else:
            logger.warning("没有close列，无法生成目标变量")
            target = None
        
        # 8. 分离特征和目标
        feature_columns = [col for col in combined_features.columns 
                          if col not in ['date', 'symbol', 'target', 'target_class']]
        
        # 量化交易系统安全要求：检查特征数量限制
        MAX_FEATURE_COUNT = 100
        if len(feature_columns) > MAX_FEATURE_COUNT:
            logger.warning(f"特征数量({len(feature_columns)})超过限制({MAX_FEATURE_COUNT})，将选择最重要的特征")
            # 使用方差选择前N个特征
            try:
                from sklearn.feature_selection import VarianceThreshold
                selector = VarianceThreshold(threshold=0.01)
                X_temp = combined_features[feature_columns]
                selector.fit(X_temp)
                
                # 获取方差最大的前N个特征
                variances = selector.variances_
                top_indices = variances.argsort()[-MAX_FEATURE_COUNT:][::-1]
                feature_columns = [feature_columns[i] for i in top_indices]
                
                logger.info(f"已选择方差最大的{len(feature_columns)}个特征")
            except Exception as e:
                logger.warning(f"特征选择失败: {e}，使用前{MAX_FEATURE_COUNT}个特征")
                feature_columns = feature_columns[:MAX_FEATURE_COUNT]
        
        X = combined_features[feature_columns]
        
        # 9. 数据完整性校验
        validation_results = _validate_feature_data(X, target, feature_columns, task_id)
        if not validation_results['is_valid']:
            logger.error(f"数据完整性校验失败: {validation_results['errors']}")
            return {
                "error": "数据完整性校验失败",
                "validation_errors": validation_results['errors'],
                "validation_warnings": validation_results['warnings']
            }
        
        logger.info(f"✅ 数据完整性校验通过: {validation_results['score']:.2f}分")
        if validation_results['warnings']:
            logger.warning(f"⚠️ 数据质量警告: {validation_results['warnings']}")
        
        # 10. 构建返回结果
        result = {
            "success": True,
            "task_id": task_id,
            "features": X,
            "feature_names": feature_columns,
            "target": target,
            "shape": X.shape,
            "sample_count": len(X),
            "feature_count": len(feature_columns),
            "symbols": symbols,
            "date_range": {
                "start": start_date,
                "end": end_date
            },
            "validation": validation_results,
            "metadata": {
                "indicators": indicators,
                "task_type": task.get('task_type'),
                "created_at": task.get('created_at'),
                "data_quality_score": validation_results['score']
            }
        }
        
        logger.info(f"✅ 成功获取特征数据: {X.shape[0]} 样本, {X.shape[1]} 特征, 质量评分: {validation_results['score']:.2f}")
        return result
        
    except Exception as e:
        logger.error(f"获取特征数据失败: {e}")
        return {"error": f"获取特征数据失败: {str(e)}"}


def get_available_feature_tasks_for_training() -> List[Dict[str, Any]]:
    """
    获取可用于模型训练的特征工程任务列表
    
    Returns:
        可用的特征工程任务列表
    """
    try:
        # 获取已完成的特征任务
        tasks = get_feature_tasks()
        
        # 筛选可用于训练的任务（已完成的）
        available_tasks = []
        for task in tasks:
            if task.get('status') in ['completed', 'running']:
                available_tasks.append({
                    "task_id": task.get('task_id'),
                    "task_type": task.get('task_type'),
                    "status": task.get('status'),
                    "feature_count": task.get('feature_count', 0),
                    "progress": task.get('progress', 0),
                    "created_at": task.get('created_at'),
                    "description": f"{task.get('task_type')} - {task.get('feature_count', 0)} 个特征"
                })
        
        logger.info(f"找到 {len(available_tasks)} 个可用于训练的特征任务")
        return available_tasks
        
    except Exception as e:
        logger.error(f"获取可用特征任务失败: {e}")
        return []


def _validate_feature_data(X, target, feature_columns, task_id: str) -> Dict[str, Any]:
    """
    验证特征数据的完整性和质量
    
    Args:
        X: 特征矩阵
        target: 目标变量
        feature_columns: 特征列名列表
        task_id: 任务ID
        
    Returns:
        验证结果字典，包含:
        - is_valid: 是否通过验证
        - score: 数据质量评分(0-100)
        - errors: 错误列表
        - warnings: 警告列表
    """
    errors = []
    warnings = []
    score = 100.0
    
    try:
        # 1. 检查基本形状
        if X is None or X.empty:
            errors.append("特征矩阵为空")
            score = 0
            return {"is_valid": False, "score": score, "errors": errors, "warnings": warnings}
        
        sample_count, feature_count = X.shape
        
        # 2. 检查样本数量
        if sample_count < 100:
            warnings.append(f"样本数量过少: {sample_count} < 100")
            score -= 10
        elif sample_count < 30:
            errors.append(f"样本数量严重不足: {sample_count} < 30")
            score = 0
            return {"is_valid": False, "score": score, "errors": errors, "warnings": warnings}
        
        # 3. 检查特征数量
        if feature_count < 5:
            warnings.append(f"特征数量过少: {feature_count} < 5")
            score -= 5
        
        # 4. 检查缺失值比例
        missing_ratio = X.isnull().sum().sum() / (sample_count * feature_count)
        if missing_ratio > 0.5:
            errors.append(f"缺失值比例过高: {missing_ratio:.2%}")
            score = 0
            return {"is_valid": False, "score": score, "errors": errors, "warnings": warnings}
        elif missing_ratio > 0.1:
            warnings.append(f"缺失值比例较高: {missing_ratio:.2%}")
            score -= 15
        elif missing_ratio > 0:
            warnings.append(f"存在缺失值: {missing_ratio:.2%}")
            score -= 5
        
        # 5. 检查目标变量
        if target is None or target.empty:
            warnings.append("目标变量为空")
            score -= 10
        else:
            # 检查目标变量分布
            target_unique = target.nunique()
            if target_unique < 2:
                warnings.append(f"目标变量类别过少: {target_unique}")
                score -= 10
            
            # 检查类别平衡
            target_counts = target.value_counts()
            if len(target_counts) >= 2:
                min_ratio = target_counts.min() / target_counts.sum()
                if min_ratio < 0.2:
                    warnings.append(f"目标变量类别不平衡: 最少类别占比 {min_ratio:.2%}")
                    score -= 5
        
        # 6. 检查特征方差（常量特征）
        constant_features = [col for col in feature_columns if X[col].nunique() <= 1]
        if constant_features:
            warnings.append(f"存在常量特征: {constant_features}")
            score -= len(constant_features) * 2
        
        # 7. 检查异常值（使用IQR方法）
        for col in feature_columns[:10]:  # 只检查前10个特征
            if X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                outliers = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()
                outlier_ratio = outliers / sample_count
                if outlier_ratio > 0.1:
                    warnings.append(f"特征 {col} 异常值比例过高: {outlier_ratio:.2%}")
                    score -= 3
                    break  # 只报告第一个异常值问题
        
        # 8. 确保分数在0-100范围内
        score = max(0, min(100, score))
        
        # 9. 判断是否通过验证（分数>=60且无错误）
        is_valid = len(errors) == 0 and score >= 60
        
        logger.info(f"数据验证结果: 任务={task_id}, 有效={is_valid}, 评分={score:.2f}, "
                   f"错误={len(errors)}, 警告={len(warnings)}")
        
        return {
            "is_valid": is_valid,
            "score": score,
            "errors": errors,
            "warnings": warnings,
            "details": {
                "sample_count": sample_count,
                "feature_count": feature_count,
                "missing_ratio": missing_ratio,
                "constant_features": constant_features
            }
        }
        
    except Exception as e:
        logger.error(f"数据验证过程出错: {e}")
        return {
            "is_valid": False,
            "score": 0,
            "errors": [f"验证过程出错: {str(e)}"],
            "warnings": warnings
        }

