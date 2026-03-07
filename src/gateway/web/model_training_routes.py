"""
模型训练API路由
提供训练任务、训练指标、资源使用等API接口
符合架构设计：使用EventBus进行事件通信，使用ServiceContainer进行依赖管理，使用BusinessProcessOrchestrator进行业务流程编排
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
import random
import logging
import time

# 量化交易系统安全要求：允许的股票代码白名单
# 可根据实际需求扩展
ALLOWED_STOCKS: Set[str] = {
    '000001', '000002', '000063', '000100', '000333', '000538', '000568', '000651', '000725', '000768',
    '000858', '000895', '002001', '002007', '002024', '002027', '002044', '002120', '002142', '002230',
    '002236', '002271', '002304', '002352', '002415', '002460', '002475', '002594', '002714', '002812',
    '300003', '300014', '300015', '300033', '300059', '300122', '300124', '300142', '300274', '300408',
    '300413', '300433', '300498', '300750', '600000', '600009', '600016', '600028', '600030', '600031',
    '600036', '600048', '600050', '600104', '600276', '600309', '600340', '600406', '600436', '600438',
    '600519', '600585', '600588', '600600', '600660', '600690', '600703', '600745', '600809', '600837',
    '600887', '600893', '600900', '601012', '601066', '601088', '601100', '601138', '601166', '601186',
    '601211', '601288', '601318', '601319', '601328', '601336', '601398', '601601', '601628', '601668',
    '601688', '601766', '601788', '601816', '601857', '601888', '601899', '601919', '601933', '601939',
    '601985', '601988', '601990', '603288', '603501', '603659', '603799', '603986', '688008', '688009',
    '688012', '688036', '688111', '688126', '688169', '688185', '688256', '688303', '688363', '688599'
}

# 量化交易系统安全要求：最大日期范围（5年）
MAX_DATE_RANGE_DAYS = 365 * 5

# 使用统一日志系统（符合架构设计：基础设施层统一日志接口）
try:
    from src.infrastructure.logging.core.unified_logger import get_unified_logger
    logger = get_unified_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

# 导入服务层
from .model_training_service import (
    get_training_jobs,
    get_training_jobs_stats,
    get_training_metrics
)

router = APIRouter()

# 全局服务容器（延迟初始化，符合架构设计）
_container = None

def _get_container():
    """获取服务容器实例（单例模式，符合架构设计）"""
    global _container
    if _container is None:
        try:
            from src.core.container.container import DependencyContainer
            _container = DependencyContainer()
            
            # 注册事件总线（符合架构设计：事件驱动通信）
            try:
                from src.core.event_bus.core import EventBus
                event_bus = EventBus()
                event_bus.initialize()
                _container.register(
                    "event_bus",
                    service=event_bus,
                    lifecycle="singleton"
                )
                logger.info("事件总线已注册到服务容器")
            except Exception as e:
                logger.warning(f"注册事件总线失败: {e}")
            
            logger.info("服务容器已初始化")
        except Exception as e:
            logger.warning(f"服务容器初始化失败: {e}")
            _container = None
    return _container

def _get_event_bus():
    """获取事件总线实例（符合架构设计）"""
    container = _get_container()
    if container:
        try:
            return container.resolve("event_bus")
        except Exception as e:
            logger.debug(f"从服务容器获取事件总线失败: {e}")
    
    # 降级方案：直接创建
    try:
        from src.core.event_bus.core import EventBus
        event_bus = EventBus()
        event_bus.initialize()
        return event_bus
    except Exception as e:
        logger.warning(f"创建事件总线失败: {e}")
        return None

def _get_orchestrator():
    """获取业务流程编排器实例（符合架构设计）"""
    container = _get_container()
    if container:
        try:
            return container.resolve("business_process_orchestrator")
        except Exception as e:
            logger.debug(f"从服务容器获取业务流程编排器失败: {e}")
    
    # 降级方案：直接创建（业务流程编排器已在MLCore中集成，这里提供全局访问点）
    try:
        from src.core.orchestration.orchestrator_refactored import BusinessProcessOrchestrator
        orchestrator = BusinessProcessOrchestrator()
        orchestrator.initialize()
        return orchestrator
    except Exception as e:
        logger.debug(f"创建业务流程编排器失败（可选功能，MLCore内部已集成）: {e}")
        return None

# ==================== 训练任务API ====================

@router.get("/ml/training/jobs")
async def get_training_jobs_endpoint() -> Dict[str, Any]:
    """获取训练任务列表 - 使用真实训练器数据，不使用模拟数据"""
    try:
        jobs = get_training_jobs()
        # 量化交易系统要求：不使用模拟数据，即使为空也返回真实结果
        stats = get_training_jobs_stats()
        
        return {
            "jobs": jobs,
            "stats": stats,
            "note": "量化交易系统要求使用真实训练数据。如果列表为空，表示当前没有训练任务。"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取训练任务失败: {str(e)}")


@router.post("/ml/training/jobs")
async def create_training_job(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    创建训练任务 - 使用真实数据，并持久化到存储，符合架构设计：事件驱动和业务流程编排
    
    支持从特征工程任务创建训练任务，实现特征层到ML层的数据流
    """
    try:
        from .model_training_service import get_model_trainer, get_ml_core
        
        model_type = request.get("model_type", "LSTM")
        config = request.get("config", {})
        
        # 检查是否从特征工程任务创建
        data_source = config.get("data_source", "historical")
        feature_task_id = config.get("feature_task_id")
        
        # 量化交易系统安全要求：强制启用风险控制功能
        deployment = config.get("deployment", {})
        risk_management = deployment.get("risk_management", {})
        
        # 强制启用风险评估
        if not risk_management.get("assess_risk"):
            logger.warning("风险评估未启用，根据量化交易系统要求强制启用")
            risk_management["assess_risk"] = True
            config["warning_messages"] = config.get("warning_messages", [])
            config["warning_messages"].append("风险评估已根据量化交易系统要求强制启用")
        
        # 强制启用异常检测
        if not risk_management.get("anomaly_detection"):
            logger.warning("异常检测未启用，根据量化交易系统要求强制启用")
            risk_management["anomaly_detection"] = True
            config["warning_messages"] = config.get("warning_messages", [])
            config["warning_messages"].append("异常检测已根据量化交易系统要求强制启用")
        
        # 更新配置
        deployment["risk_management"] = risk_management
        config["deployment"] = deployment
        
        # 量化交易系统安全要求：股票代码白名单验证
        symbols = config.get("symbols", [])
        if symbols:
            invalid_symbols = set(symbols) - ALLOWED_STOCKS
            if invalid_symbols:
                logger.error(f"包含不支持的股票代码: {invalid_symbols}")
                raise HTTPException(
                    status_code=400,
                    detail=f"包含不支持的股票代码: {list(invalid_symbols)}。"
                           f"请联系管理员添加支持或选择其他股票。"
                )
        
        # 量化交易系统安全要求：日期范围限制
        time_range = config.get("time_range", {})
        start_date_str = time_range.get("start_date")
        end_date_str = time_range.get("end_date")
        
        if start_date_str and end_date_str:
            try:
                start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
                end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
                days = (end_date - start_date).days
                
                if days > MAX_DATE_RANGE_DAYS:
                    logger.error(f"日期范围超出限制: {days}天 > {MAX_DATE_RANGE_DAYS}天")
                    raise HTTPException(
                        status_code=400,
                        detail=f"日期范围不能超过{MAX_DATE_RANGE_DAYS}天（约5年），"
                               f"当前选择: {days}天"
                    )
                
                if days < 30:
                    logger.warning(f"日期范围较短: {days}天，建议至少30天")
                    config["warning_messages"] = config.get("warning_messages", [])
                    config["warning_messages"].append(
                        f"日期范围仅{days}天，建议至少30天以获得更好的训练效果"
                    )
                    
            except ValueError as e:
                logger.error(f"日期格式错误: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"日期格式错误，请使用YYYY-MM-DD格式"
                )
        
        # 如果从特征工程创建，获取特征数据
        feature_data_info = None
        if data_source == "features" and feature_task_id:
            logger.info(f"从特征工程任务 {feature_task_id} 创建训练任务")
            try:
                from .feature_engineering_service import get_feature_data_for_training
                feature_result = get_feature_data_for_training(feature_task_id)
                
                if "error" in feature_result:
                    raise HTTPException(status_code=400, 
                                      detail=f"获取特征数据失败: {feature_result['error']}")
                
                feature_data_info = {
                    "task_id": feature_result["task_id"],
                    "feature_names": feature_result["feature_names"],
                    "shape": feature_result["shape"],
                    "sample_count": feature_result["sample_count"],
                    "feature_count": feature_result["feature_count"],
                    "symbols": feature_result["symbols"],
                    "date_range": feature_result["date_range"]
                }
                
                # 数据一致性检查
                config_symbols = config.get("symbols", [])
                feature_symbols = feature_result.get("symbols", [])
                
                if config_symbols and feature_symbols:
                    missing_symbols = set(config_symbols) - set(feature_symbols)
                    extra_symbols = set(feature_symbols) - set(config_symbols)
                    
                    if missing_symbols:
                        logger.warning(f"配置中的股票在特征工程任务中不存在: {missing_symbols}")
                        config["warning_messages"] = config.get("warning_messages", [])
                        config["warning_messages"].append(
                            f"配置中的股票 {list(missing_symbols)} 在特征工程任务中不存在，"
                            f"这些股票将使用原始数据"
                        )
                    
                    if extra_symbols:
                        logger.info(f"特征工程任务包含额外股票: {extra_symbols}，将使用特征数据")
                
                # 将特征数据存储到配置中，供训练时使用
                config["feature_data"] = feature_data_info
                config["feature_task_id"] = feature_task_id
                
                logger.info(f"成功获取特征数据: {feature_data_info['shape']}")
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"从特征工程获取数据失败: {e}")
                raise HTTPException(status_code=500, 
                                  detail=f"从特征工程获取数据失败: {str(e)}")
        
        # 可选：使用BusinessProcessOrchestrator管理模型训练任务创建业务流程（符合架构设计）
        orchestrator = _get_orchestrator()
        process_id = None
        if orchestrator:
            try:
                from src.core.orchestration.models.process_models import BusinessProcessState
                job_id_preview = f"job_{int(time.time())}"
                process_id = f"model_training_create_{job_id_preview}_{int(time.time())}"
                logger.debug(f"模型训练任务创建业务流程ID: {process_id}")
            except Exception as e:
                logger.debug(f"业务流程编排器初始化检查失败: {e}")
        
        # 尝试使用模型训练器创建任务
        model_trainer = get_model_trainer()
        ml_core = get_ml_core()
        
        job_id = f"job_{int(datetime.now().timestamp())}"
        job = {
            "job_id": job_id,
            "model_type": model_type,
            "status": "pending",
            "progress": 0,
            "accuracy": None,
            "loss": None,
            "start_time": int(datetime.now().timestamp()),
            "end_time": None,
            "training_time": 0,
            "config": config,
            "data_source": data_source,
            "feature_task_id": feature_task_id
        }
        
        # 如果模型训练器支持，尝试创建实际任务
        if model_trainer and hasattr(model_trainer, 'create_training_job'):
            try:
                created_job = model_trainer.create_training_job(model_type, config)
                if created_job:
                    if isinstance(created_job, dict):
                        job.update(created_job)
                    elif hasattr(created_job, '__dict__'):
                        job.update(created_job.__dict__)
            except Exception as e:
                logger.debug(f"使用模型训练器创建任务失败: {e}")
        
        # 持久化任务到文件系统和PostgreSQL
        try:
            from .training_job_persistence import save_training_job
            save_training_job(job)
            logger.info(f"训练任务已创建并持久化: {job_id}, 类型: {model_type}")
            
            # 发布训练任务创建事件到EventBus（符合架构设计：事件驱动通信）
            event_bus = _get_event_bus()
            if event_bus:
                try:
                    from src.core.event_bus.types import EventType
                    event_bus.publish(
                        EventType.TRAINING_JOB_CREATED,
                        {
                            "job_id": job_id,
                            "model_type": model_type,
                            "status": job.get("status", "pending"),
                            "config": config,
                            "process_id": process_id,
                            "timestamp": time.time()
                        },
                        source="model_training_routes"
                    )
                    logger.debug(f"已发布训练任务创建事件: {job_id}")
                except Exception as e:
                    logger.debug(f"发布训练任务创建事件失败: {e}")
        except Exception as e:
            logger.warning(f"保存任务到持久化存储失败: {e}")
            logger.info(f"训练任务已创建（未持久化）: {job_id}, 类型: {model_type}")
        
        # WebSocket广播模型训练任务创建事件（符合架构设计：实时更新）
        try:
            from .websocket_manager import manager
            await manager.broadcast("model_training", {
                "type": "training_job_created",
                "job_id": job_id,
                "model_type": model_type,
                "job": job,
                "timestamp": time.time()
            })
        except Exception as e:
            logger.debug(f"WebSocket广播失败: {e}")
        
        # 提交任务到统一调度器（符合架构设计）
        scheduler_task_id = None
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # 使用统一调度器（符合分布式协调器架构设计）
                from src.distributed.coordinator.unified_scheduler import (
                    get_unified_scheduler, TaskType, TaskPriority
                )
                from src.distributed.registry import get_unified_worker_registry, WorkerType
                
                scheduler = get_unified_scheduler()
                registry = get_unified_worker_registry()
                
                # 启动统一调度器（如果未启动）
                if not scheduler._running:
                    scheduler.start()
                    logger.info("✅ 统一调度器已启动")
                
                # 检查训练执行器状态
                training_executors = registry.get_workers_by_type(WorkerType.TRAINING_EXECUTOR)
                logger.info(f"👷 当前训练执行器数量: {len(training_executors)}")
                
                # 提交任务到统一调度器，使用 MODEL_TRAINING 类型
                scheduler_task_id = scheduler.submit_task(
                    task_type=TaskType.MODEL_TRAINING,
                    data=config or {},
                    priority=TaskPriority.NORMAL,
                    metadata={"job_id": job_id, "original_job": job, "model_type": model_type}
                )
                
                logger.info(f"✅ 训练任务已成功提交到统一调度器: {job_id} (调度器ID: {scheduler_task_id})")
                logger.debug(f"任务配置: model_type={model_type}, epochs={config.get('epochs', 100)}")
                
                # 获取调度器状态，确认任务已添加
                scheduler_stats = scheduler.get_scheduler_stats()
                logger.info(f"📊 统一调度器状态: 队列大小={scheduler_stats.get('queue_sizes', {})}, "
                           f"活跃工作节点={scheduler_stats.get('active_workers', 0)}")
                
                break  # 提交成功，退出重试循环
                
            except Exception as e:
                retry_count += 1
                logger.error(f"❌ 提交任务到统一调度器失败 (尝试 {retry_count}/{max_retries}): {e}")
                
                if retry_count < max_retries:
                    logger.info(f"⏱️ {retry_count}秒后重试...")
                    import time
                    time.sleep(retry_count)  # 指数退避
                else:
                    logger.error(f"❌ 提交任务到统一调度器失败，已达到最大重试次数: {job_id}")
                    # 即使调度器失败，任务记录已创建，可以手动执行
        
        return {
            "success": True,
            "job_id": job_id,
            "scheduler_task_id": scheduler_task_id,
            "job": job,
            "message": f"训练任务已创建: {model_type}",
            "scheduler_status": {
                "submitted": scheduler_task_id is not None,
                "retry_count": retry_count
            }
        }
    except HTTPException:
        # 重新抛出HTTPException，保持原始状态码
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建训练任务失败: {str(e)}")


@router.post("/ml/training/jobs/{job_id}/stop")
async def stop_training_job(job_id: str) -> Dict[str, Any]:
    """停止训练任务，并更新持久化存储，符合架构设计：事件驱动"""
    try:
        from .model_training_service import get_model_trainer
        
        # 尝试使用模型训练器停止任务
        model_trainer = get_model_trainer()
        success = False
        
        if model_trainer and hasattr(model_trainer, 'stop_training_job'):
            try:
                success = model_trainer.stop_training_job(job_id)
            except Exception as e:
                logger.debug(f"使用模型训练器停止任务失败: {e}")
        
        # 更新持久化存储中的任务状态
        try:
            from .training_job_persistence import update_training_job
            update_training_job(job_id, {
                "status": "stopped",
                "end_time": int(datetime.now().timestamp())
            })
            logger.info(f"训练任务状态已更新为停止: {job_id}")
            success = True
            
            # 发布训练任务停止事件到EventBus（符合架构设计：事件驱动通信）
            event_bus = _get_event_bus()
            if event_bus:
                try:
                    from src.core.event_bus.types import EventType
                    event_bus.publish(
                        EventType.TRAINING_JOB_STOPPED,
                        {
                            "job_id": job_id,
                            "status": "stopped",
                            "timestamp": time.time()
                        },
                        source="model_training_routes"
                    )
                    logger.debug(f"已发布训练任务停止事件: {job_id}")
                except Exception as e:
                    logger.debug(f"发布训练任务停止事件失败: {e}")
            
            # WebSocket广播模型训练任务停止事件（符合架构设计：实时更新）
            try:
                from .websocket_manager import manager
                await manager.broadcast("model_training", {
                    "type": "training_job_stopped",
                    "job_id": job_id,
                    "timestamp": time.time()
                })
            except Exception as e:
                logger.debug(f"WebSocket广播失败: {e}")
        except Exception as e:
            logger.warning(f"更新任务状态到持久化存储失败: {e}")
            if not success:
                # 如果训练器不支持且持久化更新失败，返回错误
                raise HTTPException(status_code=404, detail=f"任务 {job_id} 不存在或无法停止")
        
        return {
            "success": True,
            "message": f"训练任务 {job_id} 已停止"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"停止训练任务失败: {str(e)}")


@router.get("/ml/training/jobs/{job_id}")
async def get_training_job_details(job_id: str) -> Dict[str, Any]:
    """获取训练任务详情 - 使用真实训练器数据，不使用模拟数据"""
    try:
        jobs = get_training_jobs()
        # 量化交易系统要求：不使用模拟数据
        
        job = next((j for j in jobs if j.get('job_id') == job_id), None)
        if not job:
            raise HTTPException(status_code=404, detail=f"训练任务 {job_id} 不存在或尚未有训练数据")
        
        metrics = get_training_metrics(job_id)
        # 量化交易系统要求：不使用模拟数据，即使指标为空也返回真实结果
        job['metrics'] = metrics
        return job
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取训练任务详情失败: {str(e)}")


@router.delete("/ml/training/jobs/{job_id}")
async def delete_training_job(job_id: str) -> Dict[str, Any]:
    """删除训练任务 - 从持久化存储中删除，并广播删除事件"""
    try:
        # 检查任务是否存在
        jobs = get_training_jobs()
        job = next((j for j in jobs if j.get('job_id') == job_id), None)
        if not job:
            raise HTTPException(status_code=404, detail=f"训练任务 {job_id} 不存在")
        
        # 从持久化存储中删除任务
        try:
            from .training_job_persistence import delete_training_job
            delete_training_job(job_id)
            logger.info(f"训练任务已从持久化存储中删除: {job_id}")
        except Exception as e:
            logger.warning(f"从持久化存储中删除任务失败: {e}")
            # 继续执行，即使持久化存储删除失败
        
        # 发布训练任务删除事件到EventBus
        event_bus = _get_event_bus()
        if event_bus:
            try:
                from src.core.event_bus.types import EventType
                event_bus.publish(
                    EventType.TRAINING_JOB_DELETED,
                    {
                        "job_id": job_id,
                        "timestamp": time.time()
                    },
                    source="model_training_routes"
                )
                logger.debug(f"已发布训练任务删除事件: {job_id}")
            except Exception as e:
                logger.debug(f"发布训练任务删除事件失败: {e}")
        
        # WebSocket广播模型训练任务删除事件
        try:
            from .websocket_manager import manager
            await manager.broadcast("model_training", {
                "type": "training_job_deleted",
                "job_id": job_id,
                "timestamp": time.time()
            })
        except Exception as e:
            logger.debug(f"WebSocket广播失败: {e}")
        
        return {
            "success": True,
            "message": f"训练任务 {job_id} 已删除"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除训练任务失败: {str(e)}")


# ==================== 训练指标API ====================

@router.get("/ml/training/metrics")
async def get_training_metrics_endpoint() -> Dict[str, Any]:
    """获取训练指标 - 使用真实训练器数据，不使用模拟数据"""
    try:
        # 获取所有运行中任务的指标
        jobs = get_training_jobs()
        # 量化交易系统要求：不使用模拟数据
        
        running_jobs = [j for j in jobs if j.get('status') == 'running']
        if running_jobs:
            # 返回第一个运行中任务的指标
            job_id = running_jobs[0].get('job_id')
            metrics = get_training_metrics(job_id)
            # 量化交易系统要求：不使用模拟数据，即使指标为空也返回真实结果
            return metrics
        else:
            # 如果没有运行中的任务，返回空指标
            return {
                "history": {
                    "loss": [],
                    "accuracy": []
                },
                "resources": {
                    "gpu_usage": 0.0,
                    "cpu_usage": 0.0,
                    "memory_usage": 0.0
                },
                "hyperparameters": {},
                "note": "量化交易系统要求使用真实训练数据。当前没有运行中的训练任务。"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取训练指标失败: {str(e)}")


# ==================== 调度器API ====================

@router.get("/ml/training/scheduler/status")
async def get_scheduler_status() -> Dict[str, Any]:
    """获取模型训练调度器状态（使用统一调度器）"""
    try:
        # 使用统一调度器（符合架构设计）
        from src.distributed.coordinator.unified_scheduler import get_unified_scheduler
        from src.distributed.registry import get_unified_worker_registry, WorkerType
        
        scheduler = get_unified_scheduler()
        registry = get_unified_worker_registry()
        
        # 获取调度器状态
        is_running = scheduler._running
        
        # 获取统一调度器统计
        scheduler_stats = scheduler.get_scheduler_stats()
        
        # 获取训练执行器数量
        training_executors = registry.get_workers_by_type(WorkerType.TRAINING_EXECUTOR)
        
        # 获取任务统计（从持久化存储）
        jobs = get_training_jobs()
        total_tasks = len(jobs)
        completed_tasks = len([j for j in jobs if j.get('status') == 'completed'])
        failed_tasks = len([j for j in jobs if j.get('status') == 'failed'])
        pending_tasks = len([j for j in jobs if j.get('status') == 'pending'])
        
        return {
            "is_running": is_running,
            "stats": {
                "active_workers": len(training_executors),
                "queue_size": scheduler_stats.get('queue_sizes', {}).get('model_training', 0),
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "pending_tasks": pending_tasks
            },
            "scheduler_stats": scheduler_stats,
            "training_executors_count": len(training_executors),
            "scheduler_type": "unified_scheduler",
            "note": "使用统一调度器管理模型训练任务"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取调度器状态失败: {str(e)}")


@router.get("/ml/training/scheduler/health")
async def get_scheduler_health() -> Dict[str, Any]:
    """获取模型训练调度器健康状态（使用统一调度器）"""
    try:
        # 使用统一调度器（符合架构设计）
        from src.distributed.coordinator.unified_scheduler import get_unified_scheduler
        from src.distributed.registry import get_unified_worker_registry, WorkerType
        
        scheduler = get_unified_scheduler()
        registry = get_unified_worker_registry()
        
        # 获取调度器状态
        is_running = scheduler._running
        stats = scheduler.get_scheduler_stats()
        
        # 获取训练执行器数量
        training_executors = registry.get_workers_by_type(WorkerType.TRAINING_EXECUTOR)
        
        # 构建健康状态
        health_status = {
            "status": "healthy" if is_running else "stopped",
            "is_running": is_running,
            "active_workers": len(training_executors),
            "total_tasks": stats.get("total_tasks", 0),
            "pending_tasks": stats.get("pending_tasks", 0),
            "queue_sizes": stats.get("queue_sizes", {}),
            "scheduler_type": "unified_scheduler"
        }
        
        return {
            "success": True,
            "health": health_status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取调度器健康状态失败: {str(e)}")


@router.post("/ml/training/scheduler/start")
async def start_scheduler() -> Dict[str, Any]:
    """启动模型训练调度器（使用统一调度器）"""
    try:
        # 使用统一调度器（符合架构设计）
        from src.distributed.coordinator.unified_scheduler import get_unified_scheduler
        from src.distributed.registry import get_unified_worker_registry, WorkerType
        
        scheduler = get_unified_scheduler()
        registry = get_unified_worker_registry()
        
        # 启动统一调度器
        if not scheduler._running:
            scheduler.start()
            logger.info("✅ 统一调度器已启动")
        else:
            logger.info("✅ 统一调度器已在运行")
        
        # 检查训练执行器数量
        training_executors = registry.get_workers_by_type(WorkerType.TRAINING_EXECUTOR)
        
        return {
            "success": True,
            "message": "统一调度器已启动（管理模型训练任务）",
            "scheduler_type": "unified_scheduler",
            "is_running": scheduler._running,
            "training_executors_count": len(training_executors)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动调度器失败: {str(e)}")


@router.post("/ml/training/scheduler/stop")
async def stop_scheduler() -> Dict[str, Any]:
    """停止模型训练调度器（使用统一调度器）"""
    try:
        # 使用统一调度器（符合架构设计）
        from src.distributed.coordinator.unified_scheduler import get_unified_scheduler
        
        scheduler = get_unified_scheduler()
        
        # 停止统一调度器
        if scheduler._running:
            scheduler.stop()
            logger.info("🛑 统一调度器已停止")
        else:
            logger.info("🛑 统一调度器已经停止")
        
        return {
            "success": True,
            "message": "统一调度器已停止",
            "scheduler_type": "unified_scheduler",
            "is_running": scheduler._running
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"停止调度器失败: {str(e)}")


# ==================== 执行器管理API ====================

@router.post("/ml/training/executor/start")
async def start_executor() -> Dict[str, Any]:
    """启动模型训练任务执行器"""
    try:
        from .training_job_executor import start_training_job_executor, get_training_job_executor
        
        executor = await start_training_job_executor()
        
        if executor:
            logger.info("模型训练任务执行器已成功启动")
            return {
                "success": True,
                "message": "模型训练任务执行器已启动",
                "executor_status": {
                    "running": executor.running,
                    "worker_id": executor.worker_id
                }
            }
        else:
            logger.error("模型训练任务执行器启动失败")
            return {
                "success": False,
                "message": "模型训练任务执行器启动失败"
            }
            
    except Exception as e:
        logger.error(f"启动执行器失败: {e}")
        raise HTTPException(status_code=500, detail=f"启动执行器失败: {str(e)}")


@router.post("/ml/training/executor/stop")
async def stop_executor() -> Dict[str, Any]:
    """停止模型训练任务执行器"""
    try:
        from .training_job_executor import stop_training_job_executor
        
        await stop_training_job_executor()
        logger.info("模型训练任务执行器已成功停止")
        
        return {
            "success": True,
            "message": "模型训练任务执行器已停止"
        }
        
    except Exception as e:
        logger.error(f"停止执行器失败: {e}")
        raise HTTPException(status_code=500, detail=f"停止执行器失败: {str(e)}")


@router.get("/ml/training/executor/status")
async def get_executor_status() -> Dict[str, Any]:
    """获取模型训练任务执行器状态"""
    try:
        from .training_job_executor import get_training_job_executor
        
        executor = get_training_job_executor()
        
        if executor:
            status = {
                "running": executor.running,
                "worker_id": executor.worker_id,
                "scheduler_available": executor.scheduler is not None,
                "model_trainer_available": executor.model_trainer is not None,
                "last_updated": datetime.now().isoformat()
            }
            
            logger.info(f"执行器状态: running={status['running']}, worker_id={status['worker_id']}")
            return {
                "success": True,
                "status": status
            }
        else:
            logger.info("模型训练任务执行器未启动")
            return {
                "success": False,
                "message": "模型训练任务执行器未启动",
                "status": {
                    "running": False,
                    "last_updated": datetime.now().isoformat()
                }
            }
            
    except Exception as e:
        logger.error(f"获取执行器状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取执行器状态失败: {str(e)}")

