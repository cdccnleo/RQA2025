"""
策略生命周期路由模块
提供策略生命周期管理API端点
"""

import logging
import os
import time
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

router = APIRouter()

# 导入生命周期管理器
from .strategy_lifecycle import (
    lifecycle_manager,
    create_strategy_lifecycle,
    get_strategy_lifecycle,
    transition_strategy_status,
    get_strategy_lifecycle_timeline,
    get_strategy_lifecycle_stats,
    LifecycleStatus
)


@router.post("/api/v1/strategy/{strategy_id}/lifecycle/create")
async def create_lifecycle_api(strategy_id: str, request: Dict[str, Any]):
    """创建策略生命周期"""
    try:
        strategy_name = request.get("strategy_name", "未知策略")
        
        lifecycle = create_strategy_lifecycle(strategy_id, strategy_name)
        
        return {
            "success": True,
            "strategy_id": lifecycle.strategy_id,
            "strategy_name": lifecycle.strategy_name,
            "current_status": lifecycle.current_status.value,
            "created_at": lifecycle.created_at,
            "message": "策略生命周期创建成功"
        }
    except Exception as e:
        logger.error(f"创建策略生命周期失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建失败: {str(e)}")


@router.get("/api/v1/strategy/{strategy_id}/lifecycle")
async def get_lifecycle_api(strategy_id: str):
    """获取策略生命周期"""
    try:
        lifecycle = get_strategy_lifecycle(strategy_id)
        
        if not lifecycle:
            return {
                "exists": False,
                "message": "该策略暂无生命周期记录"
            }
        
        return {
            "exists": True,
            "strategy_id": lifecycle.strategy_id,
            "strategy_name": lifecycle.strategy_name,
            "current_status": lifecycle.current_status.value,
            "created_at": lifecycle.created_at,
            "updated_at": lifecycle.updated_at,
            "stats": get_strategy_lifecycle_stats(strategy_id)
        }
    except Exception as e:
        logger.error(f"获取策略生命周期失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.post("/api/v1/strategy/{strategy_id}/lifecycle/transition")
async def transition_lifecycle_api(strategy_id: str, request: Dict[str, Any]):
    """转换策略生命周期状态"""
    try:
        new_status = request.get("new_status")
        operator = request.get("operator", "user")
        reason = request.get("reason", "")
        
        if not new_status:
            raise HTTPException(status_code=400, detail="缺少目标状态")
        
        success = transition_strategy_status(strategy_id, new_status, operator, reason)
        
        if not success:
            raise HTTPException(status_code=400, detail="状态转换失败，请检查当前状态是否允许转换到目标状态")
        
        lifecycle = get_strategy_lifecycle(strategy_id)
        
        return {
            "success": True,
            "message": f"状态已转换到: {new_status}",
            "strategy_id": strategy_id,
            "current_status": lifecycle.current_status.value if lifecycle else new_status
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"转换策略生命周期状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"转换失败: {str(e)}")


@router.get("/api/v1/strategy/{strategy_id}/lifecycle/timeline")
async def get_lifecycle_timeline_api(strategy_id: str):
    """获取策略生命周期时间线"""
    try:
        timeline = get_strategy_lifecycle_timeline(strategy_id)
        
        return {
            "strategy_id": strategy_id,
            "timeline": timeline,
            "total_events": len(timeline)
        }
    except Exception as e:
        logger.error(f"获取策略生命周期时间线失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.get("/api/v1/strategy/{strategy_id}/lifecycle/stats")
async def get_lifecycle_stats_api(strategy_id: str):
    """获取策略生命周期统计"""
    try:
        stats = get_strategy_lifecycle_stats(strategy_id)
        
        if "error" in stats:
            raise HTTPException(status_code=404, detail=stats["error"])
        
        return stats
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取策略生命周期统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.post("/api/v1/strategy/{strategy_id}/lifecycle/archive")
async def archive_strategy_api(strategy_id: str, request: Dict[str, Any] = None):
    """归档策略"""
    try:
        reason = request.get("reason", "策略归档") if request else "策略归档"
        
        success = lifecycle_manager.archive_strategy(strategy_id, reason)
        
        if not success:
            raise HTTPException(status_code=400, detail="归档策略失败")
        
        return {
            "success": True,
            "message": "策略已归档",
            "strategy_id": strategy_id
        }
    except Exception as e:
        logger.error(f"归档策略失败: {e}")
        raise HTTPException(status_code=500, detail=f"归档失败: {str(e)}")


@router.post("/api/v1/strategy/{strategy_id}/lifecycle/pause")
async def pause_strategy_api(strategy_id: str, request: Dict[str, Any] = None):
    """暂停策略"""
    try:
        reason = request.get("reason", "策略暂停") if request else "策略暂停"
        
        success = lifecycle_manager.pause_strategy(strategy_id, reason)
        
        if not success:
            raise HTTPException(status_code=400, detail="暂停策略失败")
        
        return {
            "success": True,
            "message": "策略已暂停",
            "strategy_id": strategy_id
        }
    except Exception as e:
        logger.error(f"暂停策略失败: {e}")
        raise HTTPException(status_code=500, detail=f"暂停失败: {str(e)}")


@router.post("/api/v1/strategy/{strategy_id}/lifecycle/resume")
async def resume_strategy_api(strategy_id: str, request: Dict[str, Any]):
    """恢复策略"""
    try:
        target_status = request.get("target_status")
        reason = request.get("reason", "策略恢复")
        
        if not target_status:
            raise HTTPException(status_code=400, detail="缺少目标状态")
        
        try:
            status_enum = LifecycleStatus(target_status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"无效的目标状态: {target_status}")
        
        success = lifecycle_manager.resume_strategy(strategy_id, status_enum, reason)
        
        if not success:
            raise HTTPException(status_code=400, detail="恢复策略失败")
        
        return {
            "success": True,
            "message": f"策略已恢复到: {target_status}",
            "strategy_id": strategy_id,
            "current_status": target_status
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"恢复策略失败: {e}")
        raise HTTPException(status_code=500, detail=f"恢复失败: {str(e)}")


@router.get("/api/v1/strategy/lifecycles")
async def list_lifecycles_api(
    status: Optional[str] = Query(None, description="状态筛选"),
    strategy_id: Optional[str] = Query(None, description="策略ID筛选")
):
    """列出策略生命周期"""
    try:
        status_enum = None
        if status:
            try:
                status_enum = LifecycleStatus(status)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"无效的状态: {status}")
        
        lifecycles = lifecycle_manager.list_lifecycles(status_enum)
        
        # 如果指定了策略ID，只返回该策略的生命周期
        if strategy_id:
            lifecycles = [l for l in lifecycles if l.strategy_id == strategy_id]
        
        return {
            "lifecycles": [
                {
                    "strategy_id": l.strategy_id,
                    "strategy_name": l.strategy_name,
                    "current_status": l.current_status.value,
                    "created_at": l.created_at,
                    "updated_at": l.updated_at
                }
                for l in lifecycles
            ],
            "total": len(lifecycles)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"列出策略生命周期失败: {e}")
        raise HTTPException(status_code=500, detail=f"列出失败: {str(e)}")


@router.get("/api/v1/strategy/lifecycles/overview")
async def get_lifecycles_overview_api():
    """获取策略生命周期概览"""
    try:
        lifecycles = lifecycle_manager.list_lifecycles()
        
        # 统计各状态数量
        status_counts = {}
        for status in LifecycleStatus:
            status_counts[status.value] = 0
        
        for lifecycle in lifecycles:
            status_counts[lifecycle.current_status.value] += 1
        
        return {
            "total_strategies": len(lifecycles),
            "status_distribution": status_counts,
            "recent_active": [
                {
                    "strategy_id": l.strategy_id,
                    "strategy_name": l.strategy_name,
                    "current_status": l.current_status.value,
                    "updated_at": l.updated_at
                }
                for l in sorted(lifecycles, key=lambda x: x.updated_at, reverse=True)[:10]
            ]
        }
    except Exception as e:
        logger.error(f"获取策略生命周期概览失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.get("/api/v1/strategy/lifecycle/models/available")
async def get_available_models_api():
    """获取可用的训练好的模型列表
    
    优先从PostgreSQL数据库获取，失败则降级到文件系统
    """
    models = []
    
    # 第一步：尝试从数据库获取模型列表
    try:
        from .model_persistence_service import ModelPersistenceService
        persistence_service = ModelPersistenceService()
        db_models = persistence_service.list_available_models(status='active', limit=100)
        
        if db_models:
            logger.info(f"从数据库获取到 {len(db_models)} 个模型")
            for model in db_models:
                model_info = {
                    "model_id": model.get('model_id'),
                    "strategy_id": model.get('model_id'),  # 使用model_id作为strategy_id
                    "description": f"{model.get('model_type', '未知类型')} 模型 (准确率: {model.get('accuracy', 'N/A')})",
                    "path": f"/app/models/{model.get('model_id')}/model.pkl",
                    "created_at": model.get('trained_at', 0),
                    "model_name": model.get('model_name'),
                    "model_type": model.get('model_type'),
                    "accuracy": model.get('accuracy'),
                    "source": "database"
                }
                models.append(model_info)
            
            return {
                "models": models,
                "total": len(models),
                "source": "database"
            }
    except Exception as e:
        logger.warning(f"从数据库获取模型列表失败，将降级到文件系统: {e}")
    
    # 第二步：数据库失败，降级到文件系统
    try:
        models_dir = "/app/models"
        
        if os.path.exists(models_dir):
            for model_id in os.listdir(models_dir):
                model_dir = os.path.join(models_dir, model_id)
                if os.path.isdir(model_dir):
                    # 检查该目录是否有训练好的模型文件
                    model_file = os.path.join(model_dir, 'model.pkl')
                    metadata_file = os.path.join(model_dir, 'metadata.json')

                    if os.path.exists(model_file):
                        # 读取元数据（如果存在）
                        description = f"模型 {model_id}"
                        try:
                            if os.path.exists(metadata_file):
                                with open(metadata_file, 'r') as f:
                                    metadata = json.load(f)
                                    model_type = metadata.get('model_type', '未知类型')
                                    metrics = metadata.get('metrics', {})
                                    accuracy = metrics.get('accuracy', 'N/A')
                                    description = f"{model_type} 模型 (准确率: {accuracy})"
                        except Exception as e:
                            logger.debug(f"读取模型元数据失败 {model_id}: {e}")

                        # 获取模型信息
                        model_info = {
                            "model_id": model_id,
                            "strategy_id": model_id,
                            "description": description,
                            "path": model_file,
                            "created_at": os.path.getctime(model_file),
                            "source": "filesystem"
                        }
                        models.append(model_info)

        # 按创建时间排序，最新的在前
        models.sort(key=lambda x: x['created_at'], reverse=True)
        
        logger.info(f"从文件系统获取到 {len(models)} 个模型")

        return {
            "models": models,
            "total": len(models),
            "source": "filesystem"
        }
    except Exception as e:
        logger.error(f"从文件系统获取模型列表也失败: {e}")
        # 返回空列表而不是报错
        return {
            "models": [],
            "total": 0
        }


# 模型验证配置
MIN_MODEL_ACCURACY = 0.6  # 最低模型准确率要求
MAX_POSITION_SIZE = 0.30  # 最大仓位限制30%

# 参数验证范围
PARAM_RANGES = {
    "prediction_threshold": (0.1, 0.9),
    "confidence_threshold": (0.5, 0.95),
    "position_sizing": ["equal", "confidence_based", "kelly"],
    "max_position_size": (0.05, MAX_POSITION_SIZE),
    "stop_loss": (0.01, 0.10),      # 止损 1%-10%
    "take_profit": (0.02, 0.20),    # 止盈 2%-20%
    "max_drawdown": (0.05, 0.15)    # 最大回撤 5%-15%
}


def validate_model_for_deployment(model_id: str) -> Dict[str, Any]:
    """
    验证模型是否适合部署
    
    Args:
        model_id: 模型ID
        
    Returns:
        Dict: 验证结果，包含模型信息或错误信息
    """
    # 尝试从数据库获取模型信息
    try:
        from .model_persistence_service import ModelPersistenceService
        persistence_service = ModelPersistenceService()
        model = persistence_service.get_model(model_id)
        
        if model:
            # 验证模型状态
            model_status = model.get('status', '')
            if model_status != 'trained':
                return {
                    "valid": False,
                    "error": f"模型状态无效: {model_status}，只有训练完成的模型才能部署"
                }
            
            # 验证模型质量
            accuracy = model.get('accuracy', 0)
            if accuracy < MIN_MODEL_ACCURACY:
                return {
                    "valid": False,
                    "error": f"模型准确率 {accuracy:.2%} 低于最低要求 {MIN_MODEL_ACCURACY:.0%}"
                }
            
            return {
                "valid": True,
                "model": model,
                "source": "database"
            }
    except Exception as e:
        logger.warning(f"从数据库验证模型失败 {model_id}: {e}")
    
    # 数据库失败，尝试从文件系统验证
    try:
        models_dir = "/app/models"
        model_dir = os.path.join(models_dir, model_id)
        model_file = os.path.join(model_dir, 'model.pkl')
        metadata_file = os.path.join(model_dir, 'metadata.json')
        
        # 检查模型文件是否存在
        if not os.path.exists(model_file):
            return {
                "valid": False,
                "error": f"模型文件不存在: {model_id}"
            }
        
        # 读取元数据
        model_data = {"model_id": model_id, "status": "trained"}
        if os.path.exists(metadata_file):
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                model_data.update(metadata)
                
                # 验证准确率
                metrics = metadata.get('metrics', {})
                accuracy = metrics.get('accuracy', 0)
                if accuracy < MIN_MODEL_ACCURACY:
                    return {
                        "valid": False,
                        "error": f"模型准确率 {accuracy:.2%} 低于最低要求 {MIN_MODEL_ACCURACY:.0%}"
                    }
        
        return {
            "valid": True,
            "model": model_data,
            "source": "filesystem"
        }
    except Exception as e:
        logger.error(f"从文件系统验证模型失败 {model_id}: {e}")
        return {
            "valid": False,
            "error": f"模型验证失败: {str(e)}"
        }


def validate_deployment_params(params: Dict[str, Any]) -> Dict[str, str]:
    """
    验证部署参数
    
    Args:
        params: 部署参数字典
        
    Returns:
        Dict: 错误信息字典，空字典表示验证通过
    """
    errors = {}
    
    # 验证预测阈值
    pred_thresh = params.get("prediction_threshold", 0.5)
    if not isinstance(pred_thresh, (int, float)):
        errors["prediction_threshold"] = "预测阈值必须是数字"
    elif not (PARAM_RANGES["prediction_threshold"][0] <= pred_thresh <= PARAM_RANGES["prediction_threshold"][1]):
        errors["prediction_threshold"] = f"预测阈值必须在 {PARAM_RANGES['prediction_threshold'][0]}-{PARAM_RANGES['prediction_threshold'][1]} 之间"
    
    # 验证置信度阈值
    conf_thresh = params.get("confidence_threshold", 0.7)
    if not isinstance(conf_thresh, (int, float)):
        errors["confidence_threshold"] = "置信度阈值必须是数字"
    elif not (PARAM_RANGES["confidence_threshold"][0] <= conf_thresh <= PARAM_RANGES["confidence_threshold"][1]):
        errors["confidence_threshold"] = f"置信度阈值必须在 {PARAM_RANGES['confidence_threshold'][0]}-{PARAM_RANGES['confidence_threshold'][1]} 之间"
    
    # 验证仓位管理方式
    pos_sizing = params.get("position_sizing", "equal")
    if pos_sizing not in PARAM_RANGES["position_sizing"]:
        errors["position_sizing"] = f"仓位管理方式必须是以下之一: {', '.join(PARAM_RANGES['position_sizing'])}"
    
    # 验证最大仓位
    max_pos = params.get("max_position_size", 0.1)
    if not isinstance(max_pos, (int, float)):
        errors["max_position_size"] = "最大仓位必须是数字"
    elif not (PARAM_RANGES["max_position_size"][0] <= max_pos <= PARAM_RANGES["max_position_size"][1]):
        errors["max_position_size"] = f"最大仓位必须在 {PARAM_RANGES['max_position_size'][0]:.0%}-{PARAM_RANGES['max_position_size'][1]:.0%} 之间"
    
    # 验证风控参数（必需）
    # 止损
    stop_loss = params.get("stop_loss")
    if stop_loss is None:
        errors["stop_loss"] = "止损比例是必需参数"
    elif not isinstance(stop_loss, (int, float)):
        errors["stop_loss"] = "止损比例必须是数字"
    elif not (PARAM_RANGES["stop_loss"][0] <= stop_loss <= PARAM_RANGES["stop_loss"][1]):
        errors["stop_loss"] = f"止损比例必须在 {PARAM_RANGES['stop_loss'][0]:.0%}-{PARAM_RANGES['stop_loss'][1]:.0%} 之间"
    
    # 止盈
    take_profit = params.get("take_profit")
    if take_profit is None:
        errors["take_profit"] = "止盈比例是必需参数"
    elif not isinstance(take_profit, (int, float)):
        errors["take_profit"] = "止盈比例必须是数字"
    elif not (PARAM_RANGES["take_profit"][0] <= take_profit <= PARAM_RANGES["take_profit"][1]):
        errors["take_profit"] = f"止盈比例必须在 {PARAM_RANGES['take_profit'][0]:.0%}-{PARAM_RANGES['take_profit'][1]:.0%} 之间"
    
    # 最大回撤
    max_drawdown = params.get("max_drawdown")
    if max_drawdown is None:
        errors["max_drawdown"] = "最大回撤限制是必需参数"
    elif not isinstance(max_drawdown, (int, float)):
        errors["max_drawdown"] = "最大回撤限制必须是数字"
    elif not (PARAM_RANGES["max_drawdown"][0] <= max_drawdown <= PARAM_RANGES["max_drawdown"][1]):
        errors["max_drawdown"] = f"最大回撤限制必须在 {PARAM_RANGES['max_drawdown'][0]:.0%}-{PARAM_RANGES['max_drawdown'][1]:.0%} 之间"
    
    return errors


@router.post("/api/v1/strategy/lifecycle/model/deploy")
async def deploy_model_strategy_api(request: Dict[str, Any]):
    """部署模型策略"""
    try:
        model_id = request.get("model_id")
        name = request.get("name")
        
        if not model_id:
            raise HTTPException(status_code=400, detail="模型ID不能为空")
        
        # 1. 验证模型
        model_validation = validate_model_for_deployment(model_id)
        if not model_validation["valid"]:
            logger.warning(f"模型验证失败: {model_validation['error']}")
            raise HTTPException(status_code=400, detail=f"模型验证失败: {model_validation['error']}")
        
        model_info = model_validation["model"]
        logger.info(f"模型验证通过: {model_id}, 来源: {model_validation['source']}")
        
        # 2. 验证参数
        param_errors = validate_deployment_params(request)
        if param_errors:
            error_msg = "; ".join([f"{k}: {v}" for k, v in param_errors.items()])
            logger.warning(f"参数验证失败: {error_msg}")
            raise HTTPException(status_code=400, detail=f"参数验证失败: {error_msg}")
        
        logger.info(f"参数验证通过: {model_id}")
        
        # 创建新的策略构思
        strategy_id = f"model_strategy_{int(time.time())}"
        
        # 构建策略构思数据
        strategy_conception = {
            "id": strategy_id,
            "name": name or f"模型策略-{model_id}",
            "type": "model_based",
            "description": f"基于模型 {model_id} 的策略",
            "target_market": "stock",
            "risk_level": "medium",
            "nodes": [],
            "connections": [],
            "parameters": {
                "model_id": model_id,
                "prediction_threshold": request.get("prediction_threshold", 0.5),
                "confidence_threshold": request.get("confidence_threshold", 0.7),
                "position_sizing": request.get("position_sizing", "equal"),
                "max_position_size": request.get("max_position_size", 0.1),
                # 添加风控参数
                "stop_loss": request.get("stop_loss"),
                "take_profit": request.get("take_profit"),
                "max_drawdown": request.get("max_drawdown")
            },
            "version": 1,
            "created_at": time.time(),
            "updated_at": time.time()
        }
        
        # 保存策略构思
        from .strategy_routes import save_strategy_conception
        result = save_strategy_conception(strategy_conception, auto_increment_version=False)
        
        # 创建生命周期记录
        create_strategy_lifecycle(strategy_id, name or f"模型策略-{model_id}")
        
        # 3. 将策略状态转换为模拟交易（必须经过模拟交易验证才能进入实盘）
        transition_strategy_status(strategy_id, "paper_trading", "system", "策略部署到模拟交易")
        
        # 记录部署审计日志
        try:
            audit_record = {
                "strategy_id": strategy_id,
                "strategy_name": name or f"模型策略-{model_id}",
                "model_id": model_id,
                "model_accuracy": model_info.get('accuracy', model_info.get('metrics', {}).get('accuracy', 0)),
                "deployment_params": {
                    "prediction_threshold": request.get("prediction_threshold", 0.5),
                    "confidence_threshold": request.get("confidence_threshold", 0.7),
                    "position_sizing": request.get("position_sizing", "equal"),
                    "max_position_size": request.get("max_position_size", 0.1),
                    "stop_loss": request.get("stop_loss"),
                    "take_profit": request.get("take_profit"),
                    "max_drawdown": request.get("max_drawdown")
                },
                "timestamp": time.time(),
                "operator": request.get("operator", "system"),
                "action": "deploy",
                "status": "paper_trading"
            }
            logger.info(f"策略部署审计记录: {audit_record}")
        except Exception as e:
            logger.warning(f"记录审计日志失败: {e}")
        
        # 启动策略执行
        try:
            from .strategy_execution_service import start_strategy
            await start_strategy(strategy_id)
            logger.info(f"策略 {strategy_id} 已启动执行")
        except Exception as e:
            logger.warning(f"启动策略执行失败 {strategy_id}: {e}")
        
        return {
            "success": True,
            "strategy_id": strategy_id,
            "message": "模型策略部署成功，已进入模拟交易阶段",
            "status": "paper_trading",
            "note": "策略需经过模拟交易验证并通过审批后才能进入实盘交易"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"部署模型策略失败: {e}")
        raise HTTPException(status_code=500, detail=f"部署失败: {str(e)}")


# 模拟交易验证配置
MIN_PAPER_TRADING_DAYS = 7  # 最小模拟交易天数
MIN_SHARPE_RATIO = 1.0      # 最小夏普比率
MAX_PAPER_DRAWDOWN = 0.10   # 最大允许回撤10%


def check_paper_trading_eligibility(strategy_id: str) -> Dict[str, Any]:
    """
    检查策略是否满足从模拟交易进入实盘交易的条件
    
    Args:
        strategy_id: 策略ID
        
    Returns:
        Dict: 检查结果，包含是否满足条件及详细信息
    """
    try:
        # 获取策略生命周期
        lifecycle = get_strategy_lifecycle(strategy_id)
        if not lifecycle:
            return {
                "eligible": False,
                "reason": "策略生命周期不存在"
            }
        
        # 检查当前状态
        if lifecycle.current_status.value != "paper_trading":
            return {
                "eligible": False,
                "reason": f"策略当前状态为 {lifecycle.current_status.value}，不在模拟交易阶段"
            }
        
        # 检查模拟交易时长
        if not lifecycle.paper_trading_started_at:
            return {
                "eligible": False,
                "reason": "模拟交易开始时间未记录"
            }
        
        paper_trading_days = (time.time() - lifecycle.paper_trading_started_at) / (24 * 3600)
        if paper_trading_days < MIN_PAPER_TRADING_DAYS:
            return {
                "eligible": False,
                "reason": f"模拟交易时间不足，已运行 {paper_trading_days:.1f} 天，需要至少 {MIN_PAPER_TRADING_DAYS} 天"
            }
        
        # TODO: 从策略性能评估服务获取实际性能指标
        # 这里使用模拟数据进行演示
        mock_performance = {
            "sharpe_ratio": 1.2,  # 假设夏普比率
            "max_drawdown": 0.08,  # 假设最大回撤
            "total_return": 0.05   # 假设总收益
        }
        
        # 检查夏普比率
        if mock_performance["sharpe_ratio"] < MIN_SHARPE_RATIO:
            return {
                "eligible": False,
                "reason": f"夏普比率 {mock_performance['sharpe_ratio']:.2f} 低于最低要求 {MIN_SHARPE_RATIO}",
                "metrics": mock_performance
            }
        
        # 检查最大回撤
        if mock_performance["max_drawdown"] > MAX_PAPER_DRAWDOWN:
            return {
                "eligible": False,
                "reason": f"最大回撤 {mock_performance['max_drawdown']:.2%} 超过限制 {MAX_PAPER_DRAWDOWN:.0%}",
                "metrics": mock_performance
            }
        
        return {
            "eligible": True,
            "paper_trading_days": paper_trading_days,
            "metrics": mock_performance,
            "message": "策略满足进入实盘交易的条件"
        }
    except Exception as e:
        logger.error(f"检查模拟交易资格失败 {strategy_id}: {e}")
        return {
            "eligible": False,
            "reason": f"检查失败: {str(e)}"
        }


@router.get("/api/v1/strategy/{strategy_id}/lifecycle/paper_trading/check")
async def check_paper_trading_status_api(strategy_id: str):
    """检查策略模拟交易状态及进入实盘交易的资格"""
    try:
        result = check_paper_trading_eligibility(strategy_id)
        return {
            "strategy_id": strategy_id,
            **result
        }
    except Exception as e:
        logger.error(f"检查模拟交易状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"检查失败: {str(e)}")


@router.post("/api/v1/strategy/{strategy_id}/lifecycle/submit_for_approval")
async def submit_for_approval_api(strategy_id: str, request: Dict[str, Any] = None):
    """提交策略进入实盘交易审批"""
    try:
        # 检查是否满足条件
        eligibility = check_paper_trading_eligibility(strategy_id)
        if not eligibility["eligible"]:
            raise HTTPException(
                status_code=400, 
                detail=f"策略不满足进入实盘交易的条件: {eligibility['reason']}"
            )
        
        # 检查当前状态
        lifecycle = get_strategy_lifecycle(strategy_id)
        if not lifecycle or lifecycle.current_status.value != "paper_trading":
            raise HTTPException(
                status_code=400,
                detail="策略不在模拟交易阶段，无法提交审批"
            )
        
        # 更新策略元数据，标记为待审批
        reason = request.get("reason", "申请进入实盘交易") if request else "申请进入实盘交易"
        
        # 记录审批申请
        approval_request = {
            "strategy_id": strategy_id,
            "strategy_name": lifecycle.strategy_name,
            "submitted_at": time.time(),
            "submitted_by": request.get("operator", "user") if request else "user",
            "reason": reason,
            "status": "pending",
            "paper_trading_days": eligibility.get("paper_trading_days", 0),
            "metrics": eligibility.get("metrics", {})
        }
        
        # 将审批请求存储到策略元数据
        if strategy_id not in lifecycle.metadata:
            lifecycle.metadata[strategy_id] = {}
        lifecycle.metadata[strategy_id]["approval_request"] = approval_request
        
        logger.info(f"策略 {strategy_id} 已提交进入实盘交易审批")
        
        return {
            "success": True,
            "message": "审批申请已提交，等待管理员审核",
            "approval_id": f"{strategy_id}_{int(time.time())}",
            "status": "pending"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"提交审批失败: {e}")
        raise HTTPException(status_code=500, detail=f"提交审批失败: {str(e)}")


@router.post("/api/v1/strategy/{strategy_id}/lifecycle/approve")
async def approve_strategy_api(strategy_id: str, request: Dict[str, Any] = None):
    """审批通过策略进入实盘交易（需要管理员权限）"""
    try:
        # TODO: 添加管理员权限验证
        # check_admin_permission(request.get("operator", ""))
        
        lifecycle = get_strategy_lifecycle(strategy_id)
        if not lifecycle:
            raise HTTPException(status_code=404, detail="策略不存在")
        
        # 检查是否有待审批的申请
        approval_request = lifecycle.metadata.get(strategy_id, {}).get("approval_request")
        if not approval_request or approval_request.get("status") != "pending":
            raise HTTPException(status_code=400, detail="没有待审批的申请")
        
        # 执行状态转换
        approver = request.get("approver", "admin") if request else "admin"
        approval_comment = request.get("comment", "审批通过") if request else "审批通过"
        
        success = transition_strategy_status(
            strategy_id, 
            "live_trading", 
            approver, 
            f"审批通过进入实盘交易: {approval_comment}"
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="状态转换失败")
        
        # 更新审批记录
        approval_request["status"] = "approved"
        approval_request["approved_at"] = time.time()
        approval_request["approved_by"] = approver
        approval_request["approval_comment"] = approval_comment
        
        logger.info(f"策略 {strategy_id} 已通过审批进入实盘交易")
        
        return {
            "success": True,
            "message": "策略已通过审批，进入实盘交易",
            "strategy_id": strategy_id,
            "approved_by": approver,
            "approved_at": approval_request["approved_at"]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"审批失败: {e}")
        raise HTTPException(status_code=500, detail=f"审批失败: {str(e)}")


@router.post("/api/v1/strategy/{strategy_id}/lifecycle/reject")
async def reject_strategy_api(strategy_id: str, request: Dict[str, Any]):
    """拒绝策略进入实盘交易（需要管理员权限）"""
    try:
        lifecycle = get_strategy_lifecycle(strategy_id)
        if not lifecycle:
            raise HTTPException(status_code=404, detail="策略不存在")
        
        approval_request = lifecycle.metadata.get(strategy_id, {}).get("approval_request")
        if not approval_request or approval_request.get("status") != "pending":
            raise HTTPException(status_code=400, detail="没有待审批的申请")
        
        reject_reason = request.get("reason", "不符合要求")
        rejecter = request.get("rejecter", "admin")
        
        # 更新审批记录
        approval_request["status"] = "rejected"
        approval_request["rejected_at"] = time.time()
        approval_request["rejected_by"] = rejecter
        approval_request["reject_reason"] = reject_reason
        
        logger.info(f"策略 {strategy_id} 进入实盘交易申请被拒绝: {reject_reason}")
        
        return {
            "success": True,
            "message": "申请已拒绝",
            "strategy_id": strategy_id,
            "reject_reason": reject_reason
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"拒绝申请失败: {e}")
        raise HTTPException(status_code=500, detail=f"拒绝申请失败: {str(e)}")
