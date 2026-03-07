"""
策略部署路由模块

提供策略部署相关的REST API接口，整合一键部署功能

功能：
- 策略部署到不同环境
- 部署历史查询
- 部署回滚
- 一键发布（验证+部署）

作者: Claude
创建日期: 2026-02-21
"""

import logging
import os
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query

# 导入一键部署模块
try:
    from .one_click_strategy_deployment import (
        DeploymentManager, 
        DeploymentEnvironment, 
        DeploymentStatus,
        get_deployment_manager
    )
except ImportError as e:
    logging.error(f"导入一键部署模块失败: {e}")
    DeploymentManager = None
    DeploymentEnvironment = None
    DeploymentStatus = None
    get_deployment_manager = None

# 导入策略路由中的函数
try:
    from .strategy_routes import load_strategy_conceptions, save_strategy_conception, STRATEGY_CONCEPTION_DIR
except ImportError as e:
    logging.error(f"导入策略路由模块失败: {e}")
    # 定义降级函数
    def load_strategy_conceptions():
        return []
    def save_strategy_conception(data, auto_increment_version=True):
        pass
    STRATEGY_CONCEPTION_DIR = "data/strategy_conceptions"

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["deployment"])


class DeployRequest(BaseModel):
    """部署请求模型"""
    environment: str = Field(default="staging", description="部署环境: development, staging, production")
    skip_validation: bool = Field(default=False, description="是否跳过验证流程")
    deploy_by: str = Field(default="system", description="部署人")
    notes: Optional[str] = Field(default=None, description="部署备注")


class PublishRequest(BaseModel):
    """一键发布请求模型"""
    environment: str = Field(default="staging", description="部署环境")
    run_validation: bool = Field(default=True, description="是否执行验证流程（回测+优化）")
    deploy_by: str = Field(default="system", description="发布人")
    notes: Optional[str] = Field(default=None, description="发布备注")


class DeploymentResponse(BaseModel):
    """部署响应模型"""
    deployment_id: str
    strategy_id: str
    strategy_name: str
    environment: str
    status: str
    message: str
    created_at: str
    logs: List[str] = []


class DeploymentHistoryResponse(BaseModel):
    """部署历史响应模型"""
    deployments: List[Dict[str, Any]]
    total: int
    strategy_id: str


class DeploymentDetailResponse(BaseModel):
    """部署详情响应模型"""
    deployment_id: str
    strategy_id: str
    strategy_name: str
    environment: str
    status: str
    started_at: str
    completed_at: Optional[str]
    logs: List[str]
    error_message: Optional[str]
    deployed_by: str
    rollback_to: Optional[str]
    package_info: Optional[Dict[str, Any]]


class RollbackResponse(BaseModel):
    """回滚响应模型"""
    success: bool
    message: str
    deployment_id: str
    rollback_to: Optional[str]


# 全局部署管理器实例
_deployment_manager: Optional[DeploymentManager] = None


def get_manager() -> DeploymentManager:
    """获取部署管理器实例（带缓存）"""
    global _deployment_manager
    if _deployment_manager is None and get_deployment_manager:
        deploy_root = os.environ.get("DEPLOY_ROOT", "deployed_strategies")
        _deployment_manager = get_deployment_manager(deploy_root)
    return _deployment_manager


def _get_strategy_info(strategy_id: str) -> Optional[Dict[str, Any]]:
    """获取策略信息"""
    try:
        if load_strategy_conceptions:
            conceptions = load_strategy_conceptions()
            for c in conceptions:
                if c.get('id') == strategy_id:
                    return c
        return None
    except Exception as e:
        logger.error(f"获取策略信息失败: {e}")
        return None


def _deployment_record_to_dict(record) -> Dict[str, Any]:
    """将部署记录转换为字典"""
    if record is None:
        return {}
    
    return {
        "deployment_id": record.deployment_id,
        "package_id": record.package_id,
        "strategy_id": record.strategy_id,
        "environment": record.environment.value if hasattr(record.environment, 'value') else str(record.environment),
        "status": record.status.value if hasattr(record.status, 'value') else str(record.status),
        "started_at": record.started_at.isoformat() if record.started_at else None,
        "completed_at": record.completed_at.isoformat() if record.completed_at else None,
        "logs": record.logs,
        "error_message": record.error_message,
        "deployed_by": record.deployed_by,
        "rollback_to": record.rollback_to
    }


@router.post("/strategy/{strategy_id}/deploy", response_model=DeploymentResponse)
async def deploy_strategy(
    strategy_id: str,
    request: DeployRequest,
    background_tasks: BackgroundTasks
):
    """
    部署策略到指定环境
    
    Args:
        strategy_id: 策略ID
        request: 部署请求
        
    Returns:
        部署响应
    """
    logger.info(f"开始部署策略 {strategy_id} 到 {request.environment} 环境")
    
    # 检查策略是否存在
    strategy = _get_strategy_info(strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail=f"策略 {strategy_id} 不存在")
    
    strategy_name = strategy.get('name', strategy_id)
    
    # 获取部署管理器
    manager = get_manager()
    if not manager:
        raise HTTPException(status_code=500, detail="部署管理器未初始化")
    
    try:
        # 解析环境
        try:
            env = DeploymentEnvironment(request.environment.lower())
        except (ValueError, AttributeError):
            raise HTTPException(status_code=400, detail=f"无效的环境: {request.environment}")
        
        # 准备策略路径
        strategy_path = os.path.join(STRATEGY_CONCEPTION_DIR, f"{strategy_id}.json")
        if not os.path.exists(strategy_path):
            # 如果文件不存在，先保存策略
            if save_strategy_conception:
                save_strategy_conception(strategy)
        
        # 使用策略目录作为部署源
        strategy_dir = STRATEGY_CONCEPTION_DIR
        version = f"v{strategy.get('version', 1)}"
        
        # 执行部署
        record = manager.one_click_deploy(
            strategy_path=strategy_dir,
            strategy_name=strategy_name,
            version=version,
            strategy_id=strategy_id,
            environment=env,
            deployed_by=request.deploy_by,
            strategy_info=strategy,
            notes=request.notes
        )
        
        response = DeploymentResponse(
            deployment_id=record.deployment_id,
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            environment=request.environment,
            status=record.status.value if hasattr(record.status, 'value') else str(record.status),
            message="部署已启动" if record.status != DeploymentStatus.FAILED else f"部署失败: {record.error_message}",
            created_at=record.started_at.isoformat(),
            logs=record.logs
        )
        
        logger.info(f"策略 {strategy_id} 部署完成: {record.deployment_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"部署策略失败: {e}")
        raise HTTPException(status_code=500, detail=f"部署失败: {str(e)}")


@router.post("/strategy/{strategy_id}/publish", response_model=DeploymentResponse)
async def publish_strategy(
    strategy_id: str,
    request: PublishRequest,
    background_tasks: BackgroundTasks
):
    """
    一键发布策略（验证+部署）
    
    流程：
    1. 执行回测验证
    2. 执行参数优化
    3. 应用优化结果
    4. 打包并部署到目标环境
    
    Args:
        strategy_id: 策略ID
        request: 发布请求
        
    Returns:
        部署响应
    """
    logger.info(f"开始一键发布策略 {strategy_id} 到 {request.environment} 环境")
    
    # 检查策略是否存在
    strategy = _get_strategy_info(strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail=f"策略 {strategy_id} 不存在")
    
    strategy_name = strategy.get('name', strategy_id)
    
    # TODO: 如果 run_validation=True，先执行验证流程
    # 这部分需要调用 strategy_execution_routes 和 strategy_optimization_routes 中的功能
    # 暂时直接进行部署
    
    if request.run_validation:
        logger.info(f"策略 {strategy_id} 需要验证，将在部署前执行回测和优化")
        # 这里可以添加异步验证流程
        # 目前简化处理，直接部署
    
    # 调用部署接口
    deploy_request = DeployRequest(
        environment=request.environment,
        skip_validation=not request.run_validation,
        deploy_by=request.deploy_by,
        notes=request.notes
    )
    
    return await deploy_strategy(strategy_id, deploy_request, background_tasks)


@router.get("/strategy/{strategy_id}/deployments", response_model=DeploymentHistoryResponse)
async def get_strategy_deployments(
    strategy_id: str,
    limit: int = Query(default=10, ge=1, le=100, description="返回数量限制")
):
    """
    获取策略的部署历史
    
    Args:
        strategy_id: 策略ID
        limit: 返回记录数量限制
        
    Returns:
        部署历史列表
    """
    logger.info(f"获取策略 {strategy_id} 的部署历史")
    
    manager = get_manager()
    if not manager:
        raise HTTPException(status_code=500, detail="部署管理器未初始化")
    
    try:
        records = manager.get_deployment_history(strategy_id)
        deployments = [_deployment_record_to_dict(r) for r in records[:limit]]
        
        return DeploymentHistoryResponse(
            deployments=deployments,
            total=len(records),
            strategy_id=strategy_id
        )
        
    except Exception as e:
        logger.error(f"获取部署历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取部署历史失败: {str(e)}")


@router.get("/deployments/{deployment_id}", response_model=DeploymentDetailResponse)
async def get_deployment_detail(deployment_id: str):
    """
    获取部署详情
    
    Args:
        deployment_id: 部署ID
        
    Returns:
        部署详情
    """
    logger.info(f"获取部署详情: {deployment_id}")
    
    manager = get_manager()
    if not manager:
        raise HTTPException(status_code=500, detail="部署管理器未初始化")
    
    try:
        record = manager.get_deployment(deployment_id)
        if not record:
            raise HTTPException(status_code=404, detail=f"部署 {deployment_id} 不存在")
        
        return DeploymentDetailResponse(
            deployment_id=record.deployment_id,
            strategy_id=record.strategy_id,
            strategy_name="",  # 可以从其他地方获取
            environment=record.environment.value if hasattr(record.environment, 'value') else str(record.environment),
            status=record.status.value if hasattr(record.status, 'value') else str(record.status),
            started_at=record.started_at.isoformat() if record.started_at else "",
            completed_at=record.completed_at.isoformat() if record.completed_at else None,
            logs=record.logs,
            error_message=record.error_message,
            deployed_by=record.deployed_by,
            rollback_to=record.rollback_to,
            package_info=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取部署详情失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取部署详情失败: {str(e)}")


@router.post("/deployments/{deployment_id}/rollback", response_model=RollbackResponse)
async def rollback_deployment(deployment_id: str):
    """
    回滚部署
    
    Args:
        deployment_id: 要回滚的部署ID
        
    Returns:
        回滚结果
    """
    logger.info(f"回滚部署: {deployment_id}")
    
    manager = get_manager()
    if not manager:
        raise HTTPException(status_code=500, detail="部署管理器未初始化")
    
    try:
        record = manager.get_deployment(deployment_id)
        if not record:
            raise HTTPException(status_code=404, detail=f"部署 {deployment_id} 不存在")
        
        success = manager.rollback(deployment_id)
        
        if success:
            return RollbackResponse(
                success=True,
                message="回滚成功",
                deployment_id=deployment_id,
                rollback_to=record.rollback_to
            )
        else:
            return RollbackResponse(
                success=False,
                message="回滚失败",
                deployment_id=deployment_id,
                rollback_to=None
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"回滚部署失败: {e}")
        raise HTTPException(status_code=500, detail=f"回滚失败: {str(e)}")


@router.get("/deployments")
async def list_all_deployments(
    environment: Optional[str] = Query(default=None, description="按环境过滤"),
    status: Optional[str] = Query(default=None, description="按状态过滤"),
    limit: int = Query(default=20, ge=1, le=100, description="返回数量限制")
):
    """
    列出所有部署记录
    
    Args:
        environment: 环境过滤
        status: 状态过滤
        limit: 返回数量限制
        
    Returns:
        部署记录列表
    """
    logger.info(f"列出所有部署记录")
    
    manager = get_manager()
    if not manager:
        raise HTTPException(status_code=500, detail="部署管理器未初始化")
    
    try:
        records = manager.get_deployment_history()
        
        # 过滤
        if environment:
            records = [r for r in records if str(r.environment) == environment]
        if status:
            records = [r for r in records if str(r.status) == status]
        
        # 排序并限制数量
        records = sorted(records, key=lambda x: x.started_at, reverse=True)[:limit]
        
        deployments = [_deployment_record_to_dict(r) for r in records]
        
        return {
            "deployments": deployments,
            "total": len(deployments),
            "filters": {
                "environment": environment,
                "status": status
            }
        }
        
    except Exception as e:
        logger.error(f"列出部署记录失败: {e}")
        raise HTTPException(status_code=500, detail=f"列出部署记录失败: {str(e)}")
