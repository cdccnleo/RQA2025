"""
策略工作流路由模块
提供策略工作流管理API端点
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

router = APIRouter()

# 导入工作流引擎
from .strategy_workflow import (
    workflow_engine,
    create_strategy_workflow,
    get_strategy_workflow_progress,
    get_latest_strategy_workflow,
    transition_workflow_status,
    WorkflowStatus
)


@router.post("/api/v1/strategy/workflow/create")
async def create_workflow_api(request: Dict[str, Any]):
    """创建策略工作流"""
    try:
        strategy_id = request.get("strategy_id")
        strategy_name = request.get("strategy_name", "未知策略")
        
        if not strategy_id:
            raise HTTPException(status_code=400, detail="缺少策略ID")
        
        workflow = create_strategy_workflow(strategy_id, strategy_name)
        
        return {
            "success": True,
            "workflow_id": workflow.workflow_id,
            "strategy_id": workflow.strategy_id,
            "current_status": workflow.current_status.value,
            "message": "工作流创建成功"
        }
    except Exception as e:
        logger.error(f"创建工作流失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建工作流失败: {str(e)}")


@router.get("/api/v1/strategy/workflow/{workflow_id}/progress")
async def get_workflow_progress_api(workflow_id: str):
    """获取工作流进度"""
    try:
        progress = get_strategy_workflow_progress(workflow_id)
        
        if "error" in progress:
            raise HTTPException(status_code=404, detail=progress["error"])
        
        return progress
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取工作流进度失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取进度失败: {str(e)}")


@router.get("/api/v1/strategy/{strategy_id}/workflow/latest")
async def get_latest_workflow_api(strategy_id: str):
    """获取策略最新的工作流"""
    try:
        workflow = get_latest_strategy_workflow(strategy_id)
        
        if not workflow:
            return {
                "exists": False,
                "message": "该策略暂无工作流"
            }
        
        return {
            "exists": True,
            "workflow_id": workflow.workflow_id,
            "current_status": workflow.current_status.value,
            "progress": get_strategy_workflow_progress(workflow.workflow_id)
        }
    except Exception as e:
        logger.error(f"获取最新工作流失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.post("/api/v1/strategy/workflow/{workflow_id}/transition")
async def transition_workflow_api(workflow_id: str, request: Dict[str, Any]):
    """转换工作流状态"""
    try:
        new_status = request.get("new_status")
        step_result = request.get("step_result")
        
        if not new_status:
            raise HTTPException(status_code=400, detail="缺少目标状态")
        
        success = transition_workflow_status(workflow_id, new_status, step_result)
        
        if not success:
            raise HTTPException(status_code=400, detail="状态转换失败，请检查当前状态是否允许转换到目标状态")
        
        # 返回更新后的进度
        progress = get_strategy_workflow_progress(workflow_id)
        
        return {
            "success": True,
            "message": f"状态已转换到: {new_status}",
            "progress": progress
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"转换工作流状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"转换失败: {str(e)}")


@router.post("/api/v1/strategy/workflow/{workflow_id}/pause")
async def pause_workflow_api(workflow_id: str):
    """暂停工作流"""
    try:
        success = workflow_engine.pause_workflow(workflow_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="暂停工作流失败")
        
        return {
            "success": True,
            "message": "工作流已暂停"
        }
    except Exception as e:
        logger.error(f"暂停工作流失败: {e}")
        raise HTTPException(status_code=500, detail=f"暂停失败: {str(e)}")


@router.post("/api/v1/strategy/workflow/{workflow_id}/resume")
async def resume_workflow_api(workflow_id: str):
    """恢复工作流"""
    try:
        success = workflow_engine.resume_workflow(workflow_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="恢复工作流失败")
        
        return {
            "success": True,
            "message": "工作流已恢复"
        }
    except Exception as e:
        logger.error(f"恢复工作流失败: {e}")
        raise HTTPException(status_code=500, detail=f"恢复失败: {str(e)}")


@router.get("/api/v1/strategy/workflows")
async def list_workflows_api(
    strategy_id: Optional[str] = Query(None, description="策略ID筛选"),
    status: Optional[str] = Query(None, description="状态筛选")
):
    """列出工作流"""
    try:
        status_enum = None
        if status:
            try:
                status_enum = WorkflowStatus(status)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"无效的状态: {status}")
        
        workflows = workflow_engine.list_workflows(strategy_id, status_enum)
        
        return {
            "workflows": [
                {
                    "workflow_id": w.workflow_id,
                    "strategy_id": w.strategy_id,
                    "strategy_name": w.strategy_name,
                    "current_status": w.current_status.value,
                    "created_at": w.created_at,
                    "updated_at": w.updated_at,
                    "completed_at": w.completed_at
                }
                for w in workflows
            ],
            "total": len(workflows)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"列出工作流失败: {e}")
        raise HTTPException(status_code=500, detail=f"列出失败: {str(e)}")


@router.post("/api/v1/strategy/workflow/{workflow_id}/step/start")
async def start_step_api(workflow_id: str, request: Dict[str, Any]):
    """开始工作流步骤"""
    try:
        step_name = request.get("step_name")
        
        if not step_name:
            raise HTTPException(status_code=400, detail="缺少步骤名称")
        
        success = workflow_engine.start_step(workflow_id, step_name)
        
        if not success:
            raise HTTPException(status_code=400, detail="开始步骤失败")
        
        return {
            "success": True,
            "message": f"步骤 {step_name} 已开始"
        }
    except Exception as e:
        logger.error(f"开始步骤失败: {e}")
        raise HTTPException(status_code=500, detail=f"开始步骤失败: {str(e)}")


@router.post("/api/v1/strategy/workflow/{workflow_id}/step/complete")
async def complete_step_api(workflow_id: str, request: Dict[str, Any]):
    """完成工作流步骤"""
    try:
        result = request.get("result")
        
        success = workflow_engine.complete_step(workflow_id, result)
        
        if not success:
            raise HTTPException(status_code=400, detail="完成步骤失败")
        
        return {
            "success": True,
            "message": "步骤已完成"
        }
    except Exception as e:
        logger.error(f"完成步骤失败: {e}")
        raise HTTPException(status_code=500, detail=f"完成步骤失败: {str(e)}")


@router.post("/api/v1/strategy/workflow/{workflow_id}/step/fail")
async def fail_step_api(workflow_id: str, request: Dict[str, Any]):
    """标记步骤为失败"""
    try:
        error_message = request.get("error_message", "未知错误")
        
        success = workflow_engine.fail_step(workflow_id, error_message)
        
        if not success:
            raise HTTPException(status_code=400, detail="标记步骤失败失败")
        
        return {
            "success": True,
            "message": "步骤已标记为失败"
        }
    except Exception as e:
        logger.error(f"标记步骤失败: {e}")
        raise HTTPException(status_code=500, detail=f"标记失败: {str(e)}")
