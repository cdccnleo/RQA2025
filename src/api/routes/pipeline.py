"""
管道管理API路由

提供ML自动化训练管道的REST API接口
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from ...pipeline.core.pipeline_controller import MLPipelineController
from ...pipeline.core.pipeline_config import create_default_config


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/pipeline", tags=["pipeline"])

# 全局管道控制器实例
_pipeline_controller: Optional[MLPipelineController] = None


def get_pipeline_controller() -> MLPipelineController:
    """获取或创建管道控制器实例"""
    global _pipeline_controller
    if _pipeline_controller is None:
        config = create_default_config()
        _pipeline_controller = MLPipelineController(config)
    return _pipeline_controller


# ============ 请求/响应模型 ============

class PipelineExecuteRequest(BaseModel):
    """管道执行请求"""
    config_id: str = Field(default="default", description="配置ID")
    context: Dict[str, Any] = Field(default_factory=dict, description="执行上下文")


class PipelineExecuteResponse(BaseModel):
    """管道执行响应"""
    pipeline_id: str
    status: str
    message: str


class PipelineStatusResponse(BaseModel):
    """管道状态响应"""
    pipeline_id: str
    name: str
    status: str
    current_stage: Optional[str] = None
    progress: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None


class PipelineListResponse(BaseModel):
    """管道列表响应"""
    pipelines: List[PipelineStatusResponse]
    total: int
    running: int
    completed: int
    failed: int


class StageInfo(BaseModel):
    """阶段信息"""
    name: str
    status: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class LogEntry(BaseModel):
    """日志条目"""
    timestamp: datetime
    level: str
    message: str


class PipelineDetailsResponse(BaseModel):
    """管道详情响应"""
    pipeline_id: str
    name: str
    status: str
    current_stage: Optional[str] = None
    progress: float
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    stages: List[StageInfo]
    logs: List[LogEntry]
    context: Dict[str, Any]
    error: Optional[str] = None


class CancelPipelineResponse(BaseModel):
    """取消管道响应"""
    success: bool
    message: str


# ============ API端点 ============

@router.get("/status", response_model=PipelineListResponse)
async def get_pipeline_status():
    """
    获取所有管道执行状态
    
    Returns:
        管道列表和统计信息
    """
    try:
        controller = get_pipeline_controller()
        
        # 获取所有管道执行历史
        # 这里简化处理，实际应该从持久化存储中查询
        pipelines = []
        
        # 如果存在当前执行的管道
        if controller._current_execution:
            state = controller._current_execution
            pipelines.append(PipelineStatusResponse(
                pipeline_id=state.pipeline_id,
                name=state.pipeline_name,
                status=state.status.name.lower(),
                current_stage=state.current_stage,
                progress=_calculate_progress(state),
                start_time=state.start_time,
                end_time=state.end_time,
                duration_seconds=state.duration_seconds
            ))
        
        # 计算统计
        total = len(pipelines)
        running = sum(1 for p in pipelines if p.status == "running")
        completed = sum(1 for p in pipelines if p.status == "completed")
        failed = sum(1 for p in pipelines if p.status == "failed")
        
        return PipelineListResponse(
            pipelines=pipelines,
            total=total,
            running=running,
            completed=completed,
            failed=failed
        )
        
    except Exception as e:
        logger.error(f"获取管道状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{pipeline_id}/details", response_model=PipelineDetailsResponse)
async def get_pipeline_details(pipeline_id: str):
    """
    获取管道执行详情
    
    Args:
        pipeline_id: 管道ID
        
    Returns:
        管道详细信息
    """
    try:
        controller = get_pipeline_controller()
        
        # 检查当前执行的管道
        if (controller._current_execution and 
            controller._current_execution.pipeline_id == pipeline_id):
            state = controller._current_execution
            
            # 构建阶段信息
            stages = []
            for stage_name, stage_state in state.stage_states.items():
                stages.append(StageInfo(
                    name=stage_name,
                    status=stage_state.status.name.lower(),
                    start_time=stage_state.start_time,
                    end_time=stage_state.end_time,
                    duration_seconds=stage_state.duration_seconds,
                    output=stage_state.output if hasattr(stage_state, 'output') else None,
                    error=stage_state.error
                ))
            
            # 构建日志信息
            logs = []
            for log in state.logs[-100:]:  # 最近100条日志
                logs.append(LogEntry(
                    timestamp=log.get("timestamp", datetime.now()),
                    level=log.get("level", "info"),
                    message=log.get("message", "")
                ))
            
            return PipelineDetailsResponse(
                pipeline_id=state.pipeline_id,
                name=state.pipeline_name,
                status=state.status.name.lower(),
                current_stage=state.current_stage,
                progress=_calculate_progress(state),
                start_time=state.start_time,
                end_time=state.end_time,
                duration_seconds=state.duration_seconds,
                stages=stages,
                logs=logs,
                context=state.context,
                error=state.error
            )
        
        # 如果找不到管道
        raise HTTPException(status_code=404, detail=f"管道 {pipeline_id} 不存在")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取管道详情失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute", response_model=PipelineExecuteResponse, status_code=202)
async def execute_pipeline(request: PipelineExecuteRequest):
    """
    执行管道
    
    Args:
        request: 执行请求
        
    Returns:
        执行响应
    """
    try:
        controller = get_pipeline_controller()
        
        # 检查是否已有管道在运行
        if controller._current_execution and controller._is_executing:
            raise HTTPException(
                status_code=409, 
                detail="已有管道正在执行中"
            )
        
        # 生成管道ID
        pipeline_id = f"pipe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 设置默认上下文
        context = request.context or {}
        context.setdefault("symbols", ["AAPL", "GOOGL", "MSFT"])
        context.setdefault("start_date", datetime(2024, 1, 1))
        context.setdefault("end_date", datetime(2024, 12, 31))
        context.setdefault("model_dir", "models")
        
        # 异步执行管道
        import asyncio
        asyncio.create_task(_run_pipeline_async(controller, pipeline_id, context))
        
        return PipelineExecuteResponse(
            pipeline_id=pipeline_id,
            status="pending",
            message="管道执行已启动"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"执行管道失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{pipeline_id}/cancel", response_model=CancelPipelineResponse)
async def cancel_pipeline(pipeline_id: str):
    """
    取消管道执行
    
    Args:
        pipeline_id: 管道ID
        
    Returns:
        取消响应
    """
    try:
        controller = get_pipeline_controller()
        
        # 检查管道是否存在且正在运行
        if (not controller._current_execution or 
            controller._current_execution.pipeline_id != pipeline_id):
            raise HTTPException(
                status_code=404, 
                detail=f"管道 {pipeline_id} 不存在或未在运行"
            )
        
        # 取消执行
        controller.cancel_execution()
        
        return CancelPipelineResponse(
            success=True,
            message="管道执行已取消"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"取消管道失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/configs")
async def get_pipeline_configs():
    """
    获取所有管道配置
    
    Returns:
        配置列表
    """
    try:
        # 返回默认配置
        configs = [
            {
                "id": "default",
                "name": "默认配置",
                "description": "8阶段完整ML训练管道",
                "stages_count": 8
            },
            {
                "id": "simple",
                "name": "简化配置",
                "description": "4阶段简化管道（数据准备→特征工程→训练→评估）",
                "stages_count": 4
            }
        ]
        return {"configs": configs}
        
    except Exception as e:
        logger.error(f"获取配置列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ WebSocket端点 ============

@router.websocket("/ws/{pipeline_id}")
async def pipeline_websocket(websocket: WebSocket, pipeline_id: str):
    """
    管道执行WebSocket连接
    
    实时推送管道执行状态更新
    """
    await websocket.accept()
    
    try:
        controller = get_pipeline_controller()
        
        while True:
            # 检查管道状态
            if (controller._current_execution and 
                controller._current_execution.pipeline_id == pipeline_id):
                state = controller._current_execution
                
                # 发送状态更新
                await websocket.send_json({
                    "type": "pipeline_status",
                    "timestamp": datetime.now().isoformat(),
                    "data": {
                        "pipeline_id": pipeline_id,
                        "status": state.status.name.lower(),
                        "current_stage": state.current_stage,
                        "progress": _calculate_progress(state),
                        "duration_seconds": state.duration_seconds
                    }
                })
            
            # 等待1秒后再次检查
            import asyncio
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket连接断开: {pipeline_id}")
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
        await websocket.close()


# ============ 辅助函数 ============

def _calculate_progress(state) -> float:
    """计算管道执行进度"""
    if not state.stage_states:
        return 0.0
    
    total_stages = len(state.stage_states)
    completed_stages = sum(
        1 for s in state.stage_states.values()
        if s.status.name in ["COMPLETED", "SKIPPED"]
    )
    
    return (completed_stages / total_stages) * 100 if total_stages > 0 else 0.0


async def _run_pipeline_async(controller: MLPipelineController, 
                              pipeline_id: str, 
                              context: Dict[str, Any]):
    """异步运行管道"""
    try:
        # 导入所有阶段
        from ...pipeline.stages.data_preparation import DataPreparationStage
        from ...pipeline.stages.feature_engineering import FeatureEngineeringStage
        from ...pipeline.stages.model_training import ModelTrainingStage
        from ...pipeline.stages.model_evaluation import ModelEvaluationStage
        from ...pipeline.stages.model_validation import ModelValidationStage
        from ...pipeline.stages.canary_deployment import CanaryDeploymentStage
        from ...pipeline.stages.full_deployment import FullDeploymentStage
        from ...pipeline.stages.monitoring import MonitoringStage
        
        # 注册阶段
        stages = [
            DataPreparationStage(),
            FeatureEngineeringStage(),
            ModelTrainingStage(),
            ModelEvaluationStage(),
            ModelValidationStage(),
            CanaryDeploymentStage(),
            FullDeploymentStage(),
            MonitoringStage()
        ]
        
        for stage in stages:
            controller.register_stage(stage)
        
        # 执行管道
        result = controller.execute(
            initial_context=context,
            pipeline_id=pipeline_id
        )
        
        logger.info(f"管道 {pipeline_id} 执行完成: {result.status.name}")
        
    except Exception as e:
        logger.error(f"管道 {pipeline_id} 执行失败: {e}")
