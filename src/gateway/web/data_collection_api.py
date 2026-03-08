#!/usr/bin/env python3
"""
数据采集API服务
基于FastAPI提供RESTful数据采集接口
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field, validator
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None
    HTTPException = Exception
    BackgroundTasks = None
    CORSMiddleware = None
    JSONResponse = None
    BaseModel = object
    Field = lambda **kwargs: None
    validator = lambda func: func

from src.infrastructure.orchestration.historical_data_acquisition_service import (
    HistoricalDataAcquisitionService,
    HistoricalDataConfig,
    DataSourceType
)
from src.infrastructure.orchestration.strategy_backtest_data_workflow import (
    StrategyBacktestDataWorkflow,
    WorkflowConfig
)
from src.infrastructure.orchestration.data_quality_manager import DataQualityManager
from src.core.persistence.timescale_storage import TimescaleStorage
from src.core.persistence.minio_storage import MinIOStorage
from src.core.monitoring.data_collection_monitor import DataCollectionMonitor


# Pydantic模型
class DataAcquisitionRequest(BaseModel):
    """数据采集请求"""
    symbols: List[str] = Field(..., min_items=1, max_items=1000, description="股票代码列表")
    start_date: str = Field(..., description="开始日期 (YYYY-MM-DD)")
    end_date: str = Field(..., description="结束日期 (YYYY-MM-DD)")
    data_types: List[str] = Field(default=["stock"], description="数据类型")
    priority: str = Field(default="normal", pattern="^(low|normal|high|critical)$", description="采集优先级")
    quality_threshold: float = Field(default=0.85, ge=0.0, le=1.0, description="质量阈值")
    max_concurrent: int = Field(default=5, ge=1, le=20, description="最大并发数")
    data_source: Optional[str] = Field(default=None, description="数据源 (可选)")

    @validator('end_date')
    def end_date_must_be_after_start_date(cls, v, values):
        if 'start_date' in values and v < values['start_date']:
            raise ValueError('结束日期必须晚于开始日期')
        return v

    @validator('start_date', 'end_date')
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('日期格式必须为YYYY-MM-DD')


class WorkflowCreationRequest(BaseModel):
    """工作流创建请求"""
    name: str = Field(..., description="工作流名称")
    symbols: List[str] = Field(..., min_items=1, max_items=100, description="股票代码列表")
    start_year: int = Field(..., ge=1990, le=datetime.now().year, description="开始年份")
    end_year: int = Field(..., ge=1990, le=datetime.now().year, description="结束年份")
    data_types: List[str] = Field(default=["stock"], description="数据类型")
    max_concurrent_years: int = Field(default=2, ge=1, le=5, description="最大并发年数")
    quality_threshold: float = Field(default=0.85, ge=0.0, le=1.0, description="质量阈值")
    enable_progress_tracking: bool = Field(default=True, description="启用进度跟踪")

    @validator('end_year')
    def end_year_must_be_after_start_year(cls, v, values):
        if 'start_year' in values and v < values['start_year']:
            raise ValueError('结束年份必须晚于开始年份')
        return v


class TaskStatusResponse(BaseModel):
    """任务状态响应"""
    task_id: str
    status: str
    progress: float = 0.0
    message: str = ""
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    estimated_completion: Optional[str] = None
    result_summary: Optional[Dict[str, Any]] = None
    errors: List[str] = []


class WorkflowStatusResponse(BaseModel):
    """工作流状态响应"""
    workflow_id: str
    status: str
    progress: Dict[str, Any] = {}
    message: str = ""
    created_at: str
    end_time: Optional[str] = None
    duration_seconds: float = 0.0
    batches_completed: int = 0
    total_records: int = 0
    quality_stats: Dict[str, Any] = {}
    errors: List[str] = []


class StockDataQuery(BaseModel):
    """股票数据查询"""
    symbol: str = Field(..., description="股票代码")
    start_date: str = Field(..., description="开始日期 (YYYY-MM-DD)")
    end_date: str = Field(..., description="结束日期 (YYYY-MM-DD)")
    data_type: str = Field(default="price", description="数据类型")
    adjusted: bool = Field(default=True, description="是否复权")
    limit: Optional[int] = Field(default=None, ge=1, le=10000, description="返回记录数限制")


class APIResponse(BaseModel):
    """API响应"""
    success: bool
    message: str
    data: Optional[Any] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# 全局服务实例
acquisition_service: Optional[HistoricalDataAcquisitionService] = None
workflow_service: Optional[StrategyBacktestDataWorkflow] = None
quality_manager: Optional[DataQualityManager] = None
timescale_storage: Optional[TimescaleStorage] = None
minio_storage: Optional[MinIOStorage] = None
monitor: Optional[DataCollectionMonitor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化服务
    global acquisition_service, workflow_service, quality_manager, timescale_storage, minio_storage, monitor

    logger = logging.getLogger(__name__)
    logger.info("初始化数据采集API服务...")

    try:
        # 这里应该从配置文件加载服务配置
        # 暂时使用默认配置
        config = {
            'acquisition_service_config': {
                'adapters': {
                    'akshare': {},
                    'yahoo': {},
                    'local_backup': {'backup_dir': './data/backup'}
                },
                'max_concurrent_batches': 3,
                'quality_threshold': 0.85
            },
            'timescale_config': {
                'host': 'localhost',
                'port': 5432,
                'database': 'rqa2025',
                'user': 'rqa2025_admin',
                'password': 'rqa2025_prod'
            },
            'minio': {
                'endpoint': 'localhost:9000',
                'access_key': 'minioadmin',
                'secret_key': 'minioadmin',
                'secure': False
            },
            'redis_config': {},
            'monitor_config': {}
        }

        # 初始化服务
        acquisition_service = HistoricalDataAcquisitionService(config['acquisition_service_config'])
        workflow_service = StrategyBacktestDataWorkflow(config)
        quality_manager = DataQualityManager({'stock_checker': {}})
        timescale_storage = TimescaleStorage(config['timescale_config'])
        minio_storage = MinIOStorage(config)
        monitor = DataCollectionMonitor({})

        logger.info("数据采集API服务初始化完成")

    except Exception as e:
        logger.error(f"服务初始化失败: {e}")
        # 在生产环境中，这里应该优雅地处理初始化失败

    yield

    # 关闭时清理资源
    logger.info("清理数据采集API服务资源...")


# 创建FastAPI应用
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="RQA2025 数据采集API",
        description="量化交易数据采集和处理API服务",
        version="1.0.0",
        lifespan=lifespan
    )

    # 添加CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 在生产环境中应该限制为特定域名
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 健康检查端点
    @app.get("/health", response_model=APIResponse)
    async def health_check():
        """健康检查"""
        return APIResponse(
            success=True,
            message="数据采集API服务正常",
            data={"status": "healthy", "timestamp": datetime.now().isoformat()}
        )

    # 数据采集端点 - 使用统一调度器（符合架构设计）
    @app.post("/api/v1/acquisition/start", response_model=APIResponse)
    async def start_data_acquisition(request: DataAcquisitionRequest, background_tasks: BackgroundTasks):
        """启动数据采集任务 - 提交到统一调度器"""
        try:
            # 参数验证
            if not request.symbols:
                raise HTTPException(status_code=400, detail="股票代码列表不能为空")

            if len(request.symbols) > 1000:
                raise HTTPException(status_code=400, detail="单次采集股票数量不能超过1000只")

            # 使用统一调度器提交数据采集任务（符合架构设计）
            from .data_collection_service import submit_data_collection_task
            
            result = submit_data_collection_task(
                symbols=request.symbols,
                start_date=request.start_date,
                end_date=request.end_date,
                data_types=request.data_types,
                priority=request.priority,
                data_source=request.data_source,
                metadata={
                    "quality_threshold": request.quality_threshold,
                    "max_concurrent": request.max_concurrent
                }
            )
            
            if not result.get("success"):
                raise HTTPException(status_code=500, detail=result.get("message", "提交任务失败"))

            return APIResponse(
                success=True,
                message=f"数据采集任务已提交到统一调度器: {len(request.symbols)} 只股票",
                data={
                    "task_id": result["task_id"],
                    "scheduler_task_id": result["scheduler_task_id"],
                    "config": result["config"]
                }
            )

        except HTTPException:
            raise
        except Exception as e:
            logging.error(f"启动采集任务失败: {e}")
            raise HTTPException(status_code=500, detail=f"任务启动失败: {str(e)}")

    @app.get("/api/v1/acquisition/{task_id}/status", response_model=TaskStatusResponse)
    async def get_acquisition_status(task_id: str):
        """获取采集任务状态"""
        try:
            if not acquisition_service:
                raise HTTPException(status_code=503, detail="采集服务未初始化")

            # 这里应该从持久化存储中获取任务状态
            # 暂时返回模拟状态
            status = TaskStatusResponse(
                task_id=task_id,
                status="running",
                progress=0.5,
                message="任务执行中",
                created_at=datetime.now().isoformat(),
                estimated_completion=(datetime.now() + timedelta(minutes=30)).isoformat()
            )

            return status

        except HTTPException:
            raise
        except Exception as e:
            logging.error(f"获取任务状态失败: {e}")
            raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")

    # 工作流管理端点
    @app.post("/api/v1/workflow/create", response_model=APIResponse)
    async def create_data_workflow(request: WorkflowCreationRequest, background_tasks: BackgroundTasks):
        """创建数据采集工作流"""
        try:
            if not workflow_service:
                raise HTTPException(status_code=503, detail="工作流服务未初始化")

            # 创建工作流配置
            workflow_config = WorkflowConfig(
                name=request.name,
                symbol=",".join(request.symbols),  # 多股票用逗号分隔
                start_year=request.start_year,
                end_year=request.end_year,
                data_types=request.data_types,
                max_concurrent_years=request.max_concurrent_years,
                quality_threshold=request.quality_threshold,
                enable_progress_tracking=request.enable_progress_tracking
            )

            # 启动工作流
            workflow_id = await workflow_service.start_workflow(workflow_config)

            return APIResponse(
                success=True,
                message="数据采集工作流已创建",
                data={"workflow_id": workflow_id}
            )

        except HTTPException:
            raise
        except Exception as e:
            logging.error(f"创建工作流失败: {e}")
            raise HTTPException(status_code=500, detail=f"工作流创建失败: {str(e)}")

    @app.get("/api/v1/workflow/{workflow_id}/status", response_model=WorkflowStatusResponse)
    async def get_workflow_status(workflow_id: str):
        """获取工作流状态"""
        try:
            if not workflow_service:
                raise HTTPException(status_code=503, detail="工作流服务未初始化")

            status = workflow_service.get_workflow_status(workflow_id)
            if not status:
                raise HTTPException(status_code=404, detail="工作流不存在")

            response = WorkflowStatusResponse(
                workflow_id=status.workflow_id,
                status=status.status.value,
                progress={
                    "total_years": status.progress.total_years,
                    "completed_years": status.progress.completed_years,
                    "total_batches": status.progress.total_batches,
                    "completed_batches": status.progress.completed_batches,
                    "total_records": status.progress.total_records
                },
                message=status.progress.status_message,
                created_at=status.start_time.isoformat(),
                end_time=status.end_time.isoformat() if status.end_time else None,
                duration_seconds=status.duration_seconds,
                batches_completed=status.progress.completed_batches,
                total_records=status.progress.total_records,
                quality_stats=status.quality_stats,
                errors=status.errors
            )

            return response

        except HTTPException:
            raise
        except Exception as e:
            logging.error(f"获取工作流状态失败: {e}")
            raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")

    # 数据查询端点
    @app.get("/api/v1/data/stock/{symbol}")
    async def get_stock_data(
        symbol: str,
        start_date: str,
        end_date: str,
        data_type: str = "price",
        adjusted: bool = True,
        limit: Optional[int] = None
    ):
        """获取股票历史数据"""
        try:
            if not timescale_storage:
                raise HTTPException(status_code=503, detail="存储服务未初始化")

            # 参数验证
            try:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            except ValueError:
                raise HTTPException(status_code=400, detail="日期格式必须为YYYY-MM-DD")

            if end_dt < start_dt:
                raise HTTPException(status_code=400, detail="结束日期必须晚于开始日期")

            # 查询数据
            data = await timescale_storage.query_historical_data(
                symbol=symbol,
                start_date=start_dt,
                end_date=end_dt,
                data_type=data_type,
                quality_threshold=0.8  # 只返回高质量数据
            )

            # 应用限制
            if limit and len(data) > limit:
                data = data[:limit]

            # 计算统计信息
            stats = {
                "symbol": symbol,
                "data_type": data_type,
                "start_date": start_date,
                "end_date": end_date,
                "record_count": len(data),
                "adjusted": adjusted
            }

            if data:
                # 添加数据时间范围
                dates = [datetime.strptime(record['date'], '%Y-%m-%d') for record in data]
                stats["data_start_date"] = min(dates).strftime('%Y-%m-%d')
                stats["data_end_date"] = max(dates).strftime('%Y-%m-%d')

            return APIResponse(
                success=True,
                message="数据查询成功",
                data={
                    "stats": stats,
                    "data": data
                }
            )

        except HTTPException:
            raise
        except Exception as e:
            logging.error(f"获取股票数据失败: {e}")
            raise HTTPException(status_code=500, detail=f"数据获取失败: {str(e)}")

    # 存储管理端点
    @app.get("/api/v1/storage/stats", response_model=APIResponse)
    async def get_storage_stats():
        """获取存储统计信息"""
        try:
            if not timescale_storage:
                raise HTTPException(status_code=503, detail="存储服务未初始化")

            # 获取TimescaleDB统计
            db_stats = await timescale_storage.get_storage_stats()

            # 获取MinIO统计（如果可用）
            minio_stats = None
            if minio_storage:
                try:
                    minio_stats = await minio_storage.get_storage_stats()
                    minio_stats = {
                        "total_objects": minio_stats.total_objects,
                        "total_size_bytes": minio_stats.total_size_bytes,
                        "buckets_count": minio_stats.buckets_count
                    }
                except Exception as e:
                    logging.warning(f"获取MinIO统计失败: {e}")

            return APIResponse(
                success=True,
                message="存储统计获取成功",
                data={
                    "timescale_db": db_stats,
                    "minio": minio_stats
                }
            )

        except Exception as e:
            logging.error(f"获取存储统计失败: {e}")
            raise HTTPException(status_code=500, detail=f"获取统计失败: {str(e)}")

    # 系统监控端点
    @app.get("/api/v1/monitor/metrics", response_model=APIResponse)
    async def get_system_metrics():
        """获取系统监控指标"""
        try:
            if not monitor:
                raise HTTPException(status_code=503, detail="监控服务未初始化")

            # 获取监控指标
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "system_status": "healthy",  # 这里应该从实际监控获取
                "active_workflows": len(workflow_service.active_workflows) if workflow_service else 0,
                "recent_activities": []  # 这里应该从监控历史获取
            }

            return APIResponse(
                success=True,
                message="系统监控指标获取成功",
                data=metrics
            )

        except HTTPException:
            raise
        except Exception as e:
            logging.error(f"获取系统监控指标失败: {e}")
            raise HTTPException(status_code=500, detail=f"获取指标失败: {str(e)}")

    # 自定义异常处理器
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "message": exc.detail,
                "timestamp": datetime.now().isoformat()
            }
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        logging.error(f"未处理的异常: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "服务器内部错误",
                "timestamp": datetime.now().isoformat()
            }
        )

else:
    # 如果没有安装FastAPI，创建一个占位符
    app = None
    logging.warning("FastAPI未安装，API服务不可用")


# 后台任务函数
async def execute_acquisition_task(task_id: str, symbols: List[str], start_date: str,
                                 end_date: str, data_types: List[str],
                                 data_source: Optional[DataSourceType],
                                 priority: str, quality_threshold: float,
                                 max_concurrent: int):
    """执行采集任务"""
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"开始执行API采集任务: {task_id}")

        # 记录到监控
        if monitor:
            await monitor.record_api_request(task_id, "acquisition_start", {
                "symbol_count": len(symbols),
                "date_range": f"{start_date} to {end_date}"
            })

        # 这里应该实现实际的采集逻辑
        # 暂时模拟采集过程
        await asyncio.sleep(5)  # 模拟处理时间

        # 记录任务完成
        if monitor:
            await monitor.record_api_request(task_id, "acquisition_complete", {
                "status": "success",
                "processed_symbols": len(symbols)
            })

        logger.info(f"API采集任务完成: {task_id}")

    except Exception as e:
        logger.error(f"API采集任务失败 {task_id}: {e}")

        # 记录失败
        if monitor:
            await monitor.record_api_request(task_id, "acquisition_failed", {
                "error": str(e)
            })


def create_application() -> Optional[FastAPI]:
    """创建FastAPI应用"""
    if not FASTAPI_AVAILABLE:
        logging.error("FastAPI未安装，无法创建API服务")
        return None

    return app


if __name__ == "__main__":
    # 开发环境运行
    if FASTAPI_AVAILABLE and app:
        import uvicorn
        uvicorn.run(
            "data_collection_api:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    else:
        print("FastAPI未安装或应用创建失败")