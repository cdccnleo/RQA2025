#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据服务API
提供数据获取、清洗、存储等功能的REST API接口
"""

# 使用基础设施层日志，避免依赖上层组件
try:
    from src.infrastructure.logging import get_infrastructure_logger
except ImportError:
    # 降级到标准logging
    import logging

    def get_infrastructure_logger(name):

        logger = logging.getLogger(name)
        logger.warning("无法导入基础设施层日志，使用标准logging")
        return logger

# 独立导入 DataManagerSingleton，修正相对路径
try:
    from ..core.data_manager import DataManagerSingleton
except ImportError:
    DataManagerSingleton = None  # type: ignore

import time
from typing import Dict, Optional, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import uvicorn

# 导入已在上面try/except块中处理
# from src.data.data_loader import DataLoader
# from src.data.data_validator import DataValidator

# 初始化日志
logger = get_infrastructure_logger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="RQA2025 数据服务",
    description="提供数据获取、清洗、存储等功能的REST API",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化组件
data_manager = DataManagerSingleton.get_instance() if DataManagerSingleton else None
data_loader = None  # 暂时设为None，等待实际实现
data_validator = None  # 暂时设为None，等待实际实现

# 请求模型


class DataRequest(BaseModel):

    """数据请求"""
    symbol: str
    start_date: str
    end_date: str
    data_type: str = "ohlcv"
    source: Optional[str] = "default"


class DataValidationRequest(BaseModel):

    """数据验证请求"""
    data: Any  # 支持任何类型的数据，包括列表和字典
    validation_rules: Optional[Dict[str, Any]] = None


class DataStorageRequest(BaseModel):

    """数据存储请求"""
    data: Any  # 支持任何类型的数据，包括列表和字典，但不包括None
    storage_type: str = "database"  # 添加默认值，使其成为可选字段
    metadata: Optional[Dict[str, Any]] = None

    @field_validator('data')
    @classmethod
    def validate_data_not_none(cls, v):

        if v is None:
            raise ValueError('数据不能为None')
        return v

# 响应模型


class DataResponse(BaseModel):

    """数据响应"""
    data: Any  # 支持任何类型的数据，包括列表和字典
    metadata: Dict[str, Any]
    processing_time: float
    status: str


class HealthResponse(BaseModel):

    """健康检查响应"""
    status: str
    timestamp: float
    version: str


@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("数据服务启动中...")
    try:
        # 初始化数据管理器
        data_manager.initialize()
        logger.info("数据管理器初始化完成")

        # 暂时跳过数据加载器和验证器的初始化
        # data_loader.initialize()
        # data_validator.initialize()

        logger.info("数据服务启动完成")
    except Exception as e:
        logger.error(f"服务启动失败: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version="1.0.0"
    )


@app.get("/ready")
async def readiness_check():
    """就绪检查"""
    try:
        # 检查关键组件状态
        if not data_manager.is_initialized():
            raise HTTPException(status_code=503, detail="数据管理器未就绪")

        # 暂时跳过数据加载器检查
        # if not data_loader.is_initialized():
        #     raise HTTPException(status_code=503, detail="数据加载器未就绪")

        return {"status": "ready"}
    except Exception as e:
        logger.error(f"就绪检查失败: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/api / v1 / data / fetch", response_model=DataResponse)
async def fetch_data(request: DataRequest):
    """获取数据"""
    start_time = time.time()

    try:
        logger.info(f"开始获取数据: {request.symbol}, {request.start_date} - {request.end_date}")

        # 暂时返回模拟数据，等待实际实现
        if data_loader is None:
            raise HTTPException(status_code=503, detail="数据加载器未实现")

        # 获取数据
        data = data_loader.load_data(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            data_type=request.data_type,
            source=request.source
        )

        # 验证数据
        validation_result = {"valid": True, "errors": []}  # 暂时跳过验证
        if data_validator:
            validation_result = data_validator.validate_data(data)

        processing_time = time.time() - start_time

        metadata = {
            "symbol": request.symbol,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "data_type": request.data_type,
            "source": request.source,
            "validation_result": validation_result,
            "data_points": len(data) if data else 0,
            "processing_time": processing_time
        }

        logger.info(f"数据获取完成，耗时: {processing_time:.3f}秒")

        return DataResponse(
            data=data,
            metadata=metadata,
            processing_time=processing_time,
            status="success"
        )

    except Exception as e:
        logger.error(f"数据获取失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api / v1 / data / validate")
async def validate_data(request: DataValidationRequest):
    """验证数据"""
    start_time = time.time()

    try:
        logger.info("开始验证数据")

        # 暂时跳过验证，等待实际实现
        if data_validator is None:
            raise HTTPException(status_code=503, detail="数据验证器未实现")

        # 验证数据
        validation_result = data_validator.validate_data(
            data=request.data,
            rules=request.validation_rules
        )

        processing_time = time.time() - start_time

        return {
            "validation_result": validation_result,
            "processing_time": processing_time,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"数据验证失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api / v1 / data / store")
async def store_data(request: DataStorageRequest):
    """存储数据"""
    start_time = time.time()

    try:
        logger.info(f"开始存储数据，类型: {request.storage_type}")

        # 存储数据
        storage_result = data_manager.store_data(
            data=request.data,
            storage_type=request.storage_type,
            metadata=request.metadata
        )

        processing_time = time.time() - start_time

        return {
            "storage_result": storage_result,
            "processing_time": processing_time,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"数据存储失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api / v1 / data / symbols")
async def get_available_symbols():
    """获取可用交易对列表"""
    try:
        if data_loader is None:
            raise HTTPException(status_code=503, detail="数据加载器未实现")
        symbols = data_loader.get_available_symbols()
        return {"symbols": symbols}
    except Exception as e:
        logger.error(f"获取交易对列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api / v1 / data / sources")
async def get_available_sources():
    """获取可用数据源列表"""
    try:
        if data_loader is None:
            raise HTTPException(status_code=503, detail="数据加载器未实现")
        sources = data_loader.get_available_sources()
        return {"sources": sources}
    except Exception as e:
        logger.error(f"获取数据源列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api / v1 / data / status")
async def get_data_status():
    """获取数据服务状态"""
    try:
        status = {
            "service": "data - service",
            "version": "1.0.0",
            "status": "running",
            "data_manager": data_manager.is_initialized(),
            "data_loader": data_loader.is_initialized() if data_loader else False,
            "data_validator": data_validator.is_initialized() if data_validator else False
        }
        return status
    except Exception as e:
        logger.error(f"获取状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "src.data.api:app",
        host="0.0.0.0",
        port=8002,
        reload=False
    )
