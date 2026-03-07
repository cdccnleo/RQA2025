import logging
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征工程服务API
提供特征计算、技术指标、情感分析等功能的REST API接口
"""

import time
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from ...feature_engineer import FeatureEngineer
from ...config_integration import FeatureConfigIntegrationManager
from ...monitoring.features_monitor import FeaturesMonitor

# 初始化日志
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="RQA2025 特征工程服务",
    description="提供特征计算、技术指标、情感分析等功能的REST API",
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
config_manager = FeatureConfigIntegrationManager()
feature_engineer = FeatureEngineer()
monitor = FeaturesMonitor()


@app.get("/")
def root():
    """API根路径"""
    return {
        "message": "RQA2025 特征工程服务API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


# 请求模型


class FeatureRequest(BaseModel):

    """特征计算请求"""
    data: Dict[str, Any]
    features: List[str]
    config: Optional[Dict[str, Any]] = None


class TechnicalIndicatorRequest(BaseModel):

    """技术指标请求"""
    symbol: str
    timeframe: str
    indicators: List[str]
    period: Optional[int] = 14


class SentimentRequest(BaseModel):

    """情感分析请求"""
    text: str
    model: Optional[str] = "default"

# 响应模型


class FeatureResponse(BaseModel):

    """特征计算响应"""
    features: Dict[str, Any]
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
    logger.info("特征工程服务启动中...")
    try:
        # 初始化配置管理器
        config_manager.initialize()
        logger.info("配置管理器初始化完成")

        # 初始化特征工程师
        feature_engineer.initialize()
        logger.info("特征工程师初始化完成")

        # 启动监控
        monitor.start()
        logger.info("监控系统启动完成")

        logger.info("特征工程服务启动完成")
    except Exception as e:
        logger.error(f"服务启动失败: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("特征工程服务关闭中...")
    try:
        monitor.stop()
        logger.info("监控系统已停止")
        logger.info("特征工程服务已关闭")
    except Exception as e:
        logger.error(f"服务关闭失败: {e}")


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
        if not config_manager.is_initialized():
            raise HTTPException(status_code=503, detail="配置管理器未就绪")

        if not feature_engineer.is_initialized():
            raise HTTPException(status_code=503, detail="特征工程师未就绪")

        return {"status": "ready"}
    except Exception as e:
        logger.error(f"就绪检查失败: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/api/v1/features/calculate", response_model=FeatureResponse)
async def calculate_features(request: FeatureRequest):
    """计算特征"""
    start_time = time.time()

    try:
        logger.info(f"开始计算特征: {request.features}")

        # 应用配置
        if request.config:
            config_manager.update_config(request.config)

        # 计算特征
        features = feature_engineer.calculate_features(
            data=request.data,
            feature_names=request.features
        )

        processing_time = time.time() - start_time

        # 记录指标
        monitor.record_metric("feature_calculation_time", processing_time)
        monitor.record_metric("feature_calculation_count", 1)

        logger.info(f"特征计算完成，耗时: {processing_time:.3f}秒")

        return FeatureResponse(
            features=features,
            processing_time=processing_time,
            status="success"
        )

    except Exception as e:
        logger.error(f"特征计算失败: {e}")
        monitor.record_metric("feature_calculation_errors", 1)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/features/technical")
async def calculate_technical_indicators(request: TechnicalIndicatorRequest):
    """计算技术指标"""
    start_time = time.time()

    try:
        logger.info(f"开始计算技术指标: {request.indicators}")

        # 计算技术指标
        indicators = feature_engineer.calculate_technical_indicators(
            symbol=request.symbol,
            timeframe=request.timeframe,
            indicators=request.indicators,
            period=request.period
        )

        processing_time = time.time() - start_time

        # 记录指标
        monitor.record_metric("technical_indicator_time", processing_time)

        return {
            "indicators": indicators,
            "processing_time": processing_time,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"技术指标计算失败: {e}")
        monitor.record_metric("technical_indicator_errors", 1)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/features/sentiment")
async def analyze_sentiment(request: SentimentRequest):
    """情感分析"""
    start_time = time.time()

    try:
        logger.info(f"开始情感分析，模型: {request.model}")

        # 执行情感分析
        sentiment = feature_engineer.analyze_sentiment(
            text=request.text,
            model=request.model
        )

        processing_time = time.time() - start_time

        # 记录指标
        monitor.record_metric("sentiment_analysis_time", processing_time)

        return {
            "sentiment": sentiment,
            "processing_time": processing_time,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"情感分析失败: {e}")
        monitor.record_metric("sentiment_analysis_errors", 1)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/features/config")
async def get_config():
    """获取配置"""
    try:
        config = config_manager.get_config()
        return {"config": config}
    except Exception as e:
        logger.error(f"获取配置失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api / v1 / features / config")
async def update_config(config: Dict[str, Any]):
    """更新配置"""
    try:
        config_manager.update_config(config)
        logger.info("配置更新成功")
        return {"status": "success", "message": "配置更新成功"}
    except Exception as e:
        logger.error(f"配置更新失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/features/metrics")
async def get_metrics():
    """获取指标"""
    try:
        metrics = monitor.get_metrics()
        return {"metrics": metrics}
    except Exception as e:
        logger.error(f"获取指标失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/features/status")
async def get_status():
    """获取服务状态"""
    try:
        status = {
            "service": "features - service",
            "version": "1.0.0",
            "status": "running",
            "config_manager": config_manager.is_initialized(),
            "feature_engineer": feature_engineer.is_initialized(),
            "monitor": monitor.is_running()
        }
        return status
    except Exception as e:
        logger.error(f"获取状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "src.features.api:app",
        host="0.0.0.0",
        port=8001,
        reload=False
    )
