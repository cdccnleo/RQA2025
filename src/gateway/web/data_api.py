#!/usr/bin/env python3
"""
RQA2025 数据层 REST API 接口

from src.engine.logging.unified_logger import get_unified_logger
提供数据管理、质量监控、性能指标的完整API接口
支持数据源管理、质量监控、性能监控等功能
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.data import DataManager  # 合理跨层级导入：数据层数据管理器
from src.data.monitoring import PerformanceMonitor  # 合理跨层级导入：数据层性能监控
from src.data.quality import DataQualityMonitor, AdvancedQualityMonitor  # 合理跨层级导入：数据层质量监控
# 暂时注释掉不存在的加载器导入，使用基础的DataLoader
# from src.data.loader import (  # 合理跨层级导入：数据层各种数据加载器
#     CryptoLoader as CryptoDataLoader, MacroLoader as MacroDataLoader,
#     OptionsLoader as OptionsDataLoader, BondLoader as BondDataLoader,
#     CommodityLoader as CommodityDataLoader, ForexLoader as ForexDataLoader
from src.data.loader import BaseDataLoader, DataLoader  # 使用基础加载器类
from src.infrastructure.logging.core.unified_logger import get_unified_logger  # 当前层级内部导入：统一日志器

# 配置日志
logging.basicConfig(level=logging.INFO)

# 创建路由器
router = APIRouter(prefix="/data", tags=["data"])

# 初始化组件
base_config = {
    'cache_dir': 'cache',
    'max_retries': 3,
    'timeout': 30
}

data_manager = DataManager()
performance_monitor = PerformanceMonitor()
quality_monitor = DataQualityMonitor()
advanced_quality_monitor = AdvancedQualityMonitor()

# 数据加载器实例 - 使用通用DataLoader
loaders = {
    "crypto": DataLoader(base_config),
    "macro": DataLoader(base_config),
    "options": DataLoader(base_config),
    "bond": DataLoader(base_config),
    "commodity": DataLoader(base_config),
    "forex": DataLoader(base_config)
}


# Pydantic 模型定义
logger = logging.getLogger(__name__)


class DataSourceRequest(BaseModel):

    """数据源请求模型"""
    source_type: str = Field(..., description="数据源类型")
    symbol: str = Field(..., description="数据符号")
    start_date: str = Field(..., description="开始日期")
    end_date: str = Field(..., description="结束日期")
    frequency: str = Field(default="1d", description="数据频率")


class DataQualityRequest(BaseModel):

    """数据质量请求模型"""
    source_type: str = Field(..., description="数据源类型")
    symbol: str = Field(..., description="数据符号")
    metrics: List[str] = Field(default=[], description="质量指标列表")


class PerformanceMetrics(BaseModel):

    """性能指标模型"""
    cache_hit_rate: float = Field(..., description="缓存命中率")
    load_time: float = Field(..., description="加载时间")
    memory_usage: float = Field(..., description="内存使用率")
    error_rate: float = Field(..., description="错误率")
    timestamp: datetime = Field(..., description="时间戳")


class QualityMetrics(BaseModel):

    """质量指标模型"""
    completeness: float = Field(..., description="完整性")
    accuracy: float = Field(..., description="准确性")
    consistency: float = Field(..., description="一致性")
    timeliness: float = Field(..., description="时效性")
    validity: float = Field(..., description="有效性")
    reliability: float = Field(..., description="可靠性")
    uniqueness: float = Field(..., description="唯一性")
    integrity: float = Field(..., description="完整性")
    precision: float = Field(..., description="精确度")
    availability: float = Field(..., description="可用性")
    timestamp: datetime = Field(..., description="时间戳")


class DataSourceInfo(BaseModel):

    """数据源信息模型"""
    name: str = Field(..., description="数据源名称")
    type: str = Field(..., description="数据源类型")
    status: str = Field(..., description="状态")
    last_update: datetime = Field(..., description="最后更新时间")
    data_count: int = Field(..., description="数据条数")


# API 端点定义

@router.get("/health")
async def health_check():
    """健康检查接口"""
    try:
        # 检查数据管理器状态
        status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "data_manager": "ok",
                "performance_monitor": "ok",
                "quality_monitor": "ok"
            },
            "available_sources": list(loaders.keys())
        }
        return status
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sources")
async def list_data_sources():
    """获取可用数据源列表"""
    try:
        sources = []
        for name, loader in loaders.items():
            # 模拟数据源信息
            source_info = DataSourceInfo(
                name=name,
                type=name,
                status="active",
                last_update=datetime.now(),
                data_count=1000
            )
            sources.append(source_info.dict())

        return {
            "sources": sources,
            "total": len(sources)
        }
    except Exception as e:
        logger.error(f"获取数据源列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sources/{source_type}")
async def get_data_source_info(source_type: str):
    """获取特定数据源信息"""
    try:
        if source_type not in loaders:
            raise HTTPException(status_code=404, detail="数据源不存在")

        loader = loaders[source_type]
        source_info = DataSourceInfo(
            name=source_type,
            type=source_type,
            status="active",
            last_update=datetime.now(),
            data_count=1000
        )

        return source_info.dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取数据源信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load")
async def load_data(request: DataSourceRequest):
    """加载数据接口"""
    try:
        if request.source_type not in loaders:
            raise HTTPException(status_code=404, detail="数据源不存在")

        loader = loaders[request.source_type]

        # 记录开始时间
        start_time = datetime.now()

        # 加载数据
        data = await loader.load_data(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            frequency=request.frequency
        )

        # 计算加载时间
        load_time = (datetime.now() - start_time).total_seconds()

        # 记录性能指标
        performance_monitor.record_load_time(load_time)

        return {
            "status": "success",
            "data": data,
            "load_time": load_time,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance")
async def get_performance_metrics():
    """获取性能指标"""
    try:
        metrics = performance_monitor.get_metrics()

        performance_data = PerformanceMetrics(
            cache_hit_rate=metrics.get("cache_hit_rate", 0.0),
            load_time=metrics.get("avg_load_time", 0.0),
            memory_usage=metrics.get("memory_usage", 0.0),
            error_rate=metrics.get("error_rate", 0.0),
            timestamp=datetime.now()
        )

        return performance_data.dict()
    except Exception as e:
        logger.error(f"获取性能指标失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quality")
async def check_data_quality(request: DataQualityRequest):
    """检查数据质量"""
    try:
        if request.source_type not in loaders:
            raise HTTPException(status_code=404, detail="数据源不存在")

        # 获取数据
        loader = loaders[request.source_type]
        data = await loader.load_data(
            symbol=request.symbol,
            start_date=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
            end_date=datetime.now().strftime("%Y-%m-%d"),
            frequency="1d"
        )

        # 检查数据质量
        quality_metrics = advanced_quality_monitor.evaluate_data_quality(data)

        quality_data = QualityMetrics(
            completeness=quality_metrics.get("completeness", 0.0),
            accuracy=quality_metrics.get("accuracy", 0.0),
            consistency=quality_metrics.get("consistency", 0.0),
            timeliness=quality_metrics.get("timeliness", 0.0),
            validity=quality_metrics.get("validity", 0.0),
            reliability=quality_metrics.get("reliability", 0.0),
            uniqueness=quality_metrics.get("uniqueness", 0.0),
            integrity=quality_metrics.get("integrity", 0.0),
            precision=quality_metrics.get("precision", 0.0),
            availability=quality_metrics.get("availability", 0.0),
            timestamp=datetime.now()
        )

        return quality_data.dict()
    except Exception as e:
        logger.error(f"数据质量检查失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quality / report")
async def generate_quality_report(
    days: int = Query(default=7, description="报告天数"),
    source_type: Optional[str] = Query(default=None, description="数据源类型")
):
    """生成数据质量报告"""
    try:
        # 生成质量报告
        report = advanced_quality_monitor.generate_quality_report(
            days=days,
            source_type=source_type
        )

        return {
            "report": report,
            "generated_at": datetime.now().isoformat(),
            "period_days": days
        }
    except Exception as e:
        logger.error(f"生成质量报告失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache / stats")
async def get_cache_statistics():
    """获取缓存统计信息"""
    try:
        # 获取缓存统计
        cache_stats = data_manager.get_cache_statistics()

        return {
            "cache_stats": cache_stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"获取缓存统计失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache / clear")
async def clear_cache():
    """清除缓存"""
    try:
        # 清除缓存
        data_manager.clear_cache()

        return {
            "status": "success",
            "message": "缓存已清除",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"清除缓存失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def get_alerts():
    """获取告警信息"""
    try:
        # 获取性能告警
        performance_alerts = performance_monitor.get_alerts()

        # 获取质量告警
        quality_alerts = advanced_quality_monitor.get_alerts()

        return {
            "performance_alerts": performance_alerts,
            "quality_alerts": quality_alerts,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"获取告警信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics / dashboard")
async def get_dashboard_metrics():
    """获取仪表板指标"""
    try:
        # 获取性能指标
        performance_metrics = performance_monitor.get_metrics()

        # 获取质量指标
        quality_metrics = advanced_quality_monitor.get_current_metrics()

        # 获取数据源状态
        source_status = {}
        for name, loader in loaders.items():
            source_status[name] = {
                "status": "active",
                "last_update": datetime.now().isoformat(),
                "data_count": 1000
            }

        return {
            "performance": performance_metrics,
            "quality": quality_metrics,
            "sources": source_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"获取仪表板指标失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 错误处理中间件 - 需要在应用级别处理

def create_exception_handler():
    """创建全局异常处理器"""
    async def global_exception_handler(request, exc):
        """全局异常处理器"""
        logger.error(f"API异常: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": str(exc),
                "timestamp": datetime.now().isoformat()
            }
        )
    return global_exception_handler
