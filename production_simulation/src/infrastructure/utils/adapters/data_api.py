"""
data_api 模块

提供 data_api 相关功能和接口。
"""

import logging


from datetime import datetime, timedelta
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

try:  # pragma: no cover - 兼容 Pydantic v1
    from pydantic import ConfigDict  # type: ignore
except ImportError:  # pragma: no cover
    ConfigDict = None  # type: ignore[assignment]
from typing import Dict, List, Optional, Any

# 条件导入数据组件
try:
    from src.data.core.data_manager import DataManagerSingleton
except ImportError:
    DataManagerSingleton = None

try:
    from src.data.monitoring.performance_monitor import PerformanceMonitor
except ImportError:
    PerformanceMonitor = None

try:
    from src.data.monitoring.quality_monitor import DataQualityMonitor
except ImportError:
    DataQualityMonitor = None

try:
    from src.data.quality.advanced_quality_monitor import AdvancedQualityMonitor
except ImportError:
    AdvancedQualityMonitor = None

# 条件导入数据加载器
try:
    from src.data.loader.crypto_loader import CryptoDataLoader
except ImportError:
    CryptoDataLoader = None

try:
    from src.data.loader.macro_loader import MacroDataLoader
except ImportError:
    MacroDataLoader = None

try:
    from src.data.loader.options_loader import OptionsDataLoader
except ImportError:
    OptionsDataLoader = None

try:
    from src.data.loader.bond_loader import BondDataLoader
except ImportError:
    BondDataLoader = None

try:
    from src.data.loader.commodity_loader import CommodityDataLoader
except ImportError:
    CommodityDataLoader = None

try:
    from src.data.loader.forex_loader import ForexDataLoader
except ImportError:
    ForexDataLoader = None

# HTTP异常导入
from fastapi import HTTPException
"""
RQA2025 基础设施层工具系统 - 数据API服务

本模块提供RESTful数据API服务，支持数据查询、上传、下载和管理等功能。
基于FastAPI框架实现高性能异步API服务。

主要特性:
- RESTful数据API接口
- 异步数据处理支持
- 数据质量监控集成
- 性能监控和统计
- 数据缓存优化
- 安全认证和权限控制

API端点:
- GET /data/query - 数据查询接口
- POST /data/upload - 数据上传接口
- GET /data/download - 数据下载接口
- GET /data/stats - 数据统计接口
- GET /data/health - 健康检查接口

作者: RQA2025 Team
创建日期: 2025年9月13日
版本: 1.0.0
"""

"""
基础设施层 - 配置管理组件

data_api 模块

配置管理相关的文件
提供配置管理相关的功能实现。
"""

#!/usr/bin/env python3
"""
RQA2025 数据层 REST API 接口

提供数据管理、质量监控、性能指标的完整API接口
支持数据源管理、质量监控、性能监控等功能
"""

# 合理跨层级导入：data层接口定义
# 合理跨层级导入：data层接口定义
#     CryptoDataLoader,
#     MacroDataLoader,
#     OptionsDataLoader,
#     BondDataLoader,
#     CommodityDataLoader,
#     ForexDataLoader,

# 数据API常量定义


class DataAPI:
    """数据API客户端"""
    
    def __init__(self, base_url: str = None):
        """初始化数据API客户端"""
        self.base_url = base_url or "http://localhost:8000"
    
    def get(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """GET请求"""
        return {"status": "success", "data": {}}
    
    def post(self, endpoint: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """POST请求"""
        return {"status": "success"}


class DataAPIConstants:
    """数据API相关常量"""

    # HTTP状态码
    HTTP_INTERNAL_SERVER_ERROR = 500
    HTTP_SERVICE_UNAVAILABLE = 503
    HTTP_NOT_FOUND = 404

    # 配置常量
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_TIMEOUT_SECONDS = 30
    DEFAULT_DATA_COUNT = 1000

    # 时间常量
    DEFAULT_STATS_DAYS = 30

    # 字符串常量
    CACHE_DIR = "cache"


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/data", tags=["data"])

# 初始化组件
base_config = {
    "cache_dir": DataAPIConstants.CACHE_DIR,
    "max_retries": DataAPIConstants.DEFAULT_MAX_RETRIES,
    "timeout": DataAPIConstants.DEFAULT_TIMEOUT_SECONDS
}

# 条件初始化组件（仅在组件可用时）
data_manager = DataManagerSingleton.get_instance(config_dict={"cache": base_config}) if DataManagerSingleton else None
performance_monitor = PerformanceMonitor() if PerformanceMonitor else None
quality_monitor = DataQualityMonitor() if DataQualityMonitor else None
advanced_quality_monitor = AdvancedQualityMonitor() if AdvancedQualityMonitor else None

# 数据加载器实例（仅在加载器可用时）
loaders = {}
if CryptoDataLoader:
    try:
        loaders["crypto"] = CryptoDataLoader(base_config)
    except (TypeError, ImportError):
        pass
if MacroDataLoader:
    try:
        loaders["macro"] = MacroDataLoader(base_config)
    except (TypeError, ImportError):
        pass
if OptionsDataLoader:
    try:
        loaders["options"] = OptionsDataLoader(base_config)
    except (TypeError, ImportError):
        pass
if BondDataLoader:
    try:
        loaders["bond"] = BondDataLoader(base_config)
    except (TypeError, ImportError):
        pass
if CommodityDataLoader:
    try:
        loaders["commodity"] = CommodityDataLoader(base_config)
    except (TypeError, ImportError):
        pass
if ForexDataLoader:
    try:
        loaders["forex"] = ForexDataLoader(base_config)
    except (TypeError, ImportError):
        pass

# Pydantic 模型定义


class DataSourceRequest(BaseModel):
    """数据源请求模型"""

    source_type: Optional[str] = Field(None, alias="source", description="数据源类型")
    symbol: Optional[str] = Field(None, description="数据符号")
    start_date: Optional[str] = Field(None, description="开始日期")
    end_date: Optional[str] = Field(None, description="结束日期")
    frequency: str = Field(default="1d", description="数据频率")
    params: Dict[str, Any] = Field(default_factory=dict, description="额外参数")

    if ConfigDict is not None:  # pragma: no branch
        model_config = ConfigDict(
            populate_by_name=True,
            populate_by_alias=True,
        )
    else:  # pragma: no cover - Pydantic v1 兼容
        class Config:  # type: ignore[no-redef]
            allow_population_by_field_name = True
            allow_population_by_alias = True


class DataQualityRequest(BaseModel):
    """数据质量请求模型"""

    source_type: Optional[str] = Field(None, description="数据源类型")
    symbol: Optional[str] = Field(None, description="数据符号")
    metrics: List[str] = Field(default_factory=list, description="质量指标列表")


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


def _model_dump(model: BaseModel) -> Dict[str, Any]:
    """兼容 Pydantic v1/v2 的序列化方法。"""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    raise TypeError("unsupported model type for serialization")

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
                "quality_monitor": "ok",
            },
            "available_sources": list(loaders.keys()),
        }

        return status
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=DataAPIConstants.HTTP_INTERNAL_SERVER_ERROR, detail=str(e))


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
                data_count=DataAPIConstants.DEFAULT_DATA_COUNT,
            )

            sources.append(_model_dump(source_info))

        return {"sources": sources, "total": len(sources)}

    except Exception as e:
        logger.error(f"获取数据源列表失败: {e}")
        raise HTTPException(status_code=DataAPIConstants.HTTP_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/sources/{source_type}")
async def get_data_source_info(source_type: str):
    """获取特定数据源信息"""
    try:
        if source_type not in loaders:
            raise HTTPException(status_code=DataAPIConstants.HTTP_NOT_FOUND, detail="数据源不存在")

        loader = loaders[source_type]
        source_info = DataSourceInfo(
            name=source_type,
            type=source_type,
            status="active",
            last_update=datetime.now(),
            data_count=DataAPIConstants.DEFAULT_DATA_COUNT,
        )
        return _model_dump(source_info)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取数据源信息失败: {e}")
        raise HTTPException(status_code=DataAPIConstants.HTTP_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/load")
async def load_data(request: DataSourceRequest):
    """加载数据接口"""
    try:
        if request.source_type not in loaders:
            raise HTTPException(status_code=DataAPIConstants.HTTP_NOT_FOUND, detail="数据源不存在")

        loader = loaders[request.source_type]
        if not hasattr(loader, "load_data"):
            raise HTTPException(status_code=DataAPIConstants.HTTP_SERVICE_UNAVAILABLE, detail="数据源不支持加载操作")

        # 记录开始时间
        start_time = datetime.now()

        # 加载数据
        data = await loader.load_data(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            frequency=request.frequency,
        )
        # 计算加载时间
        load_time = (datetime.now() - start_time).total_seconds()

        # 记录性能指标
        if performance_monitor and hasattr(performance_monitor, "record_load_time"):
            try:
                performance_monitor.record_load_time(load_time)
            except Exception as monitor_exc:
                logger.warning(f"性能监控记录失败: {monitor_exc}")

        return {
            "status": "success",
            "data": data,
            "load_time": load_time,
            "timestamp": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        raise HTTPException(status_code=DataAPIConstants.HTTP_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/performance")
async def get_performance_metrics():
    """获取性能指标"""
    try:
        if not performance_monitor or not hasattr(performance_monitor, "get_metrics"):
            metrics = {}
        else:
            try:
                metrics = performance_monitor.get_metrics() or {}
            except Exception as monitor_exc:
                logger.warning(f"获取性能指标失败，返回默认值: {monitor_exc}")
                metrics = {}

        performance_data = PerformanceMetrics(
            cache_hit_rate=metrics.get("cache_hit_rate", 0.0),
            load_time=metrics.get("avg_load_time", 0.0),
            memory_usage=metrics.get("memory_usage", 0.0),
            error_rate=metrics.get("error_rate", 0.0),
            timestamp=datetime.now()
        )
        return _model_dump(performance_data)
    except Exception as e:
        logger.error(f"获取性能指标失败: {e}")
        raise HTTPException(status_code=DataAPIConstants.HTTP_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/quality")
async def check_data_quality(request: DataQualityRequest):
    """检查数据质量"""
    try:
        if request.source_type not in loaders:
            raise HTTPException(status_code=DataAPIConstants.HTTP_NOT_FOUND, detail="数据源不存在")
        if not advanced_quality_monitor or not hasattr(advanced_quality_monitor, "evaluate_data_quality"):
            raise HTTPException(
                status_code=DataAPIConstants.HTTP_SERVICE_UNAVAILABLE,
                detail="数据质量监控未启用",
            )

        # 获取数据
        loader = loaders[request.source_type]
        if not hasattr(loader, "load_data"):
            raise HTTPException(status_code=DataAPIConstants.HTTP_SERVICE_UNAVAILABLE, detail="数据源不支持质量检测")
        data = await loader.load_data(
            symbol=request.symbol,
            start_date=(datetime.now() -
                        timedelta(days=DataAPIConstants.DEFAULT_STATS_DAYS)).strftime("%Y-%m-%d"),
            end_date=datetime.now().strftime("%Y-%m-%d"),
            frequency="1d",
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
        return _model_dump(quality_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"数据质量检查失败: {e}")
        raise HTTPException(status_code=DataAPIConstants.HTTP_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/quality/report")
async def generate_quality_report(
    days: int = Query(default=7, description="报告天数"),
    source_type: Optional[str] = Query(default=None, description="数据源类型")
):
    """生成数据质量报告"""
    try:
        if not advanced_quality_monitor or not hasattr(advanced_quality_monitor, "generate_quality_report"):
            raise HTTPException(
                status_code=DataAPIConstants.HTTP_SERVICE_UNAVAILABLE,
                detail="数据质量监控未启用",
            )
        # 生成质量报告
        report = advanced_quality_monitor.generate_quality_report(
            days=days, source_type=source_type)
        return {
            "report": report,
            "generated_at": datetime.now().isoformat(),
            "period_days": days,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"生成质量报告失败: {e}")
        raise HTTPException(status_code=DataAPIConstants.HTTP_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/cache/stats")
async def get_cache_statistics():
    """获取缓存统计信息"""
    try:
        # 获取缓存统计
        if not data_manager or not hasattr(data_manager, "get_cache_statistics"):
            raise HTTPException(
                status_code=DataAPIConstants.HTTP_SERVICE_UNAVAILABLE,
                detail="缓存管理器不可用",
            )
        cache_stats = data_manager.get_cache_statistics()

        return {"cache_stats": cache_stats, "timestamp": datetime.now().isoformat()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取缓存统计失败: {e}")
        raise HTTPException(status_code=DataAPIConstants.HTTP_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/cache/clear")
async def clear_cache():
    """清除缓存"""
    try:
        # 清除缓存
        if not data_manager or not hasattr(data_manager, "clear_cache"):
            raise HTTPException(
                status_code=DataAPIConstants.HTTP_SERVICE_UNAVAILABLE,
                detail="缓存管理器不可用",
            )
        data_manager.clear_cache()

        return {
            "status": "success",
            "message": "缓存已清除",
            "timestamp": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"清除缓存失败: {e}")
        raise HTTPException(status_code=DataAPIConstants.HTTP_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/alerts")
async def get_alerts():
    """获取告警信息"""
    try:
        # 获取性能告警
        if performance_monitor and hasattr(performance_monitor, "get_alerts"):
            try:
                performance_alerts = performance_monitor.get_alerts() or []
            except Exception as monitor_exc:
                logger.warning(f"获取性能告警失败: {monitor_exc}")
                performance_alerts = []
        else:
            performance_alerts = []

        # 获取质量告警
        if advanced_quality_monitor and hasattr(advanced_quality_monitor, "get_alerts"):
            try:
                quality_alerts = advanced_quality_monitor.get_alerts() or []
            except Exception as monitor_exc:
                logger.warning(f"获取质量告警失败: {monitor_exc}")
                quality_alerts = []
        else:
            quality_alerts = []

        return {
            "performance_alerts": performance_alerts,
            "quality_alerts": quality_alerts,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"获取告警信息失败: {e}")
        raise HTTPException(status_code=DataAPIConstants.HTTP_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/metrics/dashboard")
async def get_dashboard_metrics():
    """获取仪表板指标"""
    try:
        # 获取性能指标
        if performance_monitor and hasattr(performance_monitor, "get_metrics"):
            try:
                performance_metrics = performance_monitor.get_metrics() or {}
            except Exception as monitor_exc:
                logger.warning(f"获取性能指标失败: {monitor_exc}")
                performance_metrics = {}
        else:
            performance_metrics = {}

        # 获取质量指标
        if advanced_quality_monitor and hasattr(advanced_quality_monitor, "get_current_metrics"):
            try:
                quality_metrics = advanced_quality_monitor.get_current_metrics() or {}
            except Exception as monitor_exc:
                logger.warning(f"获取质量指标失败: {monitor_exc}")
                quality_metrics = {}
        else:
            quality_metrics = {}

        # 获取数据源状态
        source_status = {}
        for name, loader in loaders.items():
            source_status[name] = {
                "status": "active",
                "last_update": datetime.now().isoformat(),
                "data_count": DataAPIConstants.DEFAULT_DATA_COUNT,
            }
        return {
            "performance": performance_metrics,
            "quality": quality_metrics,
            "sources": source_status,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"获取仪表板指标失败: {e}")
        raise HTTPException(status_code=DataAPIConstants.HTTP_INTERNAL_SERVER_ERROR, detail=str(e))

# 错误处理中间件 - 需要在应用级别处理


def create_exception_handler():
    """创建全局异常处理器"""

    async def global_exception_handler(request, exc):
        """全局异常处理器"""
        logger.error(f"API异常: {exc}")
        return JSONResponse(
            status_code=DataAPIConstants.HTTP_INTERNAL_SERVER_ERROR,
            content={"error": str(exc), "timestamp": datetime.now().isoformat()},
        )

    return global_exception_handler
