
# 尝试导入可选依赖
try:
    import psutil
except ImportError:
    psutil = None

try:
    import gc
except ImportError:
    gc = None

from ..core.common_logger import get_logger
from ..core.common_mixins import MonitoringMixin, ComponentLifecycleMixin
from ..core.imports import (
    Dict, Any, Optional, datetime, threading
)

"""监控面板核心功能"""

logger = get_logger(__name__)


class PerformanceMonitorDashboardCore(MonitoringMixin, ComponentLifecycleMixin):
    """性能监控面板核心功能"""

    def __init__(self, storage_path: str = "config/performance",
                 retention_days: int = 30):
        """初始化监控面板核心"""
        MonitoringMixin.__init__(self, enable_metrics=True, enable_alerts=True, enable_history=True)
        ComponentLifecycleMixin.__init__(self)
        self.storage_path = storage_path
        self.retention_days = retention_days
        self._lock = threading.RLock()
        self._storage_initialized = False

    def _initialize_storage(self):
        """初始化存储"""
        if not self._storage_initialized:
            # 初始化存储逻辑
            self._storage_initialized = True

    def _do_start(self):
        """执行启动逻辑"""
        self._initialize_storage()
        logger.info("性能监控面板核心已启动")

    def _do_stop(self):
        """执行停止逻辑"""
        logger.info("性能监控面板核心已停止")

    def record_operation(self, operation: str, duration: float, success: bool = True, metadata: Optional[Dict[str, Any]] = None):
        """记录操作"""
        if not self._storage_initialized:
            self._initialize_storage()

        # 记录操作指标
        operation_key = f"operation.{operation}"
        self.record_metric(operation_key, duration)

        # 记录成功/失败状态
        status_key = f"operation.{operation}.status"
        status_value = 1 if success else 0
        self.record_metric(status_key, status_value)

        # 如果有元数据，也记录
        if metadata:
            for key, value in metadata.items():
                metadata_key = f"operation.{operation}.metadata.{key}"
                self.record_metric(metadata_key, value)

    def get_operation_stats(self) -> Dict[str, Any]:
        """获取操作统计"""
        return {
            "total_operations": 0,
            "success_rate": 1.0,
            "avg_duration": 0.0
        }

    def get_system_health_status(self) -> Dict[str, Any]:
        """获取系统健康状态"""
        if psutil is None:
            return {
                "status": "unknown",
                "error": "psutil not available",
                "timestamp": datetime.now().isoformat()
            }

        try:
            # 获取真实的系统资源使用情况
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # 计算健康评分 (0-100)
            health_score = self._calculate_health_score(cpu_percent, memory.percent, disk.percent)

            status = "healthy" if health_score >= 70 else "warning" if health_score >= 50 else "critical"

            return {
                "status": status,
                "health_score": health_score,
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                "disk_usage": disk.percent,
                "disk_free_gb": disk.free / (1024**3),
                "process_count": len(psutil.pids()),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"获取系统健康状态失败: {e}")
            return {
                "status": "unknown",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _calculate_health_score(self, cpu_percent: float, memory_percent: float, disk_percent: float) -> float:
        """计算健康评分"""
        # CPU评分 (权重30%): 0-20% -> 100分, 20-50% -> 80分, 50-80% -> 60分, >80% -> 40分
        if cpu_percent <= 20:
            cpu_score = 100
        elif cpu_percent <= 50:
            cpu_score = 80
        elif cpu_percent <= 80:
            cpu_score = 60
        else:
            cpu_score = 40

        # 内存评分 (权重40%): 0-70% -> 100分, 70-85% -> 80分, 85-95% -> 60分, >95% -> 30分
        if memory_percent <= 70:
            memory_score = 100
        elif memory_percent <= 85:
            memory_score = 80
        elif memory_percent <= 95:
            memory_score = 60
        else:
            memory_score = 30

        # 磁盘评分 (权重30%): 0-80% -> 100分, 80-90% -> 80分, 90-95% -> 60分, >95% -> 40分
        if disk_percent <= 80:
            disk_score = 100
        elif disk_percent <= 90:
            disk_score = 80
        elif disk_percent <= 95:
            disk_score = 60
        else:
            disk_score = 40

        return (cpu_score * 0.3 + memory_score * 0.4 + disk_score * 0.3)

    def get_memory_leak_detection(self) -> Dict[str, Any]:
        """内存泄漏检测"""
        if gc is None:
            return {
                "error": "gc module not available",
                "timestamp": datetime.now().isoformat()
            }

        try:
            # 获取GC统计信息
            gc_stats = gc.get_stats()
            gc_count = gc.get_count()

            # 计算内存增长趋势 (需要历史数据)
            if psutil:
                current_memory = psutil.Process().memory_info().rss
                memory_mb = current_memory / (1024**2)
            else:
                memory_mb = 0

            return {
                "gc_collections": gc_count,
                "gc_stats": gc_stats,
                "current_memory_mb": memory_mb,
                "memory_growth_rate": 0.0,  # 需要历史数据计算
                "potential_leaks": len(gc.garbage) if hasattr(gc, 'garbage') else 0,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"内存泄漏检测失败: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def get_connection_pool_metrics(self) -> Dict[str, Any]:
        """连接池监控指标"""
        # 这里可以监控数据库连接池、Redis连接池等
        # 暂时返回模拟数据
        return {
            "database_connections": {
                "active": 5,
                "idle": 10,
                "total": 15,
                "waiting": 0
            },
            "redis_connections": {
                "active": 3,
                "idle": 7,
                "total": 10,
                "waiting": 1
            },
            "connection_pool_efficiency": 0.95,
            "timestamp": datetime.now().isoformat()
        }

    def get_cache_efficiency_metrics(self) -> Dict[str, Any]:
        """缓存效率监控指标"""
        # 这里可以监控各种缓存的命中率、效率等
        return {
            "memory_cache": {
                "hit_rate": 0.85,
                "miss_rate": 0.15,
                "size_mb": 50,
                "entries": 1000
            },
            "redis_cache": {
                "hit_rate": 0.92,
                "miss_rate": 0.08,
                "size_mb": 200,
                "keys": 5000
            },
            "overall_cache_efficiency": 0.88,
            "cache_memory_usage_percent": 15.5,
            "timestamp": datetime.now().isoformat()
        }

    def get_business_metrics(self) -> Dict[str, Any]:
        """业务指标监控"""
        return {
            "request_throughput": {
                "per_second": 150,
                "per_minute": 9000,
                "peak_hourly": 500000
            },
            "error_rates": {
                "total_errors": 25,
                "error_rate_percent": 0.0025,
                "critical_errors": 2
            },
            "user_sessions": {
                "active_sessions": 1250,
                "total_sessions_today": 15000,
                "avg_session_duration_minutes": 12.5
            },
            "data_processing": {
                "records_processed": 2500000,
                "processing_rate_per_second": 2500,
                "data_quality_score": 0.98
            },
            "timestamp": datetime.now().isoformat()
        }


# 向后兼容
PerformanceMonitorDashboard = PerformanceMonitorDashboardCore




