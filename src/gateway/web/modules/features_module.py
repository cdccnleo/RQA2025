"""
特征层监控模块

集成特征层可视化监控面板到统一Web管理界面，
提供特征工程性能监控、数据质量监控、配置管理等功能。
"""

import asyncio
import time
import psutil
from typing import Dict, List, Any
from datetime import datetime, timedelta
import logging

from fastapi import WebSocket

from .base_module import BaseModule, ModuleConfig, ModuleData, ModuleStatus
from src.engine.logging.unified_logger import get_unified_logger

logger = logging.getLogger(__name__)


class FeaturesModuleConfig(ModuleConfig):

    """特征层监控模块配置"""
    refresh_interval: float = 5.0
    enable_real_time_monitoring: bool = True
    enable_data_quality_monitoring: bool = True
    enable_performance_monitoring: bool = True
    enable_config_monitoring: bool = True
    alert_thresholds: Dict[str, float] = {
        "feature_engineering_time": 1000.0,  # ms
        "data_quality_score": 80.0,          # %
        "memory_usage": 85.0,                # %
        "cpu_usage": 90.0,                   # %
        "error_rate": 5.0                    # %
    }
    history_hours: int = 24
    auto_refresh: bool = True
    output_dir: str = "./features_dashboard"


class FeaturesModule(BaseModule):

    """
    特征层监控模块

    提供特征工程性能监控、数据质量监控、配置管理等功能，
    集成到统一Web管理界面。
    """

    def __init__(self, config: FeaturesModuleConfig):

        super().__init__(config)
        self.logger = get_unified_logger(__name__)

        # 监控数据缓存
        self.metrics_cache = {}
        self.last_update = time.time()
        self.alerts = []

        # 模拟历史数据
        self.historical_data = {}
        self._init_historical_data()

        # WebSocket连接管理
        self.active_connections: List[WebSocket] = []

        self.logger.info("特征层监控模块初始化完成")

    def _init_historical_data(self):
        """初始化历史数据"""
        now = datetime.now()
        for i in range(24):
            timestamp = now - timedelta(hours=i)
            self.historical_data[f"feature_engineering_time_{i}"] = {
                "timestamp": timestamp.isoformat(),
                "value": 500 + (i * 20) % 300,  # 模拟数据
                "status": "normal"
            }
            self.historical_data[f"data_quality_score_{i}"] = {
                "timestamp": timestamp.isoformat(),
                "value": 85 + (i * 5) % 15,  # 模拟数据
                "status": "normal"
            }

    def _register_routes(self):
        """注册模块路由"""
        self._add_common_routes()

        @self.router.get("/metrics")
        async def get_metrics():
            """获取当前监控指标"""
            return await self._get_current_metrics()

        @self.router.get("/alerts")
        async def get_alerts():
            """获取告警信息"""
            return await self._get_alerts()

        @self.router.get("/performance")
        async def get_performance():
            """获取性能分析数据"""
            return await self._get_performance_data()

        @self.router.get("/quality")
        async def get_quality():
            """获取数据质量数据"""
            return await self._get_quality_data()

    async def get_module_data(self) -> ModuleData:
        """获取模块数据"""
        try:
            metrics = await self._get_current_metrics()

            return ModuleData(
                module_id=self.config.name,
                data_type="features_metrics",
                content={
                    "current_metrics": metrics,
                    "alerts_count": len(self.alerts),
                    "system_status": "healthy",
                    "last_updated": datetime.now().isoformat()
                },
                timestamp=datetime.now()
            )
        except Exception as e:
            self.logger.error(f"获取特征模块数据失败: {str(e)}")
            return ModuleData(
                module_id=self.config.name,
                data_type="features_metrics",
                content={"error": str(e)},
                timestamp=datetime.now()
            )

    async def get_module_status(self) -> ModuleStatus:
        """获取模块状态"""
        try:
            # 检查系统资源
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            # 检查告警数量
            critical_alerts = len([a for a in self.alerts if a.get('severity') == 'critical'])
            warning_alerts = len([a for a in self.alerts if a.get('severity') == 'warning'])

            # 确定整体状态
            if critical_alerts > 0:
                overall_status = "critical"
            elif warning_alerts > 0:
                overall_status = "warning"
            elif cpu_usage < 80 and memory.percent < 80:
                overall_status = "healthy"
            else:
                overall_status = "degraded"

            return ModuleStatus(
                module_id=self.config.name,
                status=overall_status,
                is_healthy=overall_status in ["healthy", "warning"],
                last_activity=datetime.now(),
                error_count=0,
                uptime=0.0,
                performance_metrics={
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory.percent,
                    "alerts_count": len(self.alerts),
                    "metrics_count": len(self.metrics_cache)
                },
                alerts=self.alerts
            )
        except Exception as e:
            self.logger.error(f"获取特征模块状态失败: {str(e)}")
            return ModuleStatus(
                module_id=self.config.name,
                status="error",
                is_healthy=False,
                last_activity=datetime.now(),
                error_count=1,
                uptime=0.0,
                performance_metrics={"error": str(e)},
                alerts=[]
            )

    async def validate_permissions(self, user_permissions: List[str]) -> bool:
        """验证用户权限"""
        required_permissions = ["read"]
        return all(perm in user_permissions for perm in required_permissions)

    async def _initialize_module(self):
        """初始化模块内部逻辑"""
        self.logger.info("特征层监控模块初始化完成")
        return True

    async def _start_module(self):
        """启动模块内部逻辑"""
        # 启动监控任务
        asyncio.create_task(self._update_metrics_loop())
        self.logger.info("特征层监控模块启动完成")
        return True

    async def _stop_module(self):
        """停止模块内部逻辑"""
        self.logger.info("特征层监控模块停止完成")
        return True

    async def _update_metrics_loop(self):
        """更新指标的循环任务"""
        while True:
            try:
                await self._update_metrics()
                await asyncio.sleep(self.config.refresh_interval)
            except Exception as e:
                self.logger.error(f"更新指标失败: {str(e)}")
                await asyncio.sleep(5)

    async def _update_metrics(self):
        """更新监控指标"""
        try:
            # 获取系统资源使用情况
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            # 模拟特征工程指标
            feature_engineering_time = 500 + (int(time.time()) % 300)
            data_quality_score = 85 + (int(time.time()) % 15)

            # 检查告警阈值
            alerts = []
            if feature_engineering_time > self.config.alert_thresholds["feature_engineering_time"]:
                alerts.append({
                    "type": "performance",
                    "severity": "warning",
                    "message": f"特征工程时间过长: {feature_engineering_time}ms",
                    "timestamp": datetime.now().isoformat()
                })

            if data_quality_score < self.config.alert_thresholds["data_quality_score"]:
                alerts.append({
                    "type": "quality",
                    "severity": "critical",
                    "message": f"数据质量评分过低: {data_quality_score}%",
                    "timestamp": datetime.now().isoformat()
                })

            if cpu_usage > self.config.alert_thresholds["cpu_usage"]:
                alerts.append({
                    "type": "system",
                    "severity": "warning",
                    "message": f"CPU使用率过高: {cpu_usage}%",
                    "timestamp": datetime.now().isoformat()
                })

            # 更新缓存
            self.metrics_cache = {
                "feature_engineering_time": feature_engineering_time,
                "data_quality_score": data_quality_score,
                "cpu_usage": cpu_usage,
                "memory_usage": memory.percent,
                "timestamp": datetime.now().isoformat()
            }

            self.alerts = alerts
            self.last_update = time.time()

        except Exception as e:
            self.logger.error(f"更新指标失败: {str(e)}")

    async def _get_current_metrics(self) -> Dict[str, Any]:
        """获取当前指标"""
        if not self.metrics_cache:
            await self._update_metrics()

        return {
            "metrics": self.metrics_cache,
            "last_update": self.last_update,
            "refresh_interval": self.config.refresh_interval
        }

    async def _get_alerts(self) -> List[Dict[str, Any]]:
        """获取告警信息"""
        return self.alerts

    async def _get_performance_data(self) -> Dict[str, Any]:
        """获取性能分析数据"""
        metrics = await self._get_current_metrics()

        return {
            "feature_engineering_time": metrics["metrics"].get("feature_engineering_time", 0),
            "cpu_usage": metrics["metrics"].get("cpu_usage", 0),
            "memory_usage": metrics["metrics"].get("memory_usage", 0),
            "performance_score": self._calculate_performance_score(metrics["metrics"]),
            "bottleneck": self._identify_bottleneck(metrics["metrics"])
        }

    async def _get_quality_data(self) -> Dict[str, Any]:
        """获取数据质量数据"""
        metrics = await self._get_current_metrics()

        return {
            "data_quality_score": metrics["metrics"].get("data_quality_score", 0),
            "quality_trend": "stable",
            "quality_issues": [],
            "last_quality_check": datetime.now().isoformat()
        }

    def _calculate_performance_score(self, metrics: Dict[str, Any]) -> float:
        """计算性能评分"""
        score = 100.0

        # 根据特征工程时间调整评分
        if metrics.get("feature_engineering_time", 0) > 800:
            score -= 20
        elif metrics.get("feature_engineering_time", 0) > 600:
            score -= 10

        # 根据CPU使用率调整评分
        if metrics.get("cpu_usage", 0) > 90:
            score -= 30
        elif metrics.get("cpu_usage", 0) > 80:
            score -= 15

        # 根据内存使用率调整评分
        if metrics.get("memory_usage", 0) > 90:
            score -= 20
        elif metrics.get("memory_usage", 0) > 80:
            score -= 10

        return max(0, score)

    def _identify_bottleneck(self, metrics: Dict[str, Any]) -> str:
        """识别性能瓶颈"""
        if metrics.get("cpu_usage", 0) > 90:
            return "CPU资源不足"
        elif metrics.get("memory_usage", 0) > 90:
            return "内存资源不足"
        elif metrics.get("feature_engineering_time", 0) > 800:
            return "特征工程性能问题"
        else:
            return "无瓶颈"

    @classmethod
    def get_default_config(cls) -> FeaturesModuleConfig:
        """获取默认配置"""
        return FeaturesModuleConfig(
            name="features_monitoring",
            display_name="特征层监控",
            description="特征工程性能和数据质量监控",
            icon="fas fa - chart - line",
            route="/features",
            refresh_interval=5.0,
            enable_real_time_monitoring=True,
            enable_data_quality_monitoring=True,
            enable_performance_monitoring=True,
            enable_config_monitoring=True,
            alert_thresholds={
                "feature_engineering_time": 1000.0,
                "data_quality_score": 80.0,
                "memory_usage": 85.0,
                "cpu_usage": 90.0,
                "error_rate": 5.0
            },
            history_hours=24,
            auto_refresh=True,
            output_dir="./features_dashboard"
        )
