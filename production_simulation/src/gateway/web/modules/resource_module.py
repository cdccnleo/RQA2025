import logging
#!/usr / bin / env python
# -*- coding: utf-8 -*-

"""
RQA2025 统一Web管理界面 - 资源监控模块
整合系统资源监控功能到统一管理平台
"""

import asyncio
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .base_module import BaseModule, ModuleConfig, ModuleData, ModuleStatus  # 当前层级内部导入：基础模块类
from src.infrastructure.resource.resource_manager import ResourceManager  # 合理跨层级导入：基础设施层资源管理
from src.infrastructure.resource.gpu_manager import GPUManager  # 合理跨层级导入：基础设施层GPU管理
from src.infrastructure.resource.quota_manager import QuotaManager  # 合理跨层级导入：基础设施层配额管理

logger = logging.getLogger(__name__)


class SystemMetrics(BaseModel):

    """系统资源指标模型"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    load_average: List[float]
    timestamp: datetime


class GPUMetrics(BaseModel):

    """GPU资源指标模型"""
    device_id: str
    memory_used: float
    memory_total: float
    memory_usage: float
    utilization: float
    temperature: float
    power_consumption: float
    timestamp: datetime


class ResourceAlert(BaseModel):

    """资源告警模型"""
    type: str
    severity: str
    message: str
    threshold: float
    current_value: float
    timestamp: datetime
    resource_id: str


class ResourceModuleConfig(ModuleConfig):

    """资源监控模块配置"""
    refresh_interval: int = 5  # 刷新间隔(秒)
    alert_thresholds: Dict[str, float] = {
        "cpu_usage": 90.0,  # CPU使用率阈值(%)
        "memory_usage": 85.0,  # 内存使用率阈值(%)
        "disk_usage": 90.0,  # 磁盘使用率阈值(%)
        "gpu_memory_usage": 90.0,  # GPU显存使用率阈值(%)
        "gpu_utilization": 95.0,  # GPU利用率阈值(%)
        "gpu_temperature": 85.0  # GPU温度阈值(°C)
    }
    enable_alerts: bool = True
    enable_history: bool = True
    history_hours: int = 24
    monitor_gpu: bool = True
    monitor_network: bool = True


class ResourceModule(BaseModule):

    """资源监控模块"""

    def __init__(self, config: ResourceModuleConfig):

        super().__init__(config)
        self.resource_manager = ResourceManager()
        self.gpu_manager = GPUManager()
        self.quota_manager = QuotaManager()
        self.system_metrics_history: List[SystemMetrics] = []
        self.gpu_metrics_history: List[GPUMetrics] = []
        self.alerts: List[ResourceAlert] = []
        self._setup_routes()

    def _setup_routes(self):
        """设置路由"""
        router = APIRouter(prefix="/resources", tags=["资源监控"])

        @router.get("/system / current")
        async def get_system_metrics():
            """获取当前系统资源指标"""
            try:
                # 获取CPU使用率
                cpu_usage = psutil.cpu_percent(interval=1)

                # 获取内存使用情况
                memory = psutil.virtual_memory()
                memory_usage = memory.percent

                # 获取磁盘使用情况
                disk = psutil.disk_usage('/')
                disk_usage = (disk.used / disk.total) * 100

                # 获取网络IO
                network_io = psutil.net_io_counters()
                network_data = {
                    "bytes_sent": network_io.bytes_sent,
                    "bytes_recv": network_io.bytes_recv,
                    "packets_sent": network_io.packets_sent,
                    "packets_recv": network_io.packets_recv
                }

                # 获取负载平均值
                load_average = psutil.getloadavg()

                metrics = SystemMetrics(
                    cpu_usage=cpu_usage,
                    memory_usage=memory_usage,
                    disk_usage=disk_usage,
                    network_io=network_data,
                    load_average=list(load_average),
                    timestamp=datetime.now()
                )

                # 保存到历史记录
                if self.config.enable_history:
                    self.system_metrics_history.append(metrics)
                    # 保持历史记录在指定时间范围内
                    cutoff_time = datetime.now() - timedelta(hours=self.config.history_hours)
                    self.system_metrics_history = [
                        m for m in self.system_metrics_history
                        if m.timestamp >= cutoff_time
                    ]

                return metrics

            except Exception as e:
                logger.error(f"获取系统资源指标失败: {str(e)}")
                raise HTTPException(status_code=500, detail=f"获取系统资源指标失败: {str(e)}")

        @router.get("/gpu / current")
        async def get_gpu_metrics():
            """获取当前GPU资源指标"""
            try:
                if not self.config.monitor_gpu:
                    return {"message": "GPU监控已禁用"}

                gpu_metrics = []

                # 获取GPU信息
                gpu_devices = self.gpu_manager.get_gpu_devices()

                for device in gpu_devices:
                    try:
                        # 获取GPU内存信息
                        memory_info = self.gpu_manager.get_memory_info(device['id'])

                        # 获取GPU利用率
                        utilization = self.gpu_manager.get_utilization(device['id'])

                        # 获取GPU温度
                        temperature = self.gpu_manager.get_temperature(device['id'])

                        # 获取GPU功耗
                        power = self.gpu_manager.get_power_consumption(device['id'])

                        metrics = GPUMetrics(
                            device_id=device['id'],
                            memory_used=memory_info.get('used', 0),
                            memory_total=memory_info.get('total', 0),
                            memory_usage=(memory_info.get('used', 0) /
                                          memory_info.get('total', 1)) * 100,
                            utilization=utilization,
                            temperature=temperature,
                            power_consumption=power,
                            timestamp=datetime.now()
                        )

                        gpu_metrics.append(metrics)

                        # 保存到历史记录
                        if self.config.enable_history:
                            self.gpu_metrics_history.append(metrics)
                            # 保持历史记录在指定时间范围内
                            cutoff_time = datetime.now() - timedelta(hours=self.config.history_hours)
                            self.gpu_metrics_history = [
                                m for m in self.gpu_metrics_history
                                if m.timestamp >= cutoff_time
                            ]

                    except Exception as e:
                        logger.warning(f"获取GPU {device['id']} 指标失败: {str(e)}")
                        continue

                return {
                    "gpu_metrics": gpu_metrics,
                    "count": len(gpu_metrics)
                }

            except Exception as e:
                logger.error(f"获取GPU资源指标失败: {str(e)}")
                raise HTTPException(status_code=500, detail=f"获取GPU资源指标失败: {str(e)}")

        @router.get("/quota / current")
        async def get_quota_usage():
            """获取资源配额使用情况"""
            try:
                quota_info = self.quota_manager.get_quota_usage()
                return {
                    "cpu_quota": quota_info.get('cpu', {}),
                    "memory_quota": quota_info.get('memory', {}),
                    "gpu_quota": quota_info.get('gpu', {}),
                    "disk_quota": quota_info.get('disk', {}),
                    "last_updated": datetime.now()
                }

            except Exception as e:
                logger.error(f"获取资源配额失败: {str(e)}")
                raise HTTPException(status_code=500, detail=f"获取资源配额失败: {str(e)}")

        @router.get("/alerts")
        async def get_alerts():
            """获取资源告警信息"""
            try:
                if not self.config.enable_alerts:
                    return {"alerts": [], "count": 0}

                # 获取当前系统指标
                system_metrics = await get_system_metrics()

                # 获取当前GPU指标
                gpu_response = await get_gpu_metrics()
                gpu_metrics = gpu_response.get('gpu_metrics', [])

                new_alerts = []

                # 检查系统资源告警
                # CPU使用率告警
                if system_metrics.cpu_usage > self.config.alert_thresholds["cpu_usage"]:
                    new_alerts.append(ResourceAlert(
                        type="cpu_usage",
                        severity="warning",
                        message=f"CPU使用率过高: {system_metrics.cpu_usage:.1f}%",
                        threshold=self.config.alert_thresholds["cpu_usage"],
                        current_value=system_metrics.cpu_usage,
                        timestamp=datetime.now(),
                        resource_id="system_cpu"
                    ))

                # 内存使用率告警
                if system_metrics.memory_usage > self.config.alert_thresholds["memory_usage"]:
                    new_alerts.append(ResourceAlert(
                        type="memory_usage",
                        severity="warning",
                        message=f"内存使用率过高: {system_metrics.memory_usage:.1f}%",
                        threshold=self.config.alert_thresholds["memory_usage"],
                        current_value=system_metrics.memory_usage,
                        timestamp=datetime.now(),
                        resource_id="system_memory"
                    ))

                # 磁盘使用率告警
                if system_metrics.disk_usage > self.config.alert_thresholds["disk_usage"]:
                    new_alerts.append(ResourceAlert(
                        type="disk_usage",
                        severity="warning",
                        message=f"磁盘使用率过高: {system_metrics.disk_usage:.1f}%",
                        threshold=self.config.alert_thresholds["disk_usage"],
                        current_value=system_metrics.disk_usage,
                        timestamp=datetime.now(),
                        resource_id="system_disk"
                    ))

                # 检查GPU资源告警
                for gpu_metric in gpu_metrics:
                    # GPU显存使用率告警
                    if gpu_metric.memory_usage > self.config.alert_thresholds["gpu_memory_usage"]:
                        new_alerts.append(ResourceAlert(
                            type="gpu_memory_usage",
                            severity="warning",
                            message=f"GPU {gpu_metric.device_id} 显存使用率过高: {gpu_metric.memory_usage:.1f}%",
                            threshold=self.config.alert_thresholds["gpu_memory_usage"],
                            current_value=gpu_metric.memory_usage,
                            timestamp=datetime.now(),
                            resource_id=f"gpu_{gpu_metric.device_id}_memory"
                        ))

                    # GPU利用率告警
                    if gpu_metric.utilization > self.config.alert_thresholds["gpu_utilization"]:
                        new_alerts.append(ResourceAlert(
                            type="gpu_utilization",
                            severity="warning",
                            message=f"GPU {gpu_metric.device_id} 利用率过高: {gpu_metric.utilization:.1f}%",
                            threshold=self.config.alert_thresholds["gpu_utilization"],
                            current_value=gpu_metric.utilization,
                            timestamp=datetime.now(),
                            resource_id=f"gpu_{gpu_metric.device_id}_utilization"
                        ))

                    # GPU温度告警
                    if gpu_metric.temperature > self.config.alert_thresholds["gpu_temperature"]:
                        new_alerts.append(ResourceAlert(
                            type="gpu_temperature",
                            severity="critical",
                            message=f"GPU {gpu_metric.device_id} 温度过高: {gpu_metric.temperature:.1f}°C",
                            threshold=self.config.alert_thresholds["gpu_temperature"],
                            current_value=gpu_metric.temperature,
                            timestamp=datetime.now(),
                            resource_id=f"gpu_{gpu_metric.device_id}_temperature"
                        ))

                # 更新告警列表
                self.alerts.extend(new_alerts)

                # 清理过期告警(保留最近24小时)
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.alerts = [
                    a for a in self.alerts
                    if a.timestamp >= cutoff_time
                ]

                return {
                    "alerts": self.alerts,
                    "count": len(self.alerts),
                    "new_alerts": len(new_alerts)
                }

            except Exception as e:
                logger.error(f"获取资源告警失败: {str(e)}")
                raise HTTPException(status_code=500, detail=f"获取资源告警失败: {str(e)}")

        @router.get("/history / system")
        async def get_system_history(hours: int = 24):
            """获取系统资源历史数据"""
            try:
                cutoff_time = datetime.now() - timedelta(hours=hours)
                history = [
                    m for m in self.system_metrics_history
                    if m.timestamp >= cutoff_time
                ]
                return {
                    "metrics": history,
                    "count": len(history),
                    "time_range": f"最近{hours}小时"
                }

            except Exception as e:
                logger.error(f"获取系统资源历史数据失败: {str(e)}")
                raise HTTPException(status_code=500, detail=f"获取系统资源历史数据失败: {str(e)}")

        @router.get("/history / gpu")
        async def get_gpu_history(hours: int = 24):
            """获取GPU资源历史数据"""
            try:
                cutoff_time = datetime.now() - timedelta(hours=hours)
                history = [
                    m for m in self.gpu_metrics_history
                    if m.timestamp >= cutoff_time
                ]
                return {
                    "metrics": history,
                    "count": len(history),
                    "time_range": f"最近{hours}小时"
                }

            except Exception as e:
                logger.error(f"获取GPU资源历史数据失败: {str(e)}")
                raise HTTPException(status_code=500, detail=f"获取GPU资源历史数据失败: {str(e)}")

        @router.get("/config")
        async def get_module_config():
            """获取模块配置"""
            return {
                "refresh_interval": self.config.refresh_interval,
                "alert_thresholds": self.config.alert_thresholds,
                "enable_alerts": self.config.enable_alerts,
                "enable_history": self.config.enable_history,
                "history_hours": self.config.history_hours,
                "monitor_gpu": self.config.monitor_gpu,
                "monitor_network": self.config.monitor_network
            }

        @router.put("/config")
        async def update_module_config(config_update: Dict[str, Any]):
            """更新模块配置"""
            try:
                # 更新配置
                for key, value in config_update.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)

                logger.info(f"资源监控模块配置已更新: {config_update}")
                return {"message": "配置更新成功", "config": self.config.dict()}

            except Exception as e:
                logger.error(f"更新资源监控模块配置失败: {str(e)}")
                raise HTTPException(status_code=500, detail=f"更新配置失败: {str(e)}")

        self.router = router

    def get_module_data(self) -> ModuleData:
        """获取模块数据"""
        try:
            # 获取当前系统指标
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # 获取GPU指标（如果启用）
            gpu_metrics = []
            if self.config.monitor_gpu:
                try:
                    gpu_devices = self.gpu_manager.get_gpu_devices()
                    for device in gpu_devices:
                        gpu_metrics.append({
                            "device_id": device.get('id', 'unknown'),
                            "memory_used": device.get('memory_used', 0.0),
                            "memory_total": device.get('memory_total', 0.0),
                            "memory_usage": device.get('memory_usage', 0.0),
                            "utilization": device.get('utilization', 0.0),
                            "temperature": device.get('temperature', 0.0),
                            "power_consumption": device.get('power_consumption', 0.0)
                        })
                except Exception as e:
                    logger.warning(f"获取GPU指标失败: {str(e)}")

            return ModuleData(
                module_id=self.config.name,
                data_type="resource_metrics",
                content={
                    "system_metrics": {
                        "cpu_usage": cpu_usage,
                        "memory_usage": memory.percent,
                        "memory_available": memory.available,
                        "memory_total": memory.total,
                        "disk_usage": (disk.used / disk.total) * 100,
                        "disk_used": disk.used,
                        "disk_total": disk.total,
                        "disk_free": disk.free
                    },
                    "gpu_metrics": gpu_metrics,
                    "alerts_count": len(self.alerts),
                    "system_history_count": len(self.system_metrics_history),
                    "gpu_history_count": len(self.gpu_metrics_history)
                },
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"获取资源监控模块数据失败: {str(e)}")
            return ModuleData(
                module_id=self.config.name,
                data_type="resource_metrics",
                content={"error": str(e)},
                timestamp=datetime.now()
            )

    def get_module_status(self) -> ModuleStatus:
        """获取模块状态"""
        try:
            # 检查系统资源状态
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # 检查告警数量
            critical_alerts = len([a for a in self.alerts if a.severity == 'critical'])
            warning_alerts = len([a for a in self.alerts if a.severity == 'warning'])

            # 确定整体状态
            if critical_alerts > 0:
                overall_status = "critical"
            elif warning_alerts > 0:
                overall_status = "warning"
            elif (cpu_usage < 80 and memory.percent < 80
                  and (disk.used / disk.total) * 100 < 80):
                overall_status = "healthy"
            else:
                overall_status = "warning"

            return ModuleStatus(
                module_id=self.config.name,
                status=overall_status,
                is_healthy=overall_status in ["healthy", "warning"],
                last_activity=datetime.now(),
                error_count=critical_alerts,
                uptime=0.0,  # 需要计算实际运行时间
                performance_metrics={
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory.percent,
                    "disk_usage": (disk.used / disk.total) * 100,
                    "critical_alerts": critical_alerts,
                    "warning_alerts": warning_alerts,
                    "total_alerts": len(self.alerts),
                    "system_history_records": len(self.system_metrics_history),
                    "gpu_history_records": len(self.gpu_metrics_history),
                    "gpu_enabled": self.config.monitor_gpu
                },
                alerts=[{"type": a.type, "severity": a.severity, "message": a.message}
                        for a in self.alerts]
            )

        except Exception as e:
            logger.error(f"获取资源监控模块状态失败: {str(e)}")
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

    def validate_permissions(self, user: str, action: str) -> bool:
        """权限验证"""
        # 基础权限检查
        if action == "read":
            return True  # 所有用户都可以读取资源监控数据

        if action == "write":
            # 只有管理员和操作员可以修改配置
            return user in ["admin", "operator"]

        if action == "manage":
            # 只有管理员可以管理资源监控模块
            return user == "admin"

        return False

    async def _initialize_module(self) -> bool:
        """初始化模块"""
        try:
            # 初始化资源管理器
            self.resource_manager = ResourceManager()
            self.gpu_manager = GPUManager()
            self.quota_manager = QuotaManager()

            # 检查系统资源访问权限
            psutil.cpu_percent(interval=0.1)  # 测试CPU访问
            psutil.virtual_memory()  # 测试内存访问
            psutil.disk_usage('/')  # 测试磁盘访问

            logger.info("资源监控模块初始化成功")
            return True

        except Exception as e:
            logger.error(f"资源监控模块初始化失败: {str(e)}")
            return False

    async def _start_module(self) -> bool:
        """启动模块"""
        try:
            # 启动监控任务
            asyncio.create_task(self._monitor_loop())
            logger.info("资源监控模块启动成功")
            return True

        except Exception as e:
            logger.error(f"资源监控模块启动失败: {str(e)}")
            return False

    async def _stop_module(self) -> bool:
        """停止模块"""
        try:
            # 清理资源
            self.system_metrics_history.clear()
            self.gpu_metrics_history.clear()
            self.alerts.clear()
            logger.info("资源监控模块停止成功")
            return True

        except Exception as e:
            logger.error(f"资源监控模块停止失败: {str(e)}")
            return False

    async def _monitor_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                # 获取系统指标
                await self.get_module_data()

                # 获取GPU指标
                if self.config.monitor_gpu:
                    await self.get_gpu_metrics()

                # 检查告警
                if self.config.enable_alerts:
                    await self.get_alerts()

                # 等待下次刷新
                await asyncio.sleep(self.config.refresh_interval)

            except Exception as e:
                logger.error(f"资源监控循环错误: {str(e)}")
                await asyncio.sleep(5)  # 错误时等待5秒后重试


def create_resource_module(config: Optional[ResourceModuleConfig] = None) -> ResourceModule:
    """创建资源监控模块实例"""
    if config is None:
        config = ResourceModuleConfig()
    return ResourceModule(config)
