import logging
#!/usr / bin / env python
# -*- coding: utf-8 -*-

"""
RQA2025 统一Web管理界面 - FPGA监控模块
整合FPGA性能监控功能到统一管理平台
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .base_module import BaseModule, ModuleConfig, ModuleData, ModuleStatus  # 当前层级内部导入：基础模块类
from src.acceleration.fpga.fpga_performance_monitor import FPGAPerformanceMonitor  # 合理跨层级导入：加速层FPGA性能监控
from src.acceleration.fpga.fpga_manager import FPGAManager  # 合理跨层级导入：加速层FPGA管理器
from src.engine.logging.unified_logger import get_unified_logger  # 当前层级内部导入：统一日志器

logger = logging.getLogger(__name__)


class FPGAMetrics(BaseModel):

    """FPGA性能指标模型"""
    latency: float
    utilization: float
    temperature: float
    power_consumption: float
    throughput: float
    error_count: int
    timestamp: datetime


class FPGAAlert(BaseModel):

    """FPGA告警模型"""
    type: str
    severity: str
    message: str
    timestamp: datetime
    device_id: str


class FPGAModuleConfig(ModuleConfig):

    """FPGA模块配置"""
    refresh_interval: int = 5  # 刷新间隔(秒)
    alert_thresholds: Dict[str, float] = {
        "latency": 1.0,  # 延迟阈值(ms)
        "utilization": 90.0,  # 利用率阈值(%)
        "temperature": 85.0,  # 温度阈值(°C)
        "power": 200.0  # 功耗阈值(W)
    }
    enable_alerts: bool = True
    enable_history: bool = True
    history_hours: int = 24


class FPGAModule(BaseModule):

    """FPGA监控模块"""

    def __init__(self, config: FPGAModuleConfig):

        super().__init__(config)
        self.fpga_manager = FPGAManager()  # 移除设备ID参数
        self.fpga_monitor = FPGAPerformanceMonitor(fpga_manager=self.fpga_manager)
        self.metrics_history: List[FPGAMetrics] = []
        self.alerts: List[FPGAAlert] = []
        self._setup_routes()

    def _setup_routes(self):
        """设置路由"""
        router = APIRouter(prefix="/fpga", tags=["FPGA监控"])

        @router.get("/metrics / current")
        async def get_current_metrics():
            """获取当前FPGA性能指标"""
            try:
                status = self.fpga_manager.get_device_status()
                report = self.fpga_monitor.generate_report()

                metrics = FPGAMetrics(
                    latency=report.get('latency_stats', {}).get('current', 0.0),
                    utilization=report.get('utilization_stats', {}).get('current', 0.0),
                    temperature=status.get('temperature', 0.0),
                    power_consumption=status.get('power_consumption', 0.0),
                    throughput=report.get('throughput', 0.0),
                    error_count=report.get('error_count', 0),
                    timestamp=datetime.now()
                )

                # 保存到历史记录
                if self.config.enable_history:
                    self.metrics_history.append(metrics)
                    # 保持历史记录在指定时间范围内
                    cutoff_time = datetime.now() - timedelta(hours=self.config.history_hours)
                    self.metrics_history = [
                        m for m in self.metrics_history
                        if m.timestamp >= cutoff_time
                    ]

                return metrics

            except Exception as e:
                logger.error(f"获取FPGA指标失败: {str(e)}")
                raise HTTPException(status_code=500, detail=f"获取FPGA指标失败: {str(e)}")

        @router.get("/metrics / history")
        async def get_metrics_history(hours: int = 24):
            """获取历史性能数据"""
            try:
                cutoff_time = datetime.now() - timedelta(hours=hours)
                history = [
                    m for m in self.metrics_history
                    if m.timestamp >= cutoff_time
                ]
                return {
                    "metrics": history,
                    "count": len(history),
                    "time_range": f"最近{hours}小时"
                }

            except Exception as e:
                logger.error(f"获取FPGA历史数据失败: {str(e)}")
                raise HTTPException(status_code=500, detail=f"获取FPGA历史数据失败: {str(e)}")

        @router.get("/alerts")
        async def get_alerts():
            """获取FPGA告警信息"""
            try:
                if not self.config.enable_alerts:
                    return {"alerts": [], "count": 0}

                # 获取当前指标
                current_metrics = await get_current_metrics()

                # 检查告警条件
                new_alerts = []

                # 延迟告警
                if current_metrics.latency > self.config.alert_thresholds["latency"]:
                    new_alerts.append(FPGAAlert(
                        type="latency",
                        severity="warning",
                        message=f"FPGA延迟过高: {current_metrics.latency:.2f}ms",
                        timestamp=datetime.now(),
                        device_id="fpga_0"
                    ))

                # 利用率告警
                if current_metrics.utilization > self.config.alert_thresholds["utilization"]:
                    new_alerts.append(FPGAAlert(
                        type="utilization",
                        severity="warning",
                        message=f"FPGA利用率过高: {current_metrics.utilization:.1f}%",
                        timestamp=datetime.now(),
                        device_id="fpga_0"
                    ))

                # 温度告警
                if current_metrics.temperature > self.config.alert_thresholds["temperature"]:
                    new_alerts.append(FPGAAlert(
                        type="temperature",
                        severity="critical",
                        message=f"FPGA温度过高: {current_metrics.temperature:.1f}°C",
                        timestamp=datetime.now(),
                        device_id="fpga_0"
                    ))

                # 功耗告警
                if current_metrics.power_consumption > self.config.alert_thresholds["power"]:
                    new_alerts.append(FPGAAlert(
                        type="power",
                        severity="warning",
                        message=f"FPGA功耗过高: {current_metrics.power_consumption:.1f}W",
                        timestamp=datetime.now(),
                        device_id="fpga_0"
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
                logger.error(f"获取FPGA告警失败: {str(e)}")
                raise HTTPException(status_code=500, detail=f"获取FPGA告警失败: {str(e)}")

        @router.get("/status")
        async def get_device_status():
            """获取FPGA设备状态"""
            try:
                status = self.fpga_manager.get_device_status()
                return {
                    "device_id": "fpga_0",
                    "status": status.get("status", "unknown"),
                    "temperature": status.get("temperature", 0.0),
                    "power_consumption": status.get("power_consumption", 0.0),
                    "memory_usage": status.get("memory_usage", 0.0),
                    "last_updated": datetime.now()
                }

            except Exception as e:
                logger.error(f"获取FPGA设备状态失败: {str(e)}")
                raise HTTPException(status_code=500, detail=f"获取FPGA设备状态失败: {str(e)}")

        @router.get("/config")
        async def get_module_config():
            """获取模块配置"""
            return {
                "refresh_interval": self.config.refresh_interval,
                "alert_thresholds": self.config.alert_thresholds,
                "enable_alerts": self.config.enable_alerts,
                "enable_history": self.config.enable_history,
                "history_hours": self.config.history_hours
            }

        @router.put("/config")
        async def update_module_config(config_update: Dict[str, Any]):
            """更新模块配置"""
            try:
                # 更新配置
                for key, value in config_update.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)

                logger.info(f"FPGA模块配置已更新: {config_update}")
                return {"message": "配置更新成功", "config": self.config.dict()}

            except Exception as e:
                logger.error(f"更新FPGA模块配置失败: {str(e)}")
                raise HTTPException(status_code=500, detail=f"更新配置失败: {str(e)}")

        self.router = router

    def get_module_data(self) -> ModuleData:
        """获取模块数据"""
        try:
            # 获取当前FPGA指标
            metrics = self.fpga_monitor.get_current_metrics()
            status = self.fpga_manager.get_device_status()
            report = self.fpga_monitor.get_performance_report()

            return ModuleData(
                module_id=self.config.name,
                data_type="fpga_metrics",
                content={
                    "current_metrics": {
                        "latency": metrics.get('latency', 0.0),
                        "utilization": metrics.get('utilization', 0.0),
                        "temperature": metrics.get('temperature', 0.0),
                        "power_consumption": status.get('power_consumption', 0.0),
                        "throughput": report.get('throughput', 0.0),
                        "error_count": report.get('error_count', 0)
                    },
                    "alerts_count": len(self.alerts),
                    "history_count": len(self.metrics_history),
                    "device_status": status.get('status', 'unknown')
                },
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"获取FPGA模块数据失败: {str(e)}")
            return ModuleData(
                module_id=self.config.name,
                data_type="fpga_metrics",
                content={"error": str(e)},
                timestamp=datetime.now()
            )

    def get_module_status(self) -> ModuleStatus:
        """获取模块状态"""
        try:
            # 检查FPGA设备状态
            status = self.fpga_manager.get_device_status()
            device_status = status.get('status', 'unknown')

            # 检查告警数量
            critical_alerts = len([a for a in self.alerts if a.severity == 'critical'])
            warning_alerts = len([a for a in self.alerts if a.severity == 'warning'])

            # 确定整体状态
            if critical_alerts > 0:
                overall_status = "critical"
            elif warning_alerts > 0:
                overall_status = "warning"
            elif device_status == "healthy":
                overall_status = "healthy"
            else:
                overall_status = "unknown"

            return ModuleStatus(
                module_id=self.config.name,
                status=overall_status,
                is_healthy=overall_status in ["healthy", "warning"],
                last_activity=datetime.now(),
                error_count=len([a for a in self.alerts if a.severity == 'critical']),
                uptime=0.0,  # 需要计算实际运行时间
                performance_metrics={
                    "device_status": device_status,
                    "critical_alerts": critical_alerts,
                    "warning_alerts": warning_alerts,
                    "total_alerts": len(self.alerts),
                    "history_records": len(self.metrics_history)
                },
                alerts=[{"type": a.type, "severity": a.severity, "message": a.message}
                        for a in self.alerts]
            )

        except Exception as e:
            logger.error(f"获取FPGA模块状态失败: {str(e)}")
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
            return True  # 所有用户都可以读取FPGA监控数据

        if action == "write":
            # 只有管理员和操作员可以修改配置
            return user in ["admin", "operator"]

        if action == "manage":
            # 只有管理员可以管理FPGA模块
            return user == "admin"

        return False

    async def _initialize_module(self) -> bool:
        """初始化模块"""
        try:
            # 初始化FPGA监控器
            self.fpga_monitor = FPGAPerformanceMonitor()
            self.fpga_manager = FPGAManager()

            # 检查FPGA设备连接
            status = self.fpga_manager.get_device_status()
            if status.get('status') == 'connected':
                logger.info("FPGA模块初始化成功")
                return True
            else:
                logger.warning("FPGA设备未连接，模块将以离线模式运行")
                return True  # 允许离线模式运行

        except Exception as e:
            logger.error(f"FPGA模块初始化失败: {str(e)}")
            return False

    async def _start_module(self) -> bool:
        """启动模块"""
        try:
            # 启动监控任务
            asyncio.create_task(self._monitor_loop())
            logger.info("FPGA模块启动成功")
            return True

        except Exception as e:
            logger.error(f"FPGA模块启动失败: {str(e)}")
            return False

    async def _stop_module(self) -> bool:
        """停止模块"""
        try:
            # 清理资源
            self.metrics_history.clear()
            self.alerts.clear()
            logger.info("FPGA模块停止成功")
            return True

        except Exception as e:
            logger.error(f"FPGA模块停止失败: {str(e)}")
            return False

    async def _monitor_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                # 获取当前指标
                await self.get_module_data()

                # 检查告警
                if self.config.enable_alerts:
                    await self.get_alerts()

                # 等待下次刷新
                await asyncio.sleep(self.config.refresh_interval)

            except Exception as e:
                logger.error(f"FPGA监控循环错误: {str(e)}")
                await asyncio.sleep(5)  # 错误时等待5秒后重试


def create_fpga_module(config: Optional[FPGAModuleConfig] = None) -> FPGAModule:
    """创建FPGA模块实例"""
    if config is None:
        config = FPGAModuleConfig()
    return FPGAModule(config)
