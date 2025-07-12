#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
服务状态可视化监控模块
负责将系统状态以可视化方式展示，便于运维监控
"""

import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from src.infrastructure.utils.logger import get_logger
from src.infrastructure.config.config_manager import ConfigManager
from src.infrastructure.health.health_checker import HealthChecker
from src.infrastructure.circuit_breaker import CircuitBreaker
from src.infrastructure.degradation_manager import DegradationManager
from src.infrastructure.auto_recovery import AutoRecovery

logger = get_logger(__name__)

@dataclass
class ServiceStatus:
    """服务状态"""
    name: str
    health: str  # UP, DOWN, DEGRADED
    breaker_state: str  # CLOSED, OPEN, HALF_OPEN
    degradation_level: int  # 0-5
    last_updated: float

class VisualMonitor:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化可视化监控器
        :param config: 系统配置
        """
        self.config = config
        self.config_manager = ConfigManager(config)
        self.health_checker = HealthChecker(config)
        self.circuit_breaker = CircuitBreaker(config)
        self.degradation_manager = DegradationManager(config)
        self.auto_recovery = AutoRecovery(config)
        self.services: Dict[str, ServiceStatus] = {}
        self.lock = threading.Lock()
        self.running = False
        self.dashboard_data = {
            "services": [],
            "system_health": "GREEN",
            "last_updated": 0
        }

        # 加载可视化配置
        self._load_config()

    def start(self) -> None:
        """
        启动可视化监控
        """
        if self.running:
            return

        self.running = True
        monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        monitor_thread.start()
        logger.info("可视化监控已启动")

    def stop(self) -> None:
        """
        停止可视化监控
        """
        self.running = False
        logger.info("可视化监控已停止")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        获取仪表盘数据
        :return: 仪表盘数据字典
        """
        with self.lock:
            return self.dashboard_data.copy()

    def _load_config(self) -> None:
        """加载可视化配置"""
        visual_config = self.config_manager.get('visual_monitor', {})
        self.refresh_interval = visual_config.get('refresh_interval', 5)

    def _update_service_status(self) -> None:
        """
        更新所有服务状态
        """
        health_status = self.health_checker.get_status()
        breaker_status = self.circuit_breaker.get_status()
        degradation_status = self.degradation_manager.get_status_report()

        with self.lock:
            # 更新各服务状态
            for service_name in health_status.keys():
                health = health_status.get(service_name, {}).get('status', 'UNKNOWN')
                breaker = breaker_status.get(service_name, {}).get('state', 'CLOSED')
                degrade = degradation_status.get('services', {}).get(service_name, {}).get('current_level', 0)

                self.services[service_name] = ServiceStatus(
                    name=service_name,
                    health=health,
                    breaker_state=breaker,
                    degradation_level=degrade,
                    last_updated=time.time()
                )

            # 计算系统整体健康状态
            self._calculate_system_health()

            # 准备仪表盘数据
            self._prepare_dashboard_data()

    def _calculate_system_health(self) -> None:
        """
        计算系统整体健康状态
        """
        down_services = 0
        degraded_services = 0
        open_breakers = 0

        for status in self.services.values():
            if status.health == 'DOWN':
                down_services += 1
            elif status.health == 'DEGRADED':
                degraded_services += 1

            if status.breaker_state == 'OPEN':
                open_breakers += 1

        if down_services > 0:
            system_health = "RED"
        elif degraded_services > 2 or open_breakers > 0:
            system_health = "YELLOW"
        else:
            system_health = "GREEN"

        self.dashboard_data['system_health'] = system_health

    def _prepare_dashboard_data(self) -> None:
        """
        准备仪表盘展示数据
        """
        dashboard_services = []

        for name, status in self.services.items():
            dashboard_services.append({
                "name": name,
                "health": status.health,
                "breaker_state": status.breaker_state,
                "degradation_level": status.degradation_level,
                "last_updated": status.last_updated
            })

        self.dashboard_data.update({
            "services": dashboard_services,
            "last_updated": time.time()
        })

    def _monitor_loop(self) -> None:
        """
        可视化监控循环
        """
        logger.info("可视化监控循环启动")
        while self.running:
            try:
                self._update_service_status()
                time.sleep(self.refresh_interval)
            except Exception as e:
                logger.error(f"可视化监控循环出错: {str(e)}")
                time.sleep(30)

    def generate_html_report(self) -> str:
        """
        生成HTML格式的监控报告
        :return: HTML报告内容
        """
        dashboard = self.get_dashboard_data()

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RQA2025 系统监控</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ 
                    background-color: #f0f0f0; 
                    padding: 10px; 
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                .system-health {{ 
                    font-size: 24px; 
                    font-weight: bold;
                    padding: 10px;
                    border-radius: 5px;
                    text-align: center;
                    margin-bottom: 20px;
                }}
                .health-GREEN {{ background-color: #d4edda; color: #155724; }}
                .health-YELLOW {{ background-color: #fff3cd; color: #856404; }}
                .health-RED {{ background-color: #f8d7da; color: #721c24; }}
                table {{ 
                    width: 100%; 
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                th, td {{ 
                    border: 1px solid #ddd; 
                    padding: 8px; 
                    text-align: left;
                }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .status-UP {{ color: green; }}
                .status-DOWN {{ color: red; }}
                .status-DEGRADED {{ color: orange; }}
                .breaker-CLOSED {{ color: green; }}
                .breaker-OPEN {{ color: red; }}
                .breaker-HALF_OPEN {{ color: orange; }}
                .timestamp {{ font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>RQA2025 系统监控</h1>
                <p>最后更新时间: <span class="timestamp">{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(dashboard['last_updated']))}</span></p>
            </div>
            
            <div class="system-health health-{dashboard['system_health']}">
                系统状态: {dashboard['system_health']}
            </div>
            
            <h2>服务状态</h2>
            <table>
                <tr>
                    <th>服务名称</th>
                    <th>健康状态</th>
                    <th>熔断状态</th>
                    <th>降级级别</th>
                    <th>最后更新</th>
                </tr>
        """

        for service in dashboard['services']:
            html += f"""
                <tr>
                    <td>{service['name']}</td>
                    <td class="status-{service['health']}">{service['health']}</td>
                    <td class="breaker-{service['breaker_state']}">{service['breaker_state']}</td>
                    <td>{service['degradation_level']}</td>
                    <td>{time.strftime('%H:%M:%S', time.localtime(service['last_updated']))}</td>
                </tr>
            """

        html += """
            </table>
        </body>
        </html>
        """

        return html

    def generate_prometheus_metrics(self) -> str:
        """
        生成Prometheus格式的监控指标
        :return: Prometheus指标内容
        """
        dashboard = self.get_dashboard_data()
        metrics = []
        timestamp = int(dashboard['last_updated'] * 1000)

        # 系统健康指标
        health_value = 1 if dashboard['system_health'] == 'GREEN' else (
            0.5 if dashboard['system_health'] == 'YELLOW' else 0
        )
        metrics.append(f"system_health_status {health_value} {timestamp}")

        # 各服务指标
        for service in dashboard['services']:
            # 健康状态
            health_value = 1 if service['health'] == 'UP' else (
                0.5 if service['health'] == 'DEGRADED' else 0
            )
            metrics.append(f'service_health_status{{name="{service["name"]}"}} {health_value} {timestamp}')

            # 熔断状态
            breaker_value = 1 if service['breaker_state'] == 'CLOSED' else (
                0.5 if service['breaker_state'] == 'HALF_OPEN' else 0
            )
            metrics.append(f'service_breaker_status{{name="{service["name"]}"}} {breaker_value} {timestamp}')

            # 降级级别
            metrics.append(f'service_degradation_level{{name="{service["name"]}"}} {service["degradation_level"]} {timestamp}')

        return "\n".join(metrics)
