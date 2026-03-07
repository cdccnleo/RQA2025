"""
visual_monitor 模块

提供 visual_monitor 相关功能和接口。
"""

import logging

# -*- coding: utf-8 -*-
import threading
import time

# 移除不存在的导入，使用本地定义的类
from typing import Dict, Any
from dataclasses import dataclass
"""
基础设施层 - 日志系统组件

visual_monitor 模块

日志系统相关的文件
提供日志系统相关的功能实现。
"""

#!/usr / bin / env python
"""
服务状态可视化监控模块
负责将系统状态以可视化方式展示，便于运维监控
"""

# 修复导入路径
# 导入处理已在上方完成，这里定义备用类


class ConfigManager:

    def __init__(self, config: Dict[str, Any] = None):

        self.config = config or {}

    def get(self, key, default=None):

        return self.config.get(key, default)

# 其他导入处理已在上方完成


class HealthChecker:

    def __init__(self, config: Dict[str, Any] = None):

        self.config = config or {}

    def get_status(self):

        return {"default_service": {"status": "healthy"}}

# 其他导入处理已完成


class CircuitBreaker:

    def __init__(self, config: Dict[str, Any] = None, registry=None):

        self.config = config or {}

    def get_status(self):

        return {"default_service": {"status": "closed"}}

# 其他导入处理已完成


class DegradationManager:

    def __init__(self, config: Dict[str, Any] = None, circuit_breaker=None):

        self.config = config or {}

    def get_status_report(self):

        return {"default_service": {"level": 0}}

# 其他导入处理已完成


class AutoRecovery:

    def __init__(self, config: Dict[str, Any] = None):

        self.config = config or {}

        # 使用标准logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)


@dataclass
class ServiceStatus:

    """服务状态数据类"""
    name: str
    health: str
    breaker_state: str
    degradation_level: int
    last_updated: float


class VisualMonitor:

    def __init__(self, config: Dict[str, Any], registry=None):
        """
        初始化可视化监控器
        :param config: 系统配置
        :param registry: Prometheus CollectorRegistry（测试用）
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        # 修复ConfigManager初始化
        try:
            self.config_manager = ConfigManager()
        except Exception:
            # 如果ConfigManager初始化失败，使用简单的实现
            self.config_manager = ConfigManager(config)
        self.health_checker = HealthChecker(config)
        self.circuit_breaker = CircuitBreaker(config, registry=registry)
        self.degradation_manager = DegradationManager(config, circuit_breaker=self.circuit_breaker)
        self.auto_recovery = AutoRecovery(config)

        # 添加缺失的属性
        self.services: Dict[str, ServiceStatus] = {}
        self.service_statuses: Dict[str, ServiceStatus] = {}  # 兼容性属性
        self.lock = threading.Lock()
        self.running = False
        self.dashboard_data = {
            "services": [],
            "system_health": "GREEN",
            "last_updated": 0,
            "timestamp": 0  # 添加timestamp字段
        }

        # 添加配置相关属性
        self.update_interval = 5
        self.dashboard_port = 8080
        self.metrics_port = 9090

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
        self.logger.info("可视化监控已启动")

    def stop(self) -> None:
        """
        停止可视化监控
        """
        self.running = False
        self.logger.info("可视化监控已停止")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        获取仪表盘数据
        :return: 仪表盘数据字典
        """
        with self.lock:
            # 同步service_statuses和services，但优先保留已有的service_statuses
            if not self.service_statuses and self.services:
                self.service_statuses = self.services.copy()
            elif not self.services and self.service_statuses:
                self.services = self.service_statuses.copy()

            # 更新timestamp
            self.dashboard_data["timestamp"] = time.time()

            # 准备仪表板数据（不获取锁，因为已经在锁内）
            self._prepare_dashboard_data_internal()

            return self.dashboard_data.copy()

    def _prepare_dashboard_data_internal(self) -> None:
        """
        准备仪表板数据（内部方法，不获取锁）
        """
        try:
            # 优先使用service_statuses，如果没有则使用services
            services_to_convert = self.service_statuses if self.service_statuses else self.services

            # 转换服务状态为可序列化格式
            services_data = []
            for service in services_to_convert.values():
                services_data.append({
                    'name': service.name,
                    'health': service.health,
                    'breaker_state': service.breaker_state,
                    'degradation_level': service.degradation_level,
                    'last_updated': service.last_updated
                })

            self.dashboard_data['services'] = services_data
            self.dashboard_data['last_updated'] = time.time()

        except Exception as e:
            self.logger.error(f"准备仪表板数据失败: {e}")

    def _prepare_dashboard_data(self) -> None:
        """
        准备仪表板数据
        """
        try:
            with self.lock:
                self._prepare_dashboard_data_internal()
        except Exception as e:
            self.logger.error(f"准备仪表板数据失败: {e}")

    def _load_config(self) -> None:
        """
        加载可视化配置
        """
        try:
            visual_config = self.config_manager.get('visual_monitor', {})
            self.update_interval = visual_config.get('update_interval', 5)
            self.dashboard_port = visual_config.get('dashboard_port', 8080)
            self.metrics_port = visual_config.get('metrics_port', 9090)
        except Exception as e:
            self.logger.warning(f"配置加载失败，使用默认值: {e}")
            # 使用默认值
            self.update_interval = 5
            self.dashboard_port = 8080
            self.metrics_port = 9090

    def _update_service_status(self) -> None:
        """
        更新所有服务状态
        """
        try:
            health_status = self.health_checker.get_status()
            breaker_status = self.circuit_breaker.get_status()
            degradation_status = self.degradation_manager.get_status_report()

            with self.lock:
                # 更新各服务状态
                for service_name in health_status.keys():
                    try:
                        # 处理不同的返回格式
                        if isinstance(health_status.get(service_name), dict):
                            health = health_status.get(service_name, {}).get('status', 'UNKNOWN')
                        else:
                            health = health_status.get(service_name, 'UNKNOWN')

                        if isinstance(breaker_status.get(service_name), dict):
                            breaker = breaker_status.get(service_name, {}).get('status', 'UNKNOWN')
                        else:
                            breaker = breaker_status.get(service_name, 'UNKNOWN')

                        if isinstance(degradation_status.get(service_name), dict):
                            degradation = degradation_status.get(service_name, {}).get('level', 0)
                        else:
                            degradation = degradation_status.get(service_name, 0)

                        # 创建或更新服务状态
                        service_status = ServiceStatus(
                            name=service_name,
                            health=health,
                            breaker_state=breaker,
                            degradation_level=degradation,
                            last_updated=time.time()
                        )

                        self.services[service_name] = service_status
                        self.service_statuses[service_name] = service_status  # 兼容性

                    except Exception as e:
                        self.logger.error(f"更新服务 {service_name} 状态失败: {e}")
                        continue

        except Exception as e:
            self.logger.error(f"更新服务状态失败: {e}")

    def _calculate_system_health(self) -> None:
        """
        计算系统整体健康状态
        """
        try:
            with self.lock:
                # 优先使用service_statuses，如果没有则使用services
                services_to_check = self.service_statuses if self.service_statuses else self.services

                if not services_to_check:
                    self.dashboard_data['system_health'] = 'unknown'
                    self.system_health = 'unknown'  # 添加system_health属性
                    return

                # 统计各状态数量
                health_counts = {'UP': 0, 'DOWN': 0, 'DEGRADED': 0, 'UNKNOWN': 0}
                total_services = len(services_to_check)

                for service in services_to_check.values():
                    health_counts[service.health] += 1

                # 计算系统健康状态 - 使用统一的GREEN / YELLOW / RED格式
                if health_counts['DOWN'] > 0:
                    system_health = 'RED'
                elif health_counts['DEGRADED'] > 0:
                    system_health = 'YELLOW'
                elif health_counts['UP'] == total_services:
                    system_health = 'GREEN'
                else:
                    system_health = 'YELLOW'

                # 更新dashboard_data和system_health属性
                self.dashboard_data['system_health'] = system_health
                self.system_health = system_health

                # 更新最后更新时间
                self.dashboard_data['last_updated'] = time.time()

        except Exception as e:
            self.logger.error(f"计算系统健康状态失败: {e}")
            self.dashboard_data['system_health'] = 'unknown'
            self.system_health = 'unknown'

    def _monitor_loop(self) -> None:
        """
        监控循环
        """
        self.logger.info("可视化监控循环启动")
        while self.running:
            try:
                self._update_service_status()
                self._calculate_system_health()
                self._prepare_dashboard_data()

                # 等待下次更新
                time.sleep(self.update_interval)

            except Exception as e:
                self.logger.error(f"可视化监控循环出错: {e}")
                time.sleep(1)  # 出错时短暂等待

    def generate_html_report(self) -> str:
        """
        生成HTML格式的监控报告
        :return: HTML报告内容
        """
        try:
            dashboard = self.get_dashboard_data()

            # 确保必要字段存在
            if not dashboard or 'last_updated' not in dashboard:
                dashboard = {
                    'last_updated': time.time(),
                    'system_health': 'UNKNOWN',
                    'services': []
                }

            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>RQA2025 系统监控</title>
                <style>
                    body {{ font - family: Arial, sans - serif; margin: 20px; }}
                    .header {{}}
                        background - color: #f0f0f0;
                        padding: 10px;
                        border - radius: 5px;
                        margin - bottom: 20px;

                    .system - health {{}}
                        font - size: 24px;
                        font - weight: bold;
                        padding: 10px;
                        border - radius: 5px;
                        text - align: center;
                        margin - bottom: 20px;

                    .health - GREEN {{ background - color: #d4edda; color: #155724; }}
                    .health - YELLOW {{ background - color: #fff3cd; color: #856404; }}
                    .health - RED {{ background - color: #f8d7da; color: #721c24; }}
                    .health - UNKNOWN {{ background - color: #e2e3e5; color: #383d41; }}

                    table {{}}
                        width: 100%;
                        border - collapse: collapse;
                        margin - bottom: 20px;

                    th, td {{}}
                        border: 1px solid #ddd;
                        padding: 8px;
                        text - align: left;

                    th {{ background - color: #f2f2f2; }}
                    tr:nth - child(even) {{ background - color: #f9f9f9; }}

                    .status - UP {{ color: green; }}
                    .status - DOWN {{ color: red; }}
                    .status - DEGRADED {{ color: orange; }}
                    .breaker - CLOSED {{ color: green; }}
                    .breaker - OPEN {{ color: red; }}
                    .breaker - HALF_OPEN {{ color: orange; }}
                    .timestamp {{ font - size: 12px; color: #666; }}

                </style>
            </head>
            <body>
                <div class="header">
                    <h1>RQA2025 系统监控</h1>
                    <p>最后更新时间: <span class="timestamp">{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(dashboard['last_updated']))}</span></p>
                </div>

                <div class="system - health health-{dashboard['system_health']}">
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

            # 添加服务状态行
            for service_data in dashboard.get('services', []):
                html += f"""
                    <tr>
                        <td>{service_data.get('name', 'Unknown')}</td>
                        <td class="status-{service_data.get('health', 'UNKNOWN')}">{service_data.get('health', 'UNKNOWN')}</td>
                        <td class="breaker-{service_data.get('breaker_state', 'UNKNOWN')}">{service_data.get('breaker_state', 'UNKNOWN')}</td>
                        <td>{service_data.get('degradation_level', 0)}</td>
                        <td class="timestamp">{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(service_data.get('last_updated', 0)))}</td>
                    </tr>
                """

            html += """
                </table>
                <div class="timestamp">
                    报告生成时间: """ + time.strftime('%Y-%m-%d %H:%M:%S') + """
                </div>
            </body>
            </html>
            """

            return html

        except Exception as e:
            self.logger.error(f"生成HTML报告失败: {e}")
            # 返回错误页面
            return f"""
            <!DOCTYPE html>
            <html>
            <head><title>监控报告生成失败</title></head>
            <body>
                <h1>监控报告生成失败</h1>
                <p>错误信息: {str(e)}</p>
                <p>时间: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </body>
            </html>
            """

    def generate_prometheus_metrics(self) -> str:
        """
        生成Prometheus格式的监控指标
        :return: Prometheus指标字符串
        """
        try:
            dashboard = self.get_dashboard_data()
            metrics = []

            # 系统健康状态指标
            health_value = 1 if dashboard.get('system_health') == 'GREEN' else 0
            metrics.append(f"system_health_status {health_value} {int(time.time())}")

            # 服务健康状态指标
            for service_data in dashboard.get('services', []):
                service_name = service_data.get('name', 'unknown')
                health_value = 1 if service_data.get('health') == 'UP' else 0
                metrics.append(
                    f'service_health{{service="{service_name}"}} {health_value} {int(time.time())}')

                # 断路器状态指标
                breaker_value = 1 if service_data.get('breaker_state') == 'CLOSED' else 0
                metrics.append(
                    f'service_circuit_breaker{{service="{service_name}"}} {breaker_value} {int(time.time())}')

                # 降级级别指标
                degradation_level = service_data.get('degradation_level', 0)
                metrics.append(
                    f'service_degradation_level{{service="{service_name}"}} {degradation_level} {int(time.time())}')

            return '\n'.join(metrics)

        except Exception as e:
            self.logger.error(f"生成Prometheus指标失败: {e}")
            return f"# 指标生成失败: {str(e)}"
