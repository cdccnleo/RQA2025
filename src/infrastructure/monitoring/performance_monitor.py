#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
性能监控模块
负责收集、分析和报告系统各模块的性能指标
"""

import time
import threading
import psutil
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from src.infrastructure.utils.logger import get_logger
from src.infrastructure.config.config_manager import ConfigManager

logger = get_logger(__name__)

@dataclass
class PerformanceMetric:
    """性能指标数据结构"""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str]

class PerformanceMonitor:
    def __init__(self, config: Dict[str, Any], config_manager: Optional[ConfigManager] = None):
        """
        初始化性能监控器
        :param config: 系统配置
        :param config_manager: 可选的配置管理器实例，用于测试时注入mock对象
        """
        self.config = config
        
        # 测试钩子：允许注入mock的ConfigManager
        if config_manager is not None:
            self.config_manager = config_manager
        else:
            self.config_manager = ConfigManager(config)
            
        self.metrics_buffer = []
        self.metrics_lock = threading.Lock()
        self.running = False
        self.monitor_thread = None

        # 加载监控配置
        self.monitor_config = self.config_manager.get_config('monitoring', {})
        self.interval = self.monitor_config.get('interval', 5)
        self.metric_handlers = {
            'prometheus': self._handle_prometheus,
            'influxdb': self._handle_influxdb,
            'file': self._handle_file
        }
        self.storage_backends = self.monitor_config.get('storage', ['file'])

    def start(self) -> None:
        """
        启动性能监控
        """
        if self.running:
            logger.warning("性能监控已经启动")
            return

        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("性能监控启动成功")

    def stop(self) -> None:
        """
        停止性能监控
        """
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("性能监控已停止")

    def record_metric(self, metric: PerformanceMetric) -> None:
        """
        记录性能指标
        :param metric: 性能指标对象
        """
        with self.metrics_lock:
            self.metrics_buffer.append(metric)

    def _monitor_loop(self) -> None:
        """
        监控主循环
        """
        logger.info(f"性能监控循环启动，间隔: {self.interval}秒")

        while self.running:
            try:
                # 收集系统指标
                self._collect_system_metrics()

                # 处理缓冲指标
                with self.metrics_lock:
                    if self.metrics_buffer:
                        metrics_batch = self.metrics_buffer.copy()
                        self.metrics_buffer.clear()
                    else:
                        metrics_batch = []

                # 存储指标
                if metrics_batch:
                    self._store_metrics(metrics_batch)

                # 等待下一个周期
                time.sleep(self.interval)

            except Exception as e:
                logger.error(f"性能监控循环出错: {str(e)}")
                time.sleep(1)

    def _collect_system_metrics(self) -> None:
        """
        收集系统级性能指标
        """
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=None)
        self.record_metric(PerformanceMetric(
            name="system.cpu.usage",
            value=cpu_percent,
            timestamp=time.time(),
            tags={"host": "localhost", "type": "system"}
        ))

        # 内存使用
        mem = psutil.virtual_memory()
        self.record_metric(PerformanceMetric(
            name="system.memory.usage",
            value=mem.percent,
            timestamp=time.time(),
            tags={"host": "localhost", "type": "system"}
        ))

        # 磁盘IO
        disk_io = psutil.disk_io_counters()
        if disk_io is not None:
            self.record_metric(PerformanceMetric(
                name="system.disk.read_bytes",
                value=disk_io.read_bytes,
                timestamp=time.time(),
                tags={"host": "localhost", "type": "system"}
            ))
            self.record_metric(PerformanceMetric(
                name="system.disk.write_bytes",
                value=disk_io.write_bytes,
                timestamp=time.time(),
                tags={"host": "localhost", "type": "system"}
            ))

        # 网络IO
        net_io = psutil.net_io_counters()
        if net_io is not None:
            self.record_metric(PerformanceMetric(
                name="system.network.bytes_sent",
                value=net_io.bytes_sent,
                timestamp=time.time(),
                tags={"host": "localhost", "type": "system"}
            ))
            self.record_metric(PerformanceMetric(
                name="system.network.bytes_recv",
                value=net_io.bytes_recv,
                timestamp=time.time(),
                tags={"host": "localhost", "type": "system"}
            ))

    def _store_metrics(self, metrics: List[PerformanceMetric]) -> None:
        """
        存储性能指标
        :param metrics: 指标列表
        """
        for backend in self.storage_backends:
            if backend in self.metric_handlers:
                try:
                    self.metric_handlers[backend](metrics)
                except Exception as e:
                    logger.error(f"存储指标到 {backend} 失败: {str(e)}")
            else:
                logger.warning(f"未知的指标存储后端: {backend}")

    def _handle_prometheus(self, metrics: List[PerformanceMetric]) -> None:
        """
        处理指标到Prometheus
        :param metrics: 指标列表
        """
        # 实际项目中应使用Prometheus客户端
        prom_config = self.monitor_config.get('prometheus', {})
        logger.info(f"模拟将 {len(metrics)} 条指标推送到Prometheus: {prom_config.get('endpoint')}")

    def _handle_influxdb(self, metrics: List[PerformanceMetric]) -> None:
        """
        处理指标到InfluxDB
        :param metrics: 指标列表
        """
        # 实际项目中应使用InfluxDB客户端
        influx_config = self.monitor_config.get('influxdb', {})
        logger.info(f"模拟将 {len(metrics)} 条指标写入InfluxDB: {influx_config.get('database')}")

    def _handle_file(self, metrics: List[PerformanceMetric]) -> None:
        """
        处理指标到文件
        :param metrics: 指标列表
        """
        file_path = self.monitor_config.get('file', {}).get('path', 'logs/performance.log')
        try:
            with open(file_path, 'a', encoding='utf-8') as f:
                for metric in metrics:
                    f.write(f"{metric.timestamp},{metric.name},{metric.value},{metric.tags}\n")
            logger.debug(f"写入 {len(metrics)} 条指标到文件: {file_path}")
        except Exception as e:
            logger.error(f"写入指标文件失败: {str(e)}")

    def get_metrics(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        查询性能指标
        :param query: 查询条件
        :return: 指标数据列表
        """
        # 实际项目中应从存储后端查询
        return [{
            "name": "system.cpu.usage",
            "value": 15.2,
            "timestamp": time.time(),
            "tags": {"host": "localhost"}
        }]

    def get_performance_report(self, time_range: str = '1h') -> Dict[str, Any]:
        """
        获取性能报告
        :param time_range: 时间范围
        :return: 性能报告字典
        """
        # 实际项目中应聚合分析指标数据
        return {
            "cpu_usage": {"avg": 25.3, "max": 78.2, "min": 10.1},
            "memory_usage": {"avg": 45.7, "max": 80.2, "min": 30.5},
            "disk_io": {"read": 1024, "write": 2048},
            "network_io": {"sent": 512, "received": 1024},
            "service_metrics": {
                "trading": {"latency": 12.5, "throughput": 1500},
                "risk": {"latency": 8.2, "throughput": 3000}
            }
        }

    def check_health_status(self) -> Dict[str, Any]:
        """
        检查系统健康状态
        :return: 健康状态字典
        """
        # 实际项目中应根据指标判断系统状态
        return {
            "status": "healthy",
            "indicators": {
                "cpu": "normal",
                "memory": "normal",
                "disk": "normal",
                "network": "normal"
            },
            "anomalies": []
        }

    def create_alert_rule(self, condition: str, action: str) -> bool:
        """
        创建性能告警规则
        :param condition: 告警条件
        :param action: 触发动作
        :return: 是否创建成功
        """
        # 实际项目中应保存到配置
        logger.info(f"创建性能告警规则: {condition} => {action}")
        return True

    def track_service_metrics(self, service_name: str, metrics: Dict[str, float]) -> None:
        """
        跟踪服务性能指标
        :param service_name: 服务名称
        :param metrics: 指标字典
        """
        timestamp = time.time()
        for metric_name, value in metrics.items():
            self.record_metric(PerformanceMetric(
                name=f"service.{service_name}.{metric_name}",
                value=value,
                timestamp=timestamp,
                tags={"service": service_name}
            ))
