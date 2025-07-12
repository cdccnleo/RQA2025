#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
服务启动器模块
负责系统各服务的启动、停止和状态管理
"""

import subprocess
import threading
import time
from typing import Dict, List, Optional, Any
from src.infrastructure.utils.logger import get_logger
from src.infrastructure.config.deployment_manager import DeploymentManager

logger = get_logger(__name__)

class ServiceLauncher:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化服务启动器
        :param config: 系统配置
        """
        self.config = config
        self.deployment_mgr = DeploymentManager(config)
        self.running_services = {}
        self.service_status = {}
        self.lock = threading.Lock()

        # 服务定义
        self.service_definitions = {
            'data_service': {
                'command': 'python src/data/main.py',
                'dependencies': [],
                'restart_policy': 'always'
            },
            'feature_service': {
                'command': 'python src/features/main.py',
                'dependencies': ['data_service'],
                'restart_policy': 'always'
            },
            'trading_service': {
                'command': 'python src/trading/main.py',
                'dependencies': ['data_service', 'feature_service'],
                'restart_policy': 'on-failure'
            },
            'risk_service': {
                'command': 'python src/trading/risk/main.py',
                'dependencies': ['data_service'],
                'restart_policy': 'always'
            },
            'monitoring_service': {
                'command': 'python src/infrastructure/monitoring/main.py',
                'dependencies': [],
                'restart_policy': 'always'
            }
        }

    def start_service(self, service_name: str) -> bool:
        """
        启动单个服务
        :param service_name: 服务名称
        :return: 是否启动成功
        """
        if service_name not in self.service_definitions:
            logger.error(f"未知服务: {service_name}")
            return False

        # 检查依赖服务
        for dep in self.service_definitions[service_name]['dependencies']:
            if not self.is_service_running(dep):
                logger.warning(f"服务 {service_name} 依赖的服务 {dep} 未运行")
                return False

        # 检查是否已运行
        if self.is_service_running(service_name):
            logger.info(f"服务 {service_name} 已在运行")
            return True

        try:
            # 加载环境配置
            if not self.deployment_mgr.load_environment(self.config['env']):
                logger.error("加载部署配置失败")
                return False

            # 构建启动命令
            cmd = self.service_definitions[service_name]['command']
            if self.config['env'] != 'prod':
                cmd += f" --env={self.config['env']}"

            # 启动服务
            process = subprocess.Popen(
                cmd.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            with self.lock:
                self.running_services[service_name] = process
                self.service_status[service_name] = {
                    'pid': process.pid,
                    'start_time': time.time(),
                    'restart_count': 0
                }

            logger.info(f"服务 {service_name} 启动成功, PID: {process.pid}")
            return True

        except Exception as e:
            logger.error(f"启动服务 {service_name} 失败: {str(e)}")
            return False

    def stop_service(self, service_name: str) -> bool:
        """
        停止单个服务
        :param service_name: 服务名称
        :return: 是否停止成功
        """
        if not self.is_service_running(service_name):
            logger.warning(f"服务 {service_name} 未运行")
            return False

        try:
            with self.lock:
                process = self.running_services.pop(service_name)
                self.service_status.pop(service_name)

            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

            logger.info(f"服务 {service_name} 已停止")
            return True

        except Exception as e:
            logger.error(f"停止服务 {service_name} 失败: {str(e)}")
            return False

    def start_all_services(self) -> Dict[str, bool]:
        """
        启动所有服务
        :return: 各服务启动结果
        """
        results = {}
        # 按依赖顺序启动服务
        startup_order = [
            'data_service',
            'monitoring_service',
            'feature_service',
            'risk_service',
            'trading_service'
        ]

        for service in startup_order:
            results[service] = self.start_service(service)
            if not results[service]:
                logger.error(f"服务 {service} 启动失败，中止后续服务启动")
                break

        return results

    def stop_all_services(self) -> Dict[str, bool]:
        """
        停止所有服务
        :return: 各服务停止结果
        """
        results = {}
        # 按依赖逆序停止服务
        shutdown_order = [
            'trading_service',
            'risk_service',
            'feature_service',
            'monitoring_service',
            'data_service'
        ]

        for service in shutdown_order:
            results[service] = self.stop_service(service)

        return results

    def restart_service(self, service_name: str) -> bool:
        """
        重启单个服务
        :param service_name: 服务名称
        :return: 是否重启成功
        """
        if not self.stop_service(service_name):
            return False
        return self.start_service(service_name)

    def is_service_running(self, service_name: str) -> bool:
        """
        检查服务是否在运行
        :param service_name: 服务名称
        :return: 是否运行中
        """
        with self.lock:
            if service_name not in self.running_services:
                return False

            process = self.running_services[service_name]
            return process.poll() is None

    def monitor_services(self) -> None:
        """
        监控服务状态并自动恢复
        """
        while True:
            with self.lock:
                for service_name, process in list(self.running_services.items()):
                    if process.poll() is not None:  # 服务已终止
                        status = self.service_status[service_name]
                        restart_policy = self.service_definitions[service_name]['restart_policy']

                        if restart_policy == 'always' or \
                           (restart_policy == 'on-failure' and process.returncode != 0):

                            status['restart_count'] += 1
                            logger.warning(
                                f"服务 {service_name} 已终止, 返回码: {process.returncode}, "
                                f"正在重启(第 {status['restart_count']} 次)..."
                            )

                            # 重新启动服务
                            self.running_services.pop(service_name)
                            self.start_service(service_name)

            time.sleep(5)  # 每5秒检查一次

    def get_service_status(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有服务状态
        :return: 服务状态字典
        """
        status = {}
        with self.lock:
            for name, process in self.running_services.items():
                status[name] = {
                    'pid': process.pid,
                    'running': process.poll() is None,
                    'uptime': time.time() - self.service_status[name]['start_time'],
                    'restart_count': self.service_status[name]['restart_count']
                }
        return status

    def start(self) -> None:
        """
        启动服务管理系统
        """
        # 启动所有服务
        self.start_all_services()

        # 启动监控线程
        monitor_thread = threading.Thread(
            target=self.monitor_services,
            daemon=True
        )
        monitor_thread.start()
        logger.info("服务监控线程已启动")

    def graceful_shutdown(self) -> None:
        """
        优雅关闭所有服务
        """
        logger.info("开始优雅关闭所有服务...")
        self.stop_all_services()
        logger.info("所有服务已停止")
