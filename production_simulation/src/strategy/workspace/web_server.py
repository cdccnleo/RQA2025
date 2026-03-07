#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
策略工作空间Web服务器
Strategy Workspace Web Server

启动策略工作空间的Web服务，提供完整的可视化界面。
"""

import asyncio
import logging
import signal
from pathlib import Path
import uvicorn
from typing import Dict
from .web_api import StrategyWorkspaceAPI
from strategy.core.dependency_config import (
    configure_strategy_services,
    get_strategy_service,
    get_backtest_service,
    get_optimization_service
)
from strategy.core.container import DependencyContainer

logger = logging.getLogger(__name__)


class StrategyWorkspaceServer:

    """
    策略工作空间Web服务器
    Strategy Workspace Web Server

    提供完整的Web服务，包括API和静态文件服务。
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8000,


                 config_path: str = "./config / workspace_config.yaml"):
        """
        初始化Web服务器

        Args:
            host: 监听主机
            port: 监听端口
            config_path: 配置文件路径
        """
        self.host = host
        self.port = port
        self.config_path = Path(config_path)

        # 创建依赖注入容器
        self.container = DependencyContainer()

        # 创建Web API
        self.api = StrategyWorkspaceAPI()

        # 服务器状态
        self.running = False
        self.server_task = None

        logger.info(f"策略工作空间Web服务器初始化完成: {host}:{port}")

    async def initialize_services(self):
        """
        初始化服务
        """
        try:
            logger.info("初始化策略服务...")

            # 配置策略服务
            config = configure_strategy_services(self.container)

            # 获取服务实例
            strategy_service = get_strategy_service(self.container)
            backtest_service = get_backtest_service(self.container)
            optimization_service = get_optimization_service(self.container)

            # 设置API服务
            self.api.set_services(
                strategy_service=strategy_service,
                backtest_service=backtest_service,
                optimization_service=optimization_service
            )

            # 初始化监控服务（如果存在）
            try:
                from ..monitoring.monitoring_service import MonitoringService
                from ..monitoring.alert_service import AlertService

                monitoring_service = MonitoringService()
                alert_service = AlertService()

                self.api.set_services(
                    monitoring_service=monitoring_service,
                    alert_service=alert_service
                )
            except ImportError:
                logger.warning("监控服务未配置")

            logger.info("策略服务初始化完成")

        except Exception as e:
            logger.error(f"服务初始化失败: {e}")
            raise

    async def start_server(self):
        """
        启动Web服务器
        """
        try:
            # 初始化服务
            await self.initialize_services()

            # 设置信号处理

            def signal_handler(signum, frame):

                logger.info(f"收到信号 {signum}，准备关闭服务器...")
                asyncio.create_task(self.stop_server())

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            self.running = True

            logger.info(f"启动策略工作空间Web服务器: http://{self.host}:{self.port}")
            logger.info(f"API文档: http://{self.host}:{self.port}/docs")
            logger.info(f"API界面: http://{self.host}:{self.port}/redoc")

            # 启动服务器
            config = uvicorn.Config(
                app=self.api.get_app(),
                host=self.host,
                port=self.port,
                log_level="info",
                reload=False,  # 生产环境关闭自动重载
                workers=1,  # 单进程模式
                access_log=True
            )

            server = uvicorn.Server(config)

            # 保存服务器任务引用
            self.server_task = asyncio.create_task(server.serve())

            # 等待服务器运行
            await self.server_task

        except Exception as e:
            logger.error(f"Web服务器启动失败: {e}")
            self.running = False
            raise

    async def stop_server(self):
        """
        停止Web服务器
        """
        try:
            logger.info("正在停止策略工作空间Web服务器...")

            self.running = False

            if self.server_task and not self.server_task.done():
                self.server_task.cancel()

                try:
                    await self.server_task
                except asyncio.CancelledError:
                    pass

            # 关闭服务
            await self._shutdown_services()

            logger.info("策略工作空间Web服务器已停止")

        except Exception as e:
            logger.error(f"Web服务器停止失败: {e}")

    async def _shutdown_services(self):
        """
        关闭服务
        """
        try:
            # 这里可以添加服务关闭逻辑
            logger.info("服务关闭完成")

        except Exception as e:
            logger.error(f"服务关闭失败: {e}")

    def is_running(self) -> bool:
        """
        检查服务器是否正在运行

        Returns:
            bool: 运行状态
        """
        return self.running

    def get_server_info(self) -> dict:
        """
        获取服务器信息

        Returns:
            dict: 服务器信息
        """
        return {
            "host": self.host,
            "port": self.port,
            "running": self.running,
            "api_url": f"http://{self.host}:{self.port}",
            "docs_url": f"http://{self.host}:{self.port}/docs",
            "health_url": f"http://{self.host}:{self.port}/health"
        }


# 服务器管理器

class WorkspaceServerManager:

    """
    工作空间服务器管理器
    Workspace Server Manager

    管理Web服务器的启动、停止和监控。
    """

    def __init__(self):
        """初始化服务器管理器"""
        self.servers: Dict[str, StrategyWorkspaceServer] = {}
        self.default_server_id = "main"

        logger.info("工作空间服务器管理器初始化完成")

    def create_server(self, server_id: str = None,


                      host: str = "0.0.0.0", port: int = 8000,
                      config_path: str = "./config / workspace_config.yaml") -> str:
        """
        创建服务器

        Args:
            server_id: 服务器ID
            host: 监听主机
            port: 监听端口
            config_path: 配置文件路径

        Returns:
            str: 服务器ID
        """
        if server_id is None:
            server_id = f"server_{len(self.servers) + 1}"

        if server_id in self.servers:
            logger.warning(f"服务器 {server_id} 已存在，将覆盖")

        server = StrategyWorkspaceServer(host=host, port=port, config_path=config_path)
        self.servers[server_id] = server

        logger.info(f"服务器 {server_id} 创建成功: {host}:{port}")
        return server_id

    async def start_server(self, server_id: str = None):
        """
        启动服务器

        Args:
            server_id: 服务器ID
        """
        if server_id is None:
            server_id = self.default_server_id

        if server_id not in self.servers:
            raise ValueError(f"服务器 {server_id} 不存在")

        server = self.servers[server_id]

        if server.is_running():
            logger.warning(f"服务器 {server_id} 已在运行")
            return

        logger.info(f"启动服务器 {server_id}...")
        await server.start_server()

    async def stop_server(self, server_id: str = None):
        """
        停止服务器

        Args:
            server_id: 服务器ID
        """
        if server_id is None:
            server_id = self.default_server_id

        if server_id not in self.servers:
            logger.warning(f"服务器 {server_id} 不存在")
            return

        server = self.servers[server_id]

        if not server.is_running():
            logger.warning(f"服务器 {server_id} 未在运行")
            return

        logger.info(f"停止服务器 {server_id}...")
        await server.stop_server()

    async def restart_server(self, server_id: str = None):
        """
        重启服务器

        Args:
            server_id: 服务器ID
        """
        if server_id is None:
            server_id = self.default_server_id

        logger.info(f"重启服务器 {server_id}...")

        await self.stop_server(server_id)

        # 等待一段时间确保完全停止
        await asyncio.sleep(2)

        await self.start_server(server_id)

    def get_server_info(self, server_id: str = None) -> dict:
        """
        获取服务器信息

        Args:
            server_id: 服务器ID

        Returns:
            dict: 服务器信息
        """
        if server_id is None:
            server_id = self.default_server_id

        if server_id not in self.servers:
            return {"error": f"服务器 {server_id} 不存在"}

        server = self.servers[server_id]
        return server.get_server_info()

    def list_servers(self) -> Dict[str, dict]:
        """
        列出所有服务器

        Returns:
            Dict[str, dict]: 服务器信息字典
        """
        return {
            server_id: server.get_server_info()
            for server_id, server in self.servers.items()
        }

    async def start_all_servers(self):
        """
        启动所有服务器
        """
        logger.info("启动所有服务器...")

        tasks = []
        for server_id in self.servers.keys():
            task = asyncio.create_task(self.start_server(server_id))
            tasks.append(task)

        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("所有服务器启动完成")

    async def stop_all_servers(self):
        """
        停止所有服务器
        """
        logger.info("停止所有服务器...")

        tasks = []
        for server_id in self.servers.keys():
            task = asyncio.create_task(self.stop_server(server_id))
            tasks.append(task)

        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("所有服务器停止完成")


# 全局服务器管理器实例
server_manager = WorkspaceServerManager()


# 便捷函数

def create_workspace_server(host: str = "0.0.0.0", port: int = 8000) -> StrategyWorkspaceServer:
    """
    创建工作空间服务器

    Args:
        host: 监听主机
        port: 监听端口

    Returns:
        StrategyWorkspaceServer: 服务器实例
    """
    return StrategyWorkspaceServer(host=host, port=port)


async def start_workspace_server(host: str = "0.0.0.0", port: int = 8000):
    """
    启动工作空间服务器

    Args:
        host: 监听主机
        port: 监听端口
    """
    server = create_workspace_server(host=host, port=port)
    await server.start_server()


def run_workspace_server(host: str = "0.0.0.0", port: int = 8000):
    """
    运行工作空间服务器（阻塞模式）

    Args:
        host: 监听主机
        port: 监听端口
    """
    asyncio.run(start_workspace_server(host=host, port=port))


# 导出
__all__ = [
    'StrategyWorkspaceServer',
    'WorkspaceServerManager',
    'server_manager',
    'create_workspace_server',
    'start_workspace_server',
    'run_workspace_server'
]
