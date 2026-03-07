import logging
#!/usr / bin / env python
# -*- coding: utf-8 -*-

"""
模块化组件基础类
定义统一的模块接口和基础功能
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
from datetime import datetime
from pydantic import BaseModel
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from src.engine.logging.unified_logger import get_unified_logger

logger = logging.getLogger(__name__)


class ModuleConfig(BaseModel):

    """模块配置模型"""
    name: str
    display_name: str
    description: str
    icon: str
    route: str
    enabled: bool = True
    permissions: List[str] = []
    version: str = "1.0.0"
    dependencies: List[str] = []
    settings: Dict[str, Any] = {}


class ModuleData(BaseModel):

    """模块数据模型"""
    module_id: str
    data_type: str
    content: Dict[str, Any]
    timestamp: datetime
    metadata: Dict[str, Any] = {}


class ModuleStatus(BaseModel):

    """模块状态模型"""
    module_id: str
    status: str  # running, stopped, error, initializing
    is_healthy: bool
    last_activity: datetime
    error_count: int = 0
    uptime: float = 0.0  # 运行时间(秒)
    performance_metrics: Dict[str, Any] = {}
    alerts: List[Dict[str, Any]] = []


class BaseModule(ABC):

    """模块化组件基础类"""

    def __init__(self, config: ModuleConfig):

        self.config = config
        self.router = APIRouter(prefix=f"/{config.name}")
        self.logger = get_unified_logger(f"module.{config.name}")

        # 模块状态
        self.is_initialized = False
        self.is_running = False
        self.last_activity = datetime.now()

        # 注册路由
        self._register_routes()

        self.logger.info(f"模块 {config.name} 初始化完成")

    def _register_routes(self):
        """注册模块路由"""
        # 默认实现，子类可以重写

    @abstractmethod
    async def get_module_data(self) -> Dict[str, Any]:
        """获取模块数据"""

    @abstractmethod
    async def get_module_status(self) -> Dict[str, Any]:
        """获取模块状态"""

    @abstractmethod
    async def validate_permissions(self, user_permissions: List[str]) -> bool:
        """验证用户权限"""

    def get_config(self) -> ModuleConfig:
        """获取模块配置"""
        return self.config

    def get_router(self) -> APIRouter:
        """获取模块路由"""
        return self.router

    async def initialize(self) -> bool:
        """初始化模块"""
        try:
            if not self.is_initialized:
                await self._initialize_module()
                self.is_initialized = True
                self.logger.info(f"模块 {self.config.name} 初始化成功")
            return True
        except Exception as e:
            self.logger.error(f"模块 {self.config.name} 初始化失败: {str(e)}")
            return False

    async def start(self) -> bool:
        """启动模块"""
        try:
            if not self.is_running:
                await self._start_module()
                self.is_running = True
                self.logger.info(f"模块 {self.config.name} 启动成功")
            return True
        except Exception as e:
            self.logger.error(f"模块 {self.config.name} 启动失败: {str(e)}")
            return False

    async def stop(self) -> bool:
        """停止模块"""
        try:
            if self.is_running:
                await self._stop_module()
                self.is_running = False
                self.logger.info(f"模块 {self.config.name} 停止成功")
            return True
        except Exception as e:
            self.logger.error(f"模块 {self.config.name} 停止失败: {str(e)}")
            return False

    async def update_activity(self):
        """更新活动时间"""
        self.last_activity = datetime.now()

    def is_healthy(self) -> bool:
        """检查模块健康状态"""
        return self.is_initialized and self.is_running

    @abstractmethod
    async def _initialize_module(self):
        """初始化模块内部逻辑"""

    @abstractmethod
    async def _start_module(self):
        """启动模块内部逻辑"""

    @abstractmethod
    async def _stop_module(self):
        """停止模块内部逻辑"""

    def _add_common_routes(self):
        """添加通用路由"""

        @self.router.get("/", response_class=HTMLResponse)
        async def module_home(request: Request):
            """模块主页"""
            return await self._render_module_page(request)

        @self.router.get("/api / data")
        async def get_data():
            """获取模块数据"""
            return await self.get_module_data()

        @self.router.get("/api / status")
        async def get_status():
            """获取模块状态"""
            return await self.get_module_status()

        @self.router.get("/api / config")
        async def get_config():
            """获取模块配置"""
            return self.config.dict()

    async def _render_module_page(self, request: Request) -> HTMLResponse:
        """渲染模块页面"""
        # 这里应该返回模块特定的HTML模板
        return HTMLResponse(content=f"<h1>{self.config.display_name}</h1>")

    def _log_activity(self, action: str, details: Dict[str, Any] = None):
        """记录模块活动"""
        self.logger.info(f"模块 {self.config.name} {action}: {details or {}}")
        self.update_activity()
