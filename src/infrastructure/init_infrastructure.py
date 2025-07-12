#!/usr/bin/env python3
# 基础设施层初始化脚本

import logging
from pathlib import Path
from typing import Optional, Dict, Any

from .config.config_manager import ConfigManager
from .error.error_handler import ErrorHandler
from .error.retry_handler import RetryHandler
from .resource.resource_manager import ResourceManager
from .resource.gpu_manager import GPUManager
from .m_logging.log_manager import LogManager
from .monitoring.system_monitor import SystemMonitor
from .monitoring.application_monitor import ApplicationMonitor

class Infrastructure:
    """基础设施层入口类"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._components = {}
        self._logger = logging.getLogger(__name__)

        # 初始化顺序很重要
        self._init_config_manager()
        self._init_log_manager()
        self._init_error_handlers()
        self._init_resource_managers()
        self._init_monitoring()

        self._logger.info("Infrastructure layer initialized")

    def _init_config_manager(self):
        """初始化配置管理器"""
        try:
            config_dir = Path(__file__).parent.parent / "config"
            self.config = ConfigManager(config_dir=config_dir)
            self._components['config'] = self.config
            self._logger.info("ConfigManager initialized")

            # 注册热更新监听
            if self.config.get('config.watch_enabled', env='default', default=True):
                self.config.start_watching()
        except Exception as e:
            self._logger.critical(f"Failed to initialize ConfigManager: {e}")
            raise

    def _init_log_manager(self):
        """初始化日志系统"""
        try:
            log_config = self.config.get('logging', env='default', default={})
            self.log = LogManager(
                log_dir=log_config.get('dir', './logs'),
                app_name=log_config.get('app_name', 'rqa2025'),
                max_bytes=log_config.get('max_bytes', 10*1024*1024),
                backup_count=log_config.get('backup_count', 10)
            )
            self._components['log'] = self.log
            self._logger = self.log.get_logger('infrastructure')
            self._logger.info("LogManager initialized")
        except Exception as e:
            logging.critical(f"Failed to initialize LogManager: {e}")
            raise

    def _init_error_handlers(self):
        """初始化错误处理器"""
        try:
            error_config = self.config.get('error_handling', env='default', default={})

            self.retry_handler = RetryHandler(
                max_attempts=error_config.get('max_retries', 3),
                initial_delay=error_config.get('retry_delay', 1.0),
                backoff_factor=error_config.get('backoff_factor', 2.0),
                jitter=error_config.get('jitter', 0.1)
            )

            self.error_handler = ErrorHandler()

            self._components.update({
                'error_handler': self.error_handler,
                'retry_handler': self.retry_handler
            })
            self._logger.info("Error handlers initialized")
        except Exception as e:
            self._logger.critical(f"Failed to initialize error handlers: {e}")
            raise

    def _init_resource_managers(self):
        """初始化资源管理器"""
        try:
            resource_config = self.config.get('resources', env='default', default={})

            # 初始化资源管理器
            self.resource = ResourceManager(
                cpu_threshold=resource_config.get('cpu_threshold', 80.0),
                mem_threshold=resource_config.get('memory_threshold', 80.0),
                disk_threshold=resource_config.get('disk_threshold', 80.0)
            )

            # 初始化GPU管理器
            self.gpu = GPUManager()

            # 启动资源监控
            if resource_config.get('monitoring_enabled', True):
                self.resource.start_monitoring()
                if self.gpu.get_gpu_count() > 0:
                    self.gpu.start_monitoring()

            self._components.update({
                'resource_manager': self.resource,
                'gpu_manager': self.gpu
            })
            self._logger.info("Resource managers initialized")
        except Exception as e:
            self._logger.critical(f"Failed to initialize resource managers: {e}")
            raise

    def _init_monitoring(self):
        """初始化监控系统"""
        try:
            monitor_config = self.config.get('monitoring', env='default', default={})

            # 系统监控
            self.system_monitor = SystemMonitor(
                check_interval=monitor_config.get('system_interval', 60.0)
            )

            # 应用监控
            self.app_monitor = ApplicationMonitor(
                app_name=monitor_config.get('app_name', 'rqa2025')
            )

            # 启动监控
            if monitor_config.get('enabled', True):
                self.system_monitor.start_monitoring()

            self._components.update({
                'system_monitor': self.system_monitor,
                'app_monitor': self.app_monitor
            })
            self._logger.info("Monitoring systems initialized")
        except Exception as e:
            self._logger.critical(f"Failed to initialize monitoring: {e}")
            raise

    def get_component(self, name: str) -> Optional[Any]:
        """获取基础设施组件"""
        return self._components.get(name)

    def shutdown(self):
        """关闭基础设施层"""
        self._logger.info("Shutting down infrastructure layer...")

        # 关闭监控
        if hasattr(self, 'system_monitor'):
            self.system_monitor.stop_monitoring()

        if hasattr(self, 'resource'):
            self.resource.stop_monitoring()
            if hasattr(self, 'gpu') and self.gpu.has_gpu:
                self.gpu.stop_monitoring()

        # 关闭配置监听
        if hasattr(self, 'config'):
            self.config.stop_watching()

        self._logger.info("Infrastructure layer shutdown complete")

# 全局基础设施访问点
infra = Infrastructure()

def get_infrastructure() -> Infrastructure:
    """获取基础设施层实例"""
    return infra

def init_infrastructure():
    """初始化基础设施层"""
    return infra
