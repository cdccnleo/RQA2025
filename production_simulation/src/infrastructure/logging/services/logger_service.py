
import logging

from ..core.base_logger import BaseLogger
from ..core.exceptions import ResourceError
from ..core.interfaces import ILogger, LogLevel
from ..core.unified_logger import UnifiedLogger
from ..formatters import TextFormatter, JSONFormatter
from ..handlers import ConsoleHandler, FileHandler
from ..handlers.base import BaseHandler
from ..storage import MemoryStorage
from .base_service import BaseService
from typing import Any, Dict, Optional, List, Callable
"""
基础设施层 - 日志服务实现

提供统一的日志服务接口，支持多种日志记录方式。
"""


class LoggerWrapper(BaseLogger):
    """日志器包装器，提供BaseLogger接口"""

    def __init__(self, logger):
        self.logger = logger
        self.name = logger.name
        self._handlers = []

    def log(self, level, message, **kwargs):
        getattr(self.logger, level.lower(), self.logger.info)(message)

    def debug(self, message, **kwargs):
        self.logger.debug(message)

    def info(self, message, **kwargs):
        self.logger.info(message)

    def warning(self, message, **kwargs):
        self.logger.warning(message)

    def error(self, message, **kwargs):
        self.logger.error(message)

    def critical(self, message, **kwargs):
        self.logger.critical(message)

    def add_handler(self, handler):
        """添加处理器"""
        if handler not in self._handlers:
            self._handlers.append(handler)
            self.logger.addHandler(handler)

    def remove_handler(self, handler):
        """移除处理器"""
        if handler in self._handlers:
            self._handlers.remove(handler)
            self.logger.removeHandler(handler)

    def shutdown(self):
        """关闭日志器"""
        for handler in self._handlers[:]:  # 复制列表以避免修改时的问题
            self.logger.removeHandler(handler)
        self._handlers.clear()


class LoggerService(BaseService):
    """日志服务实现"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化日志服务

        Args:
            config: 服务配置
        """
        super().__init__("LoggerService", config)

        # 服务配置
        self.default_level = self.config.get('default_level', 'INFO')
        self.max_loggers = self.config.get('max_loggers', 100)
        self.enable_persistence = self.config.get('enable_persistence', True)
        self.auto_create_missing = self.config.get('auto_create_missing', config is None)

        # 组件
        self.loggers: Dict[str, BaseLogger] = {}
        self.storage = MemoryStorage(self.config.get('storage_config', {}))
        self._setup_default_components()

    def _setup_default_components(self) -> None:
        """设置默认组件"""
        # 创建默认的root logger
        if 'root' not in self.loggers:
            # 将字符串级别转换为LogLevel枚举
            try:
                level_enum = LogLevel[self.default_level.upper()]
            except KeyError:
                level_enum = LogLevel.INFO  # 默认使用INFO级别

            # 使用标准的Python logging.Logger
            import logging
            root_logger = logging.getLogger('root')
            root_logger.setLevel(getattr(logging, self.default_level.upper(), logging.INFO))

            # 避免重复添加处理器
            if not root_logger.handlers:
                # 添加默认处理器
                console_handler = logging.StreamHandler()
                file_handler = logging.FileHandler(self.config.get('log_file', 'logs/app.log'))

                # 设置格式化器
                console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
                file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

                root_logger.addHandler(console_handler)
                root_logger.addHandler(file_handler)

            # 包装为BaseLogger接口兼容的类
            self.loggers['root'] = LoggerWrapper(root_logger)

    def create_logger(self, name: str, config: Optional[Dict[str, Any]] = None) -> BaseLogger:
        """
        创建日志器

        Args:
            name: 日志器名称
            config: 日志器配置

        Returns:
            创建的日志器实例
        """
        # 验证资源限制
        self._validate_logger_limit()

        # 检查是否已存在
        if name in self.loggers:
            return self.loggers[name]

        # 构建配置
        logger_config = self._build_logger_config(name, config)

        # 创建日志器实例
        logger = self._create_logger_instance(logger_config)

        # 添加处理器
        self._add_handlers_to_logger(logger, config)

        # 注册日志器
        self._register_logger(name, logger)

        # 记录成功请求
        self._record_request(True)

        return logger

    def _validate_logger_limit(self) -> None:
        """
        验证日志器数量限制

        Raises:
            ResourceError: 当超过最大日志器数量时
        """
        # 计算用户创建的日志器数量（不包括默认的root日志器）
        user_loggers = len([name for name in self.loggers.keys() if name != 'root'])
        if user_loggers >= self.max_loggers:
            raise ResourceError(f"Maximum number of user loggers ({self.max_loggers}) exceeded")

    def _build_logger_config(self, name: str, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        构建日志器配置

        Args:
            name: 日志器名称
            config: 用户配置

        Returns:
            完整的日志器配置
        """
        return {
            'name': name,
            'level': config.get('level', self.default_level) if config else self.default_level,
            'category': config.get('category', 'general') if config else 'general',
            'format_type': config.get('format_type', 'text') if config else 'text',
            'log_dir': config.get('log_dir', self.config.get('log_dir', 'logs')) if config else self.config.get('log_dir', 'logs'),
            'handlers': config.get('handlers', []) if config else []
        }

    def _create_logger_instance(self, logger_config: Dict[str, Any]) -> BaseLogger:
        """
        创建日志器实例

        Args:
            logger_config: 日志器配置

        Returns:
            日志器实例
        """
        # 使用LoggerWrapper包装标准logger，它有add_handler方法
        name = logger_config.pop('name', 'default')
        # UnifiedLogger不接受level参数，在配置中设置level
        standard_logger = logging.getLogger(name)
        level_value = logger_config.get('level', self.default_level)
        if isinstance(level_value, int):
            numeric_level = level_value
        elif isinstance(level_value, LogLevel):
            numeric_level = level_value.level
        else:
            numeric_level = getattr(logging, str(level_value).upper(), logging.INFO)
        standard_logger.setLevel(numeric_level)
        return LoggerWrapper(standard_logger)

    def _add_handlers_to_logger(self, logger: BaseLogger, config: Optional[Dict[str, Any]]) -> None:
        """
        向日志器添加处理器

        Args:
            logger: 日志器实例
            config: 用户配置
        """
        if not (config and config.get('handlers')):
            return

        for handler_config in config['handlers']:
            # 如果是字符串，转换为字典格式
            if isinstance(handler_config, str):
                handler_config = {'type': handler_config}
            handler = self._create_handler(handler_config)
            if handler:
                self._configure_handler_formatter(handler, handler_config)
                logger.add_handler(handler)

    def _create_handler(self, handler_config: Dict[str, Any]) -> Optional[BaseHandler]:
        """
        创建处理器

        Args:
            handler_config: 处理器配置

        Returns:
            处理器实例或None
        """
        handler_type = handler_config.get('type', 'console')

        if handler_type == 'console':
            return ConsoleHandler(handler_config)
        elif handler_type == 'file':
            return FileHandler(handler_config)
        else:
            return None

    def _configure_handler_formatter(self, handler: BaseHandler, handler_config: Dict[str, Any]) -> None:
        """
        配置处理器格式化器

        Args:
            handler: 处理器实例
            handler_config: 处理器配置
        """
        formatter_type = handler_config.get('formatter', 'text')

        if formatter_type == 'json':
            handler.set_formatter(JSONFormatter())
        else:
            handler.set_formatter(TextFormatter())

    def _register_logger(self, name: str, logger: BaseLogger) -> None:
        """
        注册日志器

        Args:
            name: 日志器名称
            logger: 日志器实例
        """
        self.loggers[name] = logger

    def get_logger(self, name: str, **config) -> Optional[BaseLogger]:
        """
        获取日志器

        Args:
            name: 日志器名称

        Returns:
            日志器实例，如果不存在返回None
        """
        logger = self.loggers.get(name)
        if logger is None and self.auto_create_missing:
            creation_config = config or None
            logger = self.create_logger(name, creation_config)
        return logger

    def remove_logger(self, name: str) -> bool:
        """
        移除日志器

        Args:
            name: 日志器名称

        Returns:
            是否成功移除
        """
        if name in self.loggers:
            logger = self.loggers[name]
            logger.shutdown()
            del self.loggers[name]
            self._record_request(True)
            return True
        return False

    def list_loggers(self) -> List[str]:
        """
        列出所有日志器

        Returns:
            日志器名称列表
        """
        return list(self.loggers.keys())

    def log_message(self, logger_name: str, level: str, message: str, **kwargs) -> bool:
        """
        记录日志消息

        Args:
            logger_name: 日志器名称
            level: 日志级别
            message: 日志消息
            **kwargs: 额外参数

        Returns:
            是否成功记录
        """
        # 获取日志器
        logger = self._get_logger_for_logging(logger_name)
        if not logger:
            return False

        try:
            # 记录日志消息
            self._log_to_logger(logger, level, message, **kwargs)

            # 持久化存储（如果启用）
            self._persist_log_if_enabled(logger_name, level, message, **kwargs)

            # 记录成功请求
            self._record_request(True)
            return True

        except Exception:
            # 记录失败请求
            self._record_request(False)
            return False

    def _get_logger_for_logging(self, logger_name: str) -> Optional[BaseLogger]:
        """
        获取用于记录日志的日志器

        Args:
            logger_name: 日志器名称

        Returns:
            日志器实例或None
        """
        return self.get_logger(logger_name)

    def _log_to_logger(self, logger: BaseLogger, level: str, message: str, **kwargs) -> None:
        """
        向日志器记录消息

        Args:
            logger: 日志器实例
            level: 日志级别
            message: 日志消息
            **kwargs: 额外参数
        """
        log_method = self._get_log_method_by_level(logger, level)
        log_method(message, **kwargs)

    def _get_log_method_by_level(self, logger: BaseLogger, level: str):
        """
        根据级别获取对应的日志方法

        Args:
            logger: 日志器实例
            level: 日志级别

        Returns:
            日志方法
        """
        level_upper = level.upper()
        log_methods = self._get_log_method_mapping(logger)

        return log_methods.get(level_upper, None)

    def _get_log_method_mapping(self, logger: BaseLogger) -> Dict[str, Callable]:
        """
        获取日志级别到方法的映射

        Args:
            logger: 日志器实例

        Returns:
            级别到方法的映射字典
        """
        return {
            'DEBUG': logger.debug,
            'INFO': logger.info,
            'WARNING': logger.warning,
            'ERROR': logger.error,
            'CRITICAL': logger.critical
        }

    def _persist_log_if_enabled(self, logger_name: str, level: str, message: str, **kwargs) -> None:
        """
        如果启用持久化，则存储日志记录

        Args:
            logger_name: 日志器名称
            level: 日志级别
            message: 日志消息
            **kwargs: 额外参数
        """
        if not self.enable_persistence:
            return

        log_record = self._build_log_record(logger_name, level, message, **kwargs)
        self.storage.store(log_record)

    def _build_log_record(self, logger_name: str, level: str, message: str, **kwargs) -> Dict[str, Any]:
        """
        构建日志记录

        Args:
            logger_name: 日志器名称
            level: 日志级别
            message: 日志消息
            **kwargs: 额外参数

        Returns:
            日志记录字典
        """
        return {
            'timestamp': kwargs.get('timestamp', None),
            'logger_name': logger_name,
            'level': level,
            'message': message,
            'extra': kwargs
        }

    def _start(self) -> bool:
        """启动服务"""
        try:
            # 启动所有日志器
            for logger in self.loggers.values():
                # 日志器通常是自动启动的，这里可以添加额外的初始化逻辑
                pass

            # 启动存储
            if self.enable_persistence:
                # 存储通常也是自动初始化的
                pass

            return True
        except Exception:
            return False

    def _stop(self) -> bool:
        """停止服务"""
        try:
            # 停止所有日志器
            for logger in self.loggers.values():
                logger.shutdown()

            # 清空日志器缓存
            self.loggers.clear()

            return True
        except Exception:
            return False

    def _get_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        return {
            'logger_count': len(self.loggers),
            'max_loggers': self.max_loggers,
            'default_level': self.default_level,
            'persistence_enabled': self.enable_persistence,
            'storage_status': self.storage.get_status() if self.enable_persistence else None
        }

    def _get_info(self) -> Dict[str, Any]:
        """获取服务信息"""
        return {
            'service_name': 'LoggerService',
            'service_type': 'LoggerService',
            'active_loggers': len(self.loggers),
            'description': 'Unified Logger Service',
            'capabilities': [
                'create_logger',
                'get_logger',
                'remove_logger',
                'log_message',
                'list_loggers'
            ],
            'supported_levels': ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
            'supported_formats': ['text', 'json', 'structured'],
            'supported_handlers': ['console', 'file', 'remote']
        }
