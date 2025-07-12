import logging
from typing import Dict, Optional
from logging.handlers import TimedRotatingFileHandler

class LogCategory:
    """日志分类常量"""
    SYSTEM = "system"
    BUSINESS = "business"
    SECURITY = "security"
    PERFORMANCE = "performance"

class ComponentLogger:
    """组件级日志控制器"""

    def __init__(self,
                component: str,
                base_level: str = "INFO",
                category: str = LogCategory.BUSINESS):
        """
        Args:
            component: 组件名称 (如 'trading', 'risk')
            base_level: 默认日志级别
            category: 日志分类
        """
        self.logger = logging.getLogger(f"{category}.{component}")
        self.logger.setLevel(base_level)
        self.category = category
        self.component = component

        # 组件特定处理器
        self._handlers: Dict[str, logging.Handler] = {}

    def add_file_handler(self,
                       log_dir: str,
                       level: Optional[str] = None,
                       rotation: str = "midnight",
                       backup_count: int = 7):
        """添加文件处理器"""
        handler = TimedRotatingFileHandler(
            filename=f"{log_dir}/{self.category}_{self.component}.log",
            when=rotation,
            backupCount=backup_count,
            encoding='utf-8'
        )
        if level:
            handler.setLevel(level)

        handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)8s | %(name)20s | %(message)s'
        ))
        self.logger.addHandler(handler)
        self._handlers['file'] = handler

    def update_level(self, level: str):
        """动态更新日志级别"""
        self.logger.setLevel(level)
        for handler in self._handlers.values():
            handler.setLevel(level)

    def add_context(self, **kwargs):
        """添加上下文信息"""
        extra = getattr(self.logger, 'context', {})
        extra.update(kwargs)
        setattr(self.logger, 'context', extra)

class LoggingStrategy:
    """集中式日志策略管理器"""

    def __init__(self, config: Dict):
        """
        Args:
            config: 日志策略配置
        """
        self.config = config
        self._components: Dict[str, ComponentLogger] = {}

        # 初始化基础日志器
        self._init_base_loggers()

    def _init_base_loggers(self):
        """初始化基础日志分类"""
        for category in [LogCategory.SYSTEM, LogCategory.BUSINESS]:
            logger = logging.getLogger(category)
            logger.setLevel(self.config['base_level'])

            # 控制台输出
            if self.config['console']:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(logging.Formatter(
                    '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
                ))
                logger.addHandler(console_handler)

    def get_logger(self,
                  component: str,
                  category: str = LogCategory.BUSINESS) -> ComponentLogger:
        """获取组件日志控制器"""
        key = f"{category}.{component}"
        if key not in self._components:
            self._components[key] = ComponentLogger(
                component=component,
                base_level=self.config['base_level'],
                category=category
            )
        return self._components[key]

    def update_config(self, new_config: Dict):
        """动态更新日志配置"""
        self.config.update(new_config)

        # 更新所有组件日志级别
        for logger in self._components.values():
            logger.update_level(self.config['base_level'])
