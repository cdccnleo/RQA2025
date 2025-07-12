import logging
import json
import os
from datetime import datetime
from concurrent_log_handler import ConcurrentRotatingFileHandler
from .log_sampler import LogSampler, SamplingStrategyType


class JsonFormatter(logging.Formatter):
    """JSON日志格式化器"""

    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "module": os.path.splitext(os.path.basename(record.pathname))[0] if record.pathname else "",
            "lineNo": record.lineno
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)

class LogManager:
    """日志管理器"""

    _instance = None

    def __init__(self, app_name=None, log_dir=None, max_bytes=None, backup_count=None, log_level="INFO"):
        """初始化日志管理器
        
        Args:
            app_name: 应用名称(可选)
            log_dir: 日志目录路径(可选)
            max_bytes: 单个日志文件最大字节数(可选)
            backup_count: 保留的备份文件数(可选)
            log_level: 日志级别(DEBUG/INFO/WARNING/ERROR/CRITICAL)，默认为INFO
        """
        self._loggers = {}
        self._sampler = LogSampler()  # 默认采样器
        self.sampler = self._sampler  # 公开访问
        
        if app_name or log_dir:
            self.configure({
                'app_name': app_name,
                'log_dir': log_dir,
                'max_bytes': max_bytes,
                'backup_count': backup_count,
                'log_level': log_level
            })
        
    def configure_sampler(self, config):
        """配置日志采样器
        
        Args:
            config (dict): 采样器配置字典，包含:
                - default_rate: 默认采样率(0.0-1.0)
                - level_rates: 各日志级别的采样率
                - trading_hours_rate: 交易时段的采样率(可选)
        """
        if not isinstance(config, dict):
            raise ValueError("采样器配置必须是字典")
        self._sampler.configure(config)
        
    def debug(self, msg, *args, **kwargs):
        """记录DEBUG级别日志"""
        if self._sampler.should_sample('DEBUG'):
            self.logger.debug(msg, *args, **kwargs)
            
    def info(self, msg, *args, **kwargs):
        """记录INFO级别日志"""
        if self._sampler.should_sample('INFO'):
            self.logger.info(msg, *args, **kwargs)
        """可选的构造参数，用于直接初始化
        
        Args:
            app_name: 应用名称(可选)
            log_dir: 日志目录路径(可选)
            max_bytes: 单个日志文件最大字节数(可选)
            backup_count: 保留的备份文件数(可选)
            log_level: 日志级别(DEBUG/INFO/WARNING/ERROR/CRITICAL)，默认为INFO
        """
        self._loggers = {}
        if app_name or log_dir:
            self.configure({
                'app_name': app_name,
                'log_dir': log_dir,
                'max_bytes': max_bytes,
                'backup_count': backup_count,
                'log_level': log_level
            })

    @classmethod
    def get_instance(cls):
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def get_logger(cls, name=None):
        """获取日志记录器
        
        Args:
            name: 日志记录器名称，如果不指定则使用默认应用名
        """
        instance = cls.get_instance()
        # 获取基础应用名
        app_name = getattr(instance, '_app_name', 'test_app')
        
        # 构建完整日志记录器名称
        if name:
            logger_name = f"{app_name}.{name}" if not name.startswith(app_name) else name
        else:
            logger_name = app_name
            
        if logger_name not in instance._loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.INFO)
            instance._loggers[logger_name] = logger
        return instance._loggers[logger_name]

    def configure(self, config):
        """配置日志系统"""
        app_name = config.get('app_name', 'app')
        logger = logging.getLogger(app_name)
        
        # 清除现有处理器
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # 设置日志级别
        log_level = config.get('log_level', 'INFO')
        level = getattr(logging, log_level.upper()) if isinstance(log_level, str) else log_level
        logger.setLevel(level)

        # 配置文件处理器
        if 'log_dir' in config:
            import os

            log_dir = config['log_dir']
            os.makedirs(log_dir, exist_ok=True)

            handler = ConcurrentRotatingFileHandler(
                os.path.join(log_dir, f"{app_name}.log"),
                maxBytes=config.get('max_bytes', 1024*1024),
                backupCount=config.get('backup_count', 5)
            )
            handler.setLevel(level)  # 处理器继承日志记录器级别
            handler.setFormatter(JsonFormatter())
            logger.addHandler(handler)

            # 添加错误日志处理器
            error_handler = ConcurrentRotatingFileHandler(
                os.path.join(log_dir, f"{app_name}_error.log"),
                maxBytes=config.get('max_bytes', 1024*1024),
                backupCount=config.get('backup_count', 5)
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(JsonFormatter())
            logger.addHandler(error_handler)

    @classmethod
    def add_json_file_handler(cls, filename):
        """添加JSON格式的文件日志处理器
        
        Args:
            filename: 日志文件名或完整路径
            
        Raises:
            IOError: 如果无法创建或写入日志文件
        """
        instance = cls.get_instance()
        try:
            # 确保使用绝对路径
            if not os.path.isabs(filename):
                log_dir = getattr(instance, '_log_dir', os.getcwd())
                filename = os.path.join(log_dir, filename)
            
            # 确保目录存在
            dirname = os.path.dirname(filename)
            if dirname:  # 如果有目录路径
                os.makedirs(dirname, exist_ok=True)
            
            # 预先验证文件可写性
            try:
                with open(filename, 'a') as f:
                    f.write('')  # 测试写入空内容
            except IOError as e:
                raise IOError(f"无法写入日志文件 {filename}: {e}")
            
            # 创建文件处理器，设置delay=False确保立即写入
            handler = logging.FileHandler(filename, delay=False)
            handler.setFormatter(JsonFormatter())
            
            # 获取日志记录器并添加处理器
            logger = cls.get_logger()
            handler.setLevel(logger.level)  # 继承日志记录器级别
            logger.addHandler(handler)
            
            # 记录日志处理器添加成功
            logger.info(f"成功添加JSON文件日志处理器: {filename}")
            
        except Exception as e:
            # 使用基础日志记录错误，避免循环
            logging.basicConfig(level=logging.ERROR)
            logging.error(f"添加JSON文件日志处理器失败: {str(e)}")
            raise

    @classmethod
    def debug(cls, msg):
        cls.get_logger().debug(msg)

    @classmethod
    def info(cls, msg):
        cls.get_logger().info(msg)

    @classmethod
    def warning(cls, msg):
        cls.get_logger().warning(msg)

    @classmethod
    def error(cls, msg):
        cls.get_logger().error(msg)

    @classmethod
    def critical(cls, msg):
        cls.get_logger().critical(msg)

    @classmethod
    def set_level(cls, level):
        """设置日志级别
        
        Args:
            level: 日志级别(DEBUG/INFO/WARNING/ERROR/CRITICAL)
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        logger = cls.get_logger()
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)

    @classmethod
    def close(cls):
        """关闭所有日志处理器并清理资源
        
        Returns:
            bool: 总是返回True表示成功关闭
        """
        logger = cls.get_logger()
        
        # 获取所有处理器副本
        handlers = logger.handlers[:]
        
        # 关闭并移除每个处理器
        for handler in handlers:
            try:
                # 先关闭处理器
                handler.close()
                # 然后从logger中移除
                logger.removeHandler(handler)
            except Exception as e:
                # 记录错误但继续执行
                logging.warning(f"Error closing handler {handler}: {str(e)}")
                continue
        
        # 确保所有处理器都被移除
        if logger.handlers:
            logging.warning(f"Some handlers remain after close: {logger.handlers}")
            # 强制清除剩余处理器
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
        
        return True

# 默认实例
log_manager = LogManager.get_instance()
