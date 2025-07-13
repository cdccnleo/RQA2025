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

class RqaLogger(logging.Logger):
    """自定义量化日志Logger，兼容测试用例接口"""
    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)
        self._context = {}
        self._format = None
        self._filters = []
        self._handlers = []

    @property
    def context(self):
        return self._context

    @context.setter
    def context(self, value):
        self._context = value

    def with_format(self, fmt):
        self._format = fmt
        return self

    def add_filter(self, filter_func):
        self._filters.append(filter_func)

    def add_handler(self, handler):
        self._handlers.append(handler)
        super().addHandler(handler)

    def _write_log(self, *args, **kwargs):
        # 简单实现，实际可根据args/kwargs写入日志
        pass

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
        self.log_dir = log_dir
        self._app_name = app_name or 'app'
        
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
            logger = RqaLogger(logger_name)
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
            
        except Exception as e:
            raise IOError(f"添加JSON文件处理器失败: {e}")

    @classmethod
    def debug(cls, msg):
        """记录DEBUG级别日志"""
        cls.get_logger().debug(msg)

    @classmethod
    def info(cls, msg):
        """记录INFO级别日志"""
        cls.get_logger().info(msg)

    @classmethod
    def warning(cls, msg):
        """记录WARNING级别日志"""
        cls.get_logger().warning(msg)

    @classmethod
    def error(cls, msg):
        """记录ERROR级别日志"""
        cls.get_logger().error(msg)

    @classmethod
    def critical(cls, msg):
        """记录CRITICAL级别日志"""
        cls.get_logger().critical(msg)

    @classmethod
    def set_level(cls, level):
        """设置日志级别"""
        logger = cls.get_logger()
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        logger.setLevel(level)

    @classmethod
    def close(cls):
        """关闭所有日志处理器"""
        instance = cls.get_instance()
        for logger in instance._loggers.values():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)

    def get_metrics(self):
        """获取日志指标"""
        return {
            'total_loggers': len(self._loggers),
            'sampler_config': self._sampler.get_config() if hasattr(self._sampler, 'get_config') else {}
        }

    def cleanup_old_logs(self, days=30):
        """清理旧日志文件"""
        if not self.log_dir:
            return
            
        import glob
        import time
        current_time = time.time()
        cutoff_time = current_time - (days * 24 * 60 * 60)
        
        for log_file in glob.glob(os.path.join(self.log_dir, "*.log*")):
            if os.path.getmtime(log_file) < cutoff_time:
                try:
                    os.remove(log_file)
                except OSError:
                    pass

    def export_logs(self, output_file, start_time=None, end_time=None):
        """导出日志"""
        if not self.log_dir:
            return False
            
        try:
            with open(output_file, 'w') as f:
                for log_file in os.listdir(self.log_dir):
                    if log_file.endswith('.log'):
                        with open(os.path.join(self.log_dir, log_file), 'r') as log_f:
                            f.write(log_f.read())
            return True
        except Exception:
            return False

    def search_logs(self, pattern, case_sensitive=False):
        """搜索日志"""
        if not self.log_dir:
            return []
            
        import re
        results = []
        flags = 0 if case_sensitive else re.IGNORECASE
        
        for log_file in os.listdir(self.log_dir):
            if log_file.endswith('.log'):
                with open(os.path.join(self.log_dir, log_file), 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        if re.search(pattern, line, flags):
                            results.append({
                                'file': log_file,
                                'line': line_num,
                                'content': line.strip()
                            })
        return results

    def compress_logs(self, target_file=None):
        """压缩日志文件"""
        import gzip
        if not self.log_dir:
            return False
            
        if target_file is None:
            target_file = os.path.join(self.log_dir, "logs_archive.gz")
            
        try:
            with gzip.open(target_file, 'wt') as f:
                for log_file in os.listdir(self.log_dir):
                    if log_file.endswith('.log'):
                        with open(os.path.join(self.log_dir, log_file), 'r') as log_f:
                            f.write(log_f.read())
            return True
        except Exception:
            return False
