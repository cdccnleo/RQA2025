
# 合理跨层级导入：基础设施层工具类

from configparser import ConfigParser
import Path
import logging
import os
from typing import Optional
"""
基础设施层 - 日志系统组件

paths 模块

日志系统相关的文件
提供日志系统相关的功能实现。
"""

logger = logging.getLogger(__name__)


class PathConfig:

    """
paths - 配置管理

职责说明：
负责系统配置的统一管理、配置文件的读取、配置验证和配置分发

核心职责：
- 配置文件的读取和解析
- 配置参数的验证
- 配置的热重载
- 配置的分发和同步
- 环境变量管理
- 配置加密和安全

相关接口：
- IConfigComponent
- IConfigManager
- IConfigValidator
""" """管理项目路径配置，实现动态配置加载与目录创建

    属性：
        BASE_DIR (Path): 项目根目录
        DATA_DIR (Path): 数据存储目录
        MODEL_DIR (Path): 模型存储目录
        LOG_DIR (Path): 日志目录
        CACHE_DIR (Path): 缓存目录
    """

    def __init__(self, config: Optional[str] = None):

        self._load_config(config)

    def _load_config(self, config: Optional[str] = None):

        config_parser = ConfigParser()

        if config:
            config_path = Path(config)
            base_dir = config_path.parent
        else:
            config_path = Path(__file__).parent.parent / 'config / config.ini'
            base_dir = Path(__file__).parent.parent

        if not config_path.exists():
            raise RuntimeError(f"配置文件不存在: {config_path}")

        try:
            config_parser.read(config_path)
        except Exception as e:
            raise RuntimeError(f"配置文件解析失败: {str(e)}") from e

        base_dir_config = config_parser.get('Paths', 'BASE_DIR', fallback=os.getcwd())
        if not Path(base_dir_config).is_absolute():
            self.BASE_DIR = Path(base_dir) / base_dir_config
        else:
            self.BASE_DIR = Path(base_dir_config)

        self.BASE_DIR = self.BASE_DIR.resolve()  # 确保解析为绝对路径

        data_dir_config = config_parser.get('Paths', 'DATA_DIR', fallback='data')
        if Path(data_dir_config).is_absolute():
            self.DATA_DIR = Path(data_dir_config)
        else:
            self.DATA_DIR = self.BASE_DIR / data_dir_config

        self.MODEL_DIR = self.BASE_DIR / config_parser.get('Paths', 'MODEL_DIR', fallback='models')
        self.LOG_DIR = self.BASE_DIR / config_parser.get('Paths', 'LOG_DIR', fallback='logs')
        self.CACHE_DIR = self.BASE_DIR / config_parser.get('Paths', 'CACHE_DIR', fallback='cache')

        # 处理轮转策略
        rotation = config_parser.get('Logging', 'rotation', fallback='size')
        if isinstance(rotation, str):
            rotation = rotation.split('#')[0].strip().lower()  # 处理注释和空格
        self.rotation = rotation if rotation in ['size', 'time'] else 'size'  # 默认回退到'size'

        self._create_directories()

    def _create_directories(self):

        try:
            self.DATA_DIR.mkdir(parents=True, exist_ok=True)
            self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
            self.LOG_DIR.mkdir(parents=True, exist_ok=True)
            self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"目录创建失败: {str(e)}") from e

    def get_model_path(self, model_name: str) -> Path:
        """获取模型文件完整路径"""
        if not model_name:
            return self.MODEL_DIR / "unnamed_model.pkl"
        return self.MODEL_DIR / f"{model_name}.pkl"

    def get_cache_file(self, filename: str) -> Path:
        """获取缓存文件完整路径"""
        return self.CACHE_DIR / filename


# 单例配置实例 - 延迟初始化
_path_config = None


def get_path_config() -> 'PathConfig':
    """获取路径配置实例（延迟初始化）"""
    global _path_config
    if _path_config is None:
        _path_config = PathConfig()
    return _path_config


def get_config_path() -> Path:
    """获取配置文件路径"

    返回:
        Path: 配置文件的绝对路径
    """
    return get_path_config().BASE_DIR / "config / config.ini"


class ConfigPaths:

    """配置路径管理器"""

    def __init__(self, base_dir: str = None):

        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent.parent
        self.config_dir = self.base_dir / "config"
        self.data_dir = self.base_dir / "data"
        self.log_dir = self.base_dir / "logs"
        self.cache_dir = self.base_dir / "cache"
        self.model_dir = self.base_dir / "models"

        self._ensure_directories()

    def _ensure_directories(self):
        """确保必要的目录存在"""
        for directory in [self.config_dir, self.data_dir, self.log_dir,
                          self.cache_dir, self.model_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def get_config_file(self, filename: str) -> Path:
        """获取配置文件路径"""
        return self.config_dir / filename

    def get_data_file(self, filename: str) -> Path:
        """获取数据文件路径"""
        return self.data_dir / filename

    def get_log_file(self, filename: str) -> Path:
        """获取日志文件路径"""
        return self.log_dir / filename

    def get_cache_file(self, filename: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / filename

    def get_model_file(self, filename: str) -> Path:
        """获取模型文件路径"""
        return self.model_dir / filename




