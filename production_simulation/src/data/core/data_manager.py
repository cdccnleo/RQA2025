"""
数据管理器 - 重构版本
"""
import configparser
import asyncio
import threading
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Type
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

# 集成统一基础设施集成层
try:
    # 导入统一基础设施集成层
    from src.core.integration import get_data_adapter

    # 基础设施集成层可用
    INFRASTRUCTURE_INTEGRATION_AVAILABLE = True
    logger = logging.getLogger('data_manager')

except ImportError as e:
    # 基础设施集成层不可用，使用降级方案
    INFRASTRUCTURE_INTEGRATION_AVAILABLE = False
    logger = logging.getLogger('data_manager')
    logger.warning(f"统一基础设施集成层不可用，使用降级方案: {e}")

# 修复导入问题，使用try - except处理
try:
    from src.infrastructure.utils.exceptions import DataLoaderError
except ImportError:

    class DataLoaderError(Exception):

        """数据加载错误"""

try:
    from src.infrastructure.resource import global_resource_manager
except ImportError:

    class global_resource_manager:

        """全局资源管理器降级实现"""
        @staticmethod
        def get_resource_usage():

            return {"cpu": 50, "memory": 60}

        @staticmethod
        def register_object(obj):
            """注册对象到资源管理器（降级实现）"""

        @staticmethod
        def unregister_object(obj):
            """从资源管理器注销对象（降级实现）"""


from ..interfaces.IDataModel import IDataModel
from .base_loader import BaseDataLoader
from .registry import DataRegistry
from ..validation.china_stock_validator import ChinaStockValidator
from ..cache.cache_manager import CacheManager, CacheConfig
from ..quality.monitor import DataQualityMonitor
from ..compliance.data_compliance_manager import DataComplianceManager


class DataManagerSingleton:

    """DataManager单例管理器"""
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, config_path: Optional[Union[str, Path]] = None, config_dict: Optional[dict] = None):
        """
        获取DataManager单例实例

        Args:
            config_path: 配置文件路径
            config_dict: 配置字典

        Returns:
            DataManager: 单例实例
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = DataManager(config_path, config_dict)
        return cls._instance


class DataModel(IDataModel):

    """
    数据模型实现，用于封装数据和元数据
    """

    def __init__(self, data: pd.DataFrame, frequency: str, metadata: Dict[str, Any] = None):
        """
        初始化数据模型

        Args:
            data: 数据框
            frequency: 数据频率
            metadata: 元数据信息
        """
        self.data = data
        self._frequency = frequency
        self._user_metadata = dict(metadata) if metadata else {}
        self._metadata = dict(self._user_metadata)
        # 只有在metadata为None时才自动补充created_at，否则只补充data_shape、data_columns
        if not metadata or 'created_at' not in self._metadata:
            self._metadata['created_at'] = datetime.now().isoformat()
        self._metadata.update({
            'data_shape': data.shape if data is not None else None,
            'data_columns': data.columns.tolist() if data is not None else None,
        })

    def validate(self) -> bool:
        """
        数据有效性验证

        Returns:
            bool: 数据是否有效
        """
        if self.data is None or self.data.empty:
            return False
        return True

    def get_frequency(self) -> str:
        """
        获取数据频率

        Returns:
            str: 数据频率
        """
        return self._frequency

    def get_metadata(self, user_only: bool = False) -> Dict[str, Any]:
        """
        获取元数据信息
        user_only: 是否仅返回用户原始元数据（不含自动补充字段）
        Returns:
            Dict[str, Any]: 元数据信息
        """
        if user_only:
            return self._user_metadata
        return self._metadata

    def from_dict(self, data_dict: Dict[str, Any]) -> 'DataModel':
        """
        从字典创建数据模型

        Args:
            data_dict: 包含数据的字典

        Returns:
            DataModel: 数据模型实例
        """
        data = pd.DataFrame(data_dict.get('data', {}))
        frequency = data_dict.get('frequency', '1d')
        metadata = data_dict.get('metadata', {})
        return DataModel(data, frequency, metadata)

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典

        Returns:
            Dict[str, Any]: 数据字典
        """
        return {
            'data': self.data.to_dict() if self.data is not None else {},
            'frequency': self._frequency,
            'metadata': self._metadata
        }

    @property
    def columns(self):

        return self.data.columns

    def __len__(self):

        return len(self.data)

    def equals(self, other):

        if not isinstance(other, DataModel):
            return False
        return self.data.equals(other.data)


class DataManager:

    """
    数据管理器，负责协调数据加载、验证和缓存
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None, config_dict: Optional[dict] = None):
        """
        初始化数据管理器

        Args:
            config_path: 配置文件路径
            config_dict: 配置字典
        """
        # 初始化配置
        self.config = self._init_config(config_path, config_dict)

        # 初始化基础设施服务（统一集成层）
        self._init_infrastructure_services()

        # logger已在_init_infrastructure_services中初始化

        # 初始化组件
        self.registry = DataRegistry()
        self.validator = ChinaStockValidator()

        # 初始化缓存管理器（重构版本）
        if hasattr(self, 'cache_manager') and self.cache_manager:
            # 缓存管理器已在_init_infrastructure_services中初始化
            self.logger.info("使用统一基础设施集成层缓存管理器")
        else:
            # 使用降级缓存管理器
            try:
                cache_config = CacheConfig(
                    max_size=1000,
                    ttl=3600,
                    enable_disk_cache=True,
                    disk_cache_dir='cache',
                    compression=False,
                    encryption=False,
                    enable_stats=True,
                    cleanup_interval=300,
                    max_file_size=10 * 1024 * 1024,
                    backup_enabled=False,
                    backup_interval=3600
                )
                self.cache_manager = CacheManager(cache_config)
                # 注册缓存管理器到全局资源管理器
                global_resource_manager.register_object(self.cache_manager)
                self.logger.info("使用降级缓存管理器")
            except Exception as e:
                self.logger.warning(f"缓存管理器初始化失败: {e}")
                self.cache_manager = None

        self.quality_monitor = DataQualityMonitor()

        # 初始化合规管理器
        self.compliance_manager = DataComplianceManager()

        # 初始化加载器
        self._init_loaders()

        # 注册核心服务到服务容器
        self._register_core_services()

        # 初始化线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

        # 内部数据存储结构（降级模式下使用）
        self._data_store: Dict[str, Any] = {}
        self._metadata_store: Dict[str, Dict[str, Any]] = {}
        self._user_metadata_store: Dict[str, Dict[str, Any]] = {}

        self.logger.info("DataManager initialized successfully")

    # 集成基础设施桥接层的方法

    def get_infrastructure_health(self) -> Dict[str, Any]:
        """
        获取基础设施层的健康状态

        Returns:
            健康状态信息
        """
        if not INFRASTRUCTURE_INTEGRATION_AVAILABLE:
            return {
                'status': 'unhealthy',
                'message': '统一基础设施集成层不可用',
                'services': {}
            }

        try:
            # 获取适配器的健康状态
            data_adapter = get_data_adapter()
            health_status = data_adapter.health_check()

            return {
                'status': health_status.get('overall_status', 'unknown'),
                'services': health_status.get('services', {}),
                'performance_metrics': health_status.get('performance_metrics', {}),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'services': {},
                'timestamp': datetime.now().isoformat()
            }

    def publish_data_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        发布数据事件

        Args:
            event_type: 事件类型
            event_data: 事件数据
        """
        if hasattr(self, 'event_bus') and self.event_bus:
            try:
                self.event_bus.publish(event_type, event_data)
            except Exception as e:
                self.logger.warning(f"发布事件失败: {e}")
                self.logger.info(f"数据事件: {event_type} - {event_data}")
        else:
            self.logger.info(f"数据事件: {event_type} - {event_data}")

    def get_data_config(self, key: str, default: Any = None) -> Any:
        """
        获取数据配置

        Args:
            key: 配置键

            default: 默认值

        Returns:
            配置值
        """
        if hasattr(self, 'config_manager') and self.config_manager:
            try:
                return self.config_manager.get(key, default)
            except Exception as e:
                self.logger.warning(f"获取配置失败: {e}")
                return default
        else:
            # 从传统配置中获取
            try:
                if 'DEFAULT' in self.config:
                    return self.config.get('DEFAULT', str(key), fallback=default)
                else:
                    return default
            except Exception:
                return default

    def set_data_config(self, key: str, value: Any) -> bool:
        """
        设置数据配置

        Args:
            key: 配置键
            value: 配置值

        Returns:
            是否设置成功
        """
        if hasattr(self, 'config_manager') and self.config_manager:
            try:
                self.config_manager.set(key, value)
                return True
            except Exception as e:
                self.logger.warning(f"设置配置失败: {e}")
                return False
        else:
            # 设置到传统配置
            try:
                if 'DEFAULT' not in self.config:
                    self.config.add_section('DEFAULT')
                self.config.set('DEFAULT', str(key), str(value))
                return True
            except Exception:
                return False

    def register_data_service(self, service_name: str, service_instance: Any,


                              service_type: str = "loader") -> bool:
        """
        注册数据服务

        Args:
            service_name: 服务名称
            service_instance: 服务实例
            service_type: 服务类型

        Returns:
            是否注册成功
        """
        # 简化服务注册，不再依赖服务容器桥接器
        if not hasattr(self, '_registered_services'):
            self._registered_services = {}

        try:
            self._registered_services[service_name] = {
                'instance': service_instance,
                'type': service_type,
                'registered_at': datetime.now().isoformat()
            }
            self.logger.debug(f"数据服务已注册: {service_name}")
            return True
        except Exception as e:
            self.logger.error(f"注册数据服务失败: {service_name} - {e}")
            return False

    def get_data_service(self, service_name: str) -> Optional[Any]:
        """
        获取数据服务

        Args:
            service_name: 服务名称

        Returns:
            服务实例，如果不存在返回None
        """
        if hasattr(self, '_registered_services'):
            service_info = self._registered_services.get(service_name)
            if service_info:
                return service_info.get('instance')
        return None

    def get_service_health(self, service_name: str) -> Dict[str, Any]:
        """
        获取服务健康状态

        Args:
            service_name: 服务名称

        Returns:
            健康状态信息
        """
        service = self.get_data_service(service_name)
        if service:
            return {
                'status': 'healthy',
                'service_name': service_name,
                'service_type': type(service).__name__,
                'timestamp': datetime.now().isoformat()
            }
        else:
            return {
                'status': 'unhealthy',
                'service_name': service_name,
                'error': 'Service not found',
                'timestamp': datetime.now().isoformat()
            }

    def _init_infrastructure_services(self):
        """
        初始化基础设施服务 - 重构版本
        使用统一基础设施集成层的新接口
        """
        if not INFRASTRUCTURE_INTEGRATION_AVAILABLE:
            logger.warning("统一基础设施集成层不可用，使用标准logging")
            # 降级处理：使用标准组件
            self._init_fallback_services()
            return

        try:
            # 使用统一基础设施集成层获取适配器
            data_adapter = get_data_adapter()

            # 直接使用统一接口获取基础设施服务
            self.cache_manager = data_adapter.get_cache_manager()
            self.config_manager = data_adapter.get_config_manager()
            self.logger = data_adapter.get_logger()
            self.monitoring = data_adapter.get_monitoring()
            self.event_bus = data_adapter.get_event_bus()
            self.health_checker = data_adapter.get_health_checker()

            # 兼容性：保持旧的桥接器属性名（通过适配器获取）
            self.cache_bridge = self.cache_manager
            self.config_bridge = self.config_manager
            self.logging_bridge = self.logger
            self.service_bridge = None  # 服务容器已移除
            self.event_bus_bridge = self.event_bus
            self.health_bridge = self.health_checker

            logger.info("基础设施服务初始化完成（统一集成层）")

        except Exception as e:
            logger.error(f"基础设施服务初始化失败，使用降级方案: {e}")
            self._init_fallback_services()

    def _init_fallback_services(self):
        """
        初始化降级服务
        当统一基础设施集成层不可用时使用
        """
        logger.warning("使用降级基础设施服务")

        # 使用标准Python logging
        self.logger = logging.getLogger('data_manager')

        # 降级缓存管理器
        try:
            from .cache import CacheManager, CacheConfig
            cache_config = CacheConfig(
                max_size=1000,
                ttl=3600,
                enable_disk_cache=True,
                disk_cache_dir='cache',
                compression=False,
                encryption=False,
                enable_stats=True,
                cleanup_interval=300
            )
            self.cache_manager = CacheManager(cache_config)
            self.cache_bridge = self.cache_manager
        except Exception as e:
            logger.warning(f"降级缓存初始化失败: {e}")
            self.cache_manager = None
            self.cache_bridge = None

        # 降级配置管理器
        try:
            # 使用简单的配置管理器
            self.config_manager = self.config
            self.config_bridge = self.config
        except Exception as e:
            logger.warning(f"降级配置初始化失败: {e}")
            self.config_manager = None
            self.config_bridge = None

        # 其他服务设为None
        self.monitoring = None
        self.event_bus = None
        self.event_bus_bridge = None
        self.health_checker = None
        self.health_bridge = None
        self.service_bridge = None

    def _get_logger(self):
        """
        获取日志器
        现在logger在_init_infrastructure_services中已初始化

        Returns:
            日志器实例
        """
        if hasattr(self, 'logger') and self.logger:
            return self.logger
        else:
            # 降级：使用标准logging
            return logging.getLogger('data_manager')

    def _register_core_services(self):
        """
        注册核心服务到服务容器
        """
        try:
            # 注册缓存管理器
            if self.cache_manager:
                self.register_data_service("cache_manager", self.cache_manager, "cache")

            # 注册质量监控器
            if hasattr(self, 'quality_monitor') and self.quality_monitor:
                self.register_data_service("quality_monitor", self.quality_monitor, "monitoring")

            # 注册合规管理器
            if hasattr(self, 'compliance_manager') and self.compliance_manager:
                self.register_data_service(
                    "compliance_manager", self.compliance_manager, "compliance")

            # 注册验证器
            if hasattr(self, 'validator') and self.validator:
                self.register_data_service("validator", self.validator, "validation")

            # 注册注册表
            if hasattr(self, 'registry') and self.registry:
                self.register_data_service("registry", self.registry, "registry")

            self.logger.info("核心服务注册完成")

        except Exception as e:
            self.logger.error(f"注册核心服务失败: {e}")

    def health_check(self) -> Dict[str, Any]:
        """
        执行数据管理器健康检查

        Returns:
            健康检查结果
        """
        if hasattr(self, 'health_checker') and self.health_checker:
            try:
                return self.health_checker.health_check()
            except Exception as e:
                self.logger.warning(f"健康检查器调用失败: {e}")
                return self._perform_basic_health_check()
        else:
            # 降级健康检查
            return self._perform_basic_health_check()

    def _perform_basic_health_check(self) -> Dict[str, Any]:
        """
        执行基础健康检查

        Returns:
            健康检查结果
        """
        try:
            health_status = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'components': {}
            }

            # 检查缓存管理器
            if hasattr(self, 'cache_manager') and self.cache_manager:
                try:
                    cache_stats = self.cache_manager.get_stats() if hasattr(self.cache_manager, 'get_stats') else {}
                    health_status['components']['cache_manager'] = {
                        'status': 'healthy',
                        'stats': cache_stats
                    }
                except Exception as e:
                    health_status['components']['cache_manager'] = {
                        'status': 'unhealthy',
                        'error': str(e)
                    }
            else:
                health_status['components']['cache_manager'] = {
                    'status': 'unhealthy',
                    'error': 'Cache manager not available'
                }

            # 检查验证器
            if hasattr(self, 'validator') and self.validator:
                health_status['components']['validator'] = {
                    'status': 'healthy'
                }
            else:
                health_status['components']['validator'] = {
                    'status': 'unhealthy',
                    'error': 'Validator not available'
                }

            # 检查质量监控器
            if hasattr(self, 'quality_monitor') and self.quality_monitor:
                health_status['components']['quality_monitor'] = {
                    'status': 'healthy'
                }
            else:
                health_status['components']['quality_monitor'] = {
                    'status': 'unhealthy',
                    'error': 'Quality monitor not available'
                }

            # 检查合规管理器
            if hasattr(self, 'compliance_manager') and self.compliance_manager:
                health_status['components']['compliance_manager'] = {
                    'status': 'healthy'
                }
            else:
                health_status['components']['compliance_manager'] = {
                    'status': 'unhealthy',
                    'error': 'Compliance manager not available'
                }

            # 确定整体状态
            unhealthy_components = [comp for comp in health_status['components'].values()
                                    if comp.get('status') == 'unhealthy']
            if unhealthy_components:
                health_status['status'] = 'unhealthy'
                health_status['unhealthy_count'] = len(unhealthy_components)

            return health_status

        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def register_health_check(self, component_name: str, check_function: callable) -> bool:
        """
        注册健康检查

        Args:
            component_name: 组件名称
            check_function: 健康检查函数

        Returns:
            是否注册成功
        """
        if hasattr(self, 'health_checker') and self.health_checker:
            try:
                # 尝试注册健康检查
                if hasattr(self.health_checker, 'register_check'):
                    return self.health_checker.register_check(component_name, check_function)
                else:
                    self.logger.debug(f"健康检查器不支持注册检查: {component_name}")
                    return True
            except Exception as e:
                self.logger.warning(f"注册健康检查失败: {e}")
                return False
        else:
            # 降级处理：记录但不实际注册
            self.logger.debug(f"降级模式下注册健康检查: {component_name}")
            return True

    def get_health_history(self) -> List[Dict[str, Any]]:
        """
        获取健康检查历史

        Returns:
            健康检查历史记录
        """
        if hasattr(self, 'health_checker') and self.health_checker:
            try:
                if hasattr(self.health_checker, 'get_history'):
                    return self.health_checker.get_history()
                else:
                    return []
            except Exception as e:
                self.logger.warning(f"获取健康检查历史失败: {e}")
                return []
        else:
            # 降级处理，返回空列表
            return []

    def _init_config(self, config_path: Optional[Union[str, Path]], config_dict: Optional[dict]) -> configparser.ConfigParser:
        """
        初始化配置

        Args:
            config_path: 配置文件路径
            config_dict: 配置字典

        Returns:
            configparser.ConfigParser: 配置对象
        """
        config = configparser.ConfigParser()

        # 读取默认配置

        default_config_path = Path(__file__).parent / "data_config.ini"
        if default_config_path.exists():
            config.read(default_config_path, encoding='utf - 8')

        # 读取用户配置
        if config_path:
            if isinstance(config_path, str):
                config_path = Path(config_path)
            if config_path.exists():
                config.read(config_path, encoding='utf - 8')
            else:
                # 如果配置文件不存在，抛出FileNotFoundError
                raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # 应用配置字典
        if config_dict:
            for section, section_dict in config_dict.items():
                if not config.has_section(section):
                    config.add_section(section)
                for key, value in section_dict.items():
                    config.set(section, key, str(value))

        return config

    def _init_loaders(self) -> None:
        """
        初始化数据加载器
        """
        # 创建通用加载器适配器

        def create_adapter(loader_class, source_type: str, requires_symbol: bool = True):

            class GenericLoaderAdapter(BaseDataLoader):

                def __init__(self, config: Dict[str, Any]):

                    super().__init__(config)
                    self._loader = loader_class.create_from_config(config)
                    self.source_type = source_type
                    self.requires_symbol = requires_symbol

                def load(self, start_date: str, end_date: str, frequency: str, **kwargs) -> IDataModel:

                    if self.requires_symbol and 'symbols' not in kwargs:
                        raise ValueError(f"{self.source_type} loader requires symbols parameter")

                    symbols = kwargs.get('symbols', [])
                    if not symbols:
                        raise ValueError("symbols parameter is required")

                    # 对于股票数据，需要处理多个股票代码
                    if self.source_type == 'stock':
                        all_data = []
                        for symbol in symbols:
                            try:
                                symbol_data = self._loader.load_data(
                                    symbol=symbol,
                                    start_date=start_date,
                                    end_date=end_date,
                                    adjust="hfq"
                                )
                                if symbol_data is not None and not symbol_data.empty:
                                    symbol_data['symbol'] = symbol
                                    all_data.append(symbol_data)
                            except Exception as e:
                                self.logger.warning(f"Failed to load data for symbol {symbol}: {e}")
                                continue

                        if not all_data:
                            raise ValueError("No data loaded for any symbol")

                        # 合并所有股票数据
                        data = pd.concat(all_data, ignore_index=True)
                    else:
                        # 其他数据类型使用原始方法
                        data = self._loader.load_data(
                            symbols=symbols,
                            start_date=start_date,
                            end_date=end_date,
                            frequency=frequency
                        )

                    return DataModel(data, frequency)

                def get_required_config_fields(self) -> list:

                    return self._loader.get_required_config_fields()

                def get_metadata(self) -> Dict[str, Any]:
                    """获取加载器元数据"""
                    return {
                        'source_type': self.source_type,
                        'requires_symbol': self.requires_symbol,
                        'loader_class': self._loader.__class__.__name__
                    }

            return GenericLoaderAdapter

        # 注册默认加载器
        try:
            from .loader.stock_loader import StockDataLoader
            self.register_loader_class('stock', create_adapter(StockDataLoader, 'stock'))
        except ImportError:
            self.logger.warning("StockDataLoader not available")

        try:
            from .loaders.index_loader import IndexDataLoader
            self.register_loader_class('index', create_adapter(IndexDataLoader, 'index'))
        except ImportError:
            self.logger.warning("IndexDataLoader not available")

        try:
            from .loaders.news_loader import NewsDataLoader
            self.register_loader_class('news', create_adapter(
                NewsDataLoader, 'news', requires_symbol=False))
        except ImportError:
            self.logger.warning("NewsDataLoader not available")

        try:
            from .loaders.financial_loader import FinancialDataLoader
            self.register_loader_class('financial', create_adapter(
                FinancialDataLoader, 'financial'))
        except ImportError:
            self.logger.warning("FinancialDataLoader not available")

    def register_loader(self, name: str, loader: BaseDataLoader) -> None:
        """
        注册数据加载器

        Args:
            name: 加载器名称
            loader: 加载器实例
        """
        self.registry.register(name, loader)

    def register_loader_class(self, name: str, loader_class: Type[BaseDataLoader]) -> None:
        """
        注册数据加载器类

        Args:
            name: 加载器名称
            loader_class: 加载器类
        """
        self.registry.register_class(name, loader_class)

    async def load_data(self, data_type: str, start_date: str, end_date: str, frequency: str = "1d", compliance_policy_id: str = None, privacy_level: str = None, **kwargs) -> IDataModel:
        """
        加载数据

        Args:
            data_type: 数据类型
            start_date: 开始日期
            end_date: 结束日期
            frequency: 数据频率
            **kwargs: 其他参数

        Returns:
            IDataModel: 数据模型
        """
        # 生成缓存键
        cache_key = self._generate_cache_key(data_type, start_date, end_date, frequency, **kwargs)

        # 尝试从缓存获取
        cached_data = self.cache_manager.get(cache_key)
        if cached_data is not None:
            self.logger.info(f"Data loaded from cache: {cache_key}")
            return cached_data

        # 获取加载器
        try:
            loader = self.registry.create_loader(data_type, {})
        except ValueError as e:
            raise DataLoaderError(f"Loader not found for data type: {data_type} - {str(e)}")

        # 加载数据
        try:
            data_model = loader.load(start_date, end_date, frequency, **kwargs)

            # 验证数据
            validation_result = self.validator.validate_data_model(data_model)
            if not validation_result['is_valid']:
                self.logger.warning(f"Data validation failed: {validation_result['errors']}")

            # 质量监控
            try:
                self.quality_monitor.track_metrics(data_model, data_type)
            except Exception as e:
                self.logger.warning(f"Quality monitoring failed: {e}")

            # 合规性校验（如指定策略）
            if compliance_policy_id:
                compliance_result = self.compliance_manager.check_compliance(
                    data_model.data, compliance_policy_id)
                if not compliance_result.get("compliance", True):
                    self.logger.warning(
                        f"Data compliance check failed: {compliance_result.get('issues')}")
            # 隐私保护（如指定级别）
            if privacy_level:
                data_model.data = self.compliance_manager.protect_privacy(
                    data_model.data, privacy_level)

            # 缓存数据
            try:
                self.cache_manager.set(cache_key, data_model, ttl=3600)  # 1小时过期
            except Exception as e:
                self.logger.error(f"Failed to cache data: {e}")
                raise DataLoaderError(f"Cache write failed: {e}")

            # 记录数据血缘
            self._record_data_lineage(data_type, data_model, start_date, end_date, **kwargs)

            self.logger.info(f"Data loaded successfully: {data_type}")
            return data_model

        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise DataLoaderError(f"Data loading failed: {e}")

    async def load_multi_source(self, stock_symbols: List[str], index_symbols: List[str],
                                start: str, end: str, frequency: str = "1d") -> Dict[str, IDataModel]:
        """
        加载多源数据

        Args:
            stock_symbols: 股票代码列表
            index_symbols: 指数代码列表
            start: 开始日期
            end: 结束日期
            frequency: 数据频率

        Returns:
            Dict[str, IDataModel]: 多源数据字典
        """
        data_models = {}

        # 并行加载数据
        tasks = {}

        if stock_symbols:
            tasks['market'] = self._load_stock_data(stock_symbols, start, end, frequency)

        if index_symbols:
            tasks['index'] = self._load_index_data(index_symbols, start, end, frequency)

        # 等待所有任务完成
        if tasks:
            try:
                results = await asyncio.gather(*tasks.values(), return_exceptions=True)
                for i, (data_type, result) in enumerate(tasks.items()):
                    if isinstance(results[i], Exception):
                        self.logger.error(f"Failed to load {data_type} data: {results[i]}")
                        raise DataLoaderError(f"Multi - source data loading failed: {results[i]}")
                    data_models[data_type] = results[i]
            except Exception as e:
                self.logger.error(f"Multi - source data loading failed: {e}")
                raise DataLoaderError(f"Multi - source data loading failed: {e}")

        # 记录数据版本
        self._record_data_version(data_models, start, end)

        return data_models

    async def _load_stock_data(self, symbols: List[str], start: str, end: str, frequency: str) -> IDataModel:
        """
        加载股票数据

        Args:
            symbols: 股票代码列表
            start: 开始日期
            end: 结束日期
            frequency: 数据频率

        Returns:
            IDataModel: 股票数据模型
        """
        return await self.load_data('stock', start, end, frequency, symbols=symbols)

    async def _load_index_data(self, symbols: List[str], start: str, end: str, frequency: str) -> IDataModel:
        """
        加载指数数据

        Args:
            symbols: 指数代码列表
            start: 开始日期
            end: 结束日期
            frequency: 数据频率

        Returns:
            IDataModel: 指数数据模型
        """
        return await self.load_data('index', start, end, frequency, symbols=symbols)

    async def _load_news_data(self, start: str, end: str, frequency: str) -> IDataModel:
        """
        加载新闻数据

        Args:
            start: 开始日期
            end: 结束日期
            frequency: 数据频率

        Returns:
            IDataModel: 新闻数据模型
        """
        return await self.load_data('news', start, end, frequency)

    async def _load_financial_data(self, symbols: List[str], start: str, end: str, frequency: str) -> IDataModel:
        """
        加载财务数据

        Args:
            symbols: 股票代码列表
            start: 开始日期
            end: 结束日期
            frequency: 数据频率

        Returns:
            IDataModel: 财务数据模型
        """
        return await self.load_data('financial', start, end, frequency, symbols=symbols)

    def _generate_cache_key(self, data_type: str, start_date: str, end_date: str, frequency: str, **kwargs) -> str:
        """
        生成缓存键

        Args:
            data_type: 数据类型
            start_date: 开始日期
            end_date: 结束日期
            frequency: 数据频率
            **kwargs: 其他参数

        Returns:
            str: 缓存键
        """
        key_parts = [data_type, start_date, end_date, frequency]

        # 添加其他参数
        for key, value in sorted(kwargs.items()):
            if isinstance(value, list):
                key_parts.append(f"{key}_{'_'.join(map(str, value))}")
            else:
                key_parts.append(f"{key}_{value}")

        return "_".join(key_parts)

    def _record_data_lineage(self, data_type: str, data_model: IDataModel, start_date: str, end_date: str, **kwargs) -> None:
        """
        记录数据血缘

        Args:
            data_type: 数据类型
            data_model: 数据模型
            start_date: 开始日期
            end_date: 结束日期
            **kwargs: 其他参数
        """
        frequency = None
        metadata_info = {}

        if data_model is not None:
            try:
                frequency = data_model.get_frequency()
            except Exception:
                frequency = None

            try:
                metadata_info = data_model.get_metadata()
            except Exception:
                metadata_info = {}

        if frequency is None:
            frequency = kwargs.get('frequency')

        if not metadata_info:
            metadata_info = {
                key: value
                for key, value in kwargs.items()
                if key not in {'data_type', 'start_date', 'end_date'}
            }

        lineage_info = {
            'data_type': data_type,
            'start_date': start_date,
            'end_date': end_date,
            'frequency': frequency,
            'metadata': metadata_info,
            'parameters': kwargs,
            'timestamp': datetime.now().isoformat()
        }

        # 这里可以保存到数据库或文件
        self.logger.debug(f"Data lineage recorded: {lineage_info}")

    def _record_data_version(self, data_models: Dict[str, IDataModel], start: str, end: str) -> None:
        """
        记录数据版本

        Args:
            data_models: 数据模型字典
            start: 开始日期
            end: 结束日期
        """
        version_info = {
            'start_date': start,
            'end_date': end,
            'data_sources': list(data_models.keys()),
            'source_versions': self._get_data_source_version(),
            'processing_params': self._get_data_processing_params(),
            'timestamp': datetime.now().isoformat()
        }

        # 这里可以保存到数据库或文件
        self.logger.debug(f"Data version recorded: {version_info}")

    def _get_data_source_version(self) -> Dict:
        """
        获取数据源版本信息

        Returns:
            Dict: 数据源版本信息
        """
        return {
            'stock_data': 'v1.0',
            'index_data': 'v1.0',
            'news_data': 'v1.0',
            'financial_data': 'v1.0'
        }

    def _get_data_processing_params(self) -> Dict:
        """
        获取数据处理参数

        Returns:
            Dict: 数据处理参数
        """
        return {
            'validation_enabled': True,
            'cache_enabled': True,
            'quality_monitoring': True,
            'compression': False,
            'encryption': False
        }

    def validate_all_configs(self) -> bool:
        """
        验证所有配置

        Returns:
            bool: 配置是否有效
        """
        try:
            # 验证注册中心
            if not hasattr(self.registry, 'list_registered_loaders') or not callable(getattr(self.registry, 'list_registered_loaders', None)):
                return False

            # 实际调用方法来验证它们是否正常工作
            try:
                self.registry.list_registered_loaders()
            except Exception:
                return False

            # 验证验证器
            if not hasattr(self.validator, 'validate_data_model') or not callable(getattr(self.validator, 'validate_data_model', None)):
                return False

            # 验证缓存管理器
            if not hasattr(self.cache_manager, 'get') or not callable(getattr(self.cache_manager, 'get', None)):
                return False

            return True
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    def track_data_lineage(self, data_type: str = None) -> Dict:
        """
        跟踪数据血缘

        Args:
            data_type: 数据类型，如果为None则跟踪所有类型

        Returns:
            Dict: 血缘信息
        """
        # 这里应该从数据库或文件中读取血缘信息
        return {
            'data_type': data_type,
            'lineage_info': 'placeholder'
        }

    def clean_expired_cache(self) -> int:
        """
        清理过期缓存

        Returns:
            int: 清理的缓存项数量
        """
        return self.cache_manager._cleanup_expired()

    def get_cache_stats(self) -> Dict:
        """
        获取缓存统计信息

        Returns:
            Dict: 缓存统计信息
        """
        return self.cache_manager.get_stats()

    def shutdown(self) -> None:
        """
        关闭数据管理器
        """
        self.thread_pool.shutdown(wait=True)
        self.logger.info("DataManager shutdown completed")

    def store_data(self, data: Any, storage_type: str = "database", metadata: Optional[Dict[str, Any]] = None) -> Any:
        """
        存储数据

        Args:
            data: 要存储的数据
            storage_type: 存储类型，如"database"、"file"、"cache"等
            metadata: 元数据信息

        Returns:
            Dict[str, Any]: 存储结果
        """
        try:
            if not hasattr(self, "_data_store"):
                self._data_store = {}
            if not hasattr(self, "_metadata_store"):
                self._metadata_store = {}
            if not hasattr(self, "_user_metadata_store"):
                self._user_metadata_store = {}

            key_provided = False
            actual_data = data
            actual_metadata = metadata or {}
            actual_storage_type = storage_type
            key = actual_metadata.get('key')

            # 兼容测试场景：store_data(key, data, metadata)
            if not isinstance(storage_type, str):
                key_provided = True
                key = str(data)
                actual_data = storage_type
                actual_metadata = metadata or {}
                actual_storage_type = "database"
            else:
                if key is None:
                    if isinstance(actual_metadata, dict) and 'symbol' in actual_metadata and 'data_type' in actual_metadata:
                        key = f"{actual_metadata['symbol']}_{actual_metadata['data_type']}"
                    else:
                        key = f"data_{hash(str(actual_data))}"

            # 记录存储操作
            self.logger.info(f"开始存储数据，类型: {actual_storage_type}")

            # 根据存储类型选择不同的存储策略
            if actual_storage_type == "cache":
                # 存储到缓存
                cache_key = f"stored_data_{hash(str(actual_data))}"
                if self.cache_manager:
                    self.cache_manager.set(cache_key, actual_data, ttl=3600)
                result = {"cache_key": cache_key, "status": "cached"}
            elif actual_storage_type == "file":
                # 存储到文件（这里只是示例）
                result = {"file_path": "placeholder_path", "status": "saved"}
            else:
                # 默认存储到数据库（这里只是示例）
                result = {"database_id": "placeholder_id", "status": "saved"}

            # 在降级模式下维护内存数据存储，支持测试场景
            self._data_store[key] = actual_data
            meta_copy = dict(actual_metadata)
            meta_copy.setdefault('key', key)
            meta_copy['storage_type'] = actual_storage_type
            meta_copy['timestamp'] = datetime.now()
            meta_copy.setdefault('version', 1)
            self._metadata_store[key] = meta_copy
            self._user_metadata_store[key] = dict(actual_metadata)

            # 记录数据血缘
            if actual_metadata:
                data_type_value = None
                if isinstance(actual_metadata, dict):
                    data_type_value = actual_metadata.get('data_type')
                elif hasattr(actual_metadata, 'get'):
                    data_type_value = actual_metadata.get('data_type')

                if data_type_value:
                    if hasattr(actual_metadata, 'items'):
                        lineage_metadata = dict(actual_metadata.items())
                    else:
                        lineage_metadata = dict(self._user_metadata_store.get(key, {}))
                    lineage_metadata.pop('data_type', None)
                    start_date_value = None
                    end_date_value = None
                    if hasattr(actual_metadata, 'get'):
                        start_date_value = actual_metadata.get('start_date')
                        end_date_value = actual_metadata.get('end_date')
                    if start_date_value is None:
                        start_date_value = lineage_metadata.get('start_date', '')
                    if end_date_value is None:
                        end_date_value = lineage_metadata.get('end_date', '')
                    lineage_metadata.pop('start_date', None)
                    lineage_metadata.pop('end_date', None)
                    self._record_data_lineage(
                        data_type=data_type_value,
                        data_model=None,  # 这里可以创建DataModel
                        start_date=start_date_value or '',
                        end_date=end_date_value or '',
                        **lineage_metadata
                    )

            self.logger.info(f"数据存储完成，结果: {result}")
            return True if key_provided else result

        except Exception as e:
            self.logger.error(f"数据存储失败: {e}")
            raise e

    def retrieve_data(self, key: str) -> Any:
        """
        按键检索数据
        """
        return self._data_store.get(key)

    def has_data(self, key: str) -> bool:
        """
        判断指定键是否存在数据
        """
        return key in self._data_store

    def delete_data(self, key: str) -> bool:
        """
        删除指定键的数据
        """
        existed = key in self._data_store
        self._data_store.pop(key, None)
        self._metadata_store.pop(key, None)
        self._user_metadata_store.pop(key, None)
        return existed

    def update_data(self, key: str, new_data: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        更新存储的数据
        """
        if key not in self._data_store:
            return False
        self._data_store[key] = new_data
        meta = self._metadata_store.get(key, {})
        if metadata:
            meta.update(metadata)
            user_meta = self._user_metadata_store.get(key, {})
            user_meta.update(metadata)
            self._user_metadata_store[key] = user_meta
        meta['timestamp'] = datetime.now()
        meta['version'] = meta.get('version', 1) + 1
        self._metadata_store[key] = meta
        return True

    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """
        获取指定键的元数据
        """
        meta = self._user_metadata_store.get(key)
        if meta is None:
            return None
        return dict(meta)

    def list_data_keys(self, pattern: Optional[str] = None) -> List[str]:
        """
        列出已存储的数据键
        """
        keys = list(self._data_store.keys())
        if pattern:
            keys = [k for k in keys if pattern in k]
        return keys

    def get_stats(self) -> Dict[str, Any]:
        """
        获取数据存储统计信息
        """
        if not self._data_store:
            return {
                'total_items': 0,
                'total_size': 0,
                'oldest_item': None,
                'newest_item': None
            }

        timestamps = [
            meta.get('timestamp')
            for meta in self._metadata_store.values()
            if isinstance(meta.get('timestamp'), datetime)
        ]
        oldest = min(timestamps) if timestamps else None
        newest = max(timestamps) if timestamps else None

        total_size = 0
        for value in self._data_store.values():
            try:
                if isinstance(value, pd.DataFrame):
                    total_size += value.memory_usage(deep=True).sum()
                else:
                    total_size += len(str(value))
            except Exception:
                total_size += len(str(value))

        return {
            'total_items': len(self._data_store),
            'total_size': int(total_size),
            'oldest_item': oldest,
            'newest_item': newest
        }

    def validate_data(self, key: str, rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        执行数据质量验证
        """
        if key not in self._data_store:
            return {'valid': False, 'issues': ['数据不存在']}

        data = self._data_store[key]
        issues: List[str] = []

        if isinstance(data, pd.DataFrame):
            if data.empty:
                issues.append('数据为空')
            if not data.empty and data.isnull().all().all():
                issues.append('数据全部为null')

        if rules:
            for rule_name, rule_func in rules.items():
                try:
                    if not rule_func(data):
                        issues.append(f'规则 {rule_name} 验证失败')
                except Exception as exc:
                    issues.append(f'规则 {rule_name} 执行异常: {exc}')

        return {
            'valid': len(issues) == 0,
            'issues': issues
        }

    def check_compliance(self, key: str, policies: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        执行合规检查
        """
        if key not in self._data_store:
            return {'compliant': False, 'violations': ['数据不存在']}

        data = self._data_store[key]
        violations: List[str] = []

        if isinstance(data, pd.DataFrame):
            sensitive_columns = ['ssn', 'password', 'credit_card']
            for col in data.columns:
                if any(sensitive in col.lower() for sensitive in sensitive_columns):
                    violations.append(f'检测到敏感列: {col}')
            if len(data) > 10000:
                violations.append('数据量过大，可能需要脱敏')

        if policies:
            for policy_name, policy_func in policies.items():
                try:
                    if not policy_func(data):
                        violations.append(f'策略 {policy_name} 违反')
                except Exception as exc:
                    violations.append(f'策略 {policy_name} 执行异常: {exc}')

        return {
            'compliant': len(violations) == 0,
            'violations': violations
        }
