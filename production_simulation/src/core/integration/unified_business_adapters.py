#!/usr/bin/env python3
"""
统一业务适配器模块

整合所有business_adapters实现，提供统一的业务层适配器框架
基于BaseAdapter基类重构，消除代码重复

创建时间: 2025-11-03
版本: 2.0
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Type
import logging
from enum import Enum
from datetime import datetime
import threading

# 导入新的BaseAdapter基类
from src.core.foundation.base_adapter import BaseAdapter, AdapterStatus


class BusinessLayerType(Enum):
    """业务层类型枚举"""
    DATA = "data"
    FEATURES = "features"
    TRADING = "trading"
    RISK = "risk"
    MODELS = "models"
    ML = "ml"
    STRATEGY = "strategy"
    ENGINE = "engine"
    HEALTH = "health"


class IBusinessAdapter(ABC):
    """业务层适配器接口"""
    
    @property
    @abstractmethod
    def layer_type(self) -> BusinessLayerType:
        """获取业务层类型"""
        pass
    
    @abstractmethod
    def get_infrastructure_services(self) -> Dict[str, Any]:
        """获取基础设施服务"""
        pass
    
    @abstractmethod
    def get_service_bridge(self, service_name: str) -> Optional[Any]:
        """获取服务桥接器"""
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        pass


class UnifiedBusinessAdapter(BaseAdapter[Any, Any], IBusinessAdapter):
    """
    统一业务适配器实现
    
    基于BaseAdapter重构，提供：
    - 统一的基础设施服务访问
    - 自动化的健康检查
    - 服务降级支持
    - 性能监控
    """
    
    def __init__(
        self,
        layer_type: BusinessLayerType,
        config: Optional[Dict[str, Any]] = None,
        enable_cache: bool = True
    ):
        """
        初始化业务适配器
        
        Args:
            layer_type: 业务层类型
            config: 配置参数
            enable_cache: 是否启用缓存
        """
        super().__init__(
            name=f"{layer_type.value}_adapter",
            config=config,
            enable_cache=enable_cache
        )
        
        self._layer_type = layer_type
        self._service_bridges: Dict[str, Any] = {}
        self._infrastructure_services: Dict[str, Any] = {}
        self._initialized = False
        self._lock = threading.RLock()
        
        # 初始化基础设施服务
        self._init_infrastructure_services()
    
    @property
    def layer_type(self) -> BusinessLayerType:
        """获取业务层类型"""
        return self._layer_type
    
    def _init_infrastructure_services(self):
        """初始化基础设施服务映射"""
        try:
            # 尝试导入完整的基础设施服务
            from src.infrastructure.config.core.unified_manager import UnifiedConfigManager
            from src.infrastructure.cache.core.cache_manager import UnifiedCacheManager
            from src.infrastructure.logging.core.unified_logger import get_unified_logger
            from src.infrastructure.monitoring.continuous_monitoring_system import ContinuousMonitoringSystem
            from src.infrastructure.health.components.enhanced_health_checker import EnhancedHealthChecker
            
            self._infrastructure_services = {
                'config_manager': UnifiedConfigManager(),
                'cache_manager': UnifiedCacheManager(),
                'logger': get_unified_logger(f"{self._layer_type.value}_layer"),
                'monitoring': ContinuousMonitoringSystem(),
                'health_checker': EnhancedHealthChecker()
            }
            
            self._logger.info(f"{self._layer_type.value}层基础设施服务初始化完成")
            self._initialized = True
            
        except ImportError as e:
            self._logger.warning(f"{self._layer_type.value}层基础设施服务导入失败: {e}，使用降级服务")
            self._init_fallback_services()
    
    def _init_fallback_services(self):
        """初始化降级服务"""
        self._infrastructure_services = {
            'config_manager': None,
            'cache_manager': None,
            'logger': logging.getLogger(f"{self._layer_type.value}_layer"),
            'monitoring': None,
            'health_checker': None
        }
        self._initialized = True
        self._logger.warning(f"{self._layer_type.value}层使用降级服务模式")
    
    def get_infrastructure_services(self) -> Dict[str, Any]:
        """获取基础设施服务"""
        return self._infrastructure_services.copy()
    
    def get_service(self, service_name: str) -> Optional[Any]:
        """
        获取特定的基础设施服务
        
        Args:
            service_name: 服务名称（config_manager, cache_manager, logger, monitoring, health_checker）
            
        Returns:
            服务实例，不存在返回None
        """
        return self._infrastructure_services.get(service_name)
    
    def register_service_bridge(self, service_name: str, bridge: Any):
        """
        注册服务桥接器
        
        Args:
            service_name: 服务名称
            bridge: 桥接器实例
        """
        with self._lock:
            self._service_bridges[service_name] = bridge
            self._logger.info(f"注册服务桥接器: {service_name}")
    
    def get_service_bridge(self, service_name: str) -> Optional[Any]:
        """获取服务桥接器"""
        return self._service_bridges.get(service_name)
    
    def health_check(self) -> Dict[str, Any]:
        """
        执行健康检查
        
        Returns:
            健康检查结果
        """
        health_status = {
            'layer_type': self._layer_type.value,
            'adapter_name': self.name,
            'status': self._status.value,
            'initialized': self._initialized,
            'timestamp': datetime.now().isoformat(),
            'services': {},
            'bridges': {}
        }
        
        # 检查基础设施服务状态
        for service_name, service in self._infrastructure_services.items():
            if service is None:
                health_status['services'][service_name] = 'unavailable'
            elif hasattr(service, 'health_check'):
                try:
                    health_status['services'][service_name] = service.health_check()
                except Exception as e:
                    health_status['services'][service_name] = f'error: {e}'
            else:
                health_status['services'][service_name] = 'active'
        
        # 检查服务桥接器状态
        for bridge_name, bridge in self._service_bridges.items():
            if hasattr(bridge, 'health_check'):
                try:
                    health_status['bridges'][bridge_name] = bridge.health_check()
                except Exception as e:
                    health_status['bridges'][bridge_name] = f'error: {e}'
            else:
                health_status['bridges'][bridge_name] = 'active'
        
        # 添加适配器统计信息
        health_status['stats'] = self.get_stats()
        
        return health_status
    
    def _do_adapt(self, data: Any) -> Any:
        """
        执行适配逻辑
        
        默认实现：直接返回数据，子类可以覆盖实现特定的适配逻辑
        
        Args:
            data: 输入数据
            
        Returns:
            适配后的数据
        """
        # 默认实现：不做转换
        return data
    
    def validate_input(self, data: Any) -> bool:
        """
        验证输入数据
        
        Args:
            data: 输入数据
            
        Returns:
            数据是否有效
        """
        # 基本验证：数据不为None
        if data is None:
            self._logger.warning(f"{self._layer_type.value}层适配器收到None数据")
            return False
        return True
    
    def _preprocess(self, data: Any) -> Any:
        """数据预处理"""
        # 可以在这里添加通用的预处理逻辑
        # 例如：数据清洗、格式标准化等
        return data
    
    def _postprocess(self, data: Any) -> Any:
        """数据后处理"""
        # 可以在这里添加通用的后处理逻辑
        # 例如：添加元数据、日志记录等
        return data


class BusinessAdapterFactory:
    """
    业务适配器工厂
    
    负责创建和管理不同业务层的适配器实例
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._adapters: Dict[BusinessLayerType, UnifiedBusinessAdapter] = {}
            self._logger = logging.getLogger(self.__class__.__name__)
            self._initialized = True
    
    def get_adapter(
        self,
        layer_type: BusinessLayerType,
        config: Optional[Dict[str, Any]] = None,
        create_if_not_exists: bool = True
    ) -> Optional[UnifiedBusinessAdapter]:
        """
        获取业务层适配器
        
        Args:
            layer_type: 业务层类型
            config: 配置参数
            create_if_not_exists: 如果不存在是否创建
            
        Returns:
            适配器实例
        """
        if layer_type not in self._adapters and create_if_not_exists:
            self._adapters[layer_type] = UnifiedBusinessAdapter(
                layer_type=layer_type,
                config=config
            )
            self._logger.info(f"创建{layer_type.value}层适配器")
        
        return self._adapters.get(layer_type)
    
    def register_adapter(self, adapter: UnifiedBusinessAdapter):
        """
        注册适配器
        
        Args:
            adapter: 适配器实例
        """
        self._adapters[adapter.layer_type] = adapter
        self._logger.info(f"注册{adapter.layer_type.value}层适配器")
    
    def get_all_adapters(self) -> Dict[BusinessLayerType, UnifiedBusinessAdapter]:
        """获取所有适配器"""
        return self._adapters.copy()
    
    def health_check_all(self) -> Dict[str, Any]:
        """对所有适配器执行健康检查"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_adapters': len(self._adapters),
            'adapters': {}
        }
        
        for layer_type, adapter in self._adapters.items():
            try:
                results['adapters'][layer_type.value] = adapter.health_check()
            except Exception as e:
                results['adapters'][layer_type.value] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return results


# 便捷函数
def get_business_adapter(
    layer_type: BusinessLayerType,
    config: Optional[Dict[str, Any]] = None
) -> UnifiedBusinessAdapter:
    """
    获取业务层适配器的便捷函数
    
    Args:
        layer_type: 业务层类型
        config: 配置参数
        
    Returns:
        适配器实例
    """
    factory = BusinessAdapterFactory()
    return factory.get_adapter(layer_type, config)


__all__ = [
    'BusinessLayerType',
    'IBusinessAdapter',
    'UnifiedBusinessAdapter',
    'BusinessAdapterFactory',
    'get_business_adapter'
]

