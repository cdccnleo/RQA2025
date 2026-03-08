"""
接口统一层

提供各层之间的标准化接口和协议，确保系统集成的模块化和可扩展性。
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# 注意: ICoreComponent已迁移到interfaces.py中的统一接口
# 此处保留向后兼容性，建议使用interfaces.ICoreComponent


class ICoreComponentCompat(ABC):

    """核心组件接口基类 (兼容性版本)

    此接口已迁移到interfaces.py中的ICoreComponent
    新代码请使用: from .interfaces import ICoreComponent
    """

    @abstractmethod
    def initialize(self) -> bool:
        """初始化组件"""

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""

    @abstractmethod
    def validate_config(self) -> bool:
        """验证配置"""


# 向后兼容性别名
ICoreComponent = ICoreComponentCompat


class SystemLayerInterfaceManager:

    """系统层接口管理器，负责单层的接口标准化"""

    def __init__(self, layer_name: str):

        self.layer_name = layer_name
        self.methods: Dict[str, Callable] = {}
        self.interfaces: Dict[str, Any] = {}

    def register_method(self, method_name: str, method_func: Callable) -> None:
        """注册层方法"""
        self.methods[method_name] = method_func
        logger.info(f"注册 {self.layer_name} 层方法: {method_name}")

    def register_interface(self, interface_name: str, interface_obj: Any) -> None:
        """注册层接口"""
        self.interfaces[interface_name] = interface_obj
        logger.info(f"注册 {self.layer_name} 层接口: {interface_name}")

    def get_method(self, method_name: str) -> Optional[Callable]:
        """获取层方法"""
        return self.methods.get(method_name)

    def get_interface(self, interface_name: str) -> Optional[Any]:
        """获取层接口"""
        return self.interfaces.get(interface_name)

    def list_methods(self) -> List[str]:
        """列出所有方法"""
        return list(self.methods.keys())

    def list_interfaces(self) -> List[str]:
        """列出所有接口"""
        return list(self.interfaces.keys())

    def validate_interface(self) -> Dict[str, Any]:
        """验证接口完整性"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }

        # 检查必要的方法是否存在
        required_methods = self._get_required_methods()
        for method in required_methods:
            if method not in self.methods:
                validation_result['valid'] = False
                validation_result['errors'].append(f"缺少必要方法: {method}")

        return validation_result

    def _get_required_methods(self) -> List[str]:
        """获取必要的方法列表"""
        # 根据层名称返回必要的方法
        required_methods_map = {
            'data': ['load_data', 'validate_data', 'process_data'],
            'features': ['extract_features', 'select_features', 'engineer_features'],
            'models': ['train_model', 'predict', 'evaluate_model'],
            'trading': ['execute_trade', 'check_risk', 'manage_portfolio'],
            'services': ['provide_service', 'validate_service', 'monitor_service'],
            'application': ['start_application', 'stop_application', 'get_status']
        }
        return required_methods_map.get(self.layer_name, [])


class SystemIntegrationManager:

    """系统集成管理器，负责各层之间的接口统一"""

    def __init__(self):

        self.layer_interfaces: Dict[str, SystemLayerInterfaceManager] = {}
        self.integration_status = 'stopped'
        self.integration_config: Dict[str, Any] = {}

    def register_layer_interface(self, layer_name: str, layer_interface: SystemLayerInterfaceManager) -> None:
        """注册层接口"""
        self.layer_interfaces[layer_name] = layer_interface
        logger.info(f"注册层接口: {layer_name}")

    def get_layer_interface(self, layer_name: str) -> Optional[SystemLayerInterfaceManager]:
        """获取层接口"""
        return self.layer_interfaces.get(layer_name)

    def validate_interfaces(self) -> Dict[str, Any]:
        """验证接口兼容性"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'layer_results': {}
        }

        # 验证每个层的接口
        for layer_name, layer_interface in self.layer_interfaces.items():
            layer_result = layer_interface.validate_interface()
            validation_result['layer_results'][layer_name] = layer_result

            if not layer_result['valid']:
                validation_result['valid'] = False
                validation_result['errors'].extend([
                    f"{layer_name}: {error}" for error in layer_result['errors']
                ])

        # 验证层间依赖关系
        dependency_errors = self._validate_layer_dependencies()
        if dependency_errors:
            validation_result['valid'] = False
            validation_result['errors'].extend(dependency_errors)

        return validation_result

    def _validate_layer_dependencies(self) -> List[str]:
        """验证层间依赖关系"""
        errors = []

        # 定义层依赖关系
        layer_dependencies = {
            'features': ['data'],
            'models': ['features'],
            'trading': ['models'],
            'services': ['trading'],
            'application': ['services']
        }

        for layer_name, dependencies in layer_dependencies.items():
            if layer_name in self.layer_interfaces:
                for dependency in dependencies:
                    if dependency not in self.layer_interfaces:
                        errors.append(f"{layer_name} 层缺少依赖: {dependency}")

        return errors

    def get_unified_interface(self) -> Dict[str, Any]:
        """获取统一接口"""
        unified_interface = {}

        for layer_name, layer_interface in self.layer_interfaces.items():
            unified_interface[layer_name] = {
                'methods': layer_interface.list_methods(),
                'interfaces': layer_interface.list_interfaces()
            }

        return unified_interface

    def test_connections(self) -> Dict[str, Any]:
        """测试接口连接"""
        connection_results = {
            'success': True,
            'layer_results': {},
            'errors': []
        }

        for layer_name, layer_interface in self.layer_interfaces.items():
            try:
                # 测试层接口的基本功能
                layer_result = self._test_layer_connection(layer_name, layer_interface)
                connection_results['layer_results'][layer_name] = layer_result

                if not layer_result['success']:
                    connection_results['success'] = False
                    connection_results['errors'].extend(layer_result['errors'])

            except Exception as e:
                connection_results['success'] = False
                connection_results['errors'].append(f"{layer_name} 连接测试失败: {str(e)}")

        return connection_results

    def _test_layer_connection(self, layer_name: str, layer_interface: SystemLayerInterfaceManager) -> Dict[str, Any]:
        """测试单个层连接"""
        result = {
            'success': True,
            'errors': [],
            'methods_tested': 0,
            'methods_successful': 0
        }

        # 测试每个方法
        for method_name in layer_interface.list_methods():
            result['methods_tested'] += 1
            try:
                method_func = layer_interface.get_method(method_name)
                if method_func is not None:
                    result['methods_successful'] += 1
                else:
                    result['success'] = False
                    result['errors'].append(f"方法 {method_name} 获取失败")
            except Exception as e:
                result['success'] = False
                result['errors'].append(f"方法 {method_name} 测试失败: {str(e)}")

        return result

    def validate_system_integration(self) -> Dict[str, Any]:
        """验证系统集成"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'integration_status': self.integration_status
        }

        # 验证接口兼容性
        interface_validation = self.validate_interfaces()
        if not interface_validation['valid']:
            validation_result['valid'] = False
            validation_result['errors'].extend(interface_validation['errors'])

        # 验证连接状态
        connection_test = self.test_connections()
        if not connection_test['success']:
            validation_result['valid'] = False
            validation_result['errors'].extend(connection_test['errors'])

        # 验证配置完整性
        config_validation = self._validate_integration_config()
        if not config_validation['valid']:
            validation_result['valid'] = False
            validation_result['errors'].extend(config_validation['errors'])

        return validation_result

    def _validate_integration_config(self) -> Dict[str, Any]:
        """验证集成配置"""
        validation_result = {
            'valid': True,
            'errors': []
        }

        # 检查必要的配置项
        required_configs = ['timeout', 'retry_attempts', 'health_check_interval']
        for config_key in required_configs:
            if config_key not in self.integration_config:
                validation_result['valid'] = False
                validation_result['errors'].append(f"缺少必要配置: {config_key}")

        return validation_result

    def start_integration_services(self) -> Dict[str, Any]:
        """启动集成服务"""
        if self.integration_status == 'running':
            return {'success': True, 'message': '集成服务已在运行'}

        # 验证系统集成
        validation_result = self.validate_system_integration()
        if not validation_result['valid']:
            return {
                'success': False,
                'errors': validation_result['errors'],
                'message': '系统集成验证失败'
            }

        try:
            # 启动各层服务
            for layer_name, layer_interface in self.layer_interfaces.items():
                logger.info(f"启动 {layer_name} 层服务")
                # 这里可以添加具体的启动逻辑

            self.integration_status = 'running'
            logger.info("集成服务启动成功")

            return {
                'success': True,
                'message': '集成服务启动成功',
                'status': self.integration_status
            }

        except Exception as e:
            logger.error(f"集成服务启动失败: {str(e)}")
            return {
                'success': False,
                'errors': [str(e)],
                'message': '集成服务启动失败'
            }

    def stop_integration_services(self) -> Dict[str, Any]:
        """停止集成服务"""
        if self.integration_status == 'stopped':
            return {'success': True, 'message': '集成服务已停止'}

        try:
            # 停止各层服务
            for layer_name, layer_interface in self.layer_interfaces.items():
                logger.info(f"停止 {layer_name} 层服务")
                # 这里可以添加具体的停止逻辑

            self.integration_status = 'stopped'
            logger.info("集成服务停止成功")

            return {
                'success': True,
                'message': '集成服务停止成功',
                'status': self.integration_status
            }

        except Exception as e:
            logger.error(f"集成服务停止失败: {str(e)}")
            return {
                'success': False,
                'errors': [str(e)],
                'message': '集成服务停止失败'
            }

    def get_integration_status(self) -> Dict[str, Any]:
        """获取集成状态"""
        return {
            'status': self.integration_status,
            'layers': list(self.layer_interfaces.keys()),
            'config': self.integration_config
        }

    def configure_integration(self, config: Dict[str, Any]) -> None:
        """配置集成参数"""
        self.integration_config.update(config)
        logger.info(f"更新集成配置: {config}")
