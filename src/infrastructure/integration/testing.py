"""
集成测试工具

提供分层集成测试功能，验证各层之间的协作。
重构版本：消除重复代码，使用通用测试框架。
"""

import logging
from typing import Dict, Any, List, Callable
from .interfaces.interface import SystemIntegrationManager, SystemLayerInterfaceManager

logger = logging.getLogger(__name__)


class LayerTestConfig:
    """层测试配置"""

    def __init__(self, layer_name: str, mock_methods: Dict[str, Callable],
                 test_cases: List[Dict[str, Any]]):
        self.layer_name = layer_name
        self.mock_methods = mock_methods
        self.test_cases = test_cases


class BaseLayerTester:
    """通用层测试器基类"""

    def __init__(self, integration_manager: SystemIntegrationManager):
        self.integration_manager = integration_manager

    def create_test_result(self) -> Dict[str, Any]:
        """创建测试结果模板"""
        return {
            'success': True,
            'errors': [],
            'warnings': [],
            'tests_passed': 0,
            'tests_failed': 0
        }

    def execute_test_case(self, interface: SystemLayerInterfaceManager,
                         method_name: str, args: List[Any],
                         expected_key: str, test_result: Dict[str, Any],
                         error_msg: str) -> None:
        """执行单个测试用例"""
        try:
            method = interface.get_method(method_name)
            result = method(*args)

            if result and expected_key in result:
                test_result['tests_passed'] += 1
            else:
                test_result['tests_failed'] += 1
                test_result['errors'].append(error_msg)
        except Exception as e:
            test_result['tests_failed'] += 1
            test_result['errors'].append(f"{error_msg}: {str(e)}")

    def validate_interface(self, interface: SystemLayerInterfaceManager,
                          test_result: Dict[str, Any]) -> None:
        """验证接口完整性"""
        try:
            validation = interface.validate_interface()
            if validation['valid']:
                test_result['tests_passed'] += 1
            else:
                test_result['tests_failed'] += 1
                test_result['errors'].extend(validation['errors'])
        except Exception as e:
            test_result['tests_failed'] += 1
            test_result['errors'].append(f"接口验证失败: {str(e)}")

    def run_layer_test(self, config: LayerTestConfig) -> Dict[str, Any]:
        """运行层测试"""
        logger.info(f"开始测试{config.layer_name}层集成")

        test_result = self.create_test_result()

        try:
            # 创建层接口
            interface = SystemLayerInterfaceManager(config.layer_name)

            # 注册模拟方法
            for method_name, method_func in config.mock_methods.items():
                interface.register_method(method_name, method_func)

            # 注册到集成管理器
            self.integration_manager.register_layer_interface(config.layer_name, interface)

            # 执行测试用例
            for test_case in config.test_cases:
                self.execute_test_case(
                    interface=interface,
                    method_name=test_case['method'],
                    args=test_case['args'],
                    expected_key=test_case['expected_key'],
                    test_result=test_result,
                    error_msg=test_case['error_msg']
                )

            # 验证接口完整性
            self.validate_interface(interface, test_result)

            if test_result['tests_failed'] > 0:
                test_result['success'] = False

        except Exception as e:
            test_result['success'] = False
            test_result['errors'].append(f"{config.layer_name}层集成测试异常: {str(e)}")
            logger.error(f"{config.layer_name}层集成测试失败: {str(e)}")

        logger.info(
            f"{config.layer_name}层集成测试完成: {test_result['tests_passed']} 通过, {test_result['tests_failed']} 失败")
        return test_result


class LayerIntegrationTester:
    """分层集成测试器 - 重构版本"""

    def __init__(self):
        self.test_results: Dict[str, Dict[str, Any]] = {}
        self.integration_manager = SystemIntegrationManager()
        self.base_tester = BaseLayerTester(self.integration_manager)

    def _get_data_layer_config(self) -> LayerTestConfig:
        """获取数据层测试配置"""
        mock_methods = {
            'load_data': lambda symbol: {'symbol': symbol, 'data': [1, 2, 3, 4, 5]},
            'validate_data': lambda data: {'valid': True, 'errors': []},
            'process_data': lambda data: {'processed': True, 'result': data}
        }

        test_cases = [
            {
                'method': 'load_data',
                'args': ['AAPL'],
                'expected_key': 'symbol',
                'error_msg': '数据加载测试失败'
            },
            {
                'method': 'validate_data',
                'args': [{'symbol': 'AAPL', 'data': [1, 2, 3, 4, 5]}],
                'expected_key': 'valid',
                'error_msg': '数据验证测试失败'
            },
            {
                'method': 'process_data',
                'args': [{'symbol': 'AAPL', 'data': [1, 2, 3, 4, 5]}],
                'expected_key': 'processed',
                'error_msg': '数据处理测试失败'
            }
        ]

        return LayerTestConfig('data', mock_methods, test_cases)

    def test_data_layer_integration(self) -> Dict[str, Any]:
        """测试数据层集成 - 重构版本"""
        config = self._get_data_layer_config()
        result = self.base_tester.run_layer_test(config)
        self.test_results['data'] = result
        return result

    def _get_features_layer_config(self) -> LayerTestConfig:
        """获取特征层测试配置"""
        mock_methods = {
            'extract_features': lambda data: {'features': ['feature1', 'feature2', 'feature3']},
            'select_features': lambda features: {'selected_features': ['feature1', 'feature2']},
            'engineer_features': lambda features: {'engineered_features': ['engineered_feature1']}
        }

        test_cases = [
            {
                'method': 'extract_features',
                'args': [{'symbol': 'AAPL', 'data': [1, 2, 3, 4, 5]}],
                'expected_key': 'features',
                'error_msg': '特征提取测试失败'
            },
            {
                'method': 'select_features',
                'args': [['feature1', 'feature2', 'feature3']],
                'expected_key': 'selected_features',
                'error_msg': '特征选择测试失败'
            },
            {
                'method': 'engineer_features',
                'args': [['feature1', 'feature2']],
                'expected_key': 'engineered_features',
                'error_msg': '特征工程测试失败'
            }
        ]

        return LayerTestConfig('features', mock_methods, test_cases)

    def test_features_layer_integration(self) -> Dict[str, Any]:
        """测试特征层集成 - 重构版本"""
        config = self._get_features_layer_config()
        result = self.base_tester.run_layer_test(config)
        self.test_results['features'] = result
        return result

    def _get_models_layer_config(self) -> LayerTestConfig:
        """获取模型层测试配置"""
        mock_methods = {
            'train_model': lambda features, target: {'model_id': 'model_001', 'accuracy': 0.85},
            'predict': lambda model_id, features: {'prediction': 0.75, 'confidence': 0.8},
            'evaluate_model': lambda model_id, test_data: {'accuracy': 0.85, 'precision': 0.82, 'recall': 0.88}
        }

        test_cases = [
            {
                'method': 'train_model',
                'args': [['feature1', 'feature2'], [0, 1, 0, 1, 0]],
                'expected_key': 'model_id',
                'error_msg': '模型训练测试失败'
            },
            {
                'method': 'predict',
                'args': ['model_001', [0.1, 0.2]],
                'expected_key': 'prediction',
                'error_msg': '模型预测测试失败'
            },
            {
                'method': 'evaluate_model',
                'args': ['model_001', {'test_data': []}],
                'expected_key': 'accuracy',
                'error_msg': '模型评估测试失败'
            }
        ]

        return LayerTestConfig('models', mock_methods, test_cases)

    def test_models_layer_integration(self) -> Dict[str, Any]:
        """测试模型层集成 - 重构版本"""
        config = self._get_models_layer_config()
        result = self.base_tester.run_layer_test(config)
        self.test_results['models'] = result
        return result

    def _get_trading_layer_config(self) -> LayerTestConfig:
        """获取交易层测试配置"""
        mock_methods = {
            'execute_trade': lambda order: {'order_id': 'order_001', 'status': 'executed', 'filled_price': 150.0},
            'check_risk': lambda order: {'approved': True, 'risk_level': 'low', 'warnings': []},
            'manage_portfolio': lambda portfolio: {'rebalanced': True, 'new_positions': {'AAPL': 0.3, 'GOOGL': 0.7}}
        }

        test_cases = [
            {
                'method': 'execute_trade',
                'args': [{'symbol': 'AAPL', 'side': 'BUY', 'quantity': 100, 'price': 150.0}],
                'expected_key': 'order_id',
                'error_msg': '交易执行测试失败'
            },
            {
                'method': 'check_risk',
                'args': [{'symbol': 'AAPL', 'side': 'BUY', 'quantity': 100, 'price': 150.0}],
                'expected_key': 'approved',
                'error_msg': '风险检查测试失败'
            },
            {
                'method': 'manage_portfolio',
                'args': [{'positions': {}}],
                'expected_key': 'rebalanced',
                'error_msg': '投资组合管理测试失败'
            }
        ]

        return LayerTestConfig('trading', mock_methods, test_cases)

    def test_trading_layer_integration(self) -> Dict[str, Any]:
        """测试交易层集成 - 重构版本"""
        config = self._get_trading_layer_config()
        result = self.base_tester.run_layer_test(config)
        self.test_results['trading'] = result
        return result

    def _get_services_layer_config(self) -> LayerTestConfig:
        """获取服务层测试配置"""
        mock_methods = {
            'provide_service': lambda service_name, request: {'service_id': 'service_001', 'status': 'active', 'response': request},
            'validate_service': lambda service_id: {'valid': True, 'health_score': 0.95},
            'monitor_service': lambda service_id: {'status': 'healthy', 'metrics': {'cpu': 0.3, 'memory': 0.5}}
        }

        test_cases = [
            {
                'method': 'provide_service',
                'args': ['prediction_service', {'action': 'predict', 'data': [1, 2, 3]}],
                'expected_key': 'service_id',
                'error_msg': '服务提供测试失败'
            },
            {
                'method': 'validate_service',
                'args': ['service_001'],
                'expected_key': 'valid',
                'error_msg': '服务验证测试失败'
            },
            {
                'method': 'monitor_service',
                'args': ['service_001'],
                'expected_key': 'status',
                'error_msg': '服务监控测试失败'
            }
        ]

        return LayerTestConfig('services', mock_methods, test_cases)

    def test_services_layer_integration(self) -> Dict[str, Any]:
        """测试服务层集成 - 重构版本"""
        config = self._get_services_layer_config()
        result = self.base_tester.run_layer_test(config)
        self.test_results['services'] = result
        return result

    def _get_application_layer_config(self) -> LayerTestConfig:
        """获取应用层测试配置"""
        mock_methods = {
            'start_application': lambda config: {'app_id': 'app_001', 'status': 'started', 'pid': 12345},
            'stop_application': lambda app_id: {'app_id': app_id, 'status': 'stopped'},
            'get_status': lambda app_id: {'app_id': app_id, 'status': 'running', 'uptime': 3600}
        }

        test_cases = [
            {
                'method': 'start_application',
                'args': [{'mode': 'live', 'strategy': 'momentum'}],
                'expected_key': 'app_id',
                'error_msg': '应用启动测试失败'
            },
            {
                'method': 'stop_application',
                'args': ['app_001'],
                'expected_key': 'status',
                'error_msg': '应用停止测试失败'
            },
            {
                'method': 'get_status',
                'args': ['app_001'],
                'expected_key': 'status',
                'error_msg': '状态获取测试失败'
            }
        ]

        return LayerTestConfig('application', mock_methods, test_cases)

    def test_application_layer_integration(self) -> Dict[str, Any]:
        """测试应用层集成 - 重构版本"""
        config = self._get_application_layer_config()
        result = self.base_tester.run_layer_test(config)
        self.test_results['application'] = result
        return result

    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有集成测试"""
        logger.info("开始运行所有集成测试")

        # 运行各层测试
        self.test_data_layer_integration()
        self.test_features_layer_integration()
        self.test_models_layer_integration()
        self.test_trading_layer_integration()
        self.test_services_layer_integration()
        self.test_application_layer_integration()

        # 测试系统集成
        system_integration_result = self.integration_manager.validate_system_integration()

        # 汇总测试结果
        total_tests = 0
        total_passed = 0
        total_failed = 0
        all_errors = []

        for layer_name, result in self.test_results.items():
            total_tests += result['tests_passed'] + result['tests_failed']
            total_passed += result['tests_passed']
            total_failed += result['tests_failed']
            all_errors.extend(result['errors'])

        overall_result = {
            'success': total_failed == 0 and system_integration_result['valid'],
            'total_tests': total_tests,
            'tests_passed': total_passed,
            'tests_failed': total_failed,
            'layer_results': self.test_results,
            'system_integration': system_integration_result,
            'errors': all_errors
        }

        logger.info(f"所有集成测试完成: {total_passed} 通过, {total_failed} 失败")
        return overall_result
