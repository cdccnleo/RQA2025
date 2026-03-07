"""
基础设施层测试工具模式

提供集成测试框架和测试辅助工具。
"""

import os
import logging
import tempfile
import time
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger(__name__)

# ==================== 集成测试框架 ====================


class InfrastructureIntegrationTest:
    """基础设施层集成测试框架"""

    @staticmethod
    def create_test_environment(config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """创建测试环境"""
        temp_dir = tempfile.mkdtemp()

        # 默认测试配置
        default_config = {
            "cache": {
                "enabled": True,
                "max_size": 100,
                "ttl": 60
            },
            "logging": {
                "level": "INFO",
                "file": os.path.join(temp_dir, "test.log")
            },
            "monitoring": {
                "enabled": True,
                "interval": 30
            },
            "temp_dir": temp_dir
        }

        # 应用配置覆盖
        if config_overrides:
            InfrastructureIntegrationTest._deep_update(default_config, config_overrides)

        return default_config

    @staticmethod
    def _deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """深度更新字典"""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                InfrastructureIntegrationTest._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    @staticmethod
    def run_performance_benchmark(func: Callable, iterations: int = 100,
                                  warmup_iterations: int = 10) -> Dict[str, Any]:
        """运行性能基准测试"""
        # 预热
        for _ in range(warmup_iterations):
            func()

        # 正式测试
        times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            func()
            elapsed = time.perf_counter() - start_time
            times.append(elapsed)

        # 统计结果
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        return {
            "iterations": iterations,
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "total_time": sum(times),
            "performance_score": 1.0 / avg_time if avg_time > 0 else float('inf')
        }

    @staticmethod
    def assert_component_health(component: Any, required_methods: List[str]) -> bool:
        """断言组件健康状态"""
        if not hasattr(component, 'health_check'):
            return False

        try:
            health_result = component.health_check()
            if not isinstance(health_result, dict):
                return False

            # 检查必需的方法
            for method in required_methods:
                if not hasattr(component, method):
                    return False

            return health_result.get('healthy', False)
        except Exception:
            return False


# ==================== 测试辅助工具 ====================


class InfrastructureTestHelper:
    """基础设施层测试辅助工具"""

    @staticmethod
    def generate_test_template(class_name: str, methods: List[str]) -> str:
        """生成测试类模板"""
        template = f'''"""
测试 {class_name}
"""
import unittest
from unittest.mock import Mock, patch


class Test{class_name}(unittest.TestCase):
    """测试 {class_name} 类"""

    def setUp(self):
        """测试前置设置"""
        pass

    def tearDown(self):
        """测试后置清理"""
        pass
'''

        # 为每个方法生成测试
        for method in methods:
            template += f'''
    def test_{method}(self):
        """测试 {method} 方法"""
        # TODO: 实现测试逻辑
        pass
'''

        template += '''

if __name__ == '__main__':
    unittest.main()
'''

        return template

    @staticmethod
    def create_mock_component(component_type: str, **kwargs) -> Any:
        """创建模拟组件"""
        from unittest.mock import Mock
        
        mock = Mock()
        mock.component_type = component_type
        mock.is_healthy = Mock(return_value=True)
        mock.get_status = Mock(return_value={
            'healthy': True,
            'component_type': component_type
        })

        # 添加自定义属性
        for key, value in kwargs.items():
            setattr(mock, key, value)

        return mock

    @staticmethod
    def assert_dict_contains(actual: Dict[str, Any], expected_subset: Dict[str, Any]) -> bool:
        """断言字典包含子集"""
        for key, value in expected_subset.items():
            if key not in actual:
                return False
            if isinstance(value, dict):
                if not InfrastructureTestHelper.assert_dict_contains(actual[key], value):
                    return False
            elif actual[key] != value:
                return False
        return True


__all__ = [
    'InfrastructureIntegrationTest',
    'InfrastructureTestHelper',
]

