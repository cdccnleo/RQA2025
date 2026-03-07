#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - Formatter组件

测试logging/formatters/formatter_components.py中的所有类和方法
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from abc import ABC
from typing import Dict, Any

from src.infrastructure.logging.formatters.formatter_components import (
    IFormatterComponent, FormatterComponent, FormatterComponentFactory
)


class TestIFormatterComponent:
    """测试Formatter组件接口"""

    def test_interface_inheritance(self):
        """测试接口继承"""
        assert issubclass(IFormatterComponent, ABC)

    def test_interface_abstract_methods(self):
        """测试接口抽象方法"""
        abstract_methods = IFormatterComponent.__abstractmethods__
        expected_methods = {'process', 'get_status', 'validate_config', 'get_config_schema'}

        assert len(abstract_methods) >= len(expected_methods)
        for method in expected_methods:
            assert method in abstract_methods

    def test_interface_cannot_instantiate(self):
        """测试接口不能直接实例化"""
        with pytest.raises(TypeError):
            IFormatterComponent()

    def test_get_info_default_implementation(self):
        """测试默认的get_info实现"""
        # 创建一个实现了抽象方法的临时类来测试get_info
        class ConcreteFormatter(IFormatterComponent):
            def process(self, data):
                return data

            def get_status(self):
                return {"status": "ok"}

            def validate_config(self, config):
                return True

            def get_config_schema(self):
                return {"type": "object"}

            def get_formatter_id(self) -> int:
                return 1

        formatter = ConcreteFormatter()
        info = formatter.get_info()

        assert isinstance(info, dict)
        assert info["component_type"] == "formatter_component"
        assert info["interface"] == "IFormatterComponent"
        assert info["version"] == "2.0.0"
        assert "description" in info


class TestFormatterComponent:
    """测试Formatter组件基类"""

    def setup_method(self):
        """测试前准备"""
        self.component = FormatterComponent(3, "Formatter")

    def test_initialization(self):
        """测试初始化"""
        assert hasattr(self.component, 'formatter_id')
        assert hasattr(self.component, 'component_name')
        assert self.component.formatter_id == 3
        assert self.component.component_type == "Formatter"
        assert hasattr(self.component, 'creation_time')

        # Component should have basic attributes
        assert self.component.get_formatter_id() == 3
        assert self.component.creation_time is not None

    def test_initialization_with_config(self):
        """测试带配置的初始化"""
        config = {"max_size": 1024}
        component = FormatterComponent(1, config=config, name="TestFormatter")
        assert component.formatter_id == 1
        assert component.config == config

    # def test_process_not_implemented(self):
    #     """测试process方法未实现"""
    #     # FormatterComponent没有实现process方法，应该抛出NotImplementedError
    #     with pytest.raises(NotImplementedError):
    #         self.component.process({"test": "data"})

    def test_get_status(self):
        """测试获取状态"""
        component = FormatterComponent(1)
        status = component.get_status()
        assert "creation_time" in status

        assert isinstance(status, dict)
        assert "status" in status
        assert "name" in status
        assert "created_at" in status
        assert "last_used" in status
        assert "uptime_seconds" in status

        assert status["name"] == "FormatterComponent"
        assert status["status"] == "active"
        assert isinstance(status["uptime_seconds"], (int, float))

    def test_validate_config_default(self):
        """测试默认配置验证"""
        # 默认实现应该接受任何配置
        result1 = self.component.validate_config("test_type", {})
        assert result1 is True
        result2 = self.component.validate_config("test_type", {"key": "value"})
        assert result2 is True
        result3 = self.component.validate_config("test_type", {"number": 42})
        assert result3 is True

    def test_get_config_schema_default(self):
        """测试默认配置模式"""
        schema = self.component.get_config_schema()

        assert isinstance(schema, dict)
        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema

    def test_update_last_used(self):
        """测试更新最后使用时间"""
        component = FormatterComponent(1)
        old_time = component.last_used
        time.sleep(0.01)
        component.update_last_used()
        new_time = component.last_used
        assert new_time > old_time

    def test_config_immutability(self):
        """测试配置不可变性"""
        config = {"initial": "value"}
        component = FormatterComponent(1, config=config)

        # 修改传入的配置字典
        component.config["new"] = "added"

        # 组件的配置应该不受影响
        assert "new" in component.config and "new" not in config

    def test_component_lifecycle(self):
        """测试组件生命周期"""
        # 创建
        component = FormatterComponent(name="LifecycleTest")

        # 获取状态
        status1 = component.get_status()
        assert status1["status"] == "active"

        # 再次获取状态（应该更新last_used）
        time.sleep(0.1)
        component.update_last_used()
        status2 = component.get_status()

        assert status2["last_used"] != status1["last_used"]
        assert status2["uptime_seconds"] > status1["uptime_seconds"]

    def test_error_handling_in_status(self):
        """测试状态获取中的错误处理"""
        # Mock datetime.now抛出异常
        from datetime import datetime
        with patch('src.infrastructure.logging.formatters.formatter_components.datetime') as mock_datetime:
            mock_datetime.now.side_effect = Exception("Time error")
            status = self.component.get_status()

            # 应该仍然返回有效的状态
            assert isinstance(status, dict)
            assert "status" in status
            assert status["status"] == "error" or "error" in status

    def test_config_validation_edge_cases(self):
        """测试配置验证边界情况"""
        test_cases = [
            None,
            {},
            {"key": "value"},
            {"nested": {"key": "value"}},
            {"list": [1, 2, 3]},
            {"boolean": True, "number": 42, "string": "test"}
        ]

        for config in test_cases:
            # 默认验证应该接受所有配置
            result = self.component.validate_config("test_type", config)
            assert result is True

        result = self.component.validate_config("test_type", {"invalid": "config"})
        assert result is True  # 应该是True，因为它是字典

    def test_schema_consistency(self):
        """测试模式一致性"""
        schema = self.component.get_config_schema()

        # 验证基本结构
        assert isinstance(schema, dict)
        assert "type" in schema

        # 如果有properties，应该是一个对象
        if "properties" in schema:
            assert schema["type"] == "object"
            assert isinstance(schema["properties"], dict)


class TestFormatterComponentFactory:
    """测试Formatter组件工厂"""

    def setup_method(self):
        """测试前准备"""
        self.factory = FormatterComponentFactory()

    def test_initialization(self):
        """测试初始化"""
        assert hasattr(self.factory, 'registered_components')
        assert hasattr(self.factory, 'component_instances')
        assert isinstance(self.factory.registered_components, dict)
        assert isinstance(self.factory.component_instances, dict)

    def test_register_component(self):
        """测试注册组件"""
        # 创建一个测试组件类
        class TestFormatterComponent(FormatterComponent):
            def process(self, data):
                return {"processed": True, **data}

        # 注册组件
        self.factory.register_component("test_formatter", TestFormatterComponent)

        assert "test_formatter" in self.factory.registered_components
        assert self.factory.registered_components["test_formatter"] == TestFormatterComponent

    def test_create_component(self):
        self.factory.register_component("test_formatter", FormatterComponent)
        component = self.factory.create_component("test_formatter", config={})
        assert isinstance(component, FormatterComponent)
        assert component.formatter_id is not None

    def test_create_component_not_registered(self):
        with pytest.raises(ValueError):
            self.factory.create_component("nonexistent")

    def test_get_component_info(self):
        """测试获取组件信息"""
        # 注册一个组件
        class TestFormatterComponent(FormatterComponent):
            def process(self, data):
                return {"processed": True, **data}

        self.factory.register_component("test_formatter", TestFormatterComponent)

        info = self.factory.get_component_info(1)
        assert "type" in info

    def test_get_component_info_not_registered(self):
        with pytest.raises(ValueError, match="Component type 'nonexistent' not registered"):
            self.factory.get_component_info("nonexistent")

    def test_list_registered_components(self):
        """测试列出已注册的组件"""
        # 注册多个组件
        class ComponentA(FormatterComponent):
            pass
        class ComponentB(FormatterComponent):
            pass
        self.factory.register_component("component_a", ComponentA)
        self.factory.register_component("component_b", ComponentB)
        registered = self.factory.list_registered_components()
        assert len(registered) >= 2
        assert "component_a" in registered
        assert "component_b" in registered

    def test_component_instance_caching(self):
        """测试组件实例缓存"""
        class TestFormatterComponent(FormatterComponent):
            def __init__(self, config=None, name=None):
                super().__init__(formatter_id=1, component_type="Formatter", config=config, name=name)
                self.instance_id = id(self)  # 唯一标识符

        self.factory.register_component("cached_component", TestFormatterComponent)

        # 创建第一个实例
        instance1 = self.factory.create_component("cached_component", name="instance1")

        # 创建第二个实例（应该不同）
        instance2 = self.factory.create_component("cached_component", name="instance2")

        assert instance1.instance_id != instance2.instance_id
        assert instance1.name == "instance1"
        assert instance2.name == "instance2"

    def test_factory_error_handling(self):
        with pytest.raises(TypeError):
            self.factory.register_component("invalid", "not_a_class")

        # 测试创建组件时配置验证失败
        class FailingComponent(FormatterComponent):
            def __init__(self, config=None, name=None):
                if config and config.get("fail"):
                    raise ValueError("Configuration validation failed")
                super().__init__(config, name)

        self.factory.register_component("failing_component", FailingComponent)

        with pytest.raises(ValueError, match="Configuration validation failed"):
            self.factory.create_component("failing_component", config={"fail": True})

    def test_factory_thread_safety(self):
        """测试工厂线程安全性"""
        import threading
        from concurrent.futures import ThreadPoolExecutor

        class ThreadSafeComponent(FormatterComponent):
            def __init__(self, config=None, name=None):
                super().__init__(config, name)
                self.thread_id = threading.current_thread().ident

        self.factory.register_component("thread_safe", ThreadSafeComponent)

        def create():
            return self.factory.create_component(1, config={})
        with ThreadPoolExecutor(5) as executor:
            results = list(executor.map(lambda _: create(), range(5)))
        assert len(results) == 5
        assert all(isinstance(r, FormatterComponent) for r in results)

    def test_factory_performance(self):
        """测试工厂性能"""
        import time

        class SimpleComponent(FormatterComponent):
            def process(self, data):
                return data

        self.factory.register_component("perf_test", SimpleComponent)

        # 测试创建多个实例的性能
        start_time = time.time()

        for i in range(100):
            component = self.factory.create_component("perf_test", name=f"perf_{i}")
            assert isinstance(component, SimpleComponent)

        end_time = time.time()
        duration = end_time - start_time

        # 应该在合理时间内完成
        assert duration < 1.0  # 少于2秒创建100个实例

    def test_factory_resource_management(self):
        """测试工厂资源管理"""
        class ResourceComponent(FormatterComponent):
            def __init__(self, config=None, name=None):
                super().__init__(config, name)
                self.resources = ["resource1", "resource2"]

            def __del__(self):
                # 清理资源
                self.resources.clear()

        self.factory.register_component("resource_test", ResourceComponent)

        # 创建实例
        component = self.factory.create_component("resource_test")

        assert len(component.resources) == 2

        # 删除引用（模拟垃圾回收）
        del component

        # 在实际环境中，__del__会被调用，但这里我们主要验证创建过程

    def test_factory_configuration_isolation(self):
        component1 = self.factory.create_component(1, config={"env": "dev"})
        component2 = self.factory.create_component(1, config={"env": "prod"})
        assert component1.config["env"] == "dev"
        assert component2.config["env"] == "prod"

    def test_factory_edge_cases(self):
        with pytest.raises(ValueError):
            self.factory.create_component("invalid")

    def test_factory_component_info_caching(self):
        """测试组件信息缓存"""
        class InfoComponent(FormatterComponent):
            def process(self, data):
                return data

        self.factory.register_component("info_cache_test", InfoComponent)

        # 多次获取信息应该返回相同的结果
        info1 = self.factory.get_component_info("info_cache_test")
        info2 = self.factory.get_component_info("info_cache_test")

        assert info1 == info2
        assert info1["component_type"] == "info_cache_test"
        assert info1["class"] == InfoComponent


class TestFormatterComponentEdgeCases:
    """测试FormatterComponent的边界情况"""

    def setup_method(self):
        """测试前准备"""
        from src.infrastructure.logging.formatters.formatter_components import FormatterComponent
        self.FormatterComponent = FormatterComponent

    def test_formatter_component_with_invalid_config(self):
        """测试FormatterComponent使用无效配置"""
        class TestFormatter(self.FormatterComponent):
            def process(self, data):
                return {"result": "processed", "input": data}

        # 使用无效配置创建组件
        invalid_config = {"invalid_param": "value", "another_invalid": 123}
        component = TestFormatter(config=invalid_config)

        # 应该能够处理无效配置
        assert component is not None
        status = component.get_status()
        assert isinstance(status, dict)

    def test_formatter_component_config_validation(self):
        """测试FormatterComponent配置验证"""
        class TestFormatter(self.FormatterComponent):
            def process(self, data):
                return data

            def validate_config(self, config):
                """自定义配置验证"""
                if "required_field" not in config:
                    raise ValueError("Missing required_field")
                if not isinstance(config.get("number_field", 0), (int, float)):
                    raise ValueError("number_field must be numeric")
                return True

        # 测试有效配置
        valid_config = {"required_field": "value", "number_field": 42}
        component = TestFormatter(config=valid_config)
        assert component.validate_config(valid_config) is True

        # 测试无效配置 - 手动调用验证
        invalid_config = {"missing_required": "value"}
        component = TestFormatter(config=invalid_config)
        with pytest.raises(ValueError, match="Missing required_field"):
            component.validate_config(invalid_config)

    def test_formatter_component_config_schema(self):
        """测试FormatterComponent配置模式"""
        class TestFormatter(self.FormatterComponent):
            def process(self, data):
                return data

            def get_config_schema(self):
                """自定义配置模式"""
                return {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "enabled": {"type": "boolean"},
                        "count": {"type": "integer", "minimum": 0}
                    },
                    "required": ["name"]
                }

        component = TestFormatter()
        schema = component.get_config_schema()
        assert isinstance(schema, dict)
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "name" in schema["properties"]

    def test_formatter_component_process_with_complex_data(self):
        """测试FormatterComponent处理复杂数据"""
        class TestFormatter(self.FormatterComponent):
            def process(self, data):
                # 处理嵌套数据结构
                result = {
                    "original": data,
                    "processed_at": "test_timestamp",
                    "metadata": {
                        "input_keys": list(data.keys()),
                        "input_type": type(data).__name__
                    }
                }

                # 处理嵌套对象
                if "nested" in data and isinstance(data["nested"], dict):
                    result["nested_processed"] = True
                    result["nested_keys"] = list(data["nested"].keys())

                return result

        component = TestFormatter()

        # 测试简单数据
        simple_data = {"key": "value", "number": 42}
        result = component.process(simple_data)
        assert result["original"] == simple_data
        assert "processed_at" in result
        assert result["metadata"]["input_keys"] == ["key", "number"]

        # 测试嵌套数据
        nested_data = {
            "simple": "value",
            "nested": {"inner_key": "inner_value", "count": 10}
        }
        result = component.process(nested_data)
        assert result["nested_processed"] is True
        assert "inner_key" in result["nested_keys"]
        assert "count" in result["nested_keys"]

    def test_formatter_component_error_handling_in_process(self):
        """测试FormatterComponent在process方法中的错误处理"""
        class TestFormatter(self.FormatterComponent):
            def process(self, data):
                # 模拟处理过程中的错误
                if "error_trigger" in data:
                    raise ValueError("Simulated processing error")
                return {"result": "success", "input": data}

        component = TestFormatter()

        # 测试正常处理
        normal_data = {"normal": "data"}
        result = component.process(normal_data)
        assert result["result"] == "success"

        # 测试错误处理
        error_data = {"error_trigger": True, "data": "test"}
        with pytest.raises(ValueError, match="Simulated processing error"):
            component.process(error_data)

    def test_formatter_component_status_with_detailed_info(self):
        """测试FormatterComponent状态的详细信息"""
        class TestFormatter(self.FormatterComponent):
            def process(self, data):
                return data

            def get_status(self):
                """返回详细状态信息"""
                return {
                    "status": "active",
                    "uptime": 3600,
                    "processed_count": 150,
                    "error_count": 2,
                    "last_processed": "2024-01-01T12:00:00Z",
                    "memory_usage": 25.5,
                    "active_threads": 3,
                    "queue_size": 5
                }

        component = TestFormatter()
        status = component.get_status()

        assert status["status"] == "active"
        assert status["uptime"] == 3600
        assert status["processed_count"] == 150
        assert status["error_count"] == 2
        assert isinstance(status["memory_usage"], float)
        assert status["active_threads"] == 3
        assert status["queue_size"] == 5

    def test_formatter_component_concurrent_processing(self):
        """测试FormatterComponent的并发处理能力"""
        import threading
        import time

        class TestFormatter(self.FormatterComponent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.process_count = 0
                self.lock = threading.Lock()

            def process(self, data):
                with self.lock:
                    self.process_count += 1
                    time.sleep(0.01)  # 模拟处理时间
                    return {
                        "result": f"processed_{self.process_count}",
                        "input": data,
                        "thread_id": threading.current_thread().ident
                    }

        component = TestFormatter()
        results = []
        errors = []

        def worker(worker_id):
            """工作线程"""
            try:
                for i in range(5):
                    result = component.process({"worker": worker_id, "item": i})
                    results.append(result)
            except Exception as e:
                errors.append(f"worker_{worker_id}_error: {e}")

        # 启动多个线程
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)

        # 执行并发处理
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # 验证结果
        assert len(results) == 15  # 3 threads * 5 items each
        assert len(errors) == 0
        assert component.process_count == 15

        # 验证结果的唯一性
        processed_results = [r["result"] for r in results]
        assert len(set(processed_results)) == 15  # 所有结果都不同

    def test_formatter_component_resource_management(self):
        """测试FormatterComponent的资源管理"""
        class TestFormatter(self.FormatterComponent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.resources_created = False
                self.resources_cleaned = False

            def process(self, data):
                if not self.resources_created:
                    self.resources_created = True
                    # 模拟资源创建
                return {"processed": True, "data": data}

            def cleanup(self):
                """清理资源"""
                self.resources_cleaned = True

        component = TestFormatter()

        # 执行处理
        result = component.process({"test": "data"})
        assert result["processed"] is True

        # 执行清理
        component.cleanup()
        assert component.resources_cleaned is True

    def test_formatter_component_config_updates(self):
        """测试FormatterComponent配置更新"""
        class TestFormatter(self.FormatterComponent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.dynamic_config = {}

            def process(self, data):
                # 使用动态配置
                if self.dynamic_config.get("uppercase", False):
                    return {"result": str(data).upper()}
                return {"result": str(data).lower()}

            def update_config(self, new_config):
                """更新配置"""
                self.dynamic_config.update(new_config)
                return True

        component = TestFormatter()

        # 测试默认行为
        result1 = component.process("Hello World")
        assert result1["result"] == "hello world"

        # 更新配置
        component.update_config({"uppercase": True})

        # 测试更新后的行为
        result2 = component.process("Hello World")
        assert result2["result"] == "HELLO WORLD"