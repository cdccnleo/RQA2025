#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层根目录并发控制器组件测试

测试目标：提升concurrency_controller.py的真实覆盖率
实际导入和使用src.infrastructure.concurrency_controller模块
"""

import pytest


class TestConcurrencyController:
    """测试并发控制器类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.concurrency_controller import ConcurrencyController
        
        controller = ConcurrencyController()
        
        assert isinstance(controller.config, dict)
        assert controller._status == "healthy"
    
    def test_init_with_config(self):
        """测试使用配置初始化"""
        from src.infrastructure.concurrency_controller import ConcurrencyController
        
        config = {"max_workers": 10, "timeout": 30}
        controller = ConcurrencyController(config)
        
        assert controller.config["max_workers"] == 10
        assert controller.config["timeout"] == 30
    
    def test_init_with_kwargs(self):
        """测试使用关键字参数初始化"""
        from src.infrastructure.concurrency_controller import ConcurrencyController
        
        controller = ConcurrencyController(max_workers=5, timeout=60)
        
        assert controller.config["max_workers"] == 5
        assert controller.config["timeout"] == 60
    
    def test_init_with_config_and_kwargs(self):
        """测试使用配置和关键字参数初始化"""
        from src.infrastructure.concurrency_controller import ConcurrencyController
        
        config = {"max_workers": 10}
        controller = ConcurrencyController(config, timeout=60)
        
        # kwargs应该覆盖config中的值
        assert controller.config["max_workers"] == 10
        assert controller.config["timeout"] == 60
    
    def test_initialize(self):
        """测试初始化方法"""
        from src.infrastructure.concurrency_controller import ConcurrencyController
        
        controller = ConcurrencyController()
        result = controller.initialize()
        
        assert result is True
        assert controller._status == "healthy"
    
    def test_shutdown(self):
        """测试关闭方法"""
        from src.infrastructure.concurrency_controller import ConcurrencyController
        
        controller = ConcurrencyController()
        controller._active_tasks["task1"] = "data"
        
        result = controller.shutdown()
        
        assert result is True
        assert len(controller._active_tasks) == 0
        assert controller._status == "healthy"
    
    def test_health_check(self):
        """测试健康检查"""
        from src.infrastructure.concurrency_controller import ConcurrencyController
        
        controller = ConcurrencyController()
        health = controller.health_check()
        
        assert isinstance(health, dict)
        assert health["status"] == "healthy"
        assert "component" in health
    
    def test_get_config(self):
        """测试获取配置"""
        from src.infrastructure.concurrency_controller import ConcurrencyController
        
        controller = ConcurrencyController({"key": "value"})
        config = controller.get_config()
        
        assert isinstance(config, dict)
        assert config["key"] == "value"
        # 应该返回副本
        config["new_key"] = "new_value"
        assert "new_key" not in controller.config


class TestConcurrencyControllerFactory:
    """测试并发控制器工厂函数"""
    
    def test_create_controller(self):
        """测试创建控制器"""
        from src.infrastructure.concurrency_controller import create_infrastructure_core_async_processing_concurrency_controller
        
        controller = create_infrastructure_core_async_processing_concurrency_controller(max_workers=5)
        
        assert controller is not None
        assert controller.config.get("max_workers") == 5
    
    def test_get_controller_instance(self):
        """测试获取控制器实例"""
        from src.infrastructure.concurrency_controller import get_infrastructure_core_async_processing_concurrency_controller
        
        controller1 = get_infrastructure_core_async_processing_concurrency_controller()
        controller2 = get_infrastructure_core_async_processing_concurrency_controller()
        
        # 应该是同一个实例
        assert controller1 is controller2
    
    def test_global_instance(self):
        """测试全局实例"""
        from src.infrastructure.concurrency_controller import infrastructure_core_async_processing_concurrency_controller_instance
        
        assert infrastructure_core_async_processing_concurrency_controller_instance is not None
        assert hasattr(infrastructure_core_async_processing_concurrency_controller_instance, 'health_check')

