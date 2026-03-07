#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
容器边界测试
"""

import pytest
from unittest.mock import Mock
from src.core import DependencyContainer, Lifecycle


class TestContainerBoundary:


    """容器边界测试"""


    def test_circular_dependency(self):


        """测试循环依赖"""
        container = DependencyContainer()


        class ServiceA:


            def __init__(self, service_b):


                self.service_b = service_b


        class ServiceB:


            def __init__(self, service_a):


                self.service_a = service_a

        # 应该检测到循环依赖
        with pytest.raises(Exception):
            container.register("service_a", ServiceA, dependencies=["service_b"])
            container.register("service_b", ServiceB, dependencies=["service_a"])
            container.get("service_a")


    def test_missing_dependency(self):


        """测试缺失依赖"""
        container = DependencyContainer()


        class ServiceA:


            def __init__(self, missing_service):


                self.missing_service = missing_service

        # 应该检测到缺失依赖
        with pytest.raises(Exception):
            container.register("service_a", ServiceA, dependencies=["missing_service"])
            container.get("service_a")


    def test_large_number_of_services(self):


        """测试大量服务"""
        container = DependencyContainer()

        # 注册1000个服务
        for i in range(1000):
            container.register(f"service_{i}", Mock(), lifecycle=Lifecycle.SINGLETON)

        # 确保所有服务都能正常获取
        for i in range(1000):
            service = container.get(f"service_{i}")
            assert service is not None
