#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
健康检查模块单元测试
测试HealthChecker和HealthCheck类的核心功能
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.infrastructure.health.health_checker import HealthChecker, HealthStatus
from src.infrastructure.health.health_check import HealthCheck
from src.infrastructure.config.config_manager import ConfigManager


class TestHealthStatus:
    """测试HealthStatus数据类"""
    
    def test_health_status_creation(self):
        """测试HealthStatus创建"""
        status = HealthStatus(
            service="test_service",
            status="UP",
            timestamp=time.time(),
            details={"test": "value"},
            last_check=time.time()
        )
        
        assert status.service == "test_service"
        assert status.status == "UP"
        assert status.details == {"test": "value"}


class TestHealthChecker:
    """测试HealthChecker类"""
    
    @pytest.fixture
    def mock_config(self):
        """模拟配置"""
        return {
            "health_check": {
                "interval": 5,
                "services": ["database", "redis", "trading_engine"]
            },
            "database": {
                "host": "localhost",
                "port": 5432
            }
        }
    
    @pytest.fixture
    def mock_config_manager(self):
        """模拟配置管理器"""
        config_manager = Mock(spec=ConfigManager)
        config_manager.get_config.return_value = {
            "interval": 5,
            "services": ["database", "redis", "trading_engine"]
        }
        return config_manager
    
    def test_health_checker_initialization(self, mock_config, mock_config_manager):
        """测试健康检查器初始化"""
        checker = HealthChecker(mock_config, mock_config_manager)
        
        assert checker.config == mock_config
        assert checker.config_manager == mock_config_manager
        assert checker.check_interval == 5
        assert "database" in checker.services_to_check
        assert "redis" in checker.services_to_check
        assert not checker.running
    
    def test_start_and_stop(self, mock_config, mock_config_manager):
        """测试启动和停止"""
        checker = HealthChecker(mock_config, mock_config_manager)
        
        # 测试启动
        checker.start()
        assert checker.running
        assert checker.check_thread is not None
        assert checker.check_thread.is_alive()
        
        # 测试停止
        checker.stop()
        assert not checker.running
    
    def test_get_status(self, mock_config, mock_config_manager):
        """测试获取状态"""
        checker = HealthChecker(mock_config, mock_config_manager)
        
        # 测试获取所有状态
        status = checker.get_status()
        assert isinstance(status, dict)
        
        # 测试获取特定服务状态
        status = checker.get_status("database")
        assert isinstance(status, dict)
    
    def test_is_healthy(self, mock_config, mock_config_manager):
        """测试健康状态检查"""
        checker = HealthChecker(mock_config, mock_config_manager)
        
        # 初始状态应该不健康
        assert not checker.is_healthy("database")
        
        # 手动设置健康状态
        checker.health_status["database"] = HealthStatus(
            service="database",
            status="UP",
            timestamp=time.time(),
            details={},
            last_check=time.time()
        )
        
        assert checker.is_healthy("database")
    
    def test_check_database(self, mock_config, mock_config_manager):
        """测试数据库健康检查"""
        checker = HealthChecker(mock_config, mock_config_manager)
        
        status, details = checker._check_database()
        
        assert status in ["UP", "DOWN"]
        assert isinstance(details, dict)
    
    def test_check_redis(self, mock_config, mock_config_manager):
        """测试Redis健康检查"""
        checker = HealthChecker(mock_config, mock_config_manager)
        
        status, details = checker._check_redis()
        
        assert status == "UP"
        assert "used_memory" in details
        assert "ops_per_sec" in details
        assert "connected_clients" in details
    
    def test_check_trading_engine(self, mock_config, mock_config_manager):
        """测试交易引擎健康检查"""
        checker = HealthChecker(mock_config, mock_config_manager)
        
        status, details = checker._check_trading_engine()
        
        assert status in ["UP", "DOWN"]
        assert isinstance(details, dict)
    
    def test_check_risk_system(self, mock_config, mock_config_manager):
        """测试风控系统健康检查"""
        checker = HealthChecker(mock_config, mock_config_manager)
        
        status, details = checker._check_risk_system()
        
        assert status in ["UP", "DOWN"]
        assert isinstance(details, dict)
    
    def test_check_data_service(self, mock_config, mock_config_manager):
        """测试数据服务健康检查"""
        checker = HealthChecker(mock_config, mock_config_manager)
        
        status, details = checker._check_data_service()
        
        assert status in ["UP", "DOWN"]
        assert isinstance(details, dict)
    
    def test_get_health_report(self, mock_config, mock_config_manager):
        """测试获取健康报告"""
        checker = HealthChecker(mock_config, mock_config_manager)
        
        # 设置一些健康状态
        checker.health_status["database"] = HealthStatus(
            service="database",
            status="UP",
            timestamp=time.time(),
            details={},
            last_check=time.time()
        )
        checker.health_status["redis"] = HealthStatus(
            service="redis",
            status="DOWN",
            timestamp=time.time(),
            details={"error": "Connection failed"},
            last_check=time.time()
        )
        
        report = checker.get_health_report()
        
        assert "timestamp" in report
        assert "overall_status" in report
        assert "services" in report
        assert "degraded_services" in report
        assert "down_services" in report
        assert report["overall_status"] == "DOWN"
        assert "redis" in report["down_services"]
    
    def test_register_custom_check(self, mock_config, mock_config_manager):
        """测试注册自定义检查器"""
        checker = HealthChecker(mock_config, mock_config_manager)
        
        def custom_check():
            return "UP", {"custom": "value"}
        
        checker.register_custom_check("custom_service", custom_check)
        
        assert "custom_service" in checker.checkers
        assert "custom_service" in checker.services_to_check
    
    def test_trigger_manual_check(self, mock_config, mock_config_manager):
        """测试触发手动检查"""
        checker = HealthChecker(mock_config, mock_config_manager)
        
        # 测试检查特定服务
        checker.trigger_manual_check("database")
        
        # 测试检查所有服务
        checker.trigger_manual_check()
    
    def test_trigger_manual_check_unknown_service(self, mock_config, mock_config_manager):
        """测试检查未知服务"""
        checker = HealthChecker(mock_config, mock_config_manager)
        
        # 应该不会抛出异常
        checker.trigger_manual_check("unknown_service")
    
    def test_perform_checks_exception_handling(self, mock_config, mock_config_manager):
        """测试检查异常处理"""
        checker = HealthChecker(mock_config, mock_config_manager)
        
        # 模拟检查器抛出异常
        def failing_check():
            raise Exception("Test error")
        
        checker.checkers["database"] = failing_check
        
        # 应该不会抛出异常
        checker._perform_checks()
        
        # 应该记录错误状态
        assert "database" in checker.health_status
        assert checker.health_status["database"].status == "DOWN"


class TestHealthCheck:
    """测试HealthCheck类"""
    
    @pytest.fixture
    def health_check(self):
        """创建HealthCheck实例"""
        return HealthCheck()
    
    def test_health_check_initialization(self, health_check):
        """测试HealthCheck初始化"""
        assert health_check.router is not None
        assert len(health_check.dependencies) == 0
    
    def test_add_dependency_check(self, health_check):
        """测试添加依赖检查"""
        def test_check():
            return True
        
        health_check.add_dependency_check("test_service", test_check)
        
        assert len(health_check.dependencies) == 1
        assert health_check.dependencies[0]["name"] == "test_service"
        assert health_check.dependencies[0]["check"] == test_check
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, health_check):
        """测试健康检查端点"""
        # 添加一些依赖检查
        def good_check():
            return {"status": "ok"}
        
        def bad_check():
            return False
        
        health_check.add_dependency_check("good_service", good_check)
        health_check.add_dependency_check("bad_service", bad_check)
        
        result = await health_check.health()
        
        assert "timestamp" in result
        assert "status" in result
        assert "system" in result
        assert "dependencies" in result
        assert len(result["dependencies"]) == 2
        
        # 检查依赖状态
        good_dep = next(d for d in result["dependencies"] if d["name"] == "good_service")
        bad_dep = next(d for d in result["dependencies"] if d["name"] == "bad_service")
        
        assert good_dep["status"] == "healthy"
        assert bad_dep["status"] == "unhealthy"
    
    @pytest.mark.asyncio
    async def test_health_endpoint_with_exception(self, health_check):
        """测试健康检查端点异常处理"""
        def failing_check():
            raise Exception("Test error")
        
        health_check.add_dependency_check("failing_service", failing_check)
        
        result = await health_check.health()
        
        failing_dep = next(d for d in result["dependencies"] if d["name"] == "failing_service")
        assert failing_dep["status"] == "error"
        assert "error" in failing_dep
    
    @pytest.mark.asyncio
    async def test_ready_endpoint(self, health_check):
        """测试就绪检查端点"""
        result = await health_check.ready()
        
        assert "timestamp" in result
        assert result["status"] == "ready"
    
    def test_get_system_health(self, health_check):
        """测试获取系统健康状态"""
        # 这个方法应该是私有的，但我们可以通过反射测试
        system_health = health_check._get_system_health()

        assert isinstance(system_health, dict)
        assert "cpu" in system_health
        assert "memory" in system_health
        assert "disk" in system_health


class TestHealthCheckerIntegration:
    """测试健康检查器集成功能"""
    
    @pytest.fixture
    def mock_config(self):
        """模拟配置"""
        return {
            "health_check": {
                "interval": 1,
                "services": ["database", "redis"]
            }
        }
    
    def test_health_checker_lifecycle(self, mock_config):
        """测试健康检查器生命周期"""
        with patch('src.infrastructure.config.config_manager.ConfigManager') as mock_cm_class:
            mock_cm = Mock()
            mock_cm.get_config.return_value = {
                "interval": 1,
                "services": ["database", "redis"]
            }
            mock_cm_class.return_value = mock_cm
            
            # 传递mock的config_manager而不是config
            checker = HealthChecker(mock_config, mock_cm)
            
            # 启动
            checker.start()
            assert checker.running
            
            # 等待一小段时间让检查执行
            time.sleep(0.1)
            
            # 检查状态
            status = checker.get_status()
            assert isinstance(status, dict)
            
            # 停止
            checker.stop()
            assert not checker.running
    
    def test_health_checker_thread_safety(self, mock_config):
        """测试线程安全性"""
        with patch('src.infrastructure.config.config_manager.ConfigManager') as mock_cm_class:
            mock_cm = Mock()
            mock_cm.get_config.return_value = {
                "interval": 1,
                "services": ["database"]
            }
            mock_cm_class.return_value = mock_cm
            
            # 传递mock的config_manager而不是config
            checker = HealthChecker(mock_config, mock_cm)
            
            # 启动多个线程同时访问
            def access_status():
                for _ in range(10):
                    checker.get_status()
                    time.sleep(0.01)
            
            threads = []
            for _ in range(3):
                thread = threading.Thread(target=access_status)
                threads.append(thread)
                thread.start()
            
            # 等待所有线程完成
            for thread in threads:
                thread.join()
            
            checker.stop()


class TestHealthCheckErrorHandling:
    """测试健康检查错误处理"""
    
    def test_health_checker_with_invalid_config(self):
        """测试无效配置处理"""
        invalid_config = {}
        
        with patch('src.infrastructure.config.config_manager.ConfigManager') as mock_cm_class:
            mock_cm = Mock()
            mock_cm.get_config.return_value = {}
            mock_cm_class.return_value = mock_cm
            
            # 传递mock的config_manager而不是config
            checker = HealthChecker(invalid_config, mock_cm)
            assert checker.check_interval == 10  # 默认值
            assert len(checker.services_to_check) > 0
    
    def test_health_checker_with_missing_services(self):
        """测试缺失服务处理"""
        config = {
            "health_check": {
                "interval": 5,
                "services": ["unknown_service"]
            }
        }
        
        with patch('src.infrastructure.config.config_manager.ConfigManager') as mock_cm_class:
            mock_cm = Mock()
            mock_cm.get_config.return_value = {
                "interval": 5,
                "services": ["unknown_service"]
            }
            mock_cm_class.return_value = mock_cm
            
            # 传递mock的config_manager而不是config
            checker = HealthChecker(config, mock_cm)
            
            # 应该不会抛出异常
            checker.trigger_manual_check("unknown_service")
            
            # 应该记录警告状态
            status = checker.get_status("unknown_service")
            assert len(status) == 0  # 未知服务没有状态


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 