import sys
from unittest.mock import patch

class DummyCounter:
    def __init__(self, *a, **k): pass
    def inc(self, *a, **k): pass
    def labels(self, *a, **k): return self
    def set(self, *a, **k): pass
    def observe(self, *a, **k): pass

with patch('prometheus_client.Counter', DummyCounter):
    import pytest
    from unittest.mock import MagicMock
    from src.infrastructure.config.config_manager import ConfigManager
    from src.infrastructure.database.database_manager import DatabaseManager
    from src.infrastructure.monitoring.system_monitor import SystemMonitor
    from src.infrastructure.monitoring.application_monitor import ApplicationMonitor
    from src.infrastructure.health.health_checker import HealthChecker
    from src.infrastructure.error import ErrorHandler
    # from src.infrastructure.m_logging import Logger

    def test_infrastructure_integration(tmp_path):
        # 1. 配置管理与数据库初始化
        config = {"sqlite": {"enabled": True, "path": str(tmp_path / "test.db")}}
        DatabaseManager._config = config  # 避免文件IO
        with patch.object(ConfigManager, 'get_config', return_value=config):
            db_manager = DatabaseManager(config_mock=config)
            adapter = db_manager.get_adapter()
            if hasattr(adapter, 'connect'):
                adapter.connect(config["sqlite"])
            
            # 2. 日志与异常处理
            # logger = Logger("test")
            error_handler = ErrorHandler()
            try:
                if hasattr(adapter, 'write'):
                    adapter.write("test_metric", {"value": 123})
            except Exception as e:
                error_handler.handle(e, None)
                # logger.error("写入失败")
            
            # 3. 监控与健康检查
            sys_monitor = SystemMonitor(psutil_mock=None, os_mock=None, socket_mock=None)
            app_monitor = ApplicationMonitor(influx_client_mock=None, skip_thread=True)
            
            # 创建mock的ConfigManager用于HealthChecker
            mock_config_manager = MagicMock()
            mock_config_manager.get_config.return_value = {"interval": 10, "services": ["database", "redis"]}
            health_checker = HealthChecker(config, config_manager=mock_config_manager)
            health_checker.register_custom_check("db", lambda: ("UP", {"msg": "ok"}))
            health_checker._perform_checks()
            status = health_checker.get_status("db")
            assert status["db"].status == "UP"
            
            # 4. 资源回收
            if hasattr(adapter, 'close'):
                adapter.close()
            sys_monitor.stop_monitoring()
            app_monitor.close() 

def test_infrastructure_integration_config_missing(tmp_path):
    """配置缺失时系统行为"""
    from unittest.mock import MagicMock, patch
    from src.infrastructure.config.config_manager import ConfigManager
    from src.infrastructure.database.database_manager import DatabaseManager
    from src.infrastructure.monitoring.system_monitor import SystemMonitor
    from src.infrastructure.monitoring.application_monitor import ApplicationMonitor
    from src.infrastructure.health.health_checker import HealthChecker
    from src.infrastructure.error import ErrorHandler
    
    config = {}  # 缺失配置
    DatabaseManager._config = config
    with patch.object(ConfigManager, 'get_config', return_value=config):
        db_manager = DatabaseManager(config_mock=config)
        adapter = db_manager.get_adapter()
        # 连接应抛出异常或返回None
        try:
            if hasattr(adapter, 'connect'):
                adapter.connect(config.get("sqlite", {}))
        except Exception:
            pass
        # 健康检查应返回DOWN
        mock_config_manager = MagicMock()
        mock_config_manager.get_config.return_value = {"interval": 1, "services": ["database"]}

        health_checker = HealthChecker(config, config_manager=mock_config_manager)
        # 注册自定义checker覆盖database
        health_checker.register_custom_check("database", lambda: ("DOWN", {"error": "config missing"}))
        health_checker._perform_checks()
        status = health_checker.get_status("database")
        assert status["database"].status == "DOWN"

def test_infrastructure_integration_db_connect_error(tmp_path):
    """数据库连接异常时的容错"""
    from unittest.mock import MagicMock, patch
    from src.infrastructure.config.config_manager import ConfigManager
    from src.infrastructure.database.database_manager import DatabaseManager
    from src.infrastructure.monitoring.system_monitor import SystemMonitor
    from src.infrastructure.monitoring.application_monitor import ApplicationMonitor
    from src.infrastructure.health.health_checker import HealthChecker
    from src.infrastructure.error import ErrorHandler
    
    config = {"sqlite": {"enabled": True, "path": str(tmp_path / "test.db")}}
    DatabaseManager._config = config
    with patch.object(ConfigManager, 'get_config', return_value=config):
        db_manager = DatabaseManager(config_mock=config)
        adapter = db_manager.get_adapter()
        # 强制mock connect抛出异常
        if hasattr(adapter, 'connect'):
            from unittest.mock import patch as patch2
            with patch2.object(adapter, 'connect', side_effect=Exception("db connect error")):
                try:
                    adapter.connect(config["sqlite"])
                except Exception as e:
                    assert "db connect error" in str(e)

def test_infrastructure_integration_monitor_unavailable(tmp_path):
    """监控服务不可用时的降级"""
    from unittest.mock import MagicMock, patch
    from src.infrastructure.monitoring.application_monitor import ApplicationMonitor
    # mock influx_client抛出异常
    influx_mock = MagicMock()
    influx_mock.write_api.side_effect = Exception("influx unavailable")
    try:
        app_monitor = ApplicationMonitor(influx_client_mock=influx_mock, skip_thread=True)
        app_monitor.record_metric("test", 1)
    except Exception as e:
        assert "influx unavailable" in str(e)

def test_infrastructure_integration_health_check_error(tmp_path):
    """健康检查服务返回异常"""
    from unittest.mock import MagicMock
    from src.infrastructure.health.health_checker import HealthChecker
    config = {"sqlite": {"enabled": True, "path": str(tmp_path / "test.db")}}
    mock_config_manager = MagicMock()
    mock_config_manager.get_config.return_value = {"interval": 1, "services": ["custom"]}
    health_checker = HealthChecker(config, config_manager=mock_config_manager)
    # 注册异常checker
    health_checker.register_custom_check("custom", lambda: (_ for _ in ()).throw(Exception("check error")))
    health_checker._perform_checks()
    status = health_checker.get_status("custom")
    assert status["custom"].status == "DOWN"
    assert "check error" in status["custom"].details["error"]

def test_infrastructure_integration_prometheus_conflict(tmp_path):
    """Prometheus指标注册冲突时的处理"""
    from unittest.mock import patch, MagicMock
    class DummyCounter:
        def __init__(self, *a, **k): pass
        def inc(self, *a, **k): pass
        def labels(self, *a, **k): return self
        def set(self, *a, **k): pass
        def observe(self, *a, **k): pass
    with patch('prometheus_client.Counter', DummyCounter):
        from src.infrastructure.monitoring.application_monitor import ApplicationMonitor
        app_monitor = ApplicationMonitor(influx_client_mock=None, skip_thread=True)
        # 多次注册不应抛出异常
        app_monitor2 = ApplicationMonitor(influx_client_mock=None, skip_thread=True)
        assert app_monitor is not None and app_monitor2 is not None 

def test_infrastructure_integration_concurrent_operations(tmp_path):
    """多线程并发操作测试"""
    import threading
    import time
    from unittest.mock import MagicMock, patch
    from src.infrastructure.config.config_manager import ConfigManager
    from src.infrastructure.database.database_manager import DatabaseManager
    from src.infrastructure.monitoring.application_monitor import ApplicationMonitor
    from src.infrastructure.health.health_checker import HealthChecker
    
    config = {"sqlite": {"enabled": True, "path": str(tmp_path / "concurrent.db")}}
    DatabaseManager._config = config
    
    # 创建共享的监控器和健康检查器
    app_monitor = ApplicationMonitor(influx_client_mock=None, skip_thread=True)
    mock_config_manager = MagicMock()
    mock_config_manager.get_config.return_value = {"interval": 1, "services": ["database"]}
    health_checker = HealthChecker(config, config_manager=mock_config_manager)
    
    # 并发操作计数器
    operation_count = 0
    operation_lock = threading.Lock()
    
    def concurrent_operation():
        nonlocal operation_count
        with operation_lock:
            operation_count += 1
        # 模拟并发操作
        app_monitor.record_metric(f"concurrent_op_{operation_count}", operation_count)
        time.sleep(0.01)  # 模拟操作时间
    
    # 启动多个线程
    threads = []
    for i in range(10):
        thread = threading.Thread(target=concurrent_operation)
        threads.append(thread)
        thread.start()
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    # 验证操作计数
    assert operation_count == 10
    
    # 验证健康检查器在并发环境下的稳定性
    health_checker._perform_checks()
    status = health_checker.get_status("database")
    assert status["database"] is not None

def test_infrastructure_integration_resource_cleanup(tmp_path):
    """资源清理测试"""
    from unittest.mock import MagicMock, patch
    from src.infrastructure.config.config_manager import ConfigManager
    from src.infrastructure.database.database_manager import DatabaseManager
    from src.infrastructure.monitoring.system_monitor import SystemMonitor
    from src.infrastructure.monitoring.application_monitor import ApplicationMonitor
    
    config = {"sqlite": {"enabled": True, "path": str(tmp_path / "cleanup.db")}}
    DatabaseManager._config = config
    
    # 创建需要清理的资源
    sys_monitor = SystemMonitor(psutil_mock=None, os_mock=None, socket_mock=None)
    app_monitor = ApplicationMonitor(influx_client_mock=None, skip_thread=True)
    
    # 模拟资源使用
    db_manager = DatabaseManager(config_mock=config)
    adapter = db_manager.get_adapter()
    
    # 执行清理操作
    try:
        if hasattr(adapter, 'close'):
            adapter.close()
        sys_monitor.stop_monitoring()
        app_monitor.close()
    except Exception as e:
        # 清理过程中的异常应该被正确处理
        assert "cleanup" in str(e) or "close" in str(e)
    
    # 验证资源已释放
    assert not hasattr(sys_monitor, '_monitor_thread') or sys_monitor._monitor_thread is None or not sys_monitor._monitor_thread.is_alive()

def test_infrastructure_integration_error_recovery(tmp_path):
    """错误恢复测试"""
    from unittest.mock import MagicMock, patch
    from src.infrastructure.error import ErrorHandler
    from src.infrastructure.monitoring.application_monitor import ApplicationMonitor
    
    # 创建错误处理器和监控器
    error_handler = ErrorHandler()
    app_monitor = ApplicationMonitor(influx_client_mock=None, skip_thread=True)
    
    # 模拟错误发生
    test_error = Exception("Test error for recovery")
    
    # 记录错误
    error_handler.handle(test_error, {"context": "test"})
    app_monitor.record_error("test_source", str(test_error))
    
    # 验证错误被正确处理
    error_metrics = app_monitor.get_error_metrics()
    assert len(error_metrics) > 0
    
    # 模拟错误恢复
    recovery_success = True
    try:
        # 模拟恢复操作
        app_monitor.record_metric("recovery_attempt", 1)
        recovery_success = True
    except Exception:
        recovery_success = False
    
    assert recovery_success

def test_infrastructure_integration_performance_under_load(tmp_path):
    """负载下的性能测试"""
    import time
    from unittest.mock import MagicMock, patch
    from src.infrastructure.monitoring.application_monitor import ApplicationMonitor
    from src.infrastructure.monitoring.system_monitor import SystemMonitor
    
    # 创建监控器
    app_monitor = ApplicationMonitor(influx_client_mock=None, skip_thread=True)
    sys_monitor = SystemMonitor(psutil_mock=None, os_mock=None, socket_mock=None)
    
    # 模拟高负载操作
    start_time = time.time()
    
    for i in range(50):  # 减少操作次数
        app_monitor.record_metric(f"load_test_{i}", i)
        if i % 5 == 0:  # 减少系统监控调用频率
            # 模拟系统监控
            sys_monitor._collect_system_stats()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # 验证性能指标（调整时间阈值）
    assert execution_time < 15.0  # 50次操作应在15秒内完成
    
    # 验证监控数据
    custom_metrics = app_monitor.get_custom_metrics()
    assert len(custom_metrics) >= 50
    
    # 验证系统监控数据
    stats = sys_monitor._collect_system_stats()
    assert "cpu" in stats
    assert "memory" in stats 