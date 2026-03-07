"""
标准接口模块测试

测试覆盖:
- 枚举类 (ServiceStatus)
- 数据类 (DataRequest, DataResponse, Event, FeatureRequest, FeatureResponse)
- Protocol接口的Mock实现验证
- 抽象基类 (TradingStrategy)
"""

import pytest
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from unittest.mock import Mock, MagicMock

from src.infrastructure.interfaces.standard_interfaces import (
    ServiceStatus,
    DataRequest,
    DataResponse,
    IServiceProvider,
    ICacheProvider,
    ILogger,
    IConfigProvider,
    IHealthCheck,
    Event,
    IEventBus,
    IConfigEventBus,
    IConfigVersionManager,
    IEventSubscriber,
    IMonitor,
    FeatureRequest,
    FeatureResponse,
    IFeatureProcessor,
    TradingStrategy,
)


class TestServiceStatus:
    """服务状态枚举测试"""
    
    def test_service_status_values(self):
        """测试服务状态枚举值"""
        assert ServiceStatus.RUNNING.value == "running"
        assert ServiceStatus.STOPPED.value == "stopped"
        assert ServiceStatus.ERROR.value == "error"
        assert ServiceStatus.STARTING.value == "starting"
        assert ServiceStatus.STOPPING.value == "stopping"
    
    def test_service_status_comparison(self):
        """测试服务状态比较"""
        status1 = ServiceStatus.RUNNING
        status2 = ServiceStatus.RUNNING
        assert status1 == status2
        assert status1 != ServiceStatus.STOPPED


class TestDataRequest:
    """数据请求数据类测试"""
    
    def test_data_request_creation(self):
        """测试数据请求创建"""
        request = DataRequest(
            symbol="000001",
            market="CN",
            data_type="stock"
        )
        assert request.symbol == "000001"
        assert request.market == "CN"
        assert request.data_type == "stock"
    
    def test_data_request_defaults(self):
        """测试数据请求默认值"""
        request = DataRequest(symbol="000001")
        assert request.market == "CN"
        assert request.data_type == "stock"
        assert request.interval == "1d"
        assert request.start_date is None
        assert request.end_date is None
        assert request.params is None
    
    def test_data_request_to_dict(self):
        """测试数据请求转字典"""
        request = DataRequest(
            symbol="000001",
            market="US",
            data_type="etf",
            start_date="2024-01-01",
            end_date="2024-12-31",
            interval="1h",
            params={"adjust": "hfq"}
        )
        result = request.to_dict()
        
        assert result["symbol"] == "000001"
        assert result["market"] == "US"
        assert result["data_type"] == "etf"
        assert result["start_date"] == "2024-01-01"
        assert result["end_date"] == "2024-12-31"
        assert result["interval"] == "1h"
        assert result["params"] == {"adjust": "hfq"}
    
    def test_data_request_to_dict_with_none_params(self):
        """测试数据请求转字典（无参数）"""
        request = DataRequest(symbol="000001")
        result = request.to_dict()
        assert result["params"] == {}


class TestDataResponse:
    """数据响应数据类测试"""
    
    def test_data_response_creation(self):
        """测试数据响应创建"""
        request = DataRequest(symbol="000001")
        response = DataResponse(
            request=request,
            data={"price": 100.0},
            success=True
        )
        
        assert response.request == request
        assert response.data == {"price": 100.0}
        assert response.success is True
        assert response.error_message is None
        assert response.timestamp is not None
    
    def test_data_response_with_error(self):
        """测试数据响应（失败）"""
        request = DataRequest(symbol="000001")
        response = DataResponse(
            request=request,
            data=None,
            success=False,
            error_message="Connection timeout"
        )
        
        assert response.success is False
        assert response.error_message == "Connection timeout"
        assert response.data is None
    
    def test_data_response_timestamp_auto_generation(self):
        """测试时间戳自动生成"""
        request = DataRequest(symbol="000001")
        before = datetime.now().timestamp()
        response = DataResponse(request=request, data={})
        after = datetime.now().timestamp()
        
        assert before <= response.timestamp <= after


class TestEvent:
    """事件数据类测试"""
    
    def test_event_creation(self):
        """测试事件创建"""
        event = Event(
            event_type="data.updated",
            data={"symbol": "000001"},
            source="market_service"
        )
        
        assert event.event_type == "data.updated"
        assert event.data == {"symbol": "000001"}
        assert event.source == "market_service"
        assert event.timestamp is not None
    
    def test_event_default_values(self):
        """测试事件默认值"""
        event = Event(event_type="system.started")
        assert event.source == "system"
        assert event.data is None
        assert event.timestamp is not None


class TestFeatureRequest:
    """特征请求数据类测试"""
    
    def test_feature_request_creation(self):
        """测试特征请求创建"""
        request = FeatureRequest(
            data=[[1, 2, 3], [4, 5, 6]],
            feature_names=["f1", "f2", "f3"],
            config={"normalize": True}
        )
        
        assert request.data == [[1, 2, 3], [4, 5, 6]]
        assert request.feature_names == ["f1", "f2", "f3"]
        assert request.config == {"normalize": True}
    
    def test_feature_request_to_dict(self):
        """测试特征请求转字典"""
        request = FeatureRequest(
            data=[1, 2, 3],
            feature_names=["ma5", "ma10"]
        )
        result = request.to_dict()
        
        assert result["data"] == [1, 2, 3]
        assert result["feature_names"] == ["ma5", "ma10"]
        assert result["config"] == {}
        assert result["metadata"] == {}


class TestFeatureResponse:
    """特征响应数据类测试"""
    
    def test_feature_response_creation(self):
        """测试特征响应创建"""
        response = FeatureResponse(
            features=[[1.0, 2.0], [3.0, 4.0]],
            feature_names=["ma5", "ma10"],
            success=True
        )
        
        assert response.features == [[1.0, 2.0], [3.0, 4.0]]
        assert response.feature_names == ["ma5", "ma10"]
        assert response.success is True
        assert response.error_message is None
    
    def test_feature_response_with_error(self):
        """测试特征响应（失败）"""
        response = FeatureResponse(
            features=None,
            feature_names=[],
            success=False,
            error_message="Invalid data format"
        )
        
        assert response.success is False
        assert response.error_message == "Invalid data format"


class TestProtocolInterfaces:
    """Protocol接口测试（通过Mock实现验证）"""
    
    def test_service_provider_interface(self):
        """测试服务提供者接口"""
        # Mock实现
        provider = Mock(spec=IServiceProvider)
        provider.get_service.return_value = "service_instance"
        provider.register_service.return_value = True
        provider.unregister_service.return_value = True
        
        # 验证接口方法存在且可调用
        assert provider.get_service("test") == "service_instance"
        assert provider.register_service("test", object()) is True
        assert provider.unregister_service("test") is True
    
    def test_cache_provider_interface(self):
        """测试缓存提供者接口"""
        cache = Mock(spec=ICacheProvider)
        cache.get.return_value = "cached_value"
        cache.set.return_value = True
        cache.delete.return_value = True
        cache.clear.return_value = True
        
        assert cache.get("key") == "cached_value"
        assert cache.set("key", "value", ttl=3600) is True
        assert cache.delete("key") is True
        assert cache.clear() is True
    
    def test_logger_interface(self):
        """测试日志接口"""
        logger = Mock(spec=ILogger)
        
        logger.info("info message", extra={"key": "value"})
        logger.warning("warning message")
        logger.error("error message")
        logger.debug("debug message")
        
        logger.info.assert_called_once()
        logger.warning.assert_called_once()
        logger.error.assert_called_once()
        logger.debug.assert_called_once()
    
    def test_config_provider_interface(self):
        """测试配置提供者接口"""
        config = Mock(spec=IConfigProvider)
        config.get_config.return_value = "config_value"
        config.set_config.return_value = True
        config.load_config.return_value = True
        config.save_config.return_value = True
        
        assert config.get_config("key", default="default") == "config_value"
        assert config.set_config("key", "value") is True
        assert config.load_config("config.yaml") is True
        assert config.save_config("config.yaml") is True
    
    def test_health_check_interface(self):
        """测试健康检查接口"""
        health = Mock(spec=IHealthCheck)
        health.health_check.return_value = {"status": "healthy"}
        health.is_healthy.return_value = True
        
        assert health.health_check() == {"status": "healthy"}
        assert health.is_healthy() is True
    
    def test_event_bus_interface(self):
        """测试事件总线接口"""
        bus = Mock(spec=IEventBus)
        event = Event(event_type="test.event")
        handler = Mock()
        
        bus.publish.return_value = "event_id_123"
        bus.subscribe.return_value = True
        bus.unsubscribe.return_value = True
        
        assert bus.publish(event) == "event_id_123"
        assert bus.subscribe("test.event", handler) is True
        assert bus.unsubscribe("test.event", handler) is True
    
    def test_config_event_bus_interface(self):
        """测试配置事件总线接口"""
        bus = Mock(spec=IConfigEventBus)
        handler = Mock()
        
        bus.subscribe.return_value = "sub_id_123"
        bus.unsubscribe.return_value = True
        bus.get_subscribers.return_value = {"sub1": handler}
        bus.get_dead_letters.return_value = []
        
        # 测试订阅/取消订阅
        sub_id = bus.subscribe("config.updated", handler)
        assert sub_id == "sub_id_123"
        assert bus.unsubscribe("config.updated", sub_id) is True
        
        # 测试事件发布
        bus.publish("config.updated", {"key": "value"})
        bus.publish.assert_called_once()
        
        # 测试通知方法
        bus.notify_config_updated("key", "old", "new")
        bus.notify_config_error("error msg", {"details": "info"})
        bus.notify_config_loaded("file.yaml", {"key": "value"})
        bus.emit_config_changed("key", "old", "new")
        bus.emit_config_loaded("file.yaml")
        
        # 测试死信队列
        assert bus.get_dead_letters() == []
        bus.clear_dead_letters()
    
    def test_config_version_manager_interface(self):
        """测试配置版本管理器接口"""
        version_mgr = Mock(spec=IConfigVersionManager)
        version_mgr.get_latest_version.return_value = "1.2.3"
        version_mgr._versions = {"config1": [{"version": "1.2.3"}]}
        
        # 测试获取最新版本
        latest = version_mgr.get_latest_version()
        assert latest == "1.2.3"
        
        # 测试_versions属性
        assert version_mgr._versions == {"config1": [{"version": "1.2.3"}]}
    
    def test_event_subscriber_interface(self):
        """测试事件订阅者接口"""
        # Mock事件总线
        mock_bus = Mock(spec=IConfigEventBus)
        
        # 创建订阅者Mock
        subscriber = Mock(spec=IEventSubscriber)
        subscriber.handle_event.return_value = True
        
        # 测试处理事件
        event_data = {"type": "config.updated", "key": "test_key"}
        result = subscriber.handle_event(event_data)
        assert result is True
        subscriber.handle_event.assert_called_once_with(event_data)
    
    def test_monitor_interface(self):
        """测试监控接口"""
        monitor = Mock(spec=IMonitor)
        monitor.get_metric.return_value = {"value": 100, "timestamp": 123456}
        monitor.get_all_metrics.return_value = {"metric1": {"value": 100}}

        # 测试记录指标
        monitor.record_metric("cpu_usage", 75.5, tags={"host": "server1"})
        monitor.record_metric.assert_called_once_with("cpu_usage", 75.5, tags={"host": "server1"})

        # 测试获取单个指标
        metric = monitor.get_metric("cpu_usage")
        assert metric == {"value": 100, "timestamp": 123456}
        monitor.get_metric.assert_called_once_with("cpu_usage")

        # 测试获取所有指标
        all_metrics = monitor.get_all_metrics()
        assert all_metrics == {"metric1": {"value": 100}}
        monitor.get_all_metrics.assert_called_once()
    
    def test_feature_processor_interface(self):
        """测试特征处理器接口"""
        processor = Mock(spec=IFeatureProcessor)
        
        request = FeatureRequest(data=[1, 2, 3])
        response = FeatureResponse(
            features=[[1.0, 2.0]], 
            feature_names=["f1", "f2"],
            success=True
        )
        
        processor.process.return_value = response
        processor.get_supported_features.return_value = ["f1", "f2", "f3"]
        processor.validate_data.return_value = True
        
        result = processor.process(request)
        assert result.success is True
        assert result.feature_names == ["f1", "f2"]
        
        assert processor.get_supported_features() == ["f1", "f2", "f3"]
        assert processor.validate_data([1, 2, 3]) is True

    def test_event_bus_interface(self):
        """测试事件总线接口"""
        bus = Mock(spec=IEventBus)
        bus.publish.return_value = None
        bus.subscribe.return_value = "subscription_123"
        bus.unsubscribe.return_value = True

        # 测试发布事件
        bus.publish("market_data", {"price": 100.5})
        bus.publish.assert_called_once_with("market_data", {"price": 100.5})

        # 测试订阅事件
        sub_id = bus.subscribe("handler_func", "market_data")
        assert sub_id == "subscription_123"
        bus.subscribe.assert_called_once_with("handler_func", "market_data")

        # 测试取消订阅
        result = bus.unsubscribe("subscription_123")
        assert result == True
        bus.unsubscribe.assert_called_once_with("subscription_123")

    def test_config_event_bus_interface(self):
        """测试配置事件总线接口"""
        bus = Mock(spec=IConfigEventBus)
        handler = Mock()

        bus.subscribe.return_value = "sub_id_123"
        bus.unsubscribe.return_value = True
        bus.get_subscribers.return_value = {"sub1": handler}
        bus.get_dead_letters.return_value = []

        # 测试订阅/取消订阅
        sub_id = bus.subscribe("config.updated", handler)
        assert sub_id == "sub_id_123"
        assert bus.unsubscribe("config.updated", sub_id) is True

        # 测试事件发布
        bus.publish("config.updated", {"key": "database.host", "new_value": "new_host"})
        bus.publish.assert_called_once_with("config.updated", {"key": "database.host", "new_value": "new_host"})

        # 测试获取订阅者
        subscribers = bus.get_subscribers("config.updated")
        assert subscribers == {"sub1": handler}
        bus.get_subscribers.assert_called_once_with("config.updated")

        # 测试获取死信队列
        dead_letters = bus.get_dead_letters()
        assert dead_letters == []
        bus.get_dead_letters.assert_called_once()

    def test_config_version_manager_interface(self):
        """测试配置版本管理器接口"""
        manager = Mock(spec=IConfigVersionManager)
        manager.get_latest_version.return_value = "2.1.0"

        # 测试获取最新版本
        version = manager.get_latest_version()
        assert version == "2.1.0"
        manager.get_latest_version.assert_called_once()

        # 测试访问版本数据属性
        mock_versions = {
            "database.config": [
                {"version": "2.1.0", "changes": ["update host"], "timestamp": "2023-12-01"},
                {"version": "2.0.0", "changes": ["add cache"], "timestamp": "2023-11-15"}
            ]
        }
        manager._versions = mock_versions
        assert manager._versions == mock_versions

    def test_event_subscriber_interface(self):
        """测试事件订阅者接口"""
        # Mock事件总线
        mock_bus = Mock(spec=IConfigEventBus)

        # 创建订阅者Mock
        subscriber = Mock(spec=IEventSubscriber)
        subscriber.handle_event.return_value = True

        # 测试处理事件
        event_data = {"type": "config.updated", "key": "test_key"}
        result = subscriber.handle_event(event_data)
        assert result is True
        subscriber.handle_event.assert_called_once_with(event_data)

    def test_monitor_interface(self):
        """测试监控接口"""
        monitor = Mock(spec=IMonitor)
        monitor.get_metric.return_value = {"value": 100, "timestamp": 123456}
        monitor.get_all_metrics.return_value = {"metric1": {"value": 100}}

        # 测试记录指标
        monitor.record_metric("cpu_usage", 75.5, tags={"host": "server1"})
        monitor.record_metric.assert_called_once_with("cpu_usage", 75.5, tags={"host": "server1"})

        # 测试获取单个指标
        metric = monitor.get_metric("cpu_usage")
        assert metric == {"value": 100, "timestamp": 123456}
        monitor.get_metric.assert_called_once_with("cpu_usage")

        # 测试获取所有指标
        all_metrics = monitor.get_all_metrics()
        assert all_metrics == {"metric1": {"value": 100}}
        monitor.get_all_metrics.assert_called_once()


class TestTradingStrategyAbstractClass:
    """交易策略抽象类测试"""
    
    def test_trading_strategy_cannot_instantiate_directly(self):
        """测试不能直接实例化抽象类"""
        with pytest.raises(TypeError):
            TradingStrategy()  # type: ignore
    
    def test_trading_strategy_concrete_implementation(self):
        """测试具体实现类"""
        class ConcreteStrategy(TradingStrategy):
            def initialize(self, config: Dict[str, Any]) -> bool:
                return True
            
            def generate_signals(self, data: Any) -> List[Dict[str, Any]]:
                return [{"action": "buy", "symbol": "000001"}]
            
            def execute_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                return [{"order_id": "123", "status": "filled"}]
            
            def evaluate_performance(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
                return {"total_return": 0.15, "sharpe_ratio": 1.5}
            
            def update_parameters(self, new_params: Dict[str, Any]) -> bool:
                return True
            
            def get_strategy_info(self) -> Dict[str, Any]:
                return {"name": "TestStrategy", "version": "1.0"}
        
        # 测试具体实现
        strategy = ConcreteStrategy()
        
        assert strategy.initialize({"param": "value"}) is True
        
        signals = strategy.generate_signals(data=[])
        assert len(signals) == 1
        assert signals[0]["action"] == "buy"
        
        executions = strategy.execute_signals(signals)
        assert len(executions) == 1
        assert executions[0]["status"] == "filled"
        
        performance = strategy.evaluate_performance([])
        assert "total_return" in performance
        assert performance["sharpe_ratio"] == 1.5
        
        assert strategy.update_parameters({"new_param": "new_value"}) is True
        
        info = strategy.get_strategy_info()
        assert info["name"] == "TestStrategy"
        assert info["version"] == "1.0"
    
    def test_trading_strategy_missing_methods_raises_error(self):
        """测试缺少方法的实现会报错"""
        class IncompleteStrategy(TradingStrategy):
            def initialize(self, config: Dict[str, Any]) -> bool:
                return True
            # 缺少其他方法
        
        with pytest.raises(TypeError):
            IncompleteStrategy()  # type: ignore


class TestIntegrationScenarios:
    """集成场景测试"""
    
    def test_data_request_response_flow(self):
        """测试数据请求-响应流程"""
        # 创建请求
        request = DataRequest(
            symbol="000001",
            market="CN",
            data_type="stock",
            start_date="2024-01-01",
            end_date="2024-12-31"
        )
        
        # 模拟处理
        request_dict = request.to_dict()
        assert request_dict["symbol"] == "000001"
        
        # 创建响应
        response = DataResponse(
            request=request,
            data={"close": [100, 101, 102]},
            success=True
        )
        
        assert response.success is True
        assert response.request.symbol == "000001"
    
    def test_feature_request_response_flow(self):
        """测试特征请求-响应流程"""
        # 创建特征请求
        request = FeatureRequest(
            data=[[100, 101, 102, 103, 104]],
            feature_names=["ma5"],
            config={"window": 5}
        )
        
        # 模拟处理
        request_dict = request.to_dict()
        assert "ma5" in request_dict["feature_names"]
        
        # 创建响应
        response = FeatureResponse(
            features=[[102.0]],  # MA5 结果
            feature_names=["ma5"],
            metadata={"window": 5},
            success=True
        )
        
        assert response.success is True
        assert response.features[0][0] == 102.0
    
    def test_event_publish_subscribe_flow(self):
        """测试事件发布-订阅流程"""
        # 创建事件总线Mock
        bus = Mock(spec=IEventBus)
        bus.subscribe.return_value = True
        bus.publish.return_value = "event_id"
        
        # 订阅事件
        handler = Mock()
        assert bus.subscribe("data.updated", handler) is True
        
        # 发布事件
        event = Event(
            event_type="data.updated",
            data={"symbol": "000001", "price": 100.5}
        )
        event_id = bus.publish(event)
        assert event_id == "event_id"
        
        # 验证调用
        bus.subscribe.assert_called_once_with("data.updated", handler)
        bus.publish.assert_called_once_with(event)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

