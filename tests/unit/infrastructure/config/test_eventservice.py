import pytest
from unittest.mock import MagicMock, ANY
from src.infrastructure.config.services import EventService, VersionService

class TestEventService:
    """事件服务单元测试"""

    @pytest.fixture
    def version_service(self):
        return VersionService()

    @pytest.fixture
    def event_service(self, version_service):
        return EventService(version_service)

    @pytest.mark.unit
    def test_event_pub_sub(self, event_service):
        """测试基本事件发布订阅"""
        # 创建模拟订阅者
        subscriber = MagicMock()

        # 订阅配置变更事件
        event_service.subscribe("config_updated", subscriber)

        # 发布事件
        event_data = {"key": "db.host", "value": "new_host"}
        event_service.publish("config_updated", event_data)

        # 验证订阅者被调用，允许包含版本信息
        call_args = subscriber.call_args[0][0]
        assert call_args['key'] == event_data['key']
        assert call_args['value'] == event_data['value']

    @pytest.mark.unit
    def test_dead_letter_queue(self, event_service):
        """测试死信队列处理"""
        from src.infrastructure.config.error.exceptions import EventError
        
        # 创建会抛出异常的订阅者
        def faulty_subscriber(event):
            raise Exception("模拟处理失败")

        event_service.subscribe("config_updated", faulty_subscriber)
        event_data = {"key": "db.port", "value": 3306}
        
        with pytest.raises(EventError):
            event_service.publish("config_updated", event_data)
        
        # 验证死信队列
        assert len(event_service._dead_letters) == 1
        dead_letter = event_service._dead_letters[0]
        assert dead_letter["event"] == "config_updated"
        assert dead_letter["payload"] == event_data
        assert "模拟处理失败" in dead_letter["error"]

        # 验证死信队列
        dead_letters = event_service.get_dead_letters()
        assert len(dead_letters) == 1
        assert dead_letters[0]["event"] == "config_updated"  # 验证事件类型
        assert dead_letters[0]["payload"] == event_data  # 验证payload数据
        assert "处理失败" in dead_letters[0]["error"]

    @pytest.mark.unit
    def test_event_filtering(self, event_service):
        """测试事件过滤"""
        # 创建带过滤器的订阅者
        filtered_subscriber = MagicMock()

        def filter_func(event):
            # 过滤器需要处理自动添加的version字段
            return event.get("priority") == "high"

        event_service.subscribe("alert", filtered_subscriber, filter_func)

        # 发布不同优先级事件
        event_service.publish("alert", {"msg": "Low priority", "priority": "low"})
        event_service.publish("alert", {"msg": "High priority", "priority": "high"})

        # 验证只有高优先级事件被处理，考虑自动添加的version字段
        expected_call = {
            "msg": "High priority", 
            "priority": "high",
            "version": ANY  # 使用unittest.mock.ANY匹配任意版本值
        }
        filtered_subscriber.assert_called_once_with(expected_call)

    @pytest.mark.unit
    def test_high_volume_events(self, event_service):
        """测试高并发事件处理"""
        import threading

        results = []
        event_count = 100

        def subscriber(event):
            results.append(event)

        event_service.subscribe("high_volume", subscriber)

        def publisher():
            for i in range(event_count):
                event_service.publish("high_volume", {"index": i})

        # 启动多个发布者线程
        threads = [threading.Thread(target=publisher) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 验证所有事件都被处理
        assert len(results) == event_count * 5
