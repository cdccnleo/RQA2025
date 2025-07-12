import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from src.infrastructure.monitoring.influxdb_store import InfluxDBStore

# 统一mock InfluxDBClient及其API
@pytest.fixture(autouse=True)
def mock_influxdb_client():
    with patch('src.infrastructure.monitoring.influxdb_store.InfluxDBClient') as mock_client:
        mock_instance = MagicMock()
        mock_write_api = MagicMock()
        mock_query_api = MagicMock()
        mock_instance.write_api.return_value = mock_write_api
        mock_instance.query_api.return_value = mock_query_api
        mock_client.return_value = mock_instance
        yield

# Fixtures
@pytest.fixture
def mock_influx():
    """模拟InfluxDB客户端"""
    with patch('influxdb_client.InfluxDBClient') as mock_client:
        mock_write_api = Mock()
        mock_query_api = Mock()

        mock_client.return_value.write_api.return_value = mock_write_api
        mock_client.return_value.query_api.return_value = mock_query_api

        yield {
            'client': mock_client,
            'write_api': mock_write_api,
            'query_api': mock_query_api
        }

@pytest.fixture
def influx_store(mock_influx):
    """测试用InfluxDBStore实例"""
    return InfluxDBStore(
        url="http://localhost:8086",
        token="test-token",
        org="test-org",
        bucket="test-bucket",
        batch_size=2  # 小批量便于测试
    )

# 测试用例
class TestInfluxDBStore:
    def test_write_metric(self, influx_store, mock_influx):
        """测试指标写入"""
        test_time = datetime.utcnow()

        # 写入第一个指标
        influx_store.write_metric(
            measurement="cpu",
            fields={"usage": 50.5},
            tags={"host": "server1"},
            timestamp=test_time
        )

        # 验证尚未写入(缓冲区未满)
        mock_influx['write_api'].write.assert_not_called()

        # 写入第二个指标触发批量写入
        influx_store.write_metric(
            measurement="memory",
            fields={"used": 1024}
        )

        # 验证写入调用
        mock_influx['write_api'].write.assert_called_once()
        args, kwargs = mock_influx['write_api'].write.call_args
        assert kwargs['bucket'] == "test-bucket"
        assert len(kwargs['record']) == 2

        # 验证记录内容
        record1 = kwargs['record'][0]
        assert record1['measurement'] == "cpu"
        assert record1['tags']['host'] == "server1"
        assert record1['fields']['usage'] == 50.5
        assert record1['time'] == test_time

        record2 = kwargs['record'][1]
        assert record2['measurement'] == "memory"
        assert record2['fields']['used'] == 1024

    def test_manual_flush(self, influx_store, mock_influx):
        """测试手动刷新缓冲区"""
        influx_store.write_metric(
            measurement="test",
            fields={"value": 1}
        )

        # 手动刷新
        influx_store.flush()

        # 验证写入调用
        mock_influx['write_api'].write.assert_called_once()
        args, kwargs = mock_influx['write_api'].write.call_args
        assert len(kwargs['record']) == 1

    def test_query(self, influx_store, mock_influx):
        """测试查询功能"""
        mock_result = [{"result": "test"}]
        mock_influx['query_api'].query.return_value = mock_result

        # 执行查询
        result = influx_store.query('from(bucket:"test")')

        # 验证结果
        assert result == mock_result
        mock_influx['query_api'].query.assert_called_once_with('from(bucket:"test")')

    def test_write_error_handling(self, influx_store, mock_influx):
        """测试写入错误处理"""
        mock_influx['write_api'].write.side_effect = Exception("Write failed")

        # 触发写入
        influx_store.write_metric("test", {"value": 1})
        influx_store.write_metric("test", {"value": 2})  # 触发批量写入

        # 验证错误被捕获并记录
        assert len(influx_store.batch_buffer) == 0  # 缓冲区已清空

    def test_context_manager(self, mock_influx):
        """测试上下文管理器"""
        with InfluxDBStore(
            url="http://localhost:8086",
            token="test-token",
            org="test-org",
            bucket="test-bucket"
        ) as store:
            store.write_metric("test", {"value": 1})

        # 验证连接关闭
        mock_influx['client'].close.assert_called_once()

    def test_auto_flush_on_close(self, mock_influx):
        """测试关闭时自动刷新"""
        store = InfluxDBStore(
            url="http://localhost:8086",
            token="test-token",
            org="test-org",
            bucket="test-bucket"
        )
        store.write_metric("test", {"value": 1})
        store.close()

        # 验证关闭前刷新
        mock_influx['write_api'].write.assert_called_once()
        mock_influx['client'].close.assert_called_once()
