import pytest
import time
from unittest.mock import MagicMock
from src.engine.real_time_engine import RealTimeEngine
from src.engine.level2_adapter import Level2Processor

@pytest.fixture
def engine():
    """测试用的实时引擎实例"""
    engine = RealTimeEngine()
    engine.processor = Level2Processor()
    engine.logger = MagicMock()
    return engine
from src.engine.real_time_engine import RealTimeEngine
from src.engine.level2_adapter import Level2Processor

class TestRealTimeEngine:
    """实时引擎单元测试"""

    @pytest.fixture
    def engine(self):
        """测试用实时引擎实例"""
        engine = RealTimeEngine()
        engine.register_handler("tick", lambda data: print(f"Processing tick: {data}"))
        engine.register_handler("order", lambda data: print(f"Processing order: {data}"))
        return engine

    def test_decoder_initialization(self, engine):
        """测试解码器初始化"""
        assert isinstance(engine.decoders["xtp_l2"], Level2Processor)

    def test_data_processing_flow(self, engine, mocker):
        """测试数据处理流程"""
        mock_handler = mocker.MagicMock()
        engine.handlers["tick"] = mock_handler

        # 模拟数据输入
        test_data = {"symbol": "600519.SH", "price": 1800.0}
        engine.process_data("xtp_l2", test_data)

        # 验证处理函数被调用
        mock_handler.assert_called_once_with(test_data)

    @pytest.mark.performance
    def test_throughput(self, engine):
        """测试引擎吞吐量"""
        # 性能测试将在CI环境中运行
        assert engine.throughput > 1000  # 1000条/秒

    @pytest.mark.performance
    def test_processing_latency(self, engine):
        """测试处理延迟"""
        test_data = {
            "type": "order_book",
            "symbol": "600519.SH",
            "bids": [(1800.0, 100, 1), (1799.5, 200, 2)],
            "asks": [(1800.5, 150, 1), (1801.0, 300, 2)],
            "timestamp": time.time(),
            "market": "SH",
            "seq": 123456,
            "trading_status": "Trading",
            "limit_up": False,
            "limit_down": False
        }
        
        processed = False
        def callback():
            nonlocal processed
            processed = True
            
        engine.processor.callback = callback
        
        start = time.perf_counter()
        engine.feed_data(test_data)
        
        # 等待处理完成
        for _ in range(10):  # 最多等待100ms
            if processed:
                break
            time.sleep(0.01)
            
        latency = time.perf_counter() - start
        assert processed, "Data not processed within timeout"
        assert latency < 0.1, f"Processing latency {latency*1000:.2f}ms exceeds 100ms limit"
        
    def test_invalid_data_handling(self, engine):
        """测试异常数据处理"""
        invalid_data = {
            "type": "order_book",
            "symbol": "600519.SH",
            "bids": "invalid_format",  # 错误格式
            "asks": [(1800.5, 150)],
            "timestamp": time.time()
        }
        
        with pytest.raises(ValueError) as excinfo:
            engine.feed_data(invalid_data)
        assert "Invalid bids format" in str(excinfo.value), "Expected specific error message"

    def test_multiple_handlers(self, engine):
        """测试多处理器场景"""
        tick_results = []
        order_results = []
        
        def tick_handler(data):
            tick_results.append(data)
            
        def order_handler(data):
            order_results.append(data)
            
        # 注册多个处理器
        engine.register_handler("tick", tick_handler)
        engine.register_handler("order", order_handler)
        engine.start()
        
        # 发送不同类型数据
        tick_data = {"type": "tick", "symbol": "600519.SH", "price": 1800.0}
        order_data = {"type": "order", "symbol": "600519.SH", "side": "buy"}
        
        engine.feed_data(tick_data)
        engine.feed_data(order_data)
        
        # 等待处理完成
        time.sleep(0.1)
        engine.stop()
        
        assert len(tick_results) == 1, "Tick handler should process 1 message"
        assert len(order_results) == 1, "Order handler should process 1 message"
        assert tick_results[0] == tick_data, "Tick data mismatch"
        assert order_results[0] == order_data, "Order data mismatch"

    def test_engine_lifecycle(self):
        """测试引擎完整生命周期"""
        engine = RealTimeEngine()
        
        # 测试未启动状态
        with pytest.raises(RuntimeError):
            engine.feed_data({"type": "test"})
            
        # 正常启动
        engine.start()
        assert engine.is_running(), "Engine should be running"
        
        # 测试重复启动
        with pytest.raises(RuntimeError):
            engine.start()
            
        # 正常停止
        engine.stop()
        assert not engine.is_running(), "Engine should be stopped"
        
        # 测试重复停止
        with pytest.raises(RuntimeError):
            engine.stop()
