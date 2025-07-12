"""基础设施层性能测试"""
import time
import pytest
import statistics
from pathlib import Path
from src.infrastructure.config.config_manager import ConfigManager
from src.infrastructure.error.error_handler import ErrorHandler
from src.infrastructure.monitoring.application_monitor import ApplicationMonitor

@pytest.fixture
def config_dir(tmp_path):
    """准备测试配置目录"""
    config_file = tmp_path / "test.yaml"
    config_file.write_text("key: initial_value")
    return tmp_path

def test_config_reload_performance(config_dir):
    """测试配置热更新性能"""
    config = ConfigManager(config_dir=str(config_dir))
    config.start_watcher()

    # 准备测试数据
    test_file = Path(config_dir) / "test.yaml"
    durations = []

    for i in range(100):
        new_value = f"value_{i}"
        test_file.write_text(f"key: {new_value}")

        start = time.time()
        while config.get("key") != new_value:
            time.sleep(0.001)
        end = time.time()

        durations.append(end - start)

    config.stop_watcher()

    # 输出性能指标
    print(f"\n配置热更新性能指标(100次):")
    print(f"平均延迟: {statistics.mean(durations)*1000:.2f}ms")
    print(f"P95延迟: {statistics.quantiles(durations, n=20)[-1]*1000:.2f}ms")
    print(f"最大延迟: {max(durations)*1000:.2f}ms")

    assert statistics.mean(durations) < 0.1  # 平均延迟应小于100ms

def test_error_handling_throughput():
    """测试错误处理吞吐量"""
    handler = ErrorHandler()

    # 注册简单处理器
    def handle_error(e):
        return "handled"

    handler.register_handler(Exception, handle_error)

    # 性能测试
    start = time.time()
    count = 0

    while time.time() - start < 5:  # 运行5秒
        try:
            raise Exception("test")
        except Exception as e:
            handler.handle(e)
            count += 1

    throughput = count / 5
    print(f"\n错误处理吞吐量: {throughput:.0f}次/秒")

    assert throughput > 5000  # 应达到5000次/秒

def test_monitoring_latency():
    """测试监控数据收集延迟"""
    monitor = ApplicationMonitor()

    # 准备测试数据
    latencies = []

    @monitor.monitor_function()
    def test_function():
        time.sleep(0.001)

    for _ in range(1000):
        start = time.time()
        test_function()
        end = time.time()

        # 获取最新指标
        metrics = monitor.get_function_metrics(name="test_function", limit=1)
        if metrics:
            latencies.append(end - start)

    # 输出性能指标
    print(f"\n监控数据收集延迟(1000次):")
    print(f"平均延迟: {statistics.mean(latencies)*1000:.2f}ms")
    print(f"P99延迟: {statistics.quantiles(latencies, n=100)[-1]*1000:.2f}ms")

    assert statistics.mean(latencies) < 0.005  # 平均延迟应小于5ms
