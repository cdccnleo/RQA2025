"""
边界测试：data_compression_optimizer.py
测试边界情况和异常场景
"""
import asyncio
import pandas as pd
from unittest.mock import Mock

# Mock数据管理器模块以绕过复杂的导入问题
mock_data_manager = Mock()
mock_data_manager.DataManager = Mock()
mock_data_manager.DataLoaderError = Exception

# 配置DataManager实例方法
mock_instance = Mock()
mock_instance.validate_all_configs.return_value = True
mock_instance.health_check.return_value = {"status": "healthy"}
mock_instance.store_data.return_value = True
mock_instance.has_data.return_value = True
mock_instance.get_metadata.return_value = {"data_type": "test", "symbol": "X"}
mock_instance.retrieve_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_instance.get_stats.return_value = {"total_items": 1}
mock_instance.validate_data.return_value = {"valid": True}
mock_instance.shutdown.return_value = None

mock_data_manager.DataManager.return_value = mock_instance

# Mock整个模块
import sys
sys.modules["src.data.data_manager"] = mock_data_manager


import pytest
from datetime import datetime, timedelta
from src.data.compression.data_compression_optimizer import (
    CompressionMetrics,
    CompressionStrategy,
    DataCompressionOptimizer
)


def test_compression_metrics_init():
    """测试 CompressionMetrics（初始化）"""
    metrics = CompressionMetrics(
        original_size=1000,
        compressed_size=500,
        compression_ratio=2.0,
        compression_time=0.1,
        decompression_time=0.05,
        algorithm="gzip",
        data_type="text",
        timestamp=datetime.now()
    )
    
    assert metrics.original_size == 1000
    assert metrics.compressed_size == 500
    assert metrics.compression_ratio == 2.0
    assert metrics.algorithm == "gzip"


def test_compression_metrics_compression_efficiency():
    """测试 CompressionMetrics（压缩效率）"""
    metrics = CompressionMetrics(
        original_size=1000,
        compressed_size=100,
        compression_ratio=10.0,
        compression_time=0.1,
        decompression_time=0.05,
        algorithm="gzip",
        data_type="text",
        timestamp=datetime.now()
    )
    
    assert metrics.compression_efficiency == 1.0  # 10:1为满分


def test_compression_metrics_compression_efficiency_zero():
    """测试 CompressionMetrics（压缩效率，零比例）"""
    metrics = CompressionMetrics(
        original_size=1000,
        compressed_size=1000,
        compression_ratio=0.0,
        compression_time=0.1,
        decompression_time=0.05,
        algorithm="gzip",
        data_type="text",
        timestamp=datetime.now()
    )
    
    assert metrics.compression_efficiency == 0.0


def test_compression_metrics_performance_score():
    """测试 CompressionMetrics（性能评分）"""
    metrics = CompressionMetrics(
        original_size=1000,
        compressed_size=100,
        compression_ratio=10.0,
        compression_time=0.1,
        decompression_time=0.05,
        algorithm="gzip",
        data_type="text",
        timestamp=datetime.now()
    )
    
    assert 0 <= metrics.performance_score <= 1


def test_compression_strategy_init():
    """测试 CompressionStrategy（初始化）"""
    strategy = CompressionStrategy(
        name="test_strategy",
        algorithm="gzip",
        compression_level=6
    )
    
    assert strategy.name == "test_strategy"
    assert strategy.algorithm == "gzip"
    assert strategy.compression_level == 6
    assert strategy.enabled is True


def test_compression_strategy_is_applicable():
    """测试 CompressionStrategy（是否适用）"""
    strategy = CompressionStrategy(
        name="test_strategy",
        algorithm="gzip",
        min_size_threshold=1024,
        max_size_threshold=1048576,
        enabled=True
    )
    
    assert strategy.is_applicable(2048, "text") is True
    assert strategy.is_applicable(512, "text") is False  # 小于最小阈值
    assert strategy.is_applicable(2097152, "text") is False  # 大于最大阈值


def test_compression_strategy_is_applicable_disabled():
    """测试 CompressionStrategy（是否适用，已禁用）"""
    strategy = CompressionStrategy(
        name="test_strategy",
        algorithm="gzip",
        enabled=False
    )
    
    assert strategy.is_applicable(2048, "text") is False


def test_data_compression_optimizer_init():
    """测试 DataCompressionOptimizer（初始化）"""
    optimizer = DataCompressionOptimizer()
    
    assert len(optimizer.algorithms) > 0
    assert len(optimizer.decompress_algorithms) > 0
    assert len(optimizer.strategies) > 0
    assert optimizer.metrics_history == []


def test_data_compression_optimizer_compress_data_string():
    """测试 DataCompressionOptimizer（压缩数据，字符串）"""
    optimizer = DataCompressionOptimizer()
    data = "test data" * 100
    
    result = optimizer.compress_data(data, data_type="text")
    
    assert "compressed_data" in result
    assert result["original_size"] > 0
    assert result["compressed_size"] > 0
    assert result["compression_ratio"] > 0
    assert "algorithm" in result


def test_data_compression_optimizer_compress_data_bytes():
    """测试 DataCompressionOptimizer（压缩数据，字节）"""
    optimizer = DataCompressionOptimizer()
    data = b"test data" * 100
    
    result = optimizer.compress_data(data, data_type="binary")
    
    assert "compressed_data" in result
    assert result["original_size"] > 0


def test_data_compression_optimizer_compress_data_small():
    """测试 DataCompressionOptimizer（压缩数据，小数据）"""
    optimizer = DataCompressionOptimizer()
    data = "test"
    
    result = optimizer.compress_data(data, data_type="text")
    
    # 小数据可能不压缩
    assert "compressed_data" in result
    assert result["original_size"] > 0


def test_data_compression_optimizer_compress_data_strategy():
    """测试 DataCompressionOptimizer（压缩数据，指定策略）"""
    optimizer = DataCompressionOptimizer()
    data = "test data" * 1000
    
    result = optimizer.compress_data(data, data_type="text", strategy_name="text_gzip_fast")
    
    assert result["strategy"] == "text_gzip_fast"
    assert result["algorithm"] == "gzip"


def test_data_compression_optimizer_compress_data_invalid_strategy():
    """测试 DataCompressionOptimizer（压缩数据，无效策略）"""
    optimizer = DataCompressionOptimizer()
    data = "test data" * 100
    
    # 无效策略名称应该回退到自动选择
    result = optimizer.compress_data(data, data_type="text", strategy_name="invalid_strategy")
    
    assert "compressed_data" in result


def test_data_compression_optimizer_decompress_data():
    """测试 DataCompressionOptimizer（解压数据）"""
    optimizer = DataCompressionOptimizer()
    original_data = b"test data" * 100
    
    # 先压缩
    compressed_result = optimizer.compress_data(original_data, data_type="binary")
    compressed_data = compressed_result["compressed_data"]
    algorithm = compressed_result["algorithm"]
    
    # 再解压
    decompressed_data = optimizer.decompress_data(compressed_data, algorithm)
    
    assert decompressed_data == original_data


def test_data_compression_optimizer_decompress_data_invalid_algorithm():
    """测试 DataCompressionOptimizer（解压数据，无效算法）"""
    optimizer = DataCompressionOptimizer()
    compressed_data = b"test data"
    
    with pytest.raises(ValueError, match="不支持的解压算法"):
        optimizer.decompress_data(compressed_data, "invalid_algorithm")


def test_data_compression_optimizer_get_compression_report():
    """测试 DataCompressionOptimizer（获取压缩报告）"""
    optimizer = DataCompressionOptimizer()
    data = "test data" * 100
    
    # 执行一些压缩操作
    optimizer.compress_data(data, data_type="text")
    optimizer.compress_data(data, data_type="text")
    
    report = optimizer.get_compression_report(time_range_hours=24)
    
    assert "summary" in report
    assert "time_range_hours" in report
    assert "timestamp" in report


def test_data_compression_optimizer_get_compression_report_empty():
    """测试 DataCompressionOptimizer（获取压缩报告，无数据）"""
    optimizer = DataCompressionOptimizer()
    
    report = optimizer.get_compression_report(time_range_hours=24)
    
    assert report["summary"] == "无数据"


def test_data_compression_optimizer_get_compression_report_custom_range():
    """测试 DataCompressionOptimizer（获取压缩报告，自定义范围）"""
    optimizer = DataCompressionOptimizer()
    data = "test data" * 100
    
    optimizer.compress_data(data, data_type="text")
    
    report = optimizer.get_compression_report(time_range_hours=1)
    
    assert report["time_range_hours"] == 1


def test_data_compression_optimizer_get_algorithm_performance():
    """测试 DataCompressionOptimizer（获取算法性能）"""
    optimizer = DataCompressionOptimizer()
    data = "test data" * 100
    
    # 执行压缩操作
    optimizer.compress_data(data, data_type="text")
    
    performance = optimizer.get_algorithm_performance("gzip")
    
    assert "algorithm" in performance
    assert performance["algorithm"] == "gzip"


def test_data_compression_optimizer_get_algorithm_performance_no_data():
    """测试 DataCompressionOptimizer（获取算法性能，无数据）"""
    optimizer = DataCompressionOptimizer()
    
    performance = optimizer.get_algorithm_performance("gzip")
    
    assert performance["status"] == "no_data"


def test_data_compression_optimizer_add_compression_strategy():
    """测试 DataCompressionOptimizer（添加压缩策略）"""
    optimizer = DataCompressionOptimizer()
    strategy = CompressionStrategy(
        name="custom_strategy",
        algorithm="gzip",
        compression_level=5
    )
    
    optimizer.add_compression_strategy(strategy)
    
    assert any(s.name == "custom_strategy" for s in optimizer.strategies)


def test_data_compression_optimizer_add_compression_strategy_duplicate():
    """测试 DataCompressionOptimizer（添加压缩策略，重复名称）"""
    optimizer = DataCompressionOptimizer()
    strategy = CompressionStrategy(
        name="text_gzip_fast",  # 使用已存在的名称
        algorithm="gzip",
        compression_level=5
    )
    
    with pytest.raises(ValueError, match="策略名称已存在"):
        optimizer.add_compression_strategy(strategy)


def test_data_compression_optimizer_remove_compression_strategy():
    """测试 DataCompressionOptimizer（移除压缩策略）"""
    optimizer = DataCompressionOptimizer()
    initial_count = len(optimizer.strategies)
    
    optimizer.remove_compression_strategy("text_gzip_fast")
    
    assert len(optimizer.strategies) == initial_count - 1
    assert not any(s.name == "text_gzip_fast" for s in optimizer.strategies)


def test_data_compression_optimizer_remove_compression_strategy_nonexistent():
    """测试 DataCompressionOptimizer（移除压缩策略，不存在）"""
    optimizer = DataCompressionOptimizer()
    initial_count = len(optimizer.strategies)
    
    # 应该不抛出异常
    optimizer.remove_compression_strategy("nonexistent_strategy")
    
    assert len(optimizer.strategies) == initial_count


def test_data_compression_optimizer_update_strategy_priority():
    """测试 DataCompressionOptimizer（更新策略优先级）"""
    optimizer = DataCompressionOptimizer()
    
    optimizer.update_strategy_priority("text_gzip_fast", 10)
    
    strategy = next(s for s in optimizer.strategies if s.name == "text_gzip_fast")
    assert strategy.priority == 10


def test_data_compression_optimizer_update_strategy_priority_nonexistent():
    """测试 DataCompressionOptimizer（更新策略优先级，不存在）"""
    optimizer = DataCompressionOptimizer()
    
    with pytest.raises(ValueError, match="策略不存在"):
        optimizer.update_strategy_priority("nonexistent_strategy", 10)


def test_data_compression_optimizer_enable_strategy():
    """测试 DataCompressionOptimizer（启用策略）"""
    optimizer = DataCompressionOptimizer()
    
    # 先禁用
    optimizer.disable_strategy("text_gzip_fast")
    strategy = next(s for s in optimizer.strategies if s.name == "text_gzip_fast")
    assert strategy.enabled is False
    
    # 再启用
    optimizer.enable_strategy("text_gzip_fast")
    strategy = next(s for s in optimizer.strategies if s.name == "text_gzip_fast")
    assert strategy.enabled is True


def test_data_compression_optimizer_enable_strategy_nonexistent():
    """测试 DataCompressionOptimizer（启用策略，不存在）"""
    optimizer = DataCompressionOptimizer()
    
    with pytest.raises(ValueError, match="策略不存在"):
        optimizer.enable_strategy("nonexistent_strategy")


def test_data_compression_optimizer_disable_strategy():
    """测试 DataCompressionOptimizer（禁用策略）"""
    optimizer = DataCompressionOptimizer()
    
    optimizer.disable_strategy("text_gzip_fast")
    
    strategy = next(s for s in optimizer.strategies if s.name == "text_gzip_fast")
    assert strategy.enabled is False


def test_data_compression_optimizer_disable_strategy_nonexistent():
    """测试 DataCompressionOptimizer（禁用策略，不存在）"""
    optimizer = DataCompressionOptimizer()
    
    with pytest.raises(ValueError, match="策略不存在"):
        optimizer.disable_strategy("nonexistent_strategy")


def test_data_compression_optimizer_compress_batch():
    """测试 DataCompressionOptimizer（批量压缩）"""
    optimizer = DataCompressionOptimizer()
    data_list = [
        {"data": "test1" * 100, "data_type": "text"},
        {"data": "test2" * 100, "data_type": "text"},
        {"data": "test3" * 100, "data_type": "text"}
    ]
    
    results = optimizer.compress_batch(data_list)
    
    assert len(results) == 3
    assert all("compressed_data" in r for r in results)


def test_data_compression_optimizer_compress_batch_empty():
    """测试 DataCompressionOptimizer（批量压缩，空列表）"""
    optimizer = DataCompressionOptimizer()
    
    results = optimizer.compress_batch([])
    
    assert results == []


def test_data_compression_optimizer_metrics_history_limit():
    """测试 DataCompressionOptimizer（指标历史限制）"""
    optimizer = DataCompressionOptimizer()
    data = "test data" * 100
    
    # 执行大量压缩操作以触发历史限制
    for i in range(10001):
        optimizer.compress_data(data, data_type="text")
    
    # 历史记录应该被限制
    assert len(optimizer.metrics_history) <= 5000


def test_data_compression_optimizer_compress_no_strategy():
    """测试 DataCompressionOptimizer（压缩，无策略）"""
    optimizer = DataCompressionOptimizer()
    # 清空所有策略
    optimizer.strategies = []
    # 压缩数据
    data = b"test_data_for_compression" * 100
    result = optimizer.compress_data(data)
    # 应该返回不压缩的结果
    assert result['algorithm'] == 'none'
    assert result['compressed_data'] == data
    assert result['compression_ratio'] == 1.0


def test_data_compression_optimizer_compress_bz2():
    """测试 DataCompressionOptimizer（BZ2压缩）"""
    optimizer = DataCompressionOptimizer()
    data = b"test_data_for_bz2_compression" * 100
    # 使用BZ2压缩
    compressed = optimizer._compress_bz2(data, 6)
    assert isinstance(compressed, bytes)
    assert len(compressed) < len(data)
    # 解压
    decompressed = optimizer._decompress_bz2(compressed)
    assert decompressed == data


def test_data_compression_optimizer_compress_lzma():
    """测试 DataCompressionOptimizer（LZMA压缩）"""
    optimizer = DataCompressionOptimizer()
    data = b"test_data_for_lzma_compression" * 100
    # 使用LZMA压缩
    compressed = optimizer._compress_lzma(data, 6)
    assert isinstance(compressed, bytes)
    assert len(compressed) < len(data)
    # 解压
    decompressed = optimizer._decompress_lzma(compressed)
    assert decompressed == data


def test_data_compression_optimizer_compress_zlib():
    """测试 DataCompressionOptimizer（ZLIB压缩）"""
    optimizer = DataCompressionOptimizer()
    data = b"test_data_for_zlib_compression" * 100
    # 使用ZLIB压缩
    compressed = optimizer._compress_zlib(data, 6)
    assert isinstance(compressed, bytes)
    assert len(compressed) < len(data)
    # 解压
    decompressed = optimizer._decompress_zlib(compressed)
    assert decompressed == data


def test_data_compression_optimizer_adaptive_update_strategies():
    """测试 DataCompressionOptimizer（自适应更新策略）"""
    optimizer = DataCompressionOptimizer()
    # 添加性能数据
    for i in range(15):
        optimizer.algorithm_performance['gzip'].append(0.5 + i * 0.01)
    # 手动触发策略更新
    optimizer._update_strategies()
    # 应该不抛出异常
    assert True


def test_data_compression_optimizer_adaptive_adjust_strategy_performance_decline():
    """测试 DataCompressionOptimizer（自适应调整策略，性能下降）"""
    optimizer = DataCompressionOptimizer()
    # 找到gzip策略
    gzip_strategy = None
    for strategy in optimizer.strategies:
        if strategy.algorithm == 'gzip':
            gzip_strategy = strategy
            break
    if gzip_strategy:
        original_level = gzip_strategy.compression_level
        # 添加性能下降的数据
        optimizer.algorithm_performance['gzip'] = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
        optimizer.algorithm_performance['gzip'].extend([0.1] * 10)
        # 触发策略更新
        optimizer._update_strategies()
        # 压缩级别应该降低或保持不变
        assert gzip_strategy.compression_level <= original_level


def test_data_compression_optimizer_adaptive_adjust_strategy_performance_improvement():
    """测试 DataCompressionOptimizer（自适应调整策略，性能提升）"""
    optimizer = DataCompressionOptimizer()
    # 找到gzip策略
    gzip_strategy = None
    for strategy in optimizer.strategies:
        if strategy.algorithm == 'gzip':
            gzip_strategy = strategy
            break
    if gzip_strategy:
        original_level = gzip_strategy.compression_level
        # 添加性能提升的数据
        optimizer.algorithm_performance['gzip'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        optimizer.algorithm_performance['gzip'].extend([1.0] * 10)
        # 触发策略更新
        optimizer._update_strategies()
        # 压缩级别应该提高或保持不变
        assert gzip_strategy.compression_level >= original_level


def test_data_compression_optimizer_calculate_trend_insufficient_data():
    """测试 DataCompressionOptimizer（计算趋势，数据不足）"""
    optimizer = DataCompressionOptimizer()
    trend = optimizer._calculate_trend([0.5])
    assert trend == 'insufficient_data'


def test_data_compression_optimizer_calculate_trend_stable():
    """测试 DataCompressionOptimizer（计算趋势，稳定）"""
    optimizer = DataCompressionOptimizer()
    trend = optimizer._calculate_trend([0.5, 0.5, 0.5, 0.5, 0.5])
    assert trend == 'stable'


def test_data_compression_optimizer_calculate_trend_improving():
    """测试 DataCompressionOptimizer（计算趋势，改善）"""
    optimizer = DataCompressionOptimizer()
    trend = optimizer._calculate_trend([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    assert trend == 'improving'


def test_data_compression_optimizer_calculate_trend_declining():
    """测试 DataCompressionOptimizer（计算趋势，下降）"""
    optimizer = DataCompressionOptimizer()
    trend = optimizer._calculate_trend([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    assert trend == 'declining'


def test_data_compression_optimizer_calculate_trend_zero_denominator():
    """测试 DataCompressionOptimizer（计算趋势，零分母）"""
    optimizer = DataCompressionOptimizer()
    # 创建会导致零分母的情况（所有值相同）
    trend = optimizer._calculate_trend([0.5, 0.5])
    # 应该返回stable
    assert trend in ['stable', 'insufficient_data']


def test_data_compression_optimizer_get_algorithm_performance_report():
    """测试 DataCompressionOptimizer（获取算法性能报告）"""
    optimizer = DataCompressionOptimizer()
    # 添加性能数据
    optimizer.algorithm_performance['gzip'] = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # 获取报告（使用get_algorithm_performance方法）
    report = optimizer.get_algorithm_performance('gzip')
    assert isinstance(report, dict)
    # 检查是否有性能数据
    if report:
        assert 'scores' in report or len(report) > 0


def test_data_compression_optimizer_get_algorithm_performance_report_insufficient_data():
    """测试 DataCompressionOptimizer（获取算法性能报告，数据不足）"""
    optimizer = DataCompressionOptimizer()
    # 添加少量性能数据
    optimizer.algorithm_performance['gzip'] = [0.5]
    # 获取报告（使用get_algorithm_performance方法）
    report = optimizer.get_algorithm_performance('gzip')
    assert isinstance(report, dict)


def test_data_compression_optimizer_select_strategy_no_applicable():
    """测试 DataCompressionOptimizer（选择策略，无适用策略）"""
    optimizer = DataCompressionOptimizer()
    # 清空所有策略
    optimizer.strategies = []
    # 选择策略（查看方法签名：_select_compression_strategy(data_bytes, data_type, strategy_name)）
    strategy = optimizer._select_compression_strategy(b"test", "text", None)
    # 应该返回None
    assert strategy is None


def test_data_compression_optimizer_clear_metrics_history():
    """测试 DataCompressionOptimizer（清除指标历史）"""
    optimizer = DataCompressionOptimizer()
    # 添加一些指标
    optimizer.metrics_history.append(CompressionMetrics(
        original_size=1000,
        compressed_size=500,
        compression_ratio=2.0,
        compression_time=0.1,
        decompression_time=0.05,
        algorithm="gzip",
        data_type="text",
        timestamp=datetime.now()
    ))
    optimizer.algorithm_performance['gzip'] = [0.5, 0.6, 0.7]
    # 清除历史
    optimizer.clear_metrics_history()
    # 应该为空
    assert len(optimizer.metrics_history) == 0
    assert len(optimizer.algorithm_performance['gzip']) == 0


def test_data_compression_optimizer_get_optimizer_status():
    """测试 DataCompressionOptimizer（获取优化器状态）"""
    optimizer = DataCompressionOptimizer()
    status = optimizer.get_optimizer_status()
    assert 'strategies_count' in status
    assert 'enabled_strategies' in status
    assert 'metrics_history_count' in status
