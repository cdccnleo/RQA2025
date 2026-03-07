#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
回测层性能测试脚本

测试实时回测引擎的性能优化效果
"""

import time
import threading
import random
from datetime import datetime, timedelta
from typing import Dict, Any
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_market_data(symbol: str, timestamp: datetime) -> Dict[str, Any]:
    """生成模拟市场数据"""
    base_price = 100.0 + random.uniform(-10, 10)
    return {
        'timestamp': timestamp.isoformat(),
        'symbol': symbol,
        'type': 'market',
        'source': 'test',
        'price': base_price,
        'volume': random.randint(1000, 10000),
        'bid': base_price - 0.01,
        'ask': base_price + 0.01,
        'high': base_price + random.uniform(0, 2),
        'low': base_price - random.uniform(0, 2)
    }

def test_cache_manager_performance():
    """测试缓存管理器性能"""
    logger.info("开始测试缓存管理器性能...")
    
    from src.backtest.real_time_engine import CacheManager
    
    cache_manager = CacheManager(cache_size=1000)
    
    # 测试数据写入性能
    start_time = time.time()
    for i in range(10000):
        data = {'test': f'data_{i}', 'value': random.random()}
        cache_manager.cache_data(f'key_{i}', data, ttl=300)
    
    write_time = time.time() - start_time
    logger.info(f"写入10000条数据耗时: {write_time:.3f}秒")
    
    # 测试数据读取性能
    start_time = time.time()
    hit_count = 0
    for i in range(10000):
        data = cache_manager.get_cached_data(f'key_{i}')
        if data:
            hit_count += 1
    
    read_time = time.time() - start_time
    hit_rate = hit_count / 10000 * 100
    logger.info(f"读取10000条数据耗时: {read_time:.3f}秒，命中率: {hit_rate:.1f}%")
    
    # 获取缓存统计
    stats = cache_manager.get_cache_stats()
    logger.info(f"缓存统计: {stats}")
    
    return {
        'write_time': write_time,
        'read_time': read_time,
        'hit_rate': hit_rate,
        'stats': stats
    }

def test_data_processor_performance():
    """测试数据处理器性能"""
    logger.info("开始测试数据处理器性能...")
    
    from src.backtest.real_time_engine import RealTimeDataProcessor
    
    processor = RealTimeDataProcessor()
    
    # 添加模拟处理器
    def mock_processor(data):
        time.sleep(0.001)  # 模拟处理时间
        
    processor.add_processor(mock_processor)
    processor.start()
    
    # 生成测试数据
    symbols = ['000001.SZ', '000002.SZ', '000858.SZ', '002415.SZ']
    start_time = datetime.now()
    
    # 发送数据
    data_count = 1000
    for i in range(data_count):
        symbol = random.choice(symbols)
        timestamp = start_time + timedelta(seconds=i)
        data = generate_market_data(symbol, timestamp)
        processor.data_queue.put(data)
    
    # 等待处理完成
    time.sleep(5)
    processor.stop()
    
    # 获取性能统计
    stats = processor.get_performance_stats()
    logger.info(f"数据处理器性能统计: {stats}")
    
    return stats

def test_real_time_engine_performance():
    """测试实时回测引擎性能"""
    logger.info("开始测试实时回测引擎性能...")
    
    from src.backtest.real_time_engine import RealTimeBacktestEngine
    
    engine = RealTimeBacktestEngine()
    
    # 添加模拟策略
    def mock_strategy(data, state):
        return {'action': 'hold'}
    
    engine.add_strategy('test_strategy', mock_strategy)
    engine.start(initial_capital=1000000.0)
    
    # 生成测试数据
    symbols = ['000001.SZ', '000002.SZ', '000858.SZ', '002415.SZ']
    start_time = datetime.now()
    
    # 发送数据
    data_count = 500
    for i in range(data_count):
        symbol = random.choice(symbols)
        timestamp = start_time + timedelta(seconds=i)
        data = generate_market_data(symbol, timestamp)
        engine.process_data(data)
    
    # 等待处理完成
    time.sleep(3)
    
    # 获取状态和指标
    state = engine.get_current_state()
    metrics = engine.get_metrics()
    
    engine.stop()
    
    logger.info(f"引擎状态: {state}")
    logger.info(f"引擎指标: {metrics}")
    
    return {
        'state': state,
        'metrics': metrics
    }

def run_comprehensive_performance_test():
    """运行综合性能测试"""
    logger.info("开始综合性能测试...")
    
    results = {}
    
    # 测试缓存管理器
    try:
        results['cache_manager'] = test_cache_manager_performance()
    except Exception as e:
        logger.error(f"缓存管理器测试失败: {e}")
        results['cache_manager'] = {'error': str(e)}
    
    # 测试数据处理器
    try:
        results['data_processor'] = test_data_processor_performance()
    except Exception as e:
        logger.error(f"数据处理器测试失败: {e}")
        results['data_processor'] = {'error': str(e)}
    
    # 测试实时回测引擎
    try:
        results['real_time_engine'] = test_real_time_engine_performance()
    except Exception as e:
        logger.error(f"实时回测引擎测试失败: {e}")
        results['real_time_engine'] = {'error': str(e)}
    
    # 输出测试结果
    logger.info("=" * 50)
    logger.info("性能测试结果汇总:")
    logger.info("=" * 50)
    
    for test_name, result in results.items():
        logger.info(f"\n{test_name}:")
        if 'error' in result:
            logger.error(f"  错误: {result['error']}")
        else:
            for key, value in result.items():
                logger.info(f"  {key}: {value}")
    
    return results

if __name__ == "__main__":
    run_comprehensive_performance_test() 