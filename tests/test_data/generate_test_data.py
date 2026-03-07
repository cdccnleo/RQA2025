#!/usr/bin/env python3
"""
测试数据生成脚本

生成RQA2025量化交易系统测试所需的各种测试数据
包括：
1. 股票历史数据（不同时间范围和市场条件）
2. 策略测试数据
3. 性能测试数据
4. 极端市场条件测试数据
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json


def generate_stock_data(symbol, start_date, days=252, volatility=0.02, trend=0.001, market_condition='normal'):
    """
    生成股票历史数据
    
    参数：
    - symbol: 股票代码
    - start_date: 起始日期
    - days: 数据天数
    - volatility: 日波动率
    - trend: 日趋势
    - market_condition: 市场条件 ('normal', 'volatile', 'bear', 'bull')
    
    返回：
    - pandas DataFrame 包含股票历史数据
    """
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    # 根据市场条件调整参数
    if market_condition == 'volatile':
        volatility = 0.04
        trend = 0.0
    elif market_condition == 'bear':
        volatility = 0.03
        trend = -0.002
    elif market_condition == 'bull':
        volatility = 0.02
        trend = 0.002
    
    # 生成价格数据（随机游走）
    np.random.seed(42 + hash(symbol) % 1000)
    returns = np.random.normal(trend, volatility, days)
    prices = 100 * np.exp(np.cumsum(returns))  # 起始价格100
    
    # 生成OHLC数据
    high_multipliers = 1 + np.random.uniform(0, 0.02, days)
    low_multipliers = 1 - np.random.uniform(0, 0.02, days)
    volume_base = 1000000
    
    data = []
    for i, date in enumerate(dates):
        open_price = prices[i] * (1 + np.random.normal(0, 0.005))
        close_price = prices[i]
        high_price = close_price * high_multipliers[i]
        low_price = close_price * low_multipliers[i]
        volume = int(volume_base * (1 + np.random.normal(0, 0.3)))
        
        data.append({
            'symbol': symbol,
            'date': date.strftime('%Y-%m-%d'),
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': volume,
            'returns': round(returns[i], 4)
        })
    
    return pd.DataFrame(data)


def generate_strategy_test_data():
    """
    生成策略测试数据
    """
    # 生成不同市场条件的测试数据
    market_conditions = ['normal', 'volatile', 'bear', 'bull']
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    for condition in market_conditions:
        condition_dir = os.path.join('tests', 'test_data', 'market_data', condition)
        os.makedirs(condition_dir, exist_ok=True)
        
        for symbol in symbols:
            start_date = datetime(2023, 1, 1)
            df = generate_stock_data(symbol, start_date, days=252, market_condition=condition)
            output_file = os.path.join(condition_dir, f'{symbol}_{condition}.csv')
            df.to_csv(output_file, index=False)
            print(f'Generated {output_file}')


def generate_performance_test_data():
    """
    生成性能测试数据
    """
    # 生成大规模测试数据
    performance_dir = os.path.join('tests', 'test_data', 'performance_data')
    os.makedirs(performance_dir, exist_ok=True)
    
    # 生成1000万条数据的模拟
    symbols = [f'STOCK{i:03d}' for i in range(100)]
    start_date = datetime(2020, 1, 1)
    
    for i, symbol in enumerate(symbols):
        df = generate_stock_data(symbol, start_date, days=1000)
        output_file = os.path.join(performance_dir, f'{symbol}_performance.csv')
        df.to_csv(output_file, index=False)
        if (i + 1) % 10 == 0:
            print(f'Generated {i + 1}/100 performance test files')


def generate_edge_case_data():
    """
    生成极端情况测试数据
    """
    edge_case_dir = os.path.join('tests', 'test_data', 'edge_cases')
    os.makedirs(edge_case_dir, exist_ok=True)
    
    # 生成熔断场景数据
    circuit_breaker_data = []
    date = datetime(2023, 3, 1)
    
    for i in range(10):
        if i < 3:
            # 正常波动
            close = 100 - i * 2
        elif i < 6:
            # 快速下跌触发熔断
            close = 94 - (i - 2) * 5
        else:
            # 熔断后恢复
            close = 79 + (i - 5) * 3
        
        circuit_breaker_data.append({
            'symbol': 'SPX',
            'date': (date + timedelta(days=i)).strftime('%Y-%m-%d'),
            'open': close * 1.01,
            'high': close * 1.02,
            'low': close * 0.98,
            'close': close,
            'volume': 2000000 + i * 500000,
            'returns': round((close - 100) / 100, 4) if i == 0 else round((close - circuit_breaker_data[i-1]['close']) / circuit_breaker_data[i-1]['close'], 4)
        })
    
    df = pd.DataFrame(circuit_breaker_data)
    df.to_csv(os.path.join(edge_case_dir, 'market_crash.csv'), index=False)
    print('Generated edge case data: market_crash.csv')


def generate_strategy_config_data():
    """
    生成策略配置测试数据
    """
    config_dir = os.path.join('tests', 'test_data', 'strategy_data')
    os.makedirs(config_dir, exist_ok=True)
    
    # 生成不同策略类型的配置
    strategies = {
        'momentum_strategy': {
            'name': '动量策略',
            'parameters': {
                'lookback_period': 20,
                'threshold': 0.05,
                'position_size': 0.1
            }
        },
        'mean_reversion_strategy': {
            'name': '均值回归策略',
            'parameters': {
                'lookback_period': 50,
                'z_score_threshold': 2.0,
                'position_size': 0.08
            }
        },
        'trend_following_strategy': {
            'name': '趋势跟随策略',
            'parameters': {
                'short_period': 20,
                'long_period': 50,
                'position_size': 0.12
            }
        },
        'ml_strategy': {
            'name': '机器学习策略',
            'parameters': {
                'model_id': 'ml_model_001',
                'feature_set': 'technical_indicators',
                'position_size': 0.09
            }
        }
    }
    
    with open(os.path.join(config_dir, 'strategy_configs.json'), 'w', encoding='utf-8') as f:
        json.dump(strategies, f, indent=2, ensure_ascii=False)
    
    print('Generated strategy config data: strategy_configs.json')


def main():
    """
    主函数：生成所有测试数据
    """
    print('开始生成测试数据...')
    
    # 1. 生成策略测试数据
    print('\n1. 生成策略测试数据...')
    generate_strategy_test_data()
    
    # 2. 生成性能测试数据
    print('\n2. 生成性能测试数据...')
    generate_performance_test_data()
    
    # 3. 生成极端情况测试数据
    print('\n3. 生成极端情况测试数据...')
    generate_edge_case_data()
    
    # 4. 生成策略配置数据
    print('\n4. 生成策略配置数据...')
    generate_strategy_config_data()
    
    print('\n测试数据生成完成！')


if __name__ == '__main__':
    main()
