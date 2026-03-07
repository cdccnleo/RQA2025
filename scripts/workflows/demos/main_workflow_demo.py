#!/usr/bin/env python3
"""
RQA2025 主流程演示脚本
展示完整的量化交易流程：数据加载 → 特征计算 → 模型预测 → 策略信号 → 风控检查 → 交易执行
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_main_workflow():
    """主流程演示"""
    print("🚀 RQA2025 主流程演示开始")
    print("=" * 60)

    try:
        # 步骤1: 数据加载
        print("📊 步骤1: 数据加载")
        market_data = load_market_data()
        print(f"✅ 成功加载 {len(market_data)} 条市场数据")

        # 步骤2: 特征计算
        print("\n🔧 步骤2: 特征计算")
        features = calculate_features(market_data)
        print(f"✅ 成功计算 {len(features.columns)} 个特征")

        # 步骤3: 模型预测
        print("\n🤖 步骤3: 模型预测")
        predictions = run_model_predictions(features)
        print(f"✅ 成功生成 {len(predictions)} 个预测结果")

        # 步骤4: 策略信号
        print("\n📈 步骤4: 策略信号生成")
        signals = generate_strategy_signals(market_data, predictions)
        print(f"✅ 成功生成 {len(signals)} 个交易信号")

        # 步骤5: 风控检查
        print("\n🛡️ 步骤5: 风控检查")
        approved_signals = risk_control_check(signals)
        print(f"✅ 风控通过 {len(approved_signals)} 个信号")

        # 步骤6: 交易执行
        print("\n💼 步骤6: 交易执行")
        executions = execute_trades(approved_signals)
        print(f"✅ 成功执行 {len(executions)} 笔交易")

        # 步骤7: 结果汇总
        print("\n📋 步骤7: 结果汇总")
        summary = generate_summary(market_data, signals, executions)
        print_summary(summary)

        print("\n🎉 主流程演示完成！")
        return True

    except Exception as e:
        logger.error(f"主流程演示失败: {e}")
        return False


def load_market_data() -> pd.DataFrame:
    """加载市场数据"""
    print("  - 加载股票行情数据...")

    # 模拟市场数据
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
    symbols = ['000001.SZ', '000858.SZ', '001227.SZ']

    data = []
    for symbol in symbols:
        for date in dates:
            data.append({
                'symbol': symbol,
                'date': date,
                'open': np.random.uniform(10, 50),
                'high': np.random.uniform(10, 50),
                'low': np.random.uniform(10, 50),
                'close': np.random.uniform(10, 50),
                'volume': np.random.uniform(1000000, 10000000),
                'amount': np.random.uniform(50000000, 500000000)
            })

    return pd.DataFrame(data)


def calculate_features(data: pd.DataFrame) -> pd.DataFrame:
    """计算技术特征"""
    print("  - 计算技术指标...")

    features = data.copy()

    # 计算移动平均
    features['ma5'] = features.groupby('symbol')['close'].rolling(
        5).mean().reset_index(0, drop=True)
    features['ma10'] = features.groupby('symbol')['close'].rolling(
        10).mean().reset_index(0, drop=True)
    features['ma20'] = features.groupby('symbol')['close'].rolling(
        20).mean().reset_index(0, drop=True)

    # 计算RSI
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    features['rsi'] = features.groupby('symbol')['close'].apply(
        calculate_rsi).reset_index(0, drop=True)

    # 计算MACD
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram

    macd_data = features.groupby('symbol')['close'].apply(lambda x: calculate_macd(x)[0])
    features['macd'] = macd_data.reset_index(0, drop=True)

    # 计算布林带
    features['bb_upper'] = features.groupby('symbol')['close'].rolling(20).mean() + \
        features.groupby('symbol')['close'].rolling(20).std() * 2
    features['bb_lower'] = features.groupby('symbol')['close'].rolling(20).mean() - \
        features.groupby('symbol')['close'].rolling(20).std() * 2

    return features.dropna()


def run_model_predictions(features: pd.DataFrame) -> pd.DataFrame:
    """运行模型预测"""
    print("  - 运行LSTM模型预测...")
    print("  - 运行神经网络模型预测...")
    print("  - 运行随机森林模型预测...")

    # 模拟模型预测结果
    predictions = features[['symbol', 'date', 'close']].copy()
    predictions['lstm_pred'] = predictions['close'] * \
        (1 + np.random.normal(0, 0.02, len(predictions)))
    predictions['nn_pred'] = predictions['close'] * \
        (1 + np.random.normal(0, 0.025, len(predictions)))
    predictions['rf_pred'] = predictions['close'] * \
        (1 + np.random.normal(0, 0.03, len(predictions)))

    # 集成预测
    predictions['ensemble_pred'] = (predictions['lstm_pred'] +
                                    predictions['nn_pred'] +
                                    predictions['rf_pred']) / 3

    return predictions


def generate_strategy_signals(market_data: pd.DataFrame, predictions: pd.DataFrame) -> List[Dict]:
    """生成策略信号"""
    print("  - 涨停板策略信号...")
    print("  - 龙虎榜策略信号...")
    print("  - 融资融券策略信号...")
    print("  - ST股票策略信号...")

    signals = []

    # 模拟策略信号
    for _, row in predictions.iterrows():
        if np.random.random() > 0.7:  # 30%概率生成信号
            signal = {
                'symbol': row['symbol'],
                'date': row['date'],
                'signal_type': np.random.choice(['buy', 'sell']),
                'price': row['close'],
                'target_price': row['ensemble_pred'],
                'confidence': np.random.uniform(0.6, 0.95),
                'strategy': np.random.choice(['limit_up', 'dragon_tiger', 'margin', 'st']),
                'volume': np.random.uniform(1000, 10000),
                'reason': '技术指标 + 模型预测'
            }
            signals.append(signal)

    return signals


def risk_control_check(signals: List[Dict]) -> List[Dict]:
    """风控检查"""
    print("  - 仓位风险检查...")
    print("  - 资金风险检查...")
    print("  - 市场风险检查...")

    approved_signals = []

    for signal in signals:
        # 模拟风控检查
        risk_score = np.random.random()

        if risk_score > 0.3:  # 70%通过率
            signal['risk_approved'] = True
            signal['risk_score'] = risk_score
            approved_signals.append(signal)
        else:
            signal['risk_approved'] = False
            signal['risk_score'] = risk_score

    return approved_signals


def execute_trades(approved_signals: List[Dict]) -> List[Dict]:
    """执行交易"""
    print("  - 订单生成...")
    print("  - 订单执行...")
    print("  - 成交确认...")

    executions = []

    for signal in approved_signals:
        execution = {
            'order_id': f"ORD_{len(executions):06d}",
            'symbol': signal['symbol'],
            'date': signal['date'],
            'action': signal['signal_type'],
            'price': signal['price'],
            'volume': signal['volume'],
            'amount': signal['price'] * signal['volume'],
            'status': 'filled',
            'execution_time': datetime.now(),
            'strategy': signal['strategy'],
            'confidence': signal['confidence']
        }
        executions.append(execution)

    return executions


def generate_summary(market_data: pd.DataFrame, signals: List[Dict], executions: List[Dict]) -> Dict:
    """生成结果汇总"""
    summary = {
        'total_signals': len(signals),
        'approved_signals': len([s for s in signals if s.get('risk_approved', False)]),
        'total_executions': len(executions),
        'total_volume': sum(e['volume'] for e in executions),
        'total_amount': sum(e['amount'] for e in executions),
        'strategy_distribution': {},
        'signal_types': {'buy': 0, 'sell': 0}
    }

    # 统计策略分布
    for signal in signals:
        strategy = signal.get('strategy', 'unknown')
        summary['strategy_distribution'][strategy] = summary['strategy_distribution'].get(
            strategy, 0) + 1

    # 统计信号类型
    for signal in signals:
        signal_type = signal.get('signal_type', 'unknown')
        summary['signal_types'][signal_type] = summary['signal_types'].get(signal_type, 0) + 1

    return summary


def print_summary(summary: Dict):
    """打印结果汇总"""
    print(f"  📊 总信号数: {summary['total_signals']}")
    print(f"  ✅ 风控通过: {summary['approved_signals']}")
    print(f"  💼 执行交易: {summary['total_executions']}")
    print(f"  📈 总成交量: {summary['total_volume']:,.0f}")
    print(f"  💰 总成交额: {summary['total_amount']:,.2f}")

    print("\n  📋 策略分布:")
    for strategy, count in summary['strategy_distribution'].items():
        print(f"    - {strategy}: {count}")

    print("\n  📊 信号类型:")
    for signal_type, count in summary['signal_types'].items():
        print(f"    - {signal_type}: {count}")


if __name__ == "__main__":
    success = demo_main_workflow()
    if success:
        print("\n🎉 主流程演示成功完成！")
    else:
        print("\n❌ 主流程演示失败！")
