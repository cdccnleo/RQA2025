#!/usr/bin/env python3
"""
RQA2025简化端到端业务功能示例
展示如何使用已验证的业务组件构建完整的数据流
"""

import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demonstrate_components():
    """演示核心组件功能"""
    print("🚀 RQA2025核心组件功能演示")
    print("=" * 60)

    success_count = 0
    total_count = 0

    # 1. 测试数据加载器
    print("\n📊 1. 测试数据加载器...")
    try:
        from src.data.loaders import StockDataLoader
        loader = StockDataLoader({'data_source': 'mock'})
        data = loader.load_data('000001', '2024-01-01', '2024-01-31')
        if not data.empty:
            print(f"   ✅ 数据加载成功: {len(data)} 条记录")
            success_count += 1
        else:
            print("   ❌ 数据加载失败")
        total_count += 1
    except Exception as e:
        print(f"   ❌ 数据加载器错误: {e}")

    # 2. 测试特征处理器
    print("\n🔧 2. 测试特征处理器...")
    try:
        from src.features.processors.technical_indicator_processor import TechnicalIndicatorProcessor
        processor = TechnicalIndicatorProcessor()

        if not data.empty:
            # 计算简单移动平均
            config = {'sma': {'period': 5}}
            result = processor.calculate_indicators(data, config)
            if not result.empty:
                print(f"   ✅ 特征计算成功: 新增 {len(result.columns) - len(data.columns)} 个特征")
                success_count += 1
            else:
                print("   ❌ 特征计算失败")
        else:
            print("   ❌ 无数据进行特征计算")
        total_count += 1
    except Exception as e:
        print(f"   ❌ 特征处理器错误: {e}")

    # 3. 测试ML核心
    print("\n🤖 3. 测试ML核心...")
    try:
        from src.ml.core import MLCore
        ml_core = MLCore()

        if not data.empty and len(data) > 20:
            # 准备训练数据
            train_data = data.copy()
            train_data['target'] = (train_data['close'].shift(-1) > train_data['close']).astype(int)
            train_data = train_data.dropna()

            if len(train_data) > 10:
                feature_cols = ['open', 'high', 'low', 'close', 'volume']
                X = train_data[feature_cols]
                y = train_data['target']

                # 训练简单模型
                model_id = ml_core.train_model(X, y, model_type='rf',
                                               model_params={'n_estimators': 10})
                if model_id:
                    print(f"   ✅ 模型训练成功: {model_id}")
                    success_count += 1
                else:
                    print("   ❌ 模型训练失败")
            else:
                print("   ❌ 训练数据不足")
        else:
            print("   ❌ 无数据进行训练")
        total_count += 1
    except Exception as e:
        print(f"   ❌ ML核心错误: {e}")

    # 4. 测试交易引擎
    print("\n📈 4. 测试交易引擎...")
    try:
        from src.trading.trading_engine import TradingEngine
        engine = TradingEngine({'max_position': 100})
        print(f"   ✅ 交易引擎初始化成功: {engine}")
        success_count += 1
        total_count += 1
    except Exception as e:
        print(f"   ❌ 交易引擎错误: {e}")

    # 5. 测试适配器
    print("\n🔌 5. 测试适配器...")
    try:
        from src.adapters import MiniQMTAdapter
        adapter = MiniQMTAdapter({'account_id': 'demo'})
        if adapter.connect():
            account_info = adapter.get_data(data_type='account')
            if account_info:
                print("   ✅ 适配器连接成功")
                success_count += 1
            else:
                print("   ❌ 适配器连接失败")
            adapter.disconnect()
        else:
            print("   ❌ 适配器连接失败")
        total_count += 1
    except Exception as e:
        print(f"   ❌ 适配器错误: {e}")

    # 总结
    print("\n" + "=" * 60)
    print("📊 组件测试总结")
    print("=" * 60)
    print(f"✅ 成功组件: {success_count}/{total_count}")
    print(".1f")

    if success_count >= 3:
        print("\n🎉 核心组件功能正常！")
        print("   您现在可以开始构建量化交易系统了。")
    else:
        print("\n⚠️ 部分组件需要进一步调试")

    return success_count, total_count


def show_usage_examples():
    """展示使用示例"""
    print("\n" + "=" * 60)
    print("💡 使用示例")
    print("=" * 60)

    examples = [
        {
            'title': '数据加载和分析',
            'code': '''
from src.data.loaders import StockDataLoader

# 创建数据加载器
loader = StockDataLoader({'cache_enabled': True})

# 加载股票数据
data = loader.load_data('000001', '2024-01-01', '2024-12-31')

# 基本统计
print(f"数据条数: {len(data)}")
print(f"价格范围: {data['close'].min():.2f} - {data['close'].max():.2f}")
'''
        },
        {
            'title': '特征工程',
            'code': '''
from src.features.processors.technical_indicator_processor import TechnicalIndicatorProcessor

# 创建特征处理器
processor = TechnicalIndicatorProcessor()

# 计算技术指标
config = {
    'sma': {'period': 20},
    'rsi': {'period': 14},
    'macd': {'fast_period': 12, 'slow_period': 26}
}

# 处理数据
data_with_features = processor.calculate_indicators(data, config)
print(f"新增特征: {len(data_with_features.columns) - len(data.columns)}")
'''
        },
        {
            'title': '模型训练和预测',
            'code': '''
from src.ml.core import MLCore

# 创建ML核心
ml_core = MLCore()

# 准备训练数据
X = data_with_features[['close', 'sma_20', 'rsi_14']]
y = (data_with_features['close'].shift(-1) > data_with_features['close']).astype(int)

# 训练模型
model_id = ml_core.train_model(X, y, model_type='rf')

# 进行预测
predictions = ml_core.predict(model_id, X)
print(f"预测准确率: {(predictions == y).mean():.3f}")
'''
        },
        {
            'title': '交易集成',
            'code': '''
from src.adapters import MiniQMTAdapter

# 创建交易适配器
adapter = MiniQMTAdapter({'account_id': 'your_account'})

# 连接到交易平台
if adapter.connect():
    # 获取账户信息
    account = adapter.get_data(data_type='account')

    # 基于模型预测进行交易
    if predictions[-1] > 0.7:  # 强看涨信号
        order = adapter.place_order('000001', 'buy', 100, price=10.5)
        print(f"下单结果: {order}")

    adapter.disconnect()
'''
        }
    ]

    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['title']}")
        print("-" * 40)
        # 只显示代码的前几行
        code_lines = example['code'].strip().split('\n')
        for line in code_lines[:5]:
            if line.strip():
                print(f"   {line}")
        if len(code_lines) > 5:
            print("   ...")


def main():
    """主函数"""
    print("RQA2025业务功能展示")
    print("展示已验证的业务组件如何协同工作")

    # 演示组件功能
    success_count, total_count = demonstrate_components()

    # 展示使用示例
    show_usage_examples()

    print("\n" + "=" * 60)
    print("🎯 开发建议")
    print("=" * 60)
    print("1. 从数据加载开始，建立您的数据获取管道")
    print("2. 逐步添加特征工程，提升模型性能")
    print("3. 实现基础交易策略，验证整体流程")
    print("4. 集成适配器，实现自动化交易")
    print("5. 添加风险管理，确保交易安全")

    print("\n🚀 您已经准备好开始量化交易系统开发了！")


if __name__ == "__main__":
    main()
