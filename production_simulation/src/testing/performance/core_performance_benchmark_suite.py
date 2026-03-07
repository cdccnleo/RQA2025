#!/usr/bin/env python3
"""
RQA2025 核心组件性能基准测试套件
涵盖核心服务层、数据管理层、交易系统、策略系统等关键组件
"""

from ...enhanced_performance_benchmark import PerformanceBenchmarkFramework, TestCategory
import sys
import time
import random
from pathlib import Path
from typing import List, Optional
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class CoreComponentBenchmarkSuite:
    """核心组件性能基准测试套件"""

    def __init__(self):
        self.framework = PerformanceBenchmarkFramework()
        self._register_all_test_suites()

    def _register_all_test_suites(self):
        """注册所有测试套件"""
        # 核心服务层测试
        self.framework.register_test_suite(
            "event_bus_performance",
            self._test_event_bus_performance,
            TestCategory.CORE_SERVICE,
            {"target_ops_per_sec": 10000}
        )

        self.framework.register_test_suite(
            "dependency_injection_performance",
            self._test_dependency_injection_performance,
            TestCategory.CORE_SERVICE,
            {"target_ops_per_sec": 100000}
        )

        self.framework.register_test_suite(
            "business_process_orchestration",
            self._test_business_process_orchestration,
            TestCategory.CORE_SERVICE,
            {"target_ops_per_sec": 1000}
        )

        # 数据管理层测试
        self.framework.register_test_suite(
            "data_ingestion_performance",
            self._test_data_ingestion_performance,
            TestCategory.DATA_MANAGEMENT,
            {"target_ops_per_sec": 50000}
        )

        self.framework.register_test_suite(
            "cache_performance",
            self._test_cache_performance,
            TestCategory.DATA_MANAGEMENT,
            {"target_ops_per_sec": 100000}
        )

        self.framework.register_test_suite(
            "database_operations",
            self._test_database_operations,
            TestCategory.DATA_MANAGEMENT,
            {"target_ops_per_sec": 10000}
        )

        # 交易系统测试
        self.framework.register_test_suite(
            "order_processing",
            self._test_order_processing,
            TestCategory.TRADING_SYSTEM,
            {"target_ops_per_sec": 50000}
        )

        self.framework.register_test_suite(
            "risk_management",
            self._test_risk_management,
            TestCategory.TRADING_SYSTEM,
            {"target_ops_per_sec": 100000}
        )

        self.framework.register_test_suite(
            "market_data_processing",
            self._test_market_data_processing,
            TestCategory.TRADING_SYSTEM,
            {"target_ops_per_sec": 1000000}
        )

        # 策略系统测试
        self.framework.register_test_suite(
            "strategy_execution",
            self._test_strategy_execution,
            TestCategory.STRATEGY_SYSTEM,
            {"target_ops_per_sec": 10000}
        )

        self.framework.register_test_suite(
            "backtesting_performance",
            self._test_backtesting_performance,
            TestCategory.STRATEGY_SYSTEM,
            {"target_ops_per_sec": 1000}
        )

        self.framework.register_test_suite(
            "ml_model_inference",
            self._test_ml_model_inference,
            TestCategory.ML_SYSTEM,
            {"target_ops_per_sec": 5000}
        )

    # ============ 核心服务层测试 ============

    def _test_event_bus_performance(self):
        """测试事件总线性能"""
        # 模拟事件发布和订阅
        event_data = {
            'type': 'market_data_update',
            'symbol': 'AAPL',
            'price': 150.0 + random.random() * 10,
            'volume': random.randint(1000, 10000),
            'timestamp': time.time()
        }

        # 模拟事件处理
        processed_events = []
        for _ in range(10):  # 模拟10个订阅者
            processed_event = {
                'subscriber_id': f"subscriber_{random.randint(1, 100)}",
                'processed_at': time.time(),
                'event_type': event_data['type']
            }
            processed_events.append(processed_event)

        return len(processed_events)

    def _test_dependency_injection_performance(self):
        """测试依赖注入性能"""
        # 模拟依赖解析和对象创建
        dependencies = []

        # 创建多层依赖关系
        for i in range(5):
            dependency = {
                'id': f"dep_{i}",
                'type': f"Service{i}",
                'dependencies': [f"dep_{j}" for j in range(i)],
                'created_at': time.time()
            }
            dependencies.append(dependency)

        # 模拟依赖解析
        resolved_count = 0
        for dep in dependencies:
            # 简单的解析逻辑
            if all(d in [d['id'] for d in dependencies[:resolved_count]] for d in dep['dependencies']):
                resolved_count += 1

        return resolved_count

    def _test_business_process_orchestration(self):
        """测试业务流程编排性能"""
        # 模拟复杂的业务流程
        process_steps = [
            {'name': 'validate_data', 'duration': 0.001},
            {'name': 'transform_data', 'duration': 0.002},
            {'name': 'enrich_data', 'duration': 0.001},
            {'name': 'apply_business_rules', 'duration': 0.003},
            {'name': 'persist_data', 'duration': 0.002}
        ]

        # 执行流程
        total_duration = 0
        for step in process_steps:
            # 模拟步骤执行
            step_start = time.perf_counter()
            time.sleep(step['duration'] / 1000)  # 微秒级延迟
            step_end = time.perf_counter()
            total_duration += (step_end - step_start)

        return len(process_steps)

    # ============ 数据管理层测试 ============

    def _test_data_ingestion_performance(self):
        """测试数据摄取性能"""
        # 生成模拟市场数据
        records = []
        for i in range(100):  # 批量处理100条记录
            record = {
                'symbol': f'STOCK_{i % 10}',
                'price': 100.0 + random.random() * 50,
                'volume': random.randint(1000, 100000),
                'timestamp': time.time(),
                'bid': 99.5 + random.random() * 50,
                'ask': 100.5 + random.random() * 50
            }
            records.append(record)

        # 模拟数据验证和转换
        valid_records = []
        for record in records:
            if record['price'] > 0 and record['volume'] > 0:
                # 简单的数据清洗
                record['normalized_price'] = record['price'] / 100.0
                record['price_change'] = random.uniform(-0.05, 0.05)
                valid_records.append(record)

        return len(valid_records)

    def _test_cache_performance(self):
        """测试缓存性能"""
        # 模拟缓存操作
        cache_data = {}
        operations = 0

        # 写入操作
        for i in range(50):
            key = f"cache_key_{i}"
            value = {
                'data': f"cached_value_{i}",
                'timestamp': time.time(),
                'ttl': 300  # 5分钟TTL
            }
            cache_data[key] = value
            operations += 1

        # 读取操作
        for i in range(50):
            key = f"cache_key_{random.randint(0, 49)}"
            if key in cache_data:
                cached_value = cache_data[key]
                # 检查TTL
                if time.time() - cached_value['timestamp'] < cached_value['ttl']:
                    operations += 1

        return operations

    def _test_database_operations(self):
        """测试数据库操作性能"""
        # 模拟数据库CRUD操作
        operations = 0

        # 模拟数据表结构
        portfolio_data = []

        # INSERT操作
        for i in range(20):
            portfolio_record = {
                'id': i,
                'symbol': f'STOCK_{i}',
                'quantity': random.randint(100, 1000),
                'avg_price': 100.0 + random.random() * 50,
                'created_at': time.time()
            }
            portfolio_data.append(portfolio_record)
            operations += 1

        # SELECT操作
        for i in range(30):
            symbol = f'STOCK_{random.randint(0, 19)}'
            filtered_records = [r for r in portfolio_data if r['symbol'] == symbol]
            operations += 1

        # UPDATE操作
        for i in range(10):
            record_id = random.randint(0, 19)
            for record in portfolio_data:
                if record['id'] == record_id:
                    record['quantity'] += random.randint(-50, 50)
                    operations += 1
                    break

        return operations

    # ============ 交易系统测试 ============

    def _test_order_processing(self):
        """测试订单处理性能"""
        # 模拟订单生命周期
        orders = []
        processed = 0

        # 创建订单
        for i in range(50):
            order = {
                'id': f"order_{i}",
                'symbol': f'STOCK_{i % 5}',
                'side': 'buy' if random.random() > 0.5 else 'sell',
                'quantity': random.randint(100, 1000),
                'price': 100.0 + random.random() * 20,
                'order_type': 'limit',
                'status': 'pending',
                'timestamp': time.time()
            }
            orders.append(order)

        # 处理订单
        for order in orders:
            # 模拟订单验证
            if order['quantity'] > 0 and order['price'] > 0:
                order['status'] = 'validated'

                # 模拟市场匹配
                if random.random() > 0.1:  # 90%成功率
                    order['status'] = 'filled'
                    order['filled_quantity'] = order['quantity']
                    order['filled_price'] = order['price'] + random.uniform(-0.1, 0.1)
                else:
                    order['status'] = 'rejected'
                    order['reject_reason'] = 'insufficient_liquidity'

                processed += 1

        return processed

    def _test_risk_management(self):
        """测试风险管理性能"""
        # 模拟风险检查
        risk_checks = 0

        # 模拟投资组合
        portfolio = {
            'cash': 100000.0,
            'positions': {
                'AAPL': {'quantity': 100, 'avg_price': 150.0},
                'GOOGL': {'quantity': 50, 'avg_price': 2800.0},
                'MSFT': {'quantity': 200, 'avg_price': 300.0}
            }
        }

        # 风险检查场景
        for i in range(100):
            # 模拟新订单
            order = {
                'symbol': random.choice(['AAPL', 'GOOGL', 'MSFT', 'TSLA']),
                'side': random.choice(['buy', 'sell']),
                'quantity': random.randint(10, 100),
                'price': random.uniform(100, 3000)
            }

            # 执行风险检查
            checks_passed = 0

            # 1. 资金充足性检查
            if order['side'] == 'buy':
                required_cash = order['quantity'] * order['price']
                if portfolio['cash'] >= required_cash:
                    checks_passed += 1
            else:
                position = portfolio['positions'].get(order['symbol'], {'quantity': 0})
                if position['quantity'] >= order['quantity']:
                    checks_passed += 1

            # 2. 持仓集中度检查
            total_value = portfolio['cash'] + sum(
                pos['quantity'] * pos['avg_price']
                for pos in portfolio['positions'].values()
            )
            order_value = order['quantity'] * order['price']
            if order_value / total_value < 0.3:  # 不超过30%
                checks_passed += 1

            # 3. 价格合理性检查
            if 50 <= order['price'] <= 5000:
                checks_passed += 1

            if checks_passed >= 2:  # 通过大部分检查
                risk_checks += 1

        return risk_checks

    def _test_market_data_processing(self):
        """测试行情数据处理性能"""
        # 模拟高频行情数据
        ticks_processed = 0

        # 生成模拟tick数据
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']

        for i in range(200):  # 处理200个tick
            tick = {
                'symbol': random.choice(symbols),
                'price': 100.0 + random.random() * 100,
                'size': random.randint(100, 10000),
                'timestamp': time.time() + i * 0.001,  # 毫秒级间隔
                'bid': 99.5 + random.random() * 100,
                'ask': 100.5 + random.random() * 100,
                'bid_size': random.randint(100, 5000),
                'ask_size': random.randint(100, 5000)
            }

            # 模拟技术指标计算
            # 简单移动平均
            sma_5 = tick['price'] * random.uniform(0.98, 1.02)
            sma_20 = tick['price'] * random.uniform(0.95, 1.05)

            # 价格变化
            price_change = random.uniform(-0.02, 0.02)

            # 更新统计信息
            tick['sma_5'] = sma_5
            tick['sma_20'] = sma_20
            tick['price_change'] = price_change
            tick['volume_weighted_price'] = tick['price'] * (1 + random.uniform(-0.01, 0.01))

            ticks_processed += 1

        return ticks_processed

    # ============ 策略系统测试 ============

    def _test_strategy_execution(self):
        """测试策略执行性能"""
        # 模拟策略信号生成
        signals_generated = 0

        # 生成市场数据
        market_data = []
        for i in range(100):
            data_point = {
                'symbol': 'AAPL',
                'timestamp': time.time() + i,
                'open': 150.0 + random.random() * 10,
                'high': 155.0 + random.random() * 10,
                'low': 145.0 + random.random() * 10,
                'close': 150.0 + random.random() * 10,
                'volume': random.randint(100000, 1000000)
            }
            market_data.append(data_point)

        # 策略逻辑执行
        for i in range(1, len(market_data)):
            current = market_data[i]
            previous = market_data[i-1]

            # 简单移动平均策略
            if i >= 5:
                sma_5 = sum(market_data[j]['close'] for j in range(i-4, i+1)) / 5

                # 生成交易信号
                signal = None
                if current['close'] > sma_5 * 1.02:
                    signal = 'buy'
                elif current['close'] < sma_5 * 0.98:
                    signal = 'sell'

                if signal:
                    signals_generated += 1

        return signals_generated

    def _test_backtesting_performance(self):
        """测试回测性能"""
        # 模拟历史数据回测
        trades_executed = 0

        # 生成历史价格数据
        prices = []
        base_price = 100.0
        for i in range(252):  # 一年的交易日
            daily_return = random.normalvariate(0, 0.02)  # 2%的日波动率
            base_price *= (1 + daily_return)
            prices.append({
                'date': f"2023-{(i // 21) + 1:02d}-{(i % 21) + 1:02d}",
                'price': base_price,
                'volume': random.randint(1000000, 10000000)
            })

        # 回测策略
        portfolio_value = 100000.0
        position = 0

        for i in range(20, len(prices)):  # 从第20天开始，确保有足够历史数据
            current_price = prices[i]['price']
            sma_20 = sum(prices[j]['price'] for j in range(i-19, i+1)) / 20

            # 简单的移动平均策略
            if current_price > sma_20 * 1.05 and position == 0:
                # 买入信号
                shares = int(portfolio_value * 0.1 / current_price)  # 投入10%资金
                if shares > 0:
                    position = shares
                    portfolio_value -= shares * current_price
                    trades_executed += 1

            elif current_price < sma_20 * 0.95 and position > 0:
                # 卖出信号
                portfolio_value += position * current_price
                position = 0
                trades_executed += 1

        return trades_executed

    def _test_ml_model_inference(self):
        """测试机器学习模型推理性能"""
        # 模拟ML模型预测
        predictions_made = 0

        # 生成特征数据
        features = []
        for i in range(100):
            feature_vector = {
                'price_ma_5': 150.0 + random.random() * 10,
                'price_ma_20': 150.0 + random.random() * 10,
                'rsi': random.uniform(20, 80),
                'macd': random.uniform(-2, 2),
                'volume_ratio': random.uniform(0.5, 2.0),
                'volatility': random.uniform(0.1, 0.5),
                'momentum': random.uniform(-0.1, 0.1)
            }
            features.append(feature_vector)

        # 模拟模型推理
        for feature_vector in features:
            # 简单的线性模型模拟
            prediction_score = (
                feature_vector['rsi'] * 0.1 +
                feature_vector['macd'] * 0.2 +
                feature_vector['momentum'] * 0.3 +
                random.uniform(-0.1, 0.1)  # 噪声
            )

            # 转换为概率
            probability = 1 / (1 + np.exp(-prediction_score))  # sigmoid函数

            # 生成预测
            prediction = {
                'probability': probability,
                'prediction': 'buy' if probability > 0.6 else ('sell' if probability < 0.4 else 'hold'),
                'confidence': abs(probability - 0.5) * 2
            }

            predictions_made += 1

        return predictions_made

    def run_all_benchmarks(self, iterations: int = 1000, concurrent_users: Optional[List[int]] = None):
        """运行所有基准测试"""
        if concurrent_users is None:
            concurrent_users = [1, 5, 10, 20]

        results = {}

        for suite_name in self.framework.test_suites.keys():
            print(f"\n{'='*60}")
            print(f"运行基准测试: {suite_name}")
            print(f"{'='*60}")

            try:
                result = self.framework.run_benchmark_suite(
                    suite_name,
                    iterations=iterations,
                    concurrent_users=concurrent_users
                )
                results[suite_name] = result

                print(f"✅ 测试完成 - 性能水平: {result.performance_level.value}")
                print(f"   总测试数: {result.total_tests}")
                print(f"   通过测试: {result.passed_tests}")
                print(f"   失败测试: {result.failed_tests}")

                if result.recommendations:
                    print("   优化建议:")
                    for rec in result.recommendations:
                        print(f"   - {rec}")

            except Exception as e:
                print(f"❌ 测试失败: {e}")
                results[suite_name] = None

        return results


def main():
    """主函数"""
    print("🚀 RQA2025 核心组件性能基准测试套件")
    print("="*70)

    # 创建测试套件
    benchmark_suite = CoreComponentBenchmarkSuite()

    # 运行基准测试
    results = benchmark_suite.run_all_benchmarks(
        iterations=500,  # 减少迭代次数以加快测试
        concurrent_users=[1, 5, 10]
    )

    # 生成汇总报告
    print("\n" + "="*70)
    print("📊 基准测试汇总报告")
    print("="*70)

    successful_tests = sum(1 for r in results.values() if r is not None)
    total_tests = len(results)

    print(f"总测试套件: {total_tests}")
    print(f"成功套件: {successful_tests}")
    print(f"失败套件: {total_tests - successful_tests}")
    print(f"成功率: {successful_tests/total_tests*100:.1f}%")

    # 按性能水平分类
    performance_levels = {}
    for suite_name, result in results.items():
        if result:
            level = result.performance_level.value
            if level not in performance_levels:
                performance_levels[level] = []
            performance_levels[level].append(suite_name)

    print("\n📈 性能水平分布:")
    for level, suites in performance_levels.items():
        print(f"  {level.upper()}: {len(suites)} 个测试套件")
        for suite in suites:
            print(f"    - {suite}")


if __name__ == "__main__":
    main()
