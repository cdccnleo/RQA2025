#!/usr/bin/env python3
"""
测试数据生成器

生成各种测试场景所需的模拟数据
支持交易数据、用户数据、配置数据等多种类型
"""

import random
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np


class DataType(Enum):
    """数据类型枚举"""
    USER = "user"
    TRANSACTION = "transaction"
    PORTFOLIO = "portfolio"
    MARKET_DATA = "market_data"
    CONFIGURATION = "configuration"
    PERFORMANCE = "performance"
    RISK_METRICS = "risk_metrics"
    LOG_ENTRY = "log_entry"


@dataclass
class TestUser:
    """测试用户数据"""
    user_id: str
    username: str
    email: str
    account_type: str
    balance: float
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True


@dataclass
class TestTransaction:
    """测试交易数据"""
    transaction_id: str
    user_id: str
    symbol: str
    transaction_type: str  # 'BUY' or 'SELL'
    quantity: int
    price: float
    total_amount: float
    timestamp: datetime
    status: str = 'COMPLETED'


@dataclass
class TestPortfolio:
    """测试投资组合数据"""
    portfolio_id: str
    user_id: str
    holdings: Dict[str, Dict[str, Union[int, float]]]
    total_value: float
    last_updated: datetime
    risk_level: str = 'MEDIUM'


class TestDataGenerator:
    """测试数据生成器"""

    def __init__(self, seed: int = 42):
        """初始化数据生成器"""
        random.seed(seed)
        np.random.seed(seed)
        self.seed = seed

    def generate_user(self, user_type: str = 'regular') -> TestUser:
        """生成测试用户"""
        user_id = str(uuid.uuid4())
        username = f"user_{random.randint(1000, 9999)}"
        email = f"{username}@example.com"

        # 根据用户类型设置不同的余额范围
        if user_type == 'premium':
            balance = round(random.uniform(50000, 500000), 2)
        elif user_type == 'vip':
            balance = round(random.uniform(100000, 1000000), 2)
        else:
            balance = round(random.uniform(1000, 50000), 2)

        created_at = datetime.now() - timedelta(days=random.randint(1, 365))
        last_login = created_at + timedelta(days=random.randint(1, 30)) if random.choice([True, False]) else None

        return TestUser(
            user_id=user_id,
            username=username,
            email=email,
            account_type=user_type,
            balance=balance,
            created_at=created_at,
            last_login=last_login,
            is_active=random.choice([True, True, True, False])  # 75%活跃用户
        )

    def generate_users(self, count: int, user_types: Optional[List[str]] = None) -> List[TestUser]:
        """生成多个测试用户"""
        if user_types is None:
            user_types = ['regular', 'premium', 'vip']

        users = []
        for _ in range(count):
            user_type = random.choice(user_types)
            users.append(self.generate_user(user_type))

        return users

    def generate_transaction(self, user_id: str, symbols: Optional[List[str]] = None) -> TestTransaction:
        """生成测试交易"""
        if symbols is None:
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']

        transaction_id = str(uuid.uuid4())
        symbol = random.choice(symbols)
        transaction_type = random.choice(['BUY', 'SELL'])

        # 根据股票设置不同的价格范围
        price_ranges = {
            'AAPL': (150, 250),
            'GOOGL': (2000, 3500),
            'MSFT': (300, 500),
            'AMZN': (2500, 4500),
            'TSLA': (200, 400),
            'NVDA': (400, 1000),
            'META': (250, 450),
            'NFLX': (400, 700)
        }

        min_price, max_price = price_ranges.get(symbol, (100, 500))
        price = round(random.uniform(min_price, max_price), 2)
        quantity = random.randint(1, 100)

        # 考虑交易费率
        commission_rate = 0.001  # 0.1%
        commission = price * quantity * commission_rate
        total_amount = price * quantity + commission

        timestamp = datetime.now() - timedelta(days=random.randint(0, 30))

        return TestTransaction(
            transaction_id=transaction_id,
            user_id=user_id,
            symbol=symbol,
            transaction_type=transaction_type,
            quantity=quantity,
            price=price,
            total_amount=round(total_amount, 2),
            timestamp=timestamp,
            status=random.choice(['COMPLETED', 'COMPLETED', 'COMPLETED', 'PENDING', 'FAILED'])
        )

    def generate_transactions(self, user_ids: List[str], count_per_user: int = 5) -> List[TestTransaction]:
        """生成多个测试交易"""
        transactions = []
        for user_id in user_ids:
            for _ in range(count_per_user):
                transactions.append(self.generate_transaction(user_id))

        return transactions

    def generate_portfolio(self, user_id: str) -> TestPortfolio:
        """生成测试投资组合"""
        portfolio_id = str(uuid.uuid4())

        # 生成持仓数据
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        holdings = {}

        total_value = 0
        for symbol in random.sample(symbols, random.randint(2, 5)):
            quantity = random.randint(10, 200)
            price = round(random.uniform(100, 1000), 2)
            value = quantity * price
            total_value += value

            holdings[symbol] = {
                'quantity': quantity,
                'avg_price': round(random.uniform(price * 0.8, price * 1.2), 2),
                'current_price': price,
                'value': round(value, 2),
                'gain_loss': round(random.uniform(-50, 100), 2)
            }

        # 确定风险等级
        risk_levels = ['LOW', 'MEDIUM', 'HIGH']
        risk_level = random.choice(risk_levels)

        return TestPortfolio(
            portfolio_id=portfolio_id,
            user_id=user_id,
            holdings=holdings,
            total_value=round(total_value, 2),
            last_updated=datetime.now() - timedelta(hours=random.randint(1, 24)),
            risk_level=risk_level
        )

    def generate_portfolios(self, user_ids: List[str]) -> List[TestPortfolio]:
        """生成多个测试投资组合"""
        return [self.generate_portfolio(user_id) for user_id in user_ids]

    def generate_market_data(self, symbols: Optional[List[str]] = None, days: int = 30) -> Dict[str, List[Dict]]:
        """生成测试市场数据"""
        if symbols is None:
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']

        market_data = {}

        base_date = datetime.now() - timedelta(days=days)

        for symbol in symbols:
            symbol_data = []

            # 设置基础价格
            base_prices = {
                'AAPL': 180,
                'GOOGL': 2800,
                'MSFT': 380,
                'AMZN': 3500,
                'TSLA': 250
            }

            current_price = base_prices.get(symbol, 200)

            for i in range(days):
                # 模拟价格波动
                daily_return = random.gauss(0, 0.02)  # 正态分布，均值0，标准差2%
                current_price *= (1 + daily_return)
                current_price = max(current_price, 1)  # 确保价格不为负

                # 生成成交量
                base_volume = {
                    'AAPL': 50000000,
                    'GOOGL': 2000000,
                    'MSFT': 25000000,
                    'AMZN': 3000000,
                    'TSLA': 15000000
                }

                volume = int(random.gauss(base_volume.get(symbol, 10000000), base_volume.get(symbol, 10000000) * 0.3))

                data_point = {
                    'date': (base_date + timedelta(days=i)).strftime('%Y-%m-%d'),
                    'symbol': symbol,
                    'open': round(current_price * random.uniform(0.98, 1.02), 2),
                    'high': round(current_price * random.uniform(1.00, 1.05), 2),
                    'low': round(current_price * random.uniform(0.95, 1.00), 2),
                    'close': round(current_price, 2),
                    'volume': volume,
                    'adj_close': round(current_price, 2)
                }

                symbol_data.append(data_point)

            market_data[symbol] = symbol_data

        return market_data

    def generate_performance_metrics(self, user_ids: List[str]) -> Dict[str, Dict]:
        """生成测试性能指标"""
        performance_data = {}

        for user_id in user_ids:
            # 生成年度回报率
            annual_return = round(random.uniform(-20, 50), 2)
            volatility = round(random.uniform(10, 40), 2)
            sharpe_ratio = round(annual_return / volatility, 2) if volatility > 0 else 0
            max_drawdown = round(random.uniform(5, 30), 2)

            performance_data[user_id] = {
                'annual_return_pct': annual_return,
                'volatility_pct': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown_pct': max_drawdown,
                'win_rate_pct': round(random.uniform(40, 80), 2),
                'total_trades': random.randint(50, 500),
                'profit_factor': round(random.uniform(0.8, 2.5), 2),
                'calmar_ratio': round(annual_return / max_drawdown, 2) if max_drawdown > 0 else 0
            }

        return performance_data

    def generate_risk_metrics(self, user_ids: List[str]) -> Dict[str, Dict]:
        """生成测试风险指标"""
        risk_data = {}

        for user_id in user_ids:
            # 生成VaR和CVaR
            var_95 = round(random.uniform(5, 25), 2)
            cvar_95 = round(var_95 * random.uniform(1.2, 1.8), 2)

            risk_data[user_id] = {
                'var_95_pct': var_95,
                'cvar_95_pct': cvar_95,
                'beta': round(random.uniform(0.5, 1.5), 2),
                'correlation_avg': round(random.uniform(-0.3, 0.3), 2),
                'diversification_ratio': round(random.uniform(1.0, 3.0), 2),
                'concentration_risk': round(random.uniform(10, 50), 2),
                'liquidity_risk': random.choice(['LOW', 'MEDIUM', 'HIGH']),
                'stress_test_loss_pct': round(random.uniform(15, 60), 2)
            }

        return risk_data

    def generate_log_entries(self, count: int = 100, include_errors: bool = True) -> List[Dict]:
        """生成测试日志条目"""
        log_levels = ['INFO', 'WARNING', 'ERROR', 'DEBUG', 'CRITICAL']
        components = ['web_server', 'database', 'cache', 'worker', 'monitor', 'api']
        messages = [
            'Request processed successfully',
            'Database connection established',
            'Cache miss for key: {}',
            'User authentication successful',
            'Order execution completed',
            'Market data updated',
            'Performance metrics collected',
            'Health check passed'
        ]

        if include_errors:
            messages.extend([
                'Database connection timeout',
                'Cache service unavailable',
                'Authentication failed for user: {}',
                'Order validation error',
                'External API rate limit exceeded',
                'Memory usage above threshold',
                'Disk space running low'
            ])

        log_entries = []

        base_time = datetime.now() - timedelta(hours=24)

        for i in range(count):
            timestamp = base_time + timedelta(seconds=i * 60)  # 每分钟一条日志

            # 根据是否包含错误来选择日志级别
            if include_errors and random.random() < 0.1:  # 10%的错误日志
                level = random.choice(['ERROR', 'WARNING', 'CRITICAL'])
                message = random.choice([msg for msg in messages if any(word in msg.lower() for word in ['error', 'timeout', 'unavailable', 'failed', 'exceeded', 'above', 'low'])])
            else:
                level = random.choice(['INFO', 'DEBUG'])
                message = random.choice([msg for msg in messages if not any(word in msg.lower() for word in ['error', 'timeout', 'unavailable', 'failed', 'exceeded', 'above', 'low'])])

            # 格式化消息
            if '{}' in message:
                message = message.format(f"value_{random.randint(1000, 9999)}")

            log_entry = {
                'timestamp': timestamp.isoformat(),
                'level': level,
                'component': random.choice(components),
                'message': message,
                'request_id': str(uuid.uuid4()) if random.random() < 0.3 else None,
                'user_id': str(uuid.uuid4()) if random.random() < 0.2 else None,
                'duration_ms': round(random.uniform(10, 5000), 2) if random.random() < 0.4 else None
            }

            log_entries.append(log_entry)

        return log_entries

    def generate_configuration_data(self) -> Dict[str, Any]:
        """生成测试配置数据"""
        config = {
            'application': {
                'name': 'RQA2025 Trading System',
                'version': '2.1.0',
                'environment': 'testing',
                'debug': False,
                'log_level': 'INFO'
            },
            'database': {
                'host': 'test-db.example.com',
                'port': 5432,
                'name': 'trading_test',
                'pool_size': 10,
                'connection_timeout': 30,
                'ssl_enabled': True
            },
            'cache': {
                'redis_host': 'test-redis.example.com',
                'redis_port': 6379,
                'ttl_default': 3600,
                'max_memory': '512MB',
                'compression': True
            },
            'trading': {
                'max_order_size': 100000,
                'commission_rate': 0.001,
                'min_order_value': 100,
                'market_hours_start': '09:30',
                'market_hours_end': '16:00',
                'supported_exchanges': ['NYSE', 'NASDAQ', 'AMEX']
            },
            'risk_management': {
                'max_portfolio_risk': 0.1,
                'max_single_position': 0.05,
                'var_confidence_level': 0.95,
                'stress_test_frequency': 'daily',
                'alert_thresholds': {
                    'high': 0.8,
                    'medium': 0.6,
                    'low': 0.4
                }
            },
            'monitoring': {
                'metrics_collection_interval': 60,
                'alert_email_recipients': ['admin@example.com', 'devops@example.com'],
                'dashboard_refresh_rate': 30,
                'log_retention_days': 30,
                'performance_monitoring_enabled': True
            }
        }

        return config

    def save_to_json(self, data: Any, filename: str) -> None:
        """将数据保存为JSON文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

    def load_from_json(self, filename: str) -> Any:
        """从JSON文件加载数据"""
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)

    def generate_complete_test_dataset(self, user_count: int = 100) -> Dict[str, Any]:
        """生成完整的测试数据集"""
        print(f"Generating test dataset with {user_count} users...")

        # 生成用户
        users = self.generate_users(user_count)
        user_ids = [user.user_id for user in users]

        # 生成交易
        transactions = self.generate_transactions(user_ids, count_per_user=10)

        # 生成投资组合
        portfolios = self.generate_portfolios(user_ids)

        # 生成市场数据
        market_data = self.generate_market_data(days=60)

        # 生成性能指标
        performance_metrics = self.generate_performance_metrics(user_ids)

        # 生成风险指标
        risk_metrics = self.generate_risk_metrics(user_ids)

        # 生成日志
        logs = self.generate_log_entries(count=500)

        # 生成配置
        config = self.generate_configuration_data()

        dataset = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'generator_version': '1.0.0',
                'user_count': user_count,
                'transaction_count': len(transactions),
                'portfolio_count': len(portfolios),
                'log_count': len(logs),
                'market_data_symbols': list(market_data.keys())
            },
            'users': [asdict(user) for user in users],
            'transactions': [asdict(tx) for tx in transactions],
            'portfolios': [asdict(portfolio) for portfolio in portfolios],
            'market_data': market_data,
            'performance_metrics': performance_metrics,
            'risk_metrics': risk_metrics,
            'logs': logs,
            'configuration': config
        }

        print("Generated complete test dataset with:")
        print(f"  - {len(users)} users")
        print(f"  - {len(transactions)} transactions")
        print(f"  - {len(portfolios)} portfolios")
        print(f"  - {len(logs)} log entries")
        print(f"  - Market data for {len(market_data)} symbols")

        return dataset


def main():
    """主函数：生成示例测试数据"""
    generator = TestDataGenerator(seed=42)

    # 生成小型测试数据集
    print("Generating small test dataset...")
    small_dataset = generator.generate_complete_test_dataset(user_count=10)

    # 保存到文件
    generator.save_to_json(small_dataset, 'tests/test_data/small_test_dataset.json')

    # 生成用户和交易的示例
    print("\nExample user generation:")
    sample_user = generator.generate_user('premium')
    print(f"Sample user: {sample_user.username} ({sample_user.account_type}) - Balance: ${sample_user.balance}")

    print("\nExample transaction generation:")
    sample_transaction = generator.generate_transaction(sample_user.user_id)
    print(f"Sample transaction: {sample_transaction.symbol} {sample_transaction.transaction_type} {sample_transaction.quantity} shares @ ${sample_transaction.price}")

    print("\nTest data generation completed!")
    print("Files saved:")
    print("  - tests/test_data/small_test_dataset.json")


if __name__ == "__main__":
    main()
