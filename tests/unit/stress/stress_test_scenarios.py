"""压力测试场景设计"""
from datetime import datetime, timedelta
import random
from typing import List, Dict
from src.data.data_manager import DataManager
from src.trading.order_executor import OrderManager

class StressTestScenarios:
    """压力测试场景设计器"""

    def __init__(self, data_manager: DataManager, order_manager: OrderManager):
        self.data = data_manager
        self.orders = order_manager
        self.scenarios = self._init_scenarios()

    def _init_scenarios(self) -> List[Dict]:
        """初始化标准压力测试场景"""
        return [
            {
                "name": "2015股灾重现",
                "description": "模拟2015年A股市场剧烈波动场景",
                "params": {
                    "index_drop": 0.3,  # 指数下跌30%
                    "stocks_down": 0.8,  # 80%股票跌停
                    "duration": timedelta(hours=2)
                }
            },
            {
                "name": "千股跌停",
                "description": "模拟市场恐慌性抛售场景",
                "params": {
                    "limit_down_ratio": 0.9,  # 90%股票跌停
                    "liquidity_drop": 0.7,  # 流动性下降70%
                    "duration": timedelta(hours=1)
                }
            },
            {
                "name": "Level2数据风暴",
                "description": "模拟极端行情下的高频数据冲击",
                "params": {
                    "tick_rate": 10000,  # 每秒1万笔行情
                    "order_rate": 5000,  # 每秒5千笔委托
                    "duration": timedelta(minutes=30)
                }
            },
            {
                "name": "流动性危机",
                "description": "模拟市场流动性枯竭场景",
                "params": {
                    "liquidity_drop": 0.8,  # 流动性下降80%
                    "spread_increase": 0.5,  # 买卖价差扩大50%
                    "duration": timedelta(hours=4)
                }
            },
            {
                "name": "政策突变",
                "description": "模拟突发政策对市场的影响",
                "params": {
                    "policy_change": True,
                    "impact_level": "high",  # 高影响级别
                    "duration": timedelta(hours=6)
                }
            },
            {
                "name": "熔断压力测试",
                "description": "测试系统对熔断机制的响应",
                "params": {
                    "circuit_breaker_levels": [0.05, 0.07, 0.10],  # 5%,7%,10%熔断
                    "recovery_time": timedelta(minutes=15)  # 15分钟恢复交易
                }
            }
        ]

    def generate_test_data(self, scenario_name: str) -> Dict:
        """生成指定场景的测试数据"""
        scenario = next(s for s in self.scenarios if s["name"] == scenario_name)

        # 生成市场数据
        market_data = self._generate_market_data(scenario)

        # 生成订单流
        orders = self._generate_order_flow(scenario)

        return {
            "scenario": scenario,
            "market_data": market_data,
            "orders": orders
        }

    def _generate_market_data(self, scenario: Dict) -> Dict:
        """生成市场行情数据"""
        params = scenario["params"]

        if scenario["name"] == "2015股灾重现":
            # 生成指数暴跌行情
            return {
                "timestamp": datetime.now(),
                "index": {
                    "SH000001": {
                        "open": 5000,
                        "high": 5000,
                        "low": 3500,
                        "close": 3500,
                        "change": -0.3
                    }
                },
                "stocks": self._generate_stock_data(
                    down_ratio=params["stocks_down"],
                    limit_down=True
                )
            }

        elif scenario["name"] == "千股跌停":
            # 生成千股跌停行情
            return {
                "timestamp": datetime.now(),
                "stocks": self._generate_stock_data(
                    down_ratio=params["limit_down_ratio"],
                    limit_down=True
                )
            }

        elif scenario["name"] == "Level2数据风暴":
            # 生成高频Level2行情
            return {
                "timestamp": datetime.now(),
                "ticks": [
                    self._generate_tick_data()
                    for _ in range(params["tick_rate"])
                ]
            }

        # 其他场景数据生成...
        return {}

    def _generate_stock_data(self, down_ratio: float, limit_down: bool = False) -> Dict:
        """生成股票行情数据"""
        stocks = {}
        symbols = self.data.get_all_symbols()[:1000]  # 取前1000只股票

        for symbol in symbols:
            if random.random() < down_ratio:
                # 下跌股票
                stocks[symbol] = {
                    "open": 100,
                    "high": 100,
                    "low": 90 if not limit_down else 90,  # 跌停价
                    "close": 90 if not limit_down else 90,
                    "change": -0.1
                }
            else:
                # 上涨或平盘股票
                stocks[symbol] = {
                    "open": 100,
                    "high": 110,
                    "low": 100,
                    "close": 105,
                    "change": 0.05
                }

        return stocks

    def _generate_tick_data(self) -> Dict:
        """生成单笔tick数据"""
        return {
            "symbol": random.choice(self.data.get_all_symbols()),
            "price": random.uniform(90, 110),
            "volume": random.randint(100, 10000),
            "bid": [
                {"price": random.uniform(90, 100), "volume": random.randint(100, 1000)}
                for _ in range(10)
            ],
            "ask": [
                {"price": random.uniform(100, 110), "volume": random.randint(100, 1000)}
                for _ in range(10)
            ]
        }

    def _generate_order_flow(self, scenario: Dict) -> List[Dict]:
        """生成订单流数据"""
        params = scenario["params"]
        orders = []

        if scenario["name"] == "Level2数据风暴":
            # 高频订单流
            for _ in range(params["order_rate"]):
                orders.append({
                    "symbol": random.choice(self.data.get_all_symbols()),
                    "price": random.uniform(90, 110),
                    "quantity": random.randint(100, 10000),
                    "direction": random.choice(["BUY", "SELL"])
                })

        elif scenario["name"] == "2015股灾重现":
            # 恐慌性抛售订单
            for _ in range(5000):  # 5000笔卖单
                orders.append({
                    "symbol": random.choice(self.data.get_all_symbols()),
                    "price": random.uniform(90, 100),
                    "quantity": random.randint(1000, 100000),
                    "direction": "SELL"
                })

        # 其他场景订单生成...
        return orders

class StressTestExecutor:
    """压力测试执行器"""

    def __init__(self, scenarios: StressTestScenarios):
        self.scenarios = scenarios

    def run_scenario(self, scenario_name: str):
        """执行指定场景的压力测试"""
        test_data = self.scenarios.generate_test_data(scenario_name)

        # 1. 加载市场数据
        self._load_market_data(test_data["market_data"])

        # 2. 执行订单流
        self._execute_orders(test_data["orders"])

        # 3. 监控系统指标
        metrics = self._monitor_system()

        return {
            "scenario": scenario_name,
            "metrics": metrics
        }

    def _load_market_data(self, market_data: Dict):
        """加载市场数据到系统中"""
        # 实现细节...
        pass

    def _execute_orders(self, orders: List[Dict]):
        """执行订单流"""
        for order in orders:
            try:
                self.scenarios.orders.place_order(order)
            except Exception as e:
                print(f"订单执行失败: {str(e)}")

    def _monitor_system(self) -> Dict:
        """监控系统关键指标"""
        return {
            "latency": random.uniform(10, 100),  # 模拟延迟
            "throughput": random.uniform(500, 5000),  # 模拟吞吐量
            "memory_usage": random.uniform(30, 90),  # 内存使用率
            "cpu_usage": random.uniform(20, 80)  # CPU使用率
        }

class StressTestAnalyzer:
    """压力测试结果分析器"""

    @staticmethod
    def analyze(results: List[Dict]) -> Dict:
        """分析压力测试结果"""
        summary = {
            "total_scenarios": len(results),
            "passed": 0,
            "failed": 0,
            "performance_metrics": []
        }

        for result in results:
            metrics = result["metrics"]

            # 检查关键指标
            if (metrics["latency"] < 100 and
                metrics["throughput"] > 1000 and
                metrics["memory_usage"] < 90 and
                metrics["cpu_usage"] < 90):
                summary["passed"] += 1
            else:
                summary["failed"] += 1

            summary["performance_metrics"].append({
                "scenario": result["scenario"],
                "metrics": metrics
            })

        return summary
