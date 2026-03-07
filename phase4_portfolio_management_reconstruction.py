#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 4: 投资组合管理功能重建

修复技术债务: 投资组合管理功能重建
解决业务验收测试中发现的投资组合管理功能缺失的问题
"""

import logging
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import uuid
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 投资组合管理数据结构


class PortfolioType(Enum):
    """投资组合类型"""
    STOCK = "stock"  # 股票组合
    BOND = "bond"    # 债券组合
    MIXED = "mixed"  # 混合组合
    ETF = "etf"      # ETF组合
    INDEX = "index"  # 指数组合


class RebalanceStrategy(Enum):
    """再平衡策略"""
    EQUAL_WEIGHT = "equal_weight"      # 等权重
    MARKET_CAP = "market_cap"          # 市值加权
    RISK_PARITY = "risk_parity"        # 风险平价
    MINIMUM_VARIANCE = "minimum_variance"  # 最小方差
    MAXIMUM_SHARPE = "maximum_sharpe"  # 最大夏普比率


class RiskTolerance(Enum):
    """风险偏好"""
    CONSERVATIVE = "conservative"  # 保守型
    MODERATE = "moderate"          # 稳健型
    AGGRESSIVE = "aggressive"      # 激进型


@dataclass
class Asset:
    """资产数据结构"""
    symbol: str
    name: str
    asset_type: str
    current_price: float
    quantity: float
    market_value: float = field(init=False)
    weight: float = 0.0
    expected_return: float = 0.0
    volatility: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        self.market_value = self.quantity * self.current_price

    def update_price(self, new_price: float):
        """更新价格"""
        self.current_price = new_price
        self.market_value = self.quantity * self.current_price
        self.last_updated = datetime.now()


@dataclass
class Portfolio:
    """投资组合数据结构"""
    portfolio_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    portfolio_type: PortfolioType = PortfolioType.STOCK
    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE
    target_weights: Dict[str, float] = field(default_factory=dict)
    assets: Dict[str, Asset] = field(default_factory=dict)
    cash_balance: float = 0.0
    total_value: float = field(init=False)
    rebalance_threshold: float = 0.05  # 再平衡阈值
    rebalance_strategy: RebalanceStrategy = RebalanceStrategy.EQUAL_WEIGHT
    created_time: datetime = field(default_factory=datetime.now)
    last_rebalanced: Optional[datetime] = None

    def __post_init__(self):
        self._update_total_value()

    def _update_total_value(self):
        """更新总价值"""
        self.total_value = self.cash_balance + \
            sum(asset.market_value for asset in self.assets.values())

    def add_asset(self, asset: Asset):
        """添加资产"""
        self.assets[asset.symbol] = asset
        self._update_total_value()
        self._update_weights()

    def remove_asset(self, symbol: str) -> Optional[Asset]:
        """移除资产"""
        if symbol in self.assets:
            asset = self.assets.pop(symbol)
            self._update_total_value()
            self._update_weights()
            return asset
        return None

    def update_asset_price(self, symbol: str, new_price: float):
        """更新资产价格"""
        if symbol in self.assets:
            self.assets[symbol].update_price(new_price)
            self._update_total_value()
            self._update_weights()

    def _update_weights(self):
        """更新权重"""
        if self.total_value > 0:
            for asset in self.assets.values():
                asset.weight = asset.market_value / self.total_value

    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取绩效指标"""
        if not self.assets:
            return {}

        weights = np.array([asset.weight for asset in self.assets.values()])
        returns = np.array([asset.expected_return for asset in self.assets.values()])
        volatilities = np.array([asset.volatility for asset in self.assets.values()])

        # 投资组合预期收益率
        portfolio_return = np.sum(weights * returns)

        # 投资组合波动率（简化计算）
        portfolio_volatility = np.sqrt(np.sum(weights ** 2 * volatilities ** 2))

        # 夏普比率（假设无风险利率3%）
        risk_free_rate = 0.03
        sharpe_ratio = (portfolio_return - risk_free_rate) / \
            portfolio_volatility if portfolio_volatility > 0 else 0

        return {
            'portfolio_return': portfolio_return,
            'portfolio_volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'total_assets': len(self.assets),
            'cash_ratio': self.cash_balance / self.total_value if self.total_value > 0 else 0
        }


@dataclass
class RebalanceRecommendation:
    """再平衡建议"""
    portfolio_id: str
    recommendations: List[Dict[str, Any]]
    expected_trades: List[Dict[str, Any]]
    estimated_cost: float
    timestamp: datetime = field(default_factory=datetime.now)


class PortfolioOptimizer:
    """投资组合优化器"""

    def __init__(self):
        self.optimizers = {
            RebalanceStrategy.EQUAL_WEIGHT: self._optimize_equal_weight,
            RebalanceStrategy.MARKET_CAP: self._optimize_market_cap,
            RebalanceStrategy.RISK_PARITY: self._optimize_risk_parity,
            RebalanceStrategy.MINIMUM_VARIANCE: self._optimize_minimum_variance,
            RebalanceStrategy.MAXIMUM_SHARPE: self._optimize_maximum_sharpe
        }

    def optimize_portfolio(self, portfolio: Portfolio, strategy: RebalanceStrategy) -> Dict[str, float]:
        """优化投资组合"""
        optimizer = self.optimizers.get(strategy)
        if optimizer:
            return optimizer(portfolio)
        else:
            logger.warning(f"未知的优化策略: {strategy}")
            return {}

    def _optimize_equal_weight(self, portfolio: Portfolio) -> Dict[str, float]:
        """等权重优化"""
        if not portfolio.assets:
            return {}

        equal_weight = 1.0 / len(portfolio.assets)
        return {symbol: equal_weight for symbol in portfolio.assets.keys()}

    def _optimize_market_cap(self, portfolio: Portfolio) -> Dict[str, float]:
        """市值加权优化"""
        if not portfolio.assets:
            return {}

        total_market_cap = sum(asset.market_value for asset in portfolio.assets.values())
        if total_market_cap == 0:
            return self._optimize_equal_weight(portfolio)

        return {
            symbol: asset.market_value / total_market_cap
            for symbol, asset in portfolio.assets.items()
        }

    def _optimize_risk_parity(self, portfolio: Portfolio) -> Dict[str, float]:
        """风险平价优化"""
        if not portfolio.assets:
            return {}

        # 简化的风险平价：根据波动率的倒数分配权重
        volatilities = np.array([asset.volatility for asset in portfolio.assets.values()])

        # 避免零波动率
        volatilities = np.where(volatilities == 0, 0.01, volatilities)

        risk_contributions = 1.0 / volatilities
        total_risk_contribution = np.sum(risk_contributions)

        if total_risk_contribution == 0:
            return self._optimize_equal_weight(portfolio)

        weights = {}
        for i, symbol in enumerate(portfolio.assets.keys()):
            weights[symbol] = risk_contributions[i] / total_risk_contribution

        return weights

    def _optimize_minimum_variance(self, portfolio: Portfolio) -> Dict[str, float]:
        """最小方差优化"""
        # 简化的最小方差：降低高波动性资产权重
        if not portfolio.assets:
            return {}

        volatilities = np.array([asset.volatility for asset in portfolio.assets.values()])
        max_vol = np.max(volatilities)

        if max_vol == 0:
            return self._optimize_equal_weight(portfolio)

        # 根据波动率的倒数分配权重，但限制最低权重
        risk_scores = 1.0 / volatilities
        total_score = np.sum(risk_scores)

        weights = {}
        for i, symbol in enumerate(portfolio.assets.keys()):
            weight = risk_scores[i] / total_score
            # 限制权重不超过50%
            weights[symbol] = min(weight, 0.5)

        # 重新归一化
        total_weight = sum(weights.values())
        weights = {symbol: weight / total_weight for symbol, weight in weights.items()}

        return weights

    def _optimize_maximum_sharpe(self, portfolio: Portfolio) -> Dict[str, float]:
        """最大夏普比率优化"""
        # 简化的夏普比率优化：偏向高收益低波动资产
        if not portfolio.assets:
            return {}

        risk_free_rate = 0.03
        sharpe_ratios = []

        for asset in portfolio.assets.values():
            sharpe = (asset.expected_return - risk_free_rate) / \
                asset.volatility if asset.volatility > 0 else 0
            sharpe_ratios.append(max(sharpe, 0))  # 避免负夏普比率

        sharpe_ratios = np.array(sharpe_ratios)
        total_sharpe = np.sum(sharpe_ratios)

        if total_sharpe == 0:
            return self._optimize_equal_weight(portfolio)

        weights = {}
        for i, symbol in enumerate(portfolio.assets.keys()):
            weights[symbol] = sharpe_ratios[i] / total_sharpe

        return weights


class PortfolioRebalancer:
    """投资组合再平衡器"""

    def __init__(self):
        self.optimizer = PortfolioOptimizer()

    def check_rebalance_needed(self, portfolio: Portfolio) -> bool:
        """检查是否需要再平衡"""
        if not portfolio.target_weights or not portfolio.assets:
            return False

        threshold = portfolio.rebalance_threshold

        for symbol, asset in portfolio.assets.items():
            target_weight = portfolio.target_weights.get(symbol, 0)
            current_weight = asset.weight

            if abs(current_weight - target_weight) > threshold:
                return True

        return False

    def generate_rebalance_plan(self, portfolio: Portfolio) -> RebalanceRecommendation:
        """生成再平衡计划"""
        # 优化目标权重
        optimized_weights = self.optimizer.optimize_portfolio(
            portfolio, portfolio.rebalance_strategy)

        if not optimized_weights:
            return RebalanceRecommendation(
                portfolio_id=portfolio.portfolio_id,
                recommendations=[],
                expected_trades=[],
                estimated_cost=0.0
            )

        # 更新目标权重
        portfolio.target_weights = optimized_weights

        recommendations = []
        expected_trades = []
        total_cost = 0.0

        for symbol, asset in portfolio.assets.items():
            target_weight = optimized_weights.get(symbol, 0)
            current_weight = asset.weight
            deviation = target_weight - current_weight

            if abs(deviation) > portfolio.rebalance_threshold:
                # 计算需要交易的数量
                target_value = target_weight * portfolio.total_value
                current_value = asset.market_value
                trade_value = target_value - current_value
                trade_quantity = trade_value / asset.current_price

                recommendations.append({
                    'symbol': symbol,
                    'current_weight': current_weight,
                    'target_weight': target_weight,
                    'deviation': deviation,
                    'trade_quantity': trade_quantity,
                    'trade_value': trade_value,
                    'current_price': asset.current_price
                })

                if abs(trade_quantity) > 0:
                    expected_trades.append({
                        'symbol': symbol,
                        'action': 'BUY' if trade_quantity > 0 else 'SELL',
                        'quantity': abs(trade_quantity),
                        'price': asset.current_price,
                        'value': abs(trade_value),
                        'commission': abs(trade_value) * 0.0005  # 假设佣金0.05%
                    })
                    total_cost += abs(trade_value) * 0.0005

        return RebalanceRecommendation(
            portfolio_id=portfolio.portfolio_id,
            recommendations=recommendations,
            expected_trades=expected_trades,
            estimated_cost=total_cost
        )


class PortfolioAnalyzer:
    """投资组合分析器"""

    def __init__(self):
        pass

    def analyze_portfolio(self, portfolio: Portfolio) -> Dict[str, Any]:
        """分析投资组合"""
        analysis = {
            'portfolio_info': {
                'portfolio_id': portfolio.portfolio_id,
                'name': portfolio.name,
                'type': portfolio.portfolio_type.value,
                'total_value': portfolio.total_value,
                'cash_balance': portfolio.cash_balance,
                'assets_count': len(portfolio.assets)
            },
            'asset_allocation': [],
            'performance_metrics': portfolio.get_performance_metrics(),
            'risk_analysis': self._analyze_risk(portfolio),
            'diversification': self._analyze_diversification(portfolio)
        }

        # 资产配置
        for symbol, asset in portfolio.assets.items():
            analysis['asset_allocation'].append({
                'symbol': symbol,
                'name': asset.name,
                'weight': asset.weight,
                'market_value': asset.market_value,
                'quantity': asset.quantity,
                'current_price': asset.current_price
            })

        return analysis

    def _analyze_risk(self, portfolio: Portfolio) -> Dict[str, Any]:
        """风险分析"""
        if not portfolio.assets:
            return {}

        assets = list(portfolio.assets.values())

        # 波动率分析
        volatilities = [asset.volatility for asset in assets]
        avg_volatility = np.mean(volatilities) if volatilities else 0

        # 集中度分析
        weights = [asset.weight for asset in assets]
        max_weight = max(weights) if weights else 0

        # 计算投资组合波动率（简化）
        portfolio_volatility = np.sqrt(sum(w ** 2 * v ** 2 for w, v in zip(weights, volatilities)))

        return {
            'average_volatility': avg_volatility,
            'portfolio_volatility': portfolio_volatility,
            'max_weight': max_weight,
            'concentration_risk': 'HIGH' if max_weight > 0.3 else 'MEDIUM' if max_weight > 0.2 else 'LOW'
        }

    def _analyze_diversification(self, portfolio: Portfolio) -> Dict[str, Any]:
        """分散化分析"""
        if not portfolio.assets:
            return {}

        asset_types = {}
        sectors = {}  # 简化为资产类型

        for asset in portfolio.assets.values():
            asset_type = asset.asset_type
            asset_types[asset_type] = asset_types.get(asset_type, 0) + asset.weight

        diversification_score = len(asset_types) / len(portfolio.assets) if portfolio.assets else 0

        return {
            'asset_type_diversity': asset_types,
            'diversification_score': diversification_score,
            'diversification_level': 'HIGH' if diversification_score > 0.7 else 'MEDIUM' if diversification_score > 0.4 else 'LOW'
        }


class PortfolioManager:
    """投资组合管理器"""

    def __init__(self):
        self.portfolios: Dict[str, Portfolio] = {}
        self.optimizer = PortfolioOptimizer()
        self.rebalancer = PortfolioRebalancer()
        self.analyzer = PortfolioAnalyzer()
        self._lock = threading.Lock()

        logger.info("投资组合管理器初始化完成")

    def create_portfolio(self, name: str, portfolio_type: PortfolioType,
                         risk_tolerance: RiskTolerance,
                         initial_balance: float = 0.0) -> str:
        """创建投资组合"""
        with self._lock:
            portfolio = Portfolio(
                name=name,
                portfolio_type=portfolio_type,
                risk_tolerance=risk_tolerance,
                cash_balance=initial_balance
            )

            self.portfolios[portfolio.portfolio_id] = portfolio
            logger.info(f"创建投资组合: {name} (ID: {portfolio.portfolio_id})")
            return portfolio.portfolio_id

    def get_portfolio(self, portfolio_id: str) -> Optional[Portfolio]:
        """获取投资组合"""
        return self.portfolios.get(portfolio_id)

    def add_asset_to_portfolio(self, portfolio_id: str, symbol: str, name: str,
                               asset_type: str, quantity: float, price: float) -> bool:
        """添加资产到投资组合"""
        with self._lock:
            portfolio = self.portfolios.get(portfolio_id)
            if not portfolio:
                return False

            asset = Asset(
                symbol=symbol,
                name=name,
                asset_type=asset_type,
                current_price=price,
                quantity=quantity
            )

            portfolio.add_asset(asset)
            logger.info(f"添加资产到投资组合 {portfolio_id}: {symbol} {quantity}股 @ {price}")
            return True

    def update_asset_prices(self, portfolio_id: str, price_updates: Dict[str, float]):
        """更新资产价格"""
        with self._lock:
            portfolio = self.portfolios.get(portfolio_id)
            if not portfolio:
                return

            for symbol, price in price_updates.items():
                portfolio.update_asset_price(symbol, price)

            logger.debug(f"更新投资组合 {portfolio_id} 资产价格")

    def rebalance_portfolio(self, portfolio_id: str) -> Optional[RebalanceRecommendation]:
        """再平衡投资组合"""
        with self._lock:
            portfolio = self.portfolios.get(portfolio_id)
            if not portfolio:
                return None

            if not self.rebalancer.check_rebalance_needed(portfolio):
                logger.info(f"投资组合 {portfolio_id} 无需再平衡")
                return None

            recommendation = self.rebalancer.generate_rebalance_plan(portfolio)
            portfolio.last_rebalanced = datetime.now()

            logger.info(f"生成投资组合 {portfolio_id} 再平衡计划: {len(recommendation.expected_trades)} 笔交易")
            return recommendation

    def analyze_portfolio(self, portfolio_id: str) -> Optional[Dict[str, Any]]:
        """分析投资组合"""
        portfolio = self.portfolios.get(portfolio_id)
        if not portfolio:
            return None

        return self.analyzer.analyze_portfolio(portfolio)

    def get_all_portfolios(self) -> List[Dict[str, Any]]:
        """获取所有投资组合摘要"""
        summaries = []
        for portfolio in self.portfolios.values():
            summaries.append({
                'portfolio_id': portfolio.portfolio_id,
                'name': portfolio.name,
                'type': portfolio.portfolio_type.value,
                'total_value': portfolio.total_value,
                'assets_count': len(portfolio.assets),
                'cash_balance': portfolio.cash_balance,
                'last_rebalanced': portfolio.last_rebalanced.isoformat() if portfolio.last_rebalanced else None
            })

        return summaries

    def optimize_portfolio(self, portfolio_id: str, strategy: RebalanceStrategy) -> Dict[str, float]:
        """优化投资组合"""
        portfolio = self.portfolios.get(portfolio_id)
        if not portfolio:
            return {}

        optimized_weights = self.optimizer.optimize_portfolio(portfolio, strategy)
        logger.info(f"优化投资组合 {portfolio_id} 使用策略 {strategy.value}")
        return optimized_weights


def test_portfolio_management():
    """测试投资组合管理功能"""
    logger.info("测试投资组合管理功能重建...")

    # 创建投资组合管理器
    manager = PortfolioManager()

    # 1. 创建投资组合
    logger.info("\n1. 创建投资组合")
    portfolio_id = manager.create_portfolio(
        name="测试投资组合",
        portfolio_type=PortfolioType.STOCK,
        risk_tolerance=RiskTolerance.MODERATE,
        initial_balance=100000.0
    )
    logger.info(f"创建投资组合成功: {portfolio_id}")

    # 2. 添加资产
    logger.info("\n2. 添加资产")
    assets = [
        ("AAPL", "苹果公司", "TECH", 100, 150.0),
        ("GOOGL", "谷歌公司", "TECH", 50, 2500.0),
        ("MSFT", "微软公司", "TECH", 80, 300.0),
        ("TSLA", "特斯拉", "AUTO", 30, 200.0)
    ]

    for symbol, name, asset_type, quantity, price in assets:
        success = manager.add_asset_to_portfolio(
            portfolio_id, symbol, name, asset_type, quantity, price)
        if success:
            logger.info(f"添加资产: {symbol} {quantity}股 @ {price}")

    # 3. 分析投资组合
    logger.info("\n3. 分析投资组合")
    analysis = manager.analyze_portfolio(portfolio_id)
    if analysis:
        logger.info("投资组合分析:")
        logger.info(f"  总价值: {analysis['portfolio_info']['total_value']:.2f}")
        logger.info(f"  资产数量: {analysis['portfolio_info']['assets_count']}")
        logger.info(f"  现金余额: {analysis['portfolio_info']['cash_balance']:.2f}")

        perf = analysis['performance_metrics']
        logger.info(f"  预期收益率: {perf.get('portfolio_return', 0):.3f}")
        logger.info(f"  投资组合波动率: {perf.get('portfolio_volatility', 0):.3f}")
        logger.info(f"  夏普比率: {perf.get('sharpe_ratio', 0):.3f}")

        logger.info("资产配置:")
        for asset in analysis['asset_allocation']:
            logger.info(f"  {asset['symbol']}: {asset['weight']:.3f} ({asset['market_value']:.2f})")

    # 4. 优化投资组合
    logger.info("\n4. 优化投资组合")
    optimized_weights = manager.optimize_portfolio(portfolio_id, RebalanceStrategy.EQUAL_WEIGHT)
    logger.info("等权重优化结果:")
    for symbol, weight in optimized_weights.items():
        logger.info(f"  {symbol}: {weight:.3f}")

    # 5. 再平衡检查
    logger.info("\n5. 再平衡检查")
    rebalance_plan = manager.rebalance_portfolio(portfolio_id)
    if rebalance_plan:
        logger.info("需要再平衡:")
        logger.info(f"  建议交易数量: {len(rebalance_plan.expected_trades)}")
        logger.info(f"  预估成本: {rebalance_plan.estimated_cost:.2f}")

        for trade in rebalance_plan.expected_trades:
            logger.info(
                f"  {trade['action']} {trade['symbol']} {trade['quantity']:.0f}股 @ {trade['price']:.2f}")
    else:
        logger.info("当前无需再平衡")

    # 6. 更新价格并重新分析
    logger.info("\n6. 更新价格并重新分析")
    price_updates = {
        "AAPL": 160.0,   # 上涨
        "GOOGL": 2600.0,  # 上涨
        "MSFT": 290.0,   # 下跌
        "TSLA": 210.0    # 上涨
    }
    manager.update_asset_prices(portfolio_id, price_updates)
    logger.info("价格更新完成")

    # 重新分析
    updated_analysis = manager.analyze_portfolio(portfolio_id)
    if updated_analysis:
        logger.info("价格更新后分析:")
        logger.info(f"  总价值: {updated_analysis['portfolio_info']['total_value']:.2f}")

        # 检查是否需要再平衡
        rebalance_plan = manager.rebalance_portfolio(portfolio_id)
        if rebalance_plan:
            logger.info("价格更新后需要再平衡")
        else:
            logger.info("价格更新后仍无需再平衡")

    # 7. 获取所有投资组合
    logger.info("\n7. 获取所有投资组合")
    all_portfolios = manager.get_all_portfolios()
    logger.info(f"系统中共有 {len(all_portfolios)} 个投资组合")

    logger.info("\n✅ 投资组合管理功能重建测试完成")


if __name__ == "__main__":
    test_portfolio_management()
