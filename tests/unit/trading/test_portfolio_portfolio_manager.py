# -*- coding: utf-8 -*-
"""
交易层 - 投资组合管理器单元测试
测试覆盖率目标: 95%+
按照业务流程驱动架构设计测试投资组合管理器核心功能
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.trading.portfolio.portfolio_portfolio_manager import (
    PortfolioManager, BasePortfolioOptimizer, EqualWeightOptimizer,
    MeanVarianceOptimizer, PortfolioConstraints, StrategyPerformance
)


class MockOptimizer(BasePortfolioOptimizer):
    """模拟优化器，用于测试"""

    def optimize(self, returns, constraints):
        """模拟优化"""
        n_assets = len(returns.columns)
        # 返回等权重
        weights = np.ones(n_assets) / n_assets
        return weights


class TestPortfolioManager:
    """测试投资组合管理器"""

    def setup_method(self, method):
        """设置测试环境"""
        self.optimizer = MockOptimizer()
        self.manager = PortfolioManager(
            optimizer=self.optimizer,
            initial_capital=1000000.0,
            rebalance_threshold=0.05
        )

    def test_init(self):
        """测试初始化"""
        assert self.manager.optimizer == self.optimizer
        assert self.manager.cash_balance == 1000000.0
        assert self.manager.initial_capital == 1000000.0
        assert self.manager.rebalance_threshold == 0.05
        assert isinstance(self.manager.positions, dict)
        assert isinstance(self.manager.current_weights, dict)

    def test_add_position(self):
        """测试添加持仓"""
        symbol = "000001.SZ"
        quantity = 1000
        price = 10.0

        result = self.manager.add_position(symbol, quantity, price)

        assert result is True
        assert symbol in self.manager.positions
        assert self.manager.positions[symbol]["quantity"] == quantity
        assert self.manager.positions[symbol]["avg_price"] == price

    def test_remove_position(self):
        """测试移除持仓"""
        symbol = "000001.SZ"
        self.manager.add_position(symbol, 1000, 10.0)

        result = self.manager.remove_position(symbol)

        assert result is True
        assert symbol not in self.manager.positions

    def test_remove_position_nonexistent(self):
        """测试移除不存在的持仓"""
        result = self.manager.remove_position("NONEXISTENT")
        assert result is False

    def test_update_position_price(self):
        """测试更新持仓价格"""
        symbol = "000001.SZ"
        self.manager.add_position(symbol, 1000, 10.0)

        result = self.manager.update_position_price(symbol, 12.0)

        assert result is True
        assert self.manager.positions[symbol]["current_price"] == 12.0

    def test_update_position_price_nonexistent(self):
        """测试更新不存在持仓的价格"""
        result = self.manager.update_position_price("NONEXISTENT", 12.0)
        assert result is False

    def test_get_portfolio_value(self):
        """测试获取组合价值"""
        # 添加一些持仓
        self.manager.add_position("000001.SZ", 1000, 10.0)
        self.manager.add_position("000002.SZ", 500, 20.0)

        # 设置当前价格
        self.manager.update_position_price("000001.SZ", 11.0)
        self.manager.update_position_price("000002.SZ", 22.0)

        total_value = self.manager.get_portfolio_value()

        # 计算预期价值：(1000*11.0) + (500*22.0) + 现金余额
        expected_value = 1000 * 11.0 + 500 * 22.0 + self.manager.cash_balance
        assert total_value == expected_value

    def test_get_portfolio_value_no_positions(self):
        """测试空组合的价值"""
        total_value = self.manager.get_portfolio_value()
        assert total_value == self.manager.cash_balance

    def test_calculate_returns(self):
        """测试计算收益率"""
        # 创建模拟的价格数据
        dates = pd.date_range('2023-01-01', periods=5)
        prices = pd.DataFrame({
            '000001.SZ': [10.0, 10.5, 11.0, 10.8, 11.2],
            '000002.SZ': [20.0, 20.2, 20.8, 20.5, 21.0]
        }, index=dates)

        returns = self.manager.calculate_returns(prices)

        assert isinstance(returns, pd.DataFrame)
        assert len(returns) == 4  # 5个价格点产生4个收益率
        assert '000001.SZ' in returns.columns
        assert '000002.SZ' in returns.columns

    def test_optimize_portfolio(self):
        """测试组合优化"""
        # 创建模拟的收益率数据
        dates = pd.date_range('2023-01-01', periods=30)
        returns_data = pd.DataFrame({
            '000001.SZ': np.random.normal(0.001, 0.02, 30),
            '000002.SZ': np.random.normal(0.001, 0.02, 30),
            '000003.SZ': np.random.normal(0.001, 0.02, 30)
        }, index=dates)

        constraints = PortfolioConstraints(
            min_weight=0.0,
            max_weight=0.5,
            target_return=0.02
        )

        weights = self.manager.optimize_portfolio(returns_data, constraints)

        assert isinstance(weights, np.ndarray)
        assert len(weights) == 3
        assert abs(np.sum(weights) - 1.0) < 0.01  # 权重之和应为1

    def test_needs_rebalance(self):
        """测试是否需要再平衡"""
        # 设置初始权重
        self.manager.current_weights = {
            '000001.SZ': 0.5,
            '000002.SZ': 0.5
        }

        # 设置当前持仓
        self.manager.add_position('000001.SZ', 50000, 10.0)  # 50万市值
        self.manager.add_position('000002.SZ', 50000, 10.0)  # 50万市值

        # 当前权重应该是 50%:50%，不需要再平衡
        needs_rebalance = self.manager.needs_rebalance()
        assert needs_rebalance is False

        # 改变价格，使权重偏差超过阈值
        self.manager.update_position_price('000001.SZ', 15.0)  # 市值变为75万
        self.manager.update_position_price('000002.SZ', 5.0)   # 市值变为12.5万

        # 现在权重变为 75/87.5 ≈ 85.7% vs 12.5/87.5 ≈ 14.3%
        # 偏差超过5%的阈值，需要再平衡
        needs_rebalance = self.manager.needs_rebalance()
        assert needs_rebalance is True


class TestEqualWeightOptimizer:
    """测试等权重优化器"""

    def test_optimize(self):
        """测试优化"""
        optimizer = EqualWeightOptimizer()

        # 创建模拟数据
        returns = pd.DataFrame({
            'A': [0.01, 0.02, 0.03],
            'B': [0.02, 0.01, 0.04],
            'C': [0.03, 0.02, 0.01]
        })

        constraints = PortfolioConstraints()

        # 将DataFrame转换为字典格式
        performances = {col: returns[col] for col in returns.columns}

        weights = optimizer.optimize(performances, constraints)

        assert len(weights) == 3
        assert abs(weights['A'] - 1/3) < 0.001
        assert abs(weights['B'] - 1/3) < 0.001
        assert abs(weights['C'] - 1/3) < 0.001
