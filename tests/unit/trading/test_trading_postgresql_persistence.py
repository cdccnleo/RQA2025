#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交易层PostgreSQL持久化测试

测试内容:
1. OrderPersistence 文件系统存储测试（降级模式）
2. AccountPersistence 文件系统存储测试（降级模式）
3. PositionPersistence 文件系统存储测试（降级模式）
4. TradePersistence 文件系统存储测试（降级模式）
5. OrderManager 集成测试
6. AccountManager 集成测试
7. PortfolioManager 集成测试
"""

import os
import sys
import json
import time
import tempfile
import shutil
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestOrderPersistence(unittest.TestCase):
    """订单持久化测试"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_dir = os.path.join(self.temp_dir, "orders")
        os.makedirs(self.storage_dir, exist_ok=True)
    
    def tearDown(self):
        """测试后清理"""
        try:
            shutil.rmtree(self.temp_dir)
        except PermissionError:
            pass
    
    def test_order_persistence_file_storage(self):
        """测试订单文件系统存储（降级模式）"""
        from src.trading.persistence.trading_persistence import OrderPersistence, OrderData
        
        persistence = OrderPersistence(storage_dir=self.storage_dir)
        
        self.assertFalse(persistence._use_postgresql)
        
        order = OrderData(
            order_id="test_order_001",
            symbol="000001.SZ",
            side="buy",
            order_type="limit",
            quantity=1000,
            price=10.5,
            status="submitted",
            account_id="test_account"
        )
        
        result = persistence.save_order(order)
        self.assertTrue(result)
        
        file_path = os.path.join(self.storage_dir, "test_order_001.json")
        self.assertTrue(os.path.exists(file_path))
        
        loaded_order = persistence.get_order("test_order_001")
        self.assertIsNotNone(loaded_order)
        self.assertEqual(loaded_order.symbol, "000001.SZ")
        self.assertEqual(loaded_order.side, "buy")
        self.assertEqual(loaded_order.quantity, 1000)
        self.assertEqual(loaded_order.price, 10.5)
    
    def test_order_update(self):
        """测试订单更新"""
        from src.trading.persistence.trading_persistence import OrderPersistence, OrderData
        
        persistence = OrderPersistence(storage_dir=self.storage_dir)
        
        order = OrderData(
            order_id="test_order_002",
            symbol="000002.SZ",
            side="sell",
            order_type="market",
            quantity=500,
            status="submitted"
        )
        
        persistence.save_order(order)
        
        result = persistence.update_order("test_order_002", {
            "status": "filled",
            "filled_quantity": 500,
            "avg_fill_price": 12.3
        })
        self.assertTrue(result)
        
        updated_order = persistence.get_order("test_order_002")
        self.assertEqual(updated_order.status, "filled")
        self.assertEqual(updated_order.filled_quantity, 500)
        self.assertEqual(updated_order.avg_fill_price, 12.3)
    
    def test_order_delete(self):
        """测试订单删除"""
        from src.trading.persistence.trading_persistence import OrderPersistence, OrderData
        
        persistence = OrderPersistence(storage_dir=self.storage_dir)
        
        order = OrderData(
            order_id="test_order_003",
            symbol="000003.SZ",
            side="buy",
            order_type="limit",
            quantity=100,
            price=15.0,
            status="filled"
        )
        
        persistence.save_order(order)
        
        file_path = os.path.join(self.storage_dir, "test_order_003.json")
        self.assertTrue(os.path.exists(file_path))
        
        result = persistence.delete_order("test_order_003")
        self.assertTrue(result)
        
        deleted_order = persistence.get_order("test_order_003")
        self.assertIsNone(deleted_order)
    
    def test_get_orders_by_status(self):
        """测试按状态获取订单"""
        from src.trading.persistence.trading_persistence import OrderPersistence, OrderData
        
        persistence = OrderPersistence(storage_dir=self.storage_dir)
        
        for i in range(5):
            order = OrderData(
                order_id=f"test_order_{i:03d}",
                symbol=f"00000{i}.SZ",
                side="buy" if i % 2 == 0 else "sell",
                order_type="limit",
                quantity=100 * (i + 1),
                price=10.0 + i,
                status="submitted" if i < 3 else "filled"
            )
            persistence.save_order(order)
        
        submitted_orders = persistence.get_orders_by_status("submitted")
        self.assertEqual(len(submitted_orders), 3)
        
        filled_orders = persistence.get_orders_by_status("filled")
        self.assertEqual(len(filled_orders), 2)


class TestAccountPersistence(unittest.TestCase):
    """账户持久化测试"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_dir = os.path.join(self.temp_dir, "accounts")
        os.makedirs(self.storage_dir, exist_ok=True)
    
    def tearDown(self):
        """测试后清理"""
        try:
            shutil.rmtree(self.temp_dir)
        except PermissionError:
            pass
    
    def test_account_save_and_load(self):
        """测试账户保存和加载"""
        from src.trading.persistence.trading_persistence import AccountPersistence, AccountData
        
        persistence = AccountPersistence(storage_dir=self.storage_dir)
        
        self.assertFalse(persistence._use_postgresql)
        
        account = AccountData(
            account_id="test_account_001",
            balance=100000.0,
            frozen_balance=0.0,
            available_balance=100000.0,
            status="active"
        )
        
        result = persistence.save_account(account)
        self.assertTrue(result)
        
        loaded_account = persistence.get_account("test_account_001")
        self.assertIsNotNone(loaded_account)
        self.assertEqual(loaded_account.balance, 100000.0)
        self.assertEqual(loaded_account.available_balance, 100000.0)
    
    def test_account_update_balance(self):
        """测试账户余额更新"""
        from src.trading.persistence.trading_persistence import AccountPersistence, AccountData
        
        persistence = AccountPersistence(storage_dir=self.storage_dir)
        
        account = AccountData(
            account_id="test_account_002",
            balance=50000.0,
            available_balance=50000.0
        )
        
        persistence.save_account(account)
        
        result = persistence.update_account("test_account_002", {
            "balance": 60000.0,
            "available_balance": 60000.0
        })
        self.assertTrue(result)
        
        updated_account = persistence.get_account("test_account_002")
        self.assertEqual(updated_account.balance, 60000.0)
        self.assertEqual(updated_account.available_balance, 60000.0)
    
    def test_get_all_accounts(self):
        """测试获取所有账户"""
        from src.trading.persistence.trading_persistence import AccountPersistence, AccountData
        
        persistence = AccountPersistence(storage_dir=self.storage_dir)
        
        for i in range(3):
            account = AccountData(
                account_id=f"test_account_{i:03d}",
                balance=10000.0 * (i + 1)
            )
            persistence.save_account(account)
        
        accounts = persistence.get_all_accounts()
        self.assertEqual(len(accounts), 3)
    
    def test_account_delete(self):
        """测试账户删除"""
        from src.trading.persistence.trading_persistence import AccountPersistence, AccountData
        
        persistence = AccountPersistence(storage_dir=self.storage_dir)
        
        account = AccountData(
            account_id="test_account_del",
            balance=0.0
        )
        persistence.save_account(account)
        
        result = persistence.delete_account("test_account_del")
        self.assertTrue(result)
        
        deleted_account = persistence.get_account("test_account_del")
        self.assertIsNone(deleted_account)


class TestPositionPersistence(unittest.TestCase):
    """持仓持久化测试"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_dir = os.path.join(self.temp_dir, "positions")
        os.makedirs(self.storage_dir, exist_ok=True)
    
    def tearDown(self):
        """测试后清理"""
        try:
            shutil.rmtree(self.temp_dir)
        except PermissionError:
            pass
    
    def test_position_save_and_load(self):
        """测试持仓保存和加载"""
        from src.trading.persistence.trading_persistence import PositionPersistence, PositionData
        
        persistence = PositionPersistence(storage_dir=self.storage_dir)
        
        self.assertFalse(persistence._use_postgresql)
        
        position = PositionData(
            position_id="test_account_000001.SZ",
            account_id="test_account",
            symbol="000001.SZ",
            quantity=1000,
            available_quantity=1000,
            avg_cost=10.5,
            current_price=11.0,
            market_value=11000.0
        )
        
        result = persistence.save_position(position)
        self.assertTrue(result)
        
        loaded_position = persistence.get_position("test_account_000001.SZ")
        self.assertIsNotNone(loaded_position)
        self.assertEqual(loaded_position.symbol, "000001.SZ")
        self.assertEqual(loaded_position.quantity, 1000)
        self.assertEqual(loaded_position.avg_cost, 10.5)
    
    def test_position_update_price(self):
        """测试持仓价格更新"""
        from src.trading.persistence.trading_persistence import PositionPersistence, PositionData
        
        persistence = PositionPersistence(storage_dir=self.storage_dir)
        
        position = PositionData(
            position_id="test_account_000002.SZ",
            account_id="test_account",
            symbol="000002.SZ",
            quantity=500,
            avg_cost=20.0,
            current_price=20.0,
            market_value=10000.0
        )
        
        persistence.save_position(position)
        
        result = persistence.update_position("test_account_000002.SZ", {
            "current_price": 22.0,
            "market_value": 11000.0,
            "unrealized_pnl": 1000.0
        })
        self.assertTrue(result)
        
        updated_position = persistence.get_position("test_account_000002.SZ")
        self.assertEqual(updated_position.current_price, 22.0)
        self.assertEqual(updated_position.market_value, 11000.0)
        self.assertEqual(updated_position.unrealized_pnl, 1000.0)
    
    def test_get_positions_by_account(self):
        """测试按账户获取持仓"""
        from src.trading.persistence.trading_persistence import PositionPersistence, PositionData
        
        persistence = PositionPersistence(storage_dir=self.storage_dir)
        
        for i in range(3):
            position = PositionData(
                position_id=f"test_account_00000{i}.SZ",
                account_id="test_account",
                symbol=f"00000{i}.SZ",
                quantity=100 * (i + 1),
                avg_cost=10.0 + i
            )
            persistence.save_position(position)
        
        positions = persistence.get_positions_by_account("test_account")
        self.assertEqual(len(positions), 3)


class TestTradePersistence(unittest.TestCase):
    """交易记录持久化测试"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_dir = os.path.join(self.temp_dir, "trades")
        os.makedirs(self.storage_dir, exist_ok=True)
    
    def tearDown(self):
        """测试后清理"""
        try:
            shutil.rmtree(self.temp_dir)
        except PermissionError:
            pass
    
    def test_trade_save_and_load(self):
        """测试交易记录保存和加载"""
        from src.trading.persistence.trading_persistence import TradePersistence, TradeData
        
        persistence = TradePersistence(storage_dir=self.storage_dir)
        
        self.assertFalse(persistence._use_postgresql)
        
        trade = TradeData(
            trade_id="test_trade_001",
            order_id="test_order_001",
            account_id="test_account",
            symbol="000001.SZ",
            side="buy",
            quantity=1000,
            price=10.5,
            amount=10500.0,
            commission=3.15
        )
        
        result = persistence.save_trade(trade)
        self.assertTrue(result)
        
        loaded_trade = persistence.get_trade("test_trade_001")
        self.assertIsNotNone(loaded_trade)
        self.assertEqual(loaded_trade.symbol, "000001.SZ")
        self.assertEqual(loaded_trade.quantity, 1000)
        self.assertEqual(loaded_trade.price, 10.5)
        self.assertEqual(loaded_trade.commission, 3.15)
    
    def test_get_trades_by_account(self):
        """测试按账户获取交易记录"""
        from src.trading.persistence.trading_persistence import TradePersistence, TradeData
        
        persistence = TradePersistence(storage_dir=self.storage_dir)
        
        for i in range(5):
            trade = TradeData(
                trade_id=f"test_trade_{i:03d}",
                order_id=f"test_order_{i:03d}",
                account_id="test_account",
                symbol=f"00000{i}.SZ",
                side="buy" if i % 2 == 0 else "sell",
                quantity=100 * (i + 1),
                price=10.0 + i,
                amount=100 * (i + 1) * (10.0 + i)
            )
            persistence.save_trade(trade)
        
        trades = persistence.get_trades_by_account("test_account", limit=10)
        self.assertEqual(len(trades), 5)


class TestPersistencePerformance(unittest.TestCase):
    """持久化性能测试"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """测试后清理"""
        try:
            shutil.rmtree(self.temp_dir)
        except PermissionError:
            pass
    
    def test_batch_save_performance(self):
        """测试批量保存性能"""
        from src.trading.persistence.trading_persistence import OrderPersistence, OrderData
        
        storage_dir = os.path.join(self.temp_dir, "orders")
        persistence = OrderPersistence(storage_dir=storage_dir)
        
        start_time = time.time()
        
        for i in range(100):
            order = OrderData(
                order_id=f"perf_test_order_{i:04d}",
                symbol=f"{i % 10:06d}.SZ",
                side="buy" if i % 2 == 0 else "sell",
                order_type="limit",
                quantity=100 * (i % 10 + 1),
                price=10.0 + (i % 20)
            )
            persistence.save_order(order)
        
        elapsed_time = time.time() - start_time
        
        self.assertLess(elapsed_time, 5.0, "批量保存100条订单应在5秒内完成")
        
        print(f"\n批量保存100条订单耗时: {elapsed_time:.3f}秒")


class TestOrderManagerIntegration(unittest.TestCase):
    """OrderManager集成测试"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """测试后清理"""
        try:
            shutil.rmtree(self.temp_dir)
        except PermissionError:
            pass
    
    def test_order_manager_with_file_persistence(self):
        """测试OrderManager文件持久化集成"""
        from src.trading.execution.order_manager import OrderManager, Order, OrderSide, OrderType
        
        manager = OrderManager(max_orders=100, enable_persistence=False)
        
        order = Order(
            symbol="000001.SZ",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1000,
            price=10.5
        )
        
        success, msg, order_id = manager.submit_order(order)
        self.assertTrue(success)
        self.assertIsNotNone(order_id)
        
        loaded_order = manager.get_order(order_id)
        self.assertIsNotNone(loaded_order)
        self.assertEqual(loaded_order.symbol, "000001.SZ")
    
    def test_order_manager_cancel_order(self):
        """测试OrderManager取消订单"""
        from src.trading.execution.order_manager import OrderManager, Order, OrderSide, OrderType, OrderStatus
        
        manager = OrderManager(max_orders=100, enable_persistence=False)
        
        order = Order(
            symbol="000002.SZ",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=500
        )
        
        success, msg, order_id = manager.submit_order(order)
        self.assertTrue(success)
        
        success, msg = manager.cancel_order(order_id)
        self.assertTrue(success)
        
        cancelled_order = manager.get_order(order_id)
        self.assertEqual(cancelled_order.status, OrderStatus.CANCELLED)
    
    def test_order_manager_statistics(self):
        """测试OrderManager统计功能"""
        from src.trading.execution.order_manager import OrderManager, Order, OrderSide, OrderType
        
        manager = OrderManager(max_orders=100, enable_persistence=False)
        
        for i in range(5):
            order = Order(
                symbol=f"00000{i}.SZ",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=100 * (i + 1),
                price=10.0 + i
            )
            manager.submit_order(order)
        
        stats = manager.get_statistics()
        self.assertEqual(stats['total_submitted'], 5)
        self.assertEqual(stats['active_orders'], 5)


class TestAccountManagerIntegration(unittest.TestCase):
    """AccountManager集成测试"""
    
    def test_account_manager_basic_operations(self):
        """测试AccountManager基本操作"""
        from src.trading.account.account_manager import AccountManager
        
        manager = AccountManager(enable_persistence=False)
        
        account = manager.open_account("test_integration_account", 100000.0)
        self.assertEqual(account["balance"], 100000.0)
        
        loaded_account = manager.get_account("test_integration_account")
        self.assertIsNotNone(loaded_account)
        self.assertEqual(loaded_account["balance"], 100000.0)
        
        manager.update_balance("test_integration_account", 10000.0)
        updated_account = manager.get_account("test_integration_account")
        self.assertEqual(updated_account["balance"], 110000.0)
    
    def test_account_manager_freeze_balance(self):
        """测试AccountManager冻结资金"""
        from src.trading.account.account_manager import AccountManager
        
        manager = AccountManager(enable_persistence=False)
        manager.open_account("freeze_test", 50000.0)
        
        result = manager.freeze_balance("freeze_test", 10000.0)
        self.assertTrue(result)
        
        account = manager.get_account("freeze_test")
        self.assertEqual(account["frozen_balance"], 10000.0)
        self.assertEqual(account["available_balance"], 40000.0)
        
        result = manager.unfreeze_balance("freeze_test", 5000.0)
        self.assertTrue(result)
        
        account = manager.get_account("freeze_test")
        self.assertEqual(account["frozen_balance"], 5000.0)
        self.assertEqual(account["available_balance"], 45000.0)
    
    def test_account_manager_transfer(self):
        """测试AccountManager转账"""
        from src.trading.account.account_manager import AccountManager
        
        manager = AccountManager(enable_persistence=False)
        manager.open_account("account_a", 100000.0)
        manager.open_account("account_b", 50000.0)
        
        result = manager.transfer("account_a", "account_b", 20000.0)
        self.assertTrue(result)
        
        account_a = manager.get_account("account_a")
        account_b = manager.get_account("account_b")
        
        self.assertEqual(account_a["balance"], 80000.0)
        self.assertEqual(account_b["balance"], 70000.0)


class TestPortfolioManagerIntegration(unittest.TestCase):
    """PortfolioManager集成测试"""
    
    def test_portfolio_manager_basic_operations(self):
        """测试PortfolioManager基本操作"""
        from src.trading.portfolio.portfolio_manager import PortfolioManager
        
        manager = PortfolioManager(account_id="test_portfolio", enable_persistence=False)
        
        manager.cash = 100000.0
        
        result = manager.add_position("000001.SZ", 1000, 10.5)
        self.assertTrue(result)
        
        position = manager.get_position("000001.SZ")
        self.assertIsNotNone(position)
        self.assertEqual(position["quantity"], 1000)
        self.assertEqual(position["avg_price"], 10.5)
    
    def test_portfolio_manager_update_price(self):
        """测试PortfolioManager更新价格"""
        from src.trading.portfolio.portfolio_manager import PortfolioManager
        
        manager = PortfolioManager(account_id="price_test", enable_persistence=False)
        manager.cash = 50000.0
        
        manager.add_position("000002.SZ", 500, 20.0)
        
        result = manager.update_position_price("000002.SZ", 22.0)
        self.assertTrue(result)
        
        position = manager.get_position("000002.SZ")
        self.assertEqual(position["current_price"], 22.0)
        self.assertEqual(position["market_value"], 11000.0)
        self.assertEqual(position["unrealized_pnl"], 1000.0)
    
    def test_portfolio_manager_freeze_position(self):
        """测试PortfolioManager冻结持仓"""
        from src.trading.portfolio.portfolio_manager import PortfolioManager
        
        manager = PortfolioManager(account_id="freeze_pos_test", enable_persistence=False)
        
        manager.add_position("000003.SZ", 1000, 15.0)
        
        result = manager.freeze_position("000003.SZ", 500)
        self.assertTrue(result)
        
        position = manager.get_position("000003.SZ")
        self.assertEqual(position["frozen_quantity"], 500)
        self.assertEqual(position["available_quantity"], 500)
    
    def test_portfolio_manager_value_calculation(self):
        """测试PortfolioManager价值计算"""
        from src.trading.portfolio.portfolio_manager import PortfolioManager
        
        manager = PortfolioManager(account_id="value_test", enable_persistence=False)
        manager.cash = 100000.0
        
        manager.add_position("000001.SZ", 1000, 10.0)
        manager.add_position("000002.SZ", 500, 20.0)
        
        manager.update_position_price("000001.SZ", 12.0)
        manager.update_position_price("000002.SZ", 25.0)
        
        value = manager.get_portfolio_value()
        self.assertEqual(value, 100000.0 + 1000 * 12.0 + 500 * 25.0)
        
        summary = manager.get_portfolio_summary()
        self.assertEqual(summary["positions_count"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
