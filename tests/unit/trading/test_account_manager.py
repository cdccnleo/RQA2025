# -*- coding: utf-8 -*-
"""
交易层 - 账户管理器单元测试
测试覆盖率目标: 95%+
按照业务流程驱动架构设计测试账户管理器核心功能
"""

import pytest
from unittest.mock import Mock, patch
from decimal import Decimal

from src.trading.account.account_manager import AccountManager


class TestAccountManager:
    """测试账户管理器"""

    def setup_method(self, method):
        """设置测试环境"""
        self.manager = AccountManager()

    def test_init(self):
        """测试初始化"""
        assert isinstance(self.manager.accounts, dict)
        assert len(self.manager.accounts) == 0

    def test_open_account(self):
        """测试开户"""
        account_id = "test_account_001"
        result = self.manager.open_account(account_id, 1000.0)

        assert result["id"] == account_id
        assert result["status"] == "opened"
        assert account_id in self.manager.accounts
        assert self.manager.accounts[account_id]["balance"] == 1000.0

    def test_open_account_duplicate(self):
        """测试重复开户"""
        account_id = "test_account_002"
        self.manager.open_account(account_id, 1000.0)

        with pytest.raises(ValueError, match="Account .* already exists"):
            self.manager.open_account(account_id, 500.0)

    def test_add_account(self):
        """测试添加账户"""
        account_id = "test_account_003"
        result = self.manager.add_account(account_id, 2000.0)

        assert result["id"] == account_id
        assert result["status"] == "opened"  # add_account调用open_account，返回'opened'
        assert account_id in self.manager.accounts
        assert self.manager.accounts[account_id]["balance"] == 2000.0

    def test_get_account(self):
        """测试获取账户"""
        account_id = "test_account_004"
        self.manager.open_account(account_id, 1500.0)

        account = self.manager.get_account(account_id)
        assert account["balance"] == 1500.0  # get_account返回balance字典

    def test_get_account_nonexistent(self):
        """测试获取不存在的账户"""
        account = self.manager.get_account("nonexistent")
        assert account is None

    def test_deposit(self):
        """测试存款"""
        account_id = "test_account_005"
        self.manager.open_account(account_id, 1000.0)

        self.manager.deposit(account_id, 500.0)
        assert self.manager.accounts[account_id]["balance"] == 1500.0

    def test_withdraw(self):
        """测试取款"""
        account_id = "test_account_006"
        self.manager.open_account(account_id, 1000.0)

        self.manager.withdraw(account_id, 300.0)
        assert self.manager.accounts[account_id]["balance"] == 700.0

    def test_withdraw_insufficient_funds(self):
        """测试取款资金不足"""
        account_id = "test_account_007"
        self.manager.open_account(account_id, 1000.0)

        with pytest.raises(ValueError, match="Insufficient funds"):
            self.manager.withdraw(account_id, 1500.0)

    def test_remove_account(self):
        """测试删除账户"""
        account_id = "test_account_008"
        self.manager.open_account(account_id, 1000.0)

        self.manager.remove_account(account_id)
        assert account_id not in self.manager.accounts

    def test_remove_account_nonexistent(self):
        """测试删除不存在的账户"""
        with pytest.raises(ValueError, match="Account .* does not exist"):
            self.manager.remove_account("nonexistent")

    def test_update_balance_increase(self):
        """测试增加余额"""
        account_id = "test_account_009"
        self.manager.open_account(account_id, 1000.0)

        self.manager.update_balance(account_id, 500.0)
        assert float(self.manager.accounts[account_id]["balance"]) == 1500.0

    def test_update_balance_decrease(self):
        """测试减少余额"""
        account_id = "test_account_010"
        self.manager.open_account(account_id, 1000.0)

        self.manager.update_balance(account_id, -300.0)
        assert float(self.manager.accounts[account_id]["balance"]) == 700.0

    def test_update_balance_insufficient_funds(self):
        """测试余额不足"""
        account_id = "test_account_011"
        self.manager.open_account(account_id, 1000.0)

        with pytest.raises(ValueError, match="Insufficient funds"):
            self.manager.update_balance(account_id, -1500.0)

    def test_update_balance_nonexistent_account(self):
        """测试更新不存在账户的余额"""
        with pytest.raises(ValueError, match="Account .* does not exist"):
            self.manager.update_balance("nonexistent", 100.0)

    def test_transfer_success(self):
        """测试转账成功"""
        account1_id = "test_account_012"
        account2_id = "test_account_013"
        self.manager.open_account(account1_id, 1000.0)
        self.manager.open_account(account2_id, 500.0)

        self.manager.transfer(account1_id, account2_id, 300.0)

        assert float(self.manager.accounts[account1_id]["balance"]) == 700.0
        assert float(self.manager.accounts[account2_id]["balance"]) == 800.0

    def test_transfer_insufficient_funds(self):
        """测试转账资金不足"""
        account1_id = "test_account_014"
        account2_id = "test_account_015"
        self.manager.open_account(account1_id, 1000.0)
        self.manager.open_account(account2_id, 500.0)

        with pytest.raises(ValueError, match="Insufficient funds"):
            self.manager.transfer(account1_id, account2_id, 1500.0)

    def test_transfer_invalid_amount(self):
        """测试转账金额无效"""
        account1_id = "test_account_016"
        account2_id = "test_account_017"
        self.manager.open_account(account1_id, 1000.0)
        self.manager.open_account(account2_id, 500.0)

        with pytest.raises(ValueError, match="Transfer amount must be positive"):
            self.manager.transfer(account1_id, account2_id, -100.0)

        with pytest.raises(ValueError, match="Transfer amount must be positive"):
            self.manager.transfer(account1_id, account2_id, 0.0)

    def test_get_total_balance(self):
        """测试获取总余额"""
        self.manager.open_account("account1", 1000.0)
        self.manager.open_account("account2", 2000.0)
        self.manager.open_account("account3", 500.0)

        total = self.manager.get_total_balance()
        assert float(total) == 3500.0

    def test_get_total_balance_empty(self):
        """测试空账户总余额"""
        total = self.manager.get_total_balance()
        assert float(total) == 0.0

    def test_get_account_count(self):
        """测试获取账户数量"""
        assert self.manager.get_account_count() == 0

        self.manager.open_account("account1", 1000.0)
        assert self.manager.get_account_count() == 1

        self.manager.open_account("account2", 2000.0)
        assert self.manager.get_account_count() == 2

    def test_close_account_success(self):
        """测试关闭账户成功（余额为0）"""
        account_id = "test_account_018"
        self.manager.open_account(account_id, 0.0)

        result = self.manager.close_account(account_id)
        assert result is True
        assert account_id not in self.manager.accounts

    def test_close_account_with_balance(self):
        """测试关闭有余额的账户"""
        account_id = "test_account_019"
        self.manager.open_account(account_id, 1000.0)

        with pytest.raises(ValueError, match="Cannot close account .* with positive balance"):
            self.manager.close_account(account_id)

    def test_close_account_nonexistent(self):
        """测试关闭不存在的账户"""
        with pytest.raises(ValueError, match="Account .* does not exist"):
            self.manager.close_account("nonexistent")

    def test_deposit_invalid_amount(self):
        """测试存款金额无效"""
        account_id = "test_account_020"
        self.manager.open_account(account_id, 1000.0)

        with pytest.raises(ValueError, match="Deposit amount must be positive"):
            self.manager.deposit(account_id, -100.0)

        with pytest.raises(ValueError, match="Deposit amount must be positive"):
            self.manager.deposit(account_id, 0.0)

    def test_deposit_nonexistent_account(self):
        """测试向不存在账户存款"""
        with pytest.raises(ValueError, match="Account .* does not exist"):
            self.manager.deposit("nonexistent", 100.0)

    def test_withdraw_invalid_amount(self):
        """测试取款金额无效"""
        account_id = "test_account_021"
        self.manager.open_account(account_id, 1000.0)

        with pytest.raises(ValueError, match="Withdrawal amount must be positive"):
            self.manager.withdraw(account_id, -100.0)

        with pytest.raises(ValueError, match="Withdrawal amount must be positive"):
            self.manager.withdraw(account_id, 0.0)

    def test_withdraw_nonexistent_account(self):
        """测试从不存在账户取款"""
        with pytest.raises(ValueError, match="Account .* does not exist"):
            self.manager.withdraw("nonexistent", 100.0)

    def test_list_accounts(self):
        """测试列出所有账户"""
        self.manager.open_account("account1", 1000.0)
        self.manager.open_account("account2", 2000.0)

        accounts = self.manager.list_accounts()
        assert len(accounts) == 2
        assert "account1" in accounts
        assert "account2" in accounts
        assert accounts["account1"]["balance"] == Decimal("1000.0")
        assert accounts["account2"]["balance"] == Decimal("2000.0")

    def test_list_accounts_empty(self):
        """测试列出空账户列表"""
        accounts = self.manager.list_accounts()
        assert len(accounts) == 0

    def test_account_created_at_updated_at(self):
        """测试账户创建和更新时间"""
        account_id = "test_account_022"
        result = self.manager.open_account(account_id, 1000.0)

        assert "created_at" in result
        assert "updated_at" in result
        assert result["created_at"] is not None
        assert result["updated_at"] is not None

    def test_update_balance_updates_timestamp(self):
        """测试更新余额时更新时间戳"""
        account_id = "test_account_023"
        self.manager.open_account(account_id, 1000.0)
        original_updated_at = self.manager.accounts[account_id]["updated_at"]

        import time
        time.sleep(0.01)  # 确保时间戳不同

        self.manager.update_balance(account_id, 100.0)
        new_updated_at = self.manager.accounts[account_id]["updated_at"]

        assert new_updated_at > original_updated_at
