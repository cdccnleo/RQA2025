# -*- coding: utf-8 -*-
"""
交易层 - 账户管理器
负责账户的创建、管理和状态维护
"""

import logging
from decimal import Decimal
from typing import Dict, Any, Optional
from datetime import datetime


class AccountManager:
    """账户管理器"""

    def __init__(self):
        """初始化账户管理器"""
        self.accounts: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)

    def open_account(self, account_id: str, initial_balance: float = 0.0) -> Dict[str, Any]:
        """开户

        Args:
            account_id: 账户ID
            initial_balance: 初始余额

        Returns:
            账户信息字典

        Raises:
            ValueError: 账户已存在
        """
        if account_id in self.accounts:
            raise ValueError(f"Account {account_id} already exists")

        account = {
            "id": account_id,
            "balance": Decimal(str(initial_balance)),
            "status": "opened",
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }

        self.accounts[account_id] = account
        self.logger.info(f"Account {account_id} opened with balance {initial_balance}")
        return account

    def add_account(self, account_id: str, initial_balance: float = 0.0) -> Dict[str, Any]:
        """添加账户（与open_account相同）

        Args:
            account_id: 账户ID
            initial_balance: 初始余额

        Returns:
            账户信息字典
        """
        return self.open_account(account_id, initial_balance)

    def close_account(self, account_id: str) -> bool:
        """关闭账户

        Args:
            account_id: 账户ID

        Returns:
            是否成功关闭

        Raises:
            ValueError: 账户不存在
        """
        if account_id not in self.accounts:
            raise ValueError(f"Account {account_id} does not exist")

        if self.accounts[account_id]["balance"] > 0:
            raise ValueError(f"Cannot close account {account_id} with positive balance")

        del self.accounts[account_id]
        self.logger.info(f"Account {account_id} closed")
        return True

    def get_account(self, account_id: str) -> Optional[Dict[str, Any]]:
        """获取账户信息

        Args:
            account_id: 账户ID

        Returns:
            账户信息字典或None
        """
        return self.accounts.get(account_id)

    def update_balance(self, account_id: str, amount: float) -> bool:
        """更新账户余额

        Args:
            account_id: 账户ID
            amount: 变动金额（正数增加，负数减少）

        Returns:
            是否成功更新

        Raises:
            ValueError: 账户不存在或余额不足
        """
        if account_id not in self.accounts:
            raise ValueError(f"Account {account_id} does not exist")

        account = self.accounts[account_id]
        new_balance = account["balance"] + Decimal(str(amount))

        if new_balance < 0:
            raise ValueError("Insufficient funds")

        account["balance"] = new_balance
        account["updated_at"] = datetime.now()

        self.logger.info(f"Account {account_id} balance updated by {amount}, new balance: {new_balance}")
        return True

    def transfer(self, from_account: str, to_account: str, amount: float) -> bool:
        """账户间转账

        Args:
            from_account: 转出账户ID
            to_account: 转入账户ID
            amount: 转账金额

        Returns:
            是否成功转账

        Raises:
            ValueError: 账户不存在或余额不足
        """
        if amount <= 0:
            raise ValueError("Transfer amount must be positive")

        self.update_balance(from_account, -amount)
        self.update_balance(to_account, amount)

        self.logger.info(f"Transferred {amount} from {from_account} to {to_account}")
        return True

    def get_total_balance(self) -> Decimal:
        """获取所有账户总余额

        Returns:
            总余额
        """
        return sum(account["balance"] for account in self.accounts.values())

    def get_account_count(self) -> int:
        """获取账户数量

        Returns:
            账户数量
        """
        return len(self.accounts)

    def deposit(self, account_id: str, amount: float) -> bool:
        """存款

        Args:
            account_id: 账户ID
            amount: 存款金额

        Returns:
            是否成功存款

        Raises:
            ValueError: 账户不存在或金额无效
        """
        if account_id not in self.accounts:
            raise ValueError(f"Account {account_id} does not exist")

        if amount <= 0:
            raise ValueError("Deposit amount must be positive")

        self.update_balance(account_id, amount)
        return True

    def withdraw(self, account_id: str, amount: float) -> bool:
        """取款

        Args:
            account_id: 账户ID
            amount: 取款金额

        Returns:
            是否成功取款

        Raises:
            ValueError: 账户不存在或余额不足
        """
        if account_id not in self.accounts:
            raise ValueError(f"Account {account_id} does not exist")

        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")

        self.update_balance(account_id, -amount)
        return True

    def remove_account(self, account_id: str) -> bool:
        """删除账户（允许删除有余额的账户）

        Args:
            account_id: 账户ID

        Returns:
            是否成功删除

        Raises:
            ValueError: 账户不存在
        """
        if account_id not in self.accounts:
            raise ValueError(f"Account {account_id} does not exist")

        del self.accounts[account_id]
        self.logger.info(f"Account {account_id} removed")
        return True

    def list_accounts(self) -> Dict[str, Dict[str, Any]]:
        """列出所有账户

        Returns:
            所有账户信息
        """
        return self.accounts.copy()
