# -*- coding: utf-8 -*-
"""
交易层 - 账户管理器
负责账户的创建、管理和状态维护
支持PostgreSQL优先存储策略，连接失败时自动降级到内存存储
"""

import logging
import threading
from decimal import Decimal
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class AccountInfo:
    """账户信息数据类"""
    account_id: str
    balance: float = 0.0
    frozen_balance: float = 0.0
    available_balance: float = 0.0
    status: str = "active"
    currency: str = "CNY"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AccountManager:
    """
    账户管理器 - PostgreSQL持久化版本
    
    支持PostgreSQL优先存储策略，连接失败时自动降级到内存存储。
    所有账户操作会同步持久化到数据库，确保数据安全。
    """

    def __init__(self, enable_persistence: bool = True):
        """
        初始化账户管理器
        
        Args:
            enable_persistence: 是否启用持久化（默认True）
        """
        self._accounts: Dict[str, AccountInfo] = {}
        self._lock = threading.RLock()
        self._enable_persistence = enable_persistence
        self._persistence = None
        self._AccountData = None
        
        if enable_persistence:
            self._init_persistence()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"账户管理器初始化完成，持久化: {enable_persistence}")
    
    def _init_persistence(self):
        """
        初始化持久化层
        
        尝试初始化PostgreSQL持久化，失败则禁用持久化功能
        """
        try:
            from ..persistence.trading_persistence import AccountPersistence, AccountData
            self._persistence = AccountPersistence()
            self._AccountData = AccountData
            self.logger.info("账户持久化层初始化成功")
        except Exception as e:
            self.logger.warning(f"账户持久化层初始化失败: {e}，将使用纯内存模式")
            self._enable_persistence = False
            self._persistence = None
    
    def _save_account_to_persistence(self, account: AccountInfo) -> bool:
        """
        保存账户到持久化层
        
        Args:
            account: 账户信息
        
        Returns:
            是否保存成功
        """
        if not self._enable_persistence or not self._persistence:
            return True
        
        try:
            account_data = self._AccountData(
                account_id=account.account_id,
                balance=account.balance,
                frozen_balance=account.frozen_balance,
                available_balance=account.available_balance,
                status=account.status,
                currency=account.currency,
                created_at=account.created_at,
                updated_at=account.updated_at,
                metadata=account.metadata
            )
            return self._persistence.save_account(account_data)
        except Exception as e:
            self.logger.error(f"保存账户到持久化层失败: {e}")
            return False
    
    def _update_account_in_persistence(self, account_id: str, updates: Dict[str, Any]) -> bool:
        """
        更新持久化层中的账户
        
        Args:
            account_id: 账户ID
            updates: 更新字段字典
        
        Returns:
            是否更新成功
        """
        if not self._enable_persistence or not self._persistence:
            return True
        
        try:
            return self._persistence.update_account(account_id, updates)
        except Exception as e:
            self.logger.error(f"更新持久化层账户失败: {e}")
            return False
    
    def _load_account_from_persistence(self, account_id: str) -> Optional[AccountInfo]:
        """
        从持久化层加载账户
        
        Args:
            account_id: 账户ID
        
        Returns:
            账户信息，不存在返回None
        """
        if not self._enable_persistence or not self._persistence:
            return None
        
        try:
            account_data = self._persistence.get_account(account_id)
            if account_data:
                return AccountInfo(
                    account_id=account_data.account_id,
                    balance=account_data.balance,
                    frozen_balance=account_data.frozen_balance,
                    available_balance=account_data.available_balance,
                    status=account_data.status,
                    currency=account_data.currency,
                    created_at=account_data.created_at,
                    updated_at=account_data.updated_at,
                    metadata=account_data.metadata
                )
        except Exception as e:
            self.logger.error(f"从持久化层加载账户失败: {e}")
        return None
    
    def restore_from_persistence(self) -> int:
        """
        从持久化层恢复所有活跃账户
        
        Returns:
            恢复的账户数量
        """
        if not self._enable_persistence or not self._persistence:
            return 0
        
        restored_count = 0
        try:
            accounts = self._persistence.get_all_accounts()
            for account_data in accounts:
                if account_data.account_id not in self._accounts:
                    account = AccountInfo(
                        account_id=account_data.account_id,
                        balance=account_data.balance,
                        frozen_balance=account_data.frozen_balance,
                        available_balance=account_data.available_balance,
                        status=account_data.status,
                        currency=account_data.currency,
                        created_at=account_data.created_at,
                        updated_at=account_data.updated_at,
                        metadata=account_data.metadata
                    )
                    self._accounts[account_data.account_id] = account
                    restored_count += 1
            
            self.logger.info(f"从持久化层恢复了 {restored_count} 个账户")
        except Exception as e:
            self.logger.error(f"从持久化层恢复账户失败: {e}")
        
        return restored_count

    def open_account(self, account_id: str, initial_balance: float = 0.0) -> Dict[str, Any]:
        """
        开户
        
        Args:
            account_id: 账户ID
            initial_balance: 初始余额
        
        Returns:
            账户信息字典
        
        Raises:
            ValueError: 账户已存在
        """
        with self._lock:
            if account_id in self._accounts:
                raise ValueError(f"Account {account_id} already exists")
            
            now = datetime.now()
            account = AccountInfo(
                account_id=account_id,
                balance=initial_balance,
                frozen_balance=0.0,
                available_balance=initial_balance,
                status="active",
                created_at=now,
                updated_at=now
            )
            
            self._accounts[account_id] = account
            
            # 持久化账户
            self._save_account_to_persistence(account)
            
            self.logger.info(f"Account {account_id} opened with balance {initial_balance}")
            return self._account_to_dict(account)

    def add_account(self, account_id: str, initial_balance: float = 0.0) -> Dict[str, Any]:
        """
        添加账户（与open_account相同）
        
        Args:
            account_id: 账户ID
            initial_balance: 初始余额
        
        Returns:
            账户信息字典
        """
        return self.open_account(account_id, initial_balance)

    def close_account(self, account_id: str) -> bool:
        """
        关闭账户
        
        Args:
            account_id: 账户ID
        
        Returns:
            是否成功关闭
        
        Raises:
            ValueError: 账户不存在
        """
        with self._lock:
            if account_id not in self._accounts:
                raise ValueError(f"Account {account_id} does not exist")
            
            account = self._accounts[account_id]
            
            if account.balance > 0:
                raise ValueError(f"Cannot close account {account_id} with positive balance")
            
            # 更新状态为已关闭
            account.status = "closed"
            account.updated_at = datetime.now()
            
            # 持久化更新
            self._update_account_in_persistence(account_id, {
                'status': 'closed',
                'updated_at': account.updated_at
            })
            
            # 从内存中移除
            del self._accounts[account_id]
            
            self.logger.info(f"Account {account_id} closed")
            return True

    def get_account(self, account_id: str) -> Optional[Dict[str, Any]]:
        """
        获取账户信息
        
        先从内存中查找，如果没有则从持久化层加载
        
        Args:
            account_id: 账户ID
        
        Returns:
            账户信息字典或None
        """
        with self._lock:
            account = self._accounts.get(account_id)
            if account is None and self._enable_persistence:
                account = self._load_account_from_persistence(account_id)
                if account:
                    self._accounts[account_id] = account
            
            return self._account_to_dict(account) if account else None

    def update_balance(self, account_id: str, amount: float) -> bool:
        """
        更新账户余额
        
        Args:
            account_id: 账户ID
            amount: 变动金额（正数增加，负数减少）
        
        Returns:
            是否成功更新
        
        Raises:
            ValueError: 账户不存在或余额不足
        """
        with self._lock:
            if account_id not in self._accounts:
                # 尝试从持久化层加载
                account = self._load_account_from_persistence(account_id)
                if account:
                    self._accounts[account_id] = account
                else:
                    raise ValueError(f"Account {account_id} does not exist")
            
            account = self._accounts[account_id]
            new_balance = account.balance + amount
            
            if new_balance < 0:
                raise ValueError("Insufficient funds")
            
            account.balance = new_balance
            account.available_balance = new_balance - account.frozen_balance
            account.updated_at = datetime.now()
            
            # 持久化更新
            self._update_account_in_persistence(account_id, {
                'balance': account.balance,
                'available_balance': account.available_balance,
                'updated_at': account.updated_at
            })
            
            self.logger.info(f"Account {account_id} balance updated by {amount}, new balance: {new_balance}")
            return True

    def freeze_balance(self, account_id: str, amount: float) -> bool:
        """
        冻结账户资金
        
        Args:
            account_id: 账户ID
            amount: 冻结金额
        
        Returns:
            是否成功冻结
        
        Raises:
            ValueError: 账户不存在或可用资金不足
        """
        with self._lock:
            if account_id not in self._accounts:
                raise ValueError(f"Account {account_id} does not exist")
            
            account = self._accounts[account_id]
            
            if amount > account.available_balance:
                raise ValueError("Insufficient available balance")
            
            account.frozen_balance += amount
            account.available_balance -= amount
            account.updated_at = datetime.now()
            
            # 持久化更新
            self._update_account_in_persistence(account_id, {
                'frozen_balance': account.frozen_balance,
                'available_balance': account.available_balance,
                'updated_at': account.updated_at
            })
            
            self.logger.info(f"Account {account_id} frozen {amount}, available: {account.available_balance}")
            return True

    def unfreeze_balance(self, account_id: str, amount: float) -> bool:
        """
        解冻账户资金
        
        Args:
            account_id: 账户ID
            amount: 解冻金额
        
        Returns:
            是否成功解冻
        
        Raises:
            ValueError: 账户不存在或冻结资金不足
        """
        with self._lock:
            if account_id not in self._accounts:
                raise ValueError(f"Account {account_id} does not exist")
            
            account = self._accounts[account_id]
            
            if amount > account.frozen_balance:
                raise ValueError("Insufficient frozen balance")
            
            account.frozen_balance -= amount
            account.available_balance += amount
            account.updated_at = datetime.now()
            
            # 持久化更新
            self._update_account_in_persistence(account_id, {
                'frozen_balance': account.frozen_balance,
                'available_balance': account.available_balance,
                'updated_at': account.updated_at
            })
            
            self.logger.info(f"Account {account_id} unfrozen {amount}, available: {account.available_balance}")
            return True

    def transfer(self, from_account: str, to_account: str, amount: float) -> bool:
        """
        账户间转账
        
        Args:
            from_account: 转出账户ID
            to_account: 转入账户ID
            amount: 转账金额
        
        Returns:
            是否成功转账
        
        Raises:
            ValueError: 账户不存在或余额不足
        """
        with self._lock:
            if amount <= 0:
                raise ValueError("Transfer amount must be positive")
            
            self.update_balance(from_account, -amount)
            self.update_balance(to_account, amount)
            
            self.logger.info(f"Transferred {amount} from {from_account} to {to_account}")
            return True

    def get_total_balance(self) -> float:
        """
        获取所有账户总余额
        
        Returns:
            总余额
        """
        with self._lock:
            return sum(account.balance for account in self._accounts.values())

    def get_account_count(self) -> int:
        """
        获取账户数量
        
        Returns:
            账户数量
        """
        return len(self._accounts)

    def deposit(self, account_id: str, amount: float) -> bool:
        """
        存款
        
        Args:
            account_id: 账户ID
            amount: 存款金额
        
        Returns:
            是否成功存款
        
        Raises:
            ValueError: 账户不存在或金额无效
        """
        with self._lock:
            if account_id not in self._accounts:
                raise ValueError(f"Account {account_id} does not exist")
            
            if amount <= 0:
                raise ValueError("Deposit amount must be positive")
            
            self.update_balance(account_id, amount)
            return True

    def withdraw(self, account_id: str, amount: float) -> bool:
        """
        取款
        
        Args:
            account_id: 账户ID
            amount: 取款金额
        
        Returns:
            是否成功取款
        
        Raises:
            ValueError: 账户不存在或余额不足
        """
        with self._lock:
            if account_id not in self._accounts:
                raise ValueError(f"Account {account_id} does not exist")
            
            if amount <= 0:
                raise ValueError("Withdrawal amount must be positive")
            
            self.update_balance(account_id, -amount)
            return True

    def remove_account(self, account_id: str) -> bool:
        """
        删除账户（允许删除有余额的账户）
        
        Args:
            account_id: 账户ID
        
        Returns:
            是否成功删除
        
        Raises:
            ValueError: 账户不存在
        """
        with self._lock:
            if account_id not in self._accounts:
                raise ValueError(f"Account {account_id} does not exist")
            
            del self._accounts[account_id]
            
            self.logger.info(f"Account {account_id} removed")
            return True

    def list_accounts(self) -> Dict[str, Dict[str, Any]]:
        """
        列出所有账户
        
        Returns:
            所有账户信息
        """
        with self._lock:
            return {
                account_id: self._account_to_dict(account)
                for account_id, account in self._accounts.items()
            }
    
    def _account_to_dict(self, account: AccountInfo) -> Dict[str, Any]:
        """
        将账户信息转换为字典
        
        Args:
            account: 账户信息对象
        
        Returns:
            账户信息字典
        """
        return {
            "id": account.account_id,
            "balance": account.balance,
            "frozen_balance": account.frozen_balance,
            "available_balance": account.available_balance,
            "status": account.status,
            "currency": account.currency,
            "created_at": account.created_at,
            "updated_at": account.updated_at,
            "metadata": account.metadata
        }
    
    @property
    def accounts(self) -> Dict[str, Dict[str, Any]]:
        """
        兼容旧接口：获取账户字典
        
        Returns:
            账户字典
        """
        return self.list_accounts()
