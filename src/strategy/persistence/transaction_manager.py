#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
事务管理模块

提供统一的事务管理功能，支持：
1. 事务上下文管理
2. 嵌套事务支持
3. 事务超时控制
4. 自动回滚和提交
5. 跨存储事务一致性

Author: RQA2025 Development Team
Date: 2026-03-24
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Union

import psycopg2
from psycopg2.extensions import connection as PGConnection

logger = logging.getLogger(__name__)


class TransactionStatus(Enum):
    """事务状态"""
    PENDING = "pending"      # 待开始
    ACTIVE = "active"        # 进行中
    COMMITTED = "committed"  # 已提交
    ROLLED_BACK = "rolled_back"  # 已回滚
    FAILED = "failed"        # 失败


class TransactionIsolationLevel(Enum):
    """事务隔离级别"""
    READ_UNCOMMITTED = "READ UNCOMMITTED"
    READ_COMMITTED = "READ COMMITTED"
    REPEATABLE_READ = "REPEATABLE READ"
    SERIALIZABLE = "SERIALIZABLE"


@dataclass
class TransactionConfig:
    """事务配置"""
    isolation_level: TransactionIsolationLevel = TransactionIsolationLevel.READ_COMMITTED
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    readonly: bool = False
    autocommit: bool = False


class TransactionError(Exception):
    """事务异常"""
    def __init__(self, message: str, transaction_id: str = None, original_error: Exception = None):
        self.message = message
        self.transaction_id = transaction_id
        self.original_error = original_error
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        parts = [self.message]
        if self.transaction_id:
            parts.append(f"Transaction ID: {self.transaction_id}")
        if self.original_error:
            parts.append(f"Original Error: {self.original_error}")
        return " | ".join(parts)


class Transaction:
    """
    事务对象
    
    封装单个事务的状态和操作
    """
    
    _id_counter = 0
    
    def __init__(self, config: TransactionConfig = None):
        Transaction._id_counter += 1
        self.transaction_id = f"tx_{int(time.time())}_{Transaction._id_counter}"
        self.config = config or TransactionConfig()
        self.status = TransactionStatus.PENDING
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.operations: List[Dict[str, Any]] = []
        self._connection: Optional[PGConnection] = None
        self._savepoints: List[str] = []
        
    def begin(self, connection: PGConnection) -> None:
        """开始事务"""
        if self.status != TransactionStatus.PENDING:
            raise TransactionError(f"事务状态错误: {self.status}", self.transaction_id)
        
        self._connection = connection
        self.start_time = time.time()
        self.status = TransactionStatus.ACTIVE
        
        # 设置隔离级别
        cursor = self._connection.cursor()
        cursor.execute(f"SET TRANSACTION ISOLATION LEVEL {self.config.isolation_level.value}")
        cursor.close()
        
        logger.debug(f"事务开始: {self.transaction_id}")
    
    def commit(self) -> None:
        """提交事务"""
        if self.status != TransactionStatus.ACTIVE:
            raise TransactionError(f"无法提交，事务状态: {self.status}", self.transaction_id)
        
        try:
            self._connection.commit()
            self.status = TransactionStatus.COMMITTED
            self.end_time = time.time()
            duration = self.end_time - self.start_time
            logger.info(f"事务提交成功: {self.transaction_id}, 耗时: {duration:.3f}s")
        except Exception as e:
            self.status = TransactionStatus.FAILED
            raise TransactionError("事务提交失败", self.transaction_id, e)
    
    def rollback(self, savepoint: str = None) -> None:
        """回滚事务"""
        if self.status not in [TransactionStatus.ACTIVE, TransactionStatus.FAILED]:
            logger.warning(f"事务状态不支持回滚: {self.status}")
            return
        
        try:
            if savepoint and savepoint in self._savepoints:
                # 回滚到保存点
                cursor = self._connection.cursor()
                cursor.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                cursor.close()
                logger.debug(f"回滚到保存点: {savepoint}")
            else:
                # 完整回滚
                self._connection.rollback()
                self.status = TransactionStatus.ROLLED_BACK
                self.end_time = time.time()
                logger.info(f"事务已回滚: {self.transaction_id}")
        except Exception as e:
            logger.error(f"事务回滚失败: {e}")
            raise TransactionError("事务回滚失败", self.transaction_id, e)
    
    def create_savepoint(self, name: str) -> None:
        """创建保存点"""
        if self.status != TransactionStatus.ACTIVE:
            raise TransactionError("事务未激活，无法创建保存点", self.transaction_id)
        
        cursor = self._connection.cursor()
        cursor.execute(f"SAVEPOINT {name}")
        cursor.close()
        self._savepoints.append(name)
        logger.debug(f"创建保存点: {name}")
    
    def check_timeout(self) -> bool:
        """检查事务是否超时"""
        if self.start_time and self.config.timeout_seconds > 0:
            elapsed = time.time() - self.start_time
            if elapsed > self.config.timeout_seconds:
                logger.warning(f"事务超时: {self.transaction_id}, 已运行: {elapsed:.3f}s")
                return True
        return False
    
    def record_operation(self, operation_type: str, table: str, data: Dict[str, Any]) -> None:
        """记录操作日志"""
        self.operations.append({
            'type': operation_type,
            'table': table,
            'data': data,
            'timestamp': time.time()
        })
    
    def get_duration(self) -> Optional[float]:
        """获取事务持续时间"""
        if self.start_time:
            end = self.end_time or time.time()
            return end - self.start_time
        return None


class TransactionManager:
    """
    事务管理器
    
    统一管理事务的创建、提交、回滚
    支持事务嵌套和上下文管理
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._active_transactions: Dict[str, Transaction] = {}
        self._connection_factory: Optional[Callable[[], PGConnection]] = None
    
    def set_connection_factory(self, factory: Callable[[], PGConnection]) -> None:
        """设置连接工厂"""
        self._connection_factory = factory
    
    def begin_transaction(self, config: TransactionConfig = None) -> Transaction:
        """开始新事务"""
        if not self._connection_factory:
            raise TransactionError("未设置连接工厂")
        
        tx = Transaction(config)
        connection = self._connection_factory()
        tx.begin(connection)
        self._active_transactions[tx.transaction_id] = tx
        return tx
    
    def commit_transaction(self, transaction_id: str) -> None:
        """提交事务"""
        tx = self._active_transactions.get(transaction_id)
        if not tx:
            raise TransactionError(f"事务不存在: {transaction_id}")
        
        try:
            tx.commit()
        finally:
            if tx._connection:
                tx._connection.close()
            del self._active_transactions[transaction_id]
    
    def rollback_transaction(self, transaction_id: str, savepoint: str = None) -> None:
        """回滚事务"""
        tx = self._active_transactions.get(transaction_id)
        if not tx:
            raise TransactionError(f"事务不存在: {transaction_id}")
        
        try:
            tx.rollback(savepoint)
        finally:
            if not savepoint:  # 完整回滚时关闭连接
                if tx._connection:
                    tx._connection.close()
                del self._active_transactions[transaction_id]
    
    @contextmanager
    def transaction_scope(self, config: TransactionConfig = None):
        """
        事务上下文管理器
        
        使用示例:
            with transaction_manager.transaction_scope() as tx:
                # 执行数据库操作
                cur.execute("INSERT ...")
                # 自动提交或回滚
        """
        tx = None
        try:
            tx = self.begin_transaction(config)
            yield tx
            
            # 检查超时
            if tx.check_timeout():
                raise TransactionError("事务超时", tx.transaction_id)
            
            # 自动提交
            self.commit_transaction(tx.transaction_id)
            
        except Exception as e:
            if tx:
                self.logger.error(f"事务异常，执行回滚: {tx.transaction_id}, 错误: {e}")
                try:
                    self.rollback_transaction(tx.transaction_id)
                except Exception as rollback_error:
                    self.logger.error(f"回滚失败: {rollback_error}")
            raise
    
    def get_active_transactions(self) -> List[Transaction]:
        """获取所有活跃事务"""
        return list(self._active_transactions.values())
    
    def cleanup_stale_transactions(self, max_age_seconds: float = 300.0) -> int:
        """清理过期事务"""
        cleaned = 0
        current_time = time.time()
        
        for tx_id, tx in list(self._active_transactions.items()):
            if tx.start_time and (current_time - tx.start_time) > max_age_seconds:
                self.logger.warning(f"清理过期事务: {tx_id}")
                try:
                    self.rollback_transaction(tx_id)
                    cleaned += 1
                except Exception as e:
                    self.logger.error(f"清理事务失败: {e}")
        
        return cleaned


# 全局事务管理器实例
_transaction_manager: Optional[TransactionManager] = None


def get_transaction_manager() -> TransactionManager:
    """获取全局事务管理器"""
    global _transaction_manager
    if _transaction_manager is None:
        _transaction_manager = TransactionManager()
    return _transaction_manager


def set_transaction_manager(manager: TransactionManager) -> None:
    """设置全局事务管理器"""
    global _transaction_manager
    _transaction_manager = manager


# 便捷函数
@contextmanager
def transaction_scope(config: TransactionConfig = None):
    """便捷事务上下文管理器"""
    manager = get_transaction_manager()
    with manager.transaction_scope(config) as tx:
        yield tx
