# -*- coding: utf-8 -*-
"""
交易层持久化管理模块

实现交易层各组件的PostgreSQL数据库持久化，遵循PostgreSQL优先存储策略。
支持订单、账户、持仓、交易记录的完整生命周期管理。

设计原则：
1. PostgreSQL优先存储，连接失败时降级到文件系统
2. 使用统一数据库配置模块获取连接参数
3. 支持连接重试机制
4. 线程安全操作
"""

import os
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_postgresql_config() -> Optional[Dict[str, str]]:
    """
    获取PostgreSQL配置（使用统一配置模块）
    
    Returns:
        数据库配置字典，获取失败时返回None
    """
    try:
        from src.infrastructure.persistence.database_config import get_db_config
        config = get_db_config()
        return config.to_dict()
    except Exception as e:
        logger.warning(f"获取数据库配置失败: {e}，使用环境变量")
        return {
            "host": os.getenv("POSTGRES_HOST", "postgres"),
            "port": os.getenv("POSTGRES_PORT", "5432"),
            "database": os.getenv("POSTGRES_DB", "rqa2025_prod"),
            "user": os.getenv("POSTGRES_USER", "rqa2025_admin"),
            "password": os.getenv("POSTGRES_PASSWORD", "")
        }


def _retry_db_operation(func, max_retries: int = 3, delay: float = 1.0):
    """
    数据库操作重试装饰器
    
    Args:
        func: 要执行的函数
        max_retries: 最大重试次数
        delay: 重试间隔（秒）
    
    Returns:
        函数执行结果
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1))
    raise last_error


# ============================================================
# 数据类定义
# ============================================================

@dataclass
class OrderData:
    """订单数据结构"""
    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: str = "pending"
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    strategy_id: Optional[str] = None
    account_id: Optional[str] = None
    broker_order_id: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccountData:
    """账户数据结构"""
    account_id: str
    balance: float = 0.0
    frozen_balance: float = 0.0
    available_balance: float = 0.0
    status: str = "active"
    currency: str = "CNY"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionData:
    """持仓数据结构"""
    position_id: str
    account_id: str
    symbol: str
    quantity: float = 0.0
    available_quantity: float = 0.0
    frozen_quantity: float = 0.0
    avg_cost: float = 0.0
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeData:
    """交易记录数据结构"""
    trade_id: str
    order_id: str
    account_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    amount: float = 0.0
    commission: float = 0.0
    stamp_duty: float = 0.0
    transfer_fee: float = 0.0
    net_amount: float = 0.0
    pnl: Optional[float] = None
    traded_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# 抽象接口定义
# ============================================================

class IOrderPersistence(ABC):
    """订单持久化接口"""
    
    @abstractmethod
    def save_order(self, order: OrderData) -> bool:
        """保存订单"""
        pass
    
    @abstractmethod
    def update_order(self, order_id: str, updates: Dict[str, Any]) -> bool:
        """更新订单"""
        pass
    
    @abstractmethod
    def get_order(self, order_id: str) -> Optional[OrderData]:
        """获取订单"""
        pass
    
    @abstractmethod
    def get_orders_by_status(self, status: str, limit: int = 100) -> List[OrderData]:
        """按状态获取订单"""
        pass
    
    @abstractmethod
    def delete_order(self, order_id: str) -> bool:
        """删除订单"""
        pass


class IAccountPersistence(ABC):
    """账户持久化接口"""
    
    @abstractmethod
    def save_account(self, account: AccountData) -> bool:
        """保存账户"""
        pass
    
    @abstractmethod
    def update_account(self, account_id: str, updates: Dict[str, Any]) -> bool:
        """更新账户"""
        pass
    
    @abstractmethod
    def get_account(self, account_id: str) -> Optional[AccountData]:
        """获取账户"""
        pass
    
    @abstractmethod
    def get_all_accounts(self) -> List[AccountData]:
        """获取所有账户"""
        pass
    
    @abstractmethod
    def delete_account(self, account_id: str) -> bool:
        """删除账户"""
        pass


class IPositionPersistence(ABC):
    """持仓持久化接口"""
    
    @abstractmethod
    def save_position(self, position: PositionData) -> bool:
        """保存持仓"""
        pass
    
    @abstractmethod
    def update_position(self, position_id: str, updates: Dict[str, Any]) -> bool:
        """更新持仓"""
        pass
    
    @abstractmethod
    def get_position(self, position_id: str) -> Optional[PositionData]:
        """获取持仓"""
        pass
    
    @abstractmethod
    def get_positions_by_account(self, account_id: str) -> List[PositionData]:
        """按账户获取持仓"""
        pass
    
    @abstractmethod
    def delete_position(self, position_id: str) -> bool:
        """删除持仓"""
        pass


class ITradePersistence(ABC):
    """交易记录持久化接口"""
    
    @abstractmethod
    def save_trade(self, trade: TradeData) -> bool:
        """保存交易记录"""
        pass
    
    @abstractmethod
    def get_trade(self, trade_id: str) -> Optional[TradeData]:
        """获取交易记录"""
        pass
    
    @abstractmethod
    def get_trades_by_order(self, order_id: str) -> List[TradeData]:
        """按订单获取交易记录"""
        pass
    
    @abstractmethod
    def get_trades_by_account(self, account_id: str, limit: int = 100) -> List[TradeData]:
        """按账户获取交易记录"""
        pass


# ============================================================
# PostgreSQL持久化实现
# ============================================================

class OrderPersistence(IOrderPersistence):
    """
    订单持久化管理器
    
    实现PostgreSQL优先存储策略，支持自动降级到文件系统。
    """
    
    def __init__(self, storage_dir: Optional[str] = None):
        """
        初始化订单持久化管理器
        
        Args:
            storage_dir: 文件存储目录（降级时使用）
        """
        self._lock = threading.RLock()
        self._db_config = _get_postgresql_config()
        self._use_postgresql = self._test_db_connection()
        
        self._storage_dir = Path(storage_dir or os.path.join(
            os.getcwd(), "data", "trading", "orders"
        ))
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        
        self._cache: Dict[str, OrderData] = {}
        self._max_cache_size = 10000
        
        if self._use_postgresql:
            logger.info("OrderPersistence: 使用PostgreSQL存储")
        else:
            logger.warning("OrderPersistence: PostgreSQL不可用，降级到文件系统存储")
    
    def _test_db_connection(self) -> bool:
        """测试数据库连接"""
        if not self._db_config or not self._db_config.get("password"):
            return False
        try:
            import psycopg2
            conn = psycopg2.connect(**self._db_config)
            conn.close()
            return True
        except Exception as e:
            logger.warning(f"数据库连接测试失败: {e}")
            return False
    
    def _get_connection(self):
        """获取数据库连接"""
        import psycopg2
        return psycopg2.connect(**self._db_config)
    
    def save_order(self, order: OrderData) -> bool:
        """
        保存订单
        
        Args:
            order: 订单数据
        
        Returns:
            是否保存成功
        """
        with self._lock:
            try:
                if self._use_postgresql:
                    return self._save_order_to_db(order)
                else:
                    return self._save_order_to_file(order)
            except Exception as e:
                logger.error(f"保存订单失败: {e}")
                return False
    
    def _save_order_to_db(self, order: OrderData) -> bool:
        """保存订单到PostgreSQL"""
        def _do_save():
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO trading_orders (
                            order_id, symbol, side, order_type, quantity, price,
                            stop_price, status, filled_quantity, avg_fill_price,
                            strategy_id, account_id, broker_order_id, error_message,
                            created_at, updated_at, metadata
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (order_id) DO UPDATE SET
                            status = EXCLUDED.status,
                            filled_quantity = EXCLUDED.filled_quantity,
                            avg_fill_price = EXCLUDED.avg_fill_price,
                            broker_order_id = EXCLUDED.broker_order_id,
                            error_message = EXCLUDED.error_message,
                            updated_at = EXCLUDED.updated_at,
                            metadata = EXCLUDED.metadata
                    """, (
                        order.order_id, order.symbol, order.side, order.order_type,
                        order.quantity, order.price, order.stop_price, order.status,
                        order.filled_quantity, order.avg_fill_price, order.strategy_id,
                        order.account_id, order.broker_order_id, order.error_message,
                        order.created_at, order.updated_at, json.dumps(order.metadata)
                    ))
                    conn.commit()
            return True
        
        try:
            return _retry_db_operation(_do_save)
        except Exception as e:
            logger.error(f"数据库保存订单失败: {e}")
            self._use_postgresql = False
            return self._save_order_to_file(order)
    
    def _save_order_to_file(self, order: OrderData) -> bool:
        """保存订单到文件系统"""
        try:
            file_path = self._storage_dir / f"{order.order_id}.json"
            data = asdict(order)
            data['created_at'] = data['created_at'].isoformat()
            data['updated_at'] = data['updated_at'].isoformat()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            self._update_cache(order)
            return True
        except Exception as e:
            logger.error(f"文件保存订单失败: {e}")
            return False
    
    def update_order(self, order_id: str, updates: Dict[str, Any]) -> bool:
        """
        更新订单
        
        Args:
            order_id: 订单ID
            updates: 更新字段字典
        
        Returns:
            是否更新成功
        """
        with self._lock:
            updates['updated_at'] = datetime.now()
            
            try:
                if self._use_postgresql:
                    return self._update_order_in_db(order_id, updates)
                else:
                    return self._update_order_in_file(order_id, updates)
            except Exception as e:
                logger.error(f"更新订单失败: {e}")
                return False
    
    def _update_order_in_db(self, order_id: str, updates: Dict[str, Any]) -> bool:
        """在数据库中更新订单"""
        def _do_update():
            set_clauses = []
            values = []
            for key, value in updates.items():
                if key == 'metadata':
                    set_clauses.append(f"{key} = %s")
                    values.append(json.dumps(value))
                else:
                    set_clauses.append(f"{key} = %s")
                    values.append(value)
            values.append(order_id)
            
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        UPDATE trading_orders 
                        SET {', '.join(set_clauses)}
                        WHERE order_id = %s
                    """, values)
                    conn.commit()
            return True
        
        return _retry_db_operation(_do_update)
    
    def _update_order_in_file(self, order_id: str, updates: Dict[str, Any]) -> bool:
        """在文件中更新订单"""
        order = self.get_order(order_id)
        if not order:
            return False
        
        for key, value in updates.items():
            if hasattr(order, key):
                setattr(order, key, value)
        
        return self._save_order_to_file(order)
    
    def get_order(self, order_id: str) -> Optional[OrderData]:
        """
        获取订单
        
        Args:
            order_id: 订单ID
        
        Returns:
            订单数据，不存在返回None
        """
        with self._lock:
            if order_id in self._cache:
                return self._cache[order_id]
            
            try:
                if self._use_postgresql:
                    return self._get_order_from_db(order_id)
                else:
                    return self._get_order_from_file(order_id)
            except Exception as e:
                logger.error(f"获取订单失败: {e}")
                return None
    
    def _get_order_from_db(self, order_id: str) -> Optional[OrderData]:
        """从数据库获取订单"""
        def _do_get():
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT order_id, symbol, side, order_type, quantity, price,
                               stop_price, status, filled_quantity, avg_fill_price,
                               strategy_id, account_id, broker_order_id, error_message,
                               created_at, updated_at, metadata
                        FROM trading_orders WHERE order_id = %s
                    """, (order_id,))
                    row = cur.fetchone()
                    if row:
                        return OrderData(
                            order_id=row[0], symbol=row[1], side=row[2],
                            order_type=row[3], quantity=row[4], price=row[5],
                            stop_price=row[6], status=row[7], filled_quantity=row[8],
                            avg_fill_price=row[9], strategy_id=row[10], account_id=row[11],
                            broker_order_id=row[12], error_message=row[13],
                            created_at=row[14], updated_at=row[15],
                            metadata=row[16] if isinstance(row[16], dict) else json.loads(row[16] or '{}')
                        )
                    return None
        
        result = _retry_db_operation(_do_get)
        if result:
            self._update_cache(result)
        return result
    
    def _get_order_from_file(self, order_id: str) -> Optional[OrderData]:
        """从文件获取订单"""
        file_path = self._storage_dir / f"{order_id}.json"
        if not file_path.exists():
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        order = OrderData(**data)
        self._update_cache(order)
        return order
    
    def get_orders_by_status(self, status: str, limit: int = 100) -> List[OrderData]:
        """
        按状态获取订单
        
        Args:
            status: 订单状态
            limit: 返回数量限制
        
        Returns:
            订单列表
        """
        with self._lock:
            try:
                if self._use_postgresql:
                    return self._get_orders_by_status_from_db(status, limit)
                else:
                    return self._get_orders_by_status_from_files(status, limit)
            except Exception as e:
                logger.error(f"按状态获取订单失败: {e}")
                return []
    
    def _get_orders_by_status_from_db(self, status: str, limit: int) -> List[OrderData]:
        """从数据库按状态获取订单"""
        def _do_get():
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT order_id, symbol, side, order_type, quantity, price,
                               stop_price, status, filled_quantity, avg_fill_price,
                               strategy_id, account_id, broker_order_id, error_message,
                               created_at, updated_at, metadata
                        FROM trading_orders 
                        WHERE status = %s 
                        ORDER BY created_at DESC 
                        LIMIT %s
                    """, (status, limit))
                    rows = cur.fetchall()
                    return [
                        OrderData(
                            order_id=row[0], symbol=row[1], side=row[2],
                            order_type=row[3], quantity=row[4], price=row[5],
                            stop_price=row[6], status=row[7], filled_quantity=row[8],
                            avg_fill_price=row[9], strategy_id=row[10], account_id=row[11],
                            broker_order_id=row[12], error_message=row[13],
                            created_at=row[14], updated_at=row[15],
                            metadata=row[16] if isinstance(row[16], dict) else json.loads(row[16] or '{}')
                        ) for row in rows
                    ]
        
        return _retry_db_operation(_do_get)
    
    def _get_orders_by_status_from_files(self, status: str, limit: int) -> List[OrderData]:
        """从文件按状态获取订单"""
        orders = []
        for file_path in self._storage_dir.glob("*.json"):
            try:
                order = self._get_order_from_file(file_path.stem)
                if order and order.status == status:
                    orders.append(order)
                    if len(orders) >= limit:
                        break
            except Exception:
                continue
        return orders
    
    def delete_order(self, order_id: str) -> bool:
        """
        删除订单
        
        Args:
            order_id: 订单ID
        
        Returns:
            是否删除成功
        """
        with self._lock:
            try:
                if self._use_postgresql:
                    return self._delete_order_from_db(order_id)
                else:
                    return self._delete_order_from_file(order_id)
            except Exception as e:
                logger.error(f"删除订单失败: {e}")
                return False
    
    def _delete_order_from_db(self, order_id: str) -> bool:
        """从数据库删除订单"""
        def _do_delete():
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM trading_orders WHERE order_id = %s", (order_id,))
                    conn.commit()
            return True
        
        return _retry_db_operation(_do_delete)
    
    def _delete_order_from_file(self, order_id: str) -> bool:
        """从文件删除订单"""
        file_path = self._storage_dir / f"{order_id}.json"
        if file_path.exists():
            file_path.unlink()
        if order_id in self._cache:
            del self._cache[order_id]
        return True
    
    def _update_cache(self, order: OrderData):
        """更新缓存"""
        if len(self._cache) >= self._max_cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[order.order_id] = order


class AccountPersistence(IAccountPersistence):
    """
    账户持久化管理器
    
    实现PostgreSQL优先存储策略，支持自动降级到文件系统。
    """
    
    def __init__(self, storage_dir: Optional[str] = None):
        """
        初始化账户持久化管理器
        
        Args:
            storage_dir: 文件存储目录（降级时使用）
        """
        self._lock = threading.RLock()
        self._db_config = _get_postgresql_config()
        self._use_postgresql = self._test_db_connection()
        
        self._storage_dir = Path(storage_dir or os.path.join(
            os.getcwd(), "data", "trading", "accounts"
        ))
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        
        self._cache: Dict[str, AccountData] = {}
        self._max_cache_size = 1000
        
        if self._use_postgresql:
            logger.info("AccountPersistence: 使用PostgreSQL存储")
        else:
            logger.warning("AccountPersistence: PostgreSQL不可用，降级到文件系统存储")
    
    def _test_db_connection(self) -> bool:
        """测试数据库连接"""
        if not self._db_config or not self._db_config.get("password"):
            return False
        try:
            import psycopg2
            conn = psycopg2.connect(**self._db_config)
            conn.close()
            return True
        except Exception as e:
            logger.warning(f"数据库连接测试失败: {e}")
            return False
    
    def _get_connection(self):
        """获取数据库连接"""
        import psycopg2
        return psycopg2.connect(**self._db_config)
    
    def save_account(self, account: AccountData) -> bool:
        """
        保存账户
        
        Args:
            account: 账户数据
        
        Returns:
            是否保存成功
        """
        with self._lock:
            try:
                if self._use_postgresql:
                    return self._save_account_to_db(account)
                else:
                    return self._save_account_to_file(account)
            except Exception as e:
                logger.error(f"保存账户失败: {e}")
                return False
    
    def _save_account_to_db(self, account: AccountData) -> bool:
        """保存账户到PostgreSQL"""
        def _do_save():
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO trading_accounts (
                            account_id, balance, frozen_balance, available_balance,
                            status, currency, created_at, updated_at, metadata
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (account_id) DO UPDATE SET
                            balance = EXCLUDED.balance,
                            frozen_balance = EXCLUDED.frozen_balance,
                            available_balance = EXCLUDED.available_balance,
                            status = EXCLUDED.status,
                            updated_at = EXCLUDED.updated_at,
                            metadata = EXCLUDED.metadata
                    """, (
                        account.account_id, account.balance, account.frozen_balance,
                        account.available_balance, account.status, account.currency,
                        account.created_at, account.updated_at, json.dumps(account.metadata)
                    ))
                    conn.commit()
            return True
        
        try:
            return _retry_db_operation(_do_save)
        except Exception as e:
            logger.error(f"数据库保存账户失败: {e}")
            self._use_postgresql = False
            return self._save_account_to_file(account)
    
    def _save_account_to_file(self, account: AccountData) -> bool:
        """保存账户到文件系统"""
        try:
            file_path = self._storage_dir / f"{account.account_id}.json"
            data = asdict(account)
            data['created_at'] = data['created_at'].isoformat()
            data['updated_at'] = data['updated_at'].isoformat()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            self._update_cache(account)
            return True
        except Exception as e:
            logger.error(f"文件保存账户失败: {e}")
            return False
    
    def update_account(self, account_id: str, updates: Dict[str, Any]) -> bool:
        """
        更新账户
        
        Args:
            account_id: 账户ID
            updates: 更新字段字典
        
        Returns:
            是否更新成功
        """
        with self._lock:
            updates['updated_at'] = datetime.now()
            
            try:
                if self._use_postgresql:
                    return self._update_account_in_db(account_id, updates)
                else:
                    return self._update_account_in_file(account_id, updates)
            except Exception as e:
                logger.error(f"更新账户失败: {e}")
                return False
    
    def _update_account_in_db(self, account_id: str, updates: Dict[str, Any]) -> bool:
        """在数据库中更新账户"""
        def _do_update():
            set_clauses = []
            values = []
            for key, value in updates.items():
                if key == 'metadata':
                    set_clauses.append(f"{key} = %s")
                    values.append(json.dumps(value))
                else:
                    set_clauses.append(f"{key} = %s")
                    values.append(value)
            values.append(account_id)
            
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        UPDATE trading_accounts 
                        SET {', '.join(set_clauses)}
                        WHERE account_id = %s
                    """, values)
                    conn.commit()
            return True
        
        return _retry_db_operation(_do_update)
    
    def _update_account_in_file(self, account_id: str, updates: Dict[str, Any]) -> bool:
        """在文件中更新账户"""
        account = self.get_account(account_id)
        if not account:
            return False
        
        for key, value in updates.items():
            if hasattr(account, key):
                setattr(account, key, value)
        
        return self._save_account_to_file(account)
    
    def get_account(self, account_id: str) -> Optional[AccountData]:
        """
        获取账户
        
        Args:
            account_id: 账户ID
        
        Returns:
            账户数据，不存在返回None
        """
        with self._lock:
            if account_id in self._cache:
                return self._cache[account_id]
            
            try:
                if self._use_postgresql:
                    return self._get_account_from_db(account_id)
                else:
                    return self._get_account_from_file(account_id)
            except Exception as e:
                logger.error(f"获取账户失败: {e}")
                return None
    
    def _get_account_from_db(self, account_id: str) -> Optional[AccountData]:
        """从数据库获取账户"""
        def _do_get():
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT account_id, balance, frozen_balance, available_balance,
                               status, currency, created_at, updated_at, metadata
                        FROM trading_accounts WHERE account_id = %s
                    """, (account_id,))
                    row = cur.fetchone()
                    if row:
                        return AccountData(
                            account_id=row[0], balance=float(row[1]),
                            frozen_balance=float(row[2]), available_balance=float(row[3]),
                            status=row[4], currency=row[5],
                            created_at=row[6], updated_at=row[7],
                            metadata=row[8] if isinstance(row[8], dict) else json.loads(row[8] or '{}')
                        )
                    return None
        
        result = _retry_db_operation(_do_get)
        if result:
            self._update_cache(result)
        return result
    
    def _get_account_from_file(self, account_id: str) -> Optional[AccountData]:
        """从文件获取账户"""
        file_path = self._storage_dir / f"{account_id}.json"
        if not file_path.exists():
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        account = AccountData(**data)
        self._update_cache(account)
        return account
    
    def get_all_accounts(self) -> List[AccountData]:
        """
        获取所有账户
        
        Returns:
            账户列表
        """
        with self._lock:
            try:
                if self._use_postgresql:
                    return self._get_all_accounts_from_db()
                else:
                    return self._get_all_accounts_from_files()
            except Exception as e:
                logger.error(f"获取所有账户失败: {e}")
                return []
    
    def _get_all_accounts_from_db(self) -> List[AccountData]:
        """从数据库获取所有账户"""
        def _do_get():
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT account_id, balance, frozen_balance, available_balance,
                               status, currency, created_at, updated_at, metadata
                        FROM trading_accounts WHERE status = 'active'
                        ORDER BY created_at DESC
                    """)
                    rows = cur.fetchall()
                    return [
                        AccountData(
                            account_id=row[0], balance=float(row[1]),
                            frozen_balance=float(row[2]), available_balance=float(row[3]),
                            status=row[4], currency=row[5],
                            created_at=row[6], updated_at=row[7],
                            metadata=row[8] if isinstance(row[8], dict) else json.loads(row[8] or '{}')
                        ) for row in rows
                    ]
        
        return _retry_db_operation(_do_get)
    
    def _get_all_accounts_from_files(self) -> List[AccountData]:
        """从文件获取所有账户"""
        accounts = []
        for file_path in self._storage_dir.glob("*.json"):
            try:
                account = self._get_account_from_file(file_path.stem)
                if account:
                    accounts.append(account)
            except Exception:
                continue
        return accounts
    
    def delete_account(self, account_id: str) -> bool:
        """
        删除账户
        
        Args:
            account_id: 账户ID
        
        Returns:
            是否删除成功
        """
        with self._lock:
            try:
                if self._use_postgresql:
                    return self._delete_account_from_db(account_id)
                else:
                    return self._delete_account_from_file(account_id)
            except Exception as e:
                logger.error(f"删除账户失败: {e}")
                return False
    
    def _delete_account_from_db(self, account_id: str) -> bool:
        """从数据库删除账户"""
        def _do_delete():
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM trading_accounts WHERE account_id = %s", (account_id,))
                    conn.commit()
            return True
        
        return _retry_db_operation(_do_delete)
    
    def _delete_account_from_file(self, account_id: str) -> bool:
        """从文件删除账户"""
        file_path = self._storage_dir / f"{account_id}.json"
        if file_path.exists():
            file_path.unlink()
        if account_id in self._cache:
            del self._cache[account_id]
        return True
    
    def _update_cache(self, account: AccountData):
        """更新缓存"""
        if len(self._cache) >= self._max_cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[account.account_id] = account


class PositionPersistence(IPositionPersistence):
    """
    持仓持久化管理器
    
    实现PostgreSQL优先存储策略，支持自动降级到文件系统。
    """
    
    def __init__(self, storage_dir: Optional[str] = None):
        """
        初始化持仓持久化管理器
        
        Args:
            storage_dir: 文件存储目录（降级时使用）
        """
        self._lock = threading.RLock()
        self._db_config = _get_postgresql_config()
        self._use_postgresql = self._test_db_connection()
        
        self._storage_dir = Path(storage_dir or os.path.join(
            os.getcwd(), "data", "trading", "positions"
        ))
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        
        self._cache: Dict[str, PositionData] = {}
        self._max_cache_size = 5000
        
        if self._use_postgresql:
            logger.info("PositionPersistence: 使用PostgreSQL存储")
        else:
            logger.warning("PositionPersistence: PostgreSQL不可用，降级到文件系统存储")
    
    def _test_db_connection(self) -> bool:
        """测试数据库连接"""
        if not self._db_config or not self._db_config.get("password"):
            return False
        try:
            import psycopg2
            conn = psycopg2.connect(**self._db_config)
            conn.close()
            return True
        except Exception as e:
            logger.warning(f"数据库连接测试失败: {e}")
            return False
    
    def _get_connection(self):
        """获取数据库连接"""
        import psycopg2
        return psycopg2.connect(**self._db_config)
    
    def save_position(self, position: PositionData) -> bool:
        """
        保存持仓
        
        Args:
            position: 持仓数据
        
        Returns:
            是否保存成功
        """
        with self._lock:
            try:
                if self._use_postgresql:
                    return self._save_position_to_db(position)
                else:
                    return self._save_position_to_file(position)
            except Exception as e:
                logger.error(f"保存持仓失败: {e}")
                return False
    
    def _save_position_to_db(self, position: PositionData) -> bool:
        """保存持仓到PostgreSQL"""
        def _do_save():
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO trading_positions (
                            position_id, account_id, symbol, quantity, available_quantity,
                            frozen_quantity, avg_cost, current_price, market_value,
                            unrealized_pnl, realized_pnl, created_at, updated_at, metadata
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (position_id) DO UPDATE SET
                            quantity = EXCLUDED.quantity,
                            available_quantity = EXCLUDED.available_quantity,
                            frozen_quantity = EXCLUDED.frozen_quantity,
                            avg_cost = EXCLUDED.avg_cost,
                            current_price = EXCLUDED.current_price,
                            market_value = EXCLUDED.market_value,
                            unrealized_pnl = EXCLUDED.unrealized_pnl,
                            realized_pnl = EXCLUDED.realized_pnl,
                            updated_at = EXCLUDED.updated_at,
                            metadata = EXCLUDED.metadata
                    """, (
                        position.position_id, position.account_id, position.symbol,
                        position.quantity, position.available_quantity, position.frozen_quantity,
                        position.avg_cost, position.current_price, position.market_value,
                        position.unrealized_pnl, position.realized_pnl,
                        position.created_at, position.updated_at, json.dumps(position.metadata)
                    ))
                    conn.commit()
            return True
        
        try:
            return _retry_db_operation(_do_save)
        except Exception as e:
            logger.error(f"数据库保存持仓失败: {e}")
            self._use_postgresql = False
            return self._save_position_to_file(position)
    
    def _save_position_to_file(self, position: PositionData) -> bool:
        """保存持仓到文件系统"""
        try:
            file_path = self._storage_dir / f"{position.position_id}.json"
            data = asdict(position)
            data['created_at'] = data['created_at'].isoformat()
            data['updated_at'] = data['updated_at'].isoformat()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            self._update_cache(position)
            return True
        except Exception as e:
            logger.error(f"文件保存持仓失败: {e}")
            return False
    
    def update_position(self, position_id: str, updates: Dict[str, Any]) -> bool:
        """
        更新持仓
        
        Args:
            position_id: 持仓ID
            updates: 更新字段字典
        
        Returns:
            是否更新成功
        """
        with self._lock:
            updates['updated_at'] = datetime.now()
            
            try:
                if self._use_postgresql:
                    return self._update_position_in_db(position_id, updates)
                else:
                    return self._update_position_in_file(position_id, updates)
            except Exception as e:
                logger.error(f"更新持仓失败: {e}")
                return False
    
    def _update_position_in_db(self, position_id: str, updates: Dict[str, Any]) -> bool:
        """在数据库中更新持仓"""
        def _do_update():
            set_clauses = []
            values = []
            for key, value in updates.items():
                if key == 'metadata':
                    set_clauses.append(f"{key} = %s")
                    values.append(json.dumps(value))
                else:
                    set_clauses.append(f"{key} = %s")
                    values.append(value)
            values.append(position_id)
            
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        UPDATE trading_positions 
                        SET {', '.join(set_clauses)}
                        WHERE position_id = %s
                    """, values)
                    conn.commit()
            return True
        
        return _retry_db_operation(_do_update)
    
    def _update_position_in_file(self, position_id: str, updates: Dict[str, Any]) -> bool:
        """在文件中更新持仓"""
        position = self.get_position(position_id)
        if not position:
            return False
        
        for key, value in updates.items():
            if hasattr(position, key):
                setattr(position, key, value)
        
        return self._save_position_to_file(position)
    
    def get_position(self, position_id: str) -> Optional[PositionData]:
        """
        获取持仓
        
        Args:
            position_id: 持仓ID
        
        Returns:
            持仓数据，不存在返回None
        """
        with self._lock:
            if position_id in self._cache:
                return self._cache[position_id]
            
            try:
                if self._use_postgresql:
                    return self._get_position_from_db(position_id)
                else:
                    return self._get_position_from_file(position_id)
            except Exception as e:
                logger.error(f"获取持仓失败: {e}")
                return None
    
    def _get_position_from_db(self, position_id: str) -> Optional[PositionData]:
        """从数据库获取持仓"""
        def _do_get():
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT position_id, account_id, symbol, quantity, available_quantity,
                               frozen_quantity, avg_cost, current_price, market_value,
                               unrealized_pnl, realized_pnl, created_at, updated_at, metadata
                        FROM trading_positions WHERE position_id = %s
                    """, (position_id,))
                    row = cur.fetchone()
                    if row:
                        return PositionData(
                            position_id=row[0], account_id=row[1], symbol=row[2],
                            quantity=float(row[3]), available_quantity=float(row[4]),
                            frozen_quantity=float(row[5]), avg_cost=float(row[6]),
                            current_price=float(row[7]), market_value=float(row[8]),
                            unrealized_pnl=float(row[9]), realized_pnl=float(row[10]),
                            created_at=row[11], updated_at=row[12],
                            metadata=row[13] if isinstance(row[13], dict) else json.loads(row[13] or '{}')
                        )
                    return None
        
        result = _retry_db_operation(_do_get)
        if result:
            self._update_cache(result)
        return result
    
    def _get_position_from_file(self, position_id: str) -> Optional[PositionData]:
        """从文件获取持仓"""
        file_path = self._storage_dir / f"{position_id}.json"
        if not file_path.exists():
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        position = PositionData(**data)
        self._update_cache(position)
        return position
    
    def get_positions_by_account(self, account_id: str) -> List[PositionData]:
        """
        按账户获取持仓
        
        Args:
            account_id: 账户ID
        
        Returns:
            持仓列表
        """
        with self._lock:
            try:
                if self._use_postgresql:
                    return self._get_positions_by_account_from_db(account_id)
                else:
                    return self._get_positions_by_account_from_files(account_id)
            except Exception as e:
                logger.error(f"按账户获取持仓失败: {e}")
                return []
    
    def _get_positions_by_account_from_db(self, account_id: str) -> List[PositionData]:
        """从数据库按账户获取持仓"""
        def _do_get():
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT position_id, account_id, symbol, quantity, available_quantity,
                               frozen_quantity, avg_cost, current_price, market_value,
                               unrealized_pnl, realized_pnl, created_at, updated_at, metadata
                        FROM trading_positions 
                        WHERE account_id = %s AND quantity > 0
                        ORDER BY updated_at DESC
                    """, (account_id,))
                    rows = cur.fetchall()
                    return [
                        PositionData(
                            position_id=row[0], account_id=row[1], symbol=row[2],
                            quantity=float(row[3]), available_quantity=float(row[4]),
                            frozen_quantity=float(row[5]), avg_cost=float(row[6]),
                            current_price=float(row[7]), market_value=float(row[8]),
                            unrealized_pnl=float(row[9]), realized_pnl=float(row[10]),
                            created_at=row[11], updated_at=row[12],
                            metadata=row[13] if isinstance(row[13], dict) else json.loads(row[13] or '{}')
                        ) for row in rows
                    ]
        
        return _retry_db_operation(_do_get)
    
    def _get_positions_by_account_from_files(self, account_id: str) -> List[PositionData]:
        """从文件按账户获取持仓"""
        positions = []
        for file_path in self._storage_dir.glob("*.json"):
            try:
                position = self._get_position_from_file(file_path.stem)
                if position and position.account_id == account_id and position.quantity > 0:
                    positions.append(position)
            except Exception:
                continue
        return positions
    
    def delete_position(self, position_id: str) -> bool:
        """
        删除持仓
        
        Args:
            position_id: 持仓ID
        
        Returns:
            是否删除成功
        """
        with self._lock:
            try:
                if self._use_postgresql:
                    return self._delete_position_from_db(position_id)
                else:
                    return self._delete_position_from_file(position_id)
            except Exception as e:
                logger.error(f"删除持仓失败: {e}")
                return False
    
    def _delete_position_from_db(self, position_id: str) -> bool:
        """从数据库删除持仓"""
        def _do_delete():
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM trading_positions WHERE position_id = %s", (position_id,))
                    conn.commit()
            return True
        
        return _retry_db_operation(_do_delete)
    
    def _delete_position_from_file(self, position_id: str) -> bool:
        """从文件删除持仓"""
        file_path = self._storage_dir / f"{position_id}.json"
        if file_path.exists():
            file_path.unlink()
        if position_id in self._cache:
            del self._cache[position_id]
        return True
    
    def _update_cache(self, position: PositionData):
        """更新缓存"""
        if len(self._cache) >= self._max_cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[position.position_id] = position


class TradePersistence(ITradePersistence):
    """
    交易记录持久化管理器
    
    实现PostgreSQL优先存储策略，支持自动降级到文件系统。
    """
    
    def __init__(self, storage_dir: Optional[str] = None):
        """
        初始化交易记录持久化管理器
        
        Args:
            storage_dir: 文件存储目录（降级时使用）
        """
        self._lock = threading.RLock()
        self._db_config = _get_postgresql_config()
        self._use_postgresql = self._test_db_connection()
        
        self._storage_dir = Path(storage_dir or os.path.join(
            os.getcwd(), "data", "trading", "trades"
        ))
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        
        self._cache: Dict[str, TradeData] = {}
        self._max_cache_size = 10000
        
        if self._use_postgresql:
            logger.info("TradePersistence: 使用PostgreSQL存储")
        else:
            logger.warning("TradePersistence: PostgreSQL不可用，降级到文件系统存储")
    
    def _test_db_connection(self) -> bool:
        """测试数据库连接"""
        if not self._db_config or not self._db_config.get("password"):
            return False
        try:
            import psycopg2
            conn = psycopg2.connect(**self._db_config)
            conn.close()
            return True
        except Exception as e:
            logger.warning(f"数据库连接测试失败: {e}")
            return False
    
    def _get_connection(self):
        """获取数据库连接"""
        import psycopg2
        return psycopg2.connect(**self._db_config)
    
    def save_trade(self, trade: TradeData) -> bool:
        """
        保存交易记录
        
        Args:
            trade: 交易记录数据
        
        Returns:
            是否保存成功
        """
        with self._lock:
            try:
                if self._use_postgresql:
                    return self._save_trade_to_db(trade)
                else:
                    return self._save_trade_to_file(trade)
            except Exception as e:
                logger.error(f"保存交易记录失败: {e}")
                return False
    
    def _save_trade_to_db(self, trade: TradeData) -> bool:
        """保存交易记录到PostgreSQL"""
        def _do_save():
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO trading_trades (
                            trade_id, order_id, account_id, symbol, side,
                            quantity, price, amount, commission, stamp_duty,
                            transfer_fee, net_amount, pnl, traded_at, metadata
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (trade_id) DO UPDATE SET
                            pnl = EXCLUDED.pnl,
                            metadata = EXCLUDED.metadata
                    """, (
                        trade.trade_id, trade.order_id, trade.account_id,
                        trade.symbol, trade.side, trade.quantity, trade.price,
                        trade.amount, trade.commission, trade.stamp_duty,
                        trade.transfer_fee, trade.net_amount, trade.pnl,
                        trade.traded_at, json.dumps(trade.metadata)
                    ))
                    conn.commit()
            return True
        
        try:
            return _retry_db_operation(_do_save)
        except Exception as e:
            logger.error(f"数据库保存交易记录失败: {e}")
            self._use_postgresql = False
            return self._save_trade_to_file(trade)
    
    def _save_trade_to_file(self, trade: TradeData) -> bool:
        """保存交易记录到文件系统"""
        try:
            file_path = self._storage_dir / f"{trade.trade_id}.json"
            data = asdict(trade)
            data['traded_at'] = data['traded_at'].isoformat()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            self._update_cache(trade)
            return True
        except Exception as e:
            logger.error(f"文件保存交易记录失败: {e}")
            return False
    
    def get_trade(self, trade_id: str) -> Optional[TradeData]:
        """
        获取交易记录
        
        Args:
            trade_id: 交易记录ID
        
        Returns:
            交易记录数据，不存在返回None
        """
        with self._lock:
            if trade_id in self._cache:
                return self._cache[trade_id]
            
            try:
                if self._use_postgresql:
                    return self._get_trade_from_db(trade_id)
                else:
                    return self._get_trade_from_file(trade_id)
            except Exception as e:
                logger.error(f"获取交易记录失败: {e}")
                return None
    
    def _get_trade_from_db(self, trade_id: str) -> Optional[TradeData]:
        """从数据库获取交易记录"""
        def _do_get():
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT trade_id, order_id, account_id, symbol, side,
                               quantity, price, amount, commission, stamp_duty,
                               transfer_fee, net_amount, pnl, traded_at, metadata
                        FROM trading_trades WHERE trade_id = %s
                    """, (trade_id,))
                    row = cur.fetchone()
                    if row:
                        return TradeData(
                            trade_id=row[0], order_id=row[1], account_id=row[2],
                            symbol=row[3], side=row[4], quantity=float(row[5]),
                            price=float(row[6]), amount=float(row[7]),
                            commission=float(row[8]), stamp_duty=float(row[9]),
                            transfer_fee=float(row[10]), net_amount=float(row[11]),
                            pnl=float(row[12]) if row[12] else None,
                            traded_at=row[13],
                            metadata=row[14] if isinstance(row[14], dict) else json.loads(row[14] or '{}')
                        )
                    return None
        
        result = _retry_db_operation(_do_get)
        if result:
            self._update_cache(result)
        return result
    
    def _get_trade_from_file(self, trade_id: str) -> Optional[TradeData]:
        """从文件获取交易记录"""
        file_path = self._storage_dir / f"{trade_id}.json"
        if not file_path.exists():
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        data['traded_at'] = datetime.fromisoformat(data['traded_at'])
        
        trade = TradeData(**data)
        self._update_cache(trade)
        return trade
    
    def get_trades_by_order(self, order_id: str) -> List[TradeData]:
        """
        按订单获取交易记录
        
        Args:
            order_id: 订单ID
        
        Returns:
            交易记录列表
        """
        with self._lock:
            try:
                if self._use_postgresql:
                    return self._get_trades_by_order_from_db(order_id)
                else:
                    return self._get_trades_by_order_from_files(order_id)
            except Exception as e:
                logger.error(f"按订单获取交易记录失败: {e}")
                return []
    
    def _get_trades_by_order_from_db(self, order_id: str) -> List[TradeData]:
        """从数据库按订单获取交易记录"""
        def _do_get():
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT trade_id, order_id, account_id, symbol, side,
                               quantity, price, amount, commission, stamp_duty,
                               transfer_fee, net_amount, pnl, traded_at, metadata
                        FROM trading_trades 
                        WHERE order_id = %s
                        ORDER BY traded_at DESC
                    """, (order_id,))
                    rows = cur.fetchall()
                    return [
                        TradeData(
                            trade_id=row[0], order_id=row[1], account_id=row[2],
                            symbol=row[3], side=row[4], quantity=float(row[5]),
                            price=float(row[6]), amount=float(row[7]),
                            commission=float(row[8]), stamp_duty=float(row[9]),
                            transfer_fee=float(row[10]), net_amount=float(row[11]),
                            pnl=float(row[12]) if row[12] else None,
                            traded_at=row[13],
                            metadata=row[14] if isinstance(row[14], dict) else json.loads(row[14] or '{}')
                        ) for row in rows
                    ]
        
        return _retry_db_operation(_do_get)
    
    def _get_trades_by_order_from_files(self, order_id: str) -> List[TradeData]:
        """从文件按订单获取交易记录"""
        trades = []
        for file_path in self._storage_dir.glob("*.json"):
            try:
                trade = self._get_trade_from_file(file_path.stem)
                if trade and trade.order_id == order_id:
                    trades.append(trade)
            except Exception:
                continue
        return trades
    
    def get_trades_by_account(self, account_id: str, limit: int = 100) -> List[TradeData]:
        """
        按账户获取交易记录
        
        Args:
            account_id: 账户ID
            limit: 返回数量限制
        
        Returns:
            交易记录列表
        """
        with self._lock:
            try:
                if self._use_postgresql:
                    return self._get_trades_by_account_from_db(account_id, limit)
                else:
                    return self._get_trades_by_account_from_files(account_id, limit)
            except Exception as e:
                logger.error(f"按账户获取交易记录失败: {e}")
                return []
    
    def _get_trades_by_account_from_db(self, account_id: str, limit: int) -> List[TradeData]:
        """从数据库按账户获取交易记录"""
        def _do_get():
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT trade_id, order_id, account_id, symbol, side,
                               quantity, price, amount, commission, stamp_duty,
                               transfer_fee, net_amount, pnl, traded_at, metadata
                        FROM trading_trades 
                        WHERE account_id = %s
                        ORDER BY traded_at DESC
                        LIMIT %s
                    """, (account_id, limit))
                    rows = cur.fetchall()
                    return [
                        TradeData(
                            trade_id=row[0], order_id=row[1], account_id=row[2],
                            symbol=row[3], side=row[4], quantity=float(row[5]),
                            price=float(row[6]), amount=float(row[7]),
                            commission=float(row[8]), stamp_duty=float(row[9]),
                            transfer_fee=float(row[10]), net_amount=float(row[11]),
                            pnl=float(row[12]) if row[12] else None,
                            traded_at=row[13],
                            metadata=row[14] if isinstance(row[14], dict) else json.loads(row[14] or '{}')
                        ) for row in rows
                    ]
        
        return _retry_db_operation(_do_get)
    
    def _get_trades_by_account_from_files(self, account_id: str, limit: int) -> List[TradeData]:
        """从文件按账户获取交易记录"""
        trades = []
        for file_path in self._storage_dir.glob("*.json"):
            try:
                trade = self._get_trade_from_file(file_path.stem)
                if trade and trade.account_id == account_id:
                    trades.append(trade)
                    if len(trades) >= limit:
                        break
            except Exception:
                continue
        trades.sort(key=lambda x: x.traded_at, reverse=True)
        return trades[:limit]
    
    def _update_cache(self, trade: TradeData):
        """更新缓存"""
        if len(self._cache) >= self._max_cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[trade.trade_id] = trade
