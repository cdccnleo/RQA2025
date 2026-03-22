# -*- coding: utf-8 -*-
"""
投资组合管理器模块
支持PostgreSQL优先存储策略，连接失败时自动降级到内存存储
"""

import logging
import threading
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class PositionInfo:
    """持仓信息数据类"""
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


class PortfolioManager:
    """
    投资组合管理器 - PostgreSQL持久化版本
    
    支持PostgreSQL优先存储策略，连接失败时自动降级到内存存储。
    所有持仓操作会同步持久化到数据库，确保数据安全。
    """
    
    def __init__(self, account_id: str = "default", enable_persistence: bool = True):
        """
        初始化投资组合管理器
        
        Args:
            account_id: 账户ID
            enable_persistence: 是否启用持久化（默认True）
        """
        self._account_id = account_id
        self._positions: Dict[str, PositionInfo] = {}
        self._cash: float = 0.0
        self._lock = threading.RLock()
        self._enable_persistence = enable_persistence
        self._persistence = None
        self._PositionData = None
        
        if enable_persistence:
            self._init_persistence()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"投资组合管理器初始化完成，账户: {account_id}，持久化: {enable_persistence}")
    
    def _init_persistence(self):
        """
        初始化持久化层
        
        尝试初始化PostgreSQL持久化，失败则禁用持久化功能
        """
        try:
            from ..persistence.trading_persistence import PositionPersistence, PositionData
            self._persistence = PositionPersistence()
            self._PositionData = PositionData
            self.logger.info("持仓持久化层初始化成功")
        except Exception as e:
            self.logger.warning(f"持仓持久化层初始化失败: {e}，将使用纯内存模式")
            self._enable_persistence = False
            self._persistence = None
    
    def _generate_position_id(self, symbol: str) -> str:
        """
        生成持仓ID
        
        Args:
            symbol: 标的代码
        
        Returns:
            持仓ID
        """
        return f"{self._account_id}_{symbol}"
    
    def _save_position_to_persistence(self, position: PositionInfo) -> bool:
        """
        保存持仓到持久化层
        
        Args:
            position: 持仓信息
        
        Returns:
            是否保存成功
        """
        if not self._enable_persistence or not self._persistence:
            return True
        
        try:
            position_data = self._PositionData(
                position_id=position.position_id,
                account_id=position.account_id,
                symbol=position.symbol,
                quantity=position.quantity,
                available_quantity=position.available_quantity,
                frozen_quantity=position.frozen_quantity,
                avg_cost=position.avg_cost,
                current_price=position.current_price,
                market_value=position.market_value,
                unrealized_pnl=position.unrealized_pnl,
                realized_pnl=position.realized_pnl,
                created_at=position.created_at,
                updated_at=position.updated_at,
                metadata=position.metadata
            )
            return self._persistence.save_position(position_data)
        except Exception as e:
            self.logger.error(f"保存持仓到持久化层失败: {e}")
            return False
    
    def _update_position_in_persistence(self, position_id: str, updates: Dict[str, Any]) -> bool:
        """
        更新持久化层中的持仓
        
        Args:
            position_id: 持仓ID
            updates: 更新字段字典
        
        Returns:
            是否更新成功
        """
        if not self._enable_persistence or not self._persistence:
            return True
        
        try:
            return self._persistence.update_position(position_id, updates)
        except Exception as e:
            self.logger.error(f"更新持久化层持仓失败: {e}")
            return False
    
    def _load_position_from_persistence(self, position_id: str) -> Optional[PositionInfo]:
        """
        从持久化层加载持仓
        
        Args:
            position_id: 持仓ID
        
        Returns:
            持仓信息，不存在返回None
        """
        if not self._enable_persistence or not self._persistence:
            return None
        
        try:
            position_data = self._persistence.get_position(position_id)
            if position_data:
                return PositionInfo(
                    position_id=position_data.position_id,
                    account_id=position_data.account_id,
                    symbol=position_data.symbol,
                    quantity=position_data.quantity,
                    available_quantity=position_data.available_quantity,
                    frozen_quantity=position_data.frozen_quantity,
                    avg_cost=position_data.avg_cost,
                    current_price=position_data.current_price,
                    market_value=position_data.market_value,
                    unrealized_pnl=position_data.unrealized_pnl,
                    realized_pnl=position_data.realized_pnl,
                    created_at=position_data.created_at,
                    updated_at=position_data.updated_at,
                    metadata=position_data.metadata
                )
        except Exception as e:
            self.logger.error(f"从持久化层加载持仓失败: {e}")
        return None
    
    def restore_from_persistence(self) -> int:
        """
        从持久化层恢复所有持仓
        
        Returns:
            恢复的持仓数量
        """
        if not self._enable_persistence or not self._persistence:
            return 0
        
        restored_count = 0
        try:
            positions = self._persistence.get_positions_by_account(self._account_id)
            for position_data in positions:
                if position_data.symbol not in self._positions:
                    position = PositionInfo(
                        position_id=position_data.position_id,
                        account_id=position_data.account_id,
                        symbol=position_data.symbol,
                        quantity=position_data.quantity,
                        available_quantity=position_data.available_quantity,
                        frozen_quantity=position_data.frozen_quantity,
                        avg_cost=position_data.avg_cost,
                        current_price=position_data.current_price,
                        market_value=position_data.market_value,
                        unrealized_pnl=position_data.unrealized_pnl,
                        realized_pnl=position_data.realized_pnl,
                        created_at=position_data.created_at,
                        updated_at=position_data.updated_at,
                        metadata=position_data.metadata
                    )
                    self._positions[position_data.symbol] = position
                    restored_count += 1
            
            self.logger.info(f"从持久化层恢复了 {restored_count} 个持仓")
        except Exception as e:
            self.logger.error(f"从持久化层恢复持仓失败: {e}")
        
        return restored_count

    @property
    def positions(self) -> Dict[str, Dict[str, Any]]:
        """
        获取持仓字典（兼容旧接口）
        
        Returns:
            持仓字典
        """
        with self._lock:
            return {
                symbol: {
                    'quantity': pos.quantity,
                    'avg_price': pos.avg_cost,
                    'available_quantity': pos.available_quantity,
                    'frozen_quantity': pos.frozen_quantity,
                    'current_price': pos.current_price,
                    'market_value': pos.market_value,
                    'unrealized_pnl': pos.unrealized_pnl
                }
                for symbol, pos in self._positions.items()
            }
    
    @property
    def cash(self) -> float:
        """
        获取现金
        
        Returns:
            现金余额
        """
        return self._cash
    
    @cash.setter
    def cash(self, value: float):
        """
        设置现金
        
        Args:
            value: 现金余额
        """
        with self._lock:
            self._cash = value
    
    def add_position(self, symbol: str, quantity: float, price: float) -> bool:
        """
        添加持仓
        
        Args:
            symbol: 标的代码
            quantity: 数量
            price: 价格
        
        Returns:
            是否成功添加
        """
        with self._lock:
            try:
                if symbol not in self._positions:
                    position_id = self._generate_position_id(symbol)
                    now = datetime.now()
                    position = PositionInfo(
                        position_id=position_id,
                        account_id=self._account_id,
                        symbol=symbol,
                        quantity=quantity,
                        available_quantity=quantity,
                        avg_cost=price,
                        current_price=price,
                        market_value=quantity * price,
                        created_at=now,
                        updated_at=now
                    )
                    self._positions[symbol] = position
                else:
                    position = self._positions[symbol]
                    total_quantity = position.quantity + quantity
                    total_cost = position.quantity * position.avg_cost + quantity * price
                    
                    position.quantity = total_quantity
                    position.available_quantity += quantity
                    position.avg_cost = total_cost / total_quantity if total_quantity > 0 else 0
                    position.market_value = total_quantity * position.current_price
                    position.updated_at = datetime.now()
                
                # 持久化持仓
                self._save_position_to_persistence(position)
                
                self.logger.info(f"添加持仓: {symbol} 数量={quantity} 价格={price}")
                return True
                
            except Exception as e:
                self.logger.error(f"添加持仓失败: {e}")
                return False
    
    def remove_position(self, symbol: str, quantity: Optional[float] = None) -> bool:
        """
        移除持仓
        
        Args:
            symbol: 标的代码
            quantity: 移除数量（None表示全部移除）
        
        Returns:
            是否成功移除
        """
        with self._lock:
            try:
                if symbol not in self._positions:
                    return False
                
                position = self._positions[symbol]
                
                if quantity is None or quantity >= position.quantity:
                    # 全部移除
                    del self._positions[symbol]
                    # 更新持久化层
                    if self._enable_persistence and self._persistence:
                        self._persistence.delete_position(position.position_id)
                else:
                    # 部分移除
                    position.quantity -= quantity
                    position.available_quantity = max(0, position.available_quantity - quantity)
                    position.market_value = position.quantity * position.current_price
                    position.updated_at = datetime.now()
                    # 持久化更新
                    self._update_position_in_persistence(position.position_id, {
                        'quantity': position.quantity,
                        'available_quantity': position.available_quantity,
                        'market_value': position.market_value,
                        'updated_at': position.updated_at
                    })
                
                self.logger.info(f"移除持仓: {symbol} 数量={quantity}")
                return True
                
            except Exception as e:
                self.logger.error(f"移除持仓失败: {e}")
                return False
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取持仓信息
        
        Args:
            symbol: 标的代码
        
        Returns:
            持仓信息字典，不存在返回None
        """
        with self._lock:
            position = self._positions.get(symbol)
            if position is None and self._enable_persistence:
                position_id = self._generate_position_id(symbol)
                position = self._load_position_from_persistence(position_id)
                if position:
                    self._positions[symbol] = position
            
            if position:
                return {
                    'quantity': position.quantity,
                    'avg_price': position.avg_cost,
                    'available_quantity': position.available_quantity,
                    'frozen_quantity': position.frozen_quantity,
                    'current_price': position.current_price,
                    'market_value': position.market_value,
                    'unrealized_pnl': position.unrealized_pnl
                }
            return None
    
    def update_position_price(self, symbol: str, new_price: float) -> bool:
        """
        更新持仓价格
        
        Args:
            symbol: 标的代码
            new_price: 新价格
        
        Returns:
            是否成功更新
        """
        with self._lock:
            try:
                if symbol not in self._positions:
                    return False
                
                position = self._positions[symbol]
                old_market_value = position.market_value
                
                position.current_price = new_price
                position.market_value = position.quantity * new_price
                position.unrealized_pnl = position.market_value - position.quantity * position.avg_cost
                position.updated_at = datetime.now()
                
                # 持久化更新
                self._update_position_in_persistence(position.position_id, {
                    'current_price': new_price,
                    'market_value': position.market_value,
                    'unrealized_pnl': position.unrealized_pnl,
                    'updated_at': position.updated_at
                })
                
                return True
                
            except Exception as e:
                self.logger.error(f"更新持仓价格失败: {e}")
                return False
    
    def freeze_position(self, symbol: str, quantity: float) -> bool:
        """
        冻结持仓
        
        Args:
            symbol: 标的代码
            quantity: 冻结数量
        
        Returns:
            是否成功冻结
        """
        with self._lock:
            try:
                if symbol not in self._positions:
                    return False
                
                position = self._positions[symbol]
                
                if quantity > position.available_quantity:
                    return False
                
                position.frozen_quantity += quantity
                position.available_quantity -= quantity
                position.updated_at = datetime.now()
                
                # 持久化更新
                self._update_position_in_persistence(position.position_id, {
                    'frozen_quantity': position.frozen_quantity,
                    'available_quantity': position.available_quantity,
                    'updated_at': position.updated_at
                })
                
                return True
                
            except Exception as e:
                self.logger.error(f"冻结持仓失败: {e}")
                return False
    
    def unfreeze_position(self, symbol: str, quantity: float) -> bool:
        """
        解冻持仓
        
        Args:
            symbol: 标的代码
            quantity: 解冻数量
        
        Returns:
            是否成功解冻
        """
        with self._lock:
            try:
                if symbol not in self._positions:
                    return False
                
                position = self._positions[symbol]
                
                if quantity > position.frozen_quantity:
                    return False
                
                position.frozen_quantity -= quantity
                position.available_quantity += quantity
                position.updated_at = datetime.now()
                
                # 持久化更新
                self._update_position_in_persistence(position.position_id, {
                    'frozen_quantity': position.frozen_quantity,
                    'available_quantity': position.available_quantity,
                    'updated_at': position.updated_at
                })
                
                return True
                
            except Exception as e:
                self.logger.error(f"解冻持仓失败: {e}")
                return False
    
    def get_portfolio_value(self, current_prices: Optional[Dict[str, float]] = None) -> float:
        """
        获取投资组合价值
        
        Args:
            current_prices: 当前价格字典（可选）
        
        Returns:
            投资组合总价值
        """
        with self._lock:
            total = self._cash
            
            for symbol, position in self._positions.items():
                if current_prices and symbol in current_prices:
                    price = current_prices[symbol]
                    total += position.quantity * price
                else:
                    total += position.market_value
            
            return total
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        获取投资组合摘要
        
        Returns:
            投资组合摘要信息
        """
        with self._lock:
            total_value = self.get_portfolio_value()
            total_cost = sum(pos.quantity * pos.avg_cost for pos in self._positions.values())
            total_pnl = total_value - total_cost - self._cash
            
            return {
                'account_id': self._account_id,
                'cash': self._cash,
                'positions_count': len(self._positions),
                'total_value': total_value,
                'total_cost': total_cost,
                'total_pnl': total_pnl,
                'positions': self.positions
            }


__all__ = ['PortfolioManager', 'PositionInfo']
