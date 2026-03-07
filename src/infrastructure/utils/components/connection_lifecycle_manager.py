"""
连接生命周期管理器组件

负责连接的创建、销毁、清理和维护。
"""

import logging
import uuid
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConnectionInfo:
    """连接信息"""
    connection_id: str
    created_at: datetime
    last_used: datetime
    use_count: int
    is_active: bool
    error_count: int = 0
    last_error: Optional[str] = None
    connection: Optional[Any] = None


class ConnectionLifecycleManager:
    """连接生命周期管理器"""
    
    def __init__(self, connection_factory: Optional[Callable] = None,
                 idle_timeout: float = 300.0, max_lifetime: float = 3600.0,
                 max_usage: Optional[int] = None):
        """
        初始化生命周期管理器
        
        Args:
            connection_factory: 连接工厂函数
            idle_timeout: 空闲超时时间（秒）
            max_lifetime: 最大生命周期（秒）
            max_usage: 最大使用次数
        """
        self.connection_factory = connection_factory
        self.idle_timeout = idle_timeout
        self.max_lifetime = max_lifetime
        self.max_usage = max_usage
    
    def create_connection(self) -> Optional[ConnectionInfo]:
        """
        创建新连接
        
        Returns:
            连接信息对象
        """
        if not self.connection_factory:
            logger.error("连接工厂未设置，无法创建连接")
            return None
        
        try:
            # 调用工厂函数创建连接
            connection = self.connection_factory()
            
            if connection is None:
                logger.error("连接工厂返回None")
                return None
            
            # 创建连接信息
            connection_info = ConnectionInfo(
                connection_id=str(uuid.uuid4()),
                created_at=datetime.now(),
                last_used=datetime.now(),
                use_count=0,
                is_active=False,
                connection=connection
            )
            
            logger.debug(f"创建新连接: {connection_info.connection_id}")
            return connection_info
            
        except Exception as e:
            logger.error(f"创建连接失败: {e}")
            return None
    
    def destroy_connection(self, connection_info: ConnectionInfo) -> bool:
        """
        销毁连接
        
        Args:
            connection_info: 连接信息
            
        Returns:
            是否成功销毁
        """
        try:
            if connection_info.connection:
                # 尝试关闭连接
                if hasattr(connection_info.connection, 'close'):
                    connection_info.connection.close()
                
                logger.debug(f"销毁连接: {connection_info.connection_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"销毁连接失败: {e}")
            return False
    
    def cleanup_expired_connections(self, connections: List) -> List[ConnectionInfo]:
        """
        清理过期连接
        
        Args:
            connections: 连接列表
            
        Returns:
            需要删除的连接列表
        """
        expired_connections = []
        current_time = datetime.now()
        
        for conn_info in connections:
            if self._is_connection_expired(conn_info, current_time):
                expired_connections.append(conn_info)
        
        return expired_connections
    
    def ensure_min_connections(self, connections: List, min_size: int) -> int:
        """
        确保最小连接数
        
        Args:
            connections: 当前连接列表
            min_size: 最小连接数
            
        Returns:
            需要创建的连接数
        """
        current_size = len(connections)
        if current_size < min_size:
            return min_size - current_size
        return 0
    
    def _is_connection_expired(self, connection_info: ConnectionInfo, 
                               current_time: datetime) -> bool:
        """检查连接是否过期"""
        # 检查空闲超时
        if not connection_info.is_active:
            idle_time = (current_time - connection_info.last_used).total_seconds()
            if idle_time > self.idle_timeout:
                return True
        
        # 检查最大生命周期
        lifetime = (current_time - connection_info.created_at).total_seconds()
        if lifetime > self.max_lifetime:
            return True
        
        # 检查最大使用次数
        if self.max_usage and connection_info.use_count >= self.max_usage:
            return True
        
        return False
    
    def update_connection_usage(self, connection_info: ConnectionInfo) -> None:
        """更新连接使用信息"""
        connection_info.use_count += 1
        connection_info.last_used = datetime.now()
        connection_info.is_active = True
    
    def mark_connection_released(self, connection_info: ConnectionInfo) -> None:
        """标记连接已释放"""
        connection_info.is_active = False
        connection_info.last_used = datetime.now()

