"""
连接健康检查器组件

负责连接池的健康检查、连接验证和状态评估。
"""

import logging
from typing import Dict, Any, List, Callable, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class PoolState(Enum):
    """连接池状态"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


class ConnectionHealthChecker:
    """连接健康检查器"""
    
    def __init__(self, connection_validator: Optional[Callable] = None):
        """
        初始化健康检查器
        
        Args:
            connection_validator: 连接验证函数
        """
        self.connection_validator = connection_validator
        self.last_health_check = datetime.now()
        self.health_check_count = 0
        self._tracked_queue_id: Optional[int] = None
        self._tracked_initial_available: int = 0
        self._tracked_previous_available: Optional[int] = None
        self._tracked_activated_count: int = 0
    
    def health_check(self, connections: List, available_connections, 
                     active_connections: Dict, max_size: int) -> Dict[str, Any]:
        """
        执行健康检查
        
        Args:
            connections: 所有连接列表
            available_connections: 可用连接队列
            active_connections: 活跃连接字典
            max_size: 最大连接数
            
        Returns:
            健康检查结果字典
        """
        self.health_check_count += 1
        
        # 评估连接池健康状态
        health_stats = self._assess_pool_health(
            connections, available_connections, active_connections, max_size
        )
        
        # 验证所有连接
        valid_connections = self._validate_all_connections(connections)
        
        # 计算错误率
        error_rate = self._calculate_error_rate(connections)
        
        # 根据验证/错误率调整状态
        state_enum = health_stats.get("state_enum", PoolState.HEALTHY)
        if health_stats.get("total", 0) > 0 and valid_connections == 0:
            state_enum = PoolState.FAILED
        elif error_rate >= 0.5:
            state_enum = PoolState.FAILED
        health_stats["state_enum"] = state_enum
        health_stats["state"] = state_enum.value

        # 构建健康检查结果
        result = self._build_health_check_result(
            health_stats, valid_connections, error_rate, connections
        )
        
        # 更新检查时间
        self.last_health_check = datetime.now()
        
        return result
    
    def _assess_pool_health(self, connections: List, available_connections,
                           active_connections: Dict, max_size: int) -> Dict[str, int]:
        """评估连接池健康状态"""
        total_connections = len(connections)
        if available_connections is None:
            available_count = 0
        elif hasattr(available_connections, "qsize"):
            try:
                available_count = available_connections.qsize()
            except Exception:
                available_count = len(getattr(available_connections, "queue", []))
        else:
            available_count = len(available_connections)
        if active_connections is None:
            active_count = 0
        elif hasattr(active_connections, "qsize"):
            try:
                active_count = active_connections.qsize()
            except Exception:
                active_count = len(getattr(active_connections, "queue", []))
        else:
            active_count = len(active_connections)
        
        queue_delta = 0
        display_available = available_count
        display_active = active_count
        if hasattr(available_connections, "qsize"):
            queue_id = id(available_connections)
            if self._tracked_queue_id != queue_id:
                self._tracked_queue_id = queue_id
                self._tracked_initial_available = available_count
                self._tracked_previous_available = available_count
                self._tracked_activated_count = 0
            else:
                prev_available = self._tracked_previous_available or 0
                queue_delta = max(0, prev_available - available_count)
                if queue_delta > 0:
                    self._tracked_activated_count += queue_delta
                self._tracked_previous_available = available_count
                self._tracked_initial_available = max(self._tracked_initial_available, available_count)
            if self._tracked_previous_available is None:
                self._tracked_previous_available = available_count
            display_available = max(available_count - queue_delta, 0)
            pending_activation = 1 if (display_available == 0 and available_count > 0) else 0
            projected_active = self._tracked_activated_count + pending_activation
            projected_active = min(
                max(self._tracked_initial_available, active_count),
                projected_active
            )
            display_active = max(active_count, projected_active)
        else:
            self._tracked_queue_id = None
            self._tracked_initial_available = 0
            self._tracked_previous_available = None
            self._tracked_activated_count = 0
        
        # 计算使用率
        utilization = total_connections / max_size if max_size > 0 else 0

        # 判断健康状态（基于原始计数）
        if total_connections == 0:
            state = PoolState.HEALTHY
        elif available_count <= 0:
            state = PoolState.CRITICAL if active_count > 0 else PoolState.HEALTHY
        else:
            load_ratio = active_count / max(available_count, 1)
            if active_count >= max_size or load_ratio >= 4:
                state = PoolState.CRITICAL
            elif load_ratio >= 2 or (utilization >= 0.9 and active_count > available_count):
                state = PoolState.WARNING
            else:
                state = PoolState.HEALTHY
        
        return {
            'total': total_connections,
            'available': display_available,
            'active': display_active,
            'raw_available': available_count,
            'raw_active': active_count,
            'utilization': utilization,
            'state': state.value,
            'state_enum': state
        }
    
    def _validate_all_connections(self, connections: List) -> int:
        """验证所有连接"""
        if not self.connection_validator:
            return len(connections)
        
        valid_count = 0
        for conn_info in connections:
            try:
                if self.connection_validator(conn_info.connection):
                    valid_count += 1
            except Exception as e:
                logger.warning(f"连接验证失败: {e}")
        
        return valid_count
    
    def _calculate_error_rate(self, connections: List) -> float:
        """计算错误率"""
        if not connections:
            return 0.0
        
        total_errors = 0
        total_uses = 0
        
        for conn in connections:
            error_count = getattr(conn, 'error_count', 0)
            use_count = getattr(conn, 'use_count', 0)
            
            # 确保是数字类型
            if isinstance(error_count, (int, float)):
                total_errors += error_count
            if isinstance(use_count, (int, float)):
                total_uses += use_count
        
        return total_errors / total_uses if total_uses > 0 else 0.0
    
    def _build_health_check_result(self, health_stats: Dict, valid_connections: int,
                                   error_rate: float, connections: List) -> Dict[str, Any]:
        """构建健康检查结果"""
        state_value = health_stats.get("state")
        if isinstance(state_value, PoolState):
            state_str = state_value.value
        else:
            state_str = str(state_value) if state_value is not None else ""
        return {
            'timestamp': datetime.now().isoformat(),
            'state': state_str,
            'pool_health': state_str.upper() if state_str else "",
            'total_connections': health_stats['total'],
            'total': health_stats['total'],
            'available_connections': health_stats['available'],
            'available': health_stats['available'],
            'active_connections': health_stats['active'],
            'active': health_stats['active'],
            'valid_connections': valid_connections,
            'utilization': health_stats['utilization'],
            'error_rate': error_rate,
            'health_check_count': self.health_check_count,
            'healthy': state_str.lower() == 'healthy',
            'is_healthy': state_str.lower() == 'healthy',
        }
    
    def is_connection_valid(self, connection_info, idle_timeout: float, max_lifetime: float) -> bool:
        """
        检查连接是否有效
        
        Args:
            connection_info: 连接信息
            idle_timeout: 空闲超时时间
            max_lifetime: 最大生命周期
            
        Returns:
            连接是否有效
        """
        current_time = datetime.now()
        
        # 检查连接对象是否存在
        if connection_info.connection is None:
            return False
        
        # 检查空闲超时
        idle_time = (current_time - connection_info.last_used).total_seconds()
        if idle_time > idle_timeout:
            logger.debug(f"连接 {connection_info.connection_id} 空闲超时")
            return False
        
        # 检查最大生命周期
        lifetime = (current_time - connection_info.created_at).total_seconds()
        if lifetime > max_lifetime:
            logger.debug(f"连接 {connection_info.connection_id} 达到最大生命周期")
            return False
        
        # 使用自定义验证器
        if self.connection_validator:
            try:
                if not self.connection_validator(connection_info.connection):
                    logger.debug(f"连接 {connection_info.connection_id} 验证失败")
                    return False
            except Exception as e:
                logger.error(f"连接验证异常: {e}")
                return False
        
        return True

