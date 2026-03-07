"""
健康检查注册管理器

负责健康检查服务的注册、注销、配置管理等功能。
"""

from typing import Dict, Any, Optional, Callable, Set, TypeVar
from dataclasses import dataclass
from datetime import datetime

from src.infrastructure.logging.core.unified_logger import get_unified_logger

logger = get_unified_logger(__name__)

@dataclass
class HealthCheckInfo:
    """健康检查信息"""
    name: str
    check_func: Callable
    config: Dict[str, Any]
    registered_at: datetime
    last_used: Optional[datetime] = None
    call_count: int = 0
    # Type hint for config
    config: Dict[str, Any]


class HealthCheckRegistry:
    """
    健康检查注册管理器
    
    职责：
    - 管理健康检查服务的注册和注销
    - 存储检查配置信息
    - 提供注册服务查询功能
    - 跟踪服务使用统计
    """
    
    def __init__(self):
        """初始化注册管理器"""
        self._health_checks: Dict[str, HealthCheckInfo] = {}
        self._configs: Dict[str, Dict[str, Any]] = {}
    
    def register_health_check(self, 
                            name: str, 
                            check_func: Callable, 
                            config: Optional[Dict[str, Any]] = None) -> bool:
        """
        注册健康检查服务
        
        Args:
            name: 服务名称
            check_func: 检查函数
            config: 配置参数
            
        Returns:
            是否注册成功
        """
        if not name or not check_func:
            logger.error("服务名称和检查函数不能为空")
            return False
            
        if not callable(check_func):
            logger.error("检查函数必须是可调用的")
            return False
            
        try:
            check_config = config or {}
            
            # 创建健康检查信息
            health_check_info = HealthCheckInfo(
                name=name,
                check_func=check_func,
                config=check_config,
                registered_at=datetime.now()
            )
            
            self._health_checks[name] = health_check_info
            self._configs[name] = check_config.copy()
            
            logger.info(f"健康检查服务已注册: {name}")
            return True
            
        except Exception as e:
            logger.error(f"注册健康检查服务失败: {name} - {e}")
            return False
    
    def unregister_health_check(self, name: str) -> bool:
        """
        注销健康检查服务
        
        Args:
            name: 服务名称
            
        Returns:
            是否注销成功
        """
        if name not in self._health_checks:
            logger.warning(f"要注销的服务不存在: {name}")
            return False
            
        try:
            del self._health_checks[name]
            del self._configs[name]
            logger.info(f"健康检查服务已注销: {name}")
            return True
            
        except Exception as e:
            logger.error(f"注销健康检查服务失败: {name} - {e}")
            return False
    
    def get_health_check(self, name: str) -> Optional[Callable]:
        """
        获取健康检查函数
        
        Args:
            name: 服务名称
            
        Returns:
            检查函数，如果不存在返回None
        """
        if name not in self._health_checks:
            return None
            
        # 更新使用统计
        self._health_checks[name].last_used = datetime.now()
        self._health_checks[name].call_count += 1
        
        return self._health_checks[name].check_func
    
    def get_health_check_config(self, name: str) -> Optional[Dict[str, Any]]:
        """
        获取健康检查配置
        
        Args:
            name: 服务名称
            
        Returns:
            配置字典，如果不存在返回None
        """
        return self._configs.get(name)
    
    def update_health_check_config(self, name: str, config: Dict[str, Any]) -> bool:
        """
        更新健康检查配置
        
        Args:
            name: 服务名称
            config: 新的配置
            
        Returns:
            是否更新成功
        """
        if name not in self._health_checks:
            logger.error(f"要更新的服务不存在: {name}")
            return False
            
        try:
            self._configs[name] = config.copy()
            self._health_checks[name].config = config.copy()
            logger.info(f"健康检查配置已更新: {name}")
            return True
            
        except Exception as e:
            logger.error(f"更新健康检查配置失败: {name} - {e}")
            return False
    
    def get_registered_services(self) -> Set[str]:
        """
        获取所有已注册的服务名称
        
        Returns:
            服务名称集合
        """
        return set(self._health_checks.keys())
    
    def is_service_registered(self, name: str) -> bool:
        """
        检查服务是否已注册
        
        Args:
            name: 服务名称
            
        Returns:
            是否已注册
        """
        return name in self._health_checks
    
    def get_registered_count(self) -> int:
        """获取已注册服务数量"""
        return len(self._health_checks)
    
    def get_service_info(self, name: str) -> Optional[HealthCheckInfo]:
        """
        获取服务详细信息
        
        Args:
            name: 服务名称
            
        Returns:
            服务信息，如果不存在返回None
        """
        return self._health_checks.get(name)
    
    def get_all_services_info(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有服务的统计信息
        
        Returns:
            服务统计信息字典
        """
        services_info = {}
        
        for name, info in self._health_checks.items():
            services_info[name] = {
                'name': info.name,
                'registered_at': info.registered_at.isoformat(),
                'last_used': info.last_used.isoformat() if info.last_used else None,
                'call_count': info.call_count,
                'has_config': bool(info.config)
            }
            
        return services_info
    
    def clear_all_services(self) -> bool:
        """
        清除所有注册的服务
        
        Returns:
            是否清除成功
        """
        try:
            service_count = len(self._health_checks)
            self._health_checks.clear()
            self._configs.clear()
            logger.info(f"已清除 {service_count} 个注册的 health check 服务")
            return True
            
        except Exception as e:
            logger.error(f"清除服务失败: {e}")
            return False
    
    def get_services_health_summary(self) -> Dict[str, Any]:
        """
        获取服务健康摘要信息
        
        Returns:
            摘要信息
        """
        total_services = len(self._health_checks)
        
        if total_services == 0:
            return {
                'total_services': 0,
                'active_services': 0,
                'message': '没有注册的健康检查服务'
            }
        
        # 计算最近使用的服务数量
        recently_used_count = 0
        total_calls = 0
        
        for info in self._health_checks.values():
            if info.last_used and (datetime.now() - info.last_used).total_seconds() < 1 * 60 * 60:  # 1小时内
                recently_used_count += 1
            total_calls += info.call_count
        
        return {
            'total_services': total_services,
            'active_services': recently_used_count,
            'total_calls': total_calls,
            'average_calls_per_service': round(total_calls / total_services, 2) if total_services > 0 else 0
        }

