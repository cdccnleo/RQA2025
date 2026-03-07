"""负载均衡器模块"""

import logging
import threading
from typing import List, Optional
from ..gateway_types import ServiceEndpoint, ServiceStatus

logger = logging.getLogger(__name__)


class LoadBalancer:
    """负载均衡器
    
    支持多种负载均衡算法：
    - round_robin: 轮询
    - weighted: 加权轮询
    - random: 随机选择
    """
    
    def __init__(self, algorithm: str = "round_robin"):
        """初始化负载均衡器
        
        Args:
            algorithm: 负载均衡算法 ('round_robin', 'weighted', 'random')
        """
        self.algorithm = algorithm
        self.endpoints: List[ServiceEndpoint] = []
        self.current_index = 0
        self.lock = threading.Lock()
    
    def add_endpoint(self, endpoint: ServiceEndpoint):
        """添加服务端点
        
        Args:
            endpoint: 服务端点对象
        """
        self.endpoints.append(endpoint)
        logger.info(f"Added endpoint: {endpoint.service_name} - {endpoint.upstream_url}")
    
    def select_endpoint(self) -> Optional[ServiceEndpoint]:
        """选择服务端点（别名方法）
        
        Returns:
            选中的服务端点，如果没有可用端点返回None
        """
        return self.get_endpoint()
    
    def get_endpoint(self) -> Optional[ServiceEndpoint]:
        """获取服务端点
        
        Returns:
            根据负载均衡算法选择的端点
        """
        if not self.endpoints:
            return None
        
        # 过滤健康端点
        healthy_endpoints = [ep for ep in self.endpoints if ep.status == ServiceStatus.HEALTHY]
        if not healthy_endpoints:
            logger.warning("No healthy endpoints available")
            return None
        
        with self.lock:
            if self.algorithm == "round_robin":
                endpoint = healthy_endpoints[self.current_index % len(healthy_endpoints)]
                self.current_index += 1
                return endpoint
            elif self.algorithm == "weighted":
                return self._weighted_selection(healthy_endpoints)
            elif self.algorithm == "random":
                import secrets
                return secrets.choice(healthy_endpoints)
            else:
                return healthy_endpoints[0]
    
    def _weighted_selection(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """加权选择端点
        
        Args:
            endpoints: 可用端点列表
            
        Returns:
            根据权重选择的端点
        """
        total_weight = sum(ep.weight for ep in endpoints)
        if total_weight == 0:
            return endpoints[0]
        
        import secrets
        r = secrets.randbelow(total_weight)
        current_weight = 0
        
        for endpoint in endpoints:
            current_weight += endpoint.weight
            if r < current_weight:
                return endpoint
        
        return endpoints[-1]
    
    def update_endpoint_health(self, endpoint: ServiceEndpoint, is_healthy: bool):
        """更新端点健康状态
        
        Args:
            endpoint: 要更新的端点
            is_healthy: 是否健康
        """
        with self.lock:
            if is_healthy:
                endpoint.status = ServiceStatus.HEALTHY
                endpoint.success_count += 1
            else:
                endpoint.status = ServiceStatus.UNHEALTHY
                endpoint.failure_count += 1
    
    def get_stats(self) -> dict:
        """获取负载均衡统计信息
        
        Returns:
            包含统计信息的字典
        """
        with self.lock:
            total_endpoints = len(self.endpoints)
            healthy_endpoints = len([ep for ep in self.endpoints if ep.status == ServiceStatus.HEALTHY])
            
            return {
                'algorithm': self.algorithm,
                'total_endpoints': total_endpoints,
                'healthy_endpoints': healthy_endpoints,
                'current_index': self.current_index
            }


__all__ = ['LoadBalancer']

