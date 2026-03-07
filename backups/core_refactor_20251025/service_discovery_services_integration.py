"""
RQA2025 服务发现和注册模块

提供服务自动发现、注册和负载均衡功能
"""

import logging
import socket
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import threading
import requests
from concurrent.futures import ThreadPoolExecutor

from .service_communicator import ServiceEndpoint, get_service_communicator

logger = logging.getLogger(__name__)


@dataclass
class ServiceInstance:

    """服务实例"""
    service_name: str
    instance_id: str
    host: str
    port: int
    protocol: str = "http"
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    status: str = "UP"  # UP, DOWN, STARTING, OUT_OF_SERVICE

    @property
    def endpoint(self) -> ServiceEndpoint:
        """获取端点"""
        return ServiceEndpoint(
            name=f"{self.service_name}-{self.instance_id}",
            host=self.host,
            port=self.port,
            protocol=self.protocol
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'service_name': self.service_name,
            'instance_id': self.instance_id,
            'host': self.host,
            'port': self.port,
            'protocol': self.protocol,
            'metadata': self.metadata,
            'registered_at': self.registered_at.isoformat(),
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'status': self.status
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ServiceInstance':
        """从字典创建实例"""
        return cls(
            service_name=data['service_name'],
            instance_id=data['instance_id'],
            host=data['host'],
            port=data['port'],
            protocol=data.get('protocol', 'http'),
            metadata=data.get('metadata', {}),
            registered_at=datetime.fromisoformat(data['registered_at']),
            last_heartbeat=datetime.fromisoformat(data['last_heartbeat']),
            status=data.get('status', 'UP')
        )


class LoadBalancer:

    """负载均衡器"""

    def __init__(self, strategy: str = "round_robin"):

        self.strategy = strategy
        self.instances: Dict[str, List[ServiceInstance]] = {}
        self.current_index: Dict[str, int] = {}
        self.lock = threading.RLock()

    def add_instance(self, instance: ServiceInstance):
        """添加服务实例"""
        with self.lock:
            if instance.service_name not in self.instances:
                self.instances[instance.service_name] = []
                self.current_index[instance.service_name] = 0

            self.instances[instance.service_name].append(instance)
            logger.info(f"服务实例已添加: {instance.service_name} - {instance.instance_id}")

    def remove_instance(self, service_name: str, instance_id: str):
        """移除服务实例"""
        with self.lock:
            if service_name in self.instances:
                self.instances[service_name] = [
                    inst for inst in self.instances[service_name]
                    if inst.instance_id != instance_id
                ]

                if not self.instances[service_name]:
                    del self.instances[service_name]
                    del self.current_index[service_name]

                logger.info(f"服务实例已移除: {service_name} - {instance_id}")

    def get_instance(self, service_name: str) -> Optional[ServiceInstance]:
        """获取服务实例（负载均衡）"""
        with self.lock:
            if service_name not in self.instances:
                return None

            instances = [inst for inst in self.instances[service_name]
                         if inst.status == "UP"]

            if not instances:
                return None

            if self.strategy == "round_robin":
                return self._round_robin(service_name, instances)
            elif self.strategy == "random":
                return self._random(instances)
            elif self.strategy == "least_loaded":
                return self._least_loaded(instances)
            else:
                return instances[0]  # 默认返回第一个

    def _round_robin(self, service_name: str, instances: List[ServiceInstance]) -> ServiceInstance:
        """轮询负载均衡"""
        current = self.current_index[service_name]
        instance = instances[current % len(instances)]
        self.current_index[service_name] = (current + 1) % len(instances)
        return instance

    def _random(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """随机负载均衡"""
        import secrets
        return secrets.choice(instances)

    def _least_loaded(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """最少负载均衡（基于元数据）"""
        # 这里可以基于实例的负载信息进行选择
        # 目前简化为随机选择
        import secrets
        return secrets.choice(instances)

    def get_all_instances(self, service_name: str) -> List[ServiceInstance]:
        """获取服务的所有实例"""
        with self.lock:
            return self.instances.get(service_name, []).copy()

    def get_service_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        with self.lock:
            stats = {}
            for service_name, instances in self.instances.items():
                up_instances = [inst for inst in instances if inst.status == "UP"]
                stats[service_name] = {
                    'total_instances': len(instances),
                    'up_instances': len(up_instances),
                    'down_instances': len(instances) - len(up_instances)
                }
            return stats


class ServiceDiscoveryClient:

    """服务发现客户端"""

    def __init__(self, discovery_server_url: Optional[str] = None):

        self.discovery_server_url = discovery_server_url or "http://localhost:8761"
        self.load_balancer = LoadBalancer()
        self.local_services: Dict[str, ServiceInstance] = {}
        self.executor = ThreadPoolExecutor(max_workers=5)
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._running = False
        self.heartbeat_interval = 30  # 心跳间隔(秒)

    def register_service(self, service_name: str, port: int,


                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        注册服务

        Args:
            service_name: 服务名称
            port: 服务端口
            metadata: 服务元数据

        Returns:
            实例ID
        """
        import uuid

        # 获取本机IP
        host = self._get_local_ip()

        # 创建服务实例
        instance_id = str(uuid.uuid4())
        instance = ServiceInstance(
            service_name=service_name,
            instance_id=instance_id,
            host=host,
            port=port,
            metadata=metadata or {}
        )

        # 保存本地服务
        self.local_services[instance_id] = instance

        # 添加到负载均衡器
        self.load_balancer.add_instance(instance)

        # 向通信器注册
        communicator = get_service_communicator()
        communicator.register_service(instance.endpoint)

        logger.info(f"服务已注册: {service_name} - {instance_id} ({host}:{port})")

        # 尝试向发现服务器注册
        self._register_with_discovery_server(instance)

        return instance_id

    def unregister_service(self, instance_id: str):
        """注销服务"""
        if instance_id in self.local_services:
            instance = self.local_services[instance_id]

            # 从负载均衡器移除
            self.load_balancer.remove_instance(instance.service_name, instance_id)

            # 从通信器注销
            communicator = get_service_communicator()
            communicator.unregister_service(instance.endpoint.name)

            # 从发现服务器注销
            self._unregister_from_discovery_server(instance)

            # 从本地服务移除
            del self.local_services[instance_id]

            logger.info(f"服务已注销: {instance.service_name} - {instance_id}")

    def discover_service(self, service_name: str) -> Optional[ServiceInstance]:
        """发现服务"""
        return self.load_balancer.get_instance(service_name)

    def get_service_instances(self, service_name: str) -> List[ServiceInstance]:
        """获取服务的所有实例"""
        return self.load_balancer.get_all_instances(service_name)

    def start_heartbeat(self):
        """启动心跳"""
        if self._running:
            return

        self._running = True
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True
        )
        self._heartbeat_thread.start()
        logger.info("服务心跳已启动")

    def stop_heartbeat(self):
        """停止心跳"""
        self._running = False
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)
        logger.info("服务心跳已停止")

    def _heartbeat_loop(self):
        """心跳循环"""
        while self._running:
            try:
                self._send_heartbeats()
            except Exception as e:
                logger.error(f"心跳发送出错: {e}")

            time.sleep(self.heartbeat_interval)

    def _send_heartbeats(self):
        """发送心跳"""
        for instance in self.local_services.values():
            try:
                # 更新最后心跳时间
                instance.last_heartbeat = datetime.now()

                # 发送心跳到发现服务器
                self._send_heartbeat_to_discovery_server(instance)

            except Exception as e:
                logger.warning(f"心跳发送失败: {instance.service_name} - {e}")

    def _register_with_discovery_server(self, instance: ServiceInstance):
        """向发现服务器注册"""
        if not self.discovery_server_url:
            return

        try:
            url = f"{self.discovery_server_url}/eureka / apps/{instance.service_name}"
            data = instance.to_dict()

            response = requests.post(url, json=data, timeout=10)
            if response.status_code in [200, 204]:
                logger.info(f"服务已向发现服务器注册: {instance.service_name}")
            else:
                logger.warning(f"服务注册失败: {response.status_code}")

        except Exception as e:
            logger.warning(f"发现服务器注册出错: {e}")

    def _unregister_from_discovery_server(self, instance: ServiceInstance):
        """从发现服务器注销"""
        if not self.discovery_server_url:
            return

        try:
            url = f"{self.discovery_server_url}/eureka / apps/{instance.service_name}/{instance.instance_id}"
            response = requests.delete(url, timeout=10)

            if response.status_code in [200, 204]:
                logger.info(f"服务已从发现服务器注销: {instance.service_name}")
            else:
                logger.warning(f"服务注销失败: {response.status_code}")

        except Exception as e:
            logger.warning(f"发现服务器注销出错: {e}")

    def _send_heartbeat_to_discovery_server(self, instance: ServiceInstance):
        """向发现服务器发送心跳"""
        if not self.discovery_server_url:
            return

        try:
            url = f"{self.discovery_server_url}/eureka / apps/{instance.service_name}/{instance.instance_id}"
            response = requests.put(url, timeout=5)

            if response.status_code not in [200, 204]:
                logger.warning(f"心跳发送失败: {response.status_code}")

        except Exception as e:
            logger.debug(f"心跳发送出错: {e}")

    def _get_local_ip(self) -> str:
        """获取本机IP地址"""
        try:
            # 尝试连接外部地址获取本机IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            # 如果失败，返回localhost
            return "127.0.0.1"

    def get_status(self) -> Dict[str, Any]:
        """获取客户端状态"""
        return {
            'discovery_server': self.discovery_server_url,
            'local_services': {
                instance_id: {
                    'service_name': instance.service_name,
                    'endpoint': f"{instance.host}:{instance.port}",
                    'status': instance.status
                }
                for instance_id, instance in self.local_services.items()
            },
            'load_balancer_stats': self.load_balancer.get_service_stats(),
            'heartbeat_running': self._running
        }


class ServiceDiscoveryServer:

    """服务发现服务器"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8761):

        self.host = host
        self.port = port
        self.services: Dict[str, Dict[str, ServiceInstance]] = {}
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=10)

    def start(self):
        """启动发现服务器"""
        from flask import Flask, request, jsonify

        app = Flask(__name__)

        @app.route('/eureka / apps', methods=['GET'])
        def get_applications():
            """获取所有应用"""
            with self.lock:
                return jsonify({
                    'applications': {
                        service_name: {
                            'instances': [
                                instance.to_dict()
                                for instance in instances.values()
                            ]
                        }
                        for service_name, instances in self.services.items()
                    }
                })

        @app.route('/eureka / apps/<service_name>', methods=['GET', 'POST'])
        def handle_service(service_name):

            if request.method == 'GET':
                # 获取服务实例
                with self.lock:
                    instances = self.services.get(service_name, {})
                    return jsonify({
                        'application': {
                            'name': service_name,
                            'instances': [
                                instance.to_dict()
                                for instance in instances.values()
                            ]
                        }
                    })
            else:
                # 注册服务实例
                data = request.get_json()
                instance = ServiceInstance.from_dict(data)

                with self.lock:
                    if service_name not in self.services:
                        self.services[service_name] = {}
                    self.services[service_name][instance.instance_id] = instance

                logger.info(f"服务实例已注册: {service_name} - {instance.instance_id}")
                return jsonify({'status': 'registered'}), 200

        @app.route('/eureka / apps/<service_name>/<instance_id>', methods=['PUT', 'DELETE'])
        def handle_instance(service_name, instance_id):

            if request.method == 'PUT':
                # 心跳更新
                with self.lock:
                    if service_name in self.services and instance_id in self.services[service_name]:
                        self.services[service_name][instance_id].last_heartbeat = datetime.now()
                        return jsonify({'status': 'heartbeat_received'}), 200
                return jsonify({'error': 'instance_not_found'}), 404

            else:
                # 注销实例
                with self.lock:
                    if service_name in self.services and instance_id in self.services[service_name]:
                        del self.services[service_name][instance_id]
                        logger.info(f"服务实例已注销: {service_name} - {instance_id}")
                        return jsonify({'status': 'unregistered'}), 200
                return jsonify({'error': 'instance_not_found'}), 404

        logger.info(f"服务发现服务器启动: http://{self.host}:{self.port}")
        app.run(host=self.host, port=self.port, debug=False)


# 全局客户端实例
_discovery_client_instance: Optional[ServiceDiscoveryClient] = None
_client_lock = threading.Lock()


def get_discovery_client(discovery_server_url: Optional[str] = None) -> ServiceDiscoveryClient:
    """获取服务发现客户端实例（单例模式）"""
    global _discovery_client_instance

    if _discovery_client_instance is None:
        with _client_lock:
            if _discovery_client_instance is None:
                _discovery_client_instance = ServiceDiscoveryClient(discovery_server_url)

    return _discovery_client_instance


# 便捷函数

def register_current_service(service_name: str, port: int, **kwargs) -> str:
    """注册当前服务"""
    client = get_discovery_client()
    return client.register_service(service_name, port, **kwargs)


def discover_service(service_name: str) -> Optional[ServiceInstance]:
    """发现服务"""
    client = get_discovery_client()
    return client.discover_service(service_name)


if __name__ == "__main__":
    # 测试代码
    print("服务发现和注册模块测试")

    # 创建客户端
    client = get_discovery_client()

    # 注册测试服务
    instance_id = client.register_service("test - service", 5001, {"version": "1.0.0"})
    print(f"服务已注册，实例ID: {instance_id}")

    # 发现服务
    instance = client.discover_service("test - service")
    if instance:
        print(f"服务已发现: {instance.service_name} - {instance.host}:{instance.port}")

    print("服务发现状态:", client.get_status())

    print("服务发现和注册模块测试完成")
