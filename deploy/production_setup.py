import os
import yaml
from typing import Dict, List
from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ServerSpec:
    """服务器规格配置"""
    cpu_cores: int
    memory_gb: int
    gpu_count: int = 0
    fpga_count: int = 0
    storage_tb: int = 1

@dataclass
class NetworkConfig:
    """网络配置"""
    bandwidth_gbps: int
    latency_us: float
    redundancy: bool

class ProductionDeployer:
    """生产环境部署器"""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.servers = self._initialize_servers()
        self.network = self._initialize_network()

    def _load_config(self, path: str) -> Dict:
        """加载部署配置文件"""
        with open(path) as f:
            return yaml.safe_load(f)

    def _initialize_servers(self) -> List[ServerSpec]:
        """初始化服务器配置"""
        specs = []
        for server in self.config['servers']:
            specs.append(ServerSpec(
                cpu_cores=server['cpu_cores'],
                memory_gb=server['memory_gb'],
                gpu_count=server.get('gpu_count', 0),
                fpga_count=server.get('fpga_count', 0),
                storage_tb=server.get('storage_tb', 1)
            ))
        return specs

    def _initialize_network(self) -> NetworkConfig:
        """初始化网络配置"""
        net = self.config['network']
        return NetworkConfig(
            bandwidth_gbps=net['bandwidth_gbps'],
            latency_us=net['latency_us'],
            redundancy=net['redundancy']
        )

    def validate_requirements(self) -> bool:
        """验证部署需求"""
        # 检查计算资源
        total_cores = sum(s.cpu_cores for s in self.servers)
        if total_cores < self.config['requirements']['min_cpu_cores']:
            logger.error("CPU核心数不足")
            return False

        # 检查内存资源
        total_memory = sum(s.memory_gb for s in self.servers)
        if total_memory < self.config['requirements']['min_memory_gb']:
            logger.error("内存容量不足")
            return False

        # 检查网络延迟
        if self.network.latency_us > self.config['requirements']['max_latency_us']:
            logger.error("网络延迟过高")
            return False

        return True

    def generate_deployment_scripts(self, output_dir: str):
        """生成部署脚本"""
        os.makedirs(output_dir, exist_ok=True)

        # 生成Kubernetes部署文件
        self._generate_k8s_yaml(os.path.join(output_dir, 'k8s'))

        # 生成监控配置
        self._generate_monitoring_config(os.path.join(output_dir, 'monitoring'))

        # 生成安全策略
        self._generate_security_policies(os.path.join(output_dir, 'security'))

    def _generate_k8s_yaml(self, output_dir: str):
        """生成Kubernetes部署文件"""
        os.makedirs(output_dir, exist_ok=True)

        # 生成命名空间配置
        with open(os.path.join(output_dir, 'namespace.yaml'), 'w') as f:
            f.write("""apiVersion: v1
kind: Namespace
metadata:
  name: rqa-prod
""")

        # 生成部署配置
        services = [
            'data-service',
            'feature-service',
            'model-service',
            'trading-service',
            'execution-service'
        ]

        for svc in services:
            with open(os.path.join(output_dir, f'{svc}-deployment.yaml'), 'w') as f:
                f.write(f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {svc}
  namespace: rqa-prod
spec:
  replicas: 3
  selector:
    matchLabels:
      app: {svc}
  template:
    metadata:
      labels:
        app: {svc}
    spec:
      containers:
      - name: {svc}
        image: registry.rqa.com/{svc}:prod
        resources:
          limits:
            cpu: "4"
            memory: 8Gi
        ports:
        - containerPort: 8080
""")

    def _generate_monitoring_config(self, output_dir: str):
        """生成监控配置"""
        os.makedirs(output_dir, exist_ok=True)

        # Prometheus配置
        with open(os.path.join(output_dir, 'prometheus.yml'), 'w') as f:
            f.write("""global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'rqa-services'
    static_configs:
      - targets: ['data-service:8080', 'feature-service:8080', 
                 'model-service:8080', 'trading-service:8080',
                 'execution-service:8080']
""")

        # Grafana仪表盘
        with open(os.path.join(output_dir, 'grafana-dashboards.yaml'), 'w') as f:
            f.write("""apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboards
  namespace: monitoring
data:
  rqa-system.json: |-
    {
      "title": "RQA System Dashboard",
      "panels": [...]
    }
""")

    def _generate_security_policies(self, output_dir: str):
        """生成安全策略"""
        os.makedirs(output_dir, exist_ok=True)

        # 网络策略
        with open(os.path.join(output_dir, 'network-policy.yaml'), 'w') as f:
            f.write("""apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: rqa-network-policy
  namespace: rqa-prod
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 9090
""")

if __name__ == "__main__":
    deployer = ProductionDeployer("config/production.yaml")
    if deployer.validate_requirements():
        deployer.generate_deployment_scripts("output/deploy")
        print("生产部署配置生成成功")
    else:
        print("部署需求验证失败，请检查配置")
