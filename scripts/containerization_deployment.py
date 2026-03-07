#!/usr/bin/env python3
"""
容器化部署工具

完善基础设施层的容器化支持
"""

import os
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


class ContainerizationDeployment:
    """容器化部署工具"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.infrastructure_dir = self.project_root / "src" / "infrastructure"
        self.docker_dir = self.project_root / "docker"
        self.kubernetes_dir = self.project_root / "kubernetes"

        # 容器化配置
        self.config = {
            "docker": {
                "base_image": "python:3.9-slim",
                "maintainer": "RQA2025 Team",
                "expose_ports": [8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007, 8008],
                "environment": {
                    "ENV": "production",
                    "DEBUG": "false"
                }
            },
            "kubernetes": {
                "namespace": "rqa2025-infrastructure",
                "replicas": 2,
                "resources": {
                    "requests": {
                        "memory": "256Mi",
                        "cpu": "100m"
                    },
                    "limits": {
                        "memory": "512Mi",
                        "cpu": "500m"
                    }
                }
            },
            "monitoring": {
                "prometheus_enabled": True,
                "grafana_enabled": True,
                "jaeger_enabled": True
            }
        }

    def create_docker_setup(self) -> Dict[str, Any]:
        """创建Docker设置"""
        print("🐳 创建Docker设置...")

        self.docker_dir.mkdir(exist_ok=True)

        # 创建Dockerfile
        dockerfile_path = self.docker_dir / "Dockerfile"
        dockerfile_content = self._generate_dockerfile()
        with open(dockerfile_path, 'w', encoding='utf-8') as f:
            f.write(dockerfile_content)

        # 创建.dockerignore
        dockerignore_path = self.docker_dir / ".dockerignore"
        dockerignore_content = self._generate_dockerignore()
        with open(dockerignore_path, 'w', encoding='utf-8') as f:
            f.write(dockerignore_content)

        # 创建requirements.txt
        requirements_path = self.docker_dir / "requirements.txt"
        requirements_content = self._generate_requirements()
        with open(requirements_path, 'w', encoding='utf-8') as f:
            f.write(requirements_content)

        # 创建启动脚本
        entrypoint_path = self.docker_dir / "entrypoint.sh"
        entrypoint_content = self._generate_entrypoint()
        with open(entrypoint_path, 'w', encoding='utf-8') as f:
            f.write(entrypoint_content)
        os.chmod(entrypoint_path, 0o755)

        print("✅ Docker设置创建完成")
        return {
            "success": True,
            "docker_dir": str(self.docker_dir),
            "files_created": 4
        }

    def create_kubernetes_manifests(self) -> Dict[str, Any]:
        """创建Kubernetes清单文件"""
        print("☸️ 创建Kubernetes清单文件...")

        self.kubernetes_dir.mkdir(exist_ok=True)

        # 创建命名空间
        namespace_manifest = self._generate_namespace_manifest()
        with open(self.kubernetes_dir / "namespace.yaml", 'w', encoding='utf-8') as f:
            f.write(namespace_manifest)

        # 创建配置映射
        configmap_manifest = self._generate_configmap_manifest()
        with open(self.kubernetes_dir / "configmap.yaml", 'w', encoding='utf-8') as f:
            f.write(configmap_manifest)

        # 创建密钥
        secret_manifest = self._generate_secret_manifest()
        with open(self.kubernetes_dir / "secret.yaml", 'w', encoding='utf-8') as f:
            f.write(secret_manifest)

        # 创建部署文件
        deployment_manifest = self._generate_deployment_manifest()
        with open(self.kubernetes_dir / "deployment.yaml", 'w', encoding='utf-8') as f:
            f.write(deployment_manifest)

        # 创建服务文件
        service_manifest = self._generate_service_manifest()
        with open(self.kubernetes_dir / "service.yaml", 'w', encoding='utf-8') as f:
            f.write(service_manifest)

        # 创建Ingress文件
        ingress_manifest = self._generate_ingress_manifest()
        with open(self.kubernetes_dir / "ingress.yaml", 'w', encoding='utf-8') as f:
            f.write(ingress_manifest)

        # 创建HorizontalPodAutoscaler
        hpa_manifest = self._generate_hpa_manifest()
        with open(self.kubernetes_dir / "hpa.yaml", 'w', encoding='utf-8') as f:
            f.write(hpa_manifest)

        print("✅ Kubernetes清单文件创建完成")
        return {
            "success": True,
            "kubernetes_dir": str(self.kubernetes_dir),
            "files_created": 7
        }

    def create_monitoring_setup(self) -> Dict[str, Any]:
        """创建监控设置"""
        print("📊 创建监控设置...")

        monitoring_dir = self.project_root / "monitoring"
        monitoring_dir.mkdir(exist_ok=True)

        # 创建Prometheus配置
        if self.config["monitoring"]["prometheus_enabled"]:
            prometheus_config = self._generate_prometheus_config()
            with open(monitoring_dir / "prometheus.yml", 'w', encoding='utf-8') as f:
                f.write(prometheus_config)

        # 创建Grafana配置
        if self.config["monitoring"]["grafana_enabled"]:
            grafana_dashboards = self._generate_grafana_dashboards()
            with open(monitoring_dir / "grafana-dashboards.json", 'w', encoding='utf-8') as f:
                f.write(grafana_dashboards)

        # 创建Jaeger配置
        if self.config["monitoring"]["jaeger_enabled"]:
            jaeger_config = self._generate_jaeger_config()
            with open(monitoring_dir / "jaeger-config.yml", 'w', encoding='utf-8') as f:
                f.write(jaeger_config)

        # 创建监控Docker Compose文件
        monitoring_compose = self._generate_monitoring_compose()
        with open(monitoring_dir / "docker-compose.monitoring.yml", 'w', encoding='utf-8') as f:
            f.write(monitoring_compose)

        print("✅ 监控设置创建完成")
        return {
            "success": True,
            "monitoring_dir": str(monitoring_dir),
            "files_created": 4
        }

    def create_ci_cd_pipeline(self) -> Dict[str, Any]:
        """创建CI/CD流水线"""
        print("🔄 创建CI/CD流水线...")

        github_actions_dir = self.project_root / ".github" / "workflows"
        github_actions_dir.mkdir(parents=True, exist_ok=True)

        # 创建构建工作流
        build_workflow = self._generate_build_workflow()
        with open(github_actions_dir / "build.yml", 'w', encoding='utf-8') as f:
            f.write(build_workflow)

        # 创建部署工作流
        deploy_workflow = self._generate_deploy_workflow()
        with open(github_actions_dir / "deploy.yml", 'w', encoding='utf-8') as f:
            f.write(deploy_workflow)

        # 创建测试工作流
        test_workflow = self._generate_test_workflow()
        with open(github_actions_dir / "test.yml", 'w', encoding='utf-8') as f:
            f.write(test_workflow)

        print("✅ CI/CD流水线创建完成")
        return {
            "success": True,
            "workflows_dir": str(github_actions_dir),
            "files_created": 3
        }

    def _generate_dockerfile(self) -> str:
        """生成Dockerfile"""
        return f'''# Multi-stage Docker build for RQA2025 Infrastructure Layer

# Build stage
FROM {self.config["docker"]["base_image"]} as builder

LABEL maintainer="{self.config["docker"]["maintainer"]}"
LABEL version="1.0.0"
LABEL description="RQA2025 Infrastructure Layer"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# Runtime stage
FROM {self.config["docker"]["base_image"]} as runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \\
    && chown -R app:app /app
USER app

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/app/.local

# Copy application code
COPY --from=builder /app/src ./src/
COPY --from=builder /app/scripts ./scripts/

# Make scripts executable
RUN chmod +x scripts/*.py

# Set environment variables
ENV PATH="/home/app/.local/bin:$PATH"
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000

# Start command
CMD ["python", "-m", "uvicorn", "src.infrastructure.main:app", "--host", "0.0.0.0", "--port", "8000"]
'''

    def _generate_dockerignore(self) -> str:
        """生成.dockerignore"""
        return '''# Git
.git
.gitignore
.gitattributes

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Testing
.coverage
.pytest_cache/
.tox/
.cache

# Documentation
docs/_build/
*.md
!README.md

# Reports
reports/
*.log

# Docker
Dockerfile
.dockerignore
docker-compose*.yml

# Kubernetes
kubernetes/

# CI/CD
.github/

# Monitoring
monitoring/
'''

    def _generate_requirements(self) -> str:
        """生成requirements.txt"""
        return '''# Core dependencies
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
sqlalchemy==2.0.23

# Infrastructure dependencies
redis==5.0.1
pymongo==4.6.1
elasticsearch==8.11.0
prometheus-client==0.19.0

# Monitoring and observability
opentelemetry-api==1.22.0
opentelemetry-sdk==1.22.0
opentelemetry-exporter-jaeger==1.22.0
opentelemetry-exporter-prometheus==1.22.0

# Security
cryptography==41.0.7
bcrypt==4.1.2
python-jose[cryptography]==3.3.0

# Async and performance
aiofiles==23.2.1
httpx==0.26.0
gunicorn==21.2.0

# Configuration
pyyaml==6.0.1
dynaconf==3.2.4

# Logging
structlog==23.2.0

# Development
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
black==23.12.1
isort==5.13.2
flake8==7.0.0
mypy==1.8.0

# Container and deployment
docker==7.0.0
kubernetes==28.1.0
'''

    def _generate_entrypoint(self) -> str:
        """生成启动脚本"""
        return '''#!/bin/bash
set -e

# Wait for dependencies
if [ "$WAIT_FOR_DEPENDENCIES" = "true" ]; then
    echo "Waiting for dependencies..."

    # Wait for Redis
    if [ "$REDIS_HOST" ]; then
        echo "Waiting for Redis at $REDIS_HOST:$REDIS_PORT..."
        while ! nc -z $REDIS_HOST $REDIS_PORT; do
            sleep 1
        done
    fi

    # Wait for MongoDB
    if [ "$MONGODB_HOST" ]; then
        echo "Waiting for MongoDB at $MONGODB_HOST:$MONGODB_PORT..."
        while ! nc -z $MONGODB_HOST $MONGODB_PORT; do
            sleep 1
        done
    fi

    # Wait for Elasticsearch
    if [ "$ELASTICSEARCH_HOST" ]; then
        echo "Waiting for Elasticsearch at $ELASTICSEARCH_HOST:$ELASTICSEARCH_PORT..."
        while ! nc -z $ELASTICSEARCH_HOST $ELASTICSEARCH_PORT; do
            sleep 1
        done
    fi
fi

# Run pre-start scripts
if [ -d "/app/scripts/pre-start" ]; then
    for script in /app/scripts/pre-start/*.sh; do
        if [ -x "$script" ]; then
            echo "Running pre-start script: $script"
            "$script"
        fi
    done
fi

# Set Python path
export PYTHONPATH="/app/src:$PYTHONPATH"

# Start the application
echo "Starting application..."
exec "$@"
'''

    def _generate_namespace_manifest(self) -> str:
        """生成Kubernetes命名空间清单"""
        return f'''apiVersion: v1
kind: Namespace
metadata:
  name: {self.config["kubernetes"]["namespace"]}
  labels:
    name: {self.config["kubernetes"]["namespace"]}
    app: rqa2025-infrastructure
'''

    def _generate_configmap_manifest(self) -> str:
        """生成ConfigMap清单"""
        return f'''apiVersion: v1
kind: ConfigMap
metadata:
  name: rqa2025-infrastructure-config
  namespace: {self.config["kubernetes"]["namespace"]}
data:
  config.yaml: |
    app:
      name: rqa2025-infrastructure
      version: "1.0.0"
      environment: production

    server:
      host: "0.0.0.0"
      port: 8000
      workers: 4

    logging:
      level: INFO
      format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    monitoring:
      enabled: true
      prometheus_port: 9090
      metrics_path: "/metrics"
'''

    def _generate_secret_manifest(self) -> str:
        """生成Secret清单"""
        return f'''apiVersion: v1
kind: Secret
metadata:
  name: rqa2025-infrastructure-secret
  namespace: {self.config["kubernetes"]["namespace"]}
type: Opaque
data:
  # Base64 encoded secrets
  redis-password: "cmVkaXNfcGFzc3dvcmQ="  # redis_password
  mongodb-password: "bW9uZ29kYl9wYXNzd29yZA=="  # mongodb_password
  jwt-secret: "andF3LXNlY3JldA=="  # jwt-secret
'''

    def _generate_deployment_manifest(self) -> str:
        """生成Deployment清单"""
        return f'''apiVersion: apps/v1
kind: Deployment
metadata:
  name: rqa2025-infrastructure
  namespace: {self.config["kubernetes"]["namespace"]}
  labels:
    app: rqa2025-infrastructure
spec:
  replicas: {self.config["kubernetes"]["replicas"]}
  selector:
    matchLabels:
      app: rqa2025-infrastructure
  template:
    metadata:
      labels:
        app: rqa2025-infrastructure
    spec:
      containers:
      - name: infrastructure
        image: rqa2025/infrastructure:latest
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: ENV
          value: "production"
        - name: NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        resources:
          requests:
            memory: {self.config["kubernetes"]["resources"]["requests"]["memory"]}
            cpu: {self.config["kubernetes"]["resources"]["requests"]["cpu"]}
          limits:
            memory: {self.config["kubernetes"]["resources"]["limits"]["memory"]}
            cpu: {self.config["kubernetes"]["resources"]["limits"]["cpu"]}
        livenessProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
        volumeMounts:
        - name: config
          mountPath: /app/config
          readOnly: true
      volumes:
      - name: config
        configMap:
          name: rqa2025-infrastructure-config
'''

    def _generate_service_manifest(self) -> str:
        """生成Service清单"""
        return f'''apiVersion: v1
kind: Service
metadata:
  name: rqa2025-infrastructure-service
  namespace: {self.config["kubernetes"]["namespace"]}
  labels:
    app: rqa2025-infrastructure
spec:
  selector:
    app: rqa2025-infrastructure
  ports:
  - name: http
    port: 80
    targetPort: 8000
    protocol: TCP
  type: ClusterIP
'''

    def _generate_ingress_manifest(self) -> str:
        """生成Ingress清单"""
        return f'''apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rqa2025-infrastructure-ingress
  namespace: {self.config["kubernetes"]["namespace"]}
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/use-regex: "true"
spec:
  tls:
  - hosts:
    - infrastructure.rqa2025.com
    secretName: rqa2025-infrastructure-tls
  rules:
  - host: infrastructure.rqa2025.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: rqa2025-infrastructure-service
            port:
              number: 80
'''

    def _generate_hpa_manifest(self) -> str:
        """生成HPA清单"""
        return f'''apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rqa2025-infrastructure-hpa
  namespace: {self.config["kubernetes"]["namespace"]}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rqa2025-infrastructure
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
'''

    def _generate_prometheus_config(self) -> str:
        """生成Prometheus配置"""
        return '''global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'rqa2025-infrastructure'
    static_configs:
      - targets: ['rqa2025-infrastructure:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'mongodb'
    static_configs:
      - targets: ['mongodb:27017']
'''

    def _generate_grafana_dashboards(self) -> str:
        """生成Grafana仪表板配置"""
        return '''{
  "dashboard": {
    "title": "RQA2025 Infrastructure Monitoring",
    "tags": ["rqa2025", "infrastructure"],
    "timezone": "UTC",
    "panels": [
      {
        "title": "CPU Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(process_cpu_usage[5m])",
            "legendFormat": "CPU Usage"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "process_memory_usage",
            "legendFormat": "Memory Usage"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "http_request_duration_seconds",
            "legendFormat": "Response Time"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "5s"
  }
}'''

    def _generate_jaeger_config(self) -> str:
        """生成Jaeger配置"""
        return '''service:
  name: rqa2025-infrastructure
  version: "1.0.0"

reporter:
  logSpans: true
  localAgentHostPort: jaeger:6831

sampler:
  type: const
  param: 1

headers:
  jaegerDebugHeader: debug-id
  jaegerBaggageHeader: baggage
  TraceContextHeaderName: traceparent
  traceBaggageHeaderPrefix: "trace-baggage-"
'''

    def _generate_monitoring_compose(self) -> str:
        """生成监控Docker Compose文件"""
        return '''version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    networks:
      - monitoring

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
    networks:
      - monitoring
    depends_on:
      - prometheus

  jaeger:
    image: jaegertracing/all-in-one:latest
    container_name: jaeger
    ports:
      - "16686:16686"
      - "14250:14250"
    environment:
      - COLLECTOR_OTLP_ENABLED=true
    networks:
      - monitoring

volumes:
  prometheus_data:
  grafana_data:

networks:
  monitoring:
    driver: bridge
'''

    def _generate_build_workflow(self) -> str:
        """生成GitHub Actions构建工作流"""
        return '''name: Build and Test

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10]

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run tests
      run: |
        pytest tests/ -v --cov=src --cov-report=xml

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build:
    needs: test
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Build Docker image
      run: |
        docker build -t rqa2025/infrastructure:${{ github.sha }} .
        docker tag rqa2025/infrastructure:${{ github.sha }} rqa2025/infrastructure:latest

    - name: Push Docker image
      if: github.ref == 'refs/heads/main'
      run: |
        echo "${{ secrets.DOCKER_PASSWORD }}" | docker login -u "${{ secrets.DOCKER_USERNAME }}" --password-stdin
        docker push rqa2025/infrastructure:${{ github.sha }}
        docker push rqa2025/infrastructure:latest
'''

    def _generate_deploy_workflow(self) -> str:
        """生成GitHub Actions部署工作流"""
        return '''name: Deploy to Kubernetes

on:
  push:
    branches: [ main ]
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Configure kubectl
      run: |
        echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > $HOME/.kube/config
        chmod 600 $HOME/.kube/config

    - name: Deploy to Kubernetes
      run: |
        kubectl apply -f kubernetes/namespace.yaml
        kubectl apply -f kubernetes/configmap.yaml
        kubectl apply -f kubernetes/secret.yaml
        kubectl apply -f kubernetes/deployment.yaml
        kubectl apply -f kubernetes/service.yaml
        kubectl apply -f kubernetes/ingress.yaml
        kubectl apply -f kubernetes/hpa.yaml

    - name: Verify deployment
      run: |
        kubectl get pods -n rqa2025-infrastructure
        kubectl get services -n rqa2025-infrastructure
        kubectl get ingress -n rqa2025-infrastructure
'''

    def _generate_test_workflow(self) -> str:
        """生成GitHub Actions测试工作流"""
        return '''name: Security and Performance Tests

on:
  schedule:
    - cron: '0 2 * * *'  # 每天凌晨2点运行
  workflow_dispatch:

jobs:
  security:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Run security scan
      run: |
        pip install safety bandit
        safety check
        bandit -r src/ -f json -o security-report.json

    - name: Upload security report
      uses: actions/upload-artifact@v3
      with:
        name: security-report
        path: security-report.json

  performance:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Run performance tests
      run: |
        pip install locust
        locust --headless -f tests/performance/locustfile.py --csv=performance-report

    - name: Upload performance report
      uses: actions/upload-artifact@v3
      with:
        name: performance-report
        path: performance-report_*.csv
'''

    def generate_containerization_report(self) -> Dict[str, Any]:
        """生成容器化报告"""
        report_data = {
            "timestamp": datetime.now(),
            "docker_setup": self.create_docker_setup(),
            "kubernetes_setup": self.create_kubernetes_manifests(),
            "monitoring_setup": self.create_monitoring_setup(),
            "ci_cd_setup": self.create_ci_cd_pipeline(),
            "summary": {
                "total_files_created": 0,
                "directories_created": 0,
                "services_configured": len(self.config["docker"]["expose_ports"]),
                "monitoring_enabled": sum(self.config["monitoring"].values())
            }
        }

        # 计算总数
        report_data["summary"]["total_files_created"] = (
            report_data["docker_setup"]["files_created"] +
            report_data["kubernetes_setup"]["files_created"] +
            report_data["monitoring_setup"]["files_created"] +
            report_data["ci_cd_setup"]["files_created"]
        )

        # 保存报告
        report_path = self.project_root / "reports" / \
            f"containerization_deployment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)

        return {
            "success": True,
            "report_path": str(report_path),
            "data": report_data
        }


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='容器化部署工具')
    parser.add_argument('--project', default='.', help='项目根目录')
    parser.add_argument('--docker', action='store_true', help='创建Docker设置')
    parser.add_argument('--kubernetes', action='store_true', help='创建Kubernetes清单')
    parser.add_argument('--monitoring', action='store_true', help='创建监控设置')
    parser.add_argument('--ci-cd', action='store_true', help='创建CI/CD流水线')
    parser.add_argument('--all', action='store_true', help='创建所有容器化设置')
    parser.add_argument('--report', action='store_true', help='生成容器化报告')

    args = parser.parse_args()

    containerization = ContainerizationDeployment(args.project)

    if args.all:
        # 创建所有设置
        result = containerization.generate_containerization_report()
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    elif args.docker:
        result = containerization.create_docker_setup()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.kubernetes:
        result = containerization.create_kubernetes_manifests()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.monitoring:
        result = containerization.create_monitoring_setup()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.ci_cd:
        result = containerization.create_ci_cd_pipeline()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.report:
        result = containerization.generate_containerization_report()
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    else:
        print("🐳 容器化部署工具")
        print("使用 --help 查看可用命令")


if __name__ == "__main__":
    main()
