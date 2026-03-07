#!/bin/bash
# RQA2025 CI/CD环境初始化脚本

set -e

echo "🔧 RQA2025 CI/CD环境初始化开始"
echo "=================================="

# 检查必要的工具
echo "🔍 检查环境依赖..."
command -v docker >/dev/null 2>&1 || { echo "❌ Docker未安装，请先安装Docker"; exit 1; }
command -v kubectl >/dev/null 2>&1 || { echo "❌ kubectl未安装，请先安装kubectl"; exit 1; }
command -v helm >/dev/null 2>&1 || { echo "❌ Helm未安装，请先安装Helm"; exit 1; }

echo "✅ 环境依赖检查通过"

# 创建Kubernetes集群连接配置
echo "🔗 配置Kubernetes集群连接..."
mkdir -p ~/.kube

# 为不同环境创建配置文件模板
cat > ~/.kube/config.staging << EOF
apiVersion: v1
clusters:
- cluster:
    server: https://staging-cluster.example.com
  name: staging-cluster
contexts:
- context:
    cluster: staging-cluster
    user: staging-user
  name: staging
current-context: staging
EOF

cat > ~/.kube/config.preprod << EOF
apiVersion: v1
clusters:
- cluster:
    server: https://preprod-cluster.example.com
  name: preprod-cluster
contexts:
- context:
    cluster: preprod-cluster
    user: preprod-user
  name: preprod
current-context: preprod
EOF

cat > ~/.kube/config.production << EOF
apiVersion: v1
clusters:
- cluster:
    server: https://production-cluster.example.com
  name: production-cluster
contexts:
- context:
    cluster: production-cluster
    user: production-user
  name: production
current-context: production
EOF

echo "✅ Kubernetes集群配置完成"

# 创建Docker Registry配置
echo "🐳 配置Docker Registry..."
cat > docker-registry-config.yaml << EOF
apiVersion: v1
kind: Config
clusters: []
contexts: []
current-context: ""
preferences: {}
users: []
EOF

echo "✅ Docker Registry配置完成"

# 创建GitLab Runner配置
echo "🏃 配置GitLab Runner..."
cat > gitlab-runner-config.toml << EOF
concurrent = 4
check_interval = 0

[session_server]
  session_timeout = 1800

[[runners]]
  name = "rqa2025-runner"
  url = "https://gitlab.example.com/"
  token = "YOUR_RUNNER_TOKEN"
  executor = "kubernetes"
  [runners.kubernetes]
    host = ""
    bearer_token_overwrite_allowed = false
    image = "ubuntu:20.04"
    namespace = "gitlab-runner"
    privileged = true
    [runners.kubernetes.node_selector]
      kubernetes.io/arch = "amd64"
    [runners.kubernetes.volumes]
      [[runners.kubernetes.volumes.empty_dir]]
        name = "docker-certs"
        mount_path = "/certs/client"
        medium = "Memory"
EOF

echo "✅ GitLab Runner配置完成"

# 创建Helm部署脚本
echo "⛵ 创建Helm部署脚本..."
cat > scripts/deploy-with-helm.sh << 'EOF'
#!/bin/bash
# 使用Helm进行部署

set -e

ENVIRONMENT=$1
NAMESPACE="rqa2025-${ENVIRONMENT}"

if [ -z "$ENVIRONMENT" ]; then
    echo "❌ 请指定环境 (staging|preprod|production)"
    exit 1
fi

echo "⛵ 使用Helm部署到${ENVIRONMENT}环境..."

# 添加Helm仓库
helm repo add rqa2025 https://charts.rqa2025.example.com/
helm repo update

# 部署应用
helm upgrade --install rqa2025-app rqa2025/rqa2025 \
  --namespace $NAMESPACE \
  --create-namespace \
  --set image.tag=$CI_COMMIT_SHA \
  --set environment=$ENVIRONMENT \
  --wait \
  --timeout=600s

echo "✅ Helm部署完成"
EOF

chmod +x scripts/deploy-with-helm.sh

# 创建ArgoCD配置
echo "🔄 配置ArgoCD..."
mkdir -p argocd/applications

cat > argocd/applications/rqa2025-production.yaml << EOF
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: rqa2025-production
  namespace: argocd
spec:
  project: rqa2025
  source:
    repoURL: https://gitlab.example.com/rqa2025/rqa2025.git
    targetRevision: HEAD
    path: k8s/production
  destination:
    server: https://kubernetes.default.svc
    namespace: rqa2025-app
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
    - CreateNamespace=true
EOF

echo "✅ ArgoCD配置完成"

# 创建监控和告警脚本
echo "📊 创建监控配置脚本..."
cat > scripts/setup-monitoring.sh << 'EOF'
#!/bin/bash
# 部署监控栈

set -e

echo "📊 部署Prometheus + Grafana监控栈..."

# 创建监控命名空间
kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -

# 部署Prometheus
kubectl apply -f monitoring/prometheus/
kubectl apply -f monitoring/grafana/

# 等待部署完成
kubectl wait --for=condition=available --timeout=300s deployment/prometheus -n monitoring
kubectl wait --for=condition=available --timeout=300s deployment/grafana -n monitoring

echo "✅ 监控栈部署完成"

# 配置告警规则
echo "🚨 配置告警规则..."
kubectl apply -f monitoring/alertmanager/

echo "✅ 告警规则配置完成"

# 创建监控仪表板
echo "📈 创建监控仪表板..."
kubectl apply -f monitoring/dashboards/

echo "✅ 监控仪表板创建完成"
EOF

chmod +x scripts/setup-monitoring.sh

echo "🎉 CI/CD环境初始化完成！"
echo "=================================="
echo "📋 下一步操作："
echo "  1. 配置GitLab Runner令牌"
echo "  2. 更新Kubernetes集群连接信息"
echo "  3. 配置Docker Registry访问凭据"
echo "  4. 测试CI/CD流水线"
echo "=================================="
