#!/bin/bash
# RQA2025 Kubernetes生产环境部署脚本

set -e

echo "🐳 RQA2025 Kubernetes生产环境部署开始"
echo "=========================================="

# 检查kubectl连接
echo "🔍 检查Kubernetes集群连接..."
kubectl cluster-info
kubectl version --short

# 创建命名空间
echo "📦 创建命名空间..."
kubectl apply -f k8s/production/namespaces.yaml

# 创建存储类
echo "💾 创建存储类..."
kubectl apply -f k8s/production/storage-classes.yaml

# 创建配置和密钥
echo "⚙️ 创建配置和密钥..."
kubectl apply -f k8s/production/configs.yaml

# 部署数据服务
echo "🗄️ 部署数据服务..."
kubectl apply -f k8s/production/data-services-deployment.yaml

# 创建数据服务
echo "🌐 创建数据服务..."
kubectl apply -f k8s/production/services.yaml

# 部署应用服务
echo "🚀 部署应用服务..."
kubectl apply -f k8s/production/rqa2025-app-deployment.yaml

# 创建Ingress
echo "🔒 创建Ingress配置..."
kubectl apply -f k8s/production/ingress.yaml

# 验证部署
echo "✅ 验证部署状态..."
kubectl get pods -A
kubectl get services -A
kubectl get ingress -A

echo "🎉 RQA2025 Kubernetes生产环境部署完成！"
echo "=========================================="
echo "📋 访问地址: https://rqa2025.example.com"
echo "📊 监控命令: kubectl get pods -A"




