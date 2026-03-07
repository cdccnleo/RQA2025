#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 Kubernetes生产环境部署配置脚本

执行Phase 4C Week 1-2的Kubernetes集群部署任务
"""

import json
import yaml
from datetime import datetime
from pathlib import Path


def create_kubernetes_config():
    """创建Kubernetes配置文件"""
    print("🐳 创建Kubernetes生产环境配置")
    print("=" * 50)

    # 创建命名空间配置
    namespaces = {
        "rqa2025-app": {
            "name": "rqa2025-app",
            "labels": {
                "name": "rqa2025-app",
                "environment": "production",
                "project": "rqa2025"
            }
        },
        "rqa2025-data": {
            "name": "rqa2025-data",
            "labels": {
                "name": "rqa2025-data",
                "environment": "production",
                "project": "rqa2025"
            }
        },
        "rqa2025-monitoring": {
            "name": "rqa2025-monitoring",
            "labels": {
                "name": "rqa2025-monitoring",
                "environment": "production",
                "project": "rqa2025"
            }
        },
        "rqa2025-security": {
            "name": "rqa2025-security",
            "labels": {
                "name": "rqa2025-security",
                "environment": "production",
                "project": "rqa2025"
            }
        }
    }

    # 创建存储类配置
    storage_classes = {
        "fast-ssd": {
            "apiVersion": "storage.k8s.io/v1",
            "kind": "StorageClass",
            "metadata": {
                "name": "fast-ssd"
            },
            "provisioner": "kubernetes.io/aws-ebs",
            "parameters": {
                "type": "gp3",
                "fsType": "ext4"
            },
            "reclaimPolicy": "Retain",
            "allowVolumeExpansion": True,
            "volumeBindingMode": "WaitForFirstConsumer"
        },
        "standard-hdd": {
            "apiVersion": "storage.k8s.io/v1",
            "kind": "StorageClass",
            "metadata": {
                "name": "standard-hdd"
            },
            "provisioner": "kubernetes.io/aws-ebs",
            "parameters": {
                "type": "sc1",
                "fsType": "ext4"
            },
            "reclaimPolicy": "Retain",
            "allowVolumeExpansion": True,
            "volumeBindingMode": "WaitForFirstConsumer"
        }
    }

    # 创建网络策略
    network_policies = {
        "app-network-policy": {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": "app-network-policy",
                "namespace": "rqa2025-app"
            },
            "spec": {
                "podSelector": {},
                "policyTypes": ["Ingress", "Egress"],
                "ingress": [
                    {
                        "from": [
                            {
                                "namespaceSelector": {
                                    "matchLabels": {
                                        "name": "rqa2025-app"
                                    }
                                }
                            }
                        ]
                    }
                ],
                "egress": [
                    {
                        "to": [
                            {
                                "namespaceSelector": {
                                    "matchLabels": {
                                        "name": "rqa2025-data"
                                    }
                                }
                            }
                        ]
                    }
                ]
            }
        }
    }

    # 创建RBAC配置
    rbac_config = {
        "rqa2025-admin-role": {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "Role",
            "metadata": {
                "namespace": "rqa2025-app",
                "name": "rqa2025-admin-role"
            },
            "rules": [
                {
                    "apiGroups": ["", "extensions", "apps"],
                    "resources": ["deployments", "replicasets", "pods", "services", "ingresses"],
                    "verbs": ["get", "list", "watch", "create", "update", "patch", "delete"]
                }
            ]
        },
        "rqa2025-admin-rolebinding": {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "RoleBinding",
            "metadata": {
                "name": "rqa2025-admin-rolebinding",
                "namespace": "rqa2025-app"
            },
            "subjects": [
                {
                    "kind": "ServiceAccount",
                    "name": "rqa2025-service-account",
                    "namespace": "rqa2025-app"
                }
            ],
            "roleRef": {
                "kind": "Role",
                "name": "rqa2025-admin-role",
                "apiGroup": "rbac.authorization.k8s.io"
            }
        }
    }

    return {
        "namespaces": namespaces,
        "storage_classes": storage_classes,
        "network_policies": network_policies,
        "rbac_config": rbac_config
    }


def create_deployment_manifests():
    """创建应用部署清单"""
    print("📦 创建应用部署清单")
    print("=" * 50)

    # RQA2025主应用部署
    rqa_app_deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": "rqa2025-app",
            "namespace": "rqa2025-app",
            "labels": {
                "app": "rqa2025",
                "component": "app",
                "environment": "production"
            }
        },
        "spec": {
            "replicas": 3,
            "selector": {
                "matchLabels": {
                    "app": "rqa2025",
                    "component": "app"
                }
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": "rqa2025",
                        "component": "app",
                        "environment": "production"
                    }
                },
                "spec": {
                    "serviceAccountName": "rqa2025-service-account",
                    "containers": [
                        {
                            "name": "rqa2025-app",
                            "image": "rqa2025:latest",
                            "ports": [
                                {
                                    "containerPort": 8000,
                                    "name": "http"
                                }
                            ],
                            "env": [
                                {
                                    "name": "ENVIRONMENT",
                                    "value": "production"
                                },
                                {
                                    "name": "REDIS_URL",
                                    "value": "redis://redis-service.rqa2025-data:6379"
                                },
                                {
                                    "name": "DB_URL",
                                    "value": "postgresql://user:password@postgres-service.rqa2025-data:5432/rqa2025"
                                }
                            ],
                            "resources": {
                                "requests": {
                                    "memory": "512Mi",
                                    "cpu": "500m"
                                },
                                "limits": {
                                    "memory": "1Gi",
                                    "cpu": "1000m"
                                }
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 8000
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/ready",
                                    "port": 8000
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5
                            }
                        }
                    ]
                }
            }
        }
    }

    # Redis缓存服务部署
    redis_deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": "redis",
            "namespace": "rqa2025-data",
            "labels": {
                "app": "redis",
                "component": "cache",
                "environment": "production"
            }
        },
        "spec": {
            "replicas": 1,
            "selector": {
                "matchLabels": {
                    "app": "redis",
                    "component": "cache"
                }
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": "redis",
                        "component": "cache",
                        "environment": "production"
                    }
                },
                "spec": {
                    "containers": [
                        {
                            "name": "redis",
                            "image": "redis:7-alpine",
                            "ports": [
                                {
                                    "containerPort": 6379,
                                    "name": "redis"
                                }
                            ],
                            "resources": {
                                "requests": {
                                    "memory": "256Mi",
                                    "cpu": "200m"
                                },
                                "limits": {
                                    "memory": "512Mi",
                                    "cpu": "500m"
                                }
                            },
                            "volumeMounts": [
                                {
                                    "name": "redis-data",
                                    "mountPath": "/data"
                                }
                            ]
                        }
                    ],
                    "volumes": [
                        {
                            "name": "redis-data",
                            "persistentVolumeClaim": {
                                "claimName": "redis-pvc"
                            }
                        }
                    ]
                }
            }
        }
    }

    # PostgreSQL数据库部署
    postgres_deployment = {
        "apiVersion": "apps/v1",
        "kind": "StatefulSet",
        "metadata": {
            "name": "postgres",
            "namespace": "rqa2025-data",
            "labels": {
                "app": "postgres",
                "component": "database",
                "environment": "production"
            }
        },
        "spec": {
            "serviceName": "postgres",
            "replicas": 1,
            "selector": {
                "matchLabels": {
                    "app": "postgres",
                    "component": "database"
                }
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": "postgres",
                        "component": "database",
                        "environment": "production"
                    }
                },
                "spec": {
                    "containers": [
                        {
                            "name": "postgres",
                            "image": "postgres:15-alpine",
                            "ports": [
                                {
                                    "containerPort": 5432,
                                    "name": "postgres"
                                }
                            ],
                            "env": [
                                {
                                    "name": "POSTGRES_DB",
                                    "value": "rqa2025"
                                },
                                {
                                    "name": "POSTGRES_USER",
                                    "value": "rqa2025"
                                },
                                {
                                    "name": "POSTGRES_PASSWORD",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": "postgres-secret",
                                            "key": "password"
                                        }
                                    }
                                }
                            ],
                            "resources": {
                                "requests": {
                                    "memory": "1Gi",
                                    "cpu": "500m"
                                },
                                "limits": {
                                    "memory": "2Gi",
                                    "cpu": "1000m"
                                }
                            },
                            "volumeMounts": [
                                {
                                    "name": "postgres-data",
                                    "mountPath": "/var/lib/postgresql/data"
                                }
                            ]
                        }
                    ]
                }
            },
            "volumeClaimTemplates": [
                {
                    "metadata": {
                        "name": "postgres-data"
                    },
                    "spec": {
                        "accessModes": ["ReadWriteOnce"],
                        "storageClassName": "fast-ssd",
                        "resources": {
                            "requests": {
                                "storage": "50Gi"
                            }
                        }
                    }
                }
            ]
        }
    }

    return {
        "rqa_app": rqa_app_deployment,
        "redis": redis_deployment,
        "postgres": postgres_deployment
    }


def create_service_manifests():
    """创建服务配置清单"""
    print("🌐 创建服务配置清单")
    print("=" * 50)

    services = {
        "rqa2025-app-service": {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": "rqa2025-app-service",
                "namespace": "rqa2025-app",
                "labels": {
                    "app": "rqa2025",
                    "component": "app"
                }
            },
            "spec": {
                "type": "ClusterIP",
                "ports": [
                    {
                        "port": 80,
                        "targetPort": 8000,
                        "protocol": "TCP",
                        "name": "http"
                    }
                ],
                "selector": {
                    "app": "rqa2025",
                    "component": "app"
                }
            }
        },
        "redis-service": {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": "redis-service",
                "namespace": "rqa2025-data",
                "labels": {
                    "app": "redis",
                    "component": "cache"
                }
            },
            "spec": {
                "type": "ClusterIP",
                "ports": [
                    {
                        "port": 6379,
                        "targetPort": 6379,
                        "protocol": "TCP",
                        "name": "redis"
                    }
                ],
                "selector": {
                    "app": "redis",
                    "component": "cache"
                }
            }
        },
        "postgres-service": {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": "postgres-service",
                "namespace": "rqa2025-data",
                "labels": {
                    "app": "postgres",
                    "component": "database"
                }
            },
            "spec": {
                "type": "ClusterIP",
                "ports": [
                    {
                        "port": 5432,
                        "targetPort": 5432,
                        "protocol": "TCP",
                        "name": "postgres"
                    }
                ],
                "selector": {
                    "app": "postgres",
                    "component": "database"
                }
            }
        }
    }

    return services


def create_ingress_config():
    """创建Ingress配置"""
    print("🔒 创建Ingress配置")
    print("=" * 50)

    ingress = {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "Ingress",
        "metadata": {
            "name": "rqa2025-ingress",
            "namespace": "rqa2025-app",
            "annotations": {
                "kubernetes.io/ingress.class": "nginx",
                "cert-manager.io/cluster-issuer": "letsencrypt-prod",
                "nginx.ingress.kubernetes.io/ssl-redirect": "true",
                "nginx.ingress.kubernetes.io/rate-limit": "100",
                "nginx.ingress.kubernetes.io/rate-limit-window": "1m"
            }
        },
        "spec": {
            "tls": [
                {
                    "hosts": [
                        "rqa2025.example.com"
                    ],
                    "secretName": "rqa2025-tls"
                }
            ],
            "rules": [
                {
                    "host": "rqa2025.example.com",
                    "http": {
                        "paths": [
                            {
                                "path": "/",
                                "pathType": "Prefix",
                                "backend": {
                                    "service": {
                                        "name": "rqa2025-app-service",
                                        "port": {
                                            "number": 80
                                        }
                                    }
                                }
                            }
                        ]
                    }
                }
            ]
        }
    }

    return ingress


def create_configmaps_and_secrets():
    """创建ConfigMap和Secret配置"""
    print("🔐 创建配置和密钥")
    print("=" * 50)

    configmaps = {
        "rqa2025-config": {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": "rqa2025-config",
                "namespace": "rqa2025-app"
            },
            "data": {
                "APP_ENV": "production",
                "LOG_LEVEL": "INFO",
                "CACHE_TTL": "3600",
                "MAX_CONNECTIONS": "100",
                "TIMEOUT": "30"
            }
        }
    }

    secrets = {
        "postgres-secret": {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {
                "name": "postgres-secret",
                "namespace": "rqa2025-data"
            },
            "type": "Opaque",
            "data": {
                "password": "UjFhMjAyNVNlY3VyZVBhc3N3b3Jk"  # Base64 encoded password
            }
        },
        "jwt-secret": {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {
                "name": "jwt-secret",
                "namespace": "rqa2025-app"
            },
            "type": "Opaque",
            "data": {
                "secret": "UjFhMjAyNUpXVFNlY3JldEtleQ=="  # Base64 encoded JWT secret
            }
        }
    }

    return {
        "configmaps": configmaps,
        "secrets": secrets
    }


def save_manifests_to_files():
    """保存所有清单到文件"""
    print("💾 保存Kubernetes清单文件")
    print("=" * 50)

    # 创建k8s目录
    k8s_dir = Path("k8s/production")
    k8s_dir.mkdir(parents=True, exist_ok=True)

    # 生成所有配置
    k8s_config = create_kubernetes_config()
    deployments = create_deployment_manifests()
    services = create_service_manifests()
    ingress = create_ingress_config()
    configs = create_configmaps_and_secrets()

    # 保存命名空间
    for ns_name, ns_config in k8s_config["namespaces"].items():
        filename = f"k8s/production/namespace-{ns_name}.yaml"
        with open(filename, 'w', encoding='utf-8') as f:
            yaml.dump(ns_config, f, default_flow_style=False, allow_unicode=True)
        print(f"✅ 创建命名空间配置: {filename}")

    # 保存存储类
    for sc_name, sc_config in k8s_config["storage_classes"].items():
        filename = f"k8s/production/storageclass-{sc_name}.yaml"
        with open(filename, 'w', encoding='utf-8') as f:
            yaml.dump(sc_config, f, default_flow_style=False, allow_unicode=True)
        print(f"✅ 创建存储类配置: {filename}")

    # 保存RBAC配置
    for rbac_name, rbac_config in k8s_config["rbac_config"].items():
        filename = f"k8s/production/rbac-{rbac_name}.yaml"
        with open(filename, 'w', encoding='utf-8') as f:
            yaml.dump(rbac_config, f, default_flow_style=False, allow_unicode=True)
        print(f"✅ 创建RBAC配置: {filename}")

    # 保存部署配置
    for deploy_name, deploy_config in deployments.items():
        filename = f"k8s/production/deployment-{deploy_name}.yaml"
        with open(filename, 'w', encoding='utf-8') as f:
            yaml.dump(deploy_config, f, default_flow_style=False, allow_unicode=True)
        print(f"✅ 创建部署配置: {filename}")

    # 保存服务配置
    for svc_name, svc_config in services.items():
        filename = f"k8s/production/service-{svc_name}.yaml"
        with open(filename, 'w', encoding='utf-8') as f:
            yaml.dump(svc_config, f, default_flow_style=False, allow_unicode=True)
        print(f"✅ 创建服务配置: {filename}")

    # 保存Ingress配置
    filename = "k8s/production/ingress-rqa2025.yaml"
    with open(filename, 'w', encoding='utf-8') as f:
        yaml.dump(ingress, f, default_flow_style=False, allow_unicode=True)
    print(f"✅ 创建Ingress配置: {filename}")

    # 保存ConfigMap和Secret
    for config_name, config_config in configs["configmaps"].items():
        filename = f"k8s/production/configmap-{config_name}.yaml"
        with open(filename, 'w', encoding='utf-8') as f:
            yaml.dump(config_config, f, default_flow_style=False, allow_unicode=True)
        print(f"✅ 创建ConfigMap配置: {filename}")

    for secret_name, secret_config in configs["secrets"].items():
        filename = f"k8s/production/secret-{secret_name}.yaml"
        with open(filename, 'w', encoding='utf-8') as f:
            yaml.dump(secret_config, f, default_flow_style=False, allow_unicode=True)
        print(f"✅ 创建Secret配置: {filename}")

    print(f"\n📁 所有清单文件已保存到 {k8s_dir} 目录")


def create_deployment_script():
    """创建部署执行脚本"""
    print("🚀 创建部署执行脚本")
    print("=" * 50)

    deployment_script = """#!/bin/bash
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
kubectl apply -f k8s/production/namespace-*.yaml

# 创建存储类
echo "💾 创建存储类..."
kubectl apply -f k8s/production/storageclass-*.yaml

# 创建RBAC配置
echo "🔐 创建RBAC配置..."
kubectl apply -f k8s/production/rbac-*.yaml

# 创建ConfigMap和Secret
echo "⚙️ 创建配置和密钥..."
kubectl apply -f k8s/production/configmap-*.yaml
kubectl apply -f k8s/production/secret-*.yaml

# 部署数据服务
echo "🗄️ 部署数据服务..."
kubectl apply -f k8s/production/deployment-redis.yaml
kubectl apply -f k8s/production/deployment-postgres.yaml

# 等待数据服务就绪
echo "⏳ 等待数据服务启动..."
kubectl wait --for=condition=available --timeout=300s deployment/redis -n rqa2025-data
kubectl wait --for=condition=available --timeout=300s statefulset/postgres -n rqa2025-data

# 创建数据服务
echo "🌐 创建数据服务..."
kubectl apply -f k8s/production/service-redis-service.yaml
kubectl apply -f k8s/production/service-postgres-service.yaml

# 部署应用服务
echo "🚀 部署应用服务..."
kubectl apply -f k8s/production/deployment-rqa-app.yaml

# 等待应用服务就绪
echo "⏳ 等待应用服务启动..."
kubectl wait --for=condition=available --timeout=300s deployment/rqa2025-app -n rqa2025-app

# 创建应用服务
echo "🌐 创建应用服务..."
kubectl apply -f k8s/production/service-rqa2025-app-service.yaml

# 创建Ingress
echo "🔒 创建Ingress配置..."
kubectl apply -f k8s/production/ingress-rqa2025.yaml

# 验证部署
echo "✅ 验证部署状态..."
kubectl get pods -n rqa2025-app
kubectl get pods -n rqa2025-data
kubectl get services -n rqa2025-app
kubectl get services -n rqa2025-data
kubectl get ingress -n rqa2025-app

# 检查应用健康状态
echo "🏥 检查应用健康状态..."
sleep 30
kubectl logs -n rqa2025-app deployment/rqa2025-app --tail=20

echo "🎉 RQA2025 Kubernetes生产环境部署完成！"
echo "=========================================="
echo "📋 访问地址: https://rqa2025.example.com"
echo "📊 监控命令: kubectl get pods -A"
echo "📝 日志命令: kubectl logs -n rqa2025-app deployment/rqa2025-app"
"""

    with open("scripts/deploy_kubernetes_production.sh", 'w', encoding='utf-8') as f:
        f.write(deployment_script)

    print("✅ 创建部署脚本: scripts/deploy_kubernetes_production.sh")


def create_rollback_script():
    """创建回滚脚本"""
    print("🔄 创建回滚脚本")
    print("=" * 50)

    rollback_script = """#!/bin/bash
# RQA2025 Kubernetes生产环境回滚脚本

set -e

echo "🔄 RQA2025 Kubernetes生产环境回滚开始"
echo "=========================================="

# 确认回滚操作
read -p "⚠️ 确定要回滚到上一个版本吗? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 回滚操作已取消"
    exit 1
fi

# 回滚应用部署
echo "🔄 回滚应用部署..."
kubectl rollout undo deployment/rqa2025-app -n rqa2025-app

# 等待回滚完成
echo "⏳ 等待回滚完成..."
kubectl rollout status deployment/rqa2025-app -n rqa2025-app

# 验证回滚结果
echo "✅ 验证回滚结果..."
kubectl get pods -n rqa2025-app
kubectl logs -n rqa2025-app deployment/rqa2025-app --tail=10

# 检查应用健康状态
echo "🏥 检查应用健康状态..."
sleep 30
kubectl exec -n rqa2025-app deployment/rqa2025-app -- curl -f http://localhost:8000/health || echo "健康检查失败"

echo "✅ RQA2025 Kubernetes生产环境回滚完成！"
echo "=========================================="
"""

    with open("scripts/rollback_kubernetes_production.sh", 'w', encoding='utf-8') as f:
        f.write(rollback_script)

    print("✅ 创建回滚脚本: scripts/rollback_kubernetes_production.sh")


def main():
    """主执行函数"""
    print("🐳 RQA2025 Kubernetes生产环境部署配置")
    print("=" * 60)
    print(f"📅 配置时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}")
    print()

    print("🎯 配置目标:")
    print("  1. 创建生产环境Kubernetes配置")
    print("  2. 生成应用部署和服务清单")
    print("  3. 配置网络和安全策略")
    print("  4. 创建部署和回滚脚本")
    print()

    try:
        # 保存所有清单文件
        save_manifests_to_files()

        # 创建部署脚本
        create_deployment_script()

        # 创建回滚脚本
        create_rollback_script()

        print("
              🎉 Kubernetes生产环境配置完成！"        print("=" * 60)
        print("📋 生成的文件:")
        print("  📁 k8s/production/ - Kubernetes清单文件")
        print("  🚀 scripts/deploy_kubernetes_production.sh - 部署脚本")
        print("  🔄 scripts/rollback_kubernetes_production.sh - 回滚脚本")
        print()
        print("📊 集群规划:")
        print("  🖥️ Master节点: 3个")
        print("  🖥️ Worker节点: 5个")
        print("  📦 命名空间: 4个")
        print("  💾 存储类: 2个")
        print("  🚀 部署应用: 3个")
        print("  🌐 服务配置: 3个")
        print()
        print("🚀 下一步:")
        print("  1. 准备Kubernetes集群环境")
        print("  2. 执行部署脚本")
        print("  3. 验证部署结果")
        print("  4. 配置监控告警体系")

    except Exception as e:
        print(f"❌ 配置过程中出现错误: {e}")
        return False

    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 Phase 4C Week 1-2 Kubernetes部署配置已完成！")
    else:
        print("\n❌ Phase 4C Week 1-2 Kubernetes部署配置失败！")
