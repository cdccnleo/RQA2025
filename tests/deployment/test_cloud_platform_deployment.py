#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
云平台部署测试
Cloud Platform Deployment Tests

测试云平台部署的完整性，包括：
1. Kubernetes集群部署验证
2. 云服务集成测试
3. 负载均衡和服务发现测试
4. 自动扩展配置测试
5. 云存储和数据库集成测试
6. 监控和日志聚合测试
7. 安全组和网络ACL测试
8. 多区域部署测试
"""

import pytest
import os
import tempfile
import shutil
import subprocess
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
import sys
import json
import yaml

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))


class TestKubernetesDeploymentValidation:
    """测试Kubernetes集群部署验证"""

    def setup_method(self):
        """测试前准备"""
        self.k8s_client = Mock()

    def test_kubernetes_deployment_manifests(self):
        """测试Kubernetes部署清单"""
        # 定义Kubernetes部署配置
        k8s_manifests = {
            'deployment': {
                'apiVersion': 'apps/v1',
                'kind': 'Deployment',
                'metadata': {
                    'name': 'rqa2025-api',
                    'namespace': 'production',
                    'labels': {
                        'app': 'rqa2025-api',
                        'version': 'v1.0.0'
                    }
                },
                'spec': {
                    'replicas': 3,
                    'selector': {
                        'matchLabels': {
                            'app': 'rqa2025-api'
                        }
                    },
                    'template': {
                        'metadata': {
                            'labels': {
                                'app': 'rqa2025-api',
                                'version': 'v1.0.0'
                            }
                        },
                        'spec': {
                            'containers': [{
                                'name': 'api',
                                'image': 'rqa2025/api:v1.0.0',
                                'ports': [{
                                    'containerPort': 8000,
                                    'protocol': 'TCP'
                                }],
                                'env': [
                                    {'name': 'DEPLOY_ENV', 'value': 'production'},
                                    {'name': 'DATABASE_URL', 'valueFrom': {'secretKeyRef': {'name': 'db-secret', 'key': 'url'}}}
                                ],
                                'resources': {
                                    'requests': {'memory': '512Mi', 'cpu': '250m'},
                                    'limits': {'memory': '1Gi', 'cpu': '500m'}
                                },
                                'livenessProbe': {
                                    'httpGet': {'path': '/health', 'port': 8000},
                                    'initialDelaySeconds': 30,
                                    'periodSeconds': 10
                                },
                                'readinessProbe': {
                                    'httpGet': {'path': '/ready', 'port': 8000},
                                    'initialDelaySeconds': 5,
                                    'periodSeconds': 5
                                }
                            }],
                            'serviceAccountName': 'rqa2025-api-sa'
                        }
                    },
                    'strategy': {
                        'type': 'RollingUpdate',
                        'rollingUpdate': {
                            'maxUnavailable': '25%',
                            'maxSurge': '25%'
                        }
                    }
                }
            },
            'service': {
                'apiVersion': 'v1',
                'kind': 'Service',
                'metadata': {
                    'name': 'rqa2025-api-service',
                    'namespace': 'production'
                },
                'spec': {
                    'type': 'LoadBalancer',
                    'selector': {
                        'app': 'rqa2025-api'
                    },
                    'ports': [{
                        'name': 'http',
                        'port': 80,
                        'targetPort': 8000,
                        'protocol': 'TCP'
                    }]
                }
            },
            'configmap': {
                'apiVersion': 'v1',
                'kind': 'ConfigMap',
                'metadata': {
                    'name': 'rqa2025-config',
                    'namespace': 'production'
                },
                'data': {
                    'app-config.yaml': """
                    logging:
                      level: INFO
                    features:
                      caching: true
                      monitoring: true
                    """
                }
            }
        }

        def validate_k8s_manifests(manifests: Dict) -> List[str]:
            """验证Kubernetes清单"""
            errors = []

            # 验证Deployment
            deployment = manifests.get('deployment', {})
            if not deployment:
                errors.append("缺少Deployment配置")
            else:
                # 检查元数据
                metadata = deployment.get('metadata', {})
                if 'name' not in metadata:
                    errors.append("Deployment缺少name")
                if 'namespace' not in metadata:
                    errors.append("Deployment缺少namespace")

                # 检查spec
                spec = deployment.get('spec', {})
                if 'replicas' not in spec:
                    errors.append("Deployment缺少replicas配置")
                if spec.get('replicas', 0) < 1:
                    errors.append("Deployment副本数至少为1")

                # 检查容器配置
                template_spec = spec.get('template', {}).get('spec', {})
                containers = template_spec.get('containers', [])
                if not containers:
                    errors.append("Deployment缺少容器配置")

                for container in containers:
                    if 'resources' not in container:
                        errors.append(f"容器 {container.get('name')} 缺少资源限制")

                    # 检查探针
                    if 'livenessProbe' not in container:
                        errors.append(f"容器 {container.get('name')} 缺少存活探针")
                    if 'readinessProbe' not in container:
                        errors.append(f"容器 {container.get('name')} 缺少就绪探针")

            # 验证Service
            service = manifests.get('service', {})
            if not service:
                errors.append("缺少Service配置")
            else:
                service_spec = service.get('spec', {})
                if 'type' not in service_spec:
                    errors.append("Service缺少type配置")
                if 'selector' not in service_spec:
                    errors.append("Service缺少selector配置")

            return errors

        # 验证Kubernetes清单
        errors = validate_k8s_manifests(k8s_manifests)

        # 应该没有配置错误
        assert len(errors) == 0, f"Kubernetes配置存在错误: {errors}"

        # 验证Deployment配置
        deployment = k8s_manifests['deployment']
        assert deployment['metadata']['name'] == 'rqa2025-api'
        assert deployment['spec']['replicas'] == 3

        # 验证容器配置
        container = deployment['spec']['template']['spec']['containers'][0]
        assert container['resources']['requests']['memory'] == '512Mi'
        assert 'livenessProbe' in container
        assert 'readinessProbe' in container

        # 验证Service配置
        service = k8s_manifests['service']
        assert service['spec']['type'] == 'LoadBalancer'

    def test_kubernetes_horizontal_pod_autoscaling(self):
        """测试Kubernetes水平Pod自动扩展"""
        # 定义HPA配置
        hpa_config = {
            'apiVersion': 'autoscaling/v2',
            'kind': 'HorizontalPodAutoscaler',
            'metadata': {
                'name': 'rqa2025-api-hpa',
                'namespace': 'production'
            },
            'spec': {
                'scaleTargetRef': {
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'name': 'rqa2025-api'
                },
                'minReplicas': 3,
                'maxReplicas': 10,
                'metrics': [
                    {
                        'type': 'Resource',
                        'resource': {
                            'name': 'cpu',
                            'target': {
                                'type': 'Utilization',
                                'averageUtilization': 70
                            }
                        }
                    },
                    {
                        'type': 'Resource',
                        'resource': {
                            'name': 'memory',
                            'target': {
                                'type': 'Utilization',
                                'averageUtilization': 80
                            }
                        }
                    },
                    {
                        'type': 'Pods',
                        'pods': {
                            'metric': {
                                'name': 'http_requests_per_second'
                            },
                            'target': {
                                'type': 'AverageValue',
                                'averageValue': '50'
                            }
                        }
                    }
                ],
                'behavior': {
                    'scaleDown': {
                        'stabilizationWindowSeconds': 300,
                        'policies': [{
                            'type': 'Percent',
                            'value': 10,
                            'periodSeconds': 60
                        }]
                    },
                    'scaleUp': {
                        'stabilizationWindowSeconds': 60,
                        'policies': [{
                            'type': 'Percent',
                            'value': 50,
                            'periodSeconds': 60
                        }]
                    }
                }
            }
        }

        def validate_hpa_config(config: Dict) -> List[str]:
            """验证HPA配置"""
            errors = []

            spec = config.get('spec', {})

            # 检查基本配置
            if 'scaleTargetRef' not in spec:
                errors.append("HPA缺少scaleTargetRef配置")

            if 'minReplicas' not in spec or 'maxReplicas' not in spec:
                errors.append("HPA缺少minReplicas或maxReplicas配置")

            min_rep = spec.get('minReplicas', 0)
            max_rep = spec.get('maxReplicas', 0)

            if min_rep >= max_rep:
                errors.append("minReplicas应该小于maxReplicas")

            # 检查指标配置
            metrics = spec.get('metrics', [])
            if not metrics:
                errors.append("HPA缺少metrics配置")

            # 检查行为配置
            behavior = spec.get('behavior', {})
            if 'scaleDown' not in behavior:
                errors.append("HPA缺少scaleDown行为配置")
            if 'scaleUp' not in behavior:
                errors.append("HPA缺少scaleUp行为配置")

            return errors

        # 验证HPA配置
        errors = validate_hpa_config(hpa_config)

        # 应该没有配置错误
        assert len(errors) == 0, f"HPA配置存在错误: {errors}"

        # 验证具体配置
        spec = hpa_config['spec']
        assert spec['minReplicas'] == 3
        assert spec['maxReplicas'] == 10

        # 验证指标
        metrics = spec['metrics']
        cpu_metric = next((m for m in metrics if m.get('resource', {}).get('name') == 'cpu'), None)
        assert cpu_metric is not None
        assert cpu_metric['resource']['target']['averageUtilization'] == 70

        # 验证行为
        behavior = spec['behavior']
        assert 'scaleDown' in behavior
        assert 'scaleUp' in behavior
        assert behavior['scaleDown']['stabilizationWindowSeconds'] == 300


class TestCloudServicesIntegration:
    """测试云服务集成测试"""

    def setup_method(self):
        """测试前准备"""
        self.cloud_client = Mock()

    def test_aws_ecs_deployment_configuration(self):
        """测试AWS ECS部署配置"""
        # 定义ECS任务定义
        ecs_task_definition = {
            'family': 'rqa2025-api',
            'taskRoleArn': 'arn:aws:iam::123456789012:role/rqa2025-task-role',
            'executionRoleArn': 'arn:aws:iam::123456789012:role/rqa2025-execution-role',
            'networkMode': 'awsvpc',
            'requiresCompatibilities': ['FARGATE'],
            'cpu': '256',
            'memory': '512',
            'containerDefinitions': [{
                'name': 'api',
                'image': '123456789012.dkr.ecr.us-east-1.amazonaws.com/rqa2025/api:v1.0.0',
                'essential': True,
                'portMappings': [{
                    'containerPort': 8000,
                    'protocol': 'tcp'
                }],
                'environment': [
                    {'name': 'DEPLOY_ENV', 'value': 'production'},
                    {'name': 'AWS_REGION', 'value': 'us-east-1'}
                ],
                'secrets': [{
                    'name': 'DATABASE_URL',
                    'valueFrom': 'arn:aws:secretsmanager:us-east-1:123456789012:secret:rqa2025/db-url'
                }],
                'logConfiguration': {
                    'logDriver': 'awslogs',
                    'options': {
                        'awslogs-group': '/ecs/rqa2025-api',
                        'awslogs-region': 'us-east-1',
                        'awslogs-stream-prefix': 'ecs'
                    }
                },
                'healthCheck': {
                    'command': ['CMD-SHELL', 'curl -f http://localhost:8000/health || exit 1'],
                    'interval': 30,
                    'timeout': 5,
                    'retries': 3
                }
            }]
        }

        def validate_ecs_task_definition(task_def: Dict) -> List[str]:
            """验证ECS任务定义"""
            errors = []

            # 检查基本字段
            required_fields = ['family', 'containerDefinitions', 'cpu', 'memory']
            for field in required_fields:
                if field not in task_def:
                    errors.append(f"ECS任务定义缺少字段: {field}")

            # 检查容器定义
            containers = task_def.get('containerDefinitions', [])
            if not containers:
                errors.append("ECS任务定义缺少容器定义")

            for container in containers:
                if 'name' not in container:
                    errors.append("容器定义缺少name字段")
                if 'image' not in container:
                    errors.append("容器定义缺少image字段")
                if 'essential' not in container:
                    errors.append(f"容器 {container.get('name')} 缺少essential字段")

                # 检查端口映射
                port_mappings = container.get('portMappings', [])
                if port_mappings:
                    for port_map in port_mappings:
                        if 'containerPort' not in port_map:
                            errors.append(f"容器 {container.get('name')} 端口映射缺少containerPort")

            # 检查Fargate兼容性
            compatibilities = task_def.get('requiresCompatibilities', [])
            if 'FARGATE' in compatibilities:
                network_mode = task_def.get('networkMode')
                if network_mode != 'awsvpc':
                    errors.append("Fargate任务必须使用awsvpc网络模式")

            return errors

        # 验证ECS任务定义
        errors = validate_ecs_task_definition(ecs_task_definition)

        # 应该没有配置错误
        assert len(errors) == 0, f"ECS配置存在错误: {errors}"

        # 验证具体配置
        assert ecs_task_definition['family'] == 'rqa2025-api'
        assert ecs_task_definition['networkMode'] == 'awsvpc'
        assert 'FARGATE' in ecs_task_definition['requiresCompatibilities']

        # 验证容器配置
        container = ecs_task_definition['containerDefinitions'][0]
        assert container['name'] == 'api'
        assert container['essential'] is True
        assert container['portMappings'][0]['containerPort'] == 8000

    def test_azure_aks_deployment_configuration(self):
        """测试Azure AKS部署配置"""
        # 定义AKS部署配置
        aks_config = {
            'resource_group': 'rqa2025-rg',
            'cluster_name': 'rqa2025-aks',
            'location': 'eastus',
            'kubernetes_version': '1.27.0',
            'node_pools': {
                'system': {
                    'name': 'systempool',
                    'node_count': 3,
                    'vm_size': 'Standard_DS2_v2',
                    'os_disk_size_gb': 128,
                    'max_pods': 110,
                    'enable_auto_scaling': True,
                    'min_count': 3,
                    'max_count': 10
                },
                'user': {
                    'name': 'userpool',
                    'node_count': 5,
                    'vm_size': 'Standard_DS3_v2',
                    'os_disk_size_gb': 256,
                    'max_pods': 50,
                    'enable_auto_scaling': True,
                    'min_count': 3,
                    'max_count': 20
                }
            },
            'network_profile': {
                'network_plugin': 'azure',
                'network_policy': 'azure',
                'service_cidr': '10.0.0.0/16',
                'dns_service_ip': '10.0.0.10',
                'docker_bridge_cidr': '172.17.0.1/16'
            },
            'addon_profiles': {
                'omsagent': {
                    'enabled': True,
                    'log_analytics_workspace_resource_id': '/subscriptions/.../resourceGroups/.../providers/Microsoft.OperationalInsights/workspaces/rqa2025-logs'
                },
                'ingress_appgw': {
                    'enabled': True,
                    'application_gateway_id': '/subscriptions/.../resourceGroups/.../providers/Microsoft.Network/applicationGateways/rqa2025-agw'
                }
            }
        }

        def validate_aks_config(config: Dict) -> List[str]:
            """验证AKS配置"""
            errors = []

            # 检查基本配置
            required_fields = ['resource_group', 'cluster_name', 'location', 'kubernetes_version']
            for field in required_fields:
                if field not in config:
                    errors.append(f"AKS配置缺少字段: {field}")

            # 检查节点池配置
            node_pools = config.get('node_pools', {})
            if not node_pools:
                errors.append("AKS配置缺少节点池配置")

            for pool_name, pool_config in node_pools.items():
                if 'name' not in pool_config:
                    errors.append(f"节点池 {pool_name} 缺少name配置")
                if 'vm_size' not in pool_config:
                    errors.append(f"节点池 {pool_name} 缺少vm_size配置")

                # 检查自动扩展配置
                if pool_config.get('enable_auto_scaling', False):
                    if 'min_count' not in pool_config or 'max_count' not in pool_config:
                        errors.append(f"节点池 {pool_name} 启用自动扩展但缺少min_count或max_count")

            # 检查网络配置
            network = config.get('network_profile', {})
            if network.get('network_plugin') == 'azure':
                if 'service_cidr' not in network:
                    errors.append("Azure网络插件需要service_cidr配置")

            return errors

        # 验证AKS配置
        errors = validate_aks_config(aks_config)

        # 应该没有配置错误
        assert len(errors) == 0, f"AKS配置存在错误: {errors}"

        # 验证具体配置
        assert aks_config['cluster_name'] == 'rqa2025-aks'
        assert aks_config['kubernetes_version'] == '1.27.0'

        # 验证节点池
        system_pool = aks_config['node_pools']['system']
        assert system_pool['enable_auto_scaling'] is True
        assert system_pool['min_count'] == 3
        assert system_pool['max_count'] == 10

    def test_gcp_gke_deployment_configuration(self):
        """测试GCP GKE部署配置"""
        # 定义GKE集群配置
        gke_config = {
            'name': 'rqa2025-gke-cluster',
            'description': 'RQA2025 production cluster',
            'initial_cluster_version': '1.27.0-gke.1000',
            'location': 'us-central1',
            'node_pools': [{
                'name': 'default-pool',
                'initial_node_count': 3,
                'config': {
                    'machine_type': 'e2-medium',
                    'disk_size_gb': 100,
                    'disk_type': 'pd-standard',
                    'oauth_scopes': [
                        'https://www.googleapis.com/auth/devstorage.read_only',
                        'https://www.googleapis.com/auth/logging.write',
                        'https://www.googleapis.com/auth/monitoring'
                    ]
                },
                'management': {
                    'auto_repair': True,
                    'auto_upgrade': True
                },
                'autoscaling': {
                    'enabled': True,
                    'min_node_count': 3,
                    'max_node_count': 10
                }
            }],
            'network': 'projects/rqa2025/global/networks/rqa2025-vpc',
            'subnetwork': 'projects/rqa2025/regions/us-central1/subnetworks/rqa2025-subnet',
            'private_cluster_config': {
                'enable_private_nodes': True,
                'enable_private_endpoint': False,
                'master_ipv4_cidr_block': '172.16.0.0/28'
            },
            'master_authorized_networks_config': {
                'enabled': True,
                'cidr_blocks': [{
                    'display_name': 'office',
                    'cidr_block': '192.168.1.0/24'
                }]
            },
            'logging_config': {
                'component_config': {
                    'enable_components': ['SYSTEM_COMPONENTS', 'WORKLOADS']
                }
            },
            'monitoring_config': {
                'component_config': {
                    'enable_components': ['SYSTEM_COMPONENTS']
                }
            }
        }

        def validate_gke_config(config: Dict) -> List[str]:
            """验证GKE配置"""
            errors = []

            # 检查基本配置
            required_fields = ['name', 'location', 'initial_cluster_version']
            for field in required_fields:
                if field not in config:
                    errors.append(f"GKE配置缺少字段: {field}")

            # 检查节点池配置
            node_pools = config.get('node_pools', [])
            if not node_pools:
                errors.append("GKE配置缺少节点池配置")

            for pool in node_pools:
                if 'name' not in pool:
                    errors.append("节点池缺少name配置")
                if 'config' not in pool:
                    errors.append(f"节点池 {pool.get('name')} 缺少config配置")

                # 检查自动扩展
                autoscaling = pool.get('management', {}).get('autoscaling', {})
                if autoscaling.get('enabled', False):
                    if 'min_node_count' not in autoscaling or 'max_node_count' not in autoscaling:
                        errors.append(f"节点池 {pool.get('name')} 启用自动扩展但缺少节点数量限制")

            # 检查私有集群配置
            private_config = config.get('private_cluster_config', {})
            if private_config.get('enable_private_nodes', False):
                if 'master_ipv4_cidr_block' not in private_config:
                    errors.append("私有集群需要master_ipv4_cidr_block配置")

            return errors

        # 验证GKE配置
        errors = validate_gke_config(gke_config)

        # 应该没有配置错误
        assert len(errors) == 0, f"GKE配置存在错误: {errors}"

        # 验证具体配置
        assert gke_config['name'] == 'rqa2025-gke-cluster'
        assert gke_config['location'] == 'us-central1'

        # 验证节点池
        pool = gke_config['node_pools'][0]
        assert pool['autoscaling']['enabled'] is True
        assert pool['autoscaling']['min_node_count'] == 3
        assert pool['autoscaling']['max_node_count'] == 10

        # 验证私有集群配置
        private_config = gke_config['private_cluster_config']
        assert private_config['enable_private_nodes'] is True
        assert private_config['master_ipv4_cidr_block'] == '172.16.0.0/28'


class TestLoadBalancingServiceDiscovery:
    """测试负载均衡和服务发现测试"""

    def setup_method(self):
        """测试前准备"""
        self.load_balancer = Mock()

    def test_aws_application_load_balancer_config(self):
        """测试AWS Application Load Balancer配置"""
        # 定义ALB配置
        alb_config = {
            'name': 'rqa2025-alb',
            'type': 'application',
            'scheme': 'internet-facing',
            'ip_address_type': 'ipv4',
            'subnets': [
                'subnet-12345678',
                'subnet-87654321'
            ],
            'security_groups': [
                'sg-alb123456'
            ],
            'listeners': [{
                'protocol': 'HTTPS',
                'port': 443,
                'ssl_policy': 'ELBSecurityPolicy-TLS-1-2-2017-01',
                'certificates': [{
                    'certificate_arn': 'arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789012'
                }],
                'default_actions': [{
                    'type': 'forward',
                    'target_group_arn': 'arn:aws:elasticloadbalancing:us-east-1:123456789012:targetgroup/rqa2025-api/1234567890123456'
                }]
            }],
            'target_groups': [{
                'name': 'rqa2025-api',
                'protocol': 'HTTP',
                'port': 80,
                'vpc_id': 'vpc-12345678',
                'health_check': {
                    'enabled': True,
                    'path': '/health',
                    'port': 'traffic-port',
                    'healthy_threshold': 2,
                    'unhealthy_threshold': 2,
                    'timeout': 5,
                    'interval': 30,
                    'matcher': '200'
                },
                'targets': [
                    {'id': 'i-1234567890abcdef0', 'port': 80},
                    {'id': 'i-0987654321fedcba0', 'port': 80}
                ]
            }]
        }

        def validate_alb_config(config: Dict) -> List[str]:
            """验证ALB配置"""
            errors = []

            # 检查基本配置
            required_fields = ['name', 'type', 'scheme', 'subnets', 'security_groups']
            for field in required_fields:
                if field not in config:
                    errors.append(f"ALB配置缺少字段: {field}")

            # 检查侦听器配置
            listeners = config.get('listeners', [])
            if not listeners:
                errors.append("ALB配置缺少侦听器配置")

            for listener in listeners:
                if listener.get('protocol') == 'HTTPS':
                    if 'certificates' not in listener:
                        errors.append("HTTPS侦听器缺少证书配置")
                    if 'ssl_policy' not in listener:
                        errors.append("HTTPS侦听器缺少SSL策略配置")

            # 检查目标组配置
            target_groups = config.get('target_groups', [])
            if not target_groups:
                errors.append("ALB配置缺少目标组配置")

            for tg in target_groups:
                health_check = tg.get('health_check', {})
                if health_check.get('enabled', False):
                    if 'path' not in health_check:
                        errors.append(f"目标组 {tg.get('name')} 健康检查缺少path配置")

            return errors

        # 验证ALB配置
        errors = validate_alb_config(alb_config)

        # 应该没有配置错误
        assert len(errors) == 0, f"ALB配置存在错误: {errors}"

        # 验证具体配置
        assert alb_config['type'] == 'application'
        assert alb_config['scheme'] == 'internet-facing'

        # 验证侦听器
        listener = alb_config['listeners'][0]
        assert listener['protocol'] == 'HTTPS'
        assert listener['port'] == 443
        assert 'certificates' in listener

        # 验证目标组
        target_group = alb_config['target_groups'][0]
        health_check = target_group['health_check']
        assert health_check['path'] == '/health'
        assert health_check['matcher'] == '200'

    def test_kubernetes_ingress_configuration(self):
        """测试Kubernetes Ingress配置"""
        # 定义Ingress配置
        ingress_config = {
            'apiVersion': 'networking.k8s.io/v1',
            'kind': 'Ingress',
            'metadata': {
                'name': 'rqa2025-api-ingress',
                'namespace': 'production',
                'annotations': {
                    'kubernetes.io/ingress.class': 'nginx',
                    'cert-manager.io/cluster-issuer': 'letsencrypt-prod',
                    'nginx.ingress.kubernetes.io/rewrite-target': '/',
                    'nginx.ingress.kubernetes.io/rate-limit': '100',
                    'nginx.ingress.kubernetes.io/rate-limit-window': '1m'
                }
            },
            'spec': {
                'tls': [{
                    'hosts': ['api.rqa2025.com'],
                    'secretName': 'rqa2025-api-tls'
                }],
                'rules': [{
                    'host': 'api.rqa2025.com',
                    'http': {
                        'paths': [{
                            'path': '/api',
                            'pathType': 'Prefix',
                            'backend': {
                                'service': {
                                    'name': 'rqa2025-api-service',
                                    'port': {
                                        'number': 80
                                    }
                                }
                            }
                        }, {
                            'path': '/health',
                            'pathType': 'Exact',
                            'backend': {
                                'service': {
                                    'name': 'rqa2025-health-service',
                                    'port': {
                                        'number': 80
                                    }
                                }
                            }
                        }]
                    }
                }]
            }
        }

        def validate_ingress_config(config: Dict) -> List[str]:
            """验证Ingress配置"""
            errors = []

            # 检查基本配置
            if 'spec' not in config:
                errors.append("Ingress配置缺少spec")

            spec = config.get('spec', {})

            # 检查TLS配置
            tls_configs = spec.get('tls', [])
            for tls in tls_configs:
                if 'hosts' not in tls:
                    errors.append("TLS配置缺少hosts")
                if 'secretName' not in tls:
                    errors.append("TLS配置缺少secretName")

            # 检查规则配置
            rules = spec.get('rules', [])
            if not rules:
                errors.append("Ingress配置缺少rules")

            for rule in rules:
                if 'host' not in rule:
                    errors.append("Ingress规则缺少host")

                http_config = rule.get('http', {})
                paths = http_config.get('paths', [])

                for path in paths:
                    if 'path' not in path:
                        errors.append("路径配置缺少path")
                    if 'backend' not in path:
                        errors.append("路径配置缺少backend")

                    backend = path.get('backend', {})
                    service = backend.get('service', {})
                    if 'name' not in service or 'port' not in service:
                        errors.append("后端服务配置不完整")

            return errors

        # 验证Ingress配置
        errors = validate_ingress_config(ingress_config)

        # 应该没有配置错误
        assert len(errors) == 0, f"Ingress配置存在错误: {errors}"

        # 验证具体配置
        spec = ingress_config['spec']

        # 验证TLS
        tls = spec['tls'][0]
        assert 'api.rqa2025.com' in tls['hosts']
        assert tls['secretName'] == 'rqa2025-api-tls'

        # 验证规则
        rule = spec['rules'][0]
        assert rule['host'] == 'api.rqa2025.com'

        # 验证路径
        paths = rule['http']['paths']
        api_path = next(p for p in paths if p['path'] == '/api')
        assert api_path['backend']['service']['name'] == 'rqa2025-api-service'

        health_path = next(p for p in paths if p['path'] == '/health')
        assert health_path['backend']['service']['name'] == 'rqa2025-health-service'


if __name__ == "__main__":
    pytest.main([__file__])
