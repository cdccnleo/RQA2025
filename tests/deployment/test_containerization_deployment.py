#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
容器化部署测试
Containerization Deployment Tests

测试Docker容器化部署的完整性，包括：
1. Docker镜像构建验证
2. 容器运行时配置测试
3. 容器编排和网络测试
4. 容器健康检查测试
5. 容器日志和监控测试
6. 多容器应用集成测试
7. 容器安全配置测试
8. 容器资源管理测试
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


class TestDockerImageBuildValidation:
    """测试Docker镜像构建验证"""

    def setup_method(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.image_builder = Mock()

    def teardown_method(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_dockerfile_layer_optimization(self):
        """测试Dockerfile层优化"""
        # 创建测试Dockerfile
        dockerfile_content = """# 多阶段构建示例
FROM python:3.9-slim as builder

# 安装构建依赖
RUN apt-get update && apt-get install -y \\
    build-essential \\
    libpq-dev \\
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /build

# 复制依赖文件
COPY requirements.txt requirements-prod.txt ./

# 创建虚拟环境并安装依赖
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-prod.txt

# 生产镜像
FROM python:3.9-slim as production

# 安装运行时依赖
RUN apt-get update && apt-get install -y \\
    libpq5 \\
    && rm -rf /var/lib/apt/lists/* \\
    && useradd --create-home --shell /bin/bash app

# 复制虚拟环境
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 创建应用目录
WORKDIR /app

# 复制应用代码
COPY src/ ./src/

# 设置目录权限
RUN chown -R app:app /app

# 切换到非root用户
USER app

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# 启动命令
CMD ["python", "-m", "src.main"]
"""

        dockerfile_path = Path(self.temp_dir) / "Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)

        # 创建模拟的requirements文件
        for req_file in ['requirements.txt', 'requirements-prod.txt']:
            req_path = Path(self.temp_dir) / req_file
            with open(req_path, 'w') as f:
                f.write("fastapi==0.104.1\nuvicorn==0.24.0\nsqlalchemy==2.0.23\n")

        # 创建模拟的src目录
        src_dir = Path(self.temp_dir) / "src"
        src_dir.mkdir()
        (src_dir / "__init__.py").touch()
        (src_dir / "main.py").write_text('print("RQA2025 Application")')

        # 分析Dockerfile优化
        def analyze_dockerfile_optimization(dockerfile_path: Path) -> Dict:
            """分析Dockerfile优化情况"""
            with open(dockerfile_path, 'r') as f:
                content = f.read()

            analysis = {
                'multi_stage_build': False,
                'layer_optimization': True,
                'security_best_practices': True,
                'caching_optimization': True,
                'issues': []
            }

            # 检查多阶段构建
            if 'as builder' in content and 'FROM builder' in content:
                analysis['multi_stage_build'] = True
            else:
                analysis['issues'].append('未使用多阶段构建')

            # 检查RUN指令优化
            run_commands = [line for line in content.split('\n') if line.strip().startswith('RUN')]
            for cmd in run_commands:
                # 检查是否清理包管理器缓存
                if 'apt-get' in cmd and 'rm -rf /var/lib/apt/lists/*' not in cmd:
                    analysis['issues'].append('apt-get指令未清理缓存')
                    analysis['caching_optimization'] = False

            # 检查安全实践
            if 'USER root' in content or content.count('USER ') == 0:
                analysis['issues'].append('未切换到非root用户')
                analysis['security_best_practices'] = False

            # 检查健康检查
            if 'HEALTHCHECK' not in content:
                analysis['issues'].append('缺少健康检查')

            return analysis

        # 分析Dockerfile
        analysis = analyze_dockerfile_optimization(dockerfile_path)

        # 验证优化结果
        assert analysis['multi_stage_build'], "应该使用多阶段构建"
        assert analysis['caching_optimization'], "应该优化层缓存"
        assert analysis['security_best_practices'], "应该遵循安全最佳实践"
        assert len(analysis['issues']) == 0, f"Dockerfile存在优化问题: {analysis['issues']}"

    @patch('subprocess.run')
    def test_docker_image_build_process(self, mock_run):
        """测试Docker镜像构建过程"""
        # 模拟docker build命令
        mock_run.return_value = Mock(returncode=0, stdout='Successfully built image', stderr='')

        # 测试镜像构建参数
        build_context = Path(self.temp_dir)
        dockerfile_path = build_context / "Dockerfile"
        image_name = "rqa2025/api"
        image_tag = "v1.0.0"

        # 创建基本的Dockerfile
        with open(dockerfile_path, 'w') as f:
            f.write("""
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
""")

        # 创建模拟的应用文件
        (build_context / "requirements.txt").write_text("fastapi\nuvicorn\n")
        (build_context / "app.py").write_text("print('Hello RQA2025')")

        # 模拟构建过程
        def build_docker_image(context_path: Path, dockerfile: str, name: str, tag: str,
                             build_args: Dict = None) -> bool:
            """构建Docker镜像"""
            cmd = [
                'docker', 'build',
                '-f', dockerfile,
                '-t', f"{name}:{tag}",
                str(context_path)
            ]

            # 添加构建参数
            if build_args:
                for arg_name, arg_value in build_args.items():
                    cmd.extend(['--build-arg', f"{arg_name}={arg_value}"])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            return result.returncode == 0

        # 测试基本构建
        build_args = {
            'BUILD_ENV': 'production',
            'PYTHON_VERSION': '3.9'
        }

        success = build_docker_image(build_context, "Dockerfile", image_name, image_tag, build_args)

        # 验证构建调用
        assert mock_run.called, "应该调用docker build命令"

        # 检查构建参数
        call_args = mock_run.call_args[0][0]
        assert 'docker' in call_args
        assert 'build' in call_args
        assert '-f' in call_args
        assert 'Dockerfile' in call_args
        assert '-t' in call_args
        assert f"{image_name}:{image_tag}" in call_args

        # 验证构建参数
        assert '--build-arg' in call_args
        assert 'BUILD_ENV=production' in ' '.join(call_args)
        assert 'PYTHON_VERSION=3.9' in ' '.join(call_args)


class TestContainerRuntimeConfiguration:
    """测试容器运行时配置测试"""

    def setup_method(self):
        """测试前准备"""
        self.container_manager = Mock()

    @patch('subprocess.run')
    def test_container_environment_variables(self, mock_run):
        """测试容器环境变量配置"""
        # 模拟docker run命令
        mock_run.return_value = Mock(returncode=0, stdout='Container started', stderr='')

        # 定义容器配置
        container_config = {
            'image': 'rqa2025/api:v1.0.0',
            'name': 'rqa2025-api-prod',
            'ports': ['8000:8000'],
            'environment': {
                'DEPLOY_ENV': 'production',
                'DATABASE_URL': 'postgresql://prod-db:5432/rqa2025',
                'REDIS_URL': 'redis://redis-cluster:6379/0',
                'SECRET_KEY': 'prod-secret-key',
                'LOG_LEVEL': 'WARNING'
            },
            'volumes': [
                '/opt/rqa2025/logs:/app/logs',
                '/opt/rqa2025/config:/app/config:ro'
            ],
            'networks': ['rqa2025-network'],
            'restart_policy': 'unless-stopped'
        }

        def run_container_with_config(config: Dict) -> List[str]:
            """使用配置运行容器"""
            cmd = ['docker', 'run', '-d', '--name', config['name']]

            # 添加端口映射
            for port_mapping in config['ports']:
                cmd.extend(['-p', port_mapping])

            # 添加环境变量
            for env_name, env_value in config['environment'].items():
                cmd.extend(['-e', f"{env_name}={env_value}"])

            # 添加卷挂载
            for volume_mapping in config['volumes']:
                cmd.extend(['-v', volume_mapping])

            # 添加网络
            for network in config['networks']:
                cmd.extend(['--network', network])

            # 添加重启策略
            cmd.extend(['--restart', config['restart_policy']])

            # 添加镜像
            cmd.append(config['image'])

            return cmd

        # 生成运行命令
        run_cmd = run_container_with_config(container_config)

        # 验证命令结构
        assert 'docker' in run_cmd
        assert 'run' in run_cmd
        assert '-d' in run_cmd
        assert '--name' in run_cmd
        assert 'rqa2025-api-prod' in run_cmd

        # 验证端口映射
        assert '-p' in run_cmd
        assert '8000:8000' in run_cmd

        # 验证环境变量
        env_vars_in_cmd = [arg for arg in run_cmd if arg.startswith('-e')]
        assert len(env_vars_in_cmd) == len(container_config['environment'])

        # 检查关键环境变量
        cmd_str = ' '.join(run_cmd)
        assert 'DEPLOY_ENV=production' in cmd_str
        assert 'DATABASE_URL=' in cmd_str
        assert 'SECRET_KEY=' in cmd_str

        # 验证卷挂载
        assert '-v' in run_cmd
        assert '/opt/rqa2025/logs:/app/logs' in cmd_str
        assert '/app/config:ro' in cmd_str

        # 验证网络配置
        assert '--network' in run_cmd
        assert 'rqa2025-network' in cmd_str

        # 验证重启策略
        assert '--restart' in cmd_str
        assert 'unless-stopped' in cmd_str

    def test_container_resource_limits(self):
        """测试容器资源限制"""
        # 定义资源限制配置
        resource_limits = {
            'cpu_shares': 1024,
            'cpu_quota': 100000,
            'cpu_period': 100000,
            'memory': '1g',
            'memory_swap': '2g',
            'memory_reservation': '512m',
            'kernel_memory': '256m',
            'cpu_count': 2,
            'cpu_percent': 50,
            'blkio_weight': 500,
            'oom_kill_disable': False,
            'shm_size': '128m'
        }

        def generate_resource_limit_args(limits: Dict) -> List[str]:
            """生成资源限制参数"""
            args = []

            # CPU限制
            if 'cpu_shares' in limits:
                args.extend(['--cpu-shares', str(limits['cpu_shares'])])
            if 'cpu_quota' in limits and 'cpu_period' in limits:
                args.extend(['--cpu-quota', str(limits['cpu_quota'])])
                args.extend(['--cpu-period', str(limits['cpu_period'])])
            if 'cpu_count' in limits:
                args.extend(['--cpus', str(limits['cpu_count'])])

            # 内存限制
            if 'memory' in limits:
                args.extend(['--memory', limits['memory']])
            if 'memory_swap' in limits:
                args.extend(['--memory-swap', limits['memory_swap']])
            if 'memory_reservation' in limits:
                args.extend(['--memory-reservation', limits['memory_reservation']])
            if 'kernel_memory' in limits:
                args.extend(['--kernel-memory', limits['kernel_memory']])

            # I/O限制
            if 'blkio_weight' in limits:
                args.extend(['--blkio-weight', str(limits['blkio_weight'])])

            # OOM控制
            if limits.get('oom_kill_disable') is False:
                args.extend(['--oom-kill-disable', 'false'])
            elif limits.get('oom_kill_disable') is True:
                args.extend(['--oom-kill-disable', 'true'])

            # 共享内存
            if 'shm_size' in limits:
                args.extend(['--shm-size', limits['shm_size']])

            return args

        # 生成资源限制参数
        resource_args = generate_resource_limit_args(resource_limits)

        # 验证资源限制参数
        assert '--cpu-shares' in resource_args
        assert '1024' in resource_args
        assert '--cpus' in resource_args
        assert '2' in resource_args
        assert '--memory' in resource_args
        assert '1g' in resource_args
        assert '--memory-swap' in resource_args
        assert '2g' in resource_args
        assert '--blkio-weight' in resource_args
        assert '500' in resource_args
        assert '--shm-size' in resource_args
        assert '128m' in resource_args

        # 验证参数顺序和完整性
        expected_args = [
            '--cpu-shares', '1024',
            '--cpu-quota', '100000',
            '--cpu-period', '100000',
            '--cpus', '2',
            '--memory', '1g',
            '--memory-swap', '2g',
            '--memory-reservation', '512m',
            '--kernel-memory', '256m',
            '--blkio-weight', '500',
            '--oom-kill-disable', 'false',
            '--shm-size', '128m'
        ]

        assert resource_args == expected_args, f"资源参数不正确: {resource_args}"


class TestContainerOrchestrationNetworking:
    """测试容器编排和网络测试"""

    def setup_method(self):
        """测试前准备"""
        self.orchestrator = Mock()

    def test_docker_compose_service_definition(self):
        """测试Docker Compose服务定义"""
        # 创建测试docker-compose.yml
        compose_config = {
            'version': '3.8',
            'services': {
                'api': {
                    'image': 'rqa2025/api:v1.0.0',
                    'ports': ['8000:8000'],
                    'environment': {
                        'DEPLOY_ENV': 'production',
                        'DATABASE_URL': '${DATABASE_URL}'
                    },
                    'depends_on': ['database', 'redis'],
                    'healthcheck': {
                        'test': ['CMD', 'curl', '-f', 'http://localhost:8000/health'],
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': 3,
                        'start_period': '40s'
                    },
                    'deploy': {
                        'replicas': 3,
                        'restart_policy': {
                            'condition': 'on-failure',
                            'delay': '5s',
                            'max_attempts': 3,
                            'window': '120s'
                        },
                        'resources': {
                            'limits': {
                                'cpus': '1.0',
                                'memory': '1G'
                            },
                            'reservations': {
                                'cpus': '0.5',
                                'memory': '512M'
                            }
                        }
                    }
                },
                'database': {
                    'image': 'postgres:13',
                    'environment': {
                        'POSTGRES_DB': 'rqa2025',
                        'POSTGRES_USER': 'rqa2025',
                        'POSTGRES_PASSWORD': '${DB_PASSWORD}'
                    },
                    'volumes': ['postgres_data:/var/lib/postgresql/data'],
                    'ports': ['5432:5432']
                },
                'redis': {
                    'image': 'redis:7-alpine',
                    'command': 'redis-server --appendonly yes',
                    'volumes': ['redis_data:/data'],
                    'ports': ['6379:6379']
                }
            },
            'volumes': {
                'postgres_data': {},
                'redis_data': {}
            },
            'networks': {
                'default': {
                    'driver': 'bridge'
                }
            }
        }

        def validate_compose_config(config: Dict) -> List[str]:
            """验证Compose配置"""
            errors = []

            # 检查必需字段
            if 'version' not in config:
                errors.append("缺少version字段")
            if 'services' not in config:
                errors.append("缺少services字段")

            services = config.get('services', {})

            # 验证API服务
            if 'api' not in services:
                errors.append("缺少api服务定义")
            else:
                api_service = services['api']

                # 检查依赖关系
                depends_on = api_service.get('depends_on', [])
                if 'database' not in depends_on:
                    errors.append("API服务应该依赖database")
                if 'redis' not in depends_on:
                    errors.append("API服务应该依赖redis")

                # 检查健康检查
                if 'healthcheck' not in api_service:
                    errors.append("API服务缺少健康检查配置")

                # 检查部署配置
                deploy = api_service.get('deploy', {})
                if 'replicas' not in deploy:
                    errors.append("API服务缺少副本数配置")
                if 'resources' not in deploy:
                    errors.append("API服务缺少资源限制配置")

            # 验证数据库服务
            if 'database' in services:
                db_service = services['database']
                required_env = ['POSTGRES_DB', 'POSTGRES_USER', 'POSTGRES_PASSWORD']
                env_vars = db_service.get('environment', {})

                for env_var in required_env:
                    if env_var not in env_vars:
                        errors.append(f"数据库服务缺少环境变量: {env_var}")

            # 验证网络配置
            networks = config.get('networks', {})
            if not networks:
                errors.append("缺少网络配置")

            return errors

        # 验证Compose配置
        errors = validate_compose_config(compose_config)

        # 应该没有配置错误
        assert len(errors) == 0, f"Compose配置存在错误: {errors}"

        # 验证服务依赖关系
        api_service = compose_config['services']['api']
        depends_on = api_service.get('depends_on', [])
        assert 'database' in depends_on
        assert 'redis' in depends_on

        # 验证资源配置
        deploy = api_service.get('deploy', {})
        resources = deploy.get('resources', {})
        assert 'limits' in resources
        assert 'reservations' in resources

    def test_container_networking_configuration(self):
        """测试容器网络配置"""
        # 定义网络配置
        network_config = {
            'bridge_network': {
                'driver': 'bridge',
                'driver_opts': {
                    'com.docker.network.bridge.name': 'rqa2025_bridge'
                },
                'ipam': {
                    'driver': 'default',
                    'config': [{
                        'subnet': '172.20.0.0/16',
                        'gateway': '172.20.0.1'
                    }]
                },
                'internal': False,
                'attachable': True,
                'ingress': False,
                'labels': {
                    'project': 'rqa2025',
                    'environment': 'production'
                }
            },
            'overlay_network': {
                'driver': 'overlay',
                'driver_opts': {},
                'ipam': {
                    'driver': 'default',
                    'config': [{
                        'subnet': '10.0.9.0/24'
                    }]
                },
                'internal': False,
                'attachable': True,
                'ingress': False,
                'labels': {
                    'com.docker.swarm.service.name': 'rqa2025_stack'
                }
            }
        }

        def validate_network_config(networks: Dict) -> List[str]:
            """验证网络配置"""
            errors = []

            for network_name, network_def in networks.items():
                # 检查驱动
                if 'driver' not in network_def:
                    errors.append(f"网络 {network_name} 缺少driver配置")

                driver = network_def.get('driver')

                # 桥接网络验证
                if driver == 'bridge':
                    if network_def.get('internal', False):
                        errors.append(f"桥接网络 {network_name} 不应该设置为internal")

                    ipam = network_def.get('ipam', {})
                    if 'config' in ipam:
                        for config_item in ipam['config']:
                            if 'subnet' not in config_item:
                                errors.append(f"桥接网络 {network_name} IPAM配置缺少subnet")

                # Overlay网络验证
                elif driver == 'overlay':
                    if not network_def.get('attachable', False):
                        errors.append(f"Overlay网络 {network_name} 应该设置为attachable")

                    if network_def.get('internal', False):
                        errors.append(f"Overlay网络 {network_name} 不应该设置为internal")

            return errors

        # 验证网络配置
        errors = validate_network_config(network_config)

        # 应该没有配置错误
        assert len(errors) == 0, f"网络配置存在错误: {errors}"

        # 验证桥接网络配置
        bridge_net = network_config['bridge_network']
        assert bridge_net['driver'] == 'bridge'
        assert not bridge_net.get('internal', False)
        assert bridge_net.get('attachable', False)

        # 验证Overlay网络配置
        overlay_net = network_config['overlay_network']
        assert overlay_net['driver'] == 'overlay'
        assert overlay_net.get('attachable', False)
        assert not overlay_net.get('internal', False)


class TestContainerHealthMonitoring:
    """测试容器健康检查和监控测试"""

    def setup_method(self):
        """测试前准备"""
        self.health_checker = Mock()

    def test_container_health_check_configuration(self):
        """测试容器健康检查配置"""
        # 定义健康检查配置
        health_checks = {
            'api_service': {
                'test': ['CMD', 'curl', '-f', 'http://localhost:8000/health'],
                'interval': '30s',
                'timeout': '10s',
                'retries': 3,
                'start_period': '40s',
                'disable': False
            },
            'worker_service': {
                'test': ['CMD-SHELL', 'python -c "import redis; redis.Redis().ping()"'],
                'interval': '60s',
                'timeout': '30s',
                'retries': 2,
                'start_period': '60s'
            },
            'database': {
                'test': ['CMD', 'pg_isready', '-U', 'rqa2025'],
                'interval': '10s',
                'timeout': '5s',
                'retries': 5,
                'start_period': '30s'
            }
        }

        def validate_health_checks(checks: Dict) -> List[str]:
            """验证健康检查配置"""
            errors = []

            for service_name, check_config in checks.items():
                # 检查必需字段
                required_fields = ['test', 'interval', 'timeout', 'retries', 'start_period']
                for field in required_fields:
                    if field not in check_config:
                        errors.append(f"服务 {service_name} 健康检查缺少字段: {field}")

                # 验证测试命令
                test_cmd = check_config.get('test', [])
                if not test_cmd:
                    errors.append(f"服务 {service_name} 健康检查测试命令为空")
                else:
                    if test_cmd[0] not in ['CMD', 'CMD-SHELL', 'NONE']:
                        errors.append(f"服务 {service_name} 健康检查测试命令格式无效")

                # 验证时间配置
                time_fields = ['interval', 'timeout', 'start_period']
                for field in time_fields:
                    if field in check_config:
                        # 简化的时间格式验证
                        time_str = check_config[field]
                        if not (time_str.endswith('s') or time_str.endswith('m') or time_str.endswith('h')):
                            errors.append(f"服务 {service_name} {field} 时间格式无效: {time_str}")

                # 验证重试次数
                retries = check_config.get('retries', 0)
                if not isinstance(retries, int) or retries < 1:
                    errors.append(f"服务 {service_name} 重试次数无效: {retries}")

                # 验证间隔合理性
                interval = check_config.get('interval', '30s')
                timeout = check_config.get('timeout', '30s')

                # 简化的时间比较（假设都是秒格式）
                try:
                    interval_sec = int(interval.rstrip('s'))
                    timeout_sec = int(timeout.rstrip('s'))

                    if timeout_sec >= interval_sec:
                        errors.append(f"服务 {service_name} 超时时间不应大于等于检查间隔")
                except ValueError:
                    pass  # 忽略复杂的格式

            return errors

        # 验证健康检查配置
        errors = validate_health_checks(health_checks)

        # 应该没有配置错误
        assert len(errors) == 0, f"健康检查配置存在错误: {errors}"

        # 验证具体配置
        api_check = health_checks['api_service']
        assert api_check['test'][0] == 'CMD'
        assert 'curl' in api_check['test']
        assert api_check['retries'] == 3

        worker_check = health_checks['worker_service']
        assert worker_check['test'][0] == 'CMD-SHELL'
        assert 'redis' in ' '.join(worker_check['test'])

    def test_container_logging_configuration(self):
        """测试容器日志配置"""
        # 定义日志配置
        logging_config = {
            'api_service': {
                'driver': 'json-file',
                'options': {
                    'max-size': '10m',
                    'max-file': '3',
                    'labels': 'production',
                    'env': 'os,arch'
                }
            },
            'worker_service': {
                'driver': 'fluentd',
                'options': {
                    'fluentd-address': 'localhost:24224',
                    'fluentd-tag': 'rqa2025.worker',
                    'fluentd-async-connect': 'true'
                }
            },
            'database': {
                'driver': 'syslog',
                'options': {
                    'syslog-address': 'tcp://syslog-server:514',
                    'syslog-facility': 'daemon',
                    'tag': 'rqa2025-db'
                }
            }
        }

        def validate_logging_config(config: Dict) -> List[str]:
            """验证日志配置"""
            errors = []

            valid_drivers = ['json-file', 'syslog', 'journald', 'fluentd', 'awslogs', 'splunk', 'etwlogs', 'gcplogs']

            for service_name, log_config in config.items():
                # 检查驱动
                driver = log_config.get('driver')
                if not driver:
                    errors.append(f"服务 {service_name} 缺少日志驱动配置")
                elif driver not in valid_drivers:
                    errors.append(f"服务 {service_name} 日志驱动无效: {driver}")

                # 验证驱动特定配置
                options = log_config.get('options', {})

                if driver == 'json-file':
                    if 'max-size' not in options:
                        errors.append(f"服务 {service_name} json-file驱动缺少max-size配置")

                elif driver == 'fluentd':
                    if 'fluentd-address' not in options:
                        errors.append(f"服务 {service_name} fluentd驱动缺少fluentd-address配置")

                elif driver == 'syslog':
                    if 'syslog-address' not in options:
                        errors.append(f"服务 {service_name} syslog驱动缺少syslog-address配置")

            return errors

        # 验证日志配置
        errors = validate_logging_config(logging_config)

        # 应该没有配置错误
        assert len(errors) == 0, f"日志配置存在错误: {errors}"

        # 验证具体配置
        api_logging = logging_config['api_service']
        assert api_logging['driver'] == 'json-file'
        assert api_logging['options']['max-size'] == '10m'
        assert api_logging['options']['max-file'] == '3'

        fluent_logging = logging_config['worker_service']
        assert fluent_logging['driver'] == 'fluentd'
        assert 'fluentd-address' in fluent_logging['options']


if __name__ == "__main__":
    pytest.main([__file__])
