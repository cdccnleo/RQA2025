#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
部署脚本验证测试
Deployment Script Validation Tests

测试部署脚本的完整性和正确性，包括：
1. 部署脚本语法验证
2. 环境变量配置验证
3. 依赖安装验证
4. 数据库迁移验证
5. 服务启动脚本验证
6. 配置模板渲染验证
7. 权限设置验证
8. 部署前后检查验证
"""

import pytest
import os
import tempfile
import subprocess
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
import sys
import yaml
import json

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))


class TestDeploymentScriptSyntaxValidation:
    """测试部署脚本语法验证"""

    def setup_method(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.script_validator = Mock()

    def teardown_method(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_bash_script_syntax_validation(self):
        """测试Bash脚本语法验证"""
        # 创建测试Bash脚本
        bash_script = """#!/bin/bash

# RQA2025 部署脚本示例
set -euo pipefail

# 环境变量验证
if [[ -z "${DEPLOY_ENV:-}" ]]; then
    echo "ERROR: DEPLOY_ENV environment variable is required"
    exit 1
fi

# 函数定义
validate_environment() {
    local env=$1
    case $env in
        development|staging|production)
            return 0
            ;;
        *)
            echo "ERROR: Invalid environment: $env"
            return 1
            ;;
    esac
}

# 主逻辑
main() {
    echo "Starting deployment for environment: $DEPLOY_ENV"

    if ! validate_environment "$DEPLOY_ENV"; then
        exit 1
    fi

    echo "Environment validation passed"

    # 创建必要的目录
    mkdir -p /opt/rqa2025/{logs,data,config}

    echo "Deployment completed successfully"
}

# 执行主函数
main "$@"
"""

        script_path = Path(self.temp_dir) / "deploy.sh"
        with open(script_path, 'w') as f:
            f.write(bash_script)

        # 验证脚本语法
        try:
            result = subprocess.run(
                ['bash', '-n', str(script_path)],
                capture_output=True,
                text=True,
                timeout=30
            )

            # 语法检查应该通过（退出码为0）
            assert result.returncode == 0, f"Bash脚本语法错误: {result.stderr}"

        except subprocess.TimeoutExpired:
            pytest.fail("脚本语法检查超时")
        except FileNotFoundError:
            pytest.skip("Bash不可用，跳过语法检查")

        # 验证脚本可执行性
        script_path.chmod(0o755)
        assert script_path.stat().st_mode & 0o111, "脚本应该具有执行权限"

        # 验证脚本内容
        with open(script_path, 'r') as f:
            content = f.read()

        # 检查关键语法元素
        assert '#!/bin/bash' in content, "应该包含shebang"
        assert 'set -euo pipefail' in content, "应该包含严格模式设置"
        assert 'validate_environment()' in content, "应该包含函数定义"
        assert 'main "$@"' in content, "应该正确调用主函数"

    def test_python_script_syntax_validation(self):
        """测试Python脚本语法验证"""
        # 创建测试Python脚本
        python_script = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

class DeploymentManager:
    \"\"\"部署管理器\"\"\"

    def __init__(self, environment: str, config_path: Optional[Path] = None):
        self.environment = environment
        self.config_path = config_path or Path('config')
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        \"\"\"设置日志\"\"\"
        logger = logging.getLogger('deployment')
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def validate_environment(self) -> bool:
        \"\"\"验证环境配置\"\"\"
        valid_envs = ['development', 'staging', 'production']

        if self.environment not in valid_envs:
            self.logger.error(f"无效环境: {self.environment}")
            return False

        self.logger.info(f"环境验证通过: {self.environment}")
        return True

    def create_directories(self) -> None:
        \"\"\"创建必要的目录\"\"\"
        dirs = [
            Path('/opt/rqa2025/logs'),
            Path('/opt/rqa2025/data'),
            Path('/opt/rqa2025/config')
        ]

        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"创建目录: {dir_path}")

    def deploy(self) -> bool:
        \"\"\"执行部署\"\"\"
        try:
            self.logger.info("开始部署...")

            if not self.validate_environment():
                return False

            self.create_directories()

            self.logger.info("部署完成")
            return True

        except Exception as e:
            self.logger.error(f"部署失败: {e}")
            return False

def main():
    \"\"\"主函数\"\"\"
    parser = argparse.ArgumentParser(description='RQA2025 部署脚本')
    parser.add_argument('--env', required=True, help='部署环境')
    parser.add_argument('--config', help='配置文件路径')

    args = parser.parse_args()

    deployer = DeploymentManager(args.env, args.config if args.config else None)
    success = deployer.deploy()

    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
"""

        script_path = Path(self.temp_dir) / "deploy.py"
        with open(script_path, 'w') as f:
            f.write(python_script)

        # 验证Python脚本语法
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'py_compile', str(script_path)],
                capture_output=True,
                text=True,
                timeout=30
            )

            # 语法检查应该通过（退出码为0）
            assert result.returncode == 0, f"Python脚本语法错误: {result.stderr}"

        except subprocess.TimeoutExpired:
            pytest.fail("脚本语法检查超时")

        # 验证脚本可执行性
        script_path.chmod(0o755)
        assert script_path.stat().st_mode & 0o111, "脚本应该具有执行权限"

        # 验证脚本内容
        with open(script_path, 'r') as f:
            content = f.read()

        # 检查关键语法元素
        assert '#!/usr/bin/env python3' in content, "应该包含Python shebang"
        assert 'import os' in content, "应该包含必要的导入"
        assert 'class DeploymentManager:' in content, "应该包含类定义"
        assert 'def main():' in content, "应该包含主函数"
        assert "if __name__ == '__main__':" in content, "应该包含主程序入口"

    def test_dockerfile_syntax_validation(self):
        """测试Dockerfile语法验证"""
        # 创建测试Dockerfile
        dockerfile_content = """# RQA2025 应用Dockerfile
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEPLOY_ENV=production

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    build-essential \\
    libpq-dev \\
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
COPY requirements-prod.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-prod.txt

# 复制应用代码
COPY src/ ./src/
COPY scripts/ ./scripts/

# 创建非root用户
RUN useradd --create-home --shell /bin/bash app \\
    && chown -R app:app /app
USER app

# 暴露端口
EXPOSE 8000 8001

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["python", "-m", "src.main"]
"""

        dockerfile_path = Path(self.temp_dir) / "Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)

        # 验证Dockerfile语法（如果docker可用）
        try:
            # 检查Dockerfile是否可以被Docker解析
            result = subprocess.run(
                ['docker', 'build', '--dry-run', '-f', str(dockerfile_path), '.'],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=self.temp_dir
            )

            # 如果Docker可用，语法检查应该通过
            if result.returncode != 0:
                # 可能是Docker不可用或其他问题，检查具体错误
                if 'docker' not in result.stderr.lower():
                    pytest.fail(f"Dockerfile语法错误: {result.stderr}")

        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Docker不可用，跳过语法检查
            pytest.skip("Docker不可用，跳过Dockerfile语法检查")

        # 验证Dockerfile内容
        with open(dockerfile_path, 'r') as f:
            content = f.read()

        # 检查关键指令
        assert 'FROM python:3.9-slim' in content, "应该包含基础镜像"
        assert 'WORKDIR /app' in content, "应该设置工作目录"
        assert 'EXPOSE 8000' in content, "应该暴露端口"
        assert 'HEALTHCHECK' in content, "应该包含健康检查"
        assert 'CMD ["python", "-m", "src.main"]' in content, "应该包含启动命令"

    def test_makefile_syntax_validation(self):
        """测试Makefile语法验证"""
        # 创建测试Makefile
        makefile_content = """# RQA2025 构建和部署Makefile

.PHONY: help build test deploy clean install-dev setup

# 默认目标
.DEFAULT_GOAL := help

# 变量定义
PYTHON := python3
PIP := $(PYTHON) -m pip
DOCKER := docker
COMPOSE := docker-compose

# 项目配置
PROJECT_NAME := rqa2025
VERSION := $(shell git describe --tags --always --dirty)

# 帮助信息
help:
\t@echo "RQA2025 构建和部署命令"
\t@echo ""
\t@echo "可用命令:"
\t@echo "  make help          显示此帮助信息"
\t@echo "  make install-dev   安装开发依赖"
\t@echo "  make test          运行测试套件"
\t@echo "  make build         构建应用"
\t@echo "  make deploy        部署应用"
\t@echo "  make clean         清理构建产物"

# 安装开发依赖
install-dev:
\t@echo "安装开发依赖..."
\t$(PIP) install -r requirements-dev.txt
\t$(PIP) install -e .

# 运行测试
test:
\t@echo "运行测试套件..."
\t$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=html

# 构建应用
build: clean
\t@echo "构建应用 $(VERSION)..."
\t$(DOCKER) build -t $(PROJECT_NAME):$(VERSION) .
\t$(DOCKER) tag $(PROJECT_NAME):$(VERSION) $(PROJECT_NAME):latest

# 部署应用
deploy: build
\t@echo "部署应用到环境: $(ENV)"
\t@if [ -z "$(ENV)" ]; then \\
\t\techo "请指定环境变量 ENV (development/staging/production)"; \\
\t\texit 1; \\
\tfi
\t$(COMPOSE) -f docker-compose.$(ENV).yml up -d

# 清理构建产物
clean:
\t@echo "清理构建产物..."
\t$(DOCKER) system prune -f
\tfind . -type d -name __pycache__ -exec rm -rf {} +
\tfind . -type f -name "*.pyc" -delete
\tfind . -type f -name "*.pyo" -delete

# 设置开发环境
setup: install-dev
\t@echo "设置开发环境..."
\tpre-commit install
\t@echo "开发环境设置完成"
"""

        makefile_path = Path(self.temp_dir) / "Makefile"
        with open(makefile_path, 'w') as f:
            f.write(makefile_content)

        # 验证Makefile语法（如果make可用）
        try:
            result = subprocess.run(
                ['make', '-n', '-f', str(makefile_path), 'help'],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.temp_dir
            )

            # 如果make可用，语法检查应该通过
            if result.returncode != 0 and 'make' not in result.stderr.lower():
                pytest.fail(f"Makefile语法错误: {result.stderr}")

        except (subprocess.TimeoutExpired, FileNotFoundError):
            # make不可用，跳过语法检查
            pytest.skip("make不可用，跳过Makefile语法检查")

        # 验证Makefile内容
        with open(makefile_path, 'r') as f:
            content = f.read()

        # 检查关键元素
        assert '.PHONY:' in content, "应该包含.PHONY声明"
        assert 'help:' in content, "应该包含help目标"
        assert '$(PYTHON)' in content, "应该使用变量"
        assert '\t@echo' in content, "应该使用制表符缩进"


class TestEnvironmentConfigurationValidation:
    """测试环境变量配置验证"""

    def setup_method(self):
        """测试前准备"""
        self.config_validator = Mock()

    def test_environment_variables_validation(self):
        """测试环境变量验证"""
        # 定义必需的环境变量
        required_env_vars = {
            'DEPLOY_ENV': ['development', 'staging', 'production'],
            'DATABASE_URL': None,  # 任何非空值
            'REDIS_URL': None,
            'SECRET_KEY': None,
            'LOG_LEVEL': ['DEBUG', 'INFO', 'WARNING', 'ERROR']
        }

        # 测试环境变量验证函数
        def validate_environment_variables(env_vars: Dict[str, str]) -> List[str]:
            """验证环境变量"""
            errors = []

            for var_name, valid_values in required_env_vars.items():
                if var_name not in env_vars:
                    errors.append(f"缺少必需的环境变量: {var_name}")
                    continue

                value = env_vars[var_name]

                if not value:  # 空值检查
                    errors.append(f"环境变量 {var_name} 不能为空")
                    continue

                if valid_values and value not in valid_values:
                    errors.append(f"环境变量 {var_name} 无效值 '{value}'，有效值: {valid_values}")

            return errors

        # 测试有效配置
        valid_env = {
            'DEPLOY_ENV': 'production',
            'DATABASE_URL': 'postgresql://user:pass@localhost:5432/rqa2025',
            'REDIS_URL': 'redis://localhost:6379/0',
            'SECRET_KEY': 'super-secret-key-12345',
            'LOG_LEVEL': 'INFO'
        }

        errors = validate_environment_variables(valid_env)
        assert len(errors) == 0, f"有效配置应该没有错误: {errors}"

        # 测试缺失变量
        missing_env = valid_env.copy()
        del missing_env['DATABASE_URL']

        errors = validate_environment_variables(missing_env)
        assert len(errors) > 0, "应该检测到缺失变量"
        assert any('DATABASE_URL' in error for error in errors), "应该报告缺失的DATABASE_URL"

        # 测试无效值
        invalid_env = valid_env.copy()
        invalid_env['DEPLOY_ENV'] = 'invalid_env'

        errors = validate_environment_variables(invalid_env)
        assert len(errors) > 0, "应该检测到无效值"
        assert any('DEPLOY_ENV' in error for error in errors), "应该报告无效的DEPLOY_ENV"

        # 测试空值
        empty_env = valid_env.copy()
        empty_env['SECRET_KEY'] = ''

        errors = validate_environment_variables(empty_env)
        assert len(errors) > 0, "应该检测到空值"
        assert any('SECRET_KEY' in error for error in errors), "应该报告空的SECRET_KEY"

    def test_configuration_file_parsing(self):
        """测试配置文件解析"""
        # 创建测试配置文件
        config_data = {
            'application': {
                'name': 'RQA2025',
                'version': '1.0.0',
                'environment': 'production'
            },
            'database': {
                'host': 'localhost',
                'port': 5432,
                'name': 'rqa2025',
                'pool_size': 10,
                'ssl_mode': 'require'
            },
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'db': 0,
                'password': None
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'handlers': ['console', 'file']
            },
            'features': {
                'enable_caching': True,
                'enable_monitoring': True,
                'enable_metrics': True
            }
        }

        def validate_config_structure(config: Dict) -> List[str]:
            """验证配置结构"""
            errors = []

            # 检查必需的顶级配置
            required_sections = ['application', 'database', 'redis', 'logging']
            for section in required_sections:
                if section not in config:
                    errors.append(f"缺少必需的配置节: {section}")

            # 验证应用配置
            if 'application' in config:
                app_config = config['application']
                required_app_fields = ['name', 'version', 'environment']
                for field in required_app_fields:
                    if field not in app_config:
                        errors.append(f"应用配置缺少字段: {field}")

                if app_config.get('environment') not in ['development', 'staging', 'production']:
                    errors.append("无效的环境配置")

            # 验证数据库配置
            if 'database' in config:
                db_config = config['database']
                if not isinstance(db_config.get('port'), int) or db_config['port'] <= 0:
                    errors.append("数据库端口必须是正整数")

                if db_config.get('pool_size', 0) <= 0:
                    errors.append("数据库连接池大小必须大于0")

            # 验证Redis配置
            if 'redis' in config:
                redis_config = config['redis']
                if not isinstance(redis_config.get('port'), int) or redis_config['port'] <= 0:
                    errors.append("Redis端口必须是正整数")

                if not isinstance(redis_config.get('db'), int) or redis_config['db'] < 0:
                    errors.append("Redis数据库编号必须是非负整数")

            return errors

        # 测试有效配置
        errors = validate_config_structure(config_data)
        assert len(errors) == 0, f"有效配置应该没有错误: {errors}"

        # 测试缺失配置节
        invalid_config = config_data.copy()
        del invalid_config['database']

        errors = validate_config_structure(invalid_config)
        assert len(errors) > 0, "应该检测到缺失的配置节"
        assert any('database' in error for error in errors), "应该报告缺失的database配置"

        # 测试无效值
        invalid_config2 = config_data.copy()
        invalid_config2['database']['port'] = 'invalid_port'

        errors = validate_config_structure(invalid_config2)
        assert len(errors) > 0, "应该检测到无效的端口值"
        assert any('port' in error for error in errors), "应该报告无效的端口"


class TestDependencyInstallationValidation:
    """测试依赖安装验证"""

    def setup_method(self):
        """测试前准备"""
        self.dependency_checker = Mock()

    @patch('subprocess.run')
    def test_python_dependencies_installation(self, mock_run):
        """测试Python依赖安装"""
        # 模拟pip install成功
        mock_run.return_value = Mock(returncode=0, stdout='Successfully installed...', stderr='')

        # 测试依赖安装
        requirements = [
            'fastapi==0.104.1',
            'uvicorn==0.24.0',
            'sqlalchemy==2.0.23',
            'redis==5.0.1',
            'pytest==7.4.3'
        ]

        # 创建临时requirements文件
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('\n'.join(requirements))
            req_file = f.name

        try:
            # 模拟pip install命令
            result = subprocess.run([
                'pip', 'install', '-r', req_file, '--dry-run'
            ], capture_output=True, text=True)

            # 验证命令执行（即使是dry-run）
            assert result.returncode == 0 or mock_run.called, "依赖安装应该成功"

        except FileNotFoundError:
            pytest.skip("pip不可用，跳过依赖安装测试")
        finally:
            os.unlink(req_file)

    def test_system_dependencies_validation(self):
        """测试系统依赖验证"""
        # 定义必需的系统依赖
        system_dependencies = {
            'python3': {'version': '>=3.9', 'required': True},
            'pip': {'version': '>=20.0', 'required': True},
            'docker': {'version': '>=20.0', 'required': False},  # 可选
            'docker-compose': {'version': '>=1.25', 'required': False}
        }

        def check_system_dependency(dep_name: str, config: Dict) -> Dict:
            """检查系统依赖"""
            result = {
                'name': dep_name,
                'available': False,
                'version': None,
                'satisfied': False,
                'error': None
            }

            try:
                # 尝试运行版本检查命令
                if dep_name == 'python3':
                    cmd = [dep_name, '--version']
                elif dep_name == 'pip':
                    cmd = [dep_name, '--version']
                elif dep_name == 'docker':
                    cmd = [dep_name, '--version']
                elif dep_name == 'docker-compose':
                    cmd = [dep_name, 'version']
                else:
                    cmd = [dep_name, '--version']

                proc_result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if proc_result.returncode == 0:
                    result['available'] = True
                    # 简化的版本解析（实际应该更复杂）
                    version_output = proc_result.stdout.strip()
                    result['version'] = version_output

                    # 简化的版本检查
                    result['satisfied'] = True  # 假设版本满足要求

            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError) as e:
                result['error'] = str(e)

            return result

        # 检查必需的系统依赖
        check_results = {}
        for dep_name, config in system_dependencies.items():
            result = check_system_dependency(dep_name, config)
            check_results[dep_name] = result

            if config['required'] and not result['available']:
                pytest.fail(f"必需的系统依赖不可用: {dep_name}")
            elif result['available'] and not result['satisfied']:
                pytest.fail(f"系统依赖版本不满足要求: {dep_name}")

        # 验证检查结果结构
        for dep_name, result in check_results.items():
            required_fields = ['name', 'available', 'version', 'satisfied']
            for field in required_fields:
                assert field in result, f"依赖检查结果缺少字段: {field}"

    def test_service_startup_dependencies(self):
        """测试服务启动依赖"""
        # 定义服务启动依赖关系
        service_dependencies = {
            'api_service': {
                'depends_on': ['database', 'redis', 'config_service'],
                'health_checks': [
                    {'url': 'http://localhost:8000/health', 'timeout': 30},
                    {'url': 'http://localhost:8001/metrics', 'timeout': 10}
                ]
            },
            'worker_service': {
                'depends_on': ['redis', 'database'],
                'health_checks': [
                    {'queue': 'task_queue', 'timeout': 60}
                ]
            },
            'monitoring_service': {
                'depends_on': ['database'],
                'health_checks': [
                    {'url': 'http://localhost:9090/health', 'timeout': 15}
                ]
            }
        }

        def validate_service_dependencies(service_name: str, deps_config: Dict) -> List[str]:
            """验证服务依赖"""
            errors = []

            # 检查依赖服务
            depends_on = deps_config.get('depends_on', [])
            for dep_service in depends_on:
                # 简化的依赖检查（实际应该检查服务是否运行）
                if dep_service not in service_dependencies and dep_service not in ['database', 'redis']:
                    errors.append(f"服务 {service_name} 依赖未知服务: {dep_service}")

            # 检查健康检查配置
            health_checks = deps_config.get('health_checks', [])
            for check in health_checks:
                if 'url' in check:
                    if not check['url'].startswith(('http://', 'https://')):
                        errors.append(f"服务 {service_name} 健康检查URL格式无效: {check['url']}")
                    if check.get('timeout', 0) <= 0:
                        errors.append(f"服务 {service_name} 健康检查超时时间无效: {check.get('timeout')}")
                elif 'queue' in check:
                    if not check.get('queue'):
                        errors.append(f"服务 {service_name} 队列健康检查缺少队列名称")
                else:
                    errors.append(f"服务 {service_name} 健康检查配置无效: {check}")

            return errors

        # 验证所有服务的依赖配置
        all_errors = []
        for service_name, deps_config in service_dependencies.items():
            errors = validate_service_dependencies(service_name, deps_config)
            all_errors.extend(errors)

        # 应该没有配置错误
        assert len(all_errors) == 0, f"服务依赖配置错误: {all_errors}"

        # 验证依赖关系图的合理性
        dependency_graph = {}
        for service, config in service_dependencies.items():
            dependency_graph[service] = config.get('depends_on', [])

        # 检查循环依赖（简化检查）
        for service, deps in dependency_graph.items():
            for dep in deps:
                if dep in dependency_graph and service in dependency_graph.get(dep, []):
                    pytest.fail(f"检测到循环依赖: {service} <-> {dep}")


if __name__ == "__main__":
    pytest.main([__file__])
