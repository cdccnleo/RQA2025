#!/usr/bin/env python3
"""
生产环境配置验证脚本 - Phase 6.1 Day 2
用于验证生产环境配置文件的完整性和正确性

验证内容:
✅ 配置验证 - Docker Compose、环境变量、Nginx、数据库、Redis、监控配置
✅ 功能验证 - 应用结构、依赖关系、配置一致性
✅ 性能验证 - 性能参数、资源限制、连接池配置
✅ 稳定性验证 - 健康检查、日志配置、备份恢复
✅ 安全验证 - SSL配置、身份验证、网络安全

使用方法:
python scripts/production_config_validation.py --test all
python scripts/production_config_validation.py --test config
python scripts/production_config_validation.py --test security
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    import yaml
except ImportError:
    import sys
    print("❌ 需要安装PyYAML: pip install PyYAML")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """验证结果"""
    test_name: str
    status: str  # 'pass', 'fail', 'error', 'warning'
    duration: float
    message: str
    details: Dict[str, Any] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ProductionConfigValidator:
    """生产环境配置验证器"""

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or Path(__file__).parent.parent)
        self.production_env = self.project_root / "production_env"
        self.test_results: List[ValidationResult] = []
        self.configs_validated = False

        # 验证生产环境目录存在
        if not self.production_env.exists():
            raise FileNotFoundError(f"生产环境目录不存在: {self.production_env}")

        # 基础配置验证
        self._validate_basic_configs()

    def _validate_basic_configs(self):
        """验证基础配置文件"""
        required_configs = [
            'docker-compose.yml',
            '.env.production',
            'configs/nginx.conf',
            'configs/postgresql.conf',
            'configs/redis.conf'
        ]

        missing_configs = []
        for config in required_configs:
            if not (self.production_env / config).exists():
                missing_configs.append(config)

        if missing_configs:
            raise FileNotFoundError(f"缺少必要的配置文件: {missing_configs}")

        self.configs_validated = True
        logger.info("✅ 基础配置文件验证通过")

    def run_all_tests(self) -> List[ValidationResult]:
        """运行所有验证测试"""
        logger.info("🚀 开始生产环境配置全面验证...")

        # 配置验证
        self.test_config_validation()

        # 功能验证
        self.test_functional_validation()

        # 性能验证
        self.test_performance_validation()

        # 稳定性验证
        self.test_stability_validation()

        # 安全验证
        self.test_security_validation()

        logger.info(f"✅ 验证完成，共执行 {len(self.test_results)} 个测试")
        return self.test_results

    def test_config_validation(self):
        """配置验证测试"""
        logger.info("🔧 开始配置验证测试...")

        # Docker Compose配置验证
        self._test_docker_compose_config()

        # 环境变量配置验证
        self._test_environment_config()

        # Nginx配置验证
        self._test_nginx_config()

        # 数据库配置验证
        self._test_database_config()

        # Redis配置验证
        self._test_redis_config()

        # 监控配置验证
        self._test_monitoring_config()

    def test_functional_validation(self):
        """功能验证测试"""
        logger.info("🔍 开始功能验证测试...")

        # 应用代码结构验证
        self._test_application_structure()

        # 依赖关系验证
        self._test_dependencies()

        # 配置一致性验证
        self._test_config_consistency()

    def test_performance_validation(self):
        """性能验证测试"""
        logger.info("⚡ 开始性能验证测试...")

        # 配置性能参数验证
        self._test_performance_config()

        # 资源配置验证
        self._test_resource_limits()

        # 连接池配置验证
        self._test_connection_pool_config()

    def test_stability_validation(self):
        """稳定性验证测试"""
        logger.info("🔄 开始稳定性验证测试...")

        # 健康检查配置验证
        self._test_health_check_config()

        # 日志配置验证
        self._test_logging_config()

        # 备份恢复配置验证
        self._test_backup_recovery_config()

    def test_security_validation(self):
        """安全验证测试"""
        logger.info("🔒 开始安全验证测试...")

        # SSL/TLS配置验证
        self._test_ssl_config()

        # 身份验证配置验证
        self._test_auth_config()

        # 网络安全配置验证
        self._test_network_security_config()

    def _test_docker_compose_config(self):
        """Docker Compose配置验证"""
        try:
            compose_file = self.production_env / "docker-compose.yml"
            if not compose_file.exists():
                self._add_result("docker_compose_config", "fail", 0,
                                 "docker-compose.yml文件不存在")
                return

            with open(compose_file, 'r', encoding='utf-8') as f:
                compose_config = yaml.safe_load(f)

            # 验证必要的服务
            required_services = ['postgres', 'redis', 'app', 'nginx']
            services = compose_config.get('services', {})

            missing_services = []
            for service in required_services:
                if service not in services:
                    missing_services.append(service)

            if missing_services:
                self._add_result("docker_compose_config", "fail", 0,
                                 f"缺少必要的服务: {missing_services}")
            else:
                # 验证服务配置
                issues = []

                # 检查PostgreSQL配置
                if 'postgres' in services:
                    pg_config = services['postgres']
                    if 'POSTGRES_DB' not in pg_config.get('environment', {}):
                        issues.append("PostgreSQL缺少数据库名称配置")

                # 检查Redis配置
                if 'redis' in services:
                    redis_config = services['redis']
                    if not redis_config.get('command', '').startswith('redis-server'):
                        issues.append("Redis启动命令配置异常")

                # 检查应用配置
                if 'app' in services:
                    app_config = services['app']
                    if 'env_file' not in app_config:
                        issues.append("应用缺少环境变量文件配置")

                if issues:
                    self._add_result("docker_compose_config", "warning", 0,
                                     f"Docker Compose配置存在问题: {issues}")
                else:
                    self._add_result("docker_compose_config", "pass", 0,
                                     "Docker Compose配置验证通过")

        except Exception as e:
            self._add_result("docker_compose_config", "error", 0,
                             f"Docker Compose配置验证异常: {str(e)}")

    def _test_environment_config(self):
        """环境变量配置验证"""
        try:
            env_file = self.production_env / ".env.production"
            if not env_file.exists():
                self._add_result("environment_config", "fail", 0,
                                 ".env.production文件不存在")
                return

            env_vars = {}
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            env_vars[key.strip()] = value.strip()

            # 验证必要的环境变量
            required_vars = [
                'RQA_ENV', 'DEBUG', 'DATABASE_HOST', 'DATABASE_USER',
                'REDIS_HOST', 'APP_HOST', 'SECRET_KEY', 'JWT_SECRET_KEY'
            ]

            missing_vars = []
            for var in required_vars:
                if var not in env_vars:
                    missing_vars.append(var)

            if missing_vars:
                self._add_result("environment_config", "fail", 0,
                                 f"缺少必要的环境变量: {missing_vars}")
            else:
                # 验证关键配置值
                issues = []

                if env_vars.get('RQA_ENV') != 'production':
                    issues.append("RQA_ENV未设置为production")

                if env_vars.get('DEBUG', '').lower() == 'true':
                    issues.append("生产环境DEBUG不应为true")

                if not env_vars.get('SECRET_KEY', '').startswith('your-super-secure'):
                    issues.append("SECRET_KEY使用了默认值")

                if issues:
                    self._add_result("environment_config", "warning", 0,
                                     f"环境变量配置存在问题: {issues}")
                else:
                    self._add_result("environment_config", "pass", 0,
                                     "环境变量配置验证通过")

        except Exception as e:
            self._add_result("environment_config", "error", 0,
                             f"环境变量配置验证异常: {str(e)}")

    def _test_nginx_config(self):
        """Nginx配置验证"""
        try:
            nginx_file = self.production_env / "configs" / "nginx.conf"
            if not nginx_file.exists():
                self._add_result("nginx_config", "fail", 0,
                                 "nginx.conf文件不存在")
                return

            with open(nginx_file, 'r', encoding='utf-8') as f:
                nginx_config = f.read()

            # 验证关键配置
            checks = [
                ('listen 443 ssl', '缺少HTTPS配置'),
                ('ssl_certificate', '缺少SSL证书配置'),
                ('add_header Strict-Transport-Security', '缺少HSTS安全头'),
                ('add_header X-Frame-Options', '缺少X-Frame-Options安全头'),
                ('proxy_pass http://rqa2025_app', '缺少API代理配置')
            ]

            issues = []
            for check, message in checks:
                if check not in nginx_config:
                    issues.append(message)

            if issues:
                self._add_result("nginx_config", "fail", 0,
                                 f"Nginx配置存在问题: {issues}")
            else:
                self._add_result("nginx_config", "pass", 0,
                                 "Nginx配置验证通过")

        except Exception as e:
            self._add_result("nginx_config", "error", 0,
                             f"Nginx配置验证异常: {str(e)}")

    def _test_database_config(self):
        """数据库配置验证"""
        try:
            pg_file = self.production_env / "configs" / "postgresql.conf"
            if not pg_file.exists():
                self._add_result("database_config", "fail", 0,
                                 "postgresql.conf文件不存在")
                return

            with open(pg_file, 'r', encoding='utf-8') as f:
                pg_config = f.read()

            # 验证关键配置
            checks = [
                ('max_connections = 200', '连接数配置不正确'),
                ('shared_buffers = 256MB', '共享缓冲区配置不正确'),
                ('ssl = on', 'SSL未启用'),
                ('log_line_prefix', '日志前缀未配置'),
                ('pg_stat_statements', '性能监控扩展未配置')
            ]

            issues = []
            for check, message in checks:
                if check not in pg_config:
                    issues.append(message)

            if issues:
                self._add_result("database_config", "fail", 0,
                                 f"PostgreSQL配置存在问题: {issues}")
            else:
                self._add_result("database_config", "pass", 0,
                                 "PostgreSQL配置验证通过")

        except Exception as e:
            self._add_result("database_config", "error", 0,
                             f"数据库配置验证异常: {str(e)}")

    def _test_redis_config(self):
        """Redis配置验证"""
        try:
            redis_file = self.production_env / "configs" / "redis.conf"
            if not redis_file.exists():
                self._add_result("redis_config", "fail", 0,
                                 "redis.conf文件不存在")
                return

            with open(redis_file, 'r', encoding='utf-8') as f:
                redis_config = f.read()

            # 验证关键配置
            checks = [
                ('maxmemory 512mb', '内存限制配置不正确'),
                ('maxmemory-policy allkeys-lru', '淘汰策略配置不正确'),
                ('requirepass', '密码认证未配置'),
                ('save 900 1', '持久化配置不正确'),
                ('appendonly yes', 'AOF持久化未启用')
            ]

            issues = []
            for check, message in checks:
                if check not in redis_config:
                    issues.append(message)

            if issues:
                self._add_result("redis_config", "fail", 0,
                                 f"Redis配置存在问题: {issues}")
            else:
                self._add_result("redis_config", "pass", 0,
                                 "Redis配置验证通过")

        except Exception as e:
            self._add_result("redis_config", "error", 0,
                             f"Redis配置验证异常: {str(e)}")

    def _test_monitoring_config(self):
        """监控配置验证"""
        try:
            monitoring_dir = self.production_env / "monitoring"
            if not monitoring_dir.exists():
                self._add_result("monitoring_config", "warning", 0,
                                 "监控配置目录不存在，将使用默认配置")
                return

            required_files = [
                'prometheus.yml',
                'alert_rules.yml',
                'grafana/dashboards/RQA2025_Overview.json'
            ]

            missing_files = []
            for file_path in required_files:
                if not (monitoring_dir / file_path).exists():
                    missing_files.append(file_path)

            if missing_files:
                self._add_result("monitoring_config", "warning", 0,
                                 f"缺少监控配置文件: {missing_files}")
            else:
                self._add_result("monitoring_config", "pass", 0,
                                 "监控配置验证通过")

        except Exception as e:
            self._add_result("monitoring_config", "error", 0,
                             f"监控配置验证异常: {str(e)}")

    def _test_application_structure(self):
        """应用代码结构验证"""
        try:
            src_dir = self.project_root / "src"
            if not src_dir.exists():
                self._add_result("application_structure", "fail", 0,
                                 "src目录不存在")
                return

            # 验证核心模块
            required_modules = [
                'core',
                'infrastructure',
                'trading',
                'strategy',
                'monitoring',
                'data'
            ]

            missing_modules = []
            for module in required_modules:
                if not (src_dir / module).exists():
                    missing_modules.append(module)

            if missing_modules:
                self._add_result("application_structure", "fail", 0,
                                 f"缺少核心模块: {missing_modules}")
            else:
                self._add_result("application_structure", "pass", 0,
                                 "应用代码结构验证通过")

        except Exception as e:
            self._add_result("application_structure", "error", 0,
                             f"应用结构验证异常: {str(e)}")

    def _test_dependencies(self):
        """依赖关系验证"""
        try:
            requirements_file = self.project_root / "requirements.txt"
            if not requirements_file.exists():
                self._add_result("dependencies", "warning", 0,
                                 "requirements.txt文件不存在")
                return

            with open(requirements_file, 'r', encoding='utf-8') as f:
                requirements = f.read().lower()

            # 检查关键依赖
            critical_deps = [
                'fastapi',
                'uvicorn',
                'asyncpg',
                'redis',
                'pydantic',
                'sqlalchemy'
            ]

            missing_deps = []
            for dep in critical_deps:
                if dep not in requirements:
                    missing_deps.append(dep)

            if missing_deps:
                self._add_result("dependencies", "fail", 0,
                                 f"缺少关键依赖: {missing_deps}")
            else:
                self._add_result("dependencies", "pass", 0,
                                 "依赖关系验证通过")

        except Exception as e:
            self._add_result("dependencies", "error", 0,
                             f"依赖关系验证异常: {str(e)}")

    def _test_config_consistency(self):
        """配置一致性验证"""
        try:
            # 验证环境变量与Docker Compose的一致性
            env_file = self.production_env / ".env.production"
            compose_file = self.production_env / "docker-compose.yml"

            if not env_file.exists() or not compose_file.exists():
                self._add_result("config_consistency", "warning", 0,
                                 "配置文件不完整，跳过一致性检查")
                return

            with open(env_file, 'r', encoding='utf-8') as f:
                env_vars = {}
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()

            with open(compose_file, 'r', encoding='utf-8') as f:
                compose_config = yaml.safe_load(f)

            # 检查关键配置一致性
            issues = []

            # 数据库配置一致性
            if 'DATABASE_NAME' in env_vars:
                pg_config = compose_config.get('services', {}).get('postgres', {})
                pg_db = pg_config.get('environment', {}).get('POSTGRES_DB')
                if pg_db and pg_db != env_vars['DATABASE_NAME']:
                    issues.append("数据库名称配置不一致")

            if issues:
                self._add_result("config_consistency", "warning", 0,
                                 f"配置一致性存在问题: {issues}")
            else:
                self._add_result("config_consistency", "pass", 0,
                                 "配置一致性验证通过")

        except Exception as e:
            self._add_result("config_consistency", "error", 0,
                             f"配置一致性验证异常: {str(e)}")

    def _test_performance_config(self):
        """性能配置参数验证"""
        try:
            env_file = self.production_env / ".env.production"
            if not env_file.exists():
                self._add_result("performance_config", "warning", 0,
                                 "环境配置文件不存在")
                return

            with open(env_file, 'r', encoding='utf-8') as f:
                env_vars = {}
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()

            # 验证性能相关配置
            issues = []

            # 应用性能配置
            app_workers = env_vars.get('APP_WORKERS', '4')
            if int(app_workers) < 2:
                issues.append("应用工作进程数过少，建议至少2个")

            if issues:
                self._add_result("performance_config", "warning", 0,
                                 f"性能配置存在优化空间: {issues}")
            else:
                self._add_result("performance_config", "pass", 0,
                                 "性能配置参数验证通过")

        except Exception as e:
            self._add_result("performance_config", "error", 0,
                             f"性能配置验证异常: {str(e)}")

    def _test_resource_limits(self):
        """资源配置验证"""
        try:
            compose_file = self.production_env / "docker-compose.yml"
            if not compose_file.exists():
                self._add_result("resource_limits", "warning", 0,
                                 "Docker Compose文件不存在")
                return

            with open(compose_file, 'r', encoding='utf-8') as f:
                compose_config = yaml.safe_load(f)

            services = compose_config.get('services', {})
            issues = []

            # 检查资源限制配置
            for service_name, service_config in services.items():
                deploy = service_config.get('deploy', {})
                resources = deploy.get('resources', {})

                if not resources.get('limits'):
                    issues.append(f"{service_name}服务缺少资源限制配置")

            if issues:
                self._add_result("resource_limits", "warning", 0,
                                 f"资源配置存在问题: {issues}")
            else:
                self._add_result("resource_limits", "pass", 0,
                                 "资源配置验证通过")

        except Exception as e:
            self._add_result("resource_limits", "error", 0,
                             f"资源配置验证异常: {str(e)}")

    def _test_connection_pool_config(self):
        """连接池配置验证"""
        try:
            # 这里主要验证代码中的连接池配置是否合理
            env_file = self.production_env / ".env.production"
            if not env_file.exists():
                self._add_result("connection_pool_config", "warning", 0,
                                 "环境配置文件不存在")
                return

            with open(env_file, 'r', encoding='utf-8') as f:
                env_vars = {}
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()

            # 验证连接池相关配置的合理性
            issues = []

            # 数据库连接配置验证
            db_vars = {k: v for k, v in env_vars.items() if k.startswith('DATABASE_')}
            if not db_vars:
                issues.append("缺少数据库连接配置")

            # Redis连接配置验证
            redis_vars = {k: v for k, v in env_vars.items() if k.startswith('REDIS_')}
            if not redis_vars:
                issues.append("缺少Redis连接配置")

            if issues:
                self._add_result("connection_pool_config", "warning", 0,
                                 f"连接池配置存在问题: {issues}")
            else:
                self._add_result("connection_pool_config", "pass", 0,
                                 "连接池配置验证通过")

        except Exception as e:
            self._add_result("connection_pool_config", "error", 0,
                             f"连接池配置验证异常: {str(e)}")

    def _test_health_check_config(self):
        """健康检查配置验证"""
        try:
            compose_file = self.production_env / "docker-compose.yml"
            if not compose_file.exists():
                self._add_result("health_check_config", "warning", 0,
                                 "Docker Compose文件不存在")
                return

            with open(compose_file, 'r', encoding='utf-8') as f:
                compose_config = yaml.safe_load(f)

            services = compose_config.get('services', {})
            services_with_healthcheck = 0
            total_services = len(services)

            for service_name, service_config in services.items():
                if service_config.get('healthcheck'):
                    services_with_healthcheck += 1

            if services_with_healthcheck < total_services * 0.5:  # 至少50%的服务有健康检查
                self._add_result("health_check_config", "warning", 0,
                                 f"健康检查配置不足: {services_with_healthcheck}/{total_services}个服务配置了健康检查")
            else:
                self._add_result("health_check_config", "pass", 0,
                                 f"健康检查配置良好: {services_with_healthcheck}/{total_services}个服务配置了健康检查")

        except Exception as e:
            self._add_result("health_check_config", "error", 0,
                             f"健康检查配置验证异常: {str(e)}")

    def _test_logging_config(self):
        """日志配置验证"""
        try:
            env_file = self.production_env / ".env.production"
            if not env_file.exists():
                self._add_result("logging_config", "warning", 0,
                                 "环境配置文件不存在")
                return

            with open(env_file, 'r', encoding='utf-8') as f:
                env_vars = {}
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()

            # 验证日志相关配置
            issues = []

            log_level = env_vars.get('LOG_LEVEL', '').upper()
            if log_level not in ['INFO', 'WARNING', 'ERROR']:
                issues.append("日志级别配置不合理")

            log_format = env_vars.get('LOG_FORMAT', '').lower()
            if log_format not in ['json', 'text']:
                issues.append("日志格式配置不合理")

            if 'LOG_FILE' not in env_vars:
                issues.append("缺少日志文件配置")

            if issues:
                self._add_result("logging_config", "warning", 0,
                                 f"日志配置存在问题: {issues}")
            else:
                self._add_result("logging_config", "pass", 0,
                                 "日志配置验证通过")

        except Exception as e:
            self._add_result("logging_config", "error", 0,
                             f"日志配置验证异常: {str(e)}")

    def _test_backup_recovery_config(self):
        """备份恢复配置验证"""
        try:
            # 检查数据目录和备份相关配置
            data_dir = self.project_root / "data"
            backup_dir = data_dir / "backup" if data_dir.exists() else None

            issues = []

            if not data_dir.exists():
                issues.append("数据目录不存在")
            else:
                # 检查是否有数据库文件
                db_files = list(data_dir.glob("*.db"))
                if not db_files:
                    issues.append("未找到数据库文件")

                # 检查备份目录
                if not backup_dir or not backup_dir.exists():
                    issues.append("备份目录不存在")

            if issues:
                self._add_result("backup_recovery_config", "warning", 0,
                                 f"备份恢复配置存在问题: {issues}")
            else:
                self._add_result("backup_recovery_config", "pass", 0,
                                 "备份恢复配置验证通过")

        except Exception as e:
            self._add_result("backup_recovery_config", "error", 0,
                             f"备份恢复配置验证异常: {str(e)}")

    def _test_ssl_config(self):
        """SSL/TLS配置验证"""
        try:
            nginx_file = self.production_env / "configs" / "nginx.conf"
            if not nginx_file.exists():
                self._add_result("ssl_config", "warning", 0,
                                 "Nginx配置文件不存在")
                return

            with open(nginx_file, 'r', encoding='utf-8') as f:
                nginx_config = f.read()

            # 验证SSL配置
            ssl_checks = [
                'listen 443 ssl',
                'ssl_protocols TLSv1.2 TLSv1.3',
                'ssl_certificate',
                'ssl_certificate_key'
            ]

            missing_ssl = []
            for check in ssl_checks:
                if check not in nginx_config:
                    missing_ssl.append(check)

            if missing_ssl:
                self._add_result("ssl_config", "fail", 0,
                                 f"SSL配置不完整，缺少: {missing_ssl}")
            else:
                self._add_result("ssl_config", "pass", 0,
                                 "SSL/TLS配置验证通过")

        except Exception as e:
            self._add_result("ssl_config", "error", 0,
                             f"SSL配置验证异常: {str(e)}")

    def _test_auth_config(self):
        """身份验证配置验证"""
        try:
            env_file = self.production_env / ".env.production"
            if not env_file.exists():
                self._add_result("auth_config", "warning", 0,
                                 "环境配置文件不存在")
                return

            with open(env_file, 'r', encoding='utf-8') as f:
                env_vars = {}
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()

            # 验证认证相关配置
            issues = []

            if not env_vars.get('SECRET_KEY'):
                issues.append("缺少SECRET_KEY配置")

            if not env_vars.get('JWT_SECRET_KEY'):
                issues.append("缺少JWT_SECRET_KEY配置")

            bcrypt_rounds = env_vars.get('BCRYPT_ROUNDS', '12')
            if int(bcrypt_rounds) < 10:
                issues.append("Bcrypt轮数过低，不够安全")

            if issues:
                self._add_result("auth_config", "fail", 0,
                                 f"身份验证配置存在问题: {issues}")
            else:
                self._add_result("auth_config", "pass", 0,
                                 "身份验证配置验证通过")

        except Exception as e:
            self._add_result("auth_config", "error", 0,
                             f"身份验证配置验证异常: {str(e)}")

    def _test_network_security_config(self):
        """网络安全配置验证"""
        try:
            nginx_file = self.production_env / "configs" / "nginx.conf"
            if not nginx_file.exists():
                self._add_result("network_security_config", "warning", 0,
                                 "Nginx配置文件不存在")
                return

            with open(nginx_file, 'r', encoding='utf-8') as f:
                nginx_config = f.read()

            # 验证网络安全配置
            security_headers = [
                'Strict-Transport-Security',
                'X-Frame-Options',
                'X-Content-Type-Options',
                'X-XSS-Protection'
            ]

            missing_headers = []
            for header in security_headers:
                if f'add_header {header}' not in nginx_config:
                    missing_headers.append(header)

            if missing_headers:
                self._add_result("network_security_config", "fail", 0,
                                 f"缺少安全头配置: {missing_headers}")
            else:
                self._add_result("network_security_config", "pass", 0,
                                 "网络安全配置验证通过")

        except Exception as e:
            self._add_result("network_security_config", "error", 0,
                             f"网络安全配置验证异常: {str(e)}")

    def _add_result(self, test_name: str, status: str, duration: float,
                    message: str, details: Dict[str, Any] = None):
        """添加测试结果"""
        result = ValidationResult(
            test_name=test_name,
            status=status,
            duration=duration,
            message=message,
            details=details or {}
        )
        self.test_results.append(result)

        status_icon = {
            "pass": "✅",
            "fail": "❌",
            "error": "🔥",
            "warning": "⚠️"
        }.get(status, "❓")

        logger.info(f"{status_icon} {test_name}: {message}")

    def generate_report(self) -> Dict[str, Any]:
        """生成验证报告"""
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.status == "pass"])
        failed_tests = len([r for r in self.test_results if r.status == "fail"])
        error_tests = len([r for r in self.test_results if r.status == "error"])
        warning_tests = len([r for r in self.test_results if r.status == "warning"])

        success_rate = passed_tests / total_tests if total_tests > 0 else 0

        # 按类别分组结果
        categories = {
            "config": ["docker_compose_config", "environment_config", "nginx_config", "database_config", "redis_config", "monitoring_config"],
            "functional": ["application_structure", "dependencies", "config_consistency"],
            "performance": ["performance_config", "resource_limits", "connection_pool_config"],
            "stability": ["health_check_config", "logging_config", "backup_recovery_config"],
            "security": ["ssl_config", "auth_config", "network_security_config"]
        }

        category_results = {}
        for category, tests in categories.items():
            category_tests = [r for r in self.test_results if r.test_name in tests]
            if category_tests:
                category_passed = len([r for r in category_tests if r.status == "pass"])
                category_results[category] = {
                    "total": len(category_tests),
                    "passed": category_passed,
                    "success_rate": category_passed / len(category_tests)
                }

        return {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "error_tests": error_tests,
                "warning_tests": warning_tests,
                "success_rate": success_rate,
                "overall_status": "pass" if success_rate >= 0.9 else "fail"
            },
            "categories": category_results,
            "detailed_results": [asdict(r) for r in self.test_results],
            "generated_at": datetime.now().isoformat(),
            "environment": "production_simulation_config_validation"
        }


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='生产环境配置验证工具')
    parser.add_argument('--test', choices=['all', 'config', 'functional', 'performance', 'stability', 'security'],
                        default='all', help='要执行的测试类型')
    parser.add_argument(
        '--output', default='production_config_validation_report.json', help='输出报告文件')

    args = parser.parse_args()

    try:
        validator = ProductionConfigValidator()

        if args.test == 'all':
            results = validator.run_all_tests()
        elif args.test == 'config':
            validator.test_config_validation()
            results = validator.test_results
        elif args.test == 'functional':
            validator.test_functional_validation()
            results = validator.test_results
        elif args.test == 'performance':
            validator.test_performance_validation()
            results = validator.test_results
        elif args.test == 'stability':
            validator.test_stability_validation()
            results = validator.test_results
        elif args.test == 'security':
            validator.test_security_validation()
            results = validator.test_results

        # 生成报告
        report = validator.generate_report()

        # 保存详细报告
        with open(args.output, 'w', encoding='utf-8') as f:
            # 处理datetime序列化问题
            def datetime_handler(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

            json.dump(report, f, indent=2, ensure_ascii=False, default=datetime_handler)

        # 打印总结报告
        summary = report['summary']
        print("\\n" + "="*60)
        print("🎯 生产环境配置验证报告")
        print("="*60)
        print(f"📊 总测试数: {summary['total_tests']}")
        print(f"✅ 通过测试: {summary['passed_tests']}")
        print(f"❌ 失败测试: {summary['failed_tests']}")
        print(f"🔥 错误测试: {summary['error_tests']}")
        print(f"⚠️  警告测试: {summary['warning_tests']}")
        print(".1%")
        print(f"📈 总体状态: {'✅ 通过' if summary['overall_status'] == 'pass' else '❌ 需要改进'}")

        # 分类结果
        categories = report['categories']
        print("\\n📂 分类结果:")
        for category, stats in categories.items():
            status_icon = "✅" if stats['success_rate'] >= 0.9 else "❌"
            print(".1%")

        print(f"\\n📄 详细报告已保存: {args.output}")

        # 退出码基于测试结果
        exit_code = 0 if summary['overall_status'] == 'pass' else 1

    except Exception as e:
        logger.error(f"验证过程异常: {e}")
        print(f"❌ 验证失败: {e}")
        exit_code = 1

    exit(exit_code)


if __name__ == "__main__":
    main()
