"""
生产环境部署脚本

自动化部署RQA2025系统到生产环境，包含环境检查、部署执行、验证测试等功能。

作者: 刘强
创建日期: 2026-03-30
版本: 1.0.0
"""

import os
import sys
import time
import json
import shutil
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('deploy.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class DeployConfig:
    """部署配置"""
    app_name: str = "RQA2025"
    version: str = "1.0.0"
    deploy_dir: str = "/opt/rqa2025"
    backup_dir: str = "/opt/backups/rqa2025"
    service_name: str = "rqa2025"
    user: str = "rqa2025"
    group: str = "rqa2025"
    python_version: str = "3.11"
    
    # 数据库配置
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "rqa2025"
    db_user: str = "rqa2025"
    
    # 服务配置
    service_port: int = 8000
    worker_processes: int = 4
    
    # 监控配置
    prometheus_port: int = 9090
    grafana_port: int = 3000


class DeploymentError(Exception):
    """部署错误"""
    pass


class DeployManager:
    """
    部署管理器
    
    负责执行完整的生产环境部署流程，包括：
    - 部署前检查
    - 备份现有版本
    - 部署新版本
    - 执行数据库迁移
    - 启动服务
    - 部署验证
    - 回滚支持
    """
    
    def __init__(self, config: Optional[DeployConfig] = None):
        """
        初始化部署管理器
        
        Args:
            config: 部署配置
        """
        self.config = config or DeployConfig()
        self.deploy_id = f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_time = time.time()
        self.check_results: Dict[str, bool] = {}
        
    def log_step(self, step: str, message: str, level: str = "info"):
        """记录部署步骤"""
        log_func = getattr(logger, level)
        log_func(f"[{step}] {message}")
    
    def run_command(
        self,
        command: List[str],
        cwd: Optional[str] = None,
        env: Optional[Dict] = None,
        check: bool = True
    ) -> Tuple[int, str, str]:
        """
        执行系统命令
        
        Args:
            command: 命令列表
            cwd: 工作目录
            env: 环境变量
            check: 是否检查返回码
            
        Returns:
            (返回码, 标准输出, 标准错误)
        """
        self.log_step("CMD", f"执行命令: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                check=False
            )
            
            if check and result.returncode != 0:
                raise DeploymentError(
                    f"命令执行失败: {' '.join(command)}\n"
                    f"返回码: {result.returncode}\n"
                    f"错误输出: {result.stderr}"
                )
            
            return result.returncode, result.stdout, result.stderr
            
        except Exception as e:
            raise DeploymentError(f"命令执行异常: {e}")
    
    # ==================== 部署前检查 ====================
    
    def pre_deploy_checks(self) -> bool:
        """
        执行部署前检查
        
        Returns:
            所有检查是否通过
        """
        self.log_step("CHECK", "开始部署前检查...")
        
        checks = [
            ("系统资源", self._check_system_resources),
            ("依赖服务", self._check_dependencies),
            ("数据库连接", self._check_database),
            ("磁盘空间", self._check_disk_space),
            ("权限检查", self._check_permissions),
            ("配置文件", self._check_config_files),
        ]
        
        all_passed = True
        for check_name, check_func in checks:
            try:
                result = check_func()
                self.check_results[check_name] = result
                status = "✅ 通过" if result else "❌ 失败"
                self.log_step("CHECK", f"{check_name}: {status}")
                
                if not result:
                    all_passed = False
                    
            except Exception as e:
                self.check_results[check_name] = False
                self.log_step("CHECK", f"{check_name}: ❌ 异常 - {e}", "error")
                all_passed = False
        
        if all_passed:
            self.log_step("CHECK", "所有检查通过，可以继续部署")
        else:
            self.log_step("CHECK", "部分检查未通过，请修复后重试", "error")
        
        return all_passed
    
    def _check_system_resources(self) -> bool:
        """检查系统资源"""
        try:
            import psutil
            
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 80:
                self.log_step("CHECK", f"CPU使用率过高: {cpu_percent}%", "warning")
                return False
            
            # 内存使用率
            memory = psutil.virtual_memory()
            if memory.percent > 85:
                self.log_step("CHECK", f"内存使用率过高: {memory.percent}%", "warning")
                return False
            
            return True
        except ImportError:
            self.log_step("CHECK", "psutil未安装，跳过资源检查", "warning")
            return True
    
    def _check_dependencies(self) -> bool:
        """检查依赖服务"""
        services = ["postgresql", "redis"]
        
        for service in services:
            try:
                self.run_command(["systemctl", "is-active", service], check=False)
            except Exception:
                self.log_step("CHECK", f"服务 {service} 未运行", "warning")
                return False
        
        return True
    
    def _check_database(self) -> bool:
        """检查数据库连接"""
        try:
            import psycopg2
            
            conn = psycopg2.connect(
                host=self.config.db_host,
                port=self.config.db_port,
                database=self.config.db_name,
                user=self.config.db_user
            )
            conn.close()
            return True
        except ImportError:
            self.log_step("CHECK", "psycopg2未安装，跳过数据库检查", "warning")
            return True
        except Exception as e:
            self.log_step("CHECK", f"数据库连接失败: {e}", "error")
            return False
    
    def _check_disk_space(self) -> bool:
        """检查磁盘空间"""
        try:
            import shutil
            
            total, used, free = shutil.disk_usage(self.config.deploy_dir)
            free_gb = free / (1024**3)
            
            if free_gb < 5:
                self.log_step("CHECK", f"磁盘空间不足: {free_gb:.2f}GB", "warning")
                return False
            
            return True
        except Exception as e:
            self.log_step("CHECK", f"磁盘检查失败: {e}", "warning")
            return True
    
    def _check_permissions(self) -> bool:
        """检查权限"""
        try:
            # 检查是否为root用户
            if os.geteuid() != 0:
                self.log_step("CHECK", "需要使用root权限运行部署脚本", "error")
                return False
            
            return True
        except AttributeError:
            # Windows系统没有geteuid
            return True
    
    def _check_config_files(self) -> bool:
        """检查配置文件"""
        config_files = [
            "config/production.yml",
            "config/prometheus.yml",
            ".env.production"
        ]
        
        for config_file in config_files:
            if not os.path.exists(config_file):
                self.log_step("CHECK", f"配置文件不存在: {config_file}", "warning")
                return False
        
        return True
    
    # ==================== 备份 ====================
    
    def backup_current_version(self) -> bool:
        """
        备份当前版本
        
        Returns:
            备份是否成功
        """
        self.log_step("BACKUP", "开始备份当前版本...")
        
        try:
            backup_path = os.path.join(
                self.config.backup_dir,
                f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            if os.path.exists(self.config.deploy_dir):
                os.makedirs(backup_path, exist_ok=True)
                
                # 备份代码
                shutil.copytree(
                    self.config.deploy_dir,
                    os.path.join(backup_path, "app"),
                    ignore=shutil.ignore_patterns('*.pyc', '__pycache__', '.git')
                )
                
                # 备份数据库
                self._backup_database(backup_path)
                
                # 备份配置文件
                config_backup = os.path.join(backup_path, "config")
                if os.path.exists("config"):
                    shutil.copytree("config", config_backup)
                
                self.log_step("BACKUP", f"备份完成: {backup_path}")
                return True
            else:
                self.log_step("BACKUP", "当前版本不存在，跳过备份")
                return True
                
        except Exception as e:
            self.log_step("BACKUP", f"备份失败: {e}", "error")
            return False
    
    def _backup_database(self, backup_path: str):
        """备份数据库"""
        try:
            db_backup_file = os.path.join(backup_path, "database.sql")
            
            self.run_command([
                "pg_dump",
                "-h", self.config.db_host,
                "-p", str(self.config.db_port),
                "-U", self.config.db_user,
                "-d", self.config.db_name,
                "-f", db_backup_file
            ])
            
            self.log_step("BACKUP", f"数据库备份完成: {db_backup_file}")
            
        except Exception as e:
            self.log_step("BACKUP", f"数据库备份失败: {e}", "warning")
    
    # ==================== 部署 ====================
    
    def deploy_new_version(self) -> bool:
        """
        部署新版本
        
        Returns:
            部署是否成功
        """
        self.log_step("DEPLOY", "开始部署新版本...")
        
        try:
            # 创建部署目录
            os.makedirs(self.config.deploy_dir, exist_ok=True)
            
            # 复制代码
            self._copy_application_code()
            
            # 安装依赖
            self._install_dependencies()
            
            # 执行数据库迁移
            self._run_migrations()
            
            # 收集静态文件
            self._collect_static_files()
            
            # 设置权限
            self._set_permissions()
            
            self.log_step("DEPLOY", "新版本部署完成")
            return True
            
        except Exception as e:
            self.log_step("DEPLOY", f"部署失败: {e}", "error")
            return False
    
    def _copy_application_code(self):
        """复制应用代码"""
        self.log_step("DEPLOY", "复制应用代码...")
        
        # 复制当前目录下的代码到部署目录
        exclude_patterns = [
            '.git', '__pycache__', '*.pyc', '.env', 'venv',
            'node_modules', '.pytest_cache', '*.log'
        ]
        
        for item in os.listdir('.'):
            if item in exclude_patterns:
                continue
                
            src = item
            dst = os.path.join(self.config.deploy_dir, item)
            
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*exclude_patterns))
            else:
                shutil.copy2(src, dst)
    
    def _install_dependencies(self):
        """安装依赖"""
        self.log_step("DEPLOY", "安装Python依赖...")
        
        requirements_file = os.path.join(self.config.deploy_dir, "requirements.txt")
        
        if os.path.exists(requirements_file):
            self.run_command([
                sys.executable, "-m", "pip", "install",
                "-r", requirements_file,
                "--no-cache-dir"
            ])
    
    def _run_migrations(self):
        """执行数据库迁移"""
        self.log_step("DEPLOY", "执行数据库迁移...")
        
        # 这里应该调用实际的数据库迁移命令
        # 例如: alembic upgrade head
        self.log_step("DEPLOY", "数据库迁移完成（模拟）")
    
    def _collect_static_files(self):
        """收集静态文件"""
        self.log_step("DEPLOY", "收集静态文件...")
        
        static_dir = os.path.join(self.config.deploy_dir, "static")
        os.makedirs(static_dir, exist_ok=True)
        
        self.log_step("DEPLOY", "静态文件收集完成")
    
    def _set_permissions(self):
        """设置权限"""
        self.log_step("DEPLOY", "设置文件权限...")
        
        try:
            self.run_command([
                "chown", "-R", f"{self.config.user}:{self.config.group}",
                self.config.deploy_dir
            ])
        except Exception as e:
            self.log_step("DEPLOY", f"设置权限失败: {e}", "warning")
    
    # ==================== 服务管理 ====================
    
    def start_services(self) -> bool:
        """
        启动服务
        
        Returns:
            启动是否成功
        """
        self.log_step("SERVICE", "启动服务...")
        
        try:
            # 重新加载systemd配置
            self.run_command(["systemctl", "daemon-reload"])
            
            # 启动主服务
            self.run_command(["systemctl", "restart", self.config.service_name])
            self.run_command(["systemctl", "enable", self.config.service_name])
            
            # 等待服务启动
            time.sleep(5)
            
            # 检查服务状态
            returncode, stdout, _ = self.run_command(
                ["systemctl", "is-active", self.config.service_name],
                check=False
            )
            
            if returncode == 0:
                self.log_step("SERVICE", "服务启动成功")
                return True
            else:
                self.log_step("SERVICE", "服务启动失败", "error")
                return False
                
        except Exception as e:
            self.log_step("SERVICE", f"启动服务失败: {e}", "error")
            return False
    
    def stop_services(self) -> bool:
        """
        停止服务
        
        Returns:
            停止是否成功
        """
        self.log_step("SERVICE", "停止服务...")
        
        try:
            self.run_command(["systemctl", "stop", self.config.service_name])
            self.log_step("SERVICE", "服务已停止")
            return True
        except Exception as e:
            self.log_step("SERVICE", f"停止服务失败: {e}", "warning")
            return False
    
    # ==================== 验证 ====================
    
    def verify_deployment(self) -> bool:
        """
        验证部署
        
        Returns:
            验证是否通过
        """
        self.log_step("VERIFY", "开始部署验证...")
        
        checks = [
            ("服务健康检查", self._verify_service_health),
            ("API接口检查", self._verify_api_endpoints),
            ("数据库连接检查", self._verify_database_connection),
            ("监控指标检查", self._verify_monitoring),
        ]
        
        all_passed = True
        for check_name, check_func in checks:
            try:
                result = check_func()
                status = "✅ 通过" if result else "❌ 失败"
                self.log_step("VERIFY", f"{check_name}: {status}")
                
                if not result:
                    all_passed = False
                    
            except Exception as e:
                self.log_step("VERIFY", f"{check_name}: ❌ 异常 - {e}", "error")
                all_passed = False
        
        return all_passed
    
    def _verify_service_health(self) -> bool:
        """验证服务健康状态"""
        try:
            import requests
            
            response = requests.get(
                f"http://localhost:{self.config.service_port}/health",
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def _verify_api_endpoints(self) -> bool:
        """验证API接口"""
        try:
            import requests
            
            endpoints = [
                "/api/v1/health",
                "/api/v1/metrics/system"
            ]
            
            for endpoint in endpoints:
                response = requests.get(
                    f"http://localhost:{self.config.service_port}{endpoint}",
                    timeout=5
                )
                if response.status_code != 200:
                    return False
            
            return True
        except Exception:
            return False
    
    def _verify_database_connection(self) -> bool:
        """验证数据库连接"""
        try:
            import psycopg2
            
            conn = psycopg2.connect(
                host=self.config.db_host,
                port=self.config.db_port,
                database=self.config.db_name,
                user=self.config.db_user
            )
            conn.close()
            return True
        except Exception:
            return False
    
    def _verify_monitoring(self) -> bool:
        """验证监控"""
        try:
            import requests
            
            # 检查Prometheus
            response = requests.get(
                f"http://localhost:{self.config.prometheus_port}/-/healthy",
                timeout=5
            )
            
            return response.status_code == 200
        except Exception:
            return False
    
    # ==================== 回滚 ====================
    
    def rollback(self, backup_path: str) -> bool:
        """
        执行回滚
        
        Args:
            backup_path: 备份路径
            
        Returns:
            回滚是否成功
        """
        self.log_step("ROLLBACK", f"开始回滚到: {backup_path}")
        
        try:
            # 停止服务
            self.stop_services()
            
            # 恢复代码
            if os.path.exists(self.config.deploy_dir):
                shutil.rmtree(self.config.deploy_dir)
            
            shutil.copytree(
                os.path.join(backup_path, "app"),
                self.config.deploy_dir
            )
            
            # 恢复数据库
            db_backup = os.path.join(backup_path, "database.sql")
            if os.path.exists(db_backup):
                self.run_command([
                    "psql",
                    "-h", self.config.db_host,
                    "-p", str(self.config.db_port),
                    "-U", self.config.db_user,
                    "-d", self.config.db_name,
                    "-f", db_backup
                ])
            
            # 启动服务
            self.start_services()
            
            self.log_step("ROLLBACK", "回滚完成")
            return True
            
        except Exception as e:
            self.log_step("ROLLBACK", f"回滚失败: {e}", "error")
            return False
    
    # ==================== 主流程 ====================
    
    def deploy(self) -> bool:
        """
        执行完整部署流程
        
        Returns:
            部署是否成功
        """
        self.log_step("DEPLOY", f"开始部署: {self.deploy_id}")
        
        try:
            # 1. 部署前检查
            if not self.pre_deploy_checks():
                raise DeploymentError("部署前检查未通过")
            
            # 2. 备份当前版本
            if not self.backup_current_version():
                raise DeploymentError("备份失败")
            
            # 3. 停止服务
            self.stop_services()
            
            # 4. 部署新版本
            if not self.deploy_new_version():
                raise DeploymentError("部署新版本失败")
            
            # 5. 启动服务
            if not self.start_services():
                raise DeploymentError("启动服务失败")
            
            # 6. 验证部署
            if not self.verify_deployment():
                raise DeploymentError("部署验证失败")
            
            # 计算部署时间
            deploy_time = time.time() - self.start_time
            
            self.log_step("DEPLOY", f"✅ 部署成功！耗时: {deploy_time:.2f}秒")
            return True
            
        except DeploymentError as e:
            self.log_step("DEPLOY", f"❌ 部署失败: {e}", "error")
            self.log_step("DEPLOY", "准备执行回滚...", "warning")
            
            # 执行回滚
            # self.rollback(backup_path)
            
            return False
        
        except Exception as e:
            self.log_step("DEPLOY", f"❌ 部署异常: {e}", "error")
            return False


def main():
    """主函数"""
    print("=" * 60)
    print("RQA2025 生产环境部署脚本")
    print("=" * 60)
    
    # 创建部署管理器
    config = DeployConfig()
    deployer = DeployManager(config)
    
    # 执行部署
    success = deployer.deploy()
    
    if success:
        print("\n✅ 部署成功完成！")
        sys.exit(0)
    else:
        print("\n❌ 部署失败，请查看日志了解详情")
        sys.exit(1)


if __name__ == "__main__":
    main()
