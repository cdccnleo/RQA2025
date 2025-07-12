import os
import yaml
import logging
import subprocess
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ProductionDeployer:
    """生产环境部署工具"""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.env = self._setup_environment()

    def _load_config(self, path: str) -> Dict:
        """加载部署配置文件"""
        with open(path) as f:
            config = yaml.safe_load(f)

        required_keys = ['servers', 'services', 'resources']
        if not all(k in config for k in required_keys):
            raise ValueError("Invalid config format")

        return config

    def _setup_environment(self) -> Dict:
        """设置生产环境变量"""
        env = {
            'DEPLOY_ENV': 'production',
            'LOG_LEVEL': 'INFO',
            'CONFIG_PATH': '/etc/rqa/prod.yaml'
        }

        # 设置环境变量
        for k, v in env.items():
            os.environ[k] = v

        return env

    def prepare_servers(self):
        """准备服务器环境"""
        logger.info("Preparing production servers")

        # 安装基础依赖
        deps = [
            'docker-ce',
            'docker-compose',
            'prometheus',
            'grafana',
            'filebeat'
        ]

        for server in self.config['servers']:
            self._run_ssh_command(
                server['host'],
                f"sudo apt-get install -y {' '.join(deps)}"
            )

            # 创建数据目录
            self._run_ssh_command(
                server['host'],
                "mkdir -p /data/rqa/{logs,db,config}"
            )

    def deploy_services(self):
        """部署服务组件"""
        logger.info("Deploying services")

        # 构建Docker镜像
        subprocess.run([
            "docker", "build",
            "-t", "rqa:prod",
            "--build-arg", f"ENV={self.env['DEPLOY_ENV']}",
            "."
        ], check=True)

        # 推送镜像到仓库
        subprocess.run([
            "docker", "tag", "rqa:prod", "registry.rqa.com/rqa:prod"
        ], check=True)
        subprocess.run([
            "docker", "push", "registry.rqa.com/rqa:prod"
        ], check=True)

        # 在各服务器上部署
        for server in self.config['servers']:
            self._run_ssh_command(
                server['host'],
                "docker pull registry.rqa.com/rqa:prod"
            )

            # 启动服务
            self._run_ssh_command(
                server['host'],
                "docker run -d --name rqa-prod "
                "-p 8080:8080 "
                "-v /data/rqa/config:/config "
                "-v /data/rqa/logs:/logs "
                "registry.rqa.com/rqa:prod"
            )

    def setup_monitoring(self):
        """设置监控系统"""
        logger.info("Setting up monitoring")

        # 部署Prometheus
        subprocess.run([
            "helm", "install", "prometheus",
            "stable/prometheus",
            "--set", "server.global.scrape_interval=15s",
            "--set", "alertmanager.enabled=true"
        ], check=True)

        # 配置Grafana
        for server in self.config['servers']:
            self._run_ssh_command(
                server['host'],
                "echo 'datasources:\n"
                "  - name: Prometheus\n"
                "    type: prometheus\n"
                "    access: proxy\n"
                "    url: http://localhost:9090' "
                "> /etc/grafana/provisioning/datasources.yaml"
            )

            self._run_ssh_command(
                server['host'],
                "systemctl restart grafana-server"
            )

        # 导入监控面板
        subprocess.run([
            "curl", "-X", "POST",
            "-H", "Content-Type: application/json",
            "-d", "@monitoring/dashboards/rqa.json",
            "http://localhost:3000/api/dashboards/db"
        ], check=True)

    def validate_deployment(self):
        """验证部署结果"""
        logger.info("Validating deployment")

        checks = [
            ("服务状态", "docker ps | grep rqa-prod"),
            ("端口监听", "netstat -tulnp | grep 8080"),
            ("接口健康", "curl -s http://localhost:8080/health"),
            ("日志输出", "tail -n 10 /data/rqa/logs/app.log")
        ]

        for name, cmd in checks:
            for server in self.config['servers']:
                try:
                    self._run_ssh_command(server['host'], cmd)
                    logger.info(f"{server['host']} {name} 验证通过")
                except subprocess.CalledProcessError:
                    logger.error(f"{server['host']} {name} 验证失败")
                    raise

    def _run_ssh_command(self, host: str, command: str):
        """执行SSH命令"""
        return subprocess.run([
            "ssh", f"deploy@{host}", command
        ], check=True, capture_output=True, text=True)

def generate_deployment_plan():
    """生成部署计划文档"""
    plan = {
        "version": "1.0.0",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "steps": [
            {
                "name": "环境准备",
                "tasks": [
                    "服务器资源分配",
                    "基础软件安装",
                    "目录权限设置"
                ]
            },
            {
                "name": "服务部署",
                "tasks": [
                    "Docker镜像构建",
                    "服务容器启动",
                    "负载均衡配置"
                ]
            },
            {
                "name": "监控配置",
                "tasks": [
                    "Prometheus部署",
                    "Grafana面板导入",
                    "报警规则设置"
                ]
            },
            {
                "name": "验证测试",
                "tasks": [
                    "健康检查",
                    "接口测试",
                    "性能测试"
                ]
            }
        ]
    }

    with open("deployment_plan.yaml", "w") as f:
        yaml.dump(plan, f, sort_keys=False)

    logger.info("部署计划已生成: deployment_plan.yaml")

if __name__ == "__main__":
    # 初始化部署工具
    deployer = ProductionDeployer("config/deploy_prod.yaml")

    try:
        # 执行部署流程
        generate_deployment_plan()
        deployer.prepare_servers()
        deployer.deploy_services()
        deployer.setup_monitoring()
        deployer.validate_deployment()

        logger.info("生产环境部署成功")
    except Exception as e:
        logger.error(f"部署失败: {e}")
        raise
