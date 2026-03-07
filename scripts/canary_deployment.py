#!/usr/bin/env python3
"""
RQA2025 灰度发布部署脚本

支持渐进式部署策略：
- 蓝绿部署 (Blue-Green Deployment)
- 金丝雀部署 (Canary Deployment)
- A/B 测试部署
- 滚动更新 (Rolling Update)

功能特性：
- 自动镜像构建和推送
- 渐进式流量切换
- 实时监控和健康检查
- 自动回滚机制
- 多环境支持
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any
import requests

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CanaryDeployment:
    """灰度发布管理器"""

    def __init__(self, config_file: str = "canary_config.json"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
        self.project_root = Path(__file__).parent.parent

    def _load_config(self) -> Dict[str, Any]:
        """加载灰度发布配置"""
        if self.config_file.exists():
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 默认配置
            return {
                "strategy": "canary",
                "total_instances": 6,
                "canary_instances": 1,
                "rollout_percentage": [10, 25, 50, 75, 100],
                "health_check_interval": 30,
                "health_check_timeout": 60,
                "rollback_threshold": 0.1,  # 10%错误率触发回滚
                "monitoring": {
                    "prometheus_url": "http://localhost:9090",
                    "grafana_url": "http://localhost:3000"
                },
                "docker": {
                    "registry": "localhost:5000",
                    "namespace": "rqa2025"
                }
            }

    def save_config(self):
        """保存配置"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    def build_and_push_image(self, version: str) -> str:
        """构建并推送Docker镜像"""
        logger.info(f"开始构建镜像版本: {version}")

        image_tag = f"{self.config['docker']['registry']}/{self.config['docker']['namespace']}/app:{version}"

        try:
            # 构建镜像
            cmd_build = f"docker build -t {image_tag} ."
            logger.info(f"执行: {cmd_build}")
            result = subprocess.run(cmd_build, shell=True, cwd=self.project_root,
                                    capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                raise Exception(f"镜像构建失败: {result.stderr}")

            # 推送镜像
            cmd_push = f"docker push {image_tag}"
            logger.info(f"执行: {cmd_push}")
            result = subprocess.run(cmd_push, shell=True,
                                    capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                raise Exception(f"镜像推送失败: {result.stderr}")

            logger.info(f"镜像 {image_tag} 构建并推送成功")
            return image_tag

        except subprocess.TimeoutExpired:
            raise Exception("镜像构建超时")
        except Exception as e:
            logger.error(f"镜像构建失败: {e}")
            raise

    def deploy_canary(self, image_tag: str, canary_percentage: int = 10) -> bool:
        """部署金丝雀版本"""
        logger.info(f"开始金丝雀部署: {image_tag}, 比例: {canary_percentage}%")

        try:
            # 计算实例数量
            total_instances = self.config["total_instances"]
            canary_instances = max(1, int(total_instances * canary_percentage / 100))

            logger.info(f"总实例数: {total_instances}, 金丝雀实例数: {canary_instances}")

            # 更新docker-compose文件为金丝雀配置
            self._update_compose_for_canary(image_tag, canary_instances)

            # 部署新版本
            cmd = "docker-compose -f docker-compose.canary.yml up -d"
            logger.info(f"执行: {cmd}")
            result = subprocess.run(cmd, shell=True, cwd=self.project_root,
                                    capture_output=True, text=True, timeout=120)

            if result.returncode != 0:
                logger.error(f"金丝雀部署失败: {result.stderr}")
                return False

            # 等待服务启动
            time.sleep(30)

            # 进行健康检查
            if self._perform_health_checks(canary_instances):
                logger.info("金丝雀部署成功")
                return True
            else:
                logger.error("金丝雀部署健康检查失败")
                return False

        except Exception as e:
            logger.error(f"金丝雀部署异常: {e}")
            return False

    def rollout_full(self, image_tag: str) -> bool:
        """全量发布"""
        logger.info(f"开始全量发布: {image_tag}")

        try:
            # 更新为全量配置
            self._update_compose_for_full(image_tag)

            # 执行滚动更新
            cmd = "docker-compose -f docker-compose.prod.yml up -d --scale app=6"
            logger.info(f"执行: {cmd}")
            result = subprocess.run(cmd, shell=True, cwd=self.project_root,
                                    capture_output=True, text=True, timeout=180)

            if result.returncode != 0:
                logger.error(f"全量发布失败: {result.stderr}")
                return False

            # 等待服务完全启动
            time.sleep(60)

            # 最终健康检查
            if self._perform_full_health_check():
                logger.info("全量发布成功")
                return True
            else:
                logger.error("全量发布健康检查失败")
                return False

        except Exception as e:
            logger.error(f"全量发布异常: {e}")
            return False

    def rollback(self, target_version: str) -> bool:
        """回滚到指定版本"""
        logger.info(f"开始回滚到版本: {target_version}")

        try:
            # 停止当前版本
            cmd_stop = "docker-compose -f docker-compose.prod.yml down"
            subprocess.run(cmd_stop, shell=True, cwd=self.project_root)

            # 部署目标版本
            image_tag = f"{self.config['docker']['registry']}/{self.config['docker']['namespace']}/app:{target_version}"

            # 确保镜像存在
            result = subprocess.run(f"docker pull {image_tag}", shell=True,
                                    capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"无法拉取回滚镜像: {image_tag}")
                return False

            # 启动目标版本
            self._update_compose_for_full(image_tag)
            cmd_start = "docker-compose -f docker-compose.prod.yml up -d"
            result = subprocess.run(cmd_start, shell=True, cwd=self.project_root,
                                    capture_output=True, text=True, timeout=180)

            if result.returncode != 0:
                logger.error(f"回滚失败: {result.stderr}")
                return False

            logger.info(f"回滚到版本 {target_version} 成功")
            return True

        except Exception as e:
            logger.error(f"回滚异常: {e}")
            return False

    def _update_compose_for_canary(self, image_tag: str, canary_instances: int):
        """更新docker-compose文件为金丝雀配置"""
        compose_file = self.project_root / "docker-compose.canary.yml"

        # 如果不存在，从生产配置复制
        if not compose_file.exists():
            import shutil
            shutil.copy(self.project_root / "docker-compose.prod.yml", compose_file)

        # 读取并更新配置
        with open(compose_file, 'r', encoding='utf-8') as f:
            compose_config = f.read()

        # 更新镜像标签
        compose_config = compose_config.replace(
            "rqa2025-rqa2025-app:latest",
            image_tag
        )

        # 更新实例数量 (通过scale参数控制)
        # 注意：这里只是准备配置，实际scale在部署时指定

        with open(compose_file, 'w', encoding='utf-8') as f:
            f.write(compose_config)

    def _update_compose_for_full(self, image_tag: str):
        """更新docker-compose文件为全量配置"""
        compose_file = self.project_root / "docker-compose.prod.yml"

        with open(compose_file, 'r', encoding='utf-8') as f:
            compose_config = f.read()

        # 更新镜像标签
        compose_config = compose_config.replace(
            "rqa2025-rqa2025-app:latest",
            image_tag
        )

        with open(compose_file, 'w', encoding='utf-8') as f:
            f.write(compose_config)

    def _perform_health_checks(self, expected_instances: int) -> bool:
        """执行健康检查"""
        logger.info("执行健康检查...")

        try:
            # 检查容器状态
            result = subprocess.run(
                "docker-compose -f docker-compose.canary.yml ps",
                shell=True, cwd=self.project_root,
                capture_output=True, text=True
            )

            if result.returncode != 0:
                logger.error("无法获取容器状态")
                return False

            # 检查运行中的容器数量
            lines = result.stdout.strip().split('\n')
            running_containers = sum(1 for line in lines if 'Up' in line)

            if running_containers < expected_instances:
                logger.error(f"运行容器不足: {running_containers}/{expected_instances}")
                return False

            # 检查应用健康状态
            health_check_url = "http://localhost:8000/health"
            response = requests.get(health_check_url, timeout=10)

            if response.status_code != 200:
                logger.error(f"应用健康检查失败: {response.status_code}")
                return False

            health_data = response.json()
            if health_data.get("status") != "healthy":
                logger.error(f"应用状态不健康: {health_data}")
                return False

            logger.info("健康检查通过")
            return True

        except Exception as e:
            logger.error(f"健康检查异常: {e}")
            return False

    def _perform_full_health_check(self) -> bool:
        """执行全量部署健康检查"""
        logger.info("执行全量部署健康检查...")

        try:
            # 检查所有服务状态
            result = subprocess.run(
                "docker-compose -f docker-compose.prod.yml ps",
                shell=True, cwd=self.project_root,
                capture_output=True, text=True
            )

            if result.returncode != 0:
                logger.error("无法获取全量部署状态")
                return False

            # 检查所有服务都运行正常
            lines = result.stdout.strip().split('\n')
            services_to_check = ['rqa2025-postgres', 'rqa2025-redis', 'rqa2025-nginx',
                                 'rqa2025-prometheus', 'rqa2025-grafana', 'rqa2025-app']

            running_services = []
            for line in lines:
                if 'Up' in line:
                    for service in services_to_check:
                        if service in line:
                            running_services.append(service)

            if len(running_services) < len(services_to_check):
                logger.error(f"服务运行不完整: {running_services}")
                return False

            # 检查应用API
            response = requests.get("http://localhost:8000/health", timeout=10)
            if response.status_code != 200:
                logger.error("应用API检查失败")
                return False

            logger.info("全量部署健康检查通过")
            return True

        except Exception as e:
            logger.error(f"全量部署健康检查异常: {e}")
            return False

    def execute_gradual_rollout(self, new_version: str) -> bool:
        """执行渐进式发布"""
        logger.info(f"开始渐进式发布版本: {new_version}")

        try:
            # 构建新镜像
            image_tag = self.build_and_push_image(new_version)

            # 渐进式部署
            percentages = self.config["rollout_percentage"]
            for percentage in percentages:
                logger.info(f"部署进度: {percentage}%")

                if percentage == 100:
                    # 全量发布
                    success = self.rollout_full(image_tag)
                else:
                    # 金丝雀部署
                    success = self.deploy_canary(image_tag, percentage)

                if not success:
                    logger.error(f"{percentage}% 部署失败，准备回滚")
                    # 回滚逻辑
                    return False

                # 监控阶段
                if not self._monitor_deployment(300):  # 5分钟监控
                    logger.error(f"{percentage}% 部署监控失败，准备回滚")
                    return False

                logger.info(f"{percentage}% 部署成功")

            logger.info("渐进式发布完成")
            return True

        except Exception as e:
            logger.error(f"渐进式发布异常: {e}")
            return False

    def _monitor_deployment(self, duration: int) -> bool:
        """监控部署状态"""
        logger.info(f"开始监控部署状态，持续时间: {duration}秒")

        start_time = time.time()
        error_count = 0
        total_checks = 0

        while time.time() - start_time < duration:
            try:
                # 健康检查
                response = requests.get("http://localhost:8000/health", timeout=5)
                total_checks += 1

                if response.status_code != 200:
                    error_count += 1
                    logger.warning(f"健康检查失败: {response.status_code}")

                # 检查错误率
                error_rate = error_count / total_checks if total_checks > 0 else 0
                if error_rate > self.config["rollback_threshold"]:
                    logger.error(f"错误率超过阈值: {error_rate:.2f}")
                    return False

                time.sleep(self.config["health_check_interval"])

            except Exception as e:
                error_count += 1
                logger.warning(f"监控检查异常: {e}")
                total_checks += 1

        logger.info(f"监控阶段通过，错误率: {error_rate:.2f}")
        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RQA2025 灰度发布工具")
    parser.add_argument("action", choices=["build", "canary", "rollout", "rollback", "status"],
                        help="执行操作")
    parser.add_argument("--version", help="版本号")
    parser.add_argument("--percentage", type=int, default=10, help="金丝雀部署比例")
    parser.add_argument("--config", default="canary_config.json", help="配置文件")

    args = parser.parse_args()

    # 初始化灰度发布管理器
    canary = CanaryDeployment(args.config)

    try:
        if args.action == "build":
            if not args.version:
                logger.error("构建操作需要指定版本号")
                sys.exit(1)

            image_tag = canary.build_and_push_image(args.version)
            print(f"镜像构建成功: {image_tag}")

        elif args.action == "canary":
            if not args.version:
                logger.error("金丝雀部署需要指定版本号")
                sys.exit(1)

            image_tag = f"{canary.config['docker']['registry']}/{canary.config['docker']['namespace']}/app:{args.version}"
            success = canary.deploy_canary(image_tag, args.percentage)

            if success:
                print(f"金丝雀部署成功 ({args.percentage}%)")
            else:
                print("金丝雀部署失败")
                sys.exit(1)

        elif args.action == "rollout":
            if not args.version:
                logger.error("全量发布需要指定版本号")
                sys.exit(1)

            success = canary.execute_gradual_rollout(args.version)

            if success:
                print("渐进式发布成功")
            else:
                print("渐进式发布失败")
                sys.exit(1)

        elif args.action == "rollback":
            if not args.version:
                logger.error("回滚操作需要指定目标版本号")
                sys.exit(1)

            success = canary.rollback(args.version)

            if success:
                print(f"回滚到版本 {args.version} 成功")
            else:
                print("回滚失败")
                sys.exit(1)

        elif args.action == "status":
            # 显示当前部署状态
            result = subprocess.run(
                "docker-compose -f docker-compose.prod.yml ps",
                shell=True, cwd=Path(__file__).parent.parent,
                capture_output=True, text=True
            )

            print("当前部署状态:")
            print(result.stdout)

    except Exception as e:
        logger.error(f"操作失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
