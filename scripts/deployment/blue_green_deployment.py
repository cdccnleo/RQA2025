#!/usr/bin/env python3
"""
蓝绿部署自动化脚本
用于RQA2025项目的蓝绿部署管理
"""

import argparse
import json
import logging
import subprocess
import time
from datetime import datetime
from typing import Dict
import yaml

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BlueGreenDeployment:
    """蓝绿部署管理器"""

    def __init__(self, namespace: str = "production", config_path: str = "deploy/blue-green-deployment.yml"):
        self.namespace = namespace
        self.config_path = config_path
        self.current_environment = "blue"  # 默认当前环境
        self.switch_timeout = 300  # 5分钟切换超时
        self.rollback_timeout = 60  # 1分钟回滚超时

    def load_config(self) -> Dict:
        """加载部署配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ 成功加载部署配置: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"❌ 加载部署配置失败: {e}")
            raise

    def get_current_environment(self) -> str:
        """获取当前活跃环境"""
        try:
            # 检查服务选择器
            result = subprocess.run([
                "kubectl", "get", "service", "rqa2025-service",
                "-n", self.namespace, "-o", "jsonpath={.spec.selector.version}"
            ], capture_output=True, text=True, check=True)

            current_env = result.stdout.strip()
            logger.info(f"当前活跃环境: {current_env}")
            return current_env
        except subprocess.CalledProcessError as e:
            logger.error(f"获取当前环境失败: {e}")
            return "blue"  # 默认返回blue

    def check_health(self, environment: str) -> bool:
        """检查环境健康状态"""
        try:
            # 检查Pod状态
            result = subprocess.run([
                "kubectl", "get", "pods",
                "-n", self.namespace,
                "-l", f"app=rqa2025,version={environment}",
                "-o", "jsonpath={.items[*].status.phase}"
            ], capture_output=True, text=True, check=True)

            pod_statuses = result.stdout.strip().split()
            if not pod_statuses:
                logger.warning(f"环境 {environment} 没有运行中的Pod")
                return False

            # 检查所有Pod是否都是Running状态
            running_pods = [status for status in pod_statuses if status == "Running"]
            total_pods = len(pod_statuses)

            if len(running_pods) == total_pods:
                logger.info(f"✅ 环境 {environment} 健康检查通过: {len(running_pods)}/{total_pods} Pod运行中")
                return True
            else:
                logger.warning(
                    f"⚠️ 环境 {environment} 健康检查失败: {len(running_pods)}/{total_pods} Pod运行中")
                return False

        except subprocess.CalledProcessError as e:
            logger.error(f"健康检查失败: {e}")
            return False

    def check_readiness(self, environment: str) -> bool:
        """检查环境就绪状态"""
        try:
            # 检查就绪探针
            result = subprocess.run([
                "kubectl", "get", "pods",
                "-n", self.namespace,
                "-l", f"app=rqa2025,version={environment}",
                "-o", "jsonpath={.items[*].status.conditions[?(@.type=='Ready')].status}"
            ], capture_output=True, text=True, check=True)

            readiness_statuses = result.stdout.strip().split()
            if not readiness_statuses:
                logger.warning(f"环境 {environment} 没有就绪状态信息")
                return False

            # 检查所有Pod是否都是True状态
            ready_pods = [status for status in readiness_statuses if status == "True"]
            total_pods = len(readiness_statuses)

            if len(ready_pods) == total_pods:
                logger.info(f"✅ 环境 {environment} 就绪检查通过: {len(ready_pods)}/{total_pods} Pod就绪")
                return True
            else:
                logger.warning(f"⚠️ 环境 {environment} 就绪检查失败: {len(ready_pods)}/{total_pods} Pod就绪")
                return False

        except subprocess.CalledProcessError as e:
            logger.error(f"就绪检查失败: {e}")
            return False

    def deploy_environment(self, environment: str, image_tag: str) -> bool:
        """部署指定环境"""
        try:
            logger.info(f"🚀 开始部署环境 {environment}，镜像标签: {image_tag}")

            # 更新部署配置
            config = self.load_config()

            # 查找对应的Deployment
            deployment_name = f"rqa2025-{environment}"

            # 更新镜像标签
            subprocess.run([
                "kubectl", "set", "image", f"deployment/{deployment_name}",
                f"rqa2025-app=rqa2025:{image_tag}",
                "-n", self.namespace
            ], check=True)

            # 等待部署完成
            logger.info(f"⏳ 等待部署 {deployment_name} 完成...")
            subprocess.run([
                "kubectl", "rollout", "status", f"deployment/{deployment_name}",
                "-n", self.namespace, "--timeout=300s"
            ], check=True)

            logger.info(f"✅ 环境 {environment} 部署完成")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ 环境 {environment} 部署失败: {e}")
            return False

    def scale_environment(self, environment: str, replicas: int) -> bool:
        """扩缩容指定环境"""
        try:
            logger.info(f"📊 调整环境 {environment} 副本数为 {replicas}")

            deployment_name = f"rqa2025-{environment}"
            subprocess.run([
                "kubectl", "scale", "deployment", deployment_name,
                f"--replicas={replicas}",
                "-n", self.namespace
            ], check=True)

            logger.info(f"✅ 环境 {environment} 扩缩容完成")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ 环境 {environment} 扩缩容失败: {e}")
            return False

    def switch_traffic(self, target_environment: str) -> bool:
        """切换流量到指定环境"""
        try:
            logger.info(f"🔄 开始切换流量到环境 {target_environment}")

            # 更新服务选择器
            subprocess.run([
                "kubectl", "patch", "service", "rqa2025-service",
                "-n", self.namespace,
                "-p", f'{{"spec":{{"selector":{{"version":"{target_environment}"}}}}}}'
            ], check=True)

            # 等待服务更新
            time.sleep(10)

            # 验证切换结果
            current_env = self.get_current_environment()
            if current_env == target_environment:
                logger.info(f"✅ 流量切换成功，当前环境: {current_env}")
                self.current_environment = current_env
                return True
            else:
                logger.error(f"❌ 流量切换失败，期望环境: {target_environment}，实际环境: {current_env}")
                return False

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ 流量切换失败: {e}")
            return False

    def rollback(self, target_environment: str) -> bool:
        """回滚到指定环境"""
        try:
            logger.info(f"🔄 开始回滚到环境 {target_environment}")

            # 检查目标环境健康状态
            if not self.check_health(target_environment):
                logger.error(f"❌ 目标环境 {target_environment} 不健康，无法回滚")
                return False

            # 切换流量
            if self.switch_traffic(target_environment):
                logger.info(f"✅ 回滚到环境 {target_environment} 成功")
                return True
            else:
                logger.error(f"❌ 回滚到环境 {target_environment} 失败")
                return False

        except Exception as e:
            logger.error(f"❌ 回滚失败: {e}")
            return False

    def perform_blue_green_deployment(self, new_image_tag: str) -> bool:
        """执行蓝绿部署"""
        try:
            logger.info("🚀 开始蓝绿部署流程")

            # 获取当前环境
            current_env = self.get_current_environment()
            target_env = "green" if current_env == "blue" else "blue"

            logger.info(f"当前环境: {current_env}, 目标环境: {target_env}")

            # 1. 部署目标环境
            if not self.deploy_environment(target_env, new_image_tag):
                logger.error("❌ 目标环境部署失败")
                return False

            # 2. 扩缩容目标环境
            if not self.scale_environment(target_env, 3):
                logger.error("❌ 目标环境扩缩容失败")
                return False

            # 3. 等待目标环境就绪
            logger.info(f"⏳ 等待环境 {target_env} 就绪...")
            start_time = time.time()
            while time.time() - start_time < self.switch_timeout:
                if self.check_health(target_env) and self.check_readiness(target_env):
                    logger.info(f"✅ 环境 {target_env} 就绪")
                    break
                time.sleep(10)
            else:
                logger.error(f"❌ 环境 {target_env} 就绪超时")
                return False

            # 4. 切换流量
            if not self.switch_traffic(target_env):
                logger.error("❌ 流量切换失败")
                return False

            # 5. 验证新环境运行状态
            logger.info("⏳ 验证新环境运行状态...")
            time.sleep(30)  # 等待流量稳定

            if self.check_health(target_env) and self.check_readiness(target_env):
                logger.info("✅ 新环境运行正常")

                # 6. 缩容旧环境
                if not self.scale_environment(current_env, 0):
                    logger.warning(f"⚠️ 旧环境 {current_env} 缩容失败，但不影响部署")

                logger.info("🎉 蓝绿部署完成")
                return True
            else:
                logger.error("❌ 新环境运行异常，准备回滚")
                return self.rollback(current_env)

        except Exception as e:
            logger.error(f"❌ 蓝绿部署失败: {e}")
            return False

    def get_deployment_status(self) -> Dict:
        """获取部署状态"""
        try:
            status = {
                "current_environment": self.get_current_environment(),
                "blue_status": {
                    "health": self.check_health("blue"),
                    "ready": self.check_readiness("blue"),
                    "replicas": self._get_replicas("blue")
                },
                "green_status": {
                    "health": self.check_health("green"),
                    "ready": self.check_readiness("green"),
                    "replicas": self._get_replicas("green")
                },
                "timestamp": datetime.now().isoformat()
            }
            return status
        except Exception as e:
            logger.error(f"获取部署状态失败: {e}")
            return {}

    def _get_replicas(self, environment: str) -> int:
        """获取环境副本数"""
        try:
            result = subprocess.run([
                "kubectl", "get", "deployment", f"rqa2025-{environment}",
                "-n", self.namespace, "-o", "jsonpath={.spec.replicas}"
            ], capture_output=True, text=True, check=True)

            return int(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError):
            return 0


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RQA2025 蓝绿部署管理器")
    parser.add_argument("--namespace", default="production", help="Kubernetes命名空间")
    parser.add_argument("--config", default="deploy/blue-green-deployment.yml", help="部署配置文件路径")
    parser.add_argument("--action", required=True,
                        choices=["deploy", "switch", "rollback", "status", "health"],
                        help="执行操作")
    parser.add_argument("--environment", choices=["blue", "green"], help="目标环境")
    parser.add_argument("--image-tag", help="镜像标签")

    args = parser.parse_args()

    # 初始化部署管理器
    deployer = BlueGreenDeployment(args.namespace, args.config)

    try:
        if args.action == "deploy":
            if not args.image_tag:
                logger.error("❌ 部署操作需要指定 --image-tag 参数")
                return 1

            success = deployer.perform_blue_green_deployment(args.image_tag)
            return 0 if success else 1

        elif args.action == "switch":
            if not args.environment:
                logger.error("❌ 切换操作需要指定 --environment 参数")
                return 1

            success = deployer.switch_traffic(args.environment)
            return 0 if success else 1

        elif args.action == "rollback":
            if not args.environment:
                logger.error("❌ 回滚操作需要指定 --environment 参数")
                return 1

            success = deployer.rollback(args.environment)
            return 0 if success else 1

        elif args.action == "status":
            status = deployer.get_deployment_status()
            print(json.dumps(status, indent=2, ensure_ascii=False))
            return 0

        elif args.action == "health":
            if not args.environment:
                logger.error("❌ 健康检查需要指定 --environment 参数")
                return 1

            health_ok = deployer.check_health(args.environment)
            ready_ok = deployer.check_readiness(args.environment)

            print(f"环境 {args.environment} 状态:")
            print(f"  健康检查: {'✅ 通过' if health_ok else '❌ 失败'}")
            print(f"  就绪检查: {'✅ 通过' if ready_ok else '❌ 失败'}")

            return 0 if health_ok and ready_ok else 1

    except Exception as e:
        logger.error(f"❌ 操作失败: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
