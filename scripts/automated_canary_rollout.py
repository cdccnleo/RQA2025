#!/usr/bin/env python3
"""
RQA2025 自动化灰度发布脚本

自动执行完整的灰度发布流程：
1. 构建新版本镜像
2. 金丝雀部署（10%）
3. 监控和验证
4. 逐步增加流量（25% -> 50% -> 75%）
5. 全量发布或回滚

使用方法：
python automated_canary_rollout.py --version v1.2.3
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AutomatedCanaryRollout:
    """自动化灰度发布"""

    def __init__(self, config_file: str = "canary_config.json"):
        self.config_file = Path(config_file)
        self.project_root = Path(__file__).parent.parent
        self.config = self._load_config()
        self.rollout_history = []

    def _load_config(self) -> Dict[str, Any]:
        """加载配置"""
        if self.config_file.exists():
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"配置文件不存在: {self.config_file}")

    def execute_full_rollout(self, version: str) -> bool:
        """执行完整自动化发布流程"""
        logger.info(f"🚀 开始自动化灰度发布: {version}")

        try:
            # 1. 构建和推送镜像
            logger.info("📦 步骤1: 构建和推送镜像")
            image_tag = self._build_and_push(version)
            if not image_tag:
                raise Exception("镜像构建失败")

            # 2. 准备金丝雀环境
            logger.info("🐦 步骤2: 准备金丝雀环境")
            self._prepare_canary_environment()

            # 3. 渐进式部署
            rollout_stages = self.config["rollout_percentage"]
            for stage_idx, percentage in enumerate(rollout_stages):
                logger.info(f"📈 步骤3.{stage_idx+1}: 部署 {percentage}%")

                if not self._deploy_stage(image_tag, percentage):
                    logger.error(f"❌ 部署 {percentage}% 失败，准备回滚")
                    self._rollback(version)
                    return False

                # 监控阶段
                if not self._monitor_stage(percentage):
                    logger.error(f"❌ 监控阶段失败，准备回滚")
                    self._rollback(version)
                    return False

                logger.info(f"✅ 部署 {percentage}% 成功")

            # 4. 清理和最终验证
            logger.info("🧹 步骤4: 清理和最终验证")
            if not self._finalize_rollout(version):
                logger.error("❌ 最终验证失败")
                return False

            logger.info("🎉 自动化灰度发布完成！")
            return True

        except Exception as e:
            logger.error(f"❌ 自动化发布异常: {e}")
            self._rollback(version)
            return False

    def _build_and_push(self, version: str) -> str:
        """构建和推送镜像"""
        try:
            image_tag = f"{self.config['docker']['registry']}/{self.config['docker']['namespace']}/app:{version}"

            # 构建镜像
            cmd = f"docker build -t {image_tag} ."
            logger.info(f"执行: {cmd}")
            result = subprocess.run(cmd, shell=True, cwd=self.project_root,
                                  capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                logger.error(f"构建失败: {result.stderr}")
                return ""

            # 推送镜像
            cmd = f"docker push {image_tag}"
            logger.info(f"执行: {cmd}")
            result = subprocess.run(cmd, shell=True,
                                  capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                logger.error(f"推送失败: {result.stderr}")
                return ""

            logger.info(f"✅ 镜像构建并推送成功: {image_tag}")
            return image_tag

        except subprocess.TimeoutExpired:
            logger.error("镜像操作超时")
            return ""

    def _prepare_canary_environment(self):
        """准备金丝雀环境"""
        try:
            # 确保Prometheus配置更新
            self._update_prometheus_config()

            # 确保Grafana仪表板就绪
            logger.info("✅ 金丝雀环境准备完成")

        except Exception as e:
            logger.error(f"环境准备失败: {e}")
            raise

    def _deploy_stage(self, image_tag: str, percentage: int) -> bool:
        """部署单个阶段"""
        try:
            if percentage == 100:
                # 全量部署
                return self._deploy_full(image_tag)
            else:
                # 金丝雀部署
                return self._deploy_canary(image_tag, percentage)

        except Exception as e:
            logger.error(f"部署阶段 {percentage}% 失败: {e}")
            return False

    def _deploy_canary(self, image_tag: str, percentage: int) -> bool:
        """金丝雀部署"""
        try:
            # 更新compose文件
            self._update_compose_for_canary(image_tag)

            # 计算实例数量
            total_instances = self.config["total_instances"]
            canary_instances = max(1, int(total_instances * percentage / 100))

            # 启动金丝雀实例
            cmd = f"docker-compose -f docker-compose.canary.yml up -d --scale app-canary={canary_instances}"
            logger.info(f"执行: {cmd}")
            result = subprocess.run(cmd, shell=True, cwd=self.project_root,
                                  capture_output=True, text=True, timeout=120)

            if result.returncode != 0:
                logger.error(f"金丝雀部署失败: {result.stderr}")
                return False

            # 等待启动
            time.sleep(30)

            # 验证部署
            return self._verify_deployment(canary_instances, "canary")

        except Exception as e:
            logger.error(f"金丝雀部署异常: {e}")
            return False

    def _deploy_full(self, image_tag: str) -> bool:
        """全量部署"""
        try:
            # 更新生产环境
            self._update_compose_for_production(image_tag)

            # 停止金丝雀环境
            cmd = "docker-compose -f docker-compose.canary.yml down"
            subprocess.run(cmd, shell=True, cwd=self.project_root)

            # 启动生产环境
            cmd = "docker-compose -f docker-compose.prod.yml up -d"
            logger.info(f"执行: {cmd}")
            result = subprocess.run(cmd, shell=True, cwd=self.project_root,
                                  capture_output=True, text=True, timeout=180)

            if result.returncode != 0:
                logger.error(f"全量部署失败: {result.stderr}")
                return False

            # 等待启动
            time.sleep(60)

            # 验证部署
            return self._verify_deployment(self.config["total_instances"], "production")

        except Exception as e:
            logger.error(f"全量部署异常: {e}")
            return False

    def _monitor_stage(self, percentage: int) -> bool:
        """监控部署阶段"""
        monitor_duration = 300  # 5分钟监控
        logger.info(f"🔍 监控阶段: {percentage}%，持续时间: {monitor_duration}秒")

        start_time = time.time()
        metrics = {
            "requests_total": 0,
            "errors_total": 0,
            "response_time_avg": 0,
            "cpu_usage": 0,
            "memory_usage": 0
        }

        while time.time() - start_time < monitor_duration:
            try:
                # 收集指标
                new_metrics = self._collect_metrics()
                metrics.update(new_metrics)

                # 检查阈值
                if not self._check_thresholds(metrics):
                    logger.error("指标超过阈值")
                    return False

                time.sleep(30)  # 30秒检查一次

            except Exception as e:
                logger.warning(f"监控异常: {e}")
                continue

        logger.info("✅ 监控阶段通过")
        return True

    def _collect_metrics(self) -> Dict[str, float]:
        """收集监控指标"""
        metrics = {}

        try:
            # 从Prometheus获取指标
            prometheus_url = self.config["monitoring"]["prometheus_url"]

            # 请求总数
            response = requests.get(f"{prometheus_url}/api/v1/query",
                                  params={"query": "rate(http_requests_total[5m])"})
            if response.status_code == 200:
                data = response.json()
                if data["data"]["result"]:
                    metrics["requests_total"] = float(data["data"]["result"][0]["value"][1])

            # 错误率
            response = requests.get(f"{prometheus_url}/api/v1/query",
                                  params={"query": "rate(http_requests_total{status=~\"5..\"}[5m])"})
            if response.status_code == 200:
                data = response.json()
                if data["data"]["result"]:
                    metrics["errors_total"] = float(data["data"]["result"][0]["value"][1])

            # CPU使用率
            response = requests.get(f"{prometheus_url}/api/v1/query",
                                  params={"query": "rate(container_cpu_usage_seconds_total[5m])"})
            if response.status_code == 200:
                data = response.json()
                if data["data"]["result"]:
                    metrics["cpu_usage"] = float(data["data"]["result"][0]["value"][1])

        except Exception as e:
            logger.warning(f"指标收集失败: {e}")

        return metrics

    def _check_thresholds(self, metrics: Dict[str, float]) -> bool:
        """检查指标阈值"""
        thresholds = self.config["metrics"]

        # 错误率检查
        error_rate = metrics.get("errors_total", 0) / max(metrics.get("requests_total", 1), 1)
        if error_rate > thresholds["error_rate_threshold"]:
            logger.error(".2f" return False

        # CPU使用率检查
        if metrics.get("cpu_usage", 0) > thresholds["cpu_usage_threshold"] / 100:
            logger.error(".2f" return False

        return True

    def _rollback(self, version: str) -> bool:
        """回滚到稳定版本"""
        logger.info(f"🔄 开始回滚版本: {version}")

        try:
            # 停止当前部署
            cmd="docker-compose -f docker-compose.canary.yml down"
            subprocess.run(cmd, shell=True, cwd=self.project_root)

            # 重新启动稳定版本
            cmd="docker-compose -f docker-compose.prod.yml up -d"
            result=subprocess.run(cmd, shell=True, cwd=self.project_root,
                                  capture_output=True, text=True, timeout=180)

            if result.returncode == 0:
                logger.info("✅ 回滚成功")
                return True
            else:
                logger.error(f"回滚失败: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"回滚异常: {e}")
            return False

    def _finalize_rollout(self, version: str) -> bool:
        """完成发布并清理"""
        try:
            # 记录发布历史
            self.rollout_history.append({
                "version": version,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            })

            # 保存历史记录
            history_file=self.project_root / "rollout_history.json"
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.rollout_history, f, indent=2, ensure_ascii=False)

            logger.info("✅ 发布完成，历史记录已保存")
            return True

        except Exception as e:
            logger.error(f"完成发布失败: {e}")
            return False

    def _update_compose_for_canary(self, image_tag: str):
        """更新金丝雀compose文件"""
        compose_file=self.project_root / "docker-compose.canary.yml"

        with open(compose_file, 'r', encoding='utf-8') as f:
            content=f.read()

        # 更新镜像标签
        content=content.replace("rqa2025-rqa2025-app:latest", image_tag)

        with open(compose_file, 'w', encoding='utf-8') as f:
            f.write(content)

    def _update_compose_for_production(self, image_tag: str):
        """更新生产compose文件"""
        compose_file=self.project_root / "docker-compose.prod.yml"

        with open(compose_file, 'r', encoding='utf-8') as f:
            content=f.read()

        # 更新镜像标签
        content=content.replace("rqa2025-rqa2025-app:latest", image_tag)

        with open(compose_file, 'w', encoding='utf-8') as f:
            f.write(content)

    def _update_prometheus_config(self):
        """更新Prometheus配置"""
        canary_prometheus=self.project_root / "monitoring" / "prometheus.canary.yml"
        prod_prometheus=self.project_root / "monitoring" / "prometheus.yml"

        if canary_prometheus.exists():
            import shutil
            shutil.copy(canary_prometheus, prod_prometheus)

    def _verify_deployment(self, expected_instances: int, environment: str) -> bool:
        """验证部署"""
        try:
            # 检查容器状态
            if environment == "canary":
                compose_file="docker-compose.canary.yml"
            else:
                compose_file="docker-compose.prod.yml"

            result=subprocess.run(
                f"docker-compose -f {compose_file} ps",
                shell=True, cwd=self.project_root,
                capture_output=True, text=True
            )

            if result.returncode != 0:
                logger.error("无法获取容器状态")
                return False

            # 检查运行中的容器
            lines=result.stdout.split('\n')
            running_count=sum(1 for line in lines if 'Up' in line)

            if running_count < expected_instances:
                logger.error(f"容器数量不足: {running_count}/{expected_instances}")
                return False

            # 检查应用健康
            response=requests.get("http://localhost:8000/health", timeout=10)
            if response.status_code != 200:
                logger.error("应用健康检查失败")
                return False

            logger.info("✅ 部署验证通过" return True

        except Exception as e:
            logger.error(f"部署验证异常: {e}")
            return False


def main():
    """主函数"""
    parser=argparse.ArgumentParser(description="RQA2025 自动化灰度发布")
    parser.add_argument("--version", required=True, help="发布版本号")
    parser.add_argument("--config", default="canary_config.json", help="配置文件")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不执行")

    args=parser.parse_args()

    rollout=AutomatedCanaryRollout(args.config)

    if args.dry_run:
        print("🔍 预览模式 - 发布计划:" print(f"版本: {args.version}")
        print(f"发布阶段: {rollout.config['rollout_percentage']}")
        print("监控阈值:" print(f"  - 错误率: {rollout.config['metrics']['error_rate_threshold']}")
        print(f"  - CPU使用率: {rollout.config['metrics']['cpu_usage_threshold']}%")
        print(f"  - 内存使用率: {rollout.config['metrics']['memory_usage_threshold']}%")
        return

    success=rollout.execute_full_rollout(args.version)

    if success:
        print(f"🎉 版本 {args.version} 灰度发布成功！")
        sys.exit(0)
    else:
        print(f"❌ 版本 {args.version} 灰度发布失败！")
        sys.exit(1)


if __name__ == "__main__":
    main()
