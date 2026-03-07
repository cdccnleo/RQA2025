#!/usr/bin/env python3
"""
日志轮转脚本
RQA2025 生产环境日志管理工具
"""

import sys
import time
import logging
import subprocess
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('log_rotator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class LogRotationConfig:
    """日志轮转配置类"""
    namespace: str = "rqa2025-production"
    log_paths: List[str] = None
    max_size_mb: int = 100
    retention_days: int = 7
    compression: bool = True
    auto_rotation: bool = True


class LogRotator:
    """日志轮转器"""

    def __init__(self, config: LogRotationConfig):
        self.config = config
        if self.config.log_paths is None:
            self.config.log_paths = [
                "/app/logs",
                "/var/log/rqa2025"
            ]

    def rotate_logs(self) -> bool:
        """轮转日志文件"""
        logger.info("🔄 开始日志轮转...")

        try:
            # 获取Pod列表
            pods = self._get_pods()
            if not pods:
                logger.warning("⚠️ 未找到运行中的Pod")
                return True

            success_count = 0
            for pod in pods:
                if self._rotate_pod_logs(pod):
                    success_count += 1

            logger.info(f"✅ 日志轮转完成: {success_count}/{len(pods)} 个Pod")
            return success_count == len(pods)

        except Exception as e:
            logger.error(f"❌ 日志轮转失败: {e}")
            return False

    def _get_pods(self) -> List[str]:
        """获取Pod列表"""
        try:
            result = subprocess.run([
                "kubectl", "get", "pods", "-n", self.config.namespace,
                "-l", "app=rqa2025", "-o", "jsonpath={.items[*].metadata.name}"
            ], capture_output=True, text=True)

            if result.returncode == 0:
                pod_names = result.stdout.strip().split()
                return [name for name in pod_names if name]
            else:
                logger.error(f"❌ 获取Pod列表失败: {result.stderr}")
                return []

        except Exception as e:
            logger.error(f"❌ 获取Pod列表异常: {e}")
            return []

    def _rotate_pod_logs(self, pod_name: str) -> bool:
        """轮转单个Pod的日志"""
        logger.info(f"📋 轮转Pod日志: {pod_name}")

        try:
            for log_path in self.config.log_paths:
                # 检查日志文件大小
                size_result = subprocess.run([
                    "kubectl", "exec", pod_name, "-n", self.config.namespace,
                    "--", "du", "-sm", log_path
                ], capture_output=True, text=True)

                if size_result.returncode == 0:
                    try:
                        size_mb = int(size_result.stdout.split()[0])
                        if size_mb > self.config.max_size_mb:
                            logger.info(f"📏 Pod {pod_name} 日志大小: {size_mb}MB")
                            self._perform_log_rotation(pod_name, log_path)
                    except (ValueError, IndexError):
                        logger.warning(f"⚠️ 无法解析日志大小: {size_result.stdout}")
                else:
                    logger.warning(f"⚠️ 无法获取Pod {pod_name} 日志大小")

            return True

        except Exception as e:
            logger.error(f"❌ 轮转Pod {pod_name} 日志失败: {e}")
            return False

    def _perform_log_rotation(self, pod_name: str, log_path: str):
        """执行日志轮转"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 1. 压缩当前日志
            compress_result = subprocess.run([
                "kubectl", "exec", pod_name, "-n", self.config.namespace,
                "--", "find", log_path, "-name", "*.log", "-exec", "gzip", "{}", ";"
            ], capture_output=True, text=True)

            if compress_result.returncode != 0:
                logger.warning(f"⚠️ 压缩日志失败: {compress_result.stderr}")

            # 2. 重命名压缩文件
            rename_result = subprocess.run([
                "kubectl", "exec", pod_name, "-n", self.config.namespace,
                "--", "find", log_path, "-name", "*.log.gz", "-exec", "mv", "{}", "{}.{}", ";"
            ], capture_output=True, text=True)

            # 3. 清理旧日志
            self._cleanup_old_logs(pod_name, log_path)

            logger.info(f"✅ Pod {pod_name} 日志轮转完成")

        except Exception as e:
            logger.error(f"❌ 执行日志轮转失败: {e}")

    def _cleanup_old_logs(self, pod_name: str, log_path: str):
        """清理旧日志"""
        try:
            # 删除超过保留期的日志文件
            cleanup_result = subprocess.run([
                "kubectl", "exec", pod_name, "-n", self.config.namespace,
                "--", "find", log_path, "-name", "*.log.gz.*", "-mtime", f"+{self.config.retention_days}", "-delete"
            ], capture_output=True, text=True)

            if cleanup_result.returncode == 0:
                logger.info(f"🗑️ Pod {pod_name} 旧日志清理完成")
            else:
                logger.warning(f"⚠️ Pod {pod_name} 旧日志清理失败: {cleanup_result.stderr}")

        except Exception as e:
            logger.error(f"❌ 清理旧日志失败: {e}")

    def get_log_stats(self) -> Dict:
        """获取日志统计信息"""
        logger.info("📊 获取日志统计信息...")

        stats = {
            "total_pods": 0,
            "total_log_size_mb": 0,
            "log_files": 0,
            "oldest_log": None,
            "newest_log": None
        }

        try:
            pods = self._get_pods()
            stats["total_pods"] = len(pods)

            for pod in pods:
                pod_stats = self._get_pod_log_stats(pod)
                stats["total_log_size_mb"] += pod_stats.get("size_mb", 0)
                stats["log_files"] += pod_stats.get("file_count", 0)

                # 更新最旧和最新日志时间
                oldest = pod_stats.get("oldest_log")
                newest = pod_stats.get("newest_log")

                if oldest and (not stats["oldest_log"] or oldest < stats["oldest_log"]):
                    stats["oldest_log"] = oldest
                if newest and (not stats["newest_log"] or newest > stats["newest_log"]):
                    stats["newest_log"] = newest

            return stats

        except Exception as e:
            logger.error(f"❌ 获取日志统计失败: {e}")
            return stats

    def _get_pod_log_stats(self, pod_name: str) -> Dict:
        """获取单个Pod的日志统计"""
        stats = {
            "size_mb": 0,
            "file_count": 0,
            "oldest_log": None,
            "newest_log": None
        }

        try:
            for log_path in self.config.log_paths:
                # 获取日志文件大小
                size_result = subprocess.run([
                    "kubectl", "exec", pod_name, "-n", self.config.namespace,
                    "--", "du", "-sm", log_path
                ], capture_output=True, text=True)

                if size_result.returncode == 0:
                    try:
                        stats["size_mb"] += int(size_result.stdout.split()[0])
                    except (ValueError, IndexError):
                        pass

                # 获取日志文件数量
                count_result = subprocess.run([
                    "kubectl", "exec", pod_name, "-n", self.config.namespace,
                    "--", "find", log_path, "-name", "*.log*", "-type", "f", "|", "wc", "-l"
                ], capture_output=True, text=True)

                if count_result.returncode == 0:
                    try:
                        stats["file_count"] += int(count_result.stdout.strip())
                    except ValueError:
                        pass

            return stats

        except Exception as e:
            logger.error(f"❌ 获取Pod {pod_name} 日志统计失败: {e}")
            return stats

    def schedule_rotation(self, interval_hours: int = 24):
        """调度定时日志轮转"""
        logger.info(f"⏰ 设置定时日志轮转: 每{interval_hours}小时")

        def rotation_job():
            logger.info("🔄 执行定时日志轮转...")
            if self.rotate_logs():
                logger.info("✅ 定时日志轮转完成")
            else:
                logger.error("❌ 定时日志轮转失败")

        # 设置定时任务
        import schedule
        schedule.every(interval_hours).hours.do(rotation_job)

        logger.info(f"✅ 定时日志轮转已设置: 每{interval_hours}小时")

        # 运行调度器
        while True:
            schedule.run_pending()
            time.sleep(60)


def main():
    """主函数"""
    print("🔄 RQA2025 日志轮转工具")
    print("=" * 50)

    # 创建日志轮转配置
    config = LogRotationConfig()

    # 创建日志轮转器
    rotator = LogRotator(config)

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "rotate":
            success = rotator.rotate_logs()
            if success:
                print("✅ 日志轮转成功")
            else:
                print("❌ 日志轮转失败")
                sys.exit(1)

        elif command == "stats":
            stats = rotator.get_log_stats()
            print("📊 日志统计信息:")
            print(f"  Pod数量: {stats['total_pods']}")
            print(f"  总日志大小: {stats['total_log_size_mb']}MB")
            print(f"  日志文件数: {stats['log_files']}")
            print(f"  最旧日志: {stats['oldest_log']}")
            print(f"  最新日志: {stats['newest_log']}")

        elif command == "schedule":
            interval = int(sys.argv[2]) if len(sys.argv) > 2 else 24
            print(f"⏰ 启动定时日志轮转调度器 (每{interval}小时)")
            rotator.schedule_rotation(interval)

        else:
            print("❌ 未知命令")
            print("可用命令: rotate, stats, schedule [interval_hours]")
            sys.exit(1)
    else:
        print("用法: python log_rotator.py <command> [options]")
        print("命令:")
        print("  rotate              - 执行日志轮转")
        print("  stats               - 显示日志统计")
        print("  schedule [hours]    - 启动定时轮转调度器")
        sys.exit(1)


if __name__ == "__main__":
    main()
