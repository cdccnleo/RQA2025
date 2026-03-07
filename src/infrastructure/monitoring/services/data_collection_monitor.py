"""
数据采集监控服务

提供量化交易系统的数据采集监控功能，包括：
- 多层级股票池监控
- 数据质量检查
- 性能指标监控
- 告警管理
"""

import json
import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from src.infrastructure.logging.core.unified_logger import get_unified_logger

logger = get_unified_logger(__name__)


class DataCollectionMonitor:
    """
    数据采集监控服务

    监控量化交易系统的数据采集质量和性能
    """

    def __init__(self, config_path: str = "config/monitoring_config.json"):
        """
        初始化数据采集监控器

        Args:
            config_path: 监控配置文件路径
        """
        self.config_path = config_path
        self.config = {}
        self.metrics_cache = {}
        self.alerts_cache = []
        self.last_check_time = 0
        self.check_interval = 60  # 60秒检查一次

        self.load_config()

    def load_config(self):
        """加载监控配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            logger.info(f"数据采集监控配置加载成功: {self.config_path}")
        except Exception as e:
            logger.error(f"加载监控配置失败: {e}")
            # 使用默认配置
            self.config = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认监控配置"""
        return {
            "data_collection_monitoring": {
                "pools": {
                    "core_universe": {
                        "metrics": {
                            "data_freshness_threshold": 300,
                            "data_completeness_threshold": 0.999
                        }
                    }
                }
            }
        }

    def check_data_freshness(self, pool_name: str, source_id: str) -> Dict[str, Any]:
        """
        检查数据新鲜度

        Args:
            pool_name: 股票池名称
            source_id: 数据源ID

        Returns:
            检查结果
        """
        try:
            pool_config = self.config.get("data_collection_monitoring", {}).get("pools", {}).get(pool_name, {})
            threshold = pool_config.get("metrics", {}).get("data_freshness_threshold", 3600)

            # 从数据库或缓存获取最后更新时间
            last_update = self._get_last_update_time(source_id)
            current_time = time.time()
            age_seconds = current_time - last_update

            is_fresh = age_seconds <= threshold

            result = {
                "pool_name": pool_name,
                "source_id": source_id,
                "last_update": last_update,
                "age_seconds": age_seconds,
                "threshold": threshold,
                "is_fresh": is_fresh,
                "status": "fresh" if is_fresh else "stale"
            }

            # 检查是否需要告警
            if not is_fresh:
                alert = pool_config.get("alerts", {}).get("data_freshness", {})
                if alert.get("enabled", False):
                    self._trigger_alert(
                        alert_type="data_freshness",
                        severity=alert.get("severity", "LOW"),
                        message=alert.get("message", "").format(
                            value=int(age_seconds),
                            threshold=threshold
                        ),
                        details=result
                    )

            return result

        except Exception as e:
            logger.error(f"检查数据新鲜度失败: {e}")
            return {
                "error": str(e),
                "pool_name": pool_name,
                "source_id": source_id,
                "status": "error"
            }

    def check_data_completeness(self, pool_name: str, source_id: str) -> Dict[str, Any]:
        """
        检查数据完整性

        Args:
            pool_name: 股票池名称
            source_id: 数据源ID

        Returns:
            检查结果
        """
        try:
            pool_config = self.config.get("data_collection_monitoring", {}).get("pools", {}).get(pool_name, {})
            threshold = pool_config.get("metrics", {}).get("data_completeness_threshold", 0.95)

            # 计算数据完整性
            completeness = self._calculate_data_completeness(pool_name, source_id)

            is_complete = completeness >= threshold

            result = {
                "pool_name": pool_name,
                "source_id": source_id,
                "completeness": completeness,
                "threshold": threshold,
                "is_complete": is_complete,
                "status": "complete" if is_complete else "incomplete"
            }

            # 检查是否需要告警
            if not is_complete:
                alert = pool_config.get("alerts", {}).get("data_completeness", {})
                if alert.get("enabled", False):
                    self._trigger_alert(
                        alert_type="data_completeness",
                        severity=alert.get("severity", "MEDIUM"),
                        message=alert.get("message", "").format(
                            value=f"{completeness:.3f}",
                            threshold=f"{threshold:.3f}"
                        ),
                        details=result
                    )

            return result

        except Exception as e:
            logger.error(f"检查数据完整性失败: {e}")
            return {
                "error": str(e),
                "pool_name": pool_name,
                "source_id": source_id,
                "status": "error"
            }

    def check_collection_performance(self, pool_name: str, source_id: str) -> Dict[str, Any]:
        """
        检查采集性能

        Args:
            pool_name: 股票池名称
            source_id: 数据源ID

        Returns:
            性能检查结果
        """
        try:
            # 获取采集统计信息
            stats = self._get_collection_stats(source_id)

            # 检查采集成功率
            success_rate = stats.get("success_rate", 1.0)
            pool_config = self.config.get("data_collection_monitoring", {}).get("pools", {}).get(pool_name, {})
            threshold = pool_config.get("metrics", {}).get("collection_success_rate_threshold", 0.95)

            is_successful = success_rate >= threshold

            result = {
                "pool_name": pool_name,
                "source_id": source_id,
                "success_rate": success_rate,
                "threshold": threshold,
                "is_successful": is_successful,
                "total_attempts": stats.get("total_attempts", 0),
                "successful_attempts": stats.get("successful_attempts", 0),
                "failed_attempts": stats.get("failed_attempts", 0),
                "avg_latency": stats.get("avg_latency", 0),
                "status": "healthy" if is_successful else "degraded"
            }

            # 检查是否需要告警
            if not is_successful:
                alert = pool_config.get("alerts", {}).get("collection_failure_rate", {})
                if alert.get("enabled", False):
                    failure_rate = 1.0 - success_rate
                    self._trigger_alert(
                        alert_type="collection_failure_rate",
                        severity=alert.get("severity", "MEDIUM"),
                        message=alert.get("message", "").format(
                            value=f"{failure_rate:.3f}",
                            threshold=f"{1-threshold:.3f}"
                        ),
                        details=result
                    )

            return result

        except Exception as e:
            logger.error(f"检查采集性能失败: {e}")
            return {
                "error": str(e),
                "pool_name": pool_name,
                "source_id": source_id,
                "status": "error"
            }

    def check_system_health(self) -> Dict[str, Any]:
        """
        检查系统健康状态

        Returns:
            系统健康检查结果
        """
        try:
            system_config = self.config.get("data_collection_monitoring", {}).get("system_metrics", {})

            # 检查CPU使用率
            cpu_usage = self._get_cpu_usage()
            cpu_warning = system_config.get("cpu_usage", {}).get("warning_threshold", 70)
            cpu_critical = system_config.get("cpu_usage", {}).get("critical_threshold", 85)

            # 检查内存使用率
            memory_usage = self._get_memory_usage()
            memory_warning = system_config.get("memory_usage", {}).get("warning_threshold", 75)
            memory_critical = system_config.get("memory_usage", {}).get("critical_threshold", 90)

            # 检查磁盘使用率
            disk_usage = self._get_disk_usage()
            disk_warning = system_config.get("disk_usage", {}).get("warning_threshold", 80)
            disk_critical = system_config.get("disk_usage", {}).get("critical_threshold", 95)

            result = {
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "disk_usage": disk_usage,
                "cpu_status": self._get_status(cpu_usage, cpu_warning, cpu_critical),
                "memory_status": self._get_status(memory_usage, memory_warning, memory_critical),
                "disk_status": self._get_status(disk_usage, disk_warning, disk_critical),
                "overall_status": "healthy"
            }

            # 确定整体状态
            if any(status == "critical" for status in [
                result["cpu_status"], result["memory_status"], result["disk_status"]
            ]):
                result["overall_status"] = "critical"
            elif any(status == "warning" for status in [
                result["cpu_status"], result["memory_status"], result["disk_status"]
            ]):
                result["overall_status"] = "warning"

            return result

        except Exception as e:
            logger.error(f"检查系统健康状态失败: {e}")
            return {"error": str(e), "overall_status": "error"}

    def run_monitoring_cycle(self):
        """运行监控周期"""
        try:
            current_time = time.time()

            # 检查是否到了监控周期
            if current_time - self.last_check_time < self.check_interval:
                return

            self.last_check_time = current_time
            logger.info("开始数据采集监控周期...")

            # 检查各个股票池的数据质量
            pools = self.config.get("data_collection_monitoring", {}).get("pools", {})

            for pool_name, pool_config in pools.items():
                # 检查数据新鲜度
                self.check_data_freshness(pool_name, f"akshare_stock_a_{pool_name.replace('_universe', '')}")

                # 检查数据完整性
                self.check_data_completeness(pool_name, f"akshare_stock_a_{pool_name.replace('_universe', '')}")

                # 检查采集性能
                self.check_collection_performance(pool_name, f"akshare_stock_a_{pool_name.replace('_universe', '')}")

            # 检查系统健康状态
            system_health = self.check_system_health()
            if system_health.get("overall_status") != "healthy":
                logger.warning(f"系统健康状态异常: {system_health}")

            # 清理过期告警
            self._cleanup_expired_alerts()

            logger.info("数据采集监控周期完成")

        except Exception as e:
            logger.error(f"运行监控周期失败: {e}")

    def _get_last_update_time(self, source_id: str) -> float:
        """获取数据源最后更新时间"""
        # 这里应该从数据库或缓存获取实际的最后更新时间
        # 暂时返回当前时间减去一些随机时间作为模拟
        return time.time() - 300  # 模拟5分钟前更新

    def _calculate_data_completeness(self, pool_name: str, source_id: str) -> float:
        """计算数据完整性"""
        # 这里应该计算实际的数据完整性百分比
        # 暂时返回高完整性作为模拟
        return 0.995

    def _get_collection_stats(self, source_id: str) -> Dict[str, Any]:
        """获取采集统计信息"""
        # 这里应该从数据库获取实际的采集统计
        return {
            "success_rate": 0.998,
            "total_attempts": 1000,
            "successful_attempts": 998,
            "failed_attempts": 2,
            "avg_latency": 2500
        }

    def _get_cpu_usage(self) -> float:
        """获取CPU使用率"""
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except:
            return 50.0  # 默认值

    def _get_memory_usage(self) -> float:
        """获取内存使用率"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except:
            return 60.0  # 默认值

    def _get_disk_usage(self) -> float:
        """获取磁盘使用率"""
        try:
            import psutil
            return psutil.disk_usage('/').percent
        except:
            return 40.0  # 默认值

    def _get_status(self, value: float, warning_threshold: float, critical_threshold: float) -> str:
        """根据阈值获取状态"""
        if value >= critical_threshold:
            return "critical"
        elif value >= warning_threshold:
            return "warning"
        else:
            return "normal"

    def _trigger_alert(self, alert_type: str, severity: str, message: str, details: Dict[str, Any]):
        """触发告警"""
        try:
            alert = {
                "alert_type": alert_type,
                "severity": severity,
                "message": message,
                "details": details,
                "timestamp": time.time(),
                "resolved": False
            }

            self.alerts_cache.append(alert)

            # 发送告警通知
            self._send_alert_notification(alert)

            logger.warning(f"触发告警: [{severity}] {message}")

        except Exception as e:
            logger.error(f"触发告警失败: {e}")

    def _send_alert_notification(self, alert: Dict[str, Any]):
        """发送告警通知"""
        try:
            # 这里可以实现多种通知渠道：邮件、Slack、微信等
            alert_channels = self.config.get("data_collection_monitoring", {}).get("alert_channels", {})

            # 邮件通知
            if alert_channels.get("email", {}).get("enabled", False):
                self._send_email_alert(alert)

            # Slack通知
            if alert_channels.get("slack", {}).get("enabled", False):
                self._send_slack_alert(alert)

            # 微信通知
            if alert_channels.get("wechat", {}).get("enabled", False):
                self._send_wechat_alert(alert)

        except Exception as e:
            logger.error(f"发送告警通知失败: {e}")

    def _send_email_alert(self, alert: Dict[str, Any]):
        """发送邮件告警"""
        # 实现邮件发送逻辑
        logger.info(f"发送邮件告警: {alert['message']}")

    def _send_slack_alert(self, alert: Dict[str, Any]):
        """发送Slack告警"""
        # 实现Slack通知逻辑
        logger.info(f"发送Slack告警: {alert['message']}")

    def _send_wechat_alert(self, alert: Dict[str, Any]):
        """发送微信告警"""
        # 实现微信通知逻辑
        logger.info(f"发送微信告警: {alert['message']}")

    def _cleanup_expired_alerts(self):
        """清理过期告警"""
        try:
            # 保留最近24小时的告警
            cutoff_time = time.time() - 86400
            self.alerts_cache = [
                alert for alert in self.alerts_cache
                if alert["timestamp"] > cutoff_time
            ]
        except Exception as e:
            logger.error(f"清理过期告警失败: {e}")

    def get_monitoring_report(self) -> Dict[str, Any]:
        """获取监控报告"""
        try:
            report = {
                "timestamp": time.time(),
                "config": self.config,
                "active_alerts": self.alerts_cache,
                "system_health": self.check_system_health(),
                "pools_status": {}
            }

            # 检查各个池的状态
            pools = self.config.get("data_collection_monitoring", {}).get("pools", {})
            for pool_name in pools.keys():
                pool_status = {
                    "data_freshness": self.check_data_freshness(pool_name, f"akshare_stock_a_{pool_name.replace('_universe', '')}"),
                    "data_completeness": self.check_data_completeness(pool_name, f"akshare_stock_a_{pool_name.replace('_universe', '')}"),
                    "collection_performance": self.check_collection_performance(pool_name, f"akshare_stock_a_{pool_name.replace('_universe', '')}")
                }
                report["pools_status"][pool_name] = pool_status

            return report

        except Exception as e:
            logger.error(f"生成监控报告失败: {e}")
            return {"error": str(e)}


# 全局监控实例
_monitor_instance = None

def get_data_collection_monitor() -> DataCollectionMonitor:
    """获取数据采集监控器实例（单例模式）"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = DataCollectionMonitor()
    return _monitor_instance