#!/usr/bin/env python3
"""
监控反馈层部署脚本

部署内容：
1. 性能监控服务
2. 业务监控服务  
3. 告警反馈服务
4. 监控仪表板

作者: AI Assistant
创建时间: 2025-08-11
"""

from src.utils.logger import get_logger
from src.infrastructure.performance.performance_optimizer import PerformanceOptimizer
from src.infrastructure.performance.performance_dashboard import PerformanceDashboard
from src.infrastructure.health.grafana_integration import GrafanaIntegration
from src.infrastructure.health.prometheus_exporter import HealthCheckPrometheusExporter
from src.infrastructure.health.alert_manager import AlertManager
from src.infrastructure.health.enhanced_health_checker import EnhancedHealthChecker
import sys
import time
import json
from pathlib import Path
from typing import Dict

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


logger = get_logger(__name__)


class MonitoringFeedbackDeployer:
    """监控反馈层部署器"""

    def __init__(self, config_path: str = "config/monitoring.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.health_checker = None
        self.alert_manager = None
        self.prometheus_exporter = None
        self.grafana_integration = None
        self.performance_dashboard = None
        self.performance_optimizer = None

    def _load_config(self) -> Dict:
        """加载监控配置"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"加载监控配置失败: {e}")

        # 默认配置
        return {
            "health_check": {
                "enabled": True,
                "interval": 30,
                "timeout": 10
            },
            "alerting": {
                "enabled": True,
                "channels": ["email", "webhook"],
                "cooldown": 300
            },
            "prometheus": {
                "enabled": True,
                "port": 9090,
                "metrics_path": "/metrics"
            },
            "grafana": {
                "enabled": True,
                "url": "http://localhost:3000",
                "api_key": ""
            },
            "performance": {
                "enabled": True,
                "sampling_rate": 0.1,
                "retention_days": 30
            }
        }

    def deploy_health_monitoring(self) -> bool:
        """部署健康监控服务"""
        try:
            logger.info("开始部署健康监控服务...")

            # 初始化健康检查器
            health_config = {
                "check_interval": self.config["health_check"]["interval"],
                "timeout": self.config["health_check"]["timeout"],
                "grafana_enabled": self.config["grafana"]["enabled"]
            }
            self.health_checker = EnhancedHealthChecker(config=health_config)

            # 启动健康监控
            self.health_checker.start()
            logger.info("✅ 健康监控服务部署成功")
            return True

        except Exception as e:
            logger.error(f"❌ 健康监控服务部署失败: {e}")
            return False

    def deploy_alerting_service(self) -> bool:
        """部署告警服务"""
        try:
            logger.info("开始部署告警服务...")

            # 初始化告警管理器
            self.alert_manager = AlertManager(
                channels=self.config["alerting"]["channels"],
                cooldown=self.config["alerting"]["cooldown"]
            )

            # 配置告警规则
            self._setup_alert_rules()
            logger.info("✅ 告警服务部署成功")
            return True

        except Exception as e:
            logger.error(f"❌ 告警服务部署失败: {e}")
            return False

    def deploy_prometheus_exporter(self) -> bool:
        """部署Prometheus指标导出器"""
        try:
            logger.info("开始部署Prometheus指标导出器...")

            # 初始化Prometheus导出器
            self.prometheus_exporter = HealthCheckPrometheusExporter()

            # 启动指标导出（Prometheus导出器不需要显式启动）
            logger.info("✅ Prometheus指标导出器部署成功")
            return True

        except Exception as e:
            logger.error(f"❌ Prometheus指标导出器部署失败: {e}")
            return False

    def deploy_grafana_integration(self) -> bool:
        """部署Grafana集成"""
        try:
            logger.info("开始部署Grafana集成...")

            # 初始化Grafana集成
            self.grafana_integration = GrafanaIntegration(
                url=self.config["grafana"]["url"],
                api_key=self.config["grafana"]["api_key"]
            )

            # 创建默认仪表板
            self._create_default_dashboards()
            logger.info("✅ Grafana集成部署成功")
            return True

        except Exception as e:
            logger.error(f"❌ Grafana集成部署失败: {e}")
            return False

    def deploy_performance_monitoring(self) -> bool:
        """部署性能监控服务"""
        try:
            logger.info("开始部署性能监控服务...")

            # 初始化性能仪表板
            self.performance_dashboard = PerformanceDashboard(
                sampling_rate=self.config["performance"]["sampling_rate"],
                retention_days=self.config["performance"]["retention_days"]
            )

            # 初始化性能优化器
            self.performance_optimizer = PerformanceOptimizer()

            # 启动性能监控
            self.performance_dashboard.start()
            logger.info("✅ 性能监控服务部署成功")
            return True

        except Exception as e:
            logger.error(f"❌ 性能监控服务部署失败: {e}")
            return False

    def _setup_alert_rules(self):
        """设置告警规则"""
        if not self.alert_manager:
            return

        # 系统级告警规则
        system_rules = [
            {
                "name": "high_cpu_usage",
                "condition": "cpu_usage > 80",
                "severity": "warning",
                "message": "CPU使用率过高: {cpu_usage}%"
            },
            {
                "name": "high_memory_usage",
                "condition": "memory_usage > 85",
                "severity": "warning",
                "message": "内存使用率过高: {memory_usage}%"
            },
            {
                "name": "disk_space_low",
                "condition": "disk_usage > 90",
                "severity": "critical",
                "message": "磁盘空间不足: {disk_usage}%"
            }
        ]

        # 业务级告警规则
        business_rules = [
            {
                "name": "trading_error_rate",
                "condition": "error_rate > 5",
                "severity": "critical",
                "message": "交易错误率过高: {error_rate}%"
            },
            {
                "name": "risk_violation",
                "condition": "risk_score > 0.8",
                "severity": "critical",
                "message": "风险评分过高: {risk_score}"
            }
        ]

        # 添加告警规则
        for rule in system_rules + business_rules:
            self.alert_manager.add_rule(rule)

    def _create_default_dashboards(self):
        """创建默认仪表板"""
        if not self.grafana_integration:
            return

        # 系统监控仪表板
        system_dashboard = {
            "title": "系统监控",
            "panels": [
                {"title": "CPU使用率", "type": "graph", "targets": ["cpu_usage"]},
                {"title": "内存使用率", "type": "graph", "targets": ["memory_usage"]},
                {"title": "磁盘使用率", "type": "graph", "targets": ["disk_usage"]},
                {"title": "网络流量", "type": "graph", "targets": ["network_io"]}
            ]
        }

        # 业务监控仪表板
        business_dashboard = {
            "title": "业务监控",
            "panels": [
                {"title": "交易量", "type": "graph", "targets": ["trading_volume"]},
                {"title": "错误率", "type": "graph", "targets": ["error_rate"]},
                {"title": "响应时间", "type": "graph", "targets": ["response_time"]},
                {"title": "活跃用户", "type": "stat", "targets": ["active_users"]}
            ]
        }

        # 创建仪表板
        try:
            self.grafana_integration.create_dashboard(system_dashboard)
            self.grafana_integration.create_dashboard(business_dashboard)
            logger.info("默认仪表板创建成功")
        except Exception as e:
            logger.warning(f"创建默认仪表板失败: {e}")

    def deploy_all(self) -> Dict[str, bool]:
        """部署所有监控服务"""
        logger.info("🚀 开始部署监控反馈层...")

        results = {}

        # 部署健康监控
        results["health_monitoring"] = self.deploy_health_monitoring()

        # 部署告警服务
        results["alerting_service"] = self.deploy_alerting_service()

        # 部署Prometheus导出器
        results["prometheus_exporter"] = self.deploy_prometheus_exporter()

        # 部署Grafana集成
        results["grafana_integration"] = self.deploy_grafana_integration()

        # 部署性能监控
        results["performance_monitoring"] = self.deploy_performance_monitoring()

        # 汇总结果
        success_count = sum(results.values())
        total_count = len(results)

        logger.info(f"📊 部署结果汇总:")
        for service, success in results.items():
            status = "✅ 成功" if success else "❌ 失败"
            logger.info(f"  {service}: {status}")

        logger.info(f"🎯 总体结果: {success_count}/{total_count} 服务部署成功")

        if success_count == total_count:
            logger.info("🎉 监控反馈层部署完成！")
        else:
            logger.warning("⚠️ 部分服务部署失败，请检查日志")

        return results

    def health_check(self) -> Dict[str, bool]:
        """健康检查"""
        logger.info("🔍 开始健康检查...")

        health_status = {}

        # 检查健康监控服务
        if self.health_checker:
            try:
                health_status["health_monitoring"] = self.health_checker.is_healthy()
            except:
                health_status["health_monitoring"] = False

        # 检查告警服务
        if self.alert_manager:
            try:
                health_status["alerting_service"] = self.alert_manager.is_healthy()
            except:
                health_status["alerting_service"] = False

        # 检查Prometheus导出器
        if self.prometheus_exporter:
            try:
                health_status["prometheus_exporter"] = self.prometheus_exporter.is_healthy()
            except:
                health_status["prometheus_exporter"] = False

        # 检查Grafana集成
        if self.grafana_integration:
            try:
                health_status["grafana_integration"] = self.grafana_integration.is_healthy()
            except:
                health_status["grafana_integration"] = False

        # 检查性能监控
        if self.performance_dashboard:
            try:
                health_status["performance_monitoring"] = self.performance_dashboard.is_healthy()
            except:
                health_status["performance_monitoring"] = False

        # 汇总健康状态
        healthy_count = sum(health_status.values())
        total_count = len(health_status)

        logger.info(f"📊 健康检查结果:")
        for service, healthy in health_status.items():
            status = "✅ 健康" if healthy else "❌ 异常"
            logger.info(f"  {service}: {status}")

        logger.info(f"🎯 总体健康状态: {healthy_count}/{total_count} 服务健康")

        return health_status


def main():
    """主函数"""
    logger.info("🚀 监控反馈层部署脚本启动")

    # 创建部署器
    deployer = MonitoringFeedbackDeployer()

    try:
        # 部署所有服务
        deployment_results = deployer.deploy_all()

        # 等待服务启动
        logger.info("⏳ 等待服务启动...")
        time.sleep(10)

        # 健康检查
        health_status = deployer.health_check()

        # 输出最终结果
        if all(health_status.values()):
            logger.info("🎉 监控反馈层部署成功！所有服务运行正常")
            return 0
        else:
            logger.error("❌ 监控反馈层部署失败！部分服务异常")
            return 1

    except Exception as e:
        logger.error(f"❌ 部署过程中发生错误: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
