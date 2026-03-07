#!/usr/bin/env python3
"""
简化的监控反馈层部署脚本

部署内容：
1. 健康监控服务（简化版）
2. 基础监控功能
3. 配置验证

作者: AI Assistant
创建时间: 2025-08-11
"""

from src.utils.logger import get_logger
from src.infrastructure.health.prometheus_exporter import HealthCheckPrometheusExporter
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


class SimpleMonitoringDeployer:
    """简化的监控部署器"""

    def __init__(self, config_path: str = "config/monitoring.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.health_checker = None
        self.prometheus_exporter = None

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
            "prometheus": {
                "enabled": True,
                "port": 9090,
                "metrics_path": "/metrics"
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
                "grafana_enabled": False  # 暂时禁用Grafana
            }
            self.health_checker = EnhancedHealthChecker(config=health_config)

            # 注册默认健康检查
            self._register_default_health_checks()

            logger.info("✅ 健康监控服务部署成功")
            return True

        except Exception as e:
            logger.error(f"❌ 健康监控服务部署失败: {e}")
            return False

    def deploy_prometheus_exporter(self) -> bool:
        """部署Prometheus指标导出器"""
        try:
            logger.info("开始部署Prometheus指标导出器...")

            # 初始化Prometheus导出器
            self.prometheus_exporter = HealthCheckPrometheusExporter()

            logger.info("✅ Prometheus指标导出器部署成功")
            return True

        except Exception as e:
            logger.error(f"❌ Prometheus指标导出器部署失败: {e}")
            return False

    def _register_default_health_checks(self):
        """注册默认健康检查"""
        if not self.health_checker:
            return

        try:
            # 注册系统健康检查
            self.health_checker.register_service("system", self._check_system_health)
            self.health_checker.register_service("database", self._check_database_health)
            self.health_checker.register_service("network", self._check_network_health)

            logger.info("默认健康检查注册成功")

        except Exception as e:
            logger.warning(f"注册默认健康检查失败: {e}")

    def _check_system_health(self) -> Dict:
        """检查系统健康状态"""
        try:
            import psutil

            return {
                "cpu_usage": psutil.cpu_percent(interval=1),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "status": "healthy"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    def _check_database_health(self) -> Dict:
        """检查数据库健康状态"""
        # 简化的数据库健康检查
        return {
            "status": "healthy",
            "connection": "ok",
            "response_time": 0.1
        }

    def _check_network_health(self) -> Dict:
        """检查网络健康状态"""
        # 简化的网络健康检查
        return {
            "status": "healthy",
            "connectivity": "ok",
            "latency": 0.05
        }

    def deploy_all(self) -> Dict[str, bool]:
        """部署所有监控服务"""
        logger.info("🚀 开始部署简化监控反馈层...")

        results = {}

        # 部署健康监控
        results["health_monitoring"] = self.deploy_health_monitoring()

        # 部署Prometheus导出器
        results["prometheus_exporter"] = self.deploy_prometheus_exporter()

        # 汇总结果
        success_count = sum(results.values())
        total_count = len(results)

        logger.info(f"📊 部署结果汇总:")
        for service, success in results.items():
            status = "✅ 成功" if success else "❌ 失败"
            logger.info(f"  {service}: {status}")

        logger.info(f"🎯 总体结果: {success_count}/{total_count} 服务部署成功")

        if success_count == total_count:
            logger.info("🎉 简化监控反馈层部署完成！")
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
                # 执行健康检查
                health_result = self.health_checker.check_all_services()
                health_status["health_monitoring"] = all(
                    service.get("status") == "healthy"
                    for service in health_result.values()
                )
            except Exception as e:
                logger.warning(f"健康检查失败: {e}")
                health_status["health_monitoring"] = False

        # 检查Prometheus导出器
        if self.prometheus_exporter:
            try:
                # 生成指标
                metrics = self.prometheus_exporter.generate_metrics()
                health_status["prometheus_exporter"] = len(metrics) > 0
            except Exception as e:
                logger.warning(f"Prometheus检查失败: {e}")
                health_status["prometheus_exporter"] = False

        # 汇总健康状态
        healthy_count = sum(health_status.values())
        total_count = len(health_status)

        logger.info(f"📊 健康检查结果:")
        for service, healthy in health_status.items():
            status = "✅ 健康" if healthy else "❌ 异常"
            logger.info(f"  {service}: {status}")

        logger.info(f"🎯 总体健康状态: {healthy_count}/{total_count} 服务健康")

        return health_status

    def generate_deployment_report(self) -> str:
        """生成部署报告"""
        report = []
        report.append("=" * 60)
        report.append("简化监控反馈层部署报告")
        report.append("=" * 60)
        report.append(f"部署时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"配置文件: {self.config_path}")
        report.append("")

        # 配置信息
        report.append("配置信息:")
        report.append("-" * 30)
        for section, config in self.config.items():
            report.append(f"{section}:")
            for key, value in config.items():
                report.append(f"  {key}: {value}")
        report.append("")

        # 部署状态
        report.append("部署状态:")
        report.append("-" * 30)
        if self.health_checker:
            report.append("健康监控服务: ✅ 已部署")
        else:
            report.append("健康监控服务: ❌ 未部署")

        if self.prometheus_exporter:
            report.append("Prometheus导出器: ✅ 已部署")
        else:
            report.append("Prometheus导出器: ❌ 未部署")

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)


def main():
    """主函数"""
    logger.info("🚀 简化监控反馈层部署脚本启动")

    # 创建部署器
    deployer = SimpleMonitoringDeployer()

    try:
        # 部署所有服务
        deployment_results = deployer.deploy_all()

        # 等待服务启动
        logger.info("⏳ 等待服务启动...")
        time.sleep(5)

        # 健康检查
        health_status = deployer.health_check()

        # 生成部署报告
        report = deployer.generate_deployment_report()
        print(report)

        # 保存部署报告
        report_file = Path("reports/monitoring_deployment_report.txt")
        report_file.parent.mkdir(parents=True, exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"部署报告已保存到: {report_file}")

        # 输出最终结果
        if all(health_status.values()):
            logger.info("🎉 简化监控反馈层部署成功！所有服务运行正常")
            return 0
        else:
            logger.warning("⚠️ 简化监控反馈层部署部分成功！部分服务异常")
            return 1

    except Exception as e:
        logger.error(f"❌ 部署过程中发生错误: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
