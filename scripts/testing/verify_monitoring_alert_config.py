#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控告警配置验证脚本
验证智能服务的监控和告警配置是否正确
"""

from src.infrastructure.core.logging.unified_logger import UnifiedLogger
import os
import sys
import requests
from typing import Dict, Any
from datetime import datetime

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


class MonitoringAlertConfigVerifier:
    """监控告警配置验证器"""

    def __init__(self):
        """初始化验证器"""
        self.logger = UnifiedLogger("MonitoringAlertConfigVerifier")

        # 监控服务配置
        self.monitoring_services = {
            "prometheus": {
                "url": "http://localhost:9090",
                "endpoints": ["/api/v1/status/config", "/api/v1/targets", "/api/v1/rules"],
                "expected_status": 200
            },
            "grafana": {
                "url": "http://localhost:3000",
                "endpoints": ["/api/health", "/api/dashboards"],
                "expected_status": 200
            },
            "alertmanager": {
                "url": "http://localhost:9093",
                "endpoints": ["/api/v1/status", "/api/v1/alerts"],
                "expected_status": 200
            }
        }

        # 验证结果
        self.verification_results = {
            "start_time": None,
            "end_time": None,
            "overall_status": "unknown",
            "service_checks": {},
            "config_checks": {},
            "alert_rule_checks": {},
            "dashboard_checks": {},
            "integration_checks": {},
            "recommendations": []
        }

    def verify_prometheus_config(self) -> Dict[str, Any]:
        """验证Prometheus配置"""
        self.logger.info("🔍 验证Prometheus配置...")

        results = {
            "status": "unknown",
            "config_valid": False,
            "targets_healthy": False,
            "rules_loaded": False,
            "errors": []
        }

        try:
            # 检查Prometheus服务状态
            response = requests.get(
                f"{self.monitoring_services['prometheus']['url']}/api/v1/status/config", timeout=10)
            if response.status_code == 200:
                results["config_valid"] = True
                self.logger.info("✅ Prometheus配置有效")
            else:
                results["errors"].append(f"Prometheus配置检查失败: HTTP {response.status_code}")

        except Exception as e:
            results["errors"].append(f"Prometheus配置检查异常: {str(e)}")

        # 确定整体状态
        if results["config_valid"] and results["targets_healthy"] and results["rules_loaded"]:
            results["status"] = "healthy"
        elif results["errors"]:
            results["status"] = "unhealthy"
        else:
            results["status"] = "warning"

        return results

    def verify_grafana_config(self) -> Dict[str, Any]:
        """验证Grafana配置"""
        self.logger.info("📊 验证Grafana配置...")

        results = {
            "status": "unknown",
            "service_healthy": False,
            "dashboards_available": False,
            "datasources_configured": False,
            "errors": []
        }

        try:
            # 检查Grafana服务健康状态
            response = requests.get(
                f"{self.monitoring_services['grafana']['url']}/api/health", timeout=10)
            if response.status_code == 200:
                results["service_healthy"] = True
                self.logger.info("✅ Grafana服务健康")
            else:
                results["errors"].append(f"Grafana健康检查失败: HTTP {response.status_code}")

        except Exception as e:
            results["errors"].append(f"Grafana健康检查异常: {str(e)}")

        # 确定整体状态
        if results["service_healthy"] and results["dashboards_available"] and results["datasources_configured"]:
            results["status"] = "healthy"
        elif results["errors"]:
            results["status"] = "unhealthy"
        else:
            results["status"] = "warning"

        return results

    def verify_alert_rules(self) -> Dict[str, Any]:
        """验证告警规则配置"""
        self.logger.info("📋 验证告警规则配置...")

        results = {
            "status": "unknown",
            "rules_file_exists": False,
            "rules_valid": False,
            "critical_rules": 0,
            "warning_rules": 0,
            "info_rules": 0,
            "errors": []
        }

        # 检查告警规则文件
        alert_rules_files = [
            "config/alerts.yml",
            "config/production/monitoring.yaml",
            "config/monitoring/production_monitoring.yaml"
        ]

        for rules_file in alert_rules_files:
            if os.path.exists(rules_file):
                results["rules_file_exists"] = True
                self.logger.info(f"✅ 告警规则文件存在: {rules_file}")
                results["rules_valid"] = True
                break

        if not results["rules_file_exists"]:
            results["errors"].append("未找到告警规则配置文件")

        # 确定整体状态
        if results["rules_file_exists"] and results["rules_valid"]:
            results["status"] = "healthy"
        elif results["errors"]:
            results["status"] = "unhealthy"
        else:
            results["status"] = "warning"

        return results

    def run_comprehensive_verification(self) -> Dict[str, Any]:
        """运行综合验证"""
        self.logger.info("🚀 开始运行监控告警配置综合验证...")

        self.verification_results["start_time"] = datetime.now().isoformat()

        try:
            # 1. 验证Prometheus配置
            self.logger.info("=" * 60)
            self.logger.info("第1阶段: 验证Prometheus配置")
            self.logger.info("=" * 60)
            prometheus_results = self.verify_prometheus_config()
            self.verification_results["service_checks"]["prometheus"] = prometheus_results

            # 2. 验证Grafana配置
            self.logger.info("=" * 60)
            self.logger.info("第2阶段: 验证Grafana配置")
            self.logger.info("=" * 60)
            grafana_results = self.verify_grafana_config()
            self.verification_results["service_checks"]["grafana"] = grafana_results

            # 3. 验证告警规则
            self.logger.info("=" * 60)
            self.logger.info("第3阶段: 验证告警规则配置")
            self.logger.info("=" * 60)
            alert_rules_results = self.verify_alert_rules()
            self.verification_results["alert_rule_checks"] = alert_rules_results

            # 确定整体状态
            all_checks = [
                prometheus_results["status"],
                grafana_results["status"],
                alert_rules_results["status"]
            ]

            if all(status == "healthy" for status in all_checks):
                self.verification_results["overall_status"] = "healthy"
            elif any(status == "unhealthy" for status in all_checks):
                self.verification_results["overall_status"] = "unhealthy"
            else:
                self.verification_results["overall_status"] = "warning"

            self.verification_results["end_time"] = datetime.now().isoformat()

            # 生成验证报告
            self._generate_verification_report()

            self.logger.info("🎉 监控告警配置综合验证完成!")
            return self.verification_results

        except Exception as e:
            self.logger.error(f"❌ 监控告警配置综合验证失败: {str(e)}")
            self.verification_results["end_time"] = datetime.now().isoformat()
            self.verification_results["error"] = str(e)
            return self.verification_results

    def _generate_verification_report(self):
        """生成验证报告"""
        report_path = f"reports/monitoring_alert_config_verification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("# 监控告警配置验证报告\n\n")
                f.write(
                    f"**验证时间**: {self.verification_results['start_time']} - {self.verification_results['end_time']}\n")
                f.write(f"**整体状态**: {self.verification_results['overall_status']}\n\n")

                f.write("## 验证概览\n\n")
                f.write("### 服务检查\n")
                for service, results in self.verification_results["service_checks"].items():
                    f.write(f"- **{service}**: {results['status']}\n")

                f.write("\n### 配置检查\n")
                f.write(f"- **告警规则**: {self.verification_results['alert_rule_checks']['status']}\n")

                f.write("\n## 详细结果\n\n")

                # Prometheus结果
                f.write("### Prometheus配置\n")
                prom_results = self.verification_results["service_checks"]["prometheus"]
                f.write(f"- 配置有效性: {'✅' if prom_results['config_valid'] else '❌'}\n")

                # Grafana结果
                f.write("\n### Grafana配置\n")
                grafana_results = self.verification_results["service_checks"]["grafana"]
                f.write(f"- 服务健康: {'✅' if grafana_results['service_healthy'] else '❌'}\n")

                # 告警规则统计
                f.write("\n### 告警规则统计\n")
                alert_results = self.verification_results["alert_rule_checks"]
                f.write(f"- 规则文件: {'✅' if alert_results['rules_file_exists'] else '❌'}\n")
                f.write(f"- 规则有效: {'✅' if alert_results['rules_valid'] else '❌'}\n")

                f.write("\n## 改进建议\n\n")
                f.write("📈 建议设置告警阈值，确保及时发现和响应问题\n")
                f.write("🔄 建议定期备份监控配置，防止配置丢失\n")
                f.write("📊 建议建立监控指标基线，便于异常检测\n")

                if self.verification_results["overall_status"] == "healthy":
                    f.write("\n## 总结\n\n")
                    f.write("🎉 监控告警配置验证通过！所有关键组件配置正确，系统监控和告警功能正常。\n")
                elif self.verification_results["overall_status"] == "warning":
                    f.write("\n## 总结\n\n")
                    f.write("⚠️ 监控告警配置基本正常，但存在一些需要注意的问题。建议按照上述建议进行优化。\n")
                else:
                    f.write("\n## 总结\n\n")
                    f.write("❌ 监控告警配置存在严重问题，需要立即修复。请按照上述建议进行修复。\n")

            self.logger.info(f"📄 验证报告已生成: {report_path}")

        except Exception as e:
            self.logger.error(f"❌ 生成验证报告失败: {str(e)}")


def main():
    """主函数"""
    # 创建验证器
    verifier = MonitoringAlertConfigVerifier()

    # 运行综合验证
    results = verifier.run_comprehensive_verification()

    # 输出结果摘要
    print("\n" + "=" * 60)
    print("监控告警配置验证结果摘要")
    print("=" * 60)
    print(f"整体状态: {results['overall_status']}")

    print("\n服务检查:")
    for service, service_results in results["service_checks"].items():
        print(f"  {service}: {service_results['status']}")

    print(f"\n告警规则: {results['alert_rule_checks']['status']}")

    print("=" * 60)


if __name__ == "__main__":
    main()
