#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RQA2025 健康检查模块集成演示
展示新组件的协作和集成功能
"""

from src.infrastructure.health import (
    get_enhanced_health_checker,
    AlertRule,
    AlertSeverity,
    get_performance_optimizer,
    get_grafana_integration,
    get_alert_rule_engine
)
import asyncio
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入健康检查模块


class HealthCheckIntegrationDemo:
    """健康检查集成演示类"""

    def __init__(self):
        """初始化演示环境"""
        self.config = {
            'monitoring_enabled': True,
            'alerting_enabled': True,
            'cache_enabled': True,
            'grafana_enabled': False,  # 演示环境禁用Grafana
            'cache_ttl': 60,
            'system_metrics_cache_ttl': 30
        }

        # 初始化组件
        self.health_checker = None
        self.performance_optimizer = None
        self.grafana_integration = None
        self.alert_rule_engine = None

        logger.info("健康检查集成演示环境初始化完成")

    async def setup_components(self):
        """设置所有组件"""
        logger.info("正在设置健康检查组件...")

        # 创建增强健康检查器
        self.health_checker = get_enhanced_health_checker()
        logger.info("✓ 增强健康检查器已创建")

        # 创建性能优化器
        self.performance_optimizer = get_performance_optimizer(
            cache_manager=self.health_checker.cache_manager,
            prometheus_exporter=self.health_checker.prometheus_exporter
        )
        logger.info("✓ 性能优化器已创建")

        # 创建Grafana集成（可选）
        try:
            self.grafana_integration = get_grafana_integration(
                grafana_url="http://localhost:3000",
                api_key="demo-key",
                org_id=1
            )
            logger.info("✓ Grafana集成已创建")
        except Exception as e:
            logger.warning(f"Grafana集成创建失败: {e}")

        # 创建告警规则引擎
        self.alert_rule_engine = get_alert_rule_engine(
            alert_manager=self.health_checker.alert_manager,
            prometheus_exporter=self.health_checker.prometheus_exporter
        )
        logger.info("✓ 告警规则引擎已创建")

        logger.info("所有组件设置完成")

    def setup_alert_rules(self):
        """设置演示告警规则"""
        logger.info("正在设置演示告警规则...")

        # 系统资源告警规则
        system_rules = [
            AlertRule(
                name="demo_high_cpu",
                description="演示CPU使用率告警",
                query="rqa_system_cpu_percent > 70",
                severity=AlertSeverity.WARNING,
                threshold=70.0,
                duration="2m",
                auto_threshold=True,
                adjustment_factor=1.2
            ),
            AlertRule(
                name="demo_high_memory",
                description="演示内存使用率告警",
                query="rqa_system_memory_percent > 80",
                severity=AlertSeverity.WARNING,
                threshold=80.0,
                duration="2m",
                auto_threshold=True,
                adjustment_factor=1.1
            )
        ]

        # 性能告警规则
        performance_rules = [
            AlertRule(
                name="demo_slow_response",
                description="演示响应时间告警",
                query="rqa_health_response_time_seconds > 1.5",
                severity=AlertSeverity.INFO,
                threshold=1.5,
                duration="1m",
                auto_threshold=True,
                adjustment_factor=1.1
            )
        ]

        # 添加所有规则
        all_rules = system_rules + performance_rules
        for rule in all_rules:
            success = self.alert_rule_engine.add_rule(rule)
            if success:
                logger.info(f"✓ 告警规则已添加: {rule.name}")
            else:
                logger.warning(f"✗ 告警规则添加失败: {rule.name}")

        logger.info(f"告警规则设置完成，共 {len(all_rules)} 条规则")

    async def demonstrate_health_checks(self):
        """演示健康检查功能"""
        logger.info("开始演示健康检查功能...")

        # 1. 基础健康检查
        logger.info("1. 执行基础健康检查...")
        health_result = await self.health_checker.perform_health_check(
            "demo_service",
            "liveness"
        )
        logger.info(f"   健康检查结果: {health_result.status}")
        logger.info(f"   响应时间: {health_result.response_time:.3f}s")

        # 2. 系统指标收集
        logger.info("2. 收集系统指标...")
        system_metrics = await self.health_checker.get_system_metrics()
        logger.info(f"   CPU使用率: {system_metrics.cpu_percent:.1f}%")
        logger.info(f"   内存使用率: {system_metrics.memory_percent:.1f}%")
        logger.info(f"   磁盘使用情况: {len(system_metrics.disk_usage)} 个分区")

        # 3. 综合健康状态
        logger.info("3. 获取综合健康状态...")
        comprehensive_status = await self.health_checker.get_comprehensive_health_status()
        logger.info(f"   整体状态: {comprehensive_status['overall_status']}")
        logger.info(f"   检查项目数: {len(comprehensive_status['checks'])}")

        logger.info("健康检查功能演示完成")

    def demonstrate_performance_optimization(self):
        """演示性能优化功能"""
        logger.info("开始演示性能优化功能...")

        # 1. 记录性能指标
        logger.info("1. 记录性能指标...")
        metrics = [
            ("response_time", 0.15),
            ("response_time", 0.18),
            ("response_time", 0.12),
            ("cache_hit_rate", 0.85),
            ("cache_hit_rate", 0.82),
            ("cache_hit_rate", 0.88),
            ("memory_usage", 65.2),
            ("memory_usage", 67.8),
            ("memory_usage", 64.5)
        ]

        for metric_name, value in metrics:
            self.performance_optimizer.record_metric(metric_name, value)

        logger.info(f"   已记录 {len(metrics)} 个性能指标")

        # 2. 获取性能报告
        logger.info("2. 获取性能报告...")
        performance_report = self.performance_optimizer.get_performance_report()

        if 'metrics_summary' in performance_report:
            for metric_name, stats in performance_report['metrics_summary'].items():
                logger.info(f"   {metric_name}:")
                logger.info(f"     平均值: {stats.get('mean', 0):.3f}")
                logger.info(f"     标准差: {stats.get('std_dev', 0):.3f}")
                logger.info(f"     趋势: {stats.get('trend', 'unknown')}")

        # 3. 获取性能优化建议
        logger.info("3. 获取性能优化建议...")
        suggestions = self.performance_optimizer.analyze_performance()

        for i, suggestion in enumerate(suggestions, 1):
            logger.info(f"   建议 {i}: {suggestion.get('description', 'N/A')}")
            logger.info(f"     优先级: {suggestion.get('priority', 'N/A')}")
            logger.info(f"     预期收益: {suggestion.get('expected_improvement', 'N/A')}")

        logger.info("性能优化功能演示完成")

    def demonstrate_alert_management(self):
        """演示告警管理功能"""
        logger.info("开始演示告警管理功能...")

        # 1. 获取告警规则摘要
        logger.info("1. 获取告警规则摘要...")
        rules_summary = self.alert_rule_engine.get_rule_statistics()
        logger.info(f"   总规则数: {rules_summary['total_rules']}")
        logger.info(f"   启用规则数: {rules_summary['enabled_rules']}")
        logger.info(f"   活跃告警数: {rules_summary['active_alerts']}")

        # 2. 获取活跃告警
        logger.info("2. 获取活跃告警...")
        active_alerts = self.alert_rule_engine.get_active_alerts()
        logger.info(f"   当前活跃告警: {len(active_alerts)} 个")

        for alert in active_alerts[:3]:  # 显示前3个
            logger.info(f"   - {alert.rule_name}: {alert.severity.value}")

        # 3. 获取告警摘要
        logger.info("3. 获取告警摘要...")
        alert_summary = self.health_checker.get_alert_summary()
        logger.info(f"   按严重程度分类:")
        for severity, count in alert_summary['alerts_by_severity'].items():
            logger.info(f"     {severity}: {count} 个")

        logger.info("告警管理功能演示完成")

    def demonstrate_cache_management(self):
        """演示缓存管理功能"""
        logger.info("开始演示缓存管理功能...")

        # 1. 获取缓存统计
        logger.info("1. 获取缓存统计...")
        cache_stats = self.health_checker.get_cache_stats()
        logger.info(f"   缓存条目数: {cache_stats['total_entries']}")
        logger.info(f"   缓存命中率: {cache_stats['hit_rate']:.2%}")
        logger.info(f"   缓存驱逐次数: {cache_stats['evictions']}")

        # 2. 演示智能缓存
        logger.info("2. 演示智能缓存...")
        cache_key = "demo_health_data"

        # 设置缓存
        self.health_checker.cache_manager.set(
            cache_key,
            {"status": "healthy", "timestamp": datetime.now().isoformat()},
            ttl=60
        )
        logger.info(f"   已设置缓存: {cache_key}")

        # 获取缓存
        cached_data = self.health_checker.cache_manager.get(cache_key)
        if cached_data:
            logger.info(f"   缓存命中: {cached_data['status']}")
        else:
            logger.info("   缓存未命中")

        # 3. 演示预加载
        logger.info("3. 演示缓存预加载...")
        preload_keys = ["health_status", "system_metrics"]
        self.health_checker.cache_manager.set_preload_keys(preload_keys)
        logger.info(f"   已设置预加载键: {preload_keys}")

        logger.info("缓存管理功能演示完成")

    def demonstrate_monitoring_integration(self):
        """演示监控集成功能"""
        logger.info("开始演示监控集成功能...")

        # 1. 获取Prometheus指标摘要
        logger.info("1. 获取Prometheus指标摘要...")
        metrics_summary = self.health_checker.get_metrics_summary()
        logger.info(f"   总指标数: {metrics_summary.get('total_metrics', 0)}")
        logger.info(f"   指标类型: {list(metrics_summary.get('metrics_by_type', {}).keys())}")

        # 2. 记录自定义指标
        logger.info("2. 记录自定义指标...")
        self.health_checker.prometheus_exporter.record_health_check(
            service="demo_service",
            check_type="custom",
            status="healthy",
            response_time=0.25
        )
        logger.info("   已记录自定义健康检查指标")

        # 3. 记录系统指标
        logger.info("3. 记录系统指标...")
        self.health_checker.prometheus_exporter.record_system_metrics(
            host="demo-host",
            cpu_percent=45.2,
            memory_bytes=8589934592,
            disk_usage={"root": 0.65}
        )
        logger.info("   已记录系统指标")

        logger.info("监控集成功能演示完成")

    async def demonstrate_grafana_integration(self):
        """演示Grafana集成功能"""
        if not self.grafana_integration:
            logger.info("Grafana集成未启用，跳过演示")
            return

        logger.info("开始演示Grafana集成功能...")

        try:
            # 1. 部署监控仪表板
            logger.info("1. 部署监控仪表板...")
            deployment_results = self.grafana_integration.deploy_all_dashboards()

            for dashboard_name, result in deployment_results.items():
                status = "成功" if result.get('success') else "失败"
                logger.info(f"   仪表板 {dashboard_name}: {status}")

            # 2. 导出仪表板配置
            logger.info("2. 导出仪表板配置...")
            config_data = self.grafana_integration.export_dashboard_config()
            logger.info(f"   已导出 {len(config_data)} 个仪表板配置")

            # 3. 获取仪表板列表
            logger.info("3. 获取仪表板列表...")
            dashboards = self.grafana_integration.list_dashboards()
            logger.info(f"   可用仪表板: {len(dashboards)} 个")

            for dashboard in dashboards[:3]:  # 显示前3个
                logger.info(f"   - {dashboard.get('title', 'N/A')}")

        except Exception as e:
            logger.error(f"Grafana集成演示失败: {e}")

        logger.info("Grafana集成功能演示完成")

    async def run_comprehensive_demo(self):
        """运行综合演示"""
        logger.info("=" * 60)
        logger.info("RQA2025 健康检查模块集成演示")
        logger.info("=" * 60)

        try:
            # 1. 设置组件
            await self.setup_components()

            # 2. 设置告警规则
            self.setup_alert_rules()

            # 3. 运行各功能演示
            await self.demonstrate_health_checks()
            self.demonstrate_performance_optimization()
            self.demonstrate_alert_management()
            self.demonstrate_cache_management()
            self.demonstrate_monitoring_integration()
            await self.demonstrate_grafana_integration()

            # 4. 运行告警规则评估
            logger.info("运行告警规则评估...")
            triggered_alerts = self.alert_rule_engine.evaluate_rules()
            logger.info(f"   触发的告警: {len(triggered_alerts)} 个")

            # 5. 生成综合报告
            logger.info("生成综合报告...")
            await self.generate_comprehensive_report()

            logger.info("=" * 60)
            logger.info("演示完成！")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"演示过程中发生错误: {e}")
            raise

    async def generate_comprehensive_report(self):
        """生成综合报告"""
        logger.info("正在生成综合报告...")

        report = {
            "timestamp": datetime.now().isoformat(),
            "demo_version": "2.1.0",
            "components_status": {},
            "performance_metrics": {},
            "alert_summary": {},
            "recommendations": []
        }

        # 组件状态
        report["components_status"] = {
            "health_checker": "active",
            "performance_optimizer": "active",
            "alert_rule_engine": "active",
            "grafana_integration": "active" if self.grafana_integration else "disabled"
        }

        # 性能指标
        try:
            performance_report = self.performance_optimizer.get_performance_report()
            report["performance_metrics"] = {
                "total_metrics": len(performance_report.get('metrics_summary', {})),
                "optimization_suggestions": len(performance_report.get('suggestions', []))
            }
        except Exception as e:
            report["performance_metrics"]["error"] = str(e)

        # 告警摘要
        try:
            alert_summary = self.health_checker.get_alert_summary()
            report["alert_summary"] = {
                "active_alerts": alert_summary.get('active_alerts_count', 0),
                "total_rules": len(self.alert_rule_engine.rules)
            }
        except Exception as e:
            report["alert_summary"]["error"] = str(e)

        # 建议
        try:
            suggestions = self.performance_optimizer.analyze_performance()
            report["recommendations"] = [
                {
                    "description": s.get('description', ''),
                    "priority": s.get('priority', ''),
                    "expected_improvement": s.get('expected_improvement', '')
                }
                for s in suggestions[:5]  # 前5个建议
            ]
        except Exception as e:
            report["recommendations"] = [{"error": str(e)}]

        # 输出报告摘要
        logger.info("综合报告摘要:")
        logger.info(f"   组件状态: {len(report['components_status'])} 个组件")
        logger.info(f"   性能指标: {report['performance_metrics'].get('total_metrics', 0)} 个")
        logger.info(f"   活跃告警: {report['alert_summary'].get('active_alerts', 0)} 个")
        logger.info(f"   优化建议: {len(report['recommendations'])} 条")

        return report


async def main():
    """主函数"""
    demo = HealthCheckIntegrationDemo()
    await demo.run_comprehensive_demo()


if __name__ == "__main__":
    asyncio.run(main())
