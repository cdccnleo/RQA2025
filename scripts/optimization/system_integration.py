#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统集成脚本
将优化功能集成到主系统中
"""

import json
import time
import yaml
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class IntegrationConfig:
    """集成配置"""
    enable_cache_optimization: bool = True
    enable_monitoring_alert: bool = True
    enable_parameter_optimization: bool = True
    integration_mode: str = "production"  # development, testing, production
    auto_restart_enabled: bool = True
    backup_enabled: bool = True


class SystemIntegrator:
    """系统集成器"""

    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.integration_status = {}
        self.backup_files = []

    def backup_system(self) -> Dict[str, str]:
        """备份系统"""
        if not self.config.backup_enabled:
            return {}

        backup_dir = Path("backup/integration/")
        backup_dir.mkdir(parents=True, exist_ok=True)

        backup_files = {}

        # 备份配置文件
        config_files = [
            "config/risk_control_config.yaml",
            "config/deployment_config.json"
        ]

        for config_file in config_files:
            if Path(config_file).exists():
                backup_path = backup_dir / f"{Path(config_file).name}.backup"
                with open(config_file, 'r', encoding='utf-8') as src:
                    with open(backup_path, 'w', encoding='utf-8') as dst:
                        dst.write(src.read())
                backup_files[config_file] = str(backup_path)

        self.backup_files = backup_files
        return backup_files

    def integrate_cache_optimization(self) -> Dict[str, Any]:
        """集成缓存优化"""
        if not self.config.enable_cache_optimization:
            return {"status": "disabled"}

        try:
            # 导入缓存优化模块
            from scripts.optimization.cache_optimization import (
                CacheConfig, DataCacheManager, CacheOptimizer
            )

            # 创建缓存配置
            cache_config = CacheConfig(
                max_size=2000,
                ttl_seconds=7200,
                cleanup_interval=600,
                enable_persistence=True,
                compression_enabled=True,
                memory_limit_mb=200
            )

            # 创建缓存管理器
            cache_manager = DataCacheManager(cache_config)
            cache_manager.start()

            # 创建缓存优化器
            optimizer = CacheOptimizer(cache_manager)

            # 模拟优化
            usage_patterns = {
                "hit_rate": 0.75,
                "avg_request_rate": 1500,
                "memory_usage": 65
            }

            optimized_config = optimizer.optimize_cache_config(usage_patterns)

            # 停止缓存管理器
            cache_manager.stop()

            return {
                "status": "success",
                "original_config": asdict(cache_config),
                "optimized_config": asdict(optimized_config),
                "usage_patterns": usage_patterns
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def integrate_monitoring_alert(self) -> Dict[str, Any]:
        """集成监控告警"""
        if not self.config.enable_monitoring_alert:
            return {"status": "disabled"}

        try:
            # 导入监控告警模块
            from scripts.optimization.monitoring_alert_system import (
                AlertConfig, MonitoringSystem
            )

            # 创建告警配置
            alert_config = AlertConfig(
                cpu_threshold=75.0,
                memory_threshold=80.0,
                disk_threshold=85.0,
                response_time_threshold=800.0,
                error_rate_threshold=3.0,
                alert_cooldown=600,
                email_enabled=True,
                webhook_enabled=True
            )

            # 创建监控系统
            monitoring_system = MonitoringSystem(alert_config)

            # 启动监控
            monitoring_system.start()

            # 模拟运行
            time.sleep(3)

            # 获取状态
            status = monitoring_system.get_system_status()

            # 停止监控
            monitoring_system.stop()

            return {
                "status": "success",
                "config": asdict(alert_config),
                "system_status": status
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def integrate_parameter_optimization(self) -> Dict[str, Any]:
        """集成参数优化"""
        if not self.config.enable_parameter_optimization:
            return {"status": "disabled"}

        try:
            # 导入参数优化模块
            from scripts.optimization.parameter_optimization import (
                DynamicParameterManager
            )

            # 创建动态参数管理器
            manager = DynamicParameterManager()

            # 模拟市场数据
            market_data = {
                "timestamp": "2025-07-27 14:00:00",
                "volatility": 0.22,
                "trading_volume": {
                    "average_volume": 3000000
                },
                "market_conditions": {
                    "stress_index": 0.52
                }
            }

            # 更新参数
            optimized_parameters = manager.update_parameters(market_data)

            return {
                "status": "success",
                "market_data": market_data,
                "optimized_parameters": asdict(optimized_parameters)
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def update_main_config(self) -> Dict[str, Any]:
        """更新主配置文件"""
        try:
            # 读取现有配置
            config_path = Path("config/main_config.yaml")
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    main_config = yaml.safe_load(f)
            else:
                main_config = {}

            # 添加优化配置
            main_config["optimization"] = {
                "cache_enabled": self.config.enable_cache_optimization,
                "monitoring_enabled": self.config.enable_monitoring_alert,
                "parameter_optimization_enabled": self.config.enable_parameter_optimization,
                "integration_mode": self.config.integration_mode,
                "auto_restart": self.config.auto_restart_enabled
            }

            # 保存配置
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(main_config, f, default_flow_style=False, allow_unicode=True)

            return {
                "status": "success",
                "config_file": str(config_path),
                "config": main_config
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def run_integration(self) -> Dict[str, Any]:
        """运行系统集成"""
        print("🔧 开始系统集成...")

        # 备份系统
        backup_result = self.backup_system()
        self.integration_status["backup"] = backup_result

        # 集成缓存优化
        cache_result = self.integrate_cache_optimization()
        self.integration_status["cache_optimization"] = cache_result

        # 集成监控告警
        monitoring_result = self.integrate_monitoring_alert()
        self.integration_status["monitoring_alert"] = monitoring_result

        # 集成参数优化
        parameter_result = self.integrate_parameter_optimization()
        self.integration_status["parameter_optimization"] = parameter_result

        # 更新主配置
        config_result = self.update_main_config()
        self.integration_status["config_update"] = config_result

        return self.integration_status


class IntegrationValidator:
    """集成验证器"""

    def __init__(self):
        self.validation_results = {}

    def validate_integration(self, integration_status: Dict[str, Any]) -> Dict[str, Any]:
        """验证集成结果"""
        validation_results = {}

        # 验证备份
        backup_status = integration_status.get("backup", {})
        validation_results["backup"] = {
            "status": "success" if backup_status else "warning",
            "message": "备份完成" if backup_status else "备份未启用"
        }

        # 验证缓存优化
        cache_status = integration_status.get("cache_optimization", {})
        validation_results["cache_optimization"] = {
            "status": cache_status.get("status", "unknown"),
            "message": "缓存优化集成成功" if cache_status.get("status") == "success" else f"缓存优化集成失败: {cache_status.get('error', '未知错误')}"
        }

        # 验证监控告警
        monitoring_status = integration_status.get("monitoring_alert", {})
        validation_results["monitoring_alert"] = {
            "status": monitoring_status.get("status", "unknown"),
            "message": "监控告警集成成功" if monitoring_status.get("status") == "success" else f"监控告警集成失败: {monitoring_status.get('error', '未知错误')}"
        }

        # 验证参数优化
        parameter_status = integration_status.get("parameter_optimization", {})
        validation_results["parameter_optimization"] = {
            "status": parameter_status.get("status", "unknown"),
            "message": "参数优化集成成功" if parameter_status.get("status") == "success" else f"参数优化集成失败: {parameter_status.get('error', '未知错误')}"
        }

        # 验证配置更新
        config_status = integration_status.get("config_update", {})
        validation_results["config_update"] = {
            "status": config_status.get("status", "unknown"),
            "message": "配置更新成功" if config_status.get("status") == "success" else f"配置更新失败: {config_status.get('error', '未知错误')}"
        }

        # 计算总体状态
        success_count = sum(1 for result in validation_results.values()
                            if result["status"] == "success")
        total_count = len(validation_results)
        overall_status = "success" if success_count == total_count else "partial" if success_count > 0 else "failed"

        validation_results["overall"] = {
            "status": overall_status,
            "success_rate": f"{success_count}/{total_count}",
            "message": f"集成完成，成功率: {success_count}/{total_count}"
        }

        self.validation_results = validation_results
        return validation_results


def main():
    """主函数"""
    print("🔧 启动系统集成...")

    # 创建集成配置
    config = IntegrationConfig(
        enable_cache_optimization=True,
        enable_monitoring_alert=True,
        enable_parameter_optimization=True,
        integration_mode="production",
        auto_restart_enabled=True,
        backup_enabled=True
    )

    # 创建系统集成器
    integrator = SystemIntegrator(config)

    # 运行集成
    integration_status = integrator.run_integration()

    # 验证集成结果
    validator = IntegrationValidator()
    validation_results = validator.validate_integration(integration_status)

    print("✅ 系统集成完成!")

    # 打印结果
    print("\n" + "="*50)
    print("🎯 集成验证结果:")
    print("="*50)

    for component, result in validation_results.items():
        if component != "overall":
            status_icon = "✅" if result["status"] == "success" else "⚠️" if result["status"] == "warning" else "❌"
            print(f"{status_icon} {component}: {result['message']}")

    overall = validation_results["overall"]
    overall_icon = "✅" if overall["status"] == "success" else "⚠️" if overall["status"] == "partial" else "❌"
    print(f"\n{overall_icon} 总体状态: {overall['message']}")
    print("="*50)

    # 保存集成报告
    output_dir = Path("reports/optimization/")
    output_dir.mkdir(parents=True, exist_ok=True)

    integration_report = {
        "timestamp": time.time(),
        "config": asdict(config),
        "integration_status": integration_status,
        "validation_results": validation_results
    }

    report_file = output_dir / "system_integration_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(integration_report, f, ensure_ascii=False, indent=2)

    print(f"📄 集成报告已保存: {report_file}")


if __name__ == "__main__":
    main()
