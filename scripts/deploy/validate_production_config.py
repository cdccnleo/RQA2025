#!/usr/bin/env python3
"""
生产环境配置验证脚本
验证配置管理系统在生产环境中的完整功能
"""

import os
import sys
import json
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List, Optional


class ProductionConfigValidator:
    """生产环境配置验证器"""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.results = {
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "errors": [],
            "warnings": []
        }

    def run_all_validations(self) -> Dict[str, Any]:
        """运行所有生产环境验证"""
        print("🚀 开始生产环境配置验证...")
        print("=" * 60)

        # 1. 验证配置文件存在性
        self._validate_config_files()

        # 2. 验证配置内容完整性
        self._validate_config_content()

        # 3. 验证配置安全性
        self._validate_security_settings()

        # 4. 验证配置性能
        self._validate_performance_settings()

        # 5. 验证配置热重载
        self._validate_hot_reload()

        # 6. 验证配置监控
        self._validate_monitoring_setup()

        # 输出结果
        self._print_results()

        return self.results

    def _validate_config_files(self):
        """验证配置文件存在性"""
        print("\n📁 验证配置文件存在性...")

        required_files = [
            "production.json",
            "development.json",
            "test.json"
        ]

        for config_file in required_files:
            file_path = self.config_dir / config_file
            if file_path.exists():
                self.results["tests_passed"] += 1
                print(f"  ✅ {config_file} - 存在")
            else:
                self.results["tests_failed"] += 1
                self.results["errors"].append(f"配置文件不存在: {config_file}")
                print(f"  ❌ {config_file} - 不存在")

        self.results["tests_run"] += len(required_files)

    def _validate_config_content(self):
        """验证配置内容完整性"""
        print("\n📋 验证配置内容完整性...")

        prod_config = self.config_dir / "production.json"
        if not prod_config.exists():
            return

        try:
            with open(prod_config, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # 检查必需的配置项
            required_sections = [
                "application",
                "database",
                "redis",
                "trading",
                "monitoring",
                "security"
            ]

            for section in required_sections:
                if section in config:
                    self.results["tests_passed"] += 1
                    print(f"  ✅ {section} 配置段 - 存在")

                    # 验证具体字段
                    self._validate_section_fields(section, config[section])
                else:
                    self.results["tests_failed"] += 1
                    self.results["errors"].append(f"缺少必需配置段: {section}")
                    print(f"  ❌ {section} 配置段 - 缺失")

            self.results["tests_run"] += len(required_sections)

        except json.JSONDecodeError as e:
            self.results["tests_failed"] += 1
            self.results["errors"].append(f"生产配置文件JSON格式错误: {e}")
            print(f"  ❌ 配置文件JSON格式错误: {e}")

    def _validate_section_fields(self, section: str, config: Dict[str, Any]):
        """验证配置段的字段"""
        field_validations = {
            "application": ["name", "version", "environment"],
            "database": ["host", "port", "name"],
            "redis": ["host", "port"],
            "trading": ["max_positions"],
            "monitoring": ["enabled"],
            "security": ["encryption", "authentication"]
        }

        if section in field_validations:
            required_fields = field_validations[section]
            for field in required_fields:
                if field in config:
                    print(f"    ✅ {section}.{field} - 存在")
                else:
                    self.results["warnings"].append(f"建议添加字段: {section}.{field}")
                    print(f"    ⚠️  {section}.{field} - 建议添加")

    def _validate_security_settings(self):
        """验证配置安全性"""
        print("\n🔒 验证配置安全性...")

        prod_config = self.config_dir / "production.json"
        if not prod_config.exists():
            return

        try:
            with open(prod_config, 'r', encoding='utf-8') as f:
                config = json.load(f)

            security_checks = [
                ("debug", config.get("application", {}).get("debug"), False, "value"),
                ("log_level", config.get("application", {}).get("log_level"), "INFO", "value"),
                ("encryption", "encryption" in config.get("security", {}), True, "bool"),
                ("audit", config.get("security", {}).get("audit", {}).get("enabled"), True, "bool")
            ]

            for check_name, actual_value, expected_value, check_type in security_checks:
                if check_name in ["debug", "log_level"]:
                    if actual_value == expected_value:
                        self.results["tests_passed"] += 1
                        print(f"  ✅ {check_name} 设置安全")
                    else:
                        self.results["tests_failed"] += 1
                        self.results["warnings"].append(f"{check_name} 设置可能不安全: {actual_value}")
                        print(f"  ❌ {check_name} 设置可能不安全: {actual_value}")
                else:
                    if actual_value:
                        self.results["tests_passed"] += 1
                        print(f"  ✅ {check_name} 已启用")
                    else:
                        self.results["tests_failed"] += 1
                        self.results["warnings"].append(f"{check_name} 未启用")
                        print(f"  ❌ {check_name} 未启用")

            self.results["tests_run"] += len(security_checks)

        except Exception as e:
            self.results["tests_failed"] += 1
            self.results["errors"].append(f"安全验证失败: {e}")
            print(f"  ❌ 安全验证失败: {e}")

    def _validate_performance_settings(self):
        """验证配置性能"""
        print("\n⚡ 验证配置性能...")

        prod_config = self.config_dir / "production.json"
        if not prod_config.exists():
            return

        try:
            with open(prod_config, 'r', encoding='utf-8') as f:
                config = json.load(f)

            performance_checks = [
                ("database.connection_pool.max_connections",
                 config.get("database", {}).get("connection_pool", {}).get("max_connections", 0), 10),
                ("redis.connection_pool.max_connections",
                 config.get("redis", {}).get("connection_pool", {}).get("max_connections", 0), 10),
                ("monitoring.metrics_interval",
                 config.get("monitoring", {}).get("metrics_interval", 300), 300)
            ]

            for check_name, actual_value, min_expected in performance_checks:
                if actual_value >= min_expected:
                    self.results["tests_passed"] += 1
                    print(f"  ✅ {check_name}: {actual_value} - 性能良好")
                else:
                    self.results["tests_failed"] += 1
                    self.results["warnings"].append(f"{check_name} 性能设置偏低: {actual_value}")
                    print(f"  ❌ {check_name}: {actual_value} - 性能设置偏低")

            self.results["tests_run"] += len(performance_checks)

        except Exception as e:
            self.results["tests_failed"] += 1
            self.results["errors"].append(f"性能验证失败: {e}")
            print(f"  ❌ 性能验证失败: {e}")

    def _validate_hot_reload(self):
        """验证配置热重载"""
        print("\n🔄 验证配置热重载...")

        # 创建临时配置文件进行测试
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_config = {
                "test": {"value": "original"},
                "hot_reload": {"enabled": True}
            }
            json.dump(test_config, f)
            temp_config = f.name

        try:
            # 模拟配置管理器
            from tests.integration.infrastructure.config.test_config_integration import ConfigManager

            manager = ConfigManager()

            # 加载初始配置
            assert manager.load_from_json(temp_config)
            assert manager.get("test.value") == "original"

            # 修改配置文件
            test_config["test"]["value"] = "updated"
            test_config["hot_reload"]["last_modified"] = time.time()

            with open(temp_config, 'w') as f:
                json.dump(test_config, f)

            # 重新加载配置
            assert manager.load_from_json(temp_config)
            assert manager.get("test.value") == "updated"

            self.results["tests_passed"] += 1
            self.results["tests_run"] += 1
            print("  ✅ 配置热重载功能正常")

        except Exception as e:
            self.results["tests_failed"] += 1
            self.results["errors"].append(f"热重载验证失败: {e}")
            print(f"  ❌ 热重载验证失败: {e}")
        finally:
            os.unlink(temp_config)

    def _validate_monitoring_setup(self):
        """验证配置监控设置"""
        print("\n📊 验证配置监控设置...")

        prod_config = self.config_dir / "production.json"
        if not prod_config.exists():
            return

        try:
            with open(prod_config, 'r', encoding='utf-8') as f:
                config = json.load(f)

            monitoring_config = config.get("monitoring", {})

            monitoring_checks = [
                ("enabled", monitoring_config.get("enabled", False)),
                ("metrics_interval", monitoring_config.get("metrics_interval", 0) > 0),
                ("health_check_interval", monitoring_config.get("health_check_interval", 0) > 0)
            ]

            for check_name, is_valid in monitoring_checks:
                if is_valid:
                    self.results["tests_passed"] += 1
                    print(f"  ✅ {check_name} 配置正确")
                else:
                    self.results["tests_failed"] += 1
                    self.results["warnings"].append(f"{check_name} 配置可能不完整")
                    print(f"  ❌ {check_name} 配置可能不完整")

            self.results["tests_run"] += len(monitoring_checks)

        except Exception as e:
            self.results["tests_failed"] += 1
            self.results["errors"].append(f"监控验证失败: {e}")
            print(f"  ❌ 监控验证失败: {e}")

    def _print_results(self):
        """输出验证结果"""
        print("\n" + "=" * 60)
        print("📋 生产环境配置验证结果")
        print("=" * 60)

        print(f"\n总测试数: {self.results['tests_run']}")
        print(f"通过测试: {self.results['tests_passed']}")
        print(f"失败测试: {self.results['tests_failed']}")

        success_rate = (self.results['tests_passed'] / self.results['tests_run'] * 100) if self.results['tests_run'] > 0 else 0
        print(f"成功率: {success_rate:.1f}%")

        if self.results['errors']:
            print(f"\n❌ 错误 ({len(self.results['errors'])}):")
            for error in self.results['errors']:
                print(f"  • {error}")

        if self.results['warnings']:
            print(f"\n⚠️  警告 ({len(self.results['warnings'])}):")
            for warning in self.results['warnings']:
                print(f"  • {warning}")

        # 最终评估
        if success_rate >= 90:
            print("\n🎉 生产环境配置验证通过！系统已准备好部署。")
        elif success_rate >= 75:
            print("\n⚠️  生产环境配置基本合格，建议修复警告项后再部署。")
        else:
            print("\n❌ 生产环境配置存在问题，必须修复后再部署。")


def main():
    """主函数"""
    # 设置配置目录
    config_dir = Path(__file__).parent.parent.parent / "config"

    # 运行验证
    validator = ProductionConfigValidator(str(config_dir))
    results = validator.run_all_validations()

    # 返回适当的退出码
    success_rate = (results['tests_passed'] / results['tests_run'] * 100) if results['tests_run'] > 0 else 0
    return 0 if success_rate >= 75 else 1


if __name__ == "__main__":
    sys.exit(main())
