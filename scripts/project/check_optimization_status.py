#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查优化状态脚本

用于检查各层的当前状态，识别可执行的任务和需要修复的问题。
"""

import os
import subprocess
from datetime import datetime


class OptimizationStatusChecker:
    """优化状态检查器"""

    def __init__(self):
        self.project_root = os.getcwd()
        self.status_report = []

    def log_status(self, message: str, status: str = "INFO"):
        """记录状态信息"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status_entry = f"[{timestamp}] [{status}] {message}"
        print(status_entry)
        self.status_report.append(status_entry)

    def check_module_import(self, module_path: str, description: str) -> bool:
        """检查模块导入"""
        try:
            command = f"python -c \"import {module_path}; print('{description}导入成功')\""
            result = subprocess.run(command, shell=True, capture_output=True, text=True)

            if result.returncode == 0:
                self.log_status(f"✅ {description} - 导入成功", "SUCCESS")
                return True
            else:
                self.log_status(f"❌ {description} - 导入失败: {result.stderr}", "ERROR")
                return False
        except Exception as e:
            self.log_status(f"❌ {description} - 检查异常: {e}", "ERROR")
            return False

    def check_test_files(self, test_path: str, description: str) -> bool:
        """检查测试文件"""
        if os.path.exists(test_path):
            self.log_status(f"✅ {description} - 测试文件存在: {test_path}", "SUCCESS")
            return True
        else:
            self.log_status(f"⚠️ {description} - 测试文件不存在: {test_path}", "WARNING")
            return False

    def check_features_layer(self):
        """检查特征层状态"""
        self.log_status("=" * 60)
        self.log_status("检查特征层状态")
        self.log_status("=" * 60)

        # 检查核心模块导入
        features_modules = [
            ("src.features", "特征层主模块"),
            ("src.features.feature_engineer", "特征工程器"),
            ("src.features.feature_processor", "特征处理器"),
            ("src.features.feature_selector", "特征选择器"),
            ("src.features.feature_standardizer", "特征标准化器"),
            ("src.features.feature_saver", "特征保存器"),
        ]

        features_import_success = 0
        for module_path, description in features_modules:
            if self.check_module_import(module_path, description):
                features_import_success += 1

        # 检查测试文件
        features_tests = [
            ("tests/unit/features/", "特征层单元测试目录"),
            ("tests/unit/features/test_feature_engineer.py", "特征工程器测试"),
            ("tests/unit/features/test_feature_processor.py", "特征处理器测试"),
            ("tests/unit/features/test_feature_selector.py", "特征选择器测试"),
            ("tests/unit/features/test_feature_standardizer.py", "特征标准化器测试"),
            ("tests/unit/features/test_feature_saver.py", "特征保存器测试"),
        ]

        features_test_success = 0
        for test_path, description in features_tests:
            if self.check_test_files(test_path, description):
                features_test_success += 1

        self.log_status(
            f"特征层检查完成: 模块导入 {features_import_success}/{len(features_modules)}, 测试文件 {features_test_success}/{len(features_tests)}")
        return features_import_success == len(features_modules) and features_test_success == len(features_tests)

    def check_infrastructure_layer(self):
        """检查基础设施层状态"""
        self.log_status("=" * 60)
        self.log_status("检查基础设施层状态")
        self.log_status("=" * 60)

        # 检查核心模块导入
        infrastructure_modules = [
            ("src.infrastructure", "基础设施层主模块"),
            ("src.infrastructure.config", "配置管理模块"),
            ("src.infrastructure.cache", "缓存管理模块"),
            ("src.infrastructure.database", "数据库管理模块"),
            ("src.infrastructure.monitor", "监控管理模块"),
        ]

        infrastructure_import_success = 0
        for module_path, description in infrastructure_modules:
            if self.check_module_import(module_path, description):
                infrastructure_import_success += 1

        # 检查测试文件
        infrastructure_tests = [
            ("tests/unit/infrastructure/", "基础设施层单元测试目录"),
            ("tests/unit/infrastructure/test_config_manager.py", "配置管理器测试"),
            ("tests/unit/infrastructure/test_cache_manager.py", "缓存管理器测试"),
            ("tests/unit/infrastructure/test_database_manager.py", "数据库管理器测试"),
            ("tests/unit/infrastructure/test_monitor_manager.py", "监控管理器测试"),
        ]

        infrastructure_test_success = 0
        for test_path, description in infrastructure_tests:
            if self.check_test_files(test_path, description):
                infrastructure_test_success += 1

        self.log_status(
            f"基础设施层检查完成: 模块导入 {infrastructure_import_success}/{len(infrastructure_modules)}, 测试文件 {infrastructure_test_success}/{len(infrastructure_tests)}")
        return infrastructure_import_success == len(infrastructure_modules) and infrastructure_test_success == len(infrastructure_tests)

    def check_integration_layer(self):
        """检查系统集成层状态"""
        self.log_status("=" * 60)
        self.log_status("检查系统集成层状态")
        self.log_status("=" * 60)

        # 检查核心模块导入
        integration_modules = [
            ("src.integration", "系统集成层主模块"),
            ("src.integration.system_integration_manager", "系统集成管理器"),
            ("src.integration.layer_interface", "层接口"),
            ("src.integration.unified_config_manager", "统一配置管理器"),
        ]

        integration_import_success = 0
        for module_path, description in integration_modules:
            if self.check_module_import(module_path, description):
                integration_import_success += 1

        # 检查测试文件
        integration_tests = [
            ("tests/unit/integration/", "系统集成层单元测试目录"),
            ("tests/unit/integration/test_system_integration_manager.py", "系统集成管理器测试"),
            ("tests/unit/integration/test_layer_interface.py", "层接口测试"),
            ("tests/integration/test_system_integration.py", "系统集成测试"),
        ]

        integration_test_success = 0
        for test_path, description in integration_tests:
            if self.check_test_files(test_path, description):
                integration_test_success += 1

        self.log_status(
            f"系统集成层检查完成: 模块导入 {integration_import_success}/{len(integration_modules)}, 测试文件 {integration_test_success}/{len(integration_tests)}")
        return integration_import_success == len(integration_modules) and integration_test_success == len(integration_tests)

    def generate_recommendations(self):
        """生成优化建议"""
        self.log_status("=" * 60)
        self.log_status("生成优化建议")
        self.log_status("=" * 60)

        recommendations = []

        # 检查各层状态并生成建议
        features_ready = self.check_features_layer()
        infrastructure_ready = self.check_infrastructure_layer()
        integration_ready = self.check_integration_layer()

        if features_ready:
            recommendations.append("✅ 特征层已准备就绪，可以开始执行优化任务")
        else:
            recommendations.append("⚠️ 特征层需要修复模块导入或测试文件问题")

        if infrastructure_ready:
            recommendations.append("✅ 基础设施层已准备就绪，可以开始执行优化任务")
        else:
            recommendations.append("⚠️ 基础设施层需要修复模块导入或测试文件问题")

        if integration_ready:
            recommendations.append("✅ 系统集成层已准备就绪，可以开始执行优化任务")
        else:
            recommendations.append("⚠️ 系统集成层需要修复模块导入或测试文件问题")

        # 生成具体建议
        recommendations.extend([
            "",
            "## 下一步行动建议",
            "",
            "### 第一优先级（立即执行）",
            "1. 修复模块导入错误",
            "2. 创建缺失的测试文件",
            "3. 运行基础单元测试验证功能",
            "",
            "### 第二优先级（1周内）",
            "1. 完善测试覆盖率",
            "2. 运行集成测试",
            "3. 性能测试和优化",
            "",
            "### 第三优先级（2周内）",
            "1. 监控指标完善",
            "2. 文档更新",
            "3. 部署验证"
        ])

        for recommendation in recommendations:
            self.log_status(recommendation)

        return features_ready and infrastructure_ready and integration_ready

    def save_status_report(self):
        """保存状态报告"""
        report_file = "optimization_status_report.md"
        report_content = f"""# 优化状态检查报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 检查结果

```
{chr(10).join(self.status_report)}
```

## 总结

本报告显示了各层的当前状态，包括模块导入情况和测试文件完整性。
根据检查结果，可以确定哪些层已经准备就绪，哪些层需要进一步修复。

"""

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        self.log_status(f"状态报告已保存: {report_file}")

    def run_check(self):
        """运行完整检查"""
        self.log_status("开始优化状态检查")
        self.log_status(f"项目根目录: {self.project_root}")

        # 清空状态报告
        self.status_report = []

        # 执行各层检查
        all_ready = self.generate_recommendations()

        # 保存报告
        self.save_status_report()

        if all_ready:
            self.log_status("🎉 所有层都已准备就绪，可以开始执行优化任务！")
        else:
            self.log_status("⚠️ 部分层需要修复问题后才能开始优化任务")

        return all_ready


def main():
    """主函数"""
    checker = OptimizationStatusChecker()
    checker.run_check()


if __name__ == "__main__":
    main()
