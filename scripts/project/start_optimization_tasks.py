#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动优化任务脚本

用于开始执行第一优先级的优化任务，包括特征层、基础设施层和系统集成层的短期目标。
"""

import os
import subprocess
from datetime import datetime


class OptimizationTaskExecutor:
    """优化任务执行器"""

    def __init__(self):
        self.project_root = os.getcwd()
        self.log_file = "optimization_execution.log"

    def log_message(self, message: str):
        """记录日志消息"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + "\n")

    def run_command(self, command: str, description: str) -> bool:
        """运行命令并记录结果"""
        self.log_message(f"开始执行: {description}")
        self.log_message(f"命令: {command}")

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )

            if result.returncode == 0:
                self.log_message(f"✅ {description} 执行成功")
                if result.stdout:
                    self.log_message(f"输出: {result.stdout}")
                return True
            else:
                self.log_message(f"❌ {description} 执行失败")
                if result.stderr:
                    self.log_message(f"错误: {result.stderr}")
                return False

        except Exception as e:
            self.log_message(f"❌ {description} 执行异常: {e}")
            return False

    def start_features_layer_tasks(self):
        """启动特征层任务"""
        self.log_message("=" * 60)
        self.log_message("开始执行特征层优化任务")
        self.log_message("=" * 60)

        # 1. 运行特征层单元测试
        test_commands = [
            "python -m pytest tests/unit/features/ -v --tb=short",
            "python -m pytest tests/unit/features/test_feature_engineer.py -v",
            "python -m pytest tests/unit/features/test_feature_processor.py -v",
            "python -m pytest tests/unit/features/test_feature_selector.py -v",
            "python -m pytest tests/unit/features/test_feature_standardizer.py -v",
            "python -m pytest tests/unit/features/test_feature_saver.py -v"
        ]

        for i, command in enumerate(test_commands, 1):
            success = self.run_command(
                command,
                f"特征层单元测试 {i}/{len(test_commands)}"
            )
            if not success:
                self.log_message("⚠️ 特征层单元测试存在问题，需要修复")
                break

        # 2. 检查测试覆盖率
        coverage_command = "python -m pytest tests/unit/features/ --cov=src/features --cov-report=html --cov-report=term"
        self.run_command(coverage_command, "生成特征层测试覆盖率报告")

        # 3. 运行集成测试
        integration_commands = [
            "python -m pytest tests/integration/test_features_data_integration.py -v",
            "python -m pytest tests/integration/test_features_model_integration.py -v",
            "python -m pytest tests/integration/test_features_infrastructure_integration.py -v"
        ]

        for i, command in enumerate(integration_commands, 1):
            self.run_command(
                command,
                f"特征层集成测试 {i}/{len(integration_commands)}"
            )

    def start_infrastructure_layer_tasks(self):
        """启动基础设施层任务"""
        self.log_message("=" * 60)
        self.log_message("开始执行基础设施层优化任务")
        self.log_message("=" * 60)

        # 1. 检查监控配置
        config_commands = [
            "python -c \"from src.infrastructure import UnifiedConfigManager; print('配置管理器检查通过')\"",
            "python -c \"from src.infrastructure import ICacheManager; print('缓存管理器检查通过')\"",
            "python -c \"from src.infrastructure import IDatabaseManager; print('数据库管理器检查通过')\"",
            "python -c \"from src.infrastructure import IMonitorManager; print('监控管理器检查通过')\""
        ]

        for i, command in enumerate(config_commands, 1):
            self.run_command(
                command,
                f"基础设施层配置检查 {i}/{len(config_commands)}"
            )

        # 2. 运行基础设施层测试
        test_commands = [
            "python -m pytest tests/unit/infrastructure/ -v --tb=short",
            "python -m pytest tests/unit/infrastructure/test_config_manager.py -v",
            "python -m pytest tests/unit/infrastructure/test_cache_manager.py -v",
            "python -m pytest tests/unit/infrastructure/test_database_manager.py -v",
            "python -m pytest tests/unit/infrastructure/test_monitor_manager.py -v"
        ]

        for i, command in enumerate(test_commands, 1):
            success = self.run_command(
                command,
                f"基础设施层单元测试 {i}/{len(test_commands)}"
            )
            if not success:
                self.log_message("⚠️ 基础设施层单元测试存在问题，需要修复")
                break

        # 3. 检查监控指标
        monitor_command = "python -c \"from src.infrastructure import IMonitorManager; print('监控指标检查通过')\""
        self.run_command(monitor_command, "检查监控指标配置")

    def start_integration_layer_tasks(self):
        """启动系统集成层任务"""
        self.log_message("=" * 60)
        self.log_message("开始执行系统集成层优化任务")
        self.log_message("=" * 60)

        # 1. 检查集成组件
        integration_commands = [
            "python -c \"from src.integration import SystemIntegrationManager; print('系统集成管理器检查通过')\"",
            "python -c \"from src.integration import LayerInterface; print('层接口检查通过')\"",
            "python -c \"from src.integration import UnifiedConfigManager; print('统一配置管理器检查通过')\""
        ]

        for i, command in enumerate(integration_commands, 1):
            self.run_command(
                command,
                f"系统集成层组件检查 {i}/{len(integration_commands)}"
            )

        # 2. 运行集成层测试
        test_commands = [
            "python -m pytest tests/unit/integration/ -v --tb=short",
            "python -m pytest tests/unit/integration/test_system_integration_manager.py -v",
            "python -m pytest tests/unit/integration/test_layer_interface.py -v",
            "python -m pytest tests/integration/test_system_integration.py -v"
        ]

        for i, command in enumerate(test_commands, 1):
            success = self.run_command(
                command,
                f"系统集成层测试 {i}/{len(test_commands)}"
            )
            if not success:
                self.log_message("⚠️ 系统集成层测试存在问题，需要修复")
                break

    def generate_summary_report(self):
        """生成执行总结报告"""
        self.log_message("=" * 60)
        self.log_message("生成执行总结报告")
        self.log_message("=" * 60)

        # 读取日志文件
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r', encoding='utf-8') as f:
                log_content = f.read()

            # 统计成功和失败的任务
            success_count = log_content.count("✅")
            failure_count = log_content.count("❌")
            warning_count = log_content.count("⚠️")

            summary = f"""
# 优化任务执行总结报告

## 执行统计
- 成功任务: {success_count}
- 失败任务: {failure_count}
- 警告任务: {warning_count}

## 执行日志
```
{log_content}
```

## 下一步建议
1. 检查失败的任务，修复相关问题
2. 完善测试用例，提高测试覆盖率
3. 优化性能，确保系统稳定性
4. 更新文档，记录优化成果
"""

            with open("optimization_execution_summary.md", 'w', encoding='utf-8') as f:
                f.write(summary)

            self.log_message(f"总结报告已生成: optimization_execution_summary.md")

    def run_all_tasks(self):
        """运行所有优化任务"""
        self.log_message("开始执行各层优化任务")
        self.log_message(f"项目根目录: {self.project_root}")
        self.log_message(f"日志文件: {self.log_file}")

        # 清空日志文件
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"优化任务执行日志 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n")

        # 执行各层任务
        self.start_features_layer_tasks()
        self.start_infrastructure_layer_tasks()
        self.start_integration_layer_tasks()

        # 生成总结报告
        self.generate_summary_report()

        self.log_message("所有优化任务执行完成！")


def main():
    """主函数"""
    executor = OptimizationTaskExecutor()
    executor.run_all_tasks()


if __name__ == "__main__":
    main()
