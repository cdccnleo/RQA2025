#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统性修复所有基础设施层测试文件的语法错误
"""

import os
import re
import ast
from pathlib import Path
from typing import List, Dict, Tuple


class TestFileFixer:
    """测试文件修复器"""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.fixed_issues = []

    def fix_file(self) -> bool:
        """修复单个文件"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(self.file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
            except UnicodeDecodeError:
                print(f"无法读取文件: {self.file_path}")
                return False

        original_content = content

        # 应用各种修复
        content = self.fix_indentation(content)
        content = self.fix_docstring_indentation(content)
        content = self.fix_decorator_placement(content)
        content = self.fix_missing_imports(content)
        content = self.remove_duplicate_decorators(content)

        # 检查是否有所改变
        if content != original_content:
            try:
                with open(self.file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"修复了文件: {self.file_path.name}")
                return True
            except Exception as e:
                print(f"写入文件失败: {self.file_path} - {e}")
                return False
        else:
            print(f"无需修复: {self.file_path.name}")
            return False

    def fix_indentation(self, content: str) -> str:
        """修复缩进问题"""
        lines = content.split('\n')
        result = []
        in_class = False
        class_indent = 0

        for i, line in enumerate(lines):
            stripped = line.lstrip()

            # 检测类定义
            if stripped.startswith('class ') and 'Test' in stripped:
                in_class = True
                class_indent = len(line) - len(stripped)
                result.append(line)
            elif in_class and stripped and not stripped.startswith('#'):
                # 类内的非空行且不是注释
                current_indent = len(line) - len(stripped)

                # 检查是否需要修复缩进
                if stripped.startswith('def ') and current_indent != class_indent + 4:
                    result.append('    ' + stripped)
                    self.fixed_issues.append(f"修复方法缩进: {stripped[:30]}...")
                elif current_indent < class_indent + 4 and not stripped.startswith('def ') and not stripped.startswith('"""'):
                    result.append('    ' + stripped)
                    self.fixed_issues.append(f"修复类内缩进: {stripped[:30]}...")
                else:
                    result.append(line)
            else:
                result.append(line)

        return '\n'.join(result)

    def fix_docstring_indentation(self, content: str) -> str:
        """修复文档字符串缩进"""
        lines = content.split('\n')
        result = []
        in_docstring = False
        docstring_start = 0

        for i, line in enumerate(lines):
            stripped = line.lstrip()

            if stripped.startswith('"""') and not in_docstring:
                # 文档字符串开始
                in_docstring = True
                docstring_start = len(line) - len(stripped)

                # 检查是否在方法内部
                method_context = False
                for j in range(max(0, i-5), i):
                    if lines[j].strip().startswith('def '):
                        method_context = True
                        break

                if method_context and docstring_start < 8:  # 方法内文档字符串应该有8个空格
                    result.append('        ' + stripped)
                    self.fixed_issues.append("修复方法文档字符串缩进")
                else:
                    result.append(line)

            elif stripped.startswith('"""') and in_docstring:
                # 文档字符串结束
                in_docstring = False
                if method_context and len(line) - len(stripped) < 8:
                    result.append('        ' + stripped)
                else:
                    result.append(line)
            elif in_docstring:
                # 文档字符串内容
                if method_context and len(line) - len(stripped) < 8:
                    result.append('        ' + stripped)
                else:
                    result.append(line)
            else:
                result.append(line)

        return '\n'.join(result)

    def fix_decorator_placement(self, content: str) -> str:
        """修复装饰器位置"""
        lines = content.split('\n')
        result = []

        for i, line in enumerate(lines):
            if line.strip().startswith('@pytest.mark.timeout('):
                # 找到装饰器
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line.startswith('class ') and 'Test' in next_line:
                        # 装饰器在类定义之前，这是正确的
                        result.append(line)
                    else:
                        # 装饰器位置错误，移除它
                        self.fixed_issues.append("移除错误位置的装饰器")
                        continue
                else:
                    result.append(line)
            else:
                result.append(line)

        return '\n'.join(result)

    def fix_missing_imports(self, content: str) -> str:
        """修复缺失的导入"""
        # 检查是否需要添加pytest导入
        if '@pytest.mark' in content and 'import pytest' not in content:
            lines = content.split('\n')
            # 在文件开头添加导入
            if lines and not lines[0].startswith('#'):
                lines.insert(0, 'import pytest')
                self.fixed_issues.append("添加缺失的pytest导入")
            elif lines and lines[0].startswith('#'):
                # 在注释后添加导入
                for i, line in enumerate(lines):
                    if not line.startswith('#') and line.strip():
                        lines.insert(i, 'import pytest')
                        self.fixed_issues.append("添加缺失的pytest导入")
                        break

            return '\n'.join(lines)

        return content

    def remove_duplicate_decorators(self, content: str) -> str:
        """移除重复的装饰器"""
        lines = content.split('\n')
        result = []
        seen_decorators = set()

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('@pytest.mark.timeout('):
                if stripped in seen_decorators:
                    self.fixed_issues.append("移除重复的装饰器")
                    continue
                else:
                    seen_decorators.add(stripped)

            result.append(line)

        return '\n'.join(result)


def validate_file_syntax(file_path: str) -> bool:
    """验证文件语法"""
    import subprocess
    import sys

    try:
        result = subprocess.run([
            sys.executable, '-m', 'py_compile', str(file_path)
        ], capture_output=True, timeout=10)

        return result.returncode == 0
    except:
        return False


def process_all_files(root_path: str, error_files: List[str]) -> Dict[str, List[str]]:
    """处理所有错误文件"""
    results = {
        'fixed': [],
        'failed': [],
        'skipped': []
    }

    root_dir = Path(root_path)

    for error_file in error_files:
        # 构建完整的文件路径
        file_path = root_dir / error_file.replace('ERROR tests/unit/infrastructure/', '')

        if not file_path.exists():
            print(f"文件不存在: {file_path}")
            results['failed'].append(error_file)
            continue

        print(f"\n处理文件: {file_path.name}")

        fixer = TestFileFixer(str(file_path))

        if fixer.fix_file():
            # 验证修复结果
            if validate_file_syntax(str(file_path)):
                results['fixed'].append(error_file)
                print(f"✅ 成功修复: {file_path.name}")
                if fixer.fixed_issues:
                    print(f"   修复的问题: {', '.join(fixer.fixed_issues[:3])}")
            else:
                results['failed'].append(error_file)
                print(f"❌ 修复失败: {file_path.name}")
        else:
            results['skipped'].append(error_file)
            print(f"⏭️ 无需修复: {file_path.name}")

    return results


def main():
    """主函数"""
    # 从命令行参数或用户输入获取错误文件列表
    error_files = [
        "base/test_additional_infrastructure.py",
        "base/test_base.py",
        "base/test_base_infrastructure.py",
        "base/test_comprehensive_infrastructure.py",
        "cache/test_advanced_cache_manager.py",
        "cache/test_cache_advanced_cache_manager.py",
        "cache/test_cache_client_sdk.py",
        "cache/test_cache_client_sdk_simple.py",
        "cache/test_cache_core_components.py",
        "cache/test_cache_dependency.py",
        "cache/test_cache_exceptions.py",
        "cache/test_cache_global_interfaces.py",
        "cache/test_cache_multi_level_cache.py",
        "cache/test_cache_optimized_cache_service.py",
        "cache/test_cache_optimizer.py",
        "cache/test_cache_performance_config.py",
        "cache/test_cache_redis_cache.py",
        "cache/test_cache_redis_storage.py",
        "cache/test_cache_simple_memory_cache.py",
        "cache/test_cache_strategy.py",
        "cache/test_cache_system.py",
        "cache/test_cache_system_deep_coverage.py",
        "cache/test_cache_system_enhanced.py",
        "cache/test_cache_system_simple.py",
        "cache/test_cache_system_zero_coverage.py",
        "cache/test_cache_utils.py",
        "cache/test_cache_utils_prediction_cache.py",
        "cache/test_comprehensive_cache_system.py",
        "cache/test_distributed_cache.py",
        "cache/test_lru_cache.py",
        "cache/test_multi_level_cache.py",
        "cache/test_smart_cache_strategy.py",
        "cache/test_unified_cache.py",
        "config/test_config_base.py",
        "config/test_config_core_components.py",
        "config/test_config_environment.py",
        "config/test_config_factory.py",
        "config/test_config_registry.py",
        "config/test_config_simple.py",
        "config/test_config_system.py",
        "config/test_config_system_deep_coverage.py",
        "config/test_config_system_enhanced.py",
        "config/test_configuration.py",
        "config/test_unified_config_manager.py",
        "config/test_unified_config_service.py",
        "distributed/test_concurrency_controller.py",
        "distributed/test_parallel_loader.py",
        "error/test_boundary_conditions.py",
        "error/test_circuit_breaker.py",
        "error/test_error_core_components.py",
        "error/test_error_handling.py",
        "error/test_exception_handling.py",
        "error/test_health_exceptions.py",
        "error/test_retry_handler.py",
        "error/test_unified_error_handler.py",
        "health/test_database_health_monitor.py",
        "health/test_enhanced_health_checker.py",
        "health/test_health_base.py",
        "health/test_health_check_core.py",
        "health/test_health_checker.py",
        "health/test_health_core_components.py",
        "health/test_health_data_api.py",
        "health/test_health_interfaces.py",
        "logging/test_log_aggregator_plugin.py",
        "logging/test_log_correlation_plugin.py",
        "logging/test_log_level_optimizer.py",
        "logging/test_log_metrics_plugin.py",
        "logging/test_log_sampler.py",
        "logging/test_log_sampler_plugin.py",
        "logging/test_logger.py",
        "logging/test_logger_components.py",
        "logging/test_logging_advanced_features.py",
        "logging/test_logging_base.py",
        "logging/test_logging_core_components.py",
        "logging/test_logging_engine.py",
        "logging/test_logging_interfaces.py",
        "logging/test_logging_service_components.py",
        "logging/test_logging_strategy.py",
        "logging/test_logging_system.py",
        "logging/test_logging_system_comprehensive.py",
        "logging/test_logging_system_deep_performance.py",
        "logging/test_logging_utils.py",
        "logging/test_smart_log_filter.py",
        "logging/test_unified_logger.py",
        "monitoring/test_alert_manager_integration.py",
        "monitoring/test_alert_rule_engine.py",
        "monitoring/test_application_monitor.py",
        "monitoring/test_continuous_monitoring_system.py",
        "monitoring/test_infra_processor.py",
        "monitoring/test_monitoring_alert_system_comprehensive.py",
        "monitoring/test_monitoring_processor.py",
        "monitoring/test_monitoring_system.py",
        "monitoring/test_monitoring_system_deep_coverage.py",
        "monitoring/test_performance_benchmark.py",
        "monitoring/test_performance_framework.py",
        "monitoring/test_prometheus_exporter.py",
        "monitoring/test_smart_performance_monitor.py",
        "monitoring/test_system_monitor.py",
        "monitoring/test_system_processor.py",
        "resource/test_connection_pool.py",
        "resource/test_database_adapters.py",
        "resource/test_pool_components.py",
        "resource/test_quota_components.py",
        "resource/test_resource_core_components.py",
        "resource/test_resource_manager.py",
        "resource/test_resource_monitoring.py",
        "service/test_micro_service.py",
        "service/test_microservice_manager.py",
        "service/test_service_launcher.py",
        "test_boundary_conditions.py",
        "test_cache_production.py",
        "test_cache_system.py",
        "test_config_encryption.py",
        "test_config_hot_reload.py",
        "test_config_manager.py",
        "test_config_production.py",
        "test_coverage_improvement.py",
        "test_database_production.py",
        "test_health_system.py",
        "test_infrastructure_priority.py",
        "test_logging_production.py",
        "test_logging_system.py",
        "test_monitoring_production.py",
        "test_redis_production.py",
        "utils/test_async_config.py",
        "utils/test_async_metrics.py",
        "utils/test_async_optimizer.py",
        "utils/test_dynamic_executor.py",
        "utils/test_integrity_checker.py",
        "utils/test_smart_cache_optimizer.py",
        "utils/test_utils.py"
    ]

    print("开始系统性修复基础设施层测试文件...")
    print(f"需要处理的文件数量: {len(error_files)}")

    # 处理所有文件
    results = process_all_files("tests/unit/infrastructure", error_files)

    # 输出结果
    print("
=== 修复结果汇总 == ="    print(f"✅ 成功修复: {len(results['fixed'])} 个文件")
    print(f"❌ 修复失败: {len(results['failed'])} 个文件")
    print(f"⏭️ 无需修复: {len(results['skipped'])} 个文件")

    success_rate = len(results['fixed']) / len(error_files) * 100
    print(".1f"
    # 显示前10个成功修复的文件
    if results['fixed']:
        print("
✅ 成功修复的文件(前10个): " for file_name in results['fixed'][:10]:
            print(f"  - {file_name}")

    # 显示失败的文件
    if results['failed']:
        print("
❌ 修复失败的文件: " for file_name in results['failed'][:10]:
            print(f"  - {file_name}")

    print("
建议: " if success_rate > 80:
        print("- 大部分文件已修复，可以重新运行测试收集")
    else:
        print("- 仍有一些文件需要手动检查和修复")
    print("- 建议运行: python -m pytest tests/unit/infrastructure/ --collect-only")


if __name__ == "__main__":
    main()
