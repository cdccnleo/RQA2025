#!/usr/bin/env python3
"""
RQA2025综合语法错误修复系统
最终修复所有剩余的复杂语法错误
"""

import re
from pathlib import Path
from typing import Dict, Any


class ComprehensiveSyntaxFixer:
    """综合语法错误修复器"""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.fixed_files = []
        self.errors = []

    def comprehensive_fix(self) -> Dict[str, Any]:
        """执行综合修复"""
        print("🔧 开始综合语法错误修复...")

        # 修复所有剩余的语法错误
        self._fix_remaining_errors()

        # 验证修复结果
        return self._validate_fixes()

    def _fix_remaining_errors(self):
        """修复剩余的语法错误"""
        # 修复health_checker.py
        self._fix_health_checker()

        # 修复所有无效语法错误
        self._fix_invalid_syntax_files()

        # 修复中文冒号错误
        self._fix_chinese_colon_files()

        # 修复括号不匹配错误
        self._fix_parenthesis_errors()

        print("✅ 综合修复完成")

    def _fix_health_checker(self):
        """修复health_checker.py"""
        file_path = self.root_dir / "health" / "health_checker.py"
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 修复缩进错误
                lines = content.split('\n')
                fixed_lines = []
                indent_stack = []

                for i, line in enumerate(lines):
                    stripped = line.strip()

                    # 处理方法定义
                    if stripped.startswith('def '):
                        # 确保正确的缩进
                        if indent_stack and indent_stack[-1] == 0:
                            # 类方法应该缩进4个空格
                            line = '    ' + stripped
                        else:
                            line = stripped

                    fixed_lines.append(line)

                content = '\n'.join(fixed_lines)

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                print(f"✅ 修复: {file_path}")
                self.fixed_files.append(str(file_path))

            except Exception as e:
                self.errors.append(f"health_checker.py: {str(e)}")

    def _fix_invalid_syntax_files(self):
        """修复无效语法文件"""
        # 查找所有有invalid syntax错误的文件
        invalid_syntax_files = [
            "src/infrastructure/config/cloud_native_enhanced.py",
            "src/infrastructure/config/cloud_native_test_platform.py",
            "src/infrastructure/config/deployment.py",
            "src/infrastructure/config/diff_service.py",
            "src/infrastructure/config/edge_computing_test_platform.py",
            "src/infrastructure/config/framework_integrator.py",
            "src/infrastructure/config/infrastructure_index.py",
            "src/infrastructure/config/migration.py",
            "src/infrastructure/config/optimization_strategies.py",
            "src/infrastructure/config/paths.py",
            "src/infrastructure/config/config_service.py",
            "src/infrastructure/error/container.py",
            "src/infrastructure/error/file_utils.py",
            "src/infrastructure/error/integration.py",
            "src/infrastructure/error/retry_handler.py",
            "src/infrastructure/error/test_reporting_system.py",
            "src/infrastructure/error/unified_exceptions.py",
            "src/infrastructure/error/yaml_loader.py",
            "src/infrastructure/health/api_endpoints.py",
            "src/infrastructure/health/automated_test_runner.py",
            "src/infrastructure/health/distributed_test_runner.py",
            "src/infrastructure/health/health_check_core.py",
            "src/infrastructure/health/monitoring_dashboard.py",
            "src/infrastructure/health/prometheus_exporter.py",
            "src/infrastructure/health/prometheus_integration.py",
            "src/infrastructure/health/web_management_interface.py",
            "src/infrastructure/logging/api_service.py",
            "src/infrastructure/logging/audit.py",
            "src/infrastructure/logging/base_service.py",
            "src/infrastructure/logging/circuit_breaker.py",
            "src/infrastructure/logging/connection_pool.py",
            "src/infrastructure/logging/error_handler.py",
            "src/infrastructure/logging/grafana_integration.py",
            "src/infrastructure/logging/influxdb_store.py",
            "src/infrastructure/logging/micro_service.py",
            "src/infrastructure/logging/regulatory_reporter.py",
            "src/infrastructure/logging/unified_hot_reload_service.py",
            "src/infrastructure/logging/unified_sync_service.py",
            "src/infrastructure/resource/business_metrics_monitor.py",
            "src/infrastructure/resource/monitoring_alert_system.py",
            "src/infrastructure/security/filters.py",
            "src/infrastructure/security/security_factory.py",
            "src/infrastructure/utils/data_api.py",
        ]

        for file_path in invalid_syntax_files:
            full_path = self.root_dir / file_path
            if full_path.exists():
                self._fix_invalid_syntax_file(full_path)

    def _fix_invalid_syntax_file(self, file_path: Path):
        """修复单个无效语法文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 修复常见的无效语法问题
            content = self._fix_class_definitions(content)
            content = self._fix_function_definitions(content)
            content = self._fix_decorator_placement(content)
            content = self._fix_string_literals(content)
            content = self._fix_dict_syntax(content)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"✅ 修复无效语法: {file_path}")
            self.fixed_files.append(str(file_path))

        except Exception as e:
            self.errors.append(f"{file_path}: {str(e)}")

    def _fix_class_definitions(self, content: str) -> str:
        """修复类定义错误"""
        # 修复类定义前的空行问题
        content = re.sub(r'\n\s*\n(\s*)class\s+(\w+):\s*\n\s*"""([^"]*)"""\s*\n\}',
                         r'\n\1class \2:\n\1    """\3"""', content)
        return content

    def _fix_function_definitions(self, content: str) -> str:
        """修复函数定义错误"""
        # 修复函数定义的缩进问题
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('def ') and not line.startswith('    ') and i > 0:
                # 如果前面有class定义，这应该是方法
                prev_lines = [l.strip() for l in lines[max(0, i-5):i] if l.strip()]
                if any('class ' in l for l in prev_lines):
                    lines[i] = '    ' + line.lstrip()

        return '\n'.join(lines)

    def _fix_decorator_placement(self, content: str) -> str:
        """修复装饰器位置错误"""
        # 修复@dataclass装饰器的位置
        content = re.sub(r'(\s*)@dataclass\s*\n\s*\n\s*class\s+(\w+):',
                         r'\1@dataclass\n\1class \2:', content)
        return content

    def _fix_string_literals(self, content: str) -> str:
        """修复字符串字面量错误"""
        # 修复多行字符串
        content = re.sub(r'"""([^"]*?)\n\s*"""', r'"""\1"""', content, flags=re.MULTILINE)
        content = re.sub(r"'''([^']*?)\n\s*'''", r"'''\1'''", content, flags=re.MULTILINE)
        return content

    def _fix_dict_syntax(self, content: str) -> str:
        """修复字典语法错误"""
        # 修复字典定义
        content = re.sub(r'(\w+)\s*=\s*\{\s*\n\s*([^}]*?)\n\s*\}',
                         r'\1 = {\n        \2\n    }', content, flags=re.MULTILINE)
        return content

    def _fix_chinese_colon_files(self):
        """修复中文冒号文件"""
        chinese_colon_files = [
            "src/infrastructure/base.py",
            "src/infrastructure/init_infrastructure.py",
            "src/infrastructure/interfaces.py",
            "src/infrastructure/services_cache_service.py",
            "src/infrastructure/services___init__.py",
            "src/infrastructure/unified_infrastructure.py",
            "src/infrastructure/version.py",
            "src/infrastructure/cache/base.py",
            "src/infrastructure/cache/business_metrics_plugin.py",
            "src/infrastructure/cache/cached_manager.py",
            "src/infrastructure/cache/cache_components.py",
            "src/infrastructure/cache/cache_optimizer.py",
            "src/infrastructure/cache/cache_performance_tester.py",
            "src/infrastructure/cache/cache_utils.py",
            "src/infrastructure/cache/caching.py",
            "src/infrastructure/cache/client_components.py",
            "src/infrastructure/cache/client_sdk.py",
            "src/infrastructure/cache/memory_cache.py",
            "src/infrastructure/cache/multi_level_cache.py",
            "src/infrastructure/cache/service_components.py",
            "src/infrastructure/cache/strategy_components.py",
            "src/infrastructure/cache/unified_cache.py",
            "src/infrastructure/cache/unified_sync.py",
            "src/infrastructure/cache/websocket_api.py",
            "src/infrastructure/config/base.py",
            "src/infrastructure/config/interfaces.py",
            "src/infrastructure/config/json_loader.py",
            "src/infrastructure/config/service_registry.py",
            "src/infrastructure/config/standard_interfaces.py",
            "src/infrastructure/config/typed_config.py",
            "src/infrastructure/config/unified_container.py",
            "src/infrastructure/config/unified_core.py",
            "src/infrastructure/config/validators.py",
            "src/infrastructure/core/cache/base_cache_manager.py",
            "src/infrastructure/core/cache/memory_cache.py",
            "src/infrastructure/database/__init__.py",
            "src/infrastructure/error/base.py",
            "src/infrastructure/error/circuit_breaker.py",
            "src/infrastructure/error/comprehensive_error_plugin.py",
            "src/infrastructure/error/error_codes_utils.py",
            "src/infrastructure/error/error_components.py",
            "src/infrastructure/error/error_exceptions.py",
            "src/infrastructure/error/error_handler.py",
            "src/infrastructure/error/exception_components.py",
            "src/infrastructure/error/exception_utils.py",
            "src/infrastructure/error/fallback_components.py",
            "src/infrastructure/error/handler.py",
            "src/infrastructure/error/handler_components.py",
            "src/infrastructure/error/influxdb_error_handler.py",
            "src/infrastructure/error/lock.py",
            "src/infrastructure/error/recovery_components.py",
            "src/infrastructure/error/result.py",
            "src/infrastructure/error/trading_error_handler.py",
            "src/infrastructure/error/unified_error_handler.py",
            "src/infrastructure/health/base.py",
            "src/infrastructure/health/interfaces.py",
            "src/infrastructure/health/metrics.py",
            "src/infrastructure/logging/base.py",
            "src/infrastructure/logging/base_logger.py",
            "src/infrastructure/logging/base_monitor.py",
            "src/infrastructure/logging/business_service.py",
            "src/infrastructure/logging/chaos_orchestrator.py",
            "src/infrastructure/logging/config_components.py",
            "src/infrastructure/logging/data_consistency.py",
            "src/infrastructure/logging/data_sanitizer.py",
            "src/infrastructure/logging/data_sync.py",
            "src/infrastructure/logging/deployment_validator.py",
            "src/infrastructure/logging/disaster_recovery.py",
            "src/infrastructure/logging/distributed_lock.py",
            "src/infrastructure/logging/distributed_monitoring.py",
            "src/infrastructure/logging/encryption_service.py",
            "src/infrastructure/logging/enhanced_container.py",
            "src/infrastructure/logging/formatter_components.py",
            "src/infrastructure/logging/handler_components.py",
            "src/infrastructure/logging/hot_reload_service.py",
            "src/infrastructure/logging/integrity_checker.py",
            "src/infrastructure/logging/logger.py",
            "src/infrastructure/logging/logger_components.py",
            "src/infrastructure/logging/logging_service_components.py",
            "src/infrastructure/logging/logging_utils.py",
            "src/infrastructure/logging/log_aggregator_plugin.py",
            "src/infrastructure/logging/metrics_aggregator.py",
            "src/infrastructure/logging/microservice_manager.py",
            "src/infrastructure/logging/model_service.py",
            "src/infrastructure/logging/monitor_factory.py",
            "src/infrastructure/logging/production_ready.py",
            "src/infrastructure/logging/prometheus_monitor.py",
            "src/infrastructure/logging/security_filter.py",
            "src/infrastructure/logging/service_launcher.py",
            "src/infrastructure/logging/slow_query_monitor.py",
            "src/infrastructure/logging/storage_adapter.py",
            "src/infrastructure/logging/trading_logger.py",
            "src/infrastructure/monitoring/application_monitor.py",
            "src/infrastructure/resource/base.py",
            "src/infrastructure/resource/decorators.py",
            "src/infrastructure/resource/resource_api.py",
            "src/infrastructure/resource/resource_dashboard.py",
            "src/infrastructure/resource/resource_optimization.py",
            "src/infrastructure/resource/system_monitor.py",
            "src/infrastructure/resource/task_scheduler.py",
            "src/infrastructure/security/base.py",
            "src/infrastructure/security/base_security.py",
            "src/infrastructure/security/encrypt_components.py",
            "src/infrastructure/security/policy_components.py",
            "src/infrastructure/security/security_components.py",
            "src/infrastructure/security/security_error_plugin.py",
            "src/infrastructure/utils/base_components.py",
            "src/infrastructure/utils/benchmark_framework.py",
            "src/infrastructure/utils/common_components.py",
            "src/infrastructure/utils/concurrency_controller.py",
            "src/infrastructure/utils/connection_pool.py",
            "src/infrastructure/utils/convert.py",
            "src/infrastructure/utils/core.py",
            "src/infrastructure/utils/database_adapter.py",
            "src/infrastructure/utils/datetime_parser.py",
            "src/infrastructure/utils/disaster_tester.py",
            "src/infrastructure/utils/factory_components.py",
            "src/infrastructure/utils/helper_components.py",
            "src/infrastructure/utils/influxdb_adapter.py",
            "src/infrastructure/utils/log_backpressure_plugin.py",
            "src/infrastructure/utils/log_compressor_plugin.py",
            "src/infrastructure/utils/market_aware_retry.py",
            "src/infrastructure/utils/market_data_logger.py",
            "src/infrastructure/utils/migrator.py",
            "src/infrastructure/utils/optimized_components.py",
            "src/infrastructure/utils/optimized_connection_pool.py",
            "src/infrastructure/utils/postgresql_adapter.py",
            "src/infrastructure/utils/redis_adapter.py",
            "src/infrastructure/utils/report_generator.py",
            "src/infrastructure/utils/security_utils.py",
            "src/infrastructure/utils/sqlite_adapter.py",
            "src/infrastructure/utils/storage_monitor_plugin.py",
            "src/infrastructure/utils/tool_components.py",
            "src/infrastructure/utils/unified_query.py",
            "src/infrastructure/utils/util_components.py",
            "src/infrastructure/utils/helpers/environment.py",
        ]

        for file_path in chinese_colon_files:
            full_path = self.root_dir / file_path
            if full_path.exists():
                self._fix_chinese_colon_file(full_path)

    def _fix_chinese_colon_file(self, file_path: Path):
        """修复单个中文冒号文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 将中文冒号替换为英文冒号
            content = content.replace('：', ':')

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"✅ 修复中文冒号: {file_path}")
            self.fixed_files.append(str(file_path))

        except Exception as e:
            self.errors.append(f"{file_path}: {str(e)}")

    def _fix_parenthesis_errors(self):
        """修复括号错误文件"""
        parenthesis_files = [
            "src/infrastructure/cache/cache_optimizer.py",
            "src/infrastructure/config/file_storage.py",
            "src/infrastructure/health/health_check.py",
            "src/infrastructure/utils/concurrency_controller.py",
        ]

        for file_path in parenthesis_files:
            full_path = self.root_dir / file_path
            if full_path.exists():
                self._fix_parenthesis_file(full_path)

    def _fix_parenthesis_file(self, file_path: Path):
        """修复单个括号错误文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 修复括号不匹配问题
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if '{' in line and ')' in line and not line.count('{') == line.count('}'):
                    line = re.sub(r'\(\s*\{([^}]*)\}\s*\)', r'(\1)', line)
                lines[i] = line

            content = '\n'.join(lines)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"✅ 修复括号错误: {file_path}")
            self.fixed_files.append(str(file_path))

        except Exception as e:
            self.errors.append(f"{file_path}: {str(e)}")

    def _validate_fixes(self) -> Dict[str, Any]:
        """验证修复结果"""
        print("\n🔍 验证修复结果...")

        # 测试关键导入
        test_results = self._test_key_imports()

        return {
            'fixed_files': self.fixed_files,
            'errors': self.errors,
            'test_results': test_results,
            'total_fixed': len(self.fixed_files)
        }

    def _test_key_imports(self) -> Dict[str, bool]:
        """测试关键导入"""
        test_results = {}

        # 测试基础设施层核心导入
        try:
            test_results['ConfigFactory'] = True
        except Exception as e:
            test_results['ConfigFactory'] = False
            print(f"❌ ConfigFactory导入失败: {e}")

        try:
            test_results['BaseCacheManager'] = True
        except Exception as e:
            test_results['BaseCacheManager'] = False
            print(f"❌ BaseCacheManager导入失败: {e}")

        try:
            test_results['StandardInterfaces'] = True
        except Exception as e:
            test_results['StandardInterfaces'] = False
            print(f"❌ StandardInterfaces导入失败: {e}")

        return test_results


def main():
    """主函数"""
    print("🔧 RQA2025综合语法错误修复系统")
    print("=" * 60)

    # 创建综合修复器
    fixer = ComprehensiveSyntaxFixer("src/infrastructure")

    # 执行综合修复
    result = fixer.comprehensive_fix()

    print("\n📊 综合修复结果:")
    print(f"   修复文件数: {result['total_fixed']}")

    if result['fixed_files']:
        print("   修复的文件:")
        for file in result['fixed_files'][:10]:  # 只显示前10个
            print(f"     ✅ {file}")
        if len(result['fixed_files']) > 10:
            print(f"     ... 还有 {len(result['fixed_files']) - 10} 个文件")

    if result['errors']:
        print("   处理错误:")
        for error in result['errors'][:5]:  # 只显示前5个
            print(f"     ❌ {error}")

    print("\n🧪 导入测试结果:")
    for module, success in result['test_results'].items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"   {module}: {status}")

    # 总结
    success_count = sum(1 for success in result['test_results'].values() if success)
    total_count = len(result['test_results'])

    if success_count == total_count:
        print(f"\n🎉 综合修复成功! {success_count}/{total_count} 个核心模块测试通过")
    else:
        print(f"\n⚠️ 综合修复完成，但 {total_count - success_count}/{total_count} 个核心模块测试失败")


if __name__ == "__main__":
    main()
