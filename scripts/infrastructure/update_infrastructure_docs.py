#!/usr/bin/env python3
"""
基础设施层文档更新脚本

根据命名规范计划，更新所有基础设施层相关文档中的类名引用。
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

# 类名映射表 - 根据命名规范计划
CLASS_NAME_MAPPINGS = {
    # 配置管理类
    'EnhancedConfigManager': 'UnifiedConfigManager',
    'ConfigVersion': 'VersionManager',
    'DeploymentManager': 'DeploymentPlugin',
    'LegacyConfigVersionManager': 'LegacyVersionManager',
    'ConfigVersionStorage': 'VersionStorage',

    # 监控类
    'BusinessMetricsCollector': 'BusinessMetricsPlugin',
    'PerformanceOptimizer': 'PerformanceOptimizerPlugin',
    'BehaviorMonitor': 'BehaviorMonitorPlugin',
    'ModelMonitor': 'ModelMonitorPlugin',
    'BacktestMonitor': 'BacktestMonitorPlugin',
    'DisasterMonitor': 'DisasterMonitorPlugin',
    'StorageMonitor': 'StorageMonitorPlugin',

    # 日志类
    'EnhancedLogSampler': 'LogSamplerPlugin',
    'LogCorrelationQuery': 'LogCorrelationPlugin',
    'LogAggregator': 'LogAggregatorPlugin',
    'LogCompressor': 'LogCompressorPlugin',
    'LogMetrics': 'LogMetricsPlugin',
    'AdaptiveBackpressure': 'AdaptiveBackpressurePlugin',
    'BackpressureHandler': 'BackpressureHandlerPlugin',

    # 错误处理类
    'ComprehensiveErrorFramework': 'ComprehensiveErrorPlugin',
    'ErrorCodes': 'ErrorCodesUtils',
    'SecurityErrorHandler': 'SecurityErrorPlugin',
}

# 文件名映射表
FILE_NAME_MAPPINGS = {
    'enhanced_config_manager.py': 'unified_config_manager.py',
    'config_version.py': 'version_manager.py',
    'deployment_manager.py': 'deployment_plugin.py',
    'business_metrics_collector.py': 'business_metrics_plugin.py',
    'performance_optimizer.py': 'performance_optimizer_plugin.py',
    'behavior_monitor.py': 'behavior_monitor_plugin.py',
    'model_monitor.py': 'model_monitor_plugin.py',
    'backtest_monitor.py': 'backtest_monitor_plugin.py',
    'disaster_monitor.py': 'disaster_monitor_plugin.py',
    'storage_monitor.py': 'storage_monitor_plugin.py',
    'enhanced_log_sampler.py': 'log_sampler_plugin.py',
    'log_correlation_query.py': 'log_correlation_plugin.py',
    'log_aggregator.py': 'log_aggregator_plugin.py',
    'log_compressor.py': 'log_compressor_plugin.py',
    'log_metrics.py': 'log_metrics_plugin.py',
    'backpressure.py': 'log_backpressure_plugin.py',
    'comprehensive_error_framework.py': 'comprehensive_error_plugin.py',
    'error_codes.py': 'error_codes_utils.py',
    'exceptions.py': 'error_exceptions.py',
    'security_errors.py': 'security_error_plugin.py',
}


def find_documentation_files() -> List[Path]:
    """查找所有需要更新的文档文件"""
    docs_dir = Path("docs/architecture/infrastructure")
    if not docs_dir.exists():
        print(f"❌ 文档目录不存在: {docs_dir}")
        return []

    # 查找所有Markdown文件
    md_files = list(docs_dir.glob("*.md"))
    print(f"📁 找到 {len(md_files)} 个文档文件")
    return md_files


def update_file_content(file_path: Path, mappings: Dict[str, str]) -> Tuple[bool, List[str]]:
    """更新文件内容中的类名引用"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        changes = []

        # 更新类名引用
        for old_name, new_name in mappings.items():
            if old_name in content:
                # 使用正则表达式确保只替换完整的类名
                pattern = r'\b' + re.escape(old_name) + r'\b'
                new_content = re.sub(pattern, new_name, content)
                if new_content != content:
                    content = new_content
                    changes.append(f"  - {old_name} → {new_name}")

        # 更新文件名引用
        for old_file, new_file in FILE_NAME_MAPPINGS.items():
            if old_file in content:
                content = content.replace(old_file, new_file)
                changes.append(f"  - {old_file} → {new_file}")

        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, changes
        else:
            return False, []

    except Exception as e:
        print(f"❌ 更新文件失败 {file_path}: {e}")
        return False, []


def main():
    """主函数"""
    print("🚀 开始更新基础设施层文档...")
    print("=" * 60)

    # 查找文档文件
    files = find_documentation_files()
    if not files:
        return

    # 统计信息
    total_files = len(files)
    updated_files = 0
    total_changes = 0

    print(f"📋 类名映射表 ({len(CLASS_NAME_MAPPINGS)} 个):")
    for old_name, new_name in CLASS_NAME_MAPPINGS.items():
        print(f"  - {old_name} → {new_name}")

    print(f"\n📋 文件名映射表 ({len(FILE_NAME_MAPPINGS)} 个):")
    for old_file, new_file in FILE_NAME_MAPPINGS.items():
        print(f"  - {old_file} → {new_file}")

    print(f"\n🔄 开始处理 {total_files} 个文件...")
    print("-" * 60)

    # 处理每个文件
    for file_path in files:
        print(f"📄 处理: {file_path.name}")

        updated, changes = update_file_content(file_path, CLASS_NAME_MAPPINGS)

        if updated:
            updated_files += 1
            total_changes += len(changes)
            print(f"✅ 已更新 ({len(changes)} 处修改):")
            for change in changes:
                print(change)
        else:
            print("⏭️  无需更新")

        print()

    # 输出统计结果
    print("=" * 60)
    print(f"📊 更新完成!")
    print(f"  - 总文件数: {total_files}")
    print(f"  - 已更新文件: {updated_files}")
    print(f"  - 总修改数: {total_changes}")
    print(f"  - 更新率: {updated_files/total_files*100:.1f}%")

    if updated_files > 0:
        print("\n✅ 基础设施层文档已根据命名规范计划完成更新!")
    else:
        print("\nℹ️  所有文档都已符合最新的命名规范")


if __name__ == "__main__":
    main()
