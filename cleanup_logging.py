#!/usr/bin/env python3
"""
清理日志系统无效文件和目录

根据基础设施层架构设计原则，删除不应存在于日志系统中的组件。
"""

import shutil
from pathlib import Path


def cleanup_logging_system():
    """清理日志系统，删除无效的目录和文件"""

    logging_dir = Path("src/infrastructure/logging")

    # 需要删除的目录列表
    dirs_to_remove = [
        "config",           # 日志不应有配置系统
        "business",         # 日志不应有业务逻辑
        "cloud",           # 云服务集成应在更高层
        "distributed",     # 分布式功能过于复杂
        "intelligent",     # 智能功能过于复杂
        "engine",          # 引擎功能不属于基础日志
        "plugins",         # 插件系统过于复杂
    ]

    # 需要删除的文件列表（不属于日志核心功能的文件）
    files_to_remove = [
        "business_service.py",
        "chaos_orchestrator.py",
        "circuit_breaker.py",
        "connection_pool.py",
        "data_consistency.py",
        "data_sanitizer.py",
        "data_sync.py",
        "data_validation_service.py",
        "deployment_validator.py",
        "disaster_recovery.py",
        "distributed_lock.py",
        "distributed_monitoring.py",
        "encryption_service.py",
        "enhanced_container.py",
        "grafana_integration.py",
        "hot_reload_service.py",
        "influxdb_store.py",
        "integrity_checker.py",
        "log_aggregator_plugin.py",
        "log_archiver.py",
        "log_correlation_plugin.py",
        "log_level_optimizer.py",
        "log_metrics_plugin.py",
        "log_sampler_plugin.py",
        "log_sampler.py",
        "logger_components.py",
        "logging_service_components.py",
        "logging_strategy.py",
        "logging_utils.py",
        "metrics_aggregator.py",
        "micro_service.py",
        "microservice_manager.py",
        "model_service.py",
        "monitor_factory.py",
        "persistent_error_handler.py",
        "priority_queue.py",
        "production_ready.py",
        "prometheus_compat.py",
        "quant_filter.py",
        "regulatory_compliance.py",
        "regulatory_reporter.py",
        "service_launcher.py",
        "smart_log_filter.py",
        "structured_logger.py",
        "sync_conflict_manager.py",
        "sync_node_manager.py",
        "trading_service.py",
        "unified_hot_reload_service.py",
        "unified_sync_service.py",
        "config_components.py",
        "storage_adapter.py",
    ]

    print("🧹 开始清理日志系统...")
    print(f"📁 工作目录: {logging_dir.absolute()}")

    removed_items = []

    # 删除无效目录
    for dir_name in dirs_to_remove:
        dir_path = logging_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            try:
                shutil.rmtree(dir_path)
                removed_items.append(f"目录: {dir_name}/")
                print(f"✅ 删除目录: {dir_name}/")
            except Exception as e:
                print(f"❌ 删除目录失败 {dir_name}: {e}")

    # 删除无效文件
    for file_name in files_to_remove:
        file_path = logging_dir / file_name
        if file_path.exists() and file_path.is_file():
            try:
                file_path.unlink()
                removed_items.append(f"文件: {file_name}")
                print(f"✅ 删除文件: {file_name}")
            except Exception as e:
                print(f"❌ 删除文件失败 {file_name}: {e}")

    # 清理空的__pycache__目录
    for pycache_dir in logging_dir.rglob("__pycache__"):
        if pycache_dir.is_dir() and not list(pycache_dir.iterdir()):
            try:
                pycache_dir.rmdir()
                print(f"✅ 删除空目录: {pycache_dir.relative_to(logging_dir)}")
            except Exception as e:
                print(f"❌ 删除空目录失败 {pycache_dir}: {e}")

    print("\n📊 清理统计:")
    print(f"  • 删除项目数量: {len(removed_items)}")
    print(f"  • 保留核心功能: core/, handlers/, formatters/, utils/, monitors/, security/, storage/, standards/, services/")

    # 显示保留的文件
    remaining_files = list(logging_dir.glob("*.py"))
    remaining_dirs = [d for d in logging_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

    print(f"  • 保留文件数量: {len(remaining_files)}")
    print(f"  • 保留目录数量: {len(remaining_dirs)}")

    print("\n🎉 日志系统清理完成！")
    print("建议: 进行代码风格修复和功能验证")

    return len(removed_items)


if __name__ == "__main__":
    removed_count = cleanup_logging_system()
    print(f"\n总共清理了 {removed_count} 个项目")
