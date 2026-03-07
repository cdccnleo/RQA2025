#!/usr/bin/env python3
"""
从备份恢复cache测试文件
"""

import shutil
from pathlib import Path


def restore_cache_test_files():
    """恢复cache测试文件"""

    # 备份目录路径
    backup_dir = Path(
        "backup/infrastructure_tests_backup_20250823_235700/infrastructure_tests_original/cache")
    # 目标目录路径
    target_dir = Path("tests/unit/infrastructure/cache")

    if not backup_dir.exists():
        print(f"❌ 备份目录不存在: {backup_dir}")
        return

    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)

    # 需要恢复的文件列表（我们之前修复过但可能有问题的文件）
    files_to_restore = [
        'test_advanced_cache_manager.py',
        'test_base_cache_manager.py',
        'test_base.py',
        'test_business_metrics_plugin.py',
        'test_cache.py',
        'test_cache_basic.py',
        'test_cache_core.py',
        'test_cache_coverage_enhanced.py',
        'test_cache_factory.py',
        'test_cache_factory_enhanced.py',
        'test_cache_interfaces.py',
        'test_cache_manager_basic.py',
        'test_cache_manager_comprehensive.py',
        'test_cache_manager_coverage.py',
        'test_cache_managers.py',
        'test_cache_optimizer.py',
        'test_cache_performance.py',
        'test_cache_performance_tester.py',
        'test_cache_production.py',
        'test_cache_service.py',
        'test_cache_system.py',
        'test_cache_thread_cleanup.py',
        'test_cache_utils.py',
        'test_cache_utils_enhanced.py',
        'test_cached_manager.py',
        'test_caching.py',
        'test_china_cache_policy_comprehensive.py',
        'test_china_cache_policy_simple.py',
        'test_client_sdk.py',
        'test_data_cache_architecture_compliance.py',
        'test_data_cache_comprehensive.py',
        'test_data_cache_simple.py',
        'test_dependency.py',
        'test_disk_cache.py',
        'test_disk_cache_basic.py',
        'test_disk_cache_comprehensive.py',
        'test_distributed_lock.py',
        'test_enhanced_cache_manager.py',
        'test_enhanced_cache_manager_comprehensive.py',
        'test_enhanced_cache_manager_fixed.py',
        'test_enhanced_cache_strategy_comprehensive.py',
        'test_enhanced_health_checker.py',
        'test_feature_cache.py',
        'test_icache_backend.py',
        'test_inference_cache_complete.py',
        'test_inference_cache_coverage_enhanced.py',
        'test_interfaces.py',
        'test_kafka_storage.py',
        'test_kafka_storage_coverage.py',
        'test_memory_cache.py',
        'test_memory_cache_enhanced.py',
        'test_multi_level_cache.py',
        'test_multi_level_cache_basic.py',
        'test_multi_level_cache_comprehensive.py',
        'test_optimized_cache_service.py',
        'test_performance_enhanced.py',
        'test_performance_integration.py',
        'test_performance_optimizer_plugin.py',
        'test_query_cache_manager.py',
        'test_redis.py',
        'test_redis_adapter.py',
        'test_redis_cache.py',
        'test_redis_cache_adapter_comprehensive.py',
        'test_redis_cache_enhanced.py',
        'test_redis_storage.py',
        'test_simple_memory_cache.py',
        'test_simple_memory_cache_enhanced.py',
        'test_smart_cache_strategy.py',
        'test_storage.py',
        'test_storage_adapter.py',
        'test_storage_core.py',
        'test_storage_monitor_plugin.py',
        'test_storage_monitor_plugin_enhanced.py',
        'test_storage_service_enhanced.py',
        'test_storage_service_simple.py',
        'test_storage_services.py',
        'test_unified_cache.py',
        'test_unified_cache_comprehensive.py',
        'test_unified_cache_enhanced.py',
        'test_unified_cache_factory.py',
        'test_unified_sync.py',
        'test_websocket_api.py'
    ]

    restored_count = 0
    failed_count = 0

    for filename in files_to_restore:
        backup_file = backup_dir / filename
        target_file = target_dir / filename

        if backup_file.exists():
            try:
                # 创建备份
                if target_file.exists():
                    backup_target = str(target_file) + '.before_restore'
                    shutil.copy2(str(target_file), backup_target)

                # 复制备份文件
                shutil.copy2(str(backup_file), str(target_file))
                restored_count += 1
                print(f'✅ 恢复成功: {filename}')

            except Exception as e:
                failed_count += 1
                print(f'❌ 恢复失败 {filename}: {e}')
        else:
            failed_count += 1
            print(f'❌ 备份文件不存在: {filename}')

    print(f"\n恢复统计:")
    print(f"   成功恢复: {restored_count} 个文件")
    print(f"   恢复失败: {failed_count} 个文件")
    print(f"   总计文件: {len(files_to_restore)} 个")

    if restored_count > 0:
        print(f"\n提示:")
        print("   - 原文件已备份为 .before_restore 后缀")
        print("   - 请验证恢复的文件是否正常工作")


if __name__ == "__main__":
    restore_cache_test_files()
