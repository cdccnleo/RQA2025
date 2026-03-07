#!/usr/bin/env python3
"""测试文件清理脚本"""

import shutil
from pathlib import Path

# 需要删除的文件列表
files_to_delete = [
    "tests/unit/features/conftest.py",
    "tests/unit/infrastructure/config/conftest.py",
    "tests/unit/infrastructure/error/conftest.py",
]

# 需要废弃的文件列表（移动到废弃目录）
files_to_deprecate = [
    "tests/unit/features/auto_test_feature_engine.py",
    "tests/unit/features/auto_test_feature_engineer.py",
    "tests/unit/features/auto_test_feature_manager.py",
    "tests/unit/features/auto_test_sentiment_analyzer.py",
    "tests/unit/features/auto_test_signal_generator.py",
    "tests/unit/features/test_feature_engineer_isolated.py",
    "tests/unit/features/test_feature_importance_isolated.py",
    "tests/unit/features/test_feature_manager_offline.py",
    "tests/unit/features/test_feature_metadata_isolated.py",
    "tests/unit/infrastructure/auto_test_circuit_breaker.py",
    "tests/unit/infrastructure/auto_test_service_launcher.py",
    "tests/unit/infrastructure/auto_test_visual_monitor.py",
    "tests/unit/infrastructure/test_async_inference_engine_comprehensive.py",
    "tests/unit/infrastructure/test_async_inference_engine_coverage.py",
    "tests/unit/infrastructure/test_config_comprehensive.py",
    "tests/unit/infrastructure/test_config_manager_comprehensive.py",
    "tests/unit/infrastructure/test_config_manager_coverage.py",
    "tests/unit/infrastructure/test_config_manager_simple.py",
    "tests/unit/infrastructure/test_core_modules_simple.py",
    "tests/unit/infrastructure/test_core_modules_standalone.py",
    "tests/unit/infrastructure/test_error_handling_comprehensive.py",
    "tests/unit/infrastructure/test_integration_comprehensive.py",
    "tests/unit/infrastructure/test_logging_comprehensive.py",
    "tests/unit/infrastructure/test_service_launcher_coverage.py",
    "tests/unit/infrastructure/config/test_config_comprehensive.py",
    "tests/unit/infrastructure/config/test_config_exceptions_comprehensive.py",
    "tests/unit/infrastructure/config/test_config_result_comprehensive.py",
    "tests/unit/infrastructure/config/test_config_schema_comprehensive.py",
    "tests/unit/infrastructure/config/test_database_storage_comprehensive.py",
    "tests/unit/infrastructure/config/test_database_storage_standalone.py",
    "tests/unit/infrastructure/config/test_file_storage_comprehensive.py",
    "tests/unit/infrastructure/config/test_redis_storage_comprehensive.py",
    "tests/unit/infrastructure/config/test_storage_comprehensive.py",
    "tests/unit/infrastructure/config/test_strategies_comprehensive.py",
    "tests/unit/infrastructure/config/test_typed_config_comprehensive.py",
    "tests/unit/infrastructure/config/test_unified_cache_comprehensive.py",
    "tests/unit/infrastructure/config/test_unified_config_manager_simple.py",
    "tests/unit/infrastructure/database/test_database_comprehensive.py",
    "tests/unit/infrastructure/database/test_database_manager_comprehensive.py",
    "tests/unit/infrastructure/database/test_database_manager_simple.py",
    "tests/unit/infrastructure/database/test_unified_database_manager_simple.py",
    "tests/unit/infrastructure/deployment/test_deployment_manager_comprehensive.py",
    "tests/unit/infrastructure/error/test_circuit_breaker_comprehensive.py",
    "tests/unit/infrastructure/error/test_circuit_breaker_simple.py",
    "tests/unit/infrastructure/error/test_error_comprehensive.py",
    "tests/unit/infrastructure/error/test_error_handler_comprehensive.py",
    "tests/unit/infrastructure/error/test_error_handling_comprehensive.py",
    "tests/unit/infrastructure/logging/test_logging_system_comprehensive.py",
    "tests/unit/infrastructure/monitoring/test_monitoring_comprehensive.py",
    "tests/unit/infrastructure/monitoring/test_monitoring_system_comprehensive.py",
    "tests/unit/infrastructure/m_logging/test_logging_comprehensive.py",
    "tests/unit/infrastructure/m_logging/test_logging_manager_comprehensive.py",
    "tests/unit/infrastructure/m_logging/test_log_manager_simple.py",
    "tests/unit/infrastructure/network/test_network_manager_comprehensive.py",
    "tests/unit/infrastructure/scheduler/test_task_scheduler_comprehensive.py",
    "tests/unit/infrastructure/security/test_security_comprehensive.py",
    "tests/unit/infrastructure/security/test_security_system_comprehensive.py",
    "tests/unit/infrastructure/storage/test_kafka_storage_coverage.py",
    "tests/unit/infrastructure/storage/test_storage_comprehensive.py",
    "tests/unit/infrastructure/storage/test_storage_system_comprehensive.py",
    "tests/unit/infrastructure/utils/test_utils_comprehensive.py",
]


def cleanup_test_files():
    """清理测试文件"""
    print("开始清理测试文件...")

    # 创建废弃目录
    deprecated_dir = Path("tests/deprecated")
    deprecated_dir.mkdir(exist_ok=True)

    # 删除文件
    for file_path in files_to_delete:
        path = Path(file_path)
        if path.exists():
            print(f"删除文件: {file_path}")
            path.unlink()
        else:
            print(f"文件不存在: {file_path}")

    # 移动废弃文件
    for file_path in files_to_deprecate:
        path = Path(file_path)
        if path.exists():
            new_path = deprecated_dir / path.name
            print(f"移动文件: {file_path} -> {new_path}")
            shutil.move(str(path), str(new_path))
        else:
            print(f"文件不存在: {file_path}")

    print("清理完成！")


if __name__ == "__main__":
    cleanup_test_files()
