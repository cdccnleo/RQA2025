#!/usr/bin/env python3
"""
手动移动基础设施层测试文件的脚本
"""

import os
import shutil
from pathlib import Path


def create_directories():
    """创建子模块目录"""
    base_dir = Path("tests/unit/infrastructure")
    subdirs = ["cache", "config", "error", "health", "logging",
               "resource", "monitoring", "distributed", "interfaces", "utils"]

    for subdir in subdirs:
        dir_path = base_dir / subdir
        dir_path.mkdir(exist_ok=True)
        init_file = dir_path / "__init__.py"
        if not init_file.exists():
            init_file.write_text(f"# {subdir} 子模块测试用例\n")


def move_files_by_pattern(patterns, target_dir):
    """根据模式移动文件"""
    base_dir = Path("tests/unit/infrastructure")
    target_path = base_dir / target_dir

    for filename in os.listdir(base_dir):
        if not filename.endswith('.py') or filename == '__init__.py':
            continue

        filename_lower = filename.lower()
        should_move = False

        for pattern in patterns:
            if pattern in filename_lower:
                should_move = True
                break

        if should_move:
            source = base_dir / filename
            target = target_path / filename
            shutil.move(str(source), str(target))
            print(f"✅ 移动: {filename} -> {target_dir}/")


def move_remaining_files():
    """移动剩余的文件到utils目录"""
    base_dir = Path("tests/unit/infrastructure")
    utils_dir = base_dir / "utils"

    for filename in os.listdir(base_dir):
        if not filename.endswith('.py') or filename == '__init__.py':
            continue

        source = base_dir / filename
        if source.is_file():  # 只移动文件，不移动目录
            target = utils_dir / filename
            shutil.move(str(source), str(target))
            print(f"✅ 移动剩余文件: {filename} -> utils/")


def main():
    print("开始重新组织基础设施层测试文件...")

    # 创建目录
    create_directories()
    print("✅ 目录创建完成")

    # 移动缓存相关文件
    cache_patterns = ['cache', 'redis', 'memory', 'lru', 'unified_cache',
                      'multi_level', 'smart_cache', 'distributed_cache']
    move_files_by_pattern(cache_patterns, "cache")

    # 移动配置相关文件
    config_patterns = ['config', 'configuration', 'registry']
    move_files_by_pattern(config_patterns, "config")

    # 移动错误处理相关文件
    error_patterns = ['error', 'exception', 'circuit_breaker', 'retry', 'boundary']
    move_files_by_pattern(error_patterns, "error")

    # 移动健康检查相关文件
    health_patterns = ['health', 'checker']
    move_files_by_pattern(health_patterns, "health")

    # 移动日志相关文件
    logging_patterns = ['log', 'logger', 'logging']
    move_files_by_pattern(logging_patterns, "logging")

    # 移动资源管理相关文件
    resource_patterns = ['resource', 'pool', 'quota', 'connection', 'concurrency']
    move_files_by_pattern(resource_patterns, "resource")

    # 移动监控相关文件
    monitoring_patterns = ['monitor', 'alert', 'metrics', 'performance', 'system_monitor']
    move_files_by_pattern(monitoring_patterns, "monitoring")

    # 移动分布式相关文件
    distributed_patterns = ['distributed']
    move_files_by_pattern(distributed_patterns, "distributed")

    # 移动接口相关文件
    interfaces_patterns = ['interfaces', 'base']
    move_files_by_pattern(interfaces_patterns, "interfaces")

    # 移动工具相关文件
    utils_patterns = ['utils', 'async', 'processor', 'file_system', 'micro',
                      'service_launcher', 'dynamic_executor', 'parallel_loader', 'integrity_checker']
    move_files_by_pattern(utils_patterns, "utils")

    # 移动剩余文件
    move_remaining_files()

    print("🎉 测试文件重组织完成！")


if __name__ == "__main__":
    main()
