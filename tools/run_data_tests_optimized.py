#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
优化的数据层测试运行脚本
分批运行测试以避免内存问题
"""

import os
import sys
import subprocess
import time
import gc
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def get_test_groups():
    """获取测试分组"""
    return [
        # 核心模块测试
        {
            'name': 'core_modules',
            'paths': [
                'tests/unit/data/test_base_loader.py',
                'tests/unit/data/test_data_manager.py',
                'tests/unit/data/test_validator.py',
                'tests/unit/data/test_registry.py',
                'tests/unit/data/test_cache_manager.py'
            ]
        },
        # 数据处理模块测试
        {
            'name': 'processing_modules',
            'paths': [
                'tests/unit/data/processing/',
                'tests/unit/data/transformers/',
                'tests/unit/data/alignment/',
                'tests/unit/data/export/'
            ]
        },
        # 数据加载器测试
        {
            'name': 'loader_modules',
            'paths': [
                'tests/unit/data/loader/',
                'tests/unit/data/adapters/',
                'tests/unit/data/china/'
            ]
        },
        # 数据验证模块测试
        {
            'name': 'validation_modules',
            'paths': [
                'tests/unit/data/validation/',
                'tests/unit/data/quality/',
                'tests/unit/data/monitoring/'
            ]
        },
        # 数据修复和版本控制测试
        {
            'name': 'repair_version_modules',
            'paths': [
                'tests/unit/data/repair/',
                'tests/unit/data/version_control/'
            ]
        },
        # 数据湖和缓存测试
        {
            'name': 'lake_cache_modules',
            'paths': [
                'tests/unit/data/lake/',
                'tests/unit/data/cache/'
            ]
        },
        # 实时和流处理测试
        {
            'name': 'realtime_streaming_modules',
            'paths': [
                'tests/unit/data/realtime/',
                'tests/unit/data/streaming/',
                'tests/unit/data/preload/'
            ]
        },
        # 分布式和并行处理测试
        {
            'name': 'distributed_parallel_modules',
            'paths': [
                'tests/unit/data/distributed/',
                'tests/unit/data/parallel/'
            ]
        },
        # 机器学习质量评估测试
        {
            'name': 'ml_quality_modules',
            'paths': [
                'tests/unit/data/ml/',
                'tests/unit/data/performance/'
            ]
        },
        # 接口和模型测试
        {
            'name': 'interface_model_modules',
            'paths': [
                'tests/unit/data/interfaces/',
                'tests/unit/data/models.py',
                'tests/unit/data/metadata.py'
            ]
        },
        # 服务和其他模块测试
        {
            'name': 'service_other_modules',
            'paths': [
                'tests/unit/data/services/',
                'tests/unit/data/decoders/',
                'tests/unit/data/core/'
            ]
        }
    ]


def run_test_group(group_name, test_paths, max_memory_mb=500):
    """运行测试分组"""
    print(f"\n{'='*60}")
    print(f"运行测试分组: {group_name}")
    print(f"{'='*60}")

    # 构建pytest命令
    pytest_args = [
        'python', 'scripts/testing/run_tests.py',
        '--skip-coverage',
        '--pytest-args', '-v', '--tb=short'
    ]

    # 添加测试路径
    for path in test_paths:
        if os.path.exists(path):
            pytest_args.extend(['--test-path', path])

    # 添加内存限制
    pytest_args.extend(['--max-memory', str(max_memory_mb)])

    print(f"执行命令: {' '.join(pytest_args)}")

    try:
        # 执行测试
        start_time = time.time()
        result = subprocess.run(
            pytest_args,
            capture_output=True,
            text=True,
            timeout=600  # 10分钟超时
        )
        end_time = time.time()

        print(f"测试完成，耗时: {end_time - start_time:.2f}秒")
        print(f"退出码: {result.returncode}")

        if result.stdout:
            print("标准输出:")
            print(result.stdout[-2000:])  # 只显示最后2000字符

        if result.stderr:
            print("错误输出:")
            print(result.stderr[-2000:])  # 只显示最后2000字符

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print("测试超时")
        return False
    except Exception as e:
        print(f"测试执行失败: {e}")
        return False


def cleanup_memory():
    """清理内存"""
    print("执行内存清理...")
    gc.collect()
    time.sleep(1)  # 给垃圾回收器一些时间


def main():
    """主函数"""
    print("数据层测试优化运行脚本")
    print("分批运行测试以避免内存问题")

    # 获取测试分组
    test_groups = get_test_groups()

    # 统计信息
    total_groups = len(test_groups)
    successful_groups = 0
    failed_groups = []

    # 运行每个测试分组
    for i, group in enumerate(test_groups, 1):
        print(f"\n进度: {i}/{total_groups}")

        # 清理内存
        cleanup_memory()

        # 运行测试分组
        success = run_test_group(
            group['name'],
            group['paths'],
            max_memory_mb=500  # 限制每个分组最大内存使用
        )

        if success:
            successful_groups += 1
            print(f"✓ {group['name']} 测试通过")
        else:
            failed_groups.append(group['name'])
            print(f"✗ {group['name']} 测试失败")

        # 测试分组间休息
        time.sleep(2)

    # 输出总结
    print(f"\n{'='*60}")
    print("测试总结")
    print(f"{'='*60}")
    print(f"总分组数: {total_groups}")
    print(f"成功分组: {successful_groups}")
    print(f"失败分组: {len(failed_groups)}")

    if failed_groups:
        print("失败的分组:")
        for group in failed_groups:
            print(f"  - {group}")

    if successful_groups == total_groups:
        print("🎉 所有测试分组都通过了!")
        return 0
    else:
        print("⚠️  部分测试分组失败")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
