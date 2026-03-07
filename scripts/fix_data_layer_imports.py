#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复数据层导入问题脚本
"""

import os
import re


def fix_import_issues():
    """修复数据层导入问题"""

    # 修复测试文件中的导入路径
    test_files_to_fix = [
        # 基础数据加载器测试
        ('tests/unit/data/test_base_dataloader.py',
         'from src.data.base_dataloader import BaseDataLoader',
         'from src.data.base_loader import BaseDataLoader'),

        # 数据加载器测试
        ('tests/unit/data/test_data_critical.py',
         'from src.data.data_loader import DataLoader',
         'from src.data.data_manager import DataManager'),

        ('tests/unit/data/test_data_loader_integration.py',
         'from src.data.data_loader import DataLoader',
         'from src.data.data_manager import DataManager'),

        ('tests/unit/data/test_data_loader_real.py',
         'from src.data.data_loader import DataLoader',
         'from src.data.data_manager import DataManager'),

        # 数据元数据测试
        ('tests/unit/data/test_data_metadata.py',
         'from src.data.data_metadata import DataMetadata',
         'from src.data.data_manager import DataModel'),

        # 并行加载器测试
        ('tests/unit/data/test_parallel_loader.py',
         'from src.data.parallel_loader import ParallelLoader, LoadResult',
         'from src.data.loader.parallel_loader import OptimizedParallelLoader as ParallelLoader, LoadResult'),

        # 数据验证器测试
        ('tests/unit/data/test_validation_data_validator.py',
         'from src.data.validation.data_validator import DataValidator, DataType',
         'from src.data.validator import DataValidator'),

        ('tests/unit/data/test_validation_data_validator_parametrize.py',
         'from src.data.validation.data_validator import DataValidator, DataType',
         'from src.data.validator import DataValidator'),

        ('tests/unit/data/validation/test_data_validator.py',
         'import src.data.validation.data_validator',
         'import src.data.validator'),

        ('tests/unit/data/validation/test_data_validator_coverage.py',
         'from src.data.validation.data_validator import DataValidator, DataType, ValidationRule',
         'from src.data.validator import DataValidator'),
    ]

    for file_path, old_import, new_import in test_files_to_fix:
        if os.path.exists(file_path):
            print(f"修复 {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 替换导入语句
            content = content.replace(old_import, new_import)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

    # 修复数据处理测试中的FillMethod导入
    processing_test_files = [
        'tests/unit/data/processing/test_data_processor.py',
        'tests/unit/data/test_processing_data_processor_parametrize.py'
    ]

    for file_path in processing_test_files:
        if os.path.exists(file_path):
            print(f"修复 {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 确保FillMethod从正确的模块导入
            content = re.sub(
                r'from src\.data\.processing\.data_processor import DataProcessor, FillMethod',
                'from src.data.processing.data_processor import DataProcessor, FillMethod',
                content
            )

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

    # 修复并行加载器测试
    parallel_loader_test = 'tests/unit/data/loader/test_parallel_loader.py'
    if os.path.exists(parallel_loader_test):
        print(f"修复 {parallel_loader_test}")
        with open(parallel_loader_test, 'r', encoding='utf-8') as f:
            content = f.read()

        # 替换导入语句
        content = content.replace(
            'from src.data.loader.parallel_loader import ParallelDataLoader, MockResult',
            'from src.data.loader.parallel_loader import OptimizedParallelLoader as ParallelDataLoader, LoadResult as MockResult'
        )

        with open(parallel_loader_test, 'w', encoding='utf-8') as f:
            f.write(content)

    print("数据层导入问题修复完成")


if __name__ == "__main__":
    fix_import_issues()
