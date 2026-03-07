#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
核心服务层代码重复检测分析器
"""

import os
import hashlib
from collections import defaultdict
import ast
from typing import Dict, List, Tuple


class CodeDuplicateDetector:
    def __init__(self, root_path: str):
        self.root_path = root_path
        self.duplicates = defaultdict(list)
        self.function_signatures = defaultdict(list)
        self.class_signatures = defaultdict(list)
        self.import_patterns = defaultdict(list)

    def get_file_hash(self, filepath: str, block_size: int = 65536) -> str:
        """计算文件哈希"""
        hasher = hashlib.md5()
        try:
            with open(filepath, 'rb') as f:
                while True:
                    data = f.read(block_size)
                    if not data:
                        break
                    hasher.update(data)
            return hasher.hexdigest()
        except:
            return None

    def extract_function_signatures(self, content: str, filepath: str) -> List[Tuple[str, int]]:
        """提取函数签名"""
        signatures = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # 提取函数名和参数
                    args = [arg.arg for arg in node.args.args]
                    signature = f'{node.name}({", ".join(args)})'
                    signatures.append((signature, node.lineno))
        except:
            pass
        return signatures

    def extract_class_signatures(self, content: str, filepath: str) -> List[Tuple[str, int]]:
        """提取类签名"""
        signatures = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    signature = f'class {node.name}'
                    if node.bases:
                        base_names = [getattr(base, 'id', str(base)) for base in node.bases]
                        signature += f'({", ".join(base_names)})'
                    signatures.append((signature, node.lineno))
        except:
            pass
        return signatures

    def extract_import_patterns(self, content: str, filepath: str) -> List[str]:
        """提取导入模式"""
        patterns = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        patterns.append(f'import {alias.name}')
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        patterns.append(f'from {module} import {alias.name}')
        except:
            pass
        return patterns

    def find_duplicate_files(self) -> Dict[str, List[str]]:
        """查找重复文件"""
        file_hashes = defaultdict(list)

        for root, dirs, files in os.walk(self.root_path):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    file_hash = self.get_file_hash(filepath)
                    if file_hash:
                        file_hashes[file_hash].append(filepath)

        # 只保留有重复的文件
        duplicates = {k: v for k, v in file_hashes.items() if len(v) > 1}
        return duplicates

    def analyze_duplicates(self) -> Dict:
        """分析重复代码"""
        print('📁 扫描Python文件...')
        py_files = []
        for root, dirs, files in os.walk(self.root_path):
            for file in files:
                if file.endswith('.py'):
                    py_files.append(os.path.join(root, file))

        print(f'📊 发现 {len(py_files)} 个Python文件')

        # 分析每个文件
        for filepath in py_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 提取各种签名
                func_sigs = self.extract_function_signatures(content, filepath)
                class_sigs = self.extract_class_signatures(content, filepath)
                import_patterns = self.extract_import_patterns(content, filepath)

                # 记录到全局映射
                for sig, line in func_sigs:
                    self.function_signatures[sig].append((filepath, line))

                for sig, line in class_sigs:
                    self.class_signatures[sig].append((filepath, line))

                for pattern in import_patterns:
                    self.import_patterns[pattern].append(filepath)

            except Exception as e:
                print(f'⚠️ 处理文件失败 {filepath}: {e}')

        # 查找重复文件
        duplicate_files = self.find_duplicate_files()

        return {
            'duplicate_files': duplicate_files,
            'duplicate_functions': {k: v for k, v in self.function_signatures.items() if len(v) > 1},
            'duplicate_classes': {k: v for k, v in self.class_signatures.items() if len(v) > 1},
            'duplicate_imports': {k: v for k, v in self.import_patterns.items() if len(v) > 1}
        }


def main():
    print('🔍 核心服务层代码重复检测分析')
    print('=' * 60)

    # 运行重复检测
    detector = CodeDuplicateDetector('src/core')
    results = detector.analyze_duplicates()

    print('\n📋 重复代码检测结果')
    print('=' * 50)

    # 统计结果
    duplicate_files_count = len(results['duplicate_files'])
    duplicate_functions_count = len(results['duplicate_functions'])
    duplicate_classes_count = len(results['duplicate_classes'])
    duplicate_imports_count = len(results['duplicate_imports'])

    print(f'🔍 完全重复文件: {duplicate_files_count} 组')
    print(f'🔍 重复函数签名: {duplicate_functions_count} 个')
    print(f'🔍 重复类签名: {duplicate_classes_count} 个')
    print(f'🔍 重复导入模式: {duplicate_imports_count} 个')

    print('\n📁 详细分析:')

    if duplicate_files_count > 0:
        print('\n1️⃣ 完全重复文件:')
        for hash_val, files in list(results['duplicate_files'].items())[:5]:  # 只显示前5个
            print(f'   重复 {len(files)} 个文件:')
            for file in files[:3]:  # 每个组只显示前3个
                print(f'     • {os.path.relpath(file, "src/core")}')
            if len(files) > 3:
                print(f'     ... 还有 {len(files) - 3} 个文件')

    if duplicate_functions_count > 0:
        print('\n2️⃣ 重复函数签名:')
        count = 0
        for func_sig, locations in results['duplicate_functions'].items():
            if count >= 5:  # 只显示前5个
                break
            if len(locations) > 1:
                print(f'   函数 "{func_sig}" 在 {len(locations)} 个文件中出现:')
                for filepath, line in locations[:3]:
                    print(f'     • {os.path.relpath(filepath, "src/core")}:{line}')
                count += 1

    if duplicate_classes_count > 0:
        print('\n3️⃣ 重复类签名:')
        count = 0
        for class_sig, locations in results['duplicate_classes'].items():
            if count >= 3:  # 只显示前3个
                break
            if len(locations) > 1:
                print(f'   类 "{class_sig}" 在 {len(locations)} 个文件中出现:')
                for filepath, line in locations[:2]:
                    print(f'     • {os.path.relpath(filepath, "src/core")}:{line}')
                count += 1

    if duplicate_imports_count > 0:
        print('\n4️⃣ 重复导入模式:')
        count = 0
        for import_pattern, files in results['duplicate_imports'].items():
            if count >= 5:  # 只显示前5个
                break
            if len(files) > 2:  # 只显示重复次数>2的
                print(f'   导入 "{import_pattern}" 在 {len(files)} 个文件中出现')
                count += 1

    print('\n📊 重复代码影响评估')
    print('=' * 45)

    # 计算影响指标
    total_duplicate_functions = sum(len(v) for v in results['duplicate_functions'].values())
    total_duplicate_classes = sum(len(v) for v in results['duplicate_classes'].values())
    total_duplicate_imports = sum(len(v) for v in results['duplicate_imports'].values())

    print(f'🔴 高影响: 完全重复文件 {duplicate_files_count} 组')
    print(f'🟡 中影响: 重复函数 {total_duplicate_functions} 个实例')
    print(f'🟢 低影响: 重复类 {total_duplicate_classes} 个实例')
    print(f'🟢 低影响: 重复导入 {total_duplicate_imports} 个实例')

    print('\n💡 改进建议')
    print('=' * 35)
    if duplicate_files_count > 0:
        print('🔴 紧急: 合并或重构完全重复的文件')
    if duplicate_functions_count > 10:
        print('🟡 重要: 提取重复的函数到公共模块')
    if duplicate_classes_count > 5:
        print('🟡 重要: 统一重复的类定义')
    if duplicate_imports_count > 20:
        print('🟢 建议: 标准化导入语句')

    print('\n🎯 总体评估')
    print('=' * 35)
    if duplicate_files_count > 0:
        print('⚠️ 存在严重重复问题，需要立即处理')
    elif duplicate_functions_count > 10 or duplicate_classes_count > 5:
        print('📊 存在中等程度重复，可以逐步优化')
    else:
        print('✅ 重复代码控制良好')

    print('\n🏆 核心服务层重复检测分析完成')


if __name__ == "__main__":
    main()
