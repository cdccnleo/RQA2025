"""
分析代码重复模式并制定消除计划
"""

import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple


def find_similar_functions(directory: str, min_lines: int = 5) -> Dict[str, List[Tuple[str, str]]]:
    """
    查找相似的函数定义

    Args:
        directory: 要分析的目录
        min_lines: 最少行数

    Returns:
        函数相似性分组字典
    """
    function_signatures = defaultdict(list)
    function_bodies = defaultdict(list)

    # 遍历Python文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 查找函数定义
                    func_pattern = r'def\s+(\w+)\s*\([^)]*\)\s*:'  # 更精确的函数匹配
                    lines = content.split('\n')

                    for i, line in enumerate(lines):
                        match = re.match(func_pattern, line.strip())
                        if match:
                            func_name = match.group(1)
                            func_start = i

                            # 找到函数体结束（下一个def或类定义，或缩进减少）
                            func_body = []
                            base_indent = len(line) - len(line.lstrip())
                            j = func_start + 1

                            while j < len(lines):
                                current_line = lines[j]
                                if current_line.strip().startswith(('def ', 'class ')):
                                    # 检查缩进
                                    current_indent = len(current_line) - len(current_line.lstrip())
                                    if current_indent <= base_indent:
                                        break
                                elif current_line.strip() and not current_line.startswith(' ' * (base_indent + 1)):
                                    # 非空行且缩进不大于函数体
                                    break

                                func_body.append(current_line)
                                j += 1

                            # 只处理有意义的函数体
                            if len(func_body) >= min_lines:
                                # 标准化函数体（移除注释和空行）
                                normalized_body = []
                                for line in func_body:
                                    stripped = line.strip()
                                    if stripped and not stripped.startswith('#'):
                                        # 移除行首缩进差异
                                        normalized_body.append(stripped)

                                if normalized_body:
                                    body_key = '\n'.join(normalized_body[:min_lines])  # 使用前N行作为签名
                                    function_bodies[body_key].append((file_path, func_name))

                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {e}")

    # 过滤出重复的函数
    duplicates = {}
    for body_key, functions in function_bodies.items():
        if len(functions) > 1:
            duplicates[body_key] = functions

    return duplicates


def find_similar_imports(directory: str) -> Dict[str, List[str]]:
    """查找相似的导入语句"""

    import_patterns = defaultdict(list)

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                    imports = []
                    for line in lines[:30]:  # 只检查文件前30行
                        stripped = line.strip()
                        if stripped.startswith(('import ', 'from ')):
                            imports.append(stripped)
                        elif stripped and not stripped.startswith('#'):
                            # 遇到非导入非注释行，停止
                            break

                    if imports:
                        import_key = '\n'.join(sorted(imports))  # 排序以便比较
                        import_patterns[import_key].append(file_path)

                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {e}")

    # 过滤重复的导入模式
    duplicates = {}
    for import_key, files in import_patterns.items():
        if len(files) > 1:
            duplicates[import_key] = files

    return duplicates


def find_similar_error_handling(directory: str) -> Dict[str, List[Tuple[str, int]]]:
    """查找相似的错误处理模式"""

    error_patterns = defaultdict(list)

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 查找try-except块
                    try_pattern = r'try:\s*\n((?:\s+.*\n)*?)\s*except\s+([^:]+):\s*\n((?:\s+.*\n)*?)'
                    matches = re.finditer(try_pattern, content, re.MULTILINE | re.DOTALL)

                    for match in matches:
                        try_block = match.group(1).strip()
                        except_clause = match.group(2).strip()
                        except_block = match.group(3).strip()

                        # 标准化错误处理模式
                        pattern_key = f"try:\n{try_block}\nexcept {except_clause}:\n{except_block}"

                        # 只保留有意义的模式
                        if len(try_block.split('\n')) > 1 and len(except_block.split('\n')) > 1:
                            error_patterns[pattern_key].append((file_path, match.start()))

                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {e}")

    # 过滤重复的错误处理模式
    duplicates = {}
    for pattern_key, locations in error_patterns.items():
        if len(locations) > 1:
            duplicates[pattern_key] = locations

    return duplicates


def analyze_code_duplicates(directory: str):
    """综合分析代码重复"""

    print("🔍 代码重复模式分析")
    print("=" * 60)

    # 1. 相似函数分析
    print("\n1. 📋 相似函数分析")
    similar_functions = find_similar_functions(directory, min_lines=5)

    if similar_functions:
        print(f"发现 {len(similar_functions)} 组相似函数:")
        for i, (body_key, functions) in enumerate(similar_functions.items(), 1):
            print(f"\n组 {i}: {len(functions)} 个相似函数")
            for file_path, func_name in functions:
                rel_path = os.path.relpath(file_path, directory)
                print(f"  • {rel_path} - {func_name}")
    else:
        print("✅ 未发现明显的相似函数")

    # 2. 相似导入分析
    print("\n2. 📦 相似导入分析")
    similar_imports = find_similar_imports(directory)

    if similar_imports:
        print(f"发现 {len(similar_imports)} 组相似导入模式:")
        for i, (import_key, files) in enumerate(similar_imports.items(), 1):
            print(f"\n组 {i}: {len(files)} 个文件使用相同导入")
            for file_path in files:
                rel_path = os.path.relpath(file_path, directory)
                print(f"  • {rel_path}")
    else:
        print("✅ 未发现明显的相似导入模式")

    # 3. 相似错误处理分析
    print("\n3. ⚠️ 相似错误处理分析")
    similar_errors = find_similar_error_handling(directory)

    if similar_errors:
        print(f"发现 {len(similar_errors)} 组相似错误处理模式:")
        for i, (pattern_key, locations) in enumerate(similar_errors.items(), 1):
            print(f"\n组 {i}: {len(locations)} 处使用相同错误处理")
            for file_path, line_num in locations:
                rel_path = os.path.relpath(file_path, directory)
                print(f"  • {rel_path}:{line_num}")
    else:
        print("✅ 未发现明显的相似错误处理模式")

    # 生成改进建议
    print("\n💡 代码重复消除建议")
    print("-" * 40)

    suggestions = []

    if similar_functions:
        suggestions.append("• 创建通用工具函数库，提取重复的业务逻辑")
        suggestions.append("• 使用策略模式或工厂模式统一相似函数")

    if similar_imports:
        suggestions.append("• 创建统一的导入管理模块")
        suggestions.append("• 使用__init__.py进行集中导入管理")

    if similar_errors:
        suggestions.append("• 创建统一的错误处理装饰器")
        suggestions.append("• 实现通用的异常处理基类")

    if not any([similar_functions, similar_imports, similar_errors]):
        suggestions.append("• 代码重复情况良好，无需特别处理")

    for suggestion in suggestions:
        print(suggestion)

    # 量化统计
    total_duplicates = len(similar_functions) + len(similar_imports) + len(similar_errors)

    print(f"\n📊 重复分析统计")
    print(f"相似函数组数: {len(similar_functions)}")
    print(f"相似导入组数: {len(similar_imports)}")
    print(f"相似错误处理组数: {len(similar_errors)}")
    print(f"总重复模式数: {total_duplicates}")

    if total_duplicates > 0:
        print(f"\n🎯 优化建议: 重点关注前 {min(3, total_duplicates)} 个重复模式")
    else:
        print("\n🎉 代码重复情况优秀！")

    return {
        'functions': similar_functions,
        'imports': similar_imports,
        'errors': similar_errors,
        'total': total_duplicates
    }


if __name__ == '__main__':
    # 分析health目录
    results = analyze_code_duplicates('src/infrastructure/health')

    print("\n🏆 分析完成")
    print(f"共发现 {results['total']} 个代码重复模式")
