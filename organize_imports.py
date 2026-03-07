"""
统一导入语句的排序和分组规范

按照Python标准将导入语句重新排序：
1. 标准库导入
2. 第三方库导入
3. 本地模块导入

并在不同组之间添加空行分隔。
"""

import os
from pathlib import Path


def categorize_import(import_line: str) -> str:
    """将导入语句分类"""
    line = import_line.strip()

    # 标准库导入
    stdlib_modules = {
        'os', 'sys', 'time', 'datetime', 'json', 'pickle', 'threading', 'multiprocessing',
        'asyncio', 'logging', 'pathlib', 'functools', 'itertools', 'collections', 'enum',
        'abc', 'typing', 'weakref', 'contextlib', 'tempfile', 'shutil', 'glob', 'fnmatch',
        'linecache', 'inspect', 'site', 'warnings', 'contextvars', 'concurrent', 'subprocess',
        'socket', 'ssl', 'urllib', 'http', 'ftplib', 'poplib', 'imaplib', 'smtplib', 'uuid',
        'secrets', 'hashlib', 'hmac', 'base64', 'binascii', 'zlib', 'gzip', 'bz2', 'lzma',
        'zipfile', 'tarfile', 'csv', 'configparser', 'netrc', 'xdrlib', 'plistlib', 'hashlib',
        'sqlite3', 'zlib', 'gzip', 'bz2', 'lzma', 'zipfile', 'tarfile', 'csv', 'calendar',
        'datetime', 'time', 'zoneinfo', 'locale', 'gettext', 'argparse', 'optparse', 'getopt',
        'readline', 'rlcompleter', 'sqlite3', 'zlib', 'codecs', 'unicodedata', 'stringprep',
        're', 'difflib', 'textwrap', 'string', 'binary', 'struct', 'weakref', 'gc', 'inspect',
        'site', 'warnings', 'contextvars', 'concurrent', 'subprocess', 'socket', 'mmap',
        'contextlib', 'concurrent.futures', 'threading', 'multiprocessing', 'queue', 'sched',
        '_thread', 'dummy_thread', 'io', 'codecs', 'unicodedata', 'stringprep', 're', 'math',
        'cmath', 'decimal', 'fractions', 'random', 'statistics', 'datetime', 'calendar',
        'time', 'zoneinfo', 'locale', 'gettext', 'argparse', 'optparse', 'getopt', 'readline'
    }

    # 提取模块名
    if line.startswith('from '):
        module_name = line.split()[1].split('.')[0]
    elif line.startswith('import '):
        module_name = line.split()[1].split('.')[0]
    else:
        return 'unknown'

    # 检查是否是标准库
    if module_name in stdlib_modules:
        return 'stdlib'

    # 检查是否是本地模块 (以src.infrastructure开头)
    if module_name in ['src', 'infrastructure'] or line.startswith('from src.infrastructure'):
        return 'local'

    # 其他都认为是第三方库
    return 'third_party'


def organize_imports_in_file(file_path: Path) -> bool:
    """整理单个文件的导入语句"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 找到导入语句的范围 (通常在文件开头)
        import_lines = []
        import_start = -1
        import_end = -1

        for i, line in enumerate(lines):
            line = line.strip()
            if (line.startswith('from ') or line.startswith('import ')) and not line.startswith('from .'):
                if import_start == -1:
                    import_start = i
                import_end = i
            elif import_start != -1 and line and not line.startswith('#'):
                break

        if import_start == -1:
            return False  # 没有导入语句

        # 提取导入语句
        imports = []
        for i in range(import_start, import_end + 1):
            line = lines[i].rstrip()
            if line.strip() and (line.startswith('from ') or line.startswith('import ')):
                imports.append((i, line))

        if not imports:
            return False

        # 对导入语句进行分类和排序
        stdlib_imports = []
        third_party_imports = []
        local_imports = []

        for line_idx, import_line in imports:
            category = categorize_import(import_line)
            if category == 'stdlib':
                stdlib_imports.append(import_line)
            elif category == 'third_party':
                third_party_imports.append(import_line)
            elif category == 'local':
                local_imports.append(import_line)

        # 对每个类别内部排序
        stdlib_imports.sort()
        third_party_imports.sort()
        local_imports.sort()

        # 构建新的导入语句块
        new_imports = []

        # 标准库导入
        if stdlib_imports:
            new_imports.extend(stdlib_imports)
            new_imports.append('')

        # 第三方库导入
        if third_party_imports:
            new_imports.extend(third_party_imports)
            new_imports.append('')

        # 本地模块导入
        if local_imports:
            new_imports.extend(local_imports)
            new_imports.append('')

        # 替换原有的导入语句
        # 先删除原有导入语句
        for i in range(len(imports) - 1, -1, -1):
            line_idx = imports[i][0]
            del lines[line_idx]

        # 在开始位置插入新的导入语句
        for i, import_line in enumerate(reversed(new_imports)):
            if import_line:  # 跳过空行
                lines.insert(import_start, import_line + '\n')
            else:
                lines.insert(import_start, '\n')

        # 移除多余的空行
        cleaned_lines = []
        prev_empty = False
        for line in lines:
            is_empty = line.strip() == ''
            if is_empty and prev_empty:
                continue  # 跳过多余的空行
            cleaned_lines.append(line)
            prev_empty = is_empty

        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(cleaned_lines)

        return True

    except Exception as e:
        print(f'处理文件 {file_path} 时出错: {e}')
        return False


def organize_all_imports():
    """整理所有文件的导入语句"""
    infra_dir = Path('src/infrastructure')

    print('🔄 开始整理导入语句顺序和分组...')
    print('=' * 40)

    organized_count = 0
    total_files = 0

    for root, dirs, files in os.walk(infra_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                total_files += 1

                if organize_imports_in_file(file_path):
                    organized_count += 1
                    rel_path = str(file_path.relative_to(infra_dir))
                    print(f'✅ 整理完成: {rel_path}')

    print(f'\\n📊 整理结果:')
    print(f'  总文件数: {total_files}')
    print(f'  已整理文件数: {organized_count}')
    print(f'  整理率: {organized_count/total_files*100:.1f}%')


if __name__ == "__main__":
    organize_all_imports()
