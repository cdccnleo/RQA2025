"""
分析Redis文件的结构
"""

from pathlib import Path


def analyze_redis_files():
    redis_dir = Path('src/infrastructure/cache/storage')
    files_info = {}

    for file_name in ['redis_adapter_unified.py', 'redis_cache.py', 'redis_storage.py', 'redis.py']:
        file_path = redis_dir / file_name
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            lines = content.splitlines()
            classes = sum(1 for line in lines if line.strip().startswith('class '))
            functions = sum(1 for line in lines if line.strip().startswith('def '))

            files_info[file_name] = {
                'lines': len(lines),
                'classes': classes,
                'functions': functions
            }

    print('🔍 Redis文件详细分析:')
    print('=' * 30)

    for file_name, info in files_info.items():
        print(f'{file_name}:')
        print(f'  行数: {info["lines"]}')
        print(f'  类数: {info["classes"]}')
        print(f'  函数数: {info["functions"]}')
        print()

    total_lines = sum(info['lines'] for info in files_info.values())
    total_classes = sum(info['classes'] for info in files_info.values())
    total_functions = sum(info['functions'] for info in files_info.values())

    print('总计:')
    print(f'  总行数: {total_lines}')
    print(f'  总类数: {total_classes}')
    print(f'  总函数数: {total_functions}')


if __name__ == "__main__":
    analyze_redis_files()
