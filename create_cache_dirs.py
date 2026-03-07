"""
创建缓存模块新目录结构
"""

from pathlib import Path


def create_new_dirs():
    """创建新的目录结构"""
    cache_dir = Path('src/infrastructure/cache')

    new_dirs = [
        'services',
        'monitoring',
        'config',
        'utils'
    ]

    print('🏗️ 创建新的目录结构...')

    for dir_name in new_dirs:
        dir_path = cache_dir / dir_name
        dir_path.mkdir(exist_ok=True)

        # 创建__init__.py文件
        init_file = dir_path / '__init__.py'
        if not init_file.exists():
            init_content = '"""缓存模块子包"""\n'
            init_file.write_text(init_content)
            print(f'✅ 创建目录: {dir_name}/')

    print('✅ 目录结构创建完成')


if __name__ == "__main__":
    create_new_dirs()
