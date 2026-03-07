"""
重构utils/common目录结构
"""

from pathlib import Path
import shutil


def create_subdirs():
    """创建子目录"""
    common_dir = Path('src/infrastructure/utils/common')

    # 定义新的目录结构
    new_structure = {
        'core': ['base_components.py', 'core.py', 'common_components.py', 'factory_components.py', 'helper_components.py'],
        'components': ['util_components.py', 'tool_components.py', 'optimized_components.py'],
        'data': ['data_utils.py', 'data_api.py', 'convert.py', 'unified_query.py'],
        'datetime': ['date_utils.py', 'datetime_parser.py'],
        'filesystem': ['file_system.py', 'file_utils.py'],
        'async': ['async_io_optimizer.py', 'ai_optimization_enhanced.py', 'concurrency_controller.py'],
        'services': ['security_utils.py', 'market_aware_retry.py', 'memory_object_pool.py', 'math_utils.py'],
        'monitoring': ['report_generator.py', 'storage_monitor_plugin.py', 'smart_cache_optimizer.py'],
        'tools': ['migrator.py', 'disaster_tester.py']
    }

    print('🏗️ 创建utils/common的子目录结构...')

    # 创建目录和移动文件
    for subdir, files in new_structure.items():
        subdir_path = common_dir / subdir
        subdir_path.mkdir(exist_ok=True)

        # 创建__init__.py
        init_file = subdir_path / '__init__.py'
        if not init_file.exists():
            init_content = '"""utils/common子模块"""\n'
            init_file.write_text(init_content)

        # 移动文件
        moved_count = 0
        for file_name in files:
            src_path = common_dir / file_name
            dst_path = subdir_path / file_name

            if src_path.exists():
                shutil.move(str(src_path), str(dst_path))
                moved_count += 1

        print(f'✅ {subdir}/: 创建目录，移动 {moved_count} 个文件')

    print('\n✅ utils/common目录重构完成！')
    print('保留在common根目录的文件: interfaces.py, exceptions.py')

    return new_structure


def verify_restructure():
    """验证重构结果"""
    common_dir = Path('src/infrastructure/utils/common')

    print('\n🔍 验证utils/common重构结果...')

    # 检查根目录文件
    root_files = [f for f in common_dir.iterdir() if f.is_file() and f.suffix ==
                  '.py' and not f.name.startswith('__')]
    print(f'根目录保留文件: {len(root_files)} 个')
    for f in sorted(root_files):
        print(f'  - {f.name}')

    # 统计各子目录文件
    print('\n📂 各子目录文件统计:')
    subdirs = [d for d in common_dir.iterdir() if d.is_dir() and not d.name.startswith('__')]
    for subdir in sorted(subdirs, key=lambda x: x.name):
        py_files = [f for f in subdir.iterdir() if f.is_file() and f.suffix ==
                    '.py' and not f.name.startswith('__')]
        print(f'  {subdir.name}: {len(py_files)} 个文件')

    total_files = sum(len([f for f in subdir.iterdir() if f.is_file() and f.suffix ==
                      '.py' and not f.name.startswith('__')]) for subdir in subdirs)
    total_files += len(root_files)

    print(f'\n📊 总结:')
    print(f'  总文件数: {total_files}')
    print(f'  子目录数: {len(subdirs)}')
    print(f'  根目录文件数: {len(root_files)}')


def main():
    """主函数"""
    print("🚀 开始重构utils/common目录结构")
    print("=" * 45)

    try:
        # 创建子目录并移动文件
        create_subdirs()

        # 验证结果
        verify_restructure()

        print("\n✅ utils/common重构成功！")
        print("文件按功能分类到合适的子目录中")

    except Exception as e:
        print(f"\n❌ 重构过程中出现错误: {e}")
        raise


if __name__ == "__main__":
    main()
