"""
重构error模块目录结构
"""

from pathlib import Path
import shutil


def create_error_dirs():
    """创建error模块的新目录结构"""
    error_dir = Path('src/infrastructure/error')

    # 创建新的子目录
    new_dirs = [
        'foundation',  # 基础功能
        'recovery',    # 恢复机制
        'testing',     # 测试相关
        'handlers',    # 处理程序 (已存在)
        'storage',     # 存储相关
        'security',    # 安全相关
        'components',  # 组件
        'utils'        # 工具函数
    ]

    print('🏗️ 为error模块创建新的目录结构...')

    created_dirs = []
    for dir_name in new_dirs:
        dir_path = error_dir / dir_name
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)

            # 创建__init__.py文件
            init_file = dir_path / '__init__.py'
            init_content = '"""error模块子包"""\n'
            init_file.write_text(init_content)

            created_dirs.append(dir_name)
            print(f'✅ 创建目录: {dir_name}/')

    print(f'\n总共创建了 {len(created_dirs)} 个新目录')
    return new_dirs


def reorganize_error_files():
    """重新组织error模块的文件"""
    error_dir = Path('src/infrastructure/error')

    # 定义文件移动规则
    file_moves = [
        # 基础功能
        ('interfaces.py', 'foundation/interfaces.py'),

        # 恢复机制
        ('auto_recovery.py', 'recovery/auto_recovery.py'),
        ('disaster_recovery.py', 'recovery/disaster_recovery.py'),
        ('recovery_components.py', 'recovery/recovery_components.py'),

        # 测试相关
        ('automated_test_runner.py', 'testing/automated_test_runner.py'),
        ('test_reporting_system.py', 'testing/test_reporting_system.py'),

        # 处理程序 (移动到已存在的handlers目录)
        ('boundary_handler.py', 'handlers/boundary_handler.py'),
        ('enhanced_global_exception_handler.py', 'handlers/enhanced_global_exception_handler.py'),
        ('retry_handler.py', 'handlers/retry_handler.py'),
        ('retry_policy.py', 'handlers/retry_policy.py'),

        # 存储相关
        ('kafka_storage.py', 'storage/kafka_storage.py'),
        ('file_utils.py', 'storage/file_utils.py'),
        ('yaml_loader.py', 'storage/yaml_loader.py'),

        # 安全相关
        ('security.py', 'security/security.py'),

        # 组件
        ('error_components.py', 'components/error_components.py'),
        ('exception_components.py', 'components/exception_components.py'),
        ('fallback_components.py', 'components/fallback_components.py'),
        ('container.py', 'components/container.py'),
        ('comprehensive_error_plugin.py', 'components/comprehensive_error_plugin.py'),

        # 工具函数
        ('error_codes_utils.py', 'utils/error_codes_utils.py'),
        ('exception_utils.py', 'utils/exception_utils.py'),
        ('lock.py', 'utils/lock.py'),
        ('result.py', 'utils/result.py'),
        ('integration.py', 'utils/integration.py'),

        # 其他系统功能
        ('chaos_engine.py', 'utils/chaos_engine.py'),
        ('circuit_breaker.py', 'utils/circuit_breaker.py'),
        ('error_logger.py', 'utils/error_logger.py'),
    ]

    print('\n📁 开始重新组织error模块文件...')

    moved_count = 0
    for src_file, dst_path in file_moves:
        src_path = error_dir / src_file
        dst_full_path = error_dir / dst_path

        if src_path.exists():
            # 确保目标目录存在
            dst_full_path.parent.mkdir(parents=True, exist_ok=True)

            # 移动文件
            shutil.move(str(src_path), str(dst_full_path))
            print(f'✅ 移动: {src_file} -> {dst_path}')
            moved_count += 1
        else:
            print(f'⚠️  源文件不存在: {src_file}')

    print(f'\n✅ 文件重组完成，共移动 {moved_count} 个文件')
    return moved_count


def verify_error_restructure():
    """验证error模块重构结果"""
    error_dir = Path('src/infrastructure/error')

    print('\n🔍 验证error模块重构结果...')

    # 检查根目录文件
    root_files = [f for f in error_dir.iterdir() if f.is_file() and f.suffix ==
                  '.py' and not f.name.startswith('__')]
    print(f'根目录剩余文件: {len(root_files)} 个')

    if root_files:
        print('剩余文件:')
        for f in sorted(root_files, key=lambda x: x.name):
            print(f'  - {f.name}')

    # 统计各子目录文件
    print('\n📂 各子目录文件统计:')
    subdirs = [d for d in error_dir.iterdir() if d.is_dir() and not d.name.startswith('__')]
    for subdir in sorted(subdirs, key=lambda x: x.name):
        py_files = [f for f in subdir.iterdir() if f.is_file() and f.suffix ==
                    '.py' and not f.name.startswith('__')]
        print(f'  {subdir.name}: {len(py_files)} 个文件')

    total_files = sum(len([f for f in subdir.iterdir() if f.is_file() and f.suffix ==
                      '.py' and not f.name.startswith('__')]) for subdir in subdirs)
    total_files += len(root_files)

    print(f'\n📊 总结:')
    print(f'  总文件数: {total_files}')
    print(f'  根目录文件数: {len(root_files)}')
    print(f'  子目录数: {len(subdirs)}')

    return len(root_files) == 0


def main():
    """主函数"""
    print("🚀 开始重构error模块结构")
    print("=" * 40)

    try:
        # 1. 创建新目录
        create_error_dirs()

        # 2. 重新组织文件
        moved_count = reorganize_error_files()

        # 3. 验证结果
        success = verify_error_restructure()

        if success:
            print("\n✅ error模块重构成功！")
            print("根目录已清空，所有文件按功能分类")
        else:
            print("\n⚠️  error模块重构完成，但根目录仍有文件")

    except Exception as e:
        print(f"\n❌ 重构过程中出现错误: {e}")
        raise


if __name__ == "__main__":
    main()
