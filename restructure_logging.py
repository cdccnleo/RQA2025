"""
重构logging模块目录结构
"""

from pathlib import Path
import shutil


def create_logging_dirs():
    """创建logging模块的新目录结构"""
    logging_dir = Path('src/infrastructure/logging')

    # 创建新的子目录
    new_dirs = [
        'foundation',  # 基础功能
        'data',        # 数据处理
        'config',      # 配置相关
        'security',    # 安全相关
        'system',      # 系统服务
        'distributed',  # 分布式功能
        'utils'        # 工具函数
    ]

    print('🏗️ 为logging模块创建新的目录结构...')

    created_dirs = []
    for dir_name in new_dirs:
        dir_path = logging_dir / dir_name
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)

            # 创建__init__.py文件
            init_file = dir_path / '__init__.py'
            init_content = '"""logging模块子包"""\n'
            init_file.write_text(init_content)

            created_dirs.append(dir_name)
            print(f'✅ 创建目录: {dir_name}/')

    print(f'\n总共创建了 {len(created_dirs)} 个新目录')
    return new_dirs


def reorganize_logging_files():
    """重新组织logging模块的文件"""
    logging_dir = Path('src/infrastructure/logging')

    # 定义文件移动规则
    file_moves = [
        # 基础功能
        ('base.py', 'foundation/base.py'),
        ('exceptions.py', 'foundation/exceptions.py'),
        ('interfaces.py', 'foundation/interfaces.py'),

        # 数据处理
        ('data_consistency.py', 'data/data_consistency.py'),
        ('data_sanitizer.py', 'data/data_sanitizer.py'),
        ('data_sync.py', 'data/data_sync.py'),
        ('data_validation_service.py', 'data/data_validation_service.py'),

        # 配置相关
        ('config_components.py', 'config/config_components.py'),
        ('formatter_components.py', 'config/formatter_components.py'),
        ('logger_components.py', 'config/logger_components.py'),
        ('logging_service_components.py', 'config/logging_service_components.py'),

        # 安全相关
        ('encryption_service.py', 'security/encryption_service.py'),
        ('security_filter.py', 'security/security_filter.py'),
        ('integrity_checker.py', 'security/integrity_checker.py'),

        # 业务逻辑
        ('quant_filter.py', 'business/quant_filter.py'),
        ('regulatory_compliance.py', 'business/regulatory_compliance.py'),
        ('regulatory_reporter.py', 'business/regulatory_reporter.py'),
        ('smart_log_filter.py', 'business/smart_log_filter.py'),

        # 系统服务
        ('chaos_orchestrator.py', 'system/chaos_orchestrator.py'),
        ('circuit_breaker.py', 'system/circuit_breaker.py'),
        ('connection_pool.py', 'system/connection_pool.py'),
        ('deployment_validator.py', 'system/deployment_validator.py'),

        # 分布式功能
        ('distributed_lock.py', 'distributed/distributed_lock.py'),
        ('sync_conflict_manager.py', 'distributed/sync_conflict_manager.py'),
        ('sync_node_manager.py', 'distributed/sync_node_manager.py'),

        # 工具函数
        ('logging_strategy.py', 'utils/logging_strategy.py'),
        ('logging_utils.py', 'utils/logging_utils.py'),
        ('metrics_aggregator.py', 'utils/metrics_aggregator.py'),
        ('priority_queue.py', 'utils/priority_queue.py'),
        ('production_ready.py', 'utils/production_ready.py'),

        # 其他文件
        ('audit.py', 'foundation/audit.py'),
        ('disaster_recovery.py', 'system/disaster_recovery.py'),
    ]

    print('\n📁 开始重新组织logging模块文件...')

    moved_count = 0
    for src_file, dst_path in file_moves:
        src_path = logging_dir / src_file
        dst_full_path = logging_dir / dst_path

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


def verify_restructure():
    """验证重构结果"""
    logging_dir = Path('src/infrastructure/logging')

    print('\n🔍 验证logging模块重构结果...')

    # 检查根目录文件
    root_files = [f for f in logging_dir.iterdir() if f.is_file() and f.suffix ==
                  '.py' and not f.name.startswith('__')]
    print(f'根目录剩余文件: {len(root_files)} 个')

    if root_files:
        print('剩余文件:')
        for f in sorted(root_files, key=lambda x: x.name):
            print(f'  - {f.name}')

    # 统计各子目录文件
    print('\n📂 各子目录文件统计:')
    subdirs = [d for d in logging_dir.iterdir() if d.is_dir() and not d.name.startswith('__')]
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
    print("🚀 开始重构logging模块结构")
    print("=" * 40)

    try:
        # 1. 创建新目录
        create_logging_dirs()

        # 2. 重新组织文件
        moved_count = reorganize_logging_files()

        # 3. 验证结果
        success = verify_restructure()

        if success:
            print("\n✅ logging模块重构成功！")
            print("根目录已清空，所有文件按功能分类")
        else:
            print("\n⚠️  logging模块重构完成，但根目录仍有文件")

    except Exception as e:
        print(f"\n❌ 重构过程中出现错误: {e}")
        raise


if __name__ == "__main__":
    main()
