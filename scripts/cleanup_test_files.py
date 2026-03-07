#!/usr/bin/env python3
"""
清理测试文件的脚本
移除调试和临时测试文件，保留正式的测试文件
"""

import shutil
from pathlib import Path
from typing import List, Set


def get_files_to_remove() -> Set[str]:
    """获取需要移除的文件列表"""
    files_to_remove = {
        # 调试测试文件
        "test_sklearn_isolated.py",
        "test_sklearn_simple.py",
        "test_sklearn_diagnostic.py",
        "debug_sklearn_pytest.py",
        "check_sklearn_debug.py",
        "debug_pytest_sklearn.py",
        "debug_pytest_env.py",
        "check_sklearn.py",
        "test_simple_debug.py",
        "test_time_check.py",
        "test_time_debug.py",
        "debug_import_issue.py",
        "test_debug_collection.py",
        "test_original_file.py",
        "test_model_evaluator_simple.py",

        # 临时文件
        "test.txt",
        "test_model.pkl",
        "dummy.pkl",
        "test_file_list.txt",
        "testfilelist.txt",
        "dummy_path",
        "export_test",
        "pytest",

        # 日志文件
        "test_app.json.log",
        "model_landing_advanced.log",
        "model_landing.log",
        "rqa2025.log",

        # 缓存和临时目录
        "__pycache__",
        ".pytest_cache",
        ".cache",
        "htmlcov",
        "allure-results",
        ".benchmarks",
        "tmp",
        "temp",
        "venv",
        "MagicMock",
        "test_versions",
        "config_versions",
        "dummy",
        "custom_logs",
        "data_cache",
        "feature_cache",
        "cache",
        "temp",
        "output",
        "models",
        "custom",
        ".trae",
        ".cursor",
        ".idea",
        ".github",
    }
    return files_to_remove


def cleanup_files(root_path: Path, files_to_remove: Set[str]) -> List[str]:
    """清理指定的文件"""
    removed_files = []

    for item in root_path.rglob("*"):
        if item.is_file() and item.name in files_to_remove:
            try:
                item.unlink()
                removed_files.append(str(item))
                print(f"已删除文件: {item}")
            except Exception as e:
                print(f"删除文件失败 {item}: {e}")
        elif item.is_dir() and item.name in files_to_remove:
            try:
                shutil.rmtree(item)
                removed_files.append(str(item))
                print(f"已删除目录: {item}")
            except Exception as e:
                print(f"删除目录失败 {item}: {e}")

    return removed_files


def backup_important_files(root_path: Path):
    """备份重要文件"""
    backup_dir = root_path / "backup_before_cleanup"
    backup_dir.mkdir(exist_ok=True)

    important_files = [
        "requirements.txt",
        "pytest.ini",
        "pyproject.toml",
        "README.md",
        ".gitignore",
        "conftest.py",
    ]

    for file_name in important_files:
        file_path = root_path / file_name
        if file_path.exists():
            backup_path = backup_dir / file_name
            shutil.copy2(file_path, backup_path)
            print(f"已备份: {file_name}")


def main():
    """主函数"""
    root_path = Path.cwd()
    print(f"开始清理项目目录: {root_path}")

    # 备份重要文件
    print("\n1. 备份重要文件...")
    backup_important_files(root_path)

    # 获取需要移除的文件列表
    files_to_remove = get_files_to_remove()

    # 清理文件
    print("\n2. 清理调试和临时文件...")
    removed_files = cleanup_files(root_path, files_to_remove)

    # 报告结果
    print(f"\n清理完成!")
    print(f"共删除了 {len(removed_files)} 个文件/目录")

    if removed_files:
        print("\n删除的文件列表:")
        for file_path in removed_files[:10]:  # 只显示前10个
            print(f"  - {file_path}")
        if len(removed_files) > 10:
            print(f"  ... 还有 {len(removed_files) - 10} 个文件")

    print(f"\n重要文件已备份到: {root_path / 'backup_before_cleanup'}")


if __name__ == "__main__":
    main()
