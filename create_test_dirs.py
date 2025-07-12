import os
import sys

def create_test_structure(base_path):
    dirs = [
        r"tests\unit\infrastructure\config",
        r"tests\unit\infrastructure\config\performance"
    ]

    files = [
        r"tests\unit\infrastructure\config\__init__.py",
        r"tests\unit\infrastructure\config\conftest.py",
        r"tests\unit\infrastructure\config\test_config_manager.py",
        r"tests\unit\infrastructure\config\test_cache_service.py",
        r"tests\unit\infrastructure\config\test_event_service.py",
        r"tests\unit\infrastructure\config\test_version_service.py",
        r"tests\unit\infrastructure\config\performance\test_cache_perf.py",
        r"tests\unit\infrastructure\config\performance\test_lock_contention.py"
    ]

    try:
        # 创建目录
        for dir_path in dirs:
            full_path = os.path.join(base_path, dir_path)
            os.makedirs(full_path, exist_ok=True)
            print(f"创建目录: {full_path}")

        # 创建空文件
        for file_path in files:
            full_path = os.path.join(base_path, file_path)
            with open(full_path, 'w') as f:
                pass
            print(f"创建文件: {full_path}")

        return True
    except Exception as e:
        print(f"创建失败: {str(e)}")
        return False

if __name__ == "__main__":
    base_path = r"C:\PythonProject\RQA2025"
    success = create_test_structure(base_path)
    sys.exit(0 if success else 1)
