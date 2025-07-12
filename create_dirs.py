import os
import sys

def create_directory(path):
    try:
        os.makedirs(path, exist_ok=True)
        print(f"成功创建目录: {path}")
        return True
    except Exception as e:
        print(f"创建目录失败: {e}", file=sys.stderr)
        return False

if __name__ == "__main__":
    dirs_to_create = [
        "src/infrastructure/versioning",
        "src/infrastructure/monitoring/redis"
    ]

    for dir_path in dirs_to_create:
        if not create_directory(dir_path):
            sys.exit(1)
