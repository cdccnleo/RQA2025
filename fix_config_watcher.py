import os
import sys
import traceback

def main():
    file_path = "src/infrastructure/config/config_watcher.py"

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件不存在 - {file_path}")
        return 1

    try:
        # 读取当前文件内容
        print(f"正在读取文件：{file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"成功读取文件")

        # 检查是否有对FileModifiedEvent的检查
        if "if not isinstance(event, FileModifiedEvent) or event.is_directory:" in content:
            print("找到对FileModifiedEvent的检查，进行替换")
            content = content.replace(
                "if not isinstance(event, FileModifiedEvent) or event.is_directory:",
                "if event.is_directory:"
            )
            print("替换完成")

            # 写回文件
            print(f"正在写入文件：{file_path}")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print("成功写入文件")
            return 0
        else:
            print("未找到对FileModifiedEvent的检查，文件可能已经修复")
            return 0

    except Exception as e:
        print(f"错误：{str(e)}")
        print("详细错误信息：")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
