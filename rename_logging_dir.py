import os
import shutil
import sys

def main():
    try:
        # 获取绝对路径
        base_dir = os.path.dirname(os.path.abspath(__file__))
        infra_dir = os.path.join(base_dir, "src", "infrastructure")
        
        # 确保infrastructure目录存在
        if not os.path.exists(infra_dir):
            raise FileNotFoundError(f"Infrastructure directory not found: {infra_dir}")
            
        # 创建目标目录
        dst_dir = os.path.join(infra_dir, "custom_logging")
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir, exist_ok=True)
            print(f"Created directory: {dst_dir}")
            
        # 检查源目录
        src_dir = os.path.join(infra_dir, "logging")
        if not os.path.exists(src_dir):
            raise FileNotFoundError(f"Source logging directory not found: {src_dir}")
            
        # 移动所有文件
        for item in os.listdir(src_dir):
            src_item = os.path.join(src_dir, item)
            dst_item = os.path.join(dst_dir, item)
            shutil.move(src_item, dst_item)
            print(f"Moved: {src_item} -> {dst_item}")
            
        # 删除空源目录
        os.rmdir(src_dir)
        print(f"Successfully moved logging to custom_logging")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
