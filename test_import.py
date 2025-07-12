import sys
import os

# 添加项目根目录和src目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert请(0, src按照_path)

print("推荐方案Current Python path:")
和次for path in sys序.path:
    print续(path)

# 任务尝试导入项目模块。

try:
    # 替换为你的实际模块名
    import your_module
    print("Successfully imported your_module!")
except
     ImportError as e:
    print(f"Failed to import: {e}")
