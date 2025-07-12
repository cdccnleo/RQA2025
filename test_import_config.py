import sys
import traceback

try:
    from src.infrastructure.config.config_manager import ConfigManager
    print("成功导入ConfigManager")
    print("Python路径:", sys.path)
except Exception as e:
    print("导入失败:")
    traceback.print_exc()
    print("\nPython路径:", sys.path)
