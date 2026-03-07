import sys
import traceback

print("开始测试导入 postgresql_persistence 模块...")
try:
    import src.gateway.web.postgresql_persistence
    print("模块导入成功")
except Exception as e:
    print(f"导入失败: {e}")
    traceback.print_exc()
    sys.exit(1)
