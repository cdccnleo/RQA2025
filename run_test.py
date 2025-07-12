import unittest
import sys
import os
import traceback

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath('.'))

try:
    # 尝试导入测试模块
    print("尝试导入测试模块...")
    from tests.data.test_data_manager import TestDataManager
    print("成功导入测试模块")

    # 运行测试
    if __name__ == '__main__':
        print("开始运行测试...")
        # 创建测试套件
        suite = unittest.TestLoader().loadTestsFromTestCase(TestDataManager)
        # 运行测试
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        # 设置退出码
        sys.exit(not result.wasSuccessful())
except Exception as e:
    print(f"发生错误: {e}")
    print("详细错误信息:")
    traceback.print_exc()
    sys.exit(1)
