import os
import sys
import pytest

def main():
    # 添加项目根目录到Python路径
    project_root = os.path.abspath(os.path.dirname(__file__))
    sys.path.insert(0, project_root)

    # 运行测试
    pytest.main(['tests/unit/infrastructure/test_auto_recovery.py', '-v'])

if __name__ == '__main__':
    main()
