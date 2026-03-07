#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理界面启动脚本

启动特征层配置管理界面，提供可视化的配置管理功能
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    """主函数"""
    try:
        print("🚀 启动特征层配置管理界面...")

        # 导入配置管理界面
        from src.features.config_management_interface import ConfigManagementInterface

        # 创建并运行界面
        interface = ConfigManagementInterface()
        interface.run()

    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保已安装所需依赖包")
        return 1
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
