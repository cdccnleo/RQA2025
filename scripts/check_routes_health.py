#!/usr/bin/env python3
"""
路由健康检查脚本
可用于CI/CD流程中的自动化检查
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gateway.web.api import app
from src.gateway.web.route_health_check import RouteHealthChecker


def main():
    """主函数"""
    print("=" * 80)
    print("  路由健康检查")
    print("=" * 80)
    print()
    
    try:
        # 使用已创建的应用实例
        app_instance = app
        
        # 执行健康检查
        checker = RouteHealthChecker(app_instance)
        results = checker.check_routes()
        
        # 打印详细报告
        checker.print_health_report()
        
        # 返回退出码
        if results["health_status"] == "healthy":
            print("✅ 路由健康检查通过")
            return 0
        else:
            print("❌ 路由健康检查失败")
            return 1
            
    except Exception as e:
        print(f"❌ 路由健康检查执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

