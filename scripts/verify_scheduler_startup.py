#!/usr/bin/env python3
"""
调度器启动验证脚本

验证数据采集调度器是否已成功启动
如果未启动，提供诊断信息和手动启动选项
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

async def verify_scheduler_startup():
    """验证调度器是否已启动"""
    try:
        from src.core.orchestration.business_process.service_scheduler import (
            get_data_collection_scheduler
        )
        
        print("=" * 60)
        print("数据采集调度器启动验证")
        print("=" * 60)
        print(f"验证时间: {datetime.now().isoformat()}")
        print()
        
        # 获取调度器实例
        scheduler = get_data_collection_scheduler()
        
        # 检查运行状态
        is_running = scheduler.is_running()
        
        if is_running:
            print("✅ 调度器运行状态: 正在运行")
            
            # 获取详细状态
            status = scheduler.get_status()
            print("\n详细状态信息:")
            print("-" * 60)
            print(f"  启动路径: {status.get('startup_path', '未知')}")
            print(f"  启动时间: {status.get('startup_time', '未知')}")
            print(f"  启用的数据源数量: {status.get('enabled_sources_count', 0)}")
            print(f"  检查间隔: {status.get('check_interval', 0)} 秒")
            
            return True
        else:
            print("❌ 调度器运行状态: 未运行")
            print("\n诊断信息:")
            print("-" * 60)
            
            # 检查组件初始化状态
            status = scheduler.get_status()
            print(f"  启动路径: {status.get('startup_path', 'None（未启动）')}")
            print(f"  启动时间: {status.get('startup_time', 'None（未启动）')}")
            
            print("\n组件初始化状态:")
            print(f"  数据源管理器: {'✅ 已初始化' if scheduler.data_source_manager else '❌ 未初始化'}")
            print(f"  业务流程编排器: {'✅ 已初始化' if scheduler.orchestrator else '❌ 未初始化'}")
            print(f"  事件总线: {'✅ 已初始化' if scheduler.event_bus else '❌ 未初始化'}")
            
            # 提供手动启动选项
            print("\n建议:")
            print("  1. 检查应用启动日志，查看是否有调度器启动相关的错误")
            print("  2. 检查 lifespan 函数是否已执行")
            print("  3. 检查事件发布是否成功")
            print("  4. 可以尝试手动启动调度器（见下方选项）")
            
            # 询问是否手动启动
            try:
                response = input("\n是否尝试手动启动调度器? (y/n): ")
                if response.lower() == 'y':
                    print("\n正在尝试手动启动调度器...")
                    from src.core.orchestration.business_process.service_scheduler import (
                        start_data_collection_scheduler
                    )
                    success = await start_data_collection_scheduler(startup_path="manual_verification_script")
                    if success:
                        print("✅ 调度器手动启动成功")
                        return True
                    else:
                        print("❌ 调度器手动启动失败")
                        return False
            except (EOFError, KeyboardInterrupt):
                print("\n已取消手动启动")
            
            return False
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("可能的原因:")
        print("  1. 模块路径不正确")
        print("  2. 依赖模块缺失")
        return False
    except Exception as e:
        print(f"❌ 验证过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


async def check_startup_listener():
    """检查启动监听器状态"""
    try:
        from src.core.orchestration.business_process.app_startup_listener import (
            get_app_startup_listener
        )
        
        print("\n" + "=" * 60)
        print("应用启动监听器状态检查")
        print("=" * 60)
        
        listener = get_app_startup_listener()
        print(f"监听器实例: {'✅ 已创建' if listener else '❌ 未创建'}")
        
        if listener:
            print(f"监听器注册状态: {'✅ 已注册' if listener._registered else '❌ 未注册'}")
            print(f"调度器启动标记: {listener._scheduler_started}")
            
            if hasattr(listener, 'event_bus') and listener.event_bus:
                print(f"事件总线实例: ✅ 已初始化 (ID: {id(listener.event_bus)})")
            else:
                print("事件总线实例: ❌ 未初始化")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"⚠️  无法检查启动监听器: {e}")


def main():
    """主函数"""
    print("\n开始验证数据采集调度器启动状态...\n")
    
    # 运行异步验证
    is_running = asyncio.run(verify_scheduler_startup())
    
    # 检查启动监听器
    asyncio.run(check_startup_listener())
    
    # 返回状态码
    if is_running:
        print("\n✅ 验证完成: 调度器正在运行")
        sys.exit(0)
    else:
        print("\n❌ 验证完成: 调度器未运行")
        sys.exit(1)


if __name__ == "__main__":
    main()
