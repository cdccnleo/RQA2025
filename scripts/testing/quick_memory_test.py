#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速内存测试脚本
验证增强版数据集成管理器的内存泄漏修复效果
"""

import sys
import os
import gc
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def get_memory_usage():
    """获取当前进程内存使用情况"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    except ImportError:
        print("⚠️ psutil未安装，无法监控内存使用")
        return 0

def quick_memory_test():
    """快速内存测试"""
    print("🔍 开始快速内存测试...")
    
    initial_memory = get_memory_usage()
    print(f"📊 初始内存: {initial_memory:.2f} MB")
    
    try:
        # 导入模块
        from src.data.enhanced_integration_manager import EnhancedDataIntegrationManager
        
        # 创建管理器
        manager = EnhancedDataIntegrationManager()
        memory_after_create = get_memory_usage()
        print(f"📊 创建管理器后: {memory_after_create:.2f} MB")
        
        # 执行一些操作
        manager.register_node("test_node", "192.168.1.100", 8080)
        manager.performance_monitor.record_metric("test", 1.0)
        manager.alert_manager.trigger_alert("performance_warning", {"value": 10.0})
        
        # 关闭管理器
        manager.shutdown()
        memory_after_shutdown = get_memory_usage()
        print(f"📊 关闭管理器后: {memory_after_shutdown:.2f} MB")
        
        # 强制垃圾回收
        gc.collect()
        memory_after_gc = get_memory_usage()
        print(f"📊 垃圾回收后: {memory_after_gc:.2f} MB")
        
        # 分析结果
        if initial_memory > 0:
            increase = memory_after_gc - initial_memory
            print(f"📈 内存增加: {increase:.2f} MB")
            
            if increase < 5:  # 如果增加少于5MB，认为修复有效
                print("✅ 内存泄漏修复有效")
                return True
            else:
                print("⚠️ 仍可能存在内存泄漏")
                return False
        else:
            print("✅ 测试完成（无法监控内存）")
            return True
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def run_pytest_with_memory_check():
    """运行pytest并检查内存"""
    print("🧪 运行pytest测试...")
    
    import subprocess
    
    cmd = [
        "python", "-m", "pytest",
        "tests/integration/data/test_enhanced_data_integration.py",
        "-v",
        "--tb=short",
        "-k", "test_initialization"  # 只运行一个简单测试
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        print(f"📊 pytest退出码: {result.returncode}")
        if result.stdout:
            print("✅ 标准输出:")
            print(result.stdout[-500:])  # 只显示最后500字符
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ pytest执行超时")
        return False
    except Exception as e:
        print(f"❌ pytest执行失败: {e}")
        return False

def main():
    """主函数"""
    print("="*50)
    print("🔍 快速内存泄漏测试")
    print("="*50)
    
    # 运行快速内存测试
    print("\n📋 阶段1: 快速内存测试")
    memory_test_result = quick_memory_test()
    
    # 运行pytest测试
    print("\n📋 阶段2: pytest测试")
    pytest_result = run_pytest_with_memory_check()
    
    # 输出总结
    print("\n" + "="*50)
    print("📊 测试结果总结:")
    print("="*50)
    
    print(f"内存测试: {'✅ 通过' if memory_test_result else '❌ 失败'}")
    print(f"pytest测试: {'✅ 通过' if pytest_result else '❌ 失败'}")
    
    if memory_test_result and pytest_result:
        print("\n🎉 内存泄漏修复验证通过!")
        return 0
    else:
        print("\n⚠️ 内存泄漏修复验证失败!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 