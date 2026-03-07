#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接热重载功能测试

绕过__init__.py的循环导入，直接测试核心功能。
"""

import sys
import os
import time
import json
import tempfile
from pathlib import Path

# 添加src目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, src_dir)
print(f"🔧 添加路径: {src_dir}")

def test_watchdog_availability():
    """测试watchdog库是否可用"""
    try:
        import watchdog
        print("✅ watchdog库可用")
        return True
    except ImportError:
        print("❌ watchdog库不可用")
        return False

def test_config_file_handler():
    """测试配置文件处理器"""
    try:
        # 直接导入，避免通过__init__.py
        sys.path.insert(0, os.path.join(src_dir, 'infrastructure', 'core', 'config', 'services'))
        from hot_reload_service import ConfigFileHandler
        print("✅ 成功导入ConfigFileHandler")
        
        # 创建临时配置文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {"test": "value", "number": 42}
            json.dump(config_data, f, indent=2, ensure_ascii=False)
            config_file = f.name
        
        print(f"📁 创建临时配置文件: {config_file}")
        
        # 测试配置处理器
        def dummy_callback(file_path, config):
            print(f"📋 回调触发: {file_path}")
        
        handler = ConfigFileHandler(dummy_callback)
        print("✅ ConfigFileHandler创建成功")
        
        # 清理
        Path(config_file).unlink()
        print("✅ ConfigFileHandler测试完成")
        
        return True
        
    except Exception as e:
        print(f"❌ ConfigFileHandler测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hot_reload_service():
    """测试热重载服务"""
    try:
        # 直接导入，避免通过__init__.py
        sys.path.insert(0, os.path.join(src_dir, 'infrastructure', 'core', 'config', 'services'))
        from hot_reload_service import HotReloadService
        print("✅ 成功导入HotReloadService")
        
        # 测试热重载服务
        service = HotReloadService()
        print("✅ 热重载服务创建成功")
        
        # 清理
        service.stop()
        print("✅ HotReloadService测试完成")
        
        return True
        
    except Exception as e:
        print(f"❌ HotReloadService测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 直接热重载功能测试")
    print("=" * 50)
    
    # 测试watchdog可用性
    if not test_watchdog_availability():
        print("❌ 缺少必要依赖，测试终止")
        return
    
    # 测试各个组件
    results = []
    results.append(("ConfigFileHandler", test_config_file_handler()))
    results.append(("HotReloadService", test_hot_reload_service()))
    
    # 显示结果
    print("\n📊 测试结果:")
    print("-" * 30)
    for component, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{component}: {status}")
    
    # 总结
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\n🎯 总体结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！")
    else:
        print("⚠️ 部分测试失败")

if __name__ == "__main__":
    main()
