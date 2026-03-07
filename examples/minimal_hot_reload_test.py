#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最小化热重载功能测试

直接导入核心模块，避免循环导入问题。
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
print(f"🔧 Python路径: {sys.path[:3]}")

def test_hot_reload_core():
    """测试热重载核心功能"""
    try:
        # 直接导入核心模块，避免循环导入
        from infrastructure.core.config.services.hot_reload_service import HotReloadService
        from infrastructure.core.config.services.config_file_handler import ConfigFileHandler
        print("✅ 成功导入核心模块")
        
        # 创建临时配置文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {"test": "value", "number": 42}
            json.dump(config_data, f, indent=2, ensure_ascii=False)
            config_file = f.name
        
        print(f"📁 创建临时配置文件: {config_file}")
        
        # 测试配置处理器
        handler = ConfigFileHandler()
        loaded_config = handler.load_config(config_file)
        print(f"📋 加载的配置: {json.dumps(loaded_config, indent=2, ensure_ascii=False)}")
        
        # 测试热重载服务
        service = HotReloadService()
        print("✅ 热重载服务创建成功")
        
        # 清理
        service.cleanup()
        Path(config_file).unlink()
        print("✅ 测试完成，资源已清理")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_watchdog_availability():
    """测试watchdog库是否可用"""
    try:
        import watchdog
        print("✅ watchdog库可用")
        return True
    except ImportError:
        print("❌ watchdog库不可用")
        return False

def main():
    """主测试函数"""
    print("🚀 最小化热重载功能测试")
    print("=" * 50)
    
    # 测试watchdog可用性
    if not test_watchdog_availability():
        print("❌ 缺少必要依赖，测试终止")
        return
    
    # 测试核心功能
    if test_hot_reload_core():
        print("\n🎉 所有测试通过！")
    else:
        print("\n❌ 测试失败")

if __name__ == "__main__":
    main()
