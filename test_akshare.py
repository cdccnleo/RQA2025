#!/usr/bin/env python3
"""
测试 AKShare 服务可用性的脚本
"""

import sys
import time
from datetime import datetime


def test_akshare_function():
    """
    测试 AKShare 函数的可用性
    """
    print(f"[{datetime.now()}] 开始测试 AKShare 服务...")
    
    try:
        import akshare
        print(f"[{datetime.now()}] AKShare 版本: {akshare.__version__}")
        
        # 测试 stock_zh_a_spot 函数
        print(f"[{datetime.now()}] 测试 stock_zh_a_spot 函数...")
        start_time = time.time()
        data = akshare.stock_zh_a_spot()
        end_time = time.time()
        
        print(f"[{datetime.now()}] 测试成功!")
        print(f"[{datetime.now()}] 数据形状: {data.shape}")
        print(f"[{datetime.now()}] 耗时: {end_time - start_time:.2f} 秒")
        print(f"[{datetime.now()}] 前5行数据:")
        print(data.head())
        
        return True
        
    except ImportError as e:
        print(f"[{datetime.now()}] 错误: 无法导入 AKShare: {e}")
        return False
    except Exception as e:
        print(f"[{datetime.now()}] 错误: AKShare 调用失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_akshare_stock_basic():
    """
    测试 akshare_stock_basic 数据源的配置
    """
    print(f"[{datetime.now()}] 测试 akshare_stock_basic 数据源配置...")
    
    try:
        # 读取配置文件
        import json
        with open('data/data_sources_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 查找 akshare_stock_basic 数据源
        akshare_config = None
        for source in config:
            if source.get('id') == 'akshare_stock_basic':
                akshare_config = source
                break
        
        if akshare_config:
            print(f"[{datetime.now()}] 找到 akshare_stock_basic 数据源配置:")
            print(f"[{datetime.now()}] 名称: {akshare_config.get('name')}")
            print(f"[{datetime.now()}] URL: {akshare_config.get('url')}")
            print(f"[{datetime.now()}] 函数: {akshare_config.get('config', {}).get('akshare_function')}")
            return True
        else:
            print(f"[{datetime.now()}] 错误: 未找到 akshare_stock_basic 数据源配置")
            return False
            
    except Exception as e:
        print(f"[{datetime.now()}] 错误: 读取配置文件失败: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("开始测试 AKShare 服务和配置")
    print("=" * 60)
    
    # 测试配置
    config_ok = test_akshare_stock_basic()
    print()
    
    # 测试函数调用
    function_ok = test_akshare_function()
    print()
    
    print("=" * 60)
    print("测试结果汇总:")
    print(f"配置检查: {'成功' if config_ok else '失败'}")
    print(f"函数调用: {'成功' if function_ok else '失败'}")
    print("=" * 60)
    
    if config_ok and function_ok:
        print("🎉 所有测试通过! AKShare 服务正常运行。")
        sys.exit(0)
    else:
        print("❌ 测试失败! AKShare 服务存在问题。")
        sys.exit(1)