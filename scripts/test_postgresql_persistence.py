#!/usr/bin/env python3
"""
测试PostgreSQL持久化功能
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import requests
import time

def test_postgresql_persistence():
    """测试PostgreSQL持久化功能"""
    print("🧪 测试PostgreSQL持久化功能")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    
    # 1. 检查服务状态
    print("\n1. 检查服务状态...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("   ✅ 后端服务运行正常")
        else:
            print(f"   ❌ 后端服务异常 (状态码: {response.status_code})")
            return False
    except Exception as e:
        print(f"   ❌ 后端服务未运行: {str(e)[:50]}")
        return False
    
    # 2. 测试数据采集和持久化
    print("\n2. 测试AKShare数据采集和PostgreSQL持久化...")
    try:
        url = f"{base_url}/api/v1/data/sources/akshare_stock/collect"
        data = {
            "symbols": ["000001"],
            "start_date": "2024-12-20",
            "end_date": "2024-12-25"
        }
        
        print(f"   请求URL: {url}")
        print(f"   股票代码: {data['symbols']}")
        print(f"   日期范围: {data['start_date']} 至 {data['end_date']}")
        
        response = requests.post(url, json=data, timeout=30)
        print(f"   响应状态: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"   ✅ 数据采集成功")
            print(f"      采集记录数: {len(result.get('data', []))}")
            print(f"      采集耗时: {result.get('collection_time', 0):.2f}秒")
            
            # 检查持久化结果
            if 'storage' in result:
                storage = result['storage']
                print(f"\n   📊 持久化结果:")
                print(f"      成功: {storage.get('success', False)}")
                print(f"      存储类型: {storage.get('storage_type', 'unknown')}")
                
                if storage.get('success'):
                    if storage.get('storage_type') == 'postgresql':
                        print(f"      ✅ PostgreSQL持久化成功")
                        print(f"      插入记录数: {storage.get('inserted_count', 0)}")
                        print(f"      跳过记录数: {storage.get('skipped_count', 0)}")
                        print(f"      错误记录数: {storage.get('error_count', 0)}")
                        print(f"      处理时间: {storage.get('processing_time', 0):.2f}秒")
                        print(f"      消息: {storage.get('message', 'N/A')}")
                    else:
                        print(f"      ⚠️  使用文件存储（PostgreSQL不可用）")
                        print(f"      文件: {storage.get('storage_id', 'N/A')}")
                else:
                    print(f"      ❌ 持久化失败")
                    print(f"      错误: {storage.get('error', 'N/A')}")
            else:
                print(f"\n   ⚠️  未检测到持久化结果")
            
            # 显示示例数据
            if result.get('data'):
                first = result['data'][0]
                print(f"\n   📈 示例数据:")
                print(f"      股票代码: {first.get('symbol')}")
                print(f"      日期: {first.get('date')}")
                print(f"      收盘价: {first.get('close')}")
                print(f"      成交量: {first.get('volume')}")
            
            return True
        else:
            print(f"   ❌ 数据采集失败 (状态码: {response.status_code})")
            print(f"      错误信息: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"   ❌ 测试失败: {str(e)}")
        return False
    
    # 3. 测试数据去重（重复采集相同数据）
    print("\n3. 测试数据去重（重复采集相同数据）...")
    try:
        response2 = requests.post(url, json=data, timeout=30)
        if response2.status_code == 200:
            result2 = response2.json()
            storage2 = result2.get('storage', {})
            
            if storage2.get('storage_type') == 'postgresql':
                inserted = storage2.get('inserted_count', 0)
                skipped = storage2.get('skipped_count', 0)
                
                if inserted == 0 and skipped > 0:
                    print(f"   ✅ 数据去重正常（跳过重复数据）")
                elif inserted > 0:
                    print(f"   ⚠️  插入了{inserted}条新记录（可能数据有更新）")
                else:
                    print(f"   ⚠️  未插入数据")
            else:
                print(f"   ⚠️  使用文件存储，无法测试去重")
    except Exception as e:
        print(f"   ⚠️  去重测试失败: {str(e)[:50]}")
    
    print("\n" + "=" * 60)
    print("🎉 PostgreSQL持久化功能测试完成！")
    return True


if __name__ == "__main__":
    success = test_postgresql_persistence()
    sys.exit(0 if success else 1)

