#!/usr/bin/env python3
"""
测试API的PostgreSQL持久化功能
"""

import requests
import time
import os

# 设置环境变量
os.environ['DB_HOST'] = 'localhost'
os.environ['DB_PORT'] = '5432'
os.environ['DB_NAME'] = 'rqa2025'
os.environ['DB_USER'] = 'rqa2025'
os.environ['DB_PASSWORD'] = 'rqa2025pass'

def test_api_persistence():
    """测试API持久化"""
    print("🧪 测试API PostgreSQL持久化功能")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    
    # 1. 检查服务
    print("\n1. 检查API服务...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("   ✅ API服务运行正常")
        else:
            print(f"   ❌ API服务异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ API服务不可用: {e}")
        return False
    
    # 2. 测试数据采集和持久化
    print("\n2. 测试数据采集和PostgreSQL持久化...")
    url = f"{base_url}/api/v1/data/sources/akshare_stock/collect"
    data = {
        "symbols": ["000001", "600000"],
        "start_date": "2024-12-20",
        "end_date": "2024-12-25"
    }
    
    print(f"   请求: POST {url}")
    print(f"   参数: {data}")
    
    try:
        response = requests.post(url, json=data, timeout=30)
        print(f"   响应状态: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n   ✅ 数据采集成功")
            print(f"      采集记录数: {len(result.get('data', []))}")
            print(f"      采集耗时: {result.get('collection_time', 0):.2f}秒")
            
            # 检查持久化结果
            if 'storage' in result:
                storage = result['storage']
                print(f"\n   📊 持久化结果:")
                print(f"      成功: {storage.get('success', False)}")
                print(f"      存储类型: {storage.get('storage_type', 'unknown')}")
                
                if storage.get('storage_type') == 'postgresql':
                    print(f"\n      ✅ PostgreSQL持久化成功！")
                    print(f"      插入记录数: {storage.get('inserted_count', 0)}")
                    print(f"      跳过记录数: {storage.get('skipped_count', 0)}")
                    print(f"      错误记录数: {storage.get('error_count', 0)}")
                    print(f"      处理时间: {storage.get('processing_time', 0):.2f}秒")
                    return True
                elif storage.get('storage_type') == 'file':
                    print(f"\n      ⚠️  使用文件存储")
                    print(f"      原因: PostgreSQL连接可能失败")
                    print(f"      文件: {storage.get('storage_id', 'N/A')}")
                    
                    # 检查错误信息
                    if 'error' in storage:
                        print(f"      错误: {storage['error']}")
                    return False
                else:
                    print(f"\n      ⚠️  未知存储类型")
                    return False
            else:
                print(f"\n   ⚠️  未返回持久化结果")
                return False
        else:
            print(f"   ❌ 请求失败: {response.status_code}")
            print(f"      错误: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"   ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_api_persistence()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ API PostgreSQL持久化测试通过！")
    else:
        print("⚠️  API仍使用文件存储，需要检查PostgreSQL连接")
    print("=" * 60)
    
    exit(0 if success else 1)

