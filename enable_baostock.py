#!/usr/bin/env python3
"""启用baostock数据源"""
import requests
import json

print('=== 启用baostock数据源 ===')

# 1. 获取当前数据源配置
print('\n1. 获取baostock数据源当前状态')
print('-' * 60)
try:
    response = requests.get('http://localhost:8000/api/v1/data/sources', timeout=10)
    sources = response.json()
    
    baostock_source = None
    for source in sources.get('data', []):
        if 'baostock' in source.get('id', '').lower():
            baostock_source = source
            break
    
    if baostock_source:
        source_id = baostock_source.get('id')
        print(f"找到数据源: {source_id}")
        print(f"当前状态: 启用={baostock_source.get('is_active', False)}")
        
        # 2. 启用数据源
        print('\n2. 启用baostock数据源')
        print('-' * 60)
        
        # 更新配置，启用数据源
        update_data = {
            "is_active": True,
            "enabled": True
        }
        
        response = requests.put(
            f'http://localhost:8000/api/v1/data/sources/{source_id}',
            json=update_data,
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"✅ 数据源 {source_id} 已启用")
        else:
            print(f"❌ 启用失败: {response.status_code} - {response.text}")
            
        # 3. 验证更新
        print('\n3. 验证更新')
        print('-' * 60)
        response = requests.get('http://localhost:8000/api/v1/data/sources', timeout=10)
        sources = response.json()
        
        for source in sources.get('data', []):
            if 'baostock' in source.get('id', '').lower():
                print(f"更新后状态: 启用={source.get('is_active', False)}")
                if source.get('is_active', False):
                    print("✅ baostock数据源已成功启用！")
                else:
                    print("❌ 启用失败")
                break
    else:
        print("❌ 未找到baostock数据源")
        
except Exception as e:
    print(f"操作失败: {e}")
    import traceback
    traceback.print_exc()

print('\n' + '=' * 60)
print('操作完成')
print('=' * 60)
