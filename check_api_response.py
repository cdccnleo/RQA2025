#!/usr/bin/env python3
"""
检查API响应
"""

import requests

def check_api():
    print("检查API响应...")

    try:
        response = requests.get('http://localhost:8000/api/v1/data/sources')
        if response.status_code == 200:
            data = response.json()
            print("✅ API调用成功")

            for source in data.get('data_sources', []):
                if source['name'] == '财联社':
                    print(f"财联社ID: {repr(source['id'])}")
                    print(f"财联社详细信息: {source}")
                    break
        else:
            print(f"❌ API调用失败: HTTP {response.status_code}")

    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    check_api()
