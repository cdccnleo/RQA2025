#!/usr/bin/env python3
"""
测试配置加载功能
"""

from src.gateway.web.config_manager import load_data_sources

if __name__ == "__main__":
    print("开始测试配置加载...")
    sources = load_data_sources()
    print(f"加载的数据源数量: {len(sources)}")
    print("\n数据源列表:")
    for i, source in enumerate(sources):
        print(f"{i+1}. {source.get('name')} (ID: {source.get('id')})")
    print("\n测试完成！")
