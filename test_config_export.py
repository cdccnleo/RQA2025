#!/usr/bin/env python3
"""
测试配置导出功能
"""

from src.gateway.web.data_source_config_manager import get_data_source_config_manager

if __name__ == "__main__":
    print("开始测试配置导出...")
    manager = get_data_source_config_manager()
    if manager:
        config = manager.export_config()
        print(f"导出的配置包含 {len(config.get('data_sources', []))} 个数据源")
        print("\n导出的数据源列表:")
        for i, source in enumerate(config.get('data_sources', [])):
            print(f"{i+1}. {source.get('name')} (ID: {source.get('id')})")
        print("\n配置导出测试完成！")
    else:
        print("无法获取配置管理器实例")
