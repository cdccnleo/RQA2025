#!/usr/bin/env python3
"""
调试数据源加载问题
"""

import sys
import os
import json

# 设置Python路径
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/src')

def test_load_data_sources():
    """测试数据源加载"""
    print("=== 调试数据源加载 ===")

    # 检查环境变量
    env = os.getenv("RQA_ENV", "development")
    print(f"环境变量 RQA_ENV: {env}")

    # 检查文件路径
    def _get_config_file_path():
        """根据环境获取配置文件路径"""
        env = os.getenv("RQA_ENV", "development").lower()

        if env == "production":
            # 生产环境使用专用目录，避免意外覆盖
            config_file = "data/production/data_sources_config.json"
        elif env == "testing":
            # 测试环境使用测试目录
            config_file = "data/testing/data_sources_config.json"
        else:
            # 开发环境使用标准目录
            config_file = "data/data_sources_config.json"
        return config_file

    config_file = _get_config_file_path()
    print(f"配置文件路径: {config_file}")
    print(f"文件是否存在: {os.path.exists(config_file)}")

    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"文件大小: {len(content)} 字符")
                print(f"文件内容预览: {content[:200]}...")

                config_data = json.loads(content)
                print(f"配置数据类型: {type(config_data)}")
                print(f"配置数据键: {list(config_data.keys()) if isinstance(config_data, dict) else 'N/A'}")

                if isinstance(config_data, dict) and "data_sources" in config_data:
                    data_sources = config_data["data_sources"]
                elif isinstance(config_data, list):
                    data_sources = config_data
                else:
                    data_sources = []

                print(f"数据源数量: {len(data_sources)}")

                # 检查前几个数据源的ID
                for i, source in enumerate(data_sources[:5]):
                    print(f"数据源 {i}: id={repr(source.get('id'))}, name={repr(source.get('name'))}")

                # 查找 akshare_stock_a
                found = None
                for source in data_sources:
                    if source.get('id') == 'akshare_stock_a':
                        found = source
                        break

                if found:
                    print(f"✅ 找到 akshare_stock_a: {found}")
                else:
                    print("❌ 未找到 akshare_stock_a")

                    # 显示所有包含 akshare_stock_a 的数据源
                    akshare_sources = [s for s in data_sources if s.get('id', '').startswith('akshare_stock')]
                    print(f"找到的 akshare_stock 开头的ID: {[s.get('id') for s in akshare_sources]}")

        except Exception as e:
            print(f"读取文件失败: {e}")
    else:
        print(f"配置文件不存在: {config_file}")

if __name__ == "__main__":
    test_load_data_sources()
