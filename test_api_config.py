import os
import sys

sys.path.append('c:\\PythonProject\\RQA2025')

from src.gateway.web.api import _get_config_file_path, load_data_sources

# 测试配置文件路径
path = _get_config_file_path()
print('API配置文件路径:', path)
print('绝对路径:', os.path.abspath(path))
print('文件是否存在:', os.path.exists(path))

# 测试加载数据源
if os.path.exists(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
        print('文件内容:', content)

# 测试load_data_sources函数
sources = load_data_sources()
print('\nAPI加载的数据源:')
for i, source in enumerate(sources):
    print(f'数据源 {i}: id={source.get("id")}, name={source.get("name")}')
