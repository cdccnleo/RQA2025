import os
import json
import tempfile
from src.infrastructure.config.core.config_manager_complete import UnifiedConfigManager

# 创建配置管理器
manager = UnifiedConfigManager()

# 创建临时文件
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump({'file_config': {'key': 'file_value'}}, f)
    config_file = f.name

print("Config file path:", config_file)

# 设置配置文件路径
manager._data['config_file'] = config_file

# 测试文件加载
try:
    with open(config_file, 'r', encoding='utf-8') as f:
        file_config = json.load(f)
    print("File config:", file_config)

    # 将文件配置合并到当前配置中
    manager._data.update(file_config)
    print("Manager data after update:", manager._data)

    # 测试获取值
    print("Get file_config.key:", manager.get('file_config.key'))
except Exception as e:
    print("Error:", e)

# 清理临时文件
os.unlink(config_file)
