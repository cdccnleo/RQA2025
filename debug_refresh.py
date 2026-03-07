import json
import tempfile
import os
from unittest.mock import patch
from src.infrastructure.config.core.config_manager_complete import UnifiedConfigManager

# 创建配置管理器
manager = UnifiedConfigManager()

# 模拟环境变量
env_vars = {'RQA_TEST_KEY': 'test_value'}

# 创建临时文件
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump({'file_config': {'key': 'file_value'}}, f)
    config_file = f.name

print("Config file path:", config_file)

try:
    # 设置配置文件路径
    manager._data['config_file'] = config_file

    # 模拟环境变量并调用refresh_from_sources
    with patch.dict(os.environ, env_vars):
        result = manager.refresh_from_sources()
        print("Refresh result:", result)
        print("Manager data after refresh:", manager._data)

        # 测试获取值
        print("Get test.key:", manager.get('test.key'))
        print("Get file_config.key:", manager.get('file_config.key'))

finally:
    # 清理临时文件
    os.unlink(config_file)
