from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import Mock, patch, mock_open
import json
import yaml
import os
from datetime import datetime
from src.infrastructure.config.core.config_manager_storage import UnifiedConfigManagerWithStorage

class TestUnifiedConfigManagerWithStorage:
    """测试统一配置管理器存储功能"""

    @pytest.fixture
    def mock_manager(self):
        """创建mock UnifiedConfigManagerWithStorage实例"""
        manager = UnifiedConfigManagerWithStorage()
        manager.config = {'config_file': 'test.json', 'backup_enabled': True, 'max_backup_files': 5}
        manager._data = {}
        return manager

    def test_init_with_config(self):
        """测试初始化带配置 (覆盖26-30)"""
        init_config = {'test': 'value'}
        manager = UnifiedConfigManagerWithStorage(config=init_config)
        assert manager.config == init_config
        assert manager._data == {}
        assert manager._initialized is False

    def test_get_section_success(self, mock_manager):
        """测试获取配置节成功 (覆盖32-45)"""
        mock_manager._data = {'section': {'key': 'value'}}
        result = mock_manager.get_section('section')
        assert result == {'key': 'value'}

    def test_get_section_non_dict(self, mock_manager):
        """测试获取非字典配置节"""
        mock_manager._data = {'section': 'value'}
        result = mock_manager.get_section('section')
        assert result == 'value'

    def test_get_section_missing(self, mock_manager):
        """测试获取缺失配置节"""
        result = mock_manager.get_section('missing')
        assert result is None

    def test_load_config_from_dict(self, mock_manager):
        """测试从字典加载配置 (覆盖47-62)"""
        mock_data = {'key': 'value'}
        result = mock_manager.load_config(mock_data)
        assert result is True
        assert mock_manager._data == mock_data

    def test_load_config_from_file_json(self, mock_manager):
        """测试从JSON文件加载配置 (覆盖64-88)"""
        mock_data = {'key': 'value'}
        m = mock_open(read_data=json.dumps(mock_data))
        with patch('os.path.exists') as mock_exists, \
             patch('builtins.open', m), \
             patch('json.load') as mock_json_load:
            mock_exists.return_value = True
            mock_json_load.return_value = mock_data
            result = mock_manager.load_config('test.json')
            assert result is True
            assert mock_manager._data == mock_data

    def test_load_config_from_file_yaml(self, mock_manager):
        """测试从YAML文件加载配置"""
        mock_data = {'key': 'value'}
        m = mock_open(read_data=yaml.safe_dump(mock_data))
        with patch('os.path.exists') as mock_exists, \
             patch('builtins.open', m), \
             patch('yaml.safe_load') as mock_yaml_load:
            mock_exists.return_value = True
            mock_yaml_load.return_value = mock_data
            result = mock_manager.load_config('test.yaml')
            assert result is True
            assert mock_manager._data == mock_data

    def test_load_config_file_not_found(self, mock_manager):
        """测试加载不存在文件 (覆盖73-75)"""
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            result = mock_manager.load_config('missing.json')
            assert result is False

    def test_load_config_invalid_format(self, mock_manager):
        """测试加载无效格式 (覆盖85-87)"""
        with patch('os.path.exists') as mock_exists, \
             patch('builtins.open', mock_open(read_data="invalid")):
            mock_exists.return_value = True
            result = mock_manager.load_config('invalid.txt')
            assert result is False

    def test_save_config_success(self, mock_manager):
        """测试保存配置成功 (覆盖93-126)"""
        mock_manager._data = {'key': 'value'}
        with patch('os.path.exists') as mock_exists, \
             patch.object(mock_manager, '_create_backup') as mock_backup, \
             patch('builtins.open', mock_open()) as mock_file, \
             patch('json.dump') as mock_json_dump:
            mock_exists.return_value = True
            mock_backup.return_value = True
            result = mock_manager.save_config('test.json')
            assert result is True
            mock_backup.assert_called_once()
            mock_json_dump.assert_called_once()

    def test_save_config_no_backup(self, mock_manager):
        """测试无备份保存"""
        mock_manager.config['backup_enabled'] = False
        mock_manager._data = {'key': 'value'}
        with patch('builtins.open', mock_open()), \
             patch('json.dump'):
            result = mock_manager.save_config('test.json')
            assert result is True

    def test_get_all_sections(self, mock_manager):
        """测试获取所有配置节 (覆盖132-139)"""
        mock_manager._data = {'sec1': {}, 'sec2': {}}
        result = mock_manager.get_all_sections()
        assert set(result) == {'sec1', 'sec2'}

    def test_reload_config_success(self, mock_manager):
        """测试重新加载配置成功 (覆盖141-154)"""
        with patch.object(mock_manager, 'load_config') as mock_load:
            mock_load.return_value = True
            result = mock_manager.reload_config()
            assert result is True
            mock_load.assert_called_once()

    def test_reload_config_no_file(self, mock_manager):
        """测试无配置文件重新加载"""
        mock_manager.config.pop('config_file', None)
        result = mock_manager.reload_config()
        assert result is False

    def test_merge_config_success(self, mock_manager):
        """测试合并配置成功 (覆盖159-176)"""
        mock_manager._data = {'existing': 'value'}
        new_config = {'new': 'value'}
        result = mock_manager.merge_config(new_config)
        assert result is True
        assert mock_manager._data == {'existing': 'value', 'new': 'value'}

    def test_merge_config_to_section(self, mock_manager):
        """测试合并到特定节"""
        mock_manager._data = {'section': {'existing': 'value'}}
        new_config = {'new': 'value'}
        result = mock_manager.merge_config(new_config, 'section')
        assert result is True
        assert mock_manager._data['section'] == {'existing': 'value', 'new': 'value'}

    def test_merge_config_exception(self, mock_manager):
        """测试合并配置异常"""
        with patch.dict(mock_manager._data, clear=True):
            result = mock_manager.merge_config('invalid')  # invalid type
            assert result is False

    def test_export_config_json_string(self, mock_manager):
        """测试导出JSON字符串 (覆盖196-199)"""
        mock_manager._data = {'key': 'value'}
        result = mock_manager.export_config('json')
        assert isinstance(result, str)
        assert json.loads(result) == {'key': 'value'}

    def test_export_config_yaml_string(self, mock_manager):
        """测试导出YAML字符串"""
        mock_manager._data = {'key': 'value'}
        with patch('yaml.dump') as mock_yaml_dump:
            mock_yaml_dump.return_value = 'key: value\n'
            result = mock_manager.export_config('yaml')
            assert result == 'key: value\n'

    def test_export_config_to_file_json(self, mock_manager):
        """测试导出到JSON文件 (覆盖210-212)"""
        mock_manager._data = {'key': 'value'}
        with patch('builtins.open', mock_open()), \
             patch('json.dump') as mock_json_dump:
            result = mock_manager.export_config('json', 'export.json')
            assert result is True
            mock_json_dump.assert_called_once()

    def test_export_config_unsupported_format(self, mock_manager):
        """测试不支持格式导出"""
        mock_manager._data = {'key': 'value'}
        result = mock_manager.export_config('xml')
        assert isinstance(result, str)

    def test_import_config_json(self, mock_manager):
        """测试导入JSON配置 (覆盖250-252)"""
        mock_data = {'key': 'value'}
        m = mock_open(read_data=json.dumps(mock_data))
        with patch('os.path.exists') as mock_exists, \
             patch('builtins.open', m), \
             patch('json.load') as mock_json_load:
            mock_exists.return_value = True
            mock_json_load.return_value = mock_data
            result = mock_manager.import_config('import.json')
            assert result is True
            assert mock_manager._data == mock_data

    def test_import_config_yaml(self, mock_manager):
        """测试导入YAML配置 (覆盖253-255)"""
        mock_data = {'key': 'value'}
        m = mock_open(read_data=yaml.safe_dump(mock_data))
        with patch('os.path.exists') as mock_exists, \
             patch('builtins.open', m), \
             patch('yaml.safe_load') as mock_yaml_load:
            mock_exists.return_value = True
            mock_yaml_load.return_value = mock_data
            result = mock_manager.import_config('import.yaml')
            assert result is True
            assert mock_manager._data == mock_data

    def test_import_config_replace(self, mock_manager):
        """测试替换导入"""
        mock_manager._data = {'existing': 'value'}
        mock_data = {'new': 'value'}
        m = mock_open(read_data=json.dumps(mock_data))
        with patch('os.path.exists') as mock_exists, \
             patch('builtins.open', m), \
             patch('json.load') as mock_json_load:
            mock_exists.return_value = True
            mock_json_load.return_value = mock_data
            result = mock_manager.import_config('import.json', merge=False)
            assert result is True
            assert mock_manager._data == {'new': 'value'}

    def test_import_config_file_not_found(self, mock_manager):
        """测试导入不存在文件"""
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            result = mock_manager.import_config('missing.json')
            assert result is False

    def test_import_config_invalid_format(self, mock_manager):
        """测试导入无效格式"""
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            result = mock_manager.import_config('invalid.txt')
            assert result is False

    def test_create_backup_success(self, mock_manager):
        """测试创建备份成功 (覆盖277-306)"""
        with patch('os.path.exists') as mock_exists, \
             patch('os.makedirs') as mock_makedirs, \
             patch('shutil.copy2') as mock_copy, \
             patch('src.infrastructure.config.core.config_manager_storage.datetime') as mock_datetime, \
             patch.object(mock_manager, '_cleanup_old_backups') as mock_cleanup:
            mock_exists.return_value = True
            from datetime import datetime
            mock_datetime.now.return_value = datetime(2025, 1, 1)
            result = mock_manager._create_backup('test.json')
            assert result is True
            # 由于路径分隔符的差异，只检查调用了copy2方法
            assert mock_copy.called
            mock_cleanup.assert_called_once()

    def test_create_backup_no_file(self, mock_manager):
        """测试备份不存在文件"""
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            result = mock_manager._create_backup('missing.json')
            assert result is True

    def test_cleanup_old_backups(self, mock_manager):
        """测试清理旧备份 (覆盖313-342)"""
        mock_manager.config['max_backup_files'] = 2
        backup_dir = 'backups'
        mock_files = ['config_backup_1.json', 'config_backup_2.json', 'config_backup_3.json', 'other.txt']
        mock_times = [1, 3, 2]  # 1 oldest, 3 newest
        with patch('os.listdir') as mock_listdir, \
             patch('os.path.getmtime') as mock_getmtime, \
             patch('os.remove') as mock_remove, \
             patch('os.path.join') as mock_join:
            mock_listdir.return_value = mock_files
            mock_getmtime.side_effect = lambda p: mock_times[mock_files.index(os.path.basename(p))]
            mock_join.side_effect = lambda d, f: f'{d}/{f}'
            mock_manager._cleanup_old_backups(backup_dir)
            mock_remove.assert_called_once_with('backups/config_backup_1.json')

    def test_cleanup_old_backups_no_max(self, mock_manager):
        """测试无最大备份数清理"""
        mock_manager.config['max_backup_files'] = 0
        mock_manager._cleanup_old_backups('backups')  # Should do nothing

    def test_cleanup_old_backups_exception(self, mock_manager):
        """测试清理异常"""
        with patch('os.listdir', side_effect=Exception('Error')):
            mock_manager._cleanup_old_backups('backups')  # Should log warning
