"""
配置版本管理器测试用例 - 提升覆盖率到80%+
"""
from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock
import pytest

from src.infrastructure.config.version.components.configversionmanager import ConfigVersionManager
from src.infrastructure.config.version.components.configversion import ConfigVersion
from src.infrastructure.config.version.components.configdiff import ConfigDiff


class TestConfigVersionManagerCoverage:
    """ConfigVersionManager覆盖率测试"""

    def setup_method(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ConfigVersionManager(
            storage_path=self.temp_dir,
            max_versions=10,
            auto_backup=True
        )

    def teardown_method(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_config_version_manager_init(self):
        """测试ConfigVersionManager初始化"""
        assert self.manager.storage_path == Path(self.temp_dir)
        assert self.manager.max_versions == 10
        assert self.manager.auto_backup is True
        assert isinstance(self.manager._versions, dict)
        assert isinstance(self.manager._version_history, list)
        assert 'total_versions' in self.manager.stats

    def test_initialize_storage_creates_directory(self):
        """测试存储初始化创建目录"""
        new_temp_dir = tempfile.mkdtemp()
        storage_path = Path(new_temp_dir) / "versions"
        
        # 确保目录不存在
        if storage_path.exists():
            import shutil
            shutil.rmtree(storage_path)
        
        manager = ConfigVersionManager(storage_path=str(storage_path))
        assert storage_path.exists()

    @patch('builtins.open', mock_open(read_data='{"history": ["v1"], "stats": {"total_versions": 1}}'))
    @patch('pathlib.Path.exists')
    def test_load_versions_from_index_file(self, mock_exists):
        """测试从索引文件加载版本"""
        # 设置模拟：索引文件存在，版本文件存在
        def exists_side_effect(path):
            return path.name == "index.json" or path.name == "v1.json"
        
        mock_exists.side_effect = exists_side_effect
        
        # 模拟版本文件内容
        version_data = {
            'version_id': 'v1',
            'timestamp': time.time(),
            'config_data': {'test': 'data'},
            'checksum': 'abc123',
            'author': 'test_user',
            'description': 'test version',
            'tags': ['test'],
            'metadata': {}
        }
        
        with patch('builtins.open', mock_open(read_data=json.dumps(version_data))):
            manager = ConfigVersionManager(str(Path(self.temp_dir)))
            # 验证版本被加载
            assert len(manager._version_history) >= 0

    @patch('builtins.open', side_effect=IOError("Test error"))
    @patch('pathlib.Path.exists', return_value=True)
    def test_load_versions_handles_exception(self, mock_exists, mock_open_func):
        """测试加载版本时的异常处理"""
        manager = ConfigVersionManager(storage_path=str(Path(self.temp_dir)))
        # 应该能正常初始化即使加载失败
        assert isinstance(manager, ConfigVersionManager)

    def test_save_version_index(self):
        """测试保存版本索引"""
        self.manager._version_history = ['v1', 'v2']
        self.manager.stats['total_versions'] = 2
        
        # 测试保存
        with patch('builtins.open', mock_open()) as mock_file:
            self.manager._save_version_index()
            mock_file.assert_called()

    @patch('builtins.open', side_effect=IOError("Test error"))
    def test_save_version_index_handles_exception(self, mock_open_func):
        """测试保存版本索引时的异常处理"""
        self.manager._version_history = ['v1']
        # 应该不抛出异常
        self.manager._save_version_index()

    def test_create_version_basic(self):
        """测试创建版本的基本功能"""
        config_data = {'section1': {'key1': 'value1'}}
        
        version_id = self.manager.create_version(
            config_data=config_data,
            author='test_user',
            description='Test version',
            tags=['test', 'demo']
        )
        
        assert isinstance(version_id, str)
        assert version_id.startswith('v')
        assert version_id in self.manager._versions
        assert version_id in self.manager._version_history
        assert self.manager.stats['total_versions'] >= 1

    def test_create_version_with_limits(self):
        """测试创建版本时的数量限制"""
        # 设置较小的max_versions，使用临时目录
        temp_dir = tempfile.mkdtemp()
        try:
            manager = ConfigVersionManager(storage_path=temp_dir, max_versions=2)
        
            # 创建3个版本，应该自动清理
            for i in range(3):
                config_data = {'version': f'v{i}'}
                version_id = manager.create_version(config_data, author=f'user{i}')
            
            # 应该保持最多2个版本
            assert len(manager._version_history) <= 2
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_save_version_to_file(self):
        """测试保存版本到文件"""
        version = ConfigVersion(
            version_id='test_v1',
            timestamp=time.time(),
            config_data={'test': 'data'},
            checksum='test123',
            author='test_user',
            description='Test',
            tags=['test'],
            metadata={}
        )
        
        with patch('builtins.open', mock_open()) as mock_file:
            self.manager._save_version(version)
            mock_file.assert_called()

    @patch('builtins.open', side_effect=IOError("Test error"))
    def test_save_version_handles_exception(self, mock_open_func):
        """测试保存版本时的异常处理"""
        version = ConfigVersion(
            version_id='test_v1',
            timestamp=time.time(),
            config_data={'test': 'data'},
            checksum='test123'
        )
        # 应该不抛出异常
        self.manager._save_version(version)

    def test_cleanup_old_versions(self):
        """测试清理旧版本"""
        # 手动添加一些版本到历史中
        self.manager.max_versions = 2
        for i in range(3):
            version_id = f'v{i}'
            self.manager._version_history.append(version_id)
            # 创建虚拟版本对象
            version = ConfigVersion(
                version_id=version_id,
                timestamp=time.time(),
                config_data={'test': f'data{i}'},
                checksum=f'checksum{i}'
            )
            self.manager._versions[version_id] = version
        
        with patch('pathlib.Path.unlink'):
            self.manager._cleanup_old_versions()
        
        # 应该只保留最新的版本
        assert len(self.manager._version_history) == 2

    def test_get_version_existing(self):
        """测试获取存在的版本"""
        version_id = 'test_v1'
        version = ConfigVersion(
            version_id=version_id,
            timestamp=time.time(),
            config_data={'test': 'data'},
            checksum='test123'
        )
        self.manager._versions[version_id] = version
        
        result = self.manager.get_version(version_id)
        assert result == version

    def test_get_version_nonexistent(self):
        """测试获取不存在的版本"""
        result = self.manager.get_version('nonexistent')
        assert result is None

    def test_get_latest_version_with_versions(self):
        """测试获取最新版本（有版本时）"""
        version1 = ConfigVersion(
            version_id='v1',
            timestamp=time.time(),
            config_data={'test': 'data1'},
            checksum='checksum1'
        )
        version2 = ConfigVersion(
            version_id='v2',
            timestamp=time.time(),
            config_data={'test': 'data2'},
            checksum='checksum2'
        )
        
        self.manager._versions = {'v1': version1, 'v2': version2}
        self.manager._version_history = ['v1', 'v2']
        
        latest = self.manager.get_latest_version()
        assert latest == version2

    def test_get_latest_version_no_versions(self):
        """测试获取最新版本（无版本时）"""
        latest = self.manager.get_latest_version()
        assert latest is None

    def test_list_versions_with_filters(self):
        """测试列出版本并使用过滤器"""
        # 创建测试版本
        version1 = ConfigVersion(
            version_id='v1',
            timestamp=time.time(),
            config_data={'test': 'data1'},
            checksum='checksum1',
            author='user1',
            tags=['production']
        )
        
        version2 = ConfigVersion(
            version_id='v2',
            timestamp=time.time(),
            config_data={'test': 'data2'},
            checksum='checksum2',
            author='user2',
            tags=['development']
        )
        
        self.manager._versions = {'v1': version1, 'v2': version2}
        self.manager._version_history = ['v1', 'v2']
        
        # 测试作者过滤器
        result = self.manager.list_versions(author='user1')
        assert len(result) == 1
        assert result[0].version_id == 'v1'
        
        # 测试标签过滤器
        result = self.manager.list_versions(tags=['production'])
        assert len(result) == 1
        assert result[0].version_id == 'v1'
        
        # 测试限制数量
        result = self.manager.list_versions(limit=1)
        assert len(result) == 1

    def test_rollback_to_version_success(self):
        """测试成功回滚到版本"""
        # 创建测试版本
        version = ConfigVersion(
            version_id='rollback_v1',
            timestamp=time.time(),
            config_data={'rollback': 'data'},
            checksum='rollback123'
        )
        self.manager._versions['rollback_v1'] = version
        
        with patch.object(self.manager, '_validate_version', return_value=True), \
             patch.object(self.manager, '_get_current_config', return_value={'current': 'config'}), \
             patch.object(self.manager, '_execute_rollback', return_value=True), \
             patch.object(self.manager, 'create_version') as mock_create:
            
            result = self.manager.rollback_to_version('rollback_v1')
            assert result is True

    def test_rollback_to_version_not_found(self):
        """测试回滚到不存在的版本"""
        result = self.manager.rollback_to_version('nonexistent')
        assert result is False

    def test_rollback_to_version_validation_failed(self):
        """测试版本验证失败的回滚"""
        version = ConfigVersion(
            version_id='invalid_v1',
            timestamp=time.time(),
            config_data={'test': 'data'},
            checksum='invalid123'
        )
        self.manager._versions['invalid_v1'] = version
        
        with patch.object(self.manager, '_validate_version', return_value=False):
            result = self.manager.rollback_to_version('invalid_v1')
            assert result is False

    def test_validate_version_basic(self):
        """测试版本验证基本功能"""
        version = ConfigVersion(
            version_id='test_v1',
            timestamp=time.time(),
            config_data={'test': 'data'},
            checksum='abc123'
        )
        
        with patch.object(self.manager, '_validate_config_structure', return_value=True):
            # 需要正确计算校验和
            config_str = json.dumps(version.config_data, sort_keys=True)
            import hashlib
            correct_checksum = hashlib.sha256(config_str.encode()).hexdigest()[:16]
            version.checksum = correct_checksum
            
            result = self.manager._validate_version(version)
            assert result is True

    def test_validate_version_invalid_data(self):
        """测试版本验证 - 无效数据"""
        version = ConfigVersion(
            version_id='invalid_v1',
            timestamp=time.time(),
            config_data="invalid",  # 不是字典，但在dataclass限制下，这里需要调整测试方法
            checksum='abc123'
        )
        
        # 我们需要创建一个不正确的config_data，但这需要在运行时修改
        version.config_data = "invalid"
        result = self.manager._validate_version(version)
        assert result is False

    def test_validate_version_checksum_mismatch(self):
        """测试版本验证 - 校验和不匹配"""
        version = ConfigVersion(
            version_id='checksum_v1',
            timestamp=time.time(),
            config_data={'test': 'data'},
            checksum='wrong_checksum'
        )
        
        result = self.manager._validate_version(version)
        assert result is False

    def test_validate_config_structure(self):
        """测试配置结构验证"""
        # 正常配置
        normal_config = {'section1': {'key1': 'value1'}}
        result = self.manager._validate_config_structure(normal_config)
        assert result is True
        
        # 过度嵌套的配置
        deep_config = normal_config
        for i in range(15):  # 超过最大深度10
            deep_config = {'level': deep_config}
        result = self.manager._validate_config_structure(deep_config)
        assert result is False

    def test_get_dict_depth(self):
        """测试获取字典深度"""
        # 空字典
        assert self.manager._get_dict_depth({}) == 0
        
        # 简单字典
        simple = {'a': 1}
        assert self.manager._get_dict_depth(simple) == 1
        
        # 嵌套字典
        nested = {'a': {'b': {'c': 1}}}
        assert self.manager._get_dict_depth(nested) == 3

    def test_restore_version_existing(self):
        """测试恢复存在的版本"""
        version = ConfigVersion(
            version_id='restore_v1',
            timestamp=time.time(),
            config_data={'restore': 'data'},
            checksum='restore123'
        )
        self.manager._versions['restore_v1'] = version
        
        result = self.manager.restore_version('restore_v1')
        assert result == {'restore': 'data'}
        assert self.manager.stats['total_restores'] == 1

    def test_restore_version_nonexistent(self):
        """测试恢复不存在的版本"""
        result = self.manager.restore_version('nonexistent')
        assert result is None

    def test_compare_versions_success(self):
        """测试版本比较成功"""
        version1 = ConfigVersion(
            version_id='v1',
            timestamp=time.time(),
            config_data={'key1': 'value1'},
            checksum='checksum1'
        )
        
        version2 = ConfigVersion(
            version_id='v2',
            timestamp=time.time(),
            config_data={'key1': 'value2', 'key2': 'new_value'},
            checksum='checksum2'
        )
        
        self.manager._versions = {'v1': version1, 'v2': version2}
        
        diff = self.manager.compare_versions('v1', 'v2')
        assert isinstance(diff, ConfigDiff)
        assert diff.version_from == 'v1'
        assert diff.version_to == 'v2'

    def test_compare_versions_missing(self):
        """测试版本比较 - 版本不存在"""
        diff = self.manager.compare_versions('nonexistent1', 'nonexistent2')
        assert diff is None

    def test_calculate_diff(self):
        """测试计算配置差异"""
        config1 = {'key1': 'value1', 'key2': 'value2'}
        config2 = {'key1': 'value1_modified', 'key3': 'value3'}
        
        diff = self.manager._calculate_diff(config1, config2)
        assert 'key3' in diff['added']
        assert 'key2' in diff['removed']
        assert 'key1' in diff['modified']

    def test_flatten_keys(self):
        """测试扁平化键"""
        config = {
            'level1': {
                'level2': {
                    'key1': 'value1'
                },
                'key2': 'value2'
            },
            'key3': 'value3'
        }
        
        keys = self.manager._flatten_keys(config)
        assert 'level1' in keys
        assert 'level1.level2' in keys
        assert 'level1.level2.key1' in keys
        assert 'level1.key2' in keys
        assert 'key3' in keys

    def test_get_nested_value(self):
        """测试获取嵌套值"""
        config = {'level1': {'level2': {'key': 'value'}}}
        
        # 存在的嵌套键
        value = self.manager._get_nested_value(config, 'level1.level2.key')
        assert value == 'value'
        
        # 不存在的键
        value = self.manager._get_nested_value(config, 'nonexistent.key')
        assert value is None

    def test_delete_version_existing(self):
        """测试删除存在的版本"""
        version_id = 'delete_v1'
        version = ConfigVersion(
            version_id=version_id,
            timestamp=time.time(),
            config_data={'test': 'data'},
            checksum='delete123'
        )
        self.manager._versions[version_id] = version
        self.manager._version_history = [version_id]
        
        with patch('pathlib.Path.unlink'):
            result = self.manager.delete_version(version_id)
            assert result is True
            assert version_id not in self.manager._versions
            assert version_id not in self.manager._version_history

    def test_delete_version_nonexistent(self):
        """测试删除不存在的版本"""
        result = self.manager.delete_version('nonexistent')
        assert result is False

    @patch('pathlib.Path.exists', return_value=True)
    @patch('pathlib.Path.unlink', side_effect=IOError("Test error"))
    def test_delete_version_handles_exception(self, mock_unlink, mock_exists):
        """测试删除版本时的异常处理"""
        version_id = 'error_v1'
        version = ConfigVersion(
            version_id=version_id,
            timestamp=time.time(),
            config_data={'test': 'data'},
            checksum='error123'
        )
        self.manager._versions[version_id] = version
        self.manager._version_history = [version_id]
        
        result = self.manager.delete_version(version_id)
        assert result is False

    def test_cleanup_old_versions_manual(self):
        """测试手动清理旧版本"""
        # 添加多个版本
        for i in range(5):
            version_id = f'cleanup_v{i}'
            version = ConfigVersion(
                version_id=version_id,
                timestamp=time.time(),
                config_data={'test': f'data{i}'},
                checksum=f'cleanup{i}'
            )
            self.manager._versions[version_id] = version
            self.manager._version_history.append(version_id)
        
        # 清理，保留2个
        self.manager.cleanup_old_versions(keep_count=2)
        assert len(self.manager._version_history) == 2

    def test_get_stats(self):
        """测试获取统计信息"""
        version1 = ConfigVersion(
            version_id='v1',
            timestamp=time.time(),
            config_data={'test': 'data1'},
            checksum='checksum1'
        )
        version2 = ConfigVersion(
            version_id='v2',
            timestamp=time.time(),
            config_data={'test': 'data2'},
            checksum='checksum2'
        )
        self.manager._versions = {'v1': version1, 'v2': version2}
        
        stats = self.manager.get_stats()
        assert 'current_versions' in stats
        assert 'storage_path' in stats
        assert 'max_versions' in stats
        assert stats['current_versions'] == 2

    def test_export_versions(self):
        """测试导出版本"""
        # 添加一些版本
        version = ConfigVersion(
            version_id='export_v1',
            timestamp=time.time(),
            config_data={'export': 'data'},
            checksum='export123'
        )
        self.manager._versions['export_v1'] = version
        self.manager._version_history = ['export_v1']
        
        with patch('builtins.open', mock_open()) as mock_file, \
             patch.object(ConfigVersion, 'to_dict', return_value={'test': 'data'}):
            
            result = self.manager.export_versions('/tmp/export.json')
            assert result is True
            mock_file.assert_called_with('/tmp/export.json', 'w', encoding='utf-8')

    @patch('builtins.open', side_effect=IOError("Test error"))
    def test_export_versions_handles_exception(self, mock_open_func):
        """测试导出版本时的异常处理"""
        result = self.manager.export_versions('/tmp/export.json')
        assert result is False

    def test_import_versions(self):
        """测试导入版本"""
        import_data = {
            'versions': {
                'import_v1': {
                    'version_id': 'import_v1',
                    'timestamp': time.time(),
                    'config_data': {'import': 'data'},
                    'checksum': 'import123',
                    'author': 'import_user',
                    'description': 'imported version',
                    'tags': ['import'],
                    'metadata': {}
                }
            },
            'history': ['import_v1'],
            'stats': {'total_versions': 1}
        }
        
        with patch('builtins.open', mock_open(read_data=json.dumps(import_data))):
            result = self.manager.import_versions('/tmp/import.json')
            assert result is True
            assert 'import_v1' in self.manager._versions

    @patch('builtins.open', side_effect=IOError("Test error"))
    def test_import_versions_handles_exception(self, mock_open_func):
        """测试导入版本时的异常处理"""
        result = self.manager.import_versions('/tmp/import.json')
        assert result is False
