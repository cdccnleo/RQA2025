import pytest
from src.data.loader.base_loader import BaseDataLoader, LoaderConfig
from typing import Dict, Any, List

class DummyLoader(BaseDataLoader):
    def __init__(self, config=None, fail_on=None):
        super().__init__(config)
        self.fail_on = fail_on or set()

    def load(self, *args, **kwargs):
        # 模拟根据参数决定是否抛异常
        if kwargs.get('fail'):
            raise ValueError('load failed')
        return kwargs.get('data', {'value': 1})

    def get_metadata(self) -> Dict[str, Any]:
        return {
            'loader_type': 'dummy',
            'version': '1.0',
            'description': 'Dummy loader for testing'
        }

@pytest.fixture
def default_config():
    return LoaderConfig(batch_size=10, max_retries=2, timeout=5)

@pytest.fixture
def loader(default_config):
    return DummyLoader(config=default_config)

def test_init_and_config(loader, default_config):
    assert loader.config.batch_size == 10
    assert loader.config.max_retries == 2
    assert loader.config.timeout == 5
    assert loader._load_count == 0
    assert loader._last_load_time is None

def test_validate(loader):
    assert loader.validate({'a': 1}) is True
    assert loader.validate(None) is False

def test_get_metadata(loader):
    meta = loader.get_metadata()
    assert meta['loader_type'] == 'dummy'
    assert meta['version'] == '1.0'
    assert 'description' in meta

def test_batch_load_success(loader):
    params_list = [
        {'data': {'v': i}} for i in range(3)
    ]
    result = loader.batch_load(params_list)
    assert isinstance(result, list)
    assert len(result) == 3
    for i, item in enumerate(result):
        assert item == {'v': i}

def test_batch_load_with_invalid_and_exception(loader):
    params_list = [
        {'data': {'v': 1}},
        {'data': None},  # validate为False
        {'fail': True}, # load抛异常
        {'data': {'v': 2}}
    ]
    result = loader.batch_load(params_list)
    assert len(result) == 2
    assert {'v': 1} in result and {'v': 2} in result

def test_get_stats_and_update_stats(loader):
    stats = loader.get_stats()
    assert stats['load_count'] == 0
    assert stats['last_load_time'] is None
    loader._update_stats()
    stats2 = loader.get_stats()
    assert stats2['load_count'] == 1
    assert stats2['last_load_time'] is not None

def test_abstract_methods_enforced():
    class IncompleteLoader(BaseDataLoader):
        pass
    with pytest.raises(TypeError):
        IncompleteLoader() 