import pytest
from unittest.mock import patch, MagicMock, Mock
import sys
from types import ModuleType
from src.data.loader.batch_loader import BatchDataLoader

@pytest.fixture
def mock_executor():
    mock = MagicMock()
    mock.initial_workers = 4
    mock.max_workers = 8
    mock.execute_batch.side_effect = lambda tasks: [t() for t in tasks]
    return mock

@pytest.fixture
def loader(mock_executor):
    mock_config = MagicMock()
    mock_config.get = Mock(return_value=False)
    with patch('src.data.loader.batch_loader.config', mock_config):
        with patch('src.data.loader.batch_loader.DynamicExecutor', return_value=mock_executor):
            l = BatchDataLoader()
            l.executor = mock_executor  # 确保mock覆盖
            # 动态添加 _load_single 方法
            setattr(l, '_load_single', MagicMock(side_effect=lambda s, sd, ed: {'symbol': s, 'start': sd, 'end': ed}))
            yield l

def test_init_default_and_high_perf():
    mock_config = MagicMock()
    # 默认模式
    mock_config.get = Mock(return_value=False)
    with patch('src.data.loader.batch_loader.config', mock_config):
        with patch('src.data.loader.batch_loader.DynamicExecutor') as mock_exec:
            BatchDataLoader()
            mock_exec.assert_called_with(initial_workers=4, max_workers=8)
    # 高性能模式
    mock_config.get = Mock(return_value=True)
    with patch('src.data.loader.batch_loader.config', mock_config):
        with patch('src.data.loader.batch_loader.DynamicExecutor') as mock_exec:
            BatchDataLoader()
            mock_exec.assert_called_with(initial_workers=4, max_workers=16)

def test_get_metadata(loader):
    meta = loader.get_metadata()
    assert meta['loader_type'] == 'BatchDataLoader'
    assert meta['initial_workers'] == 4
    assert meta['max_workers'] == 8
    assert meta['supports_batch'] is True

def test_create_load_task(loader):
    task = loader._create_load_task('AAA', '2023-01-01', '2023-01-31')
    assert callable(task)
    result = task()
    assert isinstance(result, dict)
    assert result.get('symbol') == 'AAA'
    assert result.get('start') == '2023-01-01'
    assert result.get('end') == '2023-01-31'

def test_load_batch_normal(loader):
    symbols = ['AAA', 'BBB']
    result = loader.load_batch(symbols, '2023-01-01', '2023-01-31')
    assert set(result.keys()) == set(symbols)
    for v in result.values():
        assert isinstance(v, dict)
        assert v['start'] == '2023-01-01'

def test_load_batch_with_none(loader, mock_executor):
    # 模拟部分任务返回None
    def fake_execute_batch(tasks):
        return [tasks[0](), None]
    mock_executor.execute_batch.side_effect = fake_execute_batch
    setattr(loader, '_load_single', MagicMock(side_effect=[{'symbol': 'AAA'}, None]))
    result = loader.load_batch(['AAA', 'BBB'], '2023-01-01', '2023-01-31')
    assert 'AAA' in result and 'BBB' not in result

def test_validate(loader):
    valid = {'AAA': {'a': 1}, 'BBB': {'b': 2}}
    invalid1 = ['not', 'a', 'dict']
    invalid2 = {'AAA': 123, 'BBB': None}
    assert loader.validate(valid) is True
    assert loader.validate(invalid1) is False
    assert loader.validate(invalid2) is False

def test_load_calls_load_batch(loader):
    with patch.object(loader, 'load_batch', return_value={'AAA': {}}) as mock_lb:
        result = loader.load(['AAA'], '2023-01-01', '2023-01-31')
        mock_lb.assert_called_once()
        assert 'AAA' in result 