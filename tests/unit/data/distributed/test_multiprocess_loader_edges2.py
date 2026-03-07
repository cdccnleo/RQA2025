import asyncio
import pandas as pd
from unittest.mock import Mock

# Mock数据管理器模块以绕过复杂的导入问题
mock_data_manager = Mock()
mock_data_manager.DataManager = Mock()
mock_data_manager.DataLoaderError = Exception

# 配置DataManager实例方法
mock_instance = Mock()
mock_instance.validate_all_configs.return_value = True
mock_instance.health_check.return_value = {"status": "healthy"}
mock_instance.store_data.return_value = True
mock_instance.has_data.return_value = True
mock_instance.get_metadata.return_value = {"data_type": "test", "symbol": "X"}
mock_instance.retrieve_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_instance.get_stats.return_value = {"total_items": 1}
mock_instance.validate_data.return_value = {"valid": True}
mock_instance.shutdown.return_value = None

mock_data_manager.DataManager.return_value = mock_instance

# Mock整个模块
import sys
sys.modules["src.data.data_manager"] = mock_data_manager


import pytest
from unittest.mock import Mock, patch
from multiprocessing import Pool

from src.data.distributed.multiprocess_loader import MultiprocessDataLoader


def dummy_worker(task):
    """测试用的工作函数"""
    return task.get('value', 0) * 2


def test_multiprocess_loader_init_default_workers():
    """测试 MultiprocessDataLoader（初始化，默认工作进程数）"""
    loader = MultiprocessDataLoader(worker_fn=dummy_worker)
    assert loader.worker_fn == dummy_worker
    assert loader.num_workers > 0


def test_multiprocess_loader_init_custom_workers():
    """测试 MultiprocessDataLoader（初始化，自定义工作进程数）"""
    loader = MultiprocessDataLoader(worker_fn=dummy_worker, num_workers=4)
    assert loader.num_workers == 4


def test_multiprocess_loader_init_zero_workers():
    """测试 MultiprocessDataLoader（初始化，零工作进程数）"""
    # 零工作进程数会被 or cpu_count() 替换为 cpu_count()
    # 因为 0 是 falsy，所以实际上会使用默认值
    loader = MultiprocessDataLoader(worker_fn=dummy_worker, num_workers=0)
    # 由于 num_workers or cpu_count() 的逻辑，0 会被替换为 cpu_count()
    assert loader.num_workers > 0  # 实际上会使用 cpu_count()


def test_multiprocess_loader_init_negative_workers():
    """测试 MultiprocessDataLoader（初始化，负工作进程数）"""
    loader = MultiprocessDataLoader(worker_fn=dummy_worker, num_workers=-1)
    assert loader.num_workers == -1


@patch('src.data.distributed.multiprocess_loader.Pool')
def test_multiprocess_loader_distribute_load_empty_tasks(mock_pool):
    """测试 MultiprocessDataLoader（分发负载，空任务列表）"""
    mock_pool_instance = Mock()
    mock_pool_instance.__enter__ = Mock(return_value=mock_pool_instance)
    mock_pool_instance.__exit__ = Mock(return_value=False)
    mock_pool_instance.map = Mock(return_value=[])
    mock_pool.return_value = mock_pool_instance
    
    loader = MultiprocessDataLoader(worker_fn=dummy_worker, num_workers=2)
    results = loader.distribute_load([])
    assert results == []


@patch('src.data.distributed.multiprocess_loader.Pool')
def test_multiprocess_loader_distribute_load_single_task(mock_pool):
    """测试 MultiprocessDataLoader（分发负载，单个任务）"""
    mock_pool_instance = Mock()
    mock_pool_instance.__enter__ = Mock(return_value=mock_pool_instance)
    mock_pool_instance.__exit__ = Mock(return_value=False)
    mock_pool_instance.map = Mock(return_value=[10])
    mock_pool.return_value = mock_pool_instance
    
    loader = MultiprocessDataLoader(worker_fn=dummy_worker, num_workers=2)
    tasks = [{'value': 5}]
    results = loader.distribute_load(tasks)
    assert len(results) == 1
    assert results[0] == 10  # 5 * 2


@patch('src.data.distributed.multiprocess_loader.Pool')
def test_multiprocess_loader_distribute_load_multiple_tasks(mock_pool):
    """测试 MultiprocessDataLoader（分发负载，多个任务）"""
    mock_pool_instance = Mock()
    mock_pool_instance.__enter__ = Mock(return_value=mock_pool_instance)
    mock_pool_instance.__exit__ = Mock(return_value=False)
    mock_pool_instance.map = Mock(return_value=[2, 4, 6])
    mock_pool.return_value = mock_pool_instance
    
    loader = MultiprocessDataLoader(worker_fn=dummy_worker, num_workers=2)
    tasks = [{'value': 1}, {'value': 2}, {'value': 3}]
    results = loader.distribute_load(tasks)
    assert len(results) == 3
    assert results == [2, 4, 6]


@patch('src.data.distributed.multiprocess_loader.Pool')
def test_multiprocess_loader_distribute_load_task_without_value(mock_pool):
    """测试 MultiprocessDataLoader（分发负载，任务无值）"""
    mock_pool_instance = Mock()
    mock_pool_instance.__enter__ = Mock(return_value=mock_pool_instance)
    mock_pool_instance.__exit__ = Mock(return_value=False)
    mock_pool_instance.map = Mock(return_value=[0])
    mock_pool.return_value = mock_pool_instance
    
    loader = MultiprocessDataLoader(worker_fn=dummy_worker, num_workers=2)
    tasks = [{}]
    results = loader.distribute_load(tasks)
    assert len(results) == 1
    assert results[0] == 0  # 默认值 0 * 2


def test_multiprocess_loader_aggregate_results_none_aggregate_fn():
    """测试 MultiprocessDataLoader（聚合结果，None 聚合函数）"""
    loader = MultiprocessDataLoader(worker_fn=dummy_worker)
    results = [1, 2, 3]
    aggregated = loader.aggregate_results(results)
    assert aggregated == [1, 2, 3]


def test_multiprocess_loader_aggregate_results_empty_results():
    """测试 MultiprocessDataLoader（聚合结果，空结果）"""
    loader = MultiprocessDataLoader(worker_fn=dummy_worker)
    aggregated = loader.aggregate_results([])
    assert aggregated == []


def test_multiprocess_loader_aggregate_results_with_aggregate_fn():
    """测试 MultiprocessDataLoader（聚合结果，带聚合函数）"""
    loader = MultiprocessDataLoader(worker_fn=dummy_worker)
    results = [1, 2, 3]
    aggregated = loader.aggregate_results(results, aggregate_fn=sum)
    assert aggregated == 6


def test_multiprocess_loader_aggregate_results_custom_aggregate_fn():
    """测试 MultiprocessDataLoader（聚合结果，自定义聚合函数）"""
    loader = MultiprocessDataLoader(worker_fn=dummy_worker)
    results = [1, 2, 3]
    aggregated = loader.aggregate_results(results, aggregate_fn=lambda x: max(x))
    assert aggregated == 3


@patch('src.data.distributed.multiprocess_loader.Pool')
def test_multiprocess_loader_load_distributed(mock_pool):
    """测试 MultiprocessDataLoader（分布式加载）"""
    mock_pool_instance = Mock()
    mock_pool_instance.__enter__ = Mock(return_value=mock_pool_instance)
    mock_pool_instance.__exit__ = Mock(return_value=False)
    mock_pool_instance.map = Mock(return_value=[{'result': 'data'}])
    mock_pool.return_value = mock_pool_instance
    
    loader = MultiprocessDataLoader(worker_fn=dummy_worker, num_workers=2)
    results = loader.load_distributed("2023-01-01", "2023-01-31", "daily")
    assert isinstance(results, list)
    assert len(results) == 1


@patch('src.data.distributed.multiprocess_loader.Pool')
def test_multiprocess_loader_load_distributed_empty_dates(mock_pool):
    """测试 MultiprocessDataLoader（分布式加载，空日期）"""
    mock_pool_instance = Mock()
    mock_pool_instance.__enter__ = Mock(return_value=mock_pool_instance)
    mock_pool_instance.__exit__ = Mock(return_value=False)
    mock_pool_instance.map = Mock(return_value=[{}])
    mock_pool.return_value = mock_pool_instance
    
    loader = MultiprocessDataLoader(worker_fn=dummy_worker, num_workers=2)
    results = loader.load_distributed("", "", "daily")
    assert isinstance(results, list)


def test_multiprocess_loader_get_node_info():
    """测试 MultiprocessDataLoader（获取节点信息）"""
    loader = MultiprocessDataLoader(worker_fn=dummy_worker, num_workers=4)
    info = loader.get_node_info()
    assert info['node_type'] == 'multiprocess'
    assert info['num_workers'] == 4
    assert 'worker_function' in info


def test_multiprocess_loader_get_node_info_lambda_worker():
    """测试 MultiprocessDataLoader（获取节点信息，Lambda 工作函数）"""
    loader = MultiprocessDataLoader(worker_fn=lambda x: x, num_workers=2)
    info = loader.get_node_info()
    assert info['node_type'] == 'multiprocess'
    assert info['num_workers'] == 2
    # Lambda 函数可能没有 __name__ 属性
    assert 'worker_function' in info


def test_multiprocess_loader_get_cluster_status():
    """测试 MultiprocessDataLoader（获取集群状态）"""
    loader = MultiprocessDataLoader(worker_fn=dummy_worker, num_workers=4)
    status = loader.get_cluster_status()
    assert status['status'] == 'active'
    assert status['node_count'] == 1
    assert status['worker_count'] == 4
    assert status['total_tasks_processed'] == 0


def test_multiprocess_loader_get_cluster_status_zero_workers():
    """测试 MultiprocessDataLoader（获取集群状态，零工作进程）"""
    # 由于 num_workers or cpu_count() 的逻辑，0 会被替换为 cpu_count()
    loader = MultiprocessDataLoader(worker_fn=dummy_worker, num_workers=0)
    status = loader.get_cluster_status()
    # 实际上会使用 cpu_count()，所以 worker_count > 0
    assert status['worker_count'] > 0

