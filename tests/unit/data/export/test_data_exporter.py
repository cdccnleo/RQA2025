"""
数据导出模块测试
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import shutil
import io
import zipfile
from unittest.mock import Mock, patch

from src.data.export.data_exporter import DataExporter
from src.data.models import DataModel
from src.infrastructure.utils.exceptions import DataLoaderError


@pytest.fixture
def test_data():
    """测试数据fixture"""
    # 创建测试数据
    dates = pd.date_range(start='2023-01-01', end='2023-01-10')
    df = pd.DataFrame({
        'close': np.random.randn(len(dates)) + 100,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    return df


@pytest.fixture
def test_export_dir(tmp_path):
    """测试导出目录fixture"""
    export_dir = tmp_path / "test_exports"
    export_dir.mkdir()
    yield export_dir
    # 清理测试目录
    shutil.rmtree(export_dir)


@pytest.fixture
def data_exporter(test_export_dir):
    """数据导出器fixture"""
    return DataExporter(test_export_dir)


@pytest.fixture
def sample_data_model(test_data):
    """样本数据模型fixture"""
    model = DataModel(test_data)
    model.set_metadata({
        'source': 'test',
        'frequency': '1d',
        'symbol': '000001.SZ'
    })
    return model


def test_exporter_init(test_export_dir):
    """测试导出器初始化"""
    exporter = DataExporter(test_export_dir)

    # 验证目录创建
    assert test_export_dir.exists()
    assert test_export_dir.is_dir()

    # 验证支持的格式
    assert len(exporter.get_supported_formats()) > 0
    assert 'csv' in exporter.get_supported_formats()
    assert 'excel' in exporter.get_supported_formats()


@pytest.mark.parametrize("format,expected_suffix", [
    ('csv', '.csv'),
    ('excel', '.excel'),
    ('json', '.json'),
    ('parquet', '.parquet'),
    ('pickle', '.pickle'),
    ('html', '.html'),
    ('feather', '.feather'),
    ('stata', '.stata'),
    ('hdf', '.hdf')
])
def test_export_formats(data_exporter, sample_data_model, format, expected_suffix):
    """测试不同格式导出"""
    # 导出数据
    filepath = data_exporter.export(
        sample_data_model,
        format=format,
        filename=f"test_export{expected_suffix}",
        include_metadata=True
    )

    # 验证文件创建
    assert Path(filepath).exists()
    assert filepath.endswith(expected_suffix)

    # 验证导出历史记录
    history = data_exporter.get_export_history()
    assert len(history) > 0
    assert history[-1]['format'] == format
    assert history[-1]['filepath'] == filepath


def test_export_with_metadata(data_exporter, sample_data_model):
    """测试包含元数据的导出"""
    # 导出CSV格式（会创建单独的元数据文件）
    filepath = data_exporter.export(
        sample_data_model,
        format='csv',
        include_metadata=True
    )

    # 验证元数据文件
    metadata_path = Path(filepath).with_suffix('.metadata.json')
    assert metadata_path.exists()

    # 验证元数据内容
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    assert metadata['source'] == 'test'
    assert metadata['symbol'] == '000001.SZ'

    # 导出JSON格式（元数据包含在同一文件中）
    json_path = data_exporter.export(
        sample_data_model,
        format='json',
        include_metadata=True
    )

    # 验证JSON内容
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    assert 'metadata' in json_data
    assert 'data' in json_data
    assert json_data['metadata']['source'] == 'test'


def test_export_multiple(data_exporter, sample_data_model):
    """测试多文件导出"""
    # 创建多个数据模型
    models = [
        sample_data_model,
        sample_data_model  # 使用相同数据创建第二个模型
    ]

    # 导出多个文件到ZIP
    zip_path = data_exporter.export_multiple(
        models,
        format='csv',
        include_metadata=True
    )

    # 验证ZIP文件创建
    assert Path(zip_path).exists()
    assert zip_path.endswith('.zip')

    # 检查ZIP内容
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        files = zipf.namelist()
        assert len(files) > 0
        # 验证包含数据文件和元数据文件
        csv_files = [f for f in files if f.endswith('.csv')]
        metadata_files = [f for f in files if f.endswith('.metadata.json')]
        assert len(csv_files) == 2
        assert len(metadata_files) == 2


def test_export_to_buffer(data_exporter, sample_data_model):
    """测试导出到内存缓冲区"""
    # 测试CSV格式
    csv_buffer = data_exporter.export_to_buffer(
        sample_data_model,
        format='csv',
        include_metadata=True
    )
    assert isinstance(csv_buffer, io.BytesIO)
    assert len(csv_buffer.getvalue()) > 0

    # 测试JSON格式
    json_buffer = data_exporter.export_to_buffer(
        sample_data_model,
        format='json',
        include_metadata=True
    )
    assert isinstance(json_buffer, io.BytesIO)
    json_data = json.loads(json_buffer.getvalue().decode('utf-8'))
    assert 'metadata' in json_data
    assert 'data' in json_data


@pytest.mark.parametrize("invalid_format", [
    'invalid',
    '',
    None,
    123
])
def test_invalid_format(data_exporter, sample_data_model, invalid_format):
    """测试无效格式"""
    with pytest.raises(DataLoaderError, match="Unsupported export format"):
        data_exporter.export(
            sample_data_model,
            format=invalid_format
        )


def test_export_empty_data(data_exporter):
    """测试导出空数据"""
    # 创建空数据模型
    empty_df = pd.DataFrame()
    empty_model = DataModel(empty_df)
    empty_model.set_metadata({'source': 'test'})

    # 验证导出成功
    filepath = data_exporter.export(
        empty_model,
        format='csv'
    )
    assert Path(filepath).exists()
    assert Path(filepath).stat().st_size > 0  # 文件不为空（至少包含表头）


def test_export_large_data(data_exporter):
    """测试导出大数据集"""
    # 创建大数据集
    dates = pd.date_range(start='2020-01-01', periods=10000)
    large_df = pd.DataFrame({
        'close': np.random.randn(len(dates)) + 100,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    large_model = DataModel(large_df)

    # 测试不同格式的导出
    for format in ['csv', 'parquet']:
        filepath = data_exporter.export(
            large_model,
            format=format
        )
        assert Path(filepath).exists()


@patch('pandas.DataFrame.to_csv')
def test_export_failure(mock_to_csv, data_exporter, sample_data_model):
    """测试导出失败情况"""
    # 模拟CSV导出失败
    mock_to_csv.side_effect = Exception("Failed to export")

    with pytest.raises(DataLoaderError, match="Failed to export data"):
        data_exporter.export(
            sample_data_model,
            format='csv'
        )


def test_export_history_limit(data_exporter, sample_data_model):
    """测试导出历史记录限制"""
    # 执行多次导出
    for i in range(5):
        data_exporter.export(
            sample_data_model,
            format='csv',
            filename=f"test_{i}.csv"
        )

    # 验证历史记录
    history = data_exporter.get_export_history(limit=3)
    assert len(history) == 3
    assert history[-1]['filepath'].endswith('test_4.csv')


def test_concurrent_export(data_exporter, sample_data_model):
    """测试并发导出"""
    import threading

    # 创建多个线程同时导出
    threads = []
    for i in range(5):
        thread = threading.Thread(
            target=data_exporter.export,
            args=(sample_data_model, 'csv'),
            kwargs={'filename': f"concurrent_{i}.csv"}
        )
        threads.append(thread)

    # 启动所有线程
    for thread in threads:
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()

    # 验证所有文件都被创建
    for i in range(5):
        assert (data_exporter.export_dir / f"concurrent_{i}.csv").exists()


def test_export_with_custom_options(data_exporter, sample_data_model):
    """测试使用自定义导出选项"""
    # CSV格式自定义选项
    filepath = data_exporter.export(
        sample_data_model,
        format='csv',
        sep='|',  # 自定义分隔符
        index=False,  # 不导出索引
        date_format='%Y-%m-%d'  # 自定义日期格式
    )

    # 验证导出结果
    with open(filepath, 'r') as f:
        content = f.read()
        assert '|' in content  # 使用了自定义分隔符
        assert not content.startswith('date|')  # 索引未被导出
