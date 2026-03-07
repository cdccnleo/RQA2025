"""
数据导出模块深度测试
全面测试数据导出系统的各种功能和边界条件
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os
import json
import zipfile

# 导入实际的类
from src.data.export.data_exporter import DataExporter


class TestDataExportComprehensive:
    """数据导出综合深度测试"""

    @pytest.fixture
    def sample_stock_data(self):
        """创建样本股票数据"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        return pd.DataFrame({
            'symbol': ['AAPL'] * 100,
            'date': dates,
            'open': np.random.uniform(150, 200, 100),
            'high': np.random.uniform(155, 205, 100),
            'low': np.random.uniform(145, 195, 100),
            'close': np.random.uniform(150, 200, 100),
            'volume': np.random.randint(1000000, 10000000, 100),
            'adj_close': np.random.uniform(150, 200, 100)
        })

    @pytest.fixture
    def data_exporter(self, tmp_path):
        """创建数据导出器实例"""
        export_dir = tmp_path / "exports"
        return DataExporter(str(export_dir))

    @pytest.fixture
    def temp_export_dir(self, tmp_path):
        """创建临时导出目录"""
        export_dir = tmp_path / "test_exports"
        export_dir.mkdir(exist_ok=True)
        return export_dir

    def test_data_exporter_initialization(self, data_exporter):
        """测试数据导出器初始化"""
        assert data_exporter is not None
        assert isinstance(data_exporter.export_dir, Path)
        assert data_exporter.export_dir.exists()

    def test_export_to_csv(self, data_exporter, sample_stock_data):
        """测试导出到CSV格式"""
        filename = "test_export.csv"

        # 导出数据
        result = data_exporter.export_to_csv(sample_stock_data, filename)

        # 检查导出结果
        assert result['success'] is True
        assert 'file_path' in result

        # 检查文件是否创建
        csv_file = data_exporter.export_dir / filename
        assert csv_file.exists()

        # 检查文件内容
        exported_data = pd.read_csv(csv_file)
        assert len(exported_data) == len(sample_stock_data)
        assert list(exported_data.columns) == list(sample_stock_data.columns)

    def test_export_to_json(self, data_exporter, sample_stock_data):
        """测试导出到JSON格式"""
        filename = "test_export.json"

        # 导出数据
        result = data_exporter.export_to_json(sample_stock_data, filename)

        # 检查导出结果
        assert result['success'] is True
        assert 'file_path' in result

        # 检查文件是否创建
        json_file = data_exporter.export_dir / filename
        assert json_file.exists()

        # 检查文件内容
        with open(json_file, 'r') as f:
            exported_data = json.load(f)

        assert isinstance(exported_data, list)
        assert len(exported_data) == len(sample_stock_data)

    def test_export_to_excel(self, data_exporter, sample_stock_data):
        """测试导出到Excel格式"""
        filename = "test_export.xlsx"

        # 导出数据
        result = data_exporter.export_to_excel(sample_stock_data, filename)

        # 检查导出结果
        assert result['success'] is True
        assert 'file_path' in result

        # 检查文件是否创建
        excel_file = data_exporter.export_dir / filename
        assert excel_file.exists()

    def test_export_to_parquet(self, data_exporter, sample_stock_data):
        """测试导出到Parquet格式"""
        filename = "test_export.parquet"

        # 导出数据
        result = data_exporter.export_to_parquet(sample_stock_data, filename)

        # 检查导出结果
        assert result['success'] is True
        assert 'file_path' in result

        # 检查文件是否创建
        parquet_file = data_exporter.export_dir / filename
        assert parquet_file.exists()

    def test_export_to_pickle(self, data_exporter, sample_stock_data):
        """测试导出到Pickle格式"""
        filename = "test_export.pkl"

        # 导出数据
        result = data_exporter.export_to_pickle(sample_stock_data, filename)

        # 检查导出结果
        assert result['success'] is True
        assert 'file_path' in result

        # 检查文件是否创建
        pickle_file = data_exporter.export_dir / filename
        assert pickle_file.exists()

        # 检查文件内容
        imported_data = pd.read_pickle(pickle_file)
        pd.testing.assert_frame_equal(imported_data, sample_stock_data)

    def test_export_multiple_data_models(self, data_exporter, sample_stock_data):
        """测试导出多个数据模型"""
        # 创建多个数据模型
        data_models = [
            {'name': 'stocks', 'data': sample_stock_data},
            {'name': 'indices', 'data': sample_stock_data.head(10)},
        ]

        filename = "multiple_export.zip"

        # 导出多个数据模型
        result = data_exporter.export_multiple_data_models(data_models, filename)

        # 检查导出结果
        assert result['success'] is True
        assert 'file_path' in result

        # 检查ZIP文件是否创建
        zip_file = data_exporter.export_dir / filename
        assert zip_file.exists()

        # 检查ZIP文件内容
        with zipfile.ZipFile(zip_file, 'r') as zf:
            file_list = zf.namelist()
            assert len(file_list) == len(data_models)

    def test_export_with_metadata(self, data_exporter, sample_stock_data):
        """测试带元数据导出"""
        filename = "metadata_export.json"
        metadata = {
            'source': 'test_database',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0',
            'description': 'Test export with metadata'
        }

        # 导出带元数据的数据
        result = data_exporter.export_with_metadata(sample_stock_data, filename, metadata)

        # 检查导出结果
        assert result['success'] is True
        assert 'file_path' in result

        # 检查文件内容
        json_file = data_exporter.export_dir / filename
        with open(json_file, 'r') as f:
            exported_content = json.load(f)

        assert 'data' in exported_content
        assert 'metadata' in exported_content
        assert exported_content['metadata'] == metadata

    def test_export_to_buffer(self, data_exporter, sample_stock_data):
        """测试导出到内存缓冲区"""
        # 测试CSV格式缓冲区导出
        csv_buffer = data_exporter.export_to_buffer(sample_stock_data, 'csv')
        assert csv_buffer is not None
        assert len(csv_buffer) > 0

        # 从缓冲区重新读取数据
        buffer_data = pd.read_csv(pd.io.common.StringIO(csv_buffer))
        assert len(buffer_data) == len(sample_stock_data)

    def test_export_compression(self, data_exporter, sample_stock_data):
        """测试导出压缩"""
        filename = "compressed_export.csv.gz"

        # 导出压缩数据
        result = data_exporter.export_with_compression(sample_stock_data, filename, compression='gzip')

        # 检查导出结果
        assert result['success'] is True
        assert 'file_path' in result

        # 检查压缩文件是否创建
        compressed_file = data_exporter.export_dir / filename
        assert compressed_file.exists()

    def test_export_chunked_data(self, data_exporter):
        """测试分块导出大数据"""
        # 创建大数据集
        large_data = pd.DataFrame({
            'col1': np.random.randn(10000),
            'col2': np.random.randn(10000),
            'col3': np.random.randint(1, 100, 10000)
        })

        chunk_size = 1000
        base_filename = "chunked_export_{}.csv"

        # 分块导出
        result = data_exporter.export_chunked_data(large_data, base_filename, chunk_size)

        # 检查导出结果
        assert result['success'] is True
        assert 'files_created' in result
        assert result['files_created'] == 10  # 10000 / 1000 = 10

        # 检查文件是否创建
        for i in range(result['files_created']):
            chunk_file = data_exporter.export_dir / f"chunked_export_{i}.csv"
            assert chunk_file.exists()

            # 检查每个文件的大小
            chunk_data = pd.read_csv(chunk_file)
            assert len(chunk_data) == chunk_size

    def test_export_error_handling(self, data_exporter):
        """测试导出错误处理"""
        # 创建无效数据
        invalid_data = pd.DataFrame({
            'col1': [float('inf'), float('-inf'), float('nan')]
        })

        filename = "error_export.csv"

        # 测试错误处理
        result = data_exporter.export_with_error_handling(invalid_data, filename)

        # 检查错误处理结果
        assert 'success' in result
        assert 'errors' in result

        # 即使有错误，也应该创建文件或提供替代方案
        assert 'file_path' in result or 'buffer' in result

    def test_export_format_validation(self, data_exporter, sample_stock_data):
        """测试导出格式验证"""
        # 测试支持的格式
        supported_formats = ['csv', 'json', 'excel', 'parquet', 'pickle']

        for fmt in supported_formats:
            filename = f"format_test.{data_exporter._get_file_extension(fmt)}"

            # 尝试导出
            result = data_exporter.export_data(sample_stock_data, filename, fmt)

            # 检查导出成功
            assert result['success'] is True

    def test_export_directory_management(self, data_exporter, sample_stock_data):
        """测试导出目录管理"""
        # 测试目录自动创建
        sub_dir = "subdirectory/test"
        filename = f"{sub_dir}/test_file.csv"

        result = data_exporter.export_to_csv(sample_stock_data, filename)

        # 检查子目录是否创建
        sub_path = data_exporter.export_dir / sub_dir
        assert sub_path.exists()
        assert sub_path.is_dir()

        # 检查文件是否在子目录中
        csv_file = sub_path / "test_file.csv"
        assert csv_file.exists()

    def test_export_file_overwrite_handling(self, data_exporter, sample_stock_data):
        """测试导出文件覆盖处理"""
        filename = "overwrite_test.csv"

        # 第一次导出
        result1 = data_exporter.export_to_csv(sample_stock_data, filename)
        assert result1['success'] is True

        # 获取第一次导出文件的修改时间
        csv_file = data_exporter.export_dir / filename
        first_mtime = csv_file.stat().st_mtime

        # 等待一小段时间
        import time
        time.sleep(0.1)

        # 第二次导出（覆盖）
        result2 = data_exporter.export_to_csv(sample_stock_data, filename, overwrite=True)
        assert result2['success'] is True

        # 检查文件被覆盖
        second_mtime = csv_file.stat().st_mtime
        assert second_mtime > first_mtime

    def test_export_statistics_tracking(self, data_exporter, sample_stock_data):
        """测试导出统计跟踪"""
        # 执行多个导出操作
        formats = ['csv', 'json', 'parquet']
        for fmt in formats:
            filename = f"stats_test.{data_exporter._get_file_extension(fmt)}"
            data_exporter.export_data(sample_stock_data, filename, fmt)

        # 获取导出统计
        stats = data_exporter.get_export_statistics()

        # 检查统计信息
        assert isinstance(stats, dict)
        assert 'total_exports' in stats
        assert 'formats_used' in stats
        assert stats['total_exports'] >= len(formats)

    def test_export_cleanup_old_files(self, data_exporter, sample_stock_data):
        """测试导出清理旧文件"""
        # 创建一些旧文件
        old_files = []
        for i in range(5):
            filename = f"old_file_{i}.csv"
            data_exporter.export_to_csv(sample_stock_data, filename)
            old_files.append(data_exporter.export_dir / filename)

            # 设置旧文件时间
            import time
            old_time = time.time() - (30 * 24 * 60 * 60)  # 30天前
            os.utime(old_files[-1], (old_time, old_time))

        # 清理30天前的文件
        cleanup_result = data_exporter.cleanup_old_files(days=30)

        # 检查清理结果
        assert 'files_removed' in cleanup_result
        assert cleanup_result['files_removed'] >= len(old_files)

    def test_export_batch_processing(self, data_exporter, sample_stock_data):
        """测试导出批处理"""
        # 创建批处理导出请求
        batch_requests = [
            {'data': sample_stock_data, 'filename': 'batch_1.csv', 'format': 'csv'},
            {'data': sample_stock_data.head(10), 'filename': 'batch_2.json', 'format': 'json'},
            {'data': sample_stock_data.head(5), 'filename': 'batch_3.xlsx', 'format': 'excel'},
        ]

        # 执行批处理导出
        batch_result = data_exporter.export_batch(batch_requests)

        # 检查批处理结果
        assert isinstance(batch_result, list)
        assert len(batch_result) == len(batch_requests)

        # 检查每个请求都成功
        for result in batch_result:
            assert result['success'] is True

    def test_export_performance_monitoring(self, data_exporter, sample_stock_data):
        """测试导出性能监控"""
        # 执行多个导出操作
        operations = 10

        import time
        start_time = time.time()

        for i in range(operations):
            filename = f"perf_test_{i}.csv"
            data_exporter.export_to_csv(sample_stock_data, filename)

        end_time = time.time()

        # 获取性能统计
        perf_stats = data_exporter.get_performance_statistics()

        # 检查性能统计
        assert isinstance(perf_stats, dict)
        assert 'avg_export_time' in perf_stats
        assert 'total_data_processed' in perf_stats

        # 检查实际性能
        total_time = end_time - start_time
        assert perf_stats['avg_export_time'] > 0
        assert perf_stats['total_data_processed'] >= operations * len(sample_stock_data)

    def test_export_resource_management(self, data_exporter):
        """测试导出资源管理"""
        # 创建大数据进行资源管理测试
        large_data = pd.DataFrame({
            'col1': np.random.randn(50000),
            'col2': np.random.randn(50000),
            'col3': ['string_' + str(i) for i in range(50000)]
        })

        filename = "resource_test.csv"

        # 监控资源使用
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss

        # 执行大数据导出
        result = data_exporter.export_to_csv(large_data, filename)

        memory_after = process.memory_info().rss

        # 检查导出成功
        assert result['success'] is True

        # 检查内存使用合理（不应无限增长）
        memory_increase = memory_after - memory_before
        assert memory_increase < 500 * 1024 * 1024  # 500MB限制
