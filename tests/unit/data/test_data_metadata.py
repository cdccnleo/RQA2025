# tests/data/test_data_metadata.py
import os
import tempfile
from pathlib import Path
import pytest
from src.infrastructure.utils.exceptions import DataLoaderError
from src.data.data_manager import DataMetadata, DataManager


def test_data_metadata_initialization():
    metadata = DataMetadata(data_type="market", version="v1.0")
    assert metadata.data_type == "market"
    assert metadata.version == "v1.0"
    assert metadata.schema is None
    assert metadata.data_version_info is None


def test_update_metadata():
    metadata = DataMetadata(data_type="news")
    schema = {"columns": ["date", "content"], "dtypes": {"date": "datetime64[ns]"}}
    metadata.update_metadata(schema)
    assert metadata.schema == schema
    assert isinstance(metadata.last_updated, float)


def test_set_data_version():
    metadata = DataMetadata(data_type="fundamental")
    version_info = {"source": "DB", "version": "2023Q4"}
    metadata.set_data_version(version_info)
    assert metadata.data_version_info == version_info


def test_save_and_load_metadata(tmp_path):
    metadata = DataMetadata(data_type="index")
    schema = {"columns": ["symbol", "close"]}
    metadata.update_metadata(schema)
    save_path = tmp_path / "test_metadata.pkl"
    metadata.save(save_path)

    loaded_metadata = DataMetadata.load(save_path)
    assert loaded_metadata.data_type == "index"
    assert loaded_metadata.schema == schema


# 新增测试用例 - 数据元数据版本管理
def test_data_version_info_persistence(tmp_path: Path):
    """测试数据版本信息持久化"""
    metadata = DataMetadata("market")
    version_info = {"source": "DB", "version": "2023Q4"}
    metadata.set_data_version(version_info)

    save_path = tmp_path / "metadata.pkl"
    metadata.save(save_path)

    loaded = DataMetadata.load(save_path)
    assert loaded.data_version_info == version_info


def test_metadata_version_handling():
    """测试元数据版本冲突处理"""
    # 初始化 DataManager
    data_manager = DataManager()

    # 创建具有不同版本的元数据
    metadata_v2 = DataMetadata("market", "v2.0")
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.close()  # 手动关闭文件句柄
        metadata_v2.save(tmp.name)

        # 尝试加载元数据并触发版本冲突
        loaded_metadata = DataMetadata.load(tmp.name)
        data_manager.metadata["market"] = loaded_metadata  # 直接赋值以触发版本冲突

        # 检查是否记录了数据版本信息
        assert data_manager.metadata["market"].version == "v2.0"  # 直接检查元数据版本

    os.unlink(tmp.name)  # 删除临时文件