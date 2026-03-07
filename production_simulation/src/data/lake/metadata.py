"""
数据元数据管理类定义
"""
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, Optional
import json


@dataclass
class DataMetadata:

    """
    数据元数据类，用于描述和管理数据集的元信息
    """
    data_type: str
    source: str
    created_at: datetime
    version: str
    record_count: int
    columns: list
    description: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """将元数据转换为字典"""
        result = asdict(self)
        # 处理datetime对象的序列化
        if isinstance(result['created_at'], datetime):
            result['created_at'] = result['created_at'].isoformat()
        return result

    def to_json(self) -> str:
        """将元数据转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, metadata_dict: Dict[str, Any]) -> 'DataMetadata':
        """从字典创建DataMetadata实例"""
        # 解析datetime字符串
        if isinstance(metadata_dict.get('created_at'), str):
            metadata_dict['created_at'] = datetime.fromisoformat(metadata_dict['created_at'])
        return cls(**metadata_dict)

    def validate(self) -> bool:
        """验证元数据的完整性和有效性"""
        required_fields = ['data_type', 'source',
                           'created_at', 'version', 'record_count', 'columns']
        for field in required_fields:
            if not getattr(self, field, None):
                return False
        return True


class MetadataManager:

    """
    元数据管理器，用于处理元数据的存储和检索
    """

    def __init__(self, storage_path: str):

        self.storage_path = storage_path

    def save_metadata(self, metadata: DataMetadata, data_id: str) -> bool:
        """保存元数据到文件"""
        try:
            with open(f"{self.storage_path}/{data_id}_metadata.json", 'w', encoding='utf - 8') as f:
                f.write(metadata.to_json())
            return True
        except Exception as e:
            print(f"保存元数据失败: {e}")
            return False

    def load_metadata(self, data_id: str) -> Optional[DataMetadata]:
        """从文件加载元数据"""
        try:
            with open(f"{self.storage_path}/{data_id}_metadata.json", 'r', encoding='utf - 8') as f:
                metadata_dict = json.load(f)
            return DataMetadata.from_dict(metadata_dict)
        except FileNotFoundError:
            print(f"元数据文件不存在: {data_id}")
            return None
        except Exception as e:
            print(f"加载元数据失败: {e}")
            return None
            return None
