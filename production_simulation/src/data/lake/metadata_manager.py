# 使用基础设施层日志，避免依赖上层组件
try:
    from src.infrastructure.logging import get_infrastructure_logger
except ImportError:
    # 降级到标准logging
    import logging

    def get_infrastructure_logger(name):

        logger = logging.getLogger(name)
        logger.warning("无法导入基础设施层日志，使用标准logging")
        return logger

from typing import Dict, Any, List, Optional
from src.infrastructure.logging import get_infrastructure_logger
from dataclasses import dataclass, asdict
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

logger = get_infrastructure_logger('__name__')


@dataclass
class MetadataSchema:

    """元数据模式"""
    dataset_name: str
    description: str = ""
    schema_version: str = "1.0"
    created_at: str = ""
    updated_at: str = ""
    columns: List[Dict[str, Any]] = None
    tags: List[str] = None
    owner: str = ""
    access_level: str = "public"  # public, private, restricted

    def __post_init__(self):

        if self.columns is None:
            self.columns = []
        if self.tags is None:
            self.tags = []
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = datetime.now().isoformat()


class MetadataManager:

    """
    元数据管理器
    支持数据湖的元数据存储、查询、版本控制等
    """

    def __init__(self, metadata_path: str = "metadata"):

        self.metadata_path = Path(metadata_path)
        self.metadata_path.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    def create_schema(self, dataset_name: str, data: pd.DataFrame, **kwargs) -> MetadataSchema:
        """创建元数据模式"""
        try:
            # 分析数据模式
            columns = []
            for col_name, col_type in data.dtypes.items():
                column_info = {
                    'name': col_name,
                    'type': str(col_type),
                    'nullable': bool(data[col_name].isnull().any()),
                    'unique_values': int(data[col_name].nunique()),
                    'sample_values': data[col_name].dropna().head(5).tolist()
                }
                columns.append(column_info)

            # 创建模式
            schema = MetadataSchema(
                dataset_name=dataset_name,
                columns=columns,
                **kwargs
            )

            # 保存模式
            self._save_schema(schema)

            self.logger.info(f"为数据集 {dataset_name} 创建了元数据模式")
            return schema

        except Exception as e:
            self.logger.error(f"创建元数据模式失败: {e}")
            raise

    def get_schema(self, dataset_name: str) -> Optional[MetadataSchema]:
        """获取元数据模式"""
        try:
            schema_file = self.metadata_path / f"{dataset_name}_schema.json"

            if not schema_file.exists():
                return None

            with open(schema_file, 'r', encoding='utf-8') as f:
                schema_data = json.load(f)

            return MetadataSchema(**schema_data)

        except Exception as e:
            self.logger.error(f"获取元数据模式失败: {e}")
            return None

    def update_schema(self, dataset_name: str, **kwargs) -> bool:
        """更新元数据模式"""
        try:
            schema = self.get_schema(dataset_name)
            if not schema:
                return False

            # 更新字段
            for key, value in kwargs.items():
                if hasattr(schema, key):
                    setattr(schema, key, value)

            schema.updated_at = datetime.now().isoformat()

            # 保存更新后的模式
            self._save_schema(schema)

            self.logger.info(f"更新了数据集 {dataset_name} 的元数据模式")
            return True

        except Exception as e:
            self.logger.error(f"更新元数据模式失败: {e}")
            return False

    def list_schemas(self) -> List[str]:
        """列出所有元数据模式"""
        try:
            schemas = []
            for schema_file in self.metadata_path.glob("*_schema.json"):
                dataset_name = schema_file.stem.replace("_schema", "")
                schemas.append(dataset_name)

            return sorted(schemas)
        except Exception as e:
            self.logger.error(f"列出元数据模式失败: {e}")
            return []

    def delete_schema(self, dataset_name: str) -> bool:
        """删除元数据模式"""
        try:
            schema_file = self.metadata_path / f"{dataset_name}_schema.json"

            if schema_file.exists():
                schema_file.unlink()
                self.logger.info(f"删除了数据集 {dataset_name} 的元数据模式")
                return True
            else:
                self.logger.warning(f"元数据模式不存在: {dataset_name}")
                return False

        except Exception as e:
            self.logger.error(f"删除元数据模式失败: {e}")
            return False

    def search_schemas(self, query: str) -> List[MetadataSchema]:
        """搜索元数据模式"""
        try:
            matching_schemas = []

            for dataset_name in self.list_schemas():
                schema = self.get_schema(dataset_name)
                if schema and self._matches_query(schema, query):
                    matching_schemas.append(schema)

            return matching_schemas

        except Exception as e:
            self.logger.error(f"搜索元数据模式失败: {e}")
            return []

    def get_schema_stats(self) -> Dict[str, Any]:
        """获取元数据统计信息"""
        try:
            stats = {
                'total_datasets': 0,
                'total_columns': 0,
                'avg_columns_per_dataset': 0,
                'most_common_types': {},
                'recent_updates': [],
                'access_levels': {}
            }

            schemas = []
            for dataset_name in self.list_schemas():
                schema = self.get_schema(dataset_name)
                if schema:
                    schemas.append(schema)
                    stats['total_datasets'] += 1
                    stats['total_columns'] += len(schema.columns)

                    # 统计数据类型
                    for column in schema.columns:
                        col_type = column['type']
                        if col_type not in stats['most_common_types']:
                            stats['most_common_types'][col_type] = 0
                        stats['most_common_types'][col_type] += 1

                    # 统计访问级别
                    access_level = schema.access_level
                    if access_level not in stats['access_levels']:
                        stats['access_levels'][access_level] = 0
                    stats['access_levels'][access_level] += 1

                    # 记录最近更新
                    try:
                        updated_at = datetime.fromisoformat(schema.updated_at)
                        stats['recent_updates'].append({
                            'dataset': dataset_name,
                            'updated_at': updated_at
                        })
                    except BaseException:
                        pass

            # 计算平均列数
            if stats['total_datasets'] > 0:
                stats['avg_columns_per_dataset'] = stats['total_columns'] / stats['total_datasets']

            # 排序最近更新
            stats['recent_updates'].sort(key=lambda x: x['updated_at'], reverse=True)
            stats['recent_updates'] = stats['recent_updates'][:10]  # 只保留最近10个

            return stats

        except Exception as e:
            self.logger.error(f"获取元数据统计信息失败: {e}")
            return {}

    def validate_schema(self, dataset_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """验证数据是否符合元数据模式"""
        try:
            schema = self.get_schema(dataset_name)
            if not schema:
                return {'valid': False, 'error': 'Schema not found'}

            validation_result = {
                'valid': True,
                'warnings': [],
                'errors': []
            }

            # 检查列是否存在
            schema_columns = {col['name'] for col in schema.columns}
            data_columns = set(data.columns)

            missing_columns = schema_columns - data_columns
            extra_columns = data_columns - schema_columns

            if missing_columns:
                validation_result['errors'].append(f"Missing columns: {missing_columns}")
                validation_result['valid'] = False

            if extra_columns:
                validation_result['warnings'].append(f"Extra columns: {extra_columns}")

            # 检查数据类型
            for column in schema.columns:
                col_name = column['name']
                if col_name in data.columns:
                    expected_type = column['type']
                    actual_type = str(data[col_name].dtype)

                    if expected_type != actual_type:
                        validation_result['warnings'].append(
                            f"Column {col_name}: expected {expected_type}, got {actual_type}"
                        )

            return validation_result

        except Exception as e:
            self.logger.error(f"验证模式失败: {e}")
            return {'valid': False, 'error': str(e)}

    def _save_schema(self, schema: MetadataSchema):
        """保存元数据模式"""
        schema_file = self.metadata_path / f"{schema.dataset_name}_schema.json"

        with open(schema_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(schema), f, ensure_ascii=False, indent=2)

    def _matches_query(self, schema: MetadataSchema, query: str) -> bool:
        """检查模式是否匹配查询"""
        query_lower = query.lower()

        # 检查数据集名称
        if query_lower in schema.dataset_name.lower():
            return True

        # 检查描述
        if query_lower in schema.description.lower():
            return True

        # 检查标签
        for tag in schema.tags:
            if query_lower in tag.lower():
                return True

        # 检查列名
        for column in schema.columns:
            if query_lower in column['name'].lower():
                return True

        return False
