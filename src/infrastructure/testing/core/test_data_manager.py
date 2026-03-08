import time
#!/usr/bin/env python3
"""
RQA2025 测试数据管理器
Testing Data Manager

管理测试数据的生成、存储和清理。
"""

import json
import csv
import logging
from typing import Dict, Any, List, Optional, Iterator
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
import shutil

# 配置日志
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class TestDataSet:

    """测试数据集"""
    dataset_id: str
    name: str
    description: str
    data_type: str  # market_data, user_data, system_config, etc.
    data_format: str  # json, csv, sql, etc.
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None


    def is_expired(self) -> bool:

        """检查是否过期"""
        return self.expires_at and datetime.now() > self.expires_at


@dataclass
class DataTemplate:

    """数据模板"""
    template_id: str
    name: str
    description: str
    template_type: str
    template_data: Dict[str, Any]
    variables: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class TestDataManager:

    """
    测试数据管理器
    负责测试数据的生成、存储、检索和清理
    """


    def __init__(self, data_dir: Optional[str] = None):

        self.data_dir = Path(data_dir) if data_dir else Path(tempfile.gettempdir()) / "rqa2025_test_data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.datasets: Dict[str, TestDataSet] = {}
        self.templates: Dict[str, DataTemplate] = {}

        # 加载默认模板
        self._load_default_templates()

        logger.info(f"测试数据管理器已初始化，数据目录: {self.data_dir}")


    def create_dataset(self, dataset: TestDataSet) -> str:

        """创建数据集"""
        self.datasets[dataset.dataset_id] = dataset

        # 保存到文件系统
        self._save_dataset_to_file(dataset)

        logger.info(f"创建数据集: {dataset.dataset_id}")
        return dataset.dataset_id


    def get_dataset(self, dataset_id: str) -> Optional[TestDataSet]:

        """获取数据集"""
        return self.datasets.get(dataset_id)

    def list_datasets(self, data_type: Optional[str] = None) -> List[TestDataSet]:

        """列出数据集"""
        datasets = list(self.datasets.values())

        if data_type:
            datasets = [d for d in datasets if d.data_type == data_type]

        return datasets

    def delete_dataset(self, dataset_id: str) -> bool:

        """删除数据集"""
        if dataset_id in self.datasets:
            dataset = self.datasets[dataset_id]

            # 删除文件
            self._delete_dataset_file(dataset)

            # 从内存中删除
            del self.datasets[dataset_id]

            logger.info(f"删除数据集: {dataset_id}")
            return True

        return False


    def generate_test_data(self, template_id: str, variables: Dict[str, Any] = None) -> TestDataSet:

        """基于模板生成测试数据"""
        template = self.templates.get(template_id)
        if not template:
            raise ValueError(f"模板不存在: {template_id}")

        variables = variables or {}

        # 合并变量
        merged_vars = {**template.variables, **variables}

        # 生成数据
        generated_data = self._generate_data_from_template(template, merged_vars)

        # 创建数据集
        dataset_id = f"generated_{template_id}_{datetime.now().strftime('%Y % m % d % H % M % S % f')}"

        dataset = TestDataSet(
            dataset_id=dataset_id,
            name=f"Generated from {template.name}",
            description=f"Generated test data from template {template.name}",
            data_type=template.template_type,
            data_format="json",
            data=generated_data,
            metadata={
                'template_id': template_id,
                'variables': merged_vars,
                'generated_at': datetime.now().isoformat()
            }
        )

        return self.create_dataset(dataset)


    def create_template(self, template: DataTemplate) -> str:

        """创建数据模板"""
        self.templates[template.template_id] = template
        logger.info(f"创建数据模板: {template.template_id}")
        return template.template_id

    def get_template(self, template_id: str) -> Optional[DataTemplate]:

        """获取数据模板"""
        return self.templates.get(template_id)

    def list_templates(self, template_type: Optional[str] = None) -> List[DataTemplate]:

        """列出数据模板"""
        templates = list(self.templates.values())

        if template_type:
            templates = [t for t in templates if t.template_type == template_type]

        return templates

    def cleanup_expired_data(self) -> int:

        """清理过期数据"""
        expired_datasets = []
        current_time = datetime.now()

        for dataset_id, dataset in self.datasets.items():
            if dataset.is_expired():
                expired_datasets.append(dataset_id)

        # 删除过期数据集
        for dataset_id in expired_datasets:
            self.delete_dataset(dataset_id)

        logger.info(f"清理了 {len(expired_datasets)} 个过期数据集")
        return len(expired_datasets)


    def export_dataset(self, dataset_id: str, export_path: str, format: str = "json") -> bool:

        """导出数据集"""
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            return False

        export_path = Path(export_path)

        try:
            if format == "json":
                with open(export_path, 'w', encoding='utf - 8') as f:
                    json.dump(dataset.data, f, indent=2, default=str)
            elif format == "csv":
                self._export_as_csv(dataset, export_path)
            else:
                raise ValueError(f"不支持的导出格式: {format}")

            logger.info(f"导出数据集 {dataset_id} 到 {export_path}")
            return True

        except Exception as e:
            logger.error(f"导出数据集失败: {str(e)}")
            return False


    def import_dataset(self, import_path: str, dataset_name: str, data_type: str) -> Optional[str]:

        """导入数据集"""
        import_path = Path(import_path)

        try:
            if import_path.suffix.lower() == '.json':
                with open(import_path, 'r', encoding='utf - 8') as f:
                    data = json.load(f)
            elif import_path.suffix.lower() == '.csv':
                data = self._import_from_csv(import_path)
            else:
                raise ValueError(f"不支持的文件格式: {import_path.suffix}")

            dataset_id = f"imported_{datetime.now().strftime('%Y % m % d % H % M % S % f')}"

            dataset = TestDataSet(
                dataset_id=dataset_id,
                name=dataset_name,
                description=f"Imported from {import_path}",
                data_type=data_type,
                data_format=import_path.suffix[1:],  # Remove the dot
                data=data,
                metadata={
                    'imported_from': str(import_path),
                    'imported_at': datetime.now().isoformat()
                }
            )

            return self.create_dataset(dataset)

        except Exception as e:
            logger.error(f"导入数据集失败: {str(e)}")
            return None


    def _save_dataset_to_file(self, dataset: TestDataSet):

        """保存数据集到文件"""
        file_path = self.data_dir / f"{dataset.dataset_id}.json"

        try:
            with open(file_path, 'w', encoding='utf - 8') as f:
                json.dump({
                    'dataset_id': dataset.dataset_id,
                    'name': dataset.name,
                    'description': dataset.description,
                    'data_type': dataset.data_type,
                    'data_format': dataset.data_format,
                    'data': dataset.data,
                    'metadata': dataset.metadata,
                    'created_at': dataset.created_at.isoformat(),
                    'expires_at': dataset.expires_at.isoformat() if dataset.expires_at else None
                }, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"保存数据集到文件失败: {str(e)}")


    def _delete_dataset_file(self, dataset: TestDataSet):

        """删除数据集文件"""
        file_path = self.data_dir / f"{dataset.dataset_id}.json"

        try:
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            logger.error(f"删除数据集文件失败: {str(e)}")

    def _load_default_templates(self):

        """加载默认数据模板"""
        # 市场数据模板
        market_data_template = DataTemplate(
            template_id="market_data_template",
            name="市场数据模板",
            description="生成模拟市场数据",
            template_type="market_data",
            template_data={
                "symbol": "${symbol}",
                "price": "${price}",
                "volume": "${volume}",
                "timestamp": "${timestamp}"
            },
            variables={
                "symbol": "AAPL",
                "price": 150.0,
                "volume": 1000,
                "timestamp": datetime.now().isoformat()
            }
        )

        # 用户数据模板
        user_data_template = DataTemplate(
            template_id="user_data_template",
            name="用户数据模板",
            description="生成模拟用户数据",
            template_type="user_data",
            template_data={
                "user_id": "${user_id}",
                "username": "${username}",
                "email": "${email}",
                "balance": "${balance}",
                "created_at": "${created_at}"
            },
            variables={
                "user_id": 1,
                "username": "test_user",
                "email": "test@example.com",
                "balance": 10000.0,
                "created_at": datetime.now().isoformat()
            }
        )

        # 系统配置模板
        system_config_template = DataTemplate(
            template_id="system_config_template",
            name="系统配置模板",
            description="生成模拟系统配置",
            template_type="system_config",
            template_data={
                "service_name": "${service_name}",
                "version": "${version}",
                "environment": "${environment}",
                "database_url": "${database_url}",
                "cache_enabled": "${cache_enabled}"
            },
            variables={
                "service_name": "trading_engine",
                "version": "1.0.0",
                "environment": "test",
                "database_url": "postgresql://test:test@localhost / testdb",
                "cache_enabled": True
            }
        )

        self.templates = {
            market_data_template.template_id: market_data_template,
            user_data_template.template_id: user_data_template,
            system_config_template.template_id: system_config_template
        }


    def _generate_data_from_template(self, template: DataTemplate, variables: Dict[str, Any]) -> Any:

        """从模板生成数据"""
        # 简单的变量替换
        template_str = json.dumps(template.template_data)
        for var_name, var_value in variables.items():
            placeholder = f"${{{var_name}}}"
            template_str = template_str.replace(placeholder, str(var_value))

        return json.loads(template_str)

    def _export_as_csv(self, dataset: TestDataSet, export_path: Path):

        """导出为CSV格式"""
        if isinstance(dataset.data, list) and dataset.data:
            fieldnames = dataset.data[0].keys()
            with open(export_path, 'w', newline='', encoding='utf - 8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(dataset.data)
        else:
            raise ValueError("数据集数据格式不支持CSV导出")

    def _import_from_csv(self, import_path: Path) -> List[Dict[str, Any]]:

        """从CSV导入数据"""
        data = []
        with open(import_path, 'r', encoding='utf - 8') as f:
            reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
        return data

# 创建全局测试数据管理器实例
_test_data_manager = None

def get_test_data_manager() -> TestDataManager:

    """获取全局测试数据管理器实例"""
    global _test_data_manager
    if _test_data_manager is None:
        _test_data_manager = TestDataManager()
    return _test_data_manager


__all__ = [
    'TestDataManager', 'TestDataSet', 'DataTemplate', 'get_test_data_manager'
]
