from __future__ import annotations
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd
from datetime import datetime

from src.infrastructure.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureMetadata:
    """特征元数据管理类，用于跟踪和管理特征生成过程的元数据"""

    def __init__(
            self,
            feature_params: Optional[Dict[str, Any]] = None,
            data_source_version: str = "1.0.0",
            feature_list: Optional[List[str]] = None,
            metadata_path: Optional[str] = None
    ):
        """
        初始化特征元数据管理器

        Args:
            feature_params: 特征生成参数
            data_source_version: 数据源版本
            feature_list: 初始特征列表
            metadata_path: 元数据文件路径
        """
        self.logger = logger
        self.metadata = {
            "version": "1.0.0",  # 元数据版本
            "data_source_version": data_source_version,
            "feature_params": feature_params or {},
            "feature_columns": feature_list or [],  # 兼容旧的feature_list参数
            "creation_time": datetime.now().isoformat(),
            "last_update_time": datetime.now().isoformat(),
            "update_history": []
        }

        if metadata_path:
            self._load_metadata(metadata_path)

    @property
    def feature_list(self) -> List[str]:
        """获取特征列表（兼容旧接口）"""
        return self.metadata["feature_columns"]

    @property
    def last_updated(self) -> str:
        """获取最后更新时间（兼容旧接口）"""
        return self.metadata["last_update_time"]

    def _load_metadata(self, path: str) -> None:
        """
        从文件加载元数据

        Args:
            path: 元数据文件路径
        """
        try:
            path_obj = Path(path)
            if not path_obj.is_file():
                raise FileNotFoundError(f"元数据文件不存在: {path}")

            with open(path, 'r', encoding='utf-8') as f:
                loaded_metadata = json.load(f)

            # 验证加载的元数据
            required_keys = [
                "version",
                "data_source_version",
                "feature_params",
                "feature_columns",
                "creation_time",
                "last_update_time",
                "update_history"
            ]

            missing_keys = [key for key in required_keys if key not in loaded_metadata]
            if missing_keys:
                raise ValueError(f"元数据缺少必要字段: {', '.join(missing_keys)}")

            self.metadata = loaded_metadata
            self.logger.info(f"成功加载元数据: {path}")

        except Exception as e:
            self.logger.error(f"加载元数据失败: {str(e)}")
            raise

    def save_metadata(self, path: str) -> None:
        """
        保存元数据到文件

        Args:
            path: 元数据文件路径
        """
        try:
            # 确保目录存在
            path_obj = Path(path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)

            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)

            self.logger.info(f"成功保存元数据: {path}")

        except Exception as e:
            self.logger.error(f"保存元数据失败: {str(e)}")
            raise

    def update_feature_columns(self, columns: List[str]) -> None:
        """
        更新特征列列表

        Args:
            columns: 特征列名列表
        """
        if not isinstance(columns, list):
            raise TypeError("columns必须是列表类型")

        if not all(isinstance(col, str) for col in columns):
            raise ValueError("所有列名必须是字符串类型")

        # 记录更新
        update_record = {
            "timestamp": datetime.now().isoformat(),
            "action": "update_feature_columns",
            "old_columns": self.metadata["feature_columns"],
            "new_columns": columns
        }

        self.metadata["feature_columns"] = columns
        self.metadata["last_update_time"] = datetime.now().isoformat()
        self.metadata["update_history"].append(update_record)

        self.logger.info(f"更新特征列完成，当前特征数: {len(columns)}")

    def update_feature_params(self, params: Dict[str, Any]) -> None:
        """
        更新特征生成参数

        Args:
            params: 特征生成参数
        """
        if not isinstance(params, dict):
            raise TypeError("params必须是字典类型")

        # 记录更新
        update_record = {
            "timestamp": datetime.now().isoformat(),
            "action": "update_feature_params",
            "old_params": self.metadata["feature_params"],
            "new_params": params
        }

        self.metadata["feature_params"] = params
        self.metadata["last_update_time"] = datetime.now().isoformat()
        self.metadata["update_history"].append(update_record)

        self.logger.info("更新特征参数完成")

    def validate_features(self, data: pd.DataFrame) -> bool:
        """
        验证特征数据

        Args:
            data: 特征数据DataFrame

        Returns:
            验证是否通过
        """
        try:
            # 检查特征列是否存在
            missing_cols = [col for col in self.metadata["feature_columns"] if col not in data.columns]
            if missing_cols:
                raise ValueError(f"缺少特征列: {', '.join(missing_cols)}")

            # 检查数据类型
            non_numeric_cols = [col for col in self.metadata["feature_columns"]
                                if not pd.api.types.is_numeric_dtype(data[col])]
            if non_numeric_cols:
                raise ValueError(f"非数值类型特征列: {', '.join(non_numeric_cols)}")

            # 检查缺失值
            na_cols = data[self.metadata["feature_columns"]].isna().any()
            na_cols = na_cols[na_cols].index.tolist()
            if na_cols:
                raise ValueError(f"包含缺失值的特征列: {', '.join(na_cols)}")

            # 检查无穷值
            inf_cols = data[self.metadata["feature_columns"]].isin([float('inf'), float('-inf')]).any()
            inf_cols = inf_cols[inf_cols].index.tolist()
            if inf_cols:
                raise ValueError(f"包含无穷值的特征列: {', '.join(inf_cols)}")

            self.logger.info("特征验证通过")
            return True

        except Exception as e:
            self.logger.error(f"特征验证失败: {str(e)}")
            return False

    def get_feature_info(self) -> Dict[str, Any]:
        """
        获取特征信息摘要

        Returns:
            特征信息字典
        """
        return {
            "total_features": len(self.metadata["feature_columns"]),
            "feature_list": self.metadata["feature_columns"],
            "data_source_version": self.metadata["data_source_version"],
            "last_update": self.metadata["last_update_time"],
            "feature_params": self.metadata["feature_params"]
        }

    def __str__(self) -> str:
        """返回元数据的字符串表示"""
        info = self.get_feature_info()
        return (
            f"特征元数据:\n"
            f"- 特征总数: {info['total_features']}\n"
            f"- 数据源版本: {info['data_source_version']}\n"
            f"- 最后更新时间: {info['last_update']}\n"
            f"- 特征参数: {json.dumps(info['feature_params'], indent=2, ensure_ascii=False)}"
        )
