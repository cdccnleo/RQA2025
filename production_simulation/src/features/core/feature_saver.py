import json
import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd


logger = logging.getLogger(__name__)


class FeatureSaver:
    """特征结果持久化工具，负责保存/加载以及元数据落盘。"""

    def __init__(
        self,
        base_path: Union[str, Path] = "./feature_outputs",
        metadata_name: str = "metadata.json",
    ):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.base_path / metadata_name
        self.last_save_info: Optional[dict] = None

    def save_features(
        self,
        features: pd.DataFrame,
        output_path: Union[str, Path],
        format: str = "parquet",
        metadata: Optional[dict] = None,
    ) -> bool:
        """保存特征数据并记录元数据。"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if format == "parquet":
                features.to_parquet(output_path, index=False)
            elif format == "csv":
                features.to_csv(output_path, index=False)
            elif format == "pickle":
                features.to_pickle(output_path)
            else:
                raise ValueError(f"Unsupported format: {format}")

            save_info = {
                "path": str(output_path),
                "format": format,
                "shape": list(features.shape),
                "columns": features.columns.tolist(),
                "metadata": metadata or {},
            }
            self._write_metadata(save_info)
            self.last_save_info = save_info
            return True
        except Exception as exc:
            logger.error(f"保存特征失败: {exc}")
            return False

    def load_features(self, input_path: Union[str, Path], format: str = "parquet") -> pd.DataFrame:
        """加载先前保存的特征数据。"""
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(str(input_path))

        if format == "parquet":
            return pd.read_parquet(input_path)
        if format == "csv":
            return pd.read_csv(input_path)
        if format == "pickle":
            return pd.read_pickle(input_path)

        raise ValueError(f"Unsupported format: {format}")

    def get_last_metadata(self) -> Optional[dict]:
        """返回最近一次保存的元数据内容。"""
        if self.last_save_info:
            return self.last_save_info
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as fp:
                return json.load(fp)
        return None

    def _write_metadata(self, data: dict) -> None:
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, ensure_ascii=False, indent=2)
