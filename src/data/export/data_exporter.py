"""
数据导出器
"""
from typing import Dict, List, Any, Optional, Callable
import logging
from datetime import datetime
from pathlib import Path
import json
import pandas as pd
import zipfile
import os

from src.data.interfaces import IDataModel
from src.infrastructure.utils.exceptions import DataLoaderError

logger = logging.getLogger(__name__)


class DataExporter:
    """
    数据导出器，支持多种格式导出
    """
    def __init__(self, export_dir: str):
        """
        初始化数据导出器

        Args:
            export_dir: 导出目录
        """
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)

        # 导出历史记录文件
        self.history_file = self.export_dir / 'export_history.json'
        self.history = self._load_history()

        # 支持的导出格式
        self.supported_formats = {
            'csv': self._export_csv,
            'excel': self._export_excel,
            'json': self._export_json,
            'parquet': self._export_parquet,
            'pickle': self._export_pickle,
            'hdf': self._export_hdf
        }

        logger.info(f"DataExporter initialized with directory: {export_dir}")

    def _load_history(self) -> List[Dict[str, Any]]:
        """加载导出历史记录"""
        if not self.history_file.exists():
            return []

        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load export history: {e}")
            return []

    def _save_history(self) -> None:
        """保存导出历史记录"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save export history: {e}")

    def _record_export(self, data_model: IDataModel, format: str, filepath: Path) -> None:
        """记录导出历史"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'format': format,
            'filepath': str(filepath),
            'metadata': data_model.get_metadata()
        }
        self.history.append(record)
        self._save_history()

    def export(
        self,
        data_model: IDataModel,
        format: str,
        filename: Optional[str] = None,
        include_metadata: bool = True,
        **kwargs
    ) -> str:
        """
        导出数据

        Args:
            data_model: 数据模型
            format: 导出格式
            filename: 文件名，如果为None则自动生成
            include_metadata: 是否包含元数据
            **kwargs: 其他导出选项

        Returns:
            str: 导出文件路径

        Raises:
            DataLoaderError: 如果导出失败
        """
        # 检查格式是否支持
        if format.lower() not in self.supported_formats:
            supported = ', '.join(self.supported_formats.keys())
            raise DataLoaderError(f"Unsupported export format: {format}. Supported formats: {supported}")

        # 生成文件名
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            metadata = data_model.get_metadata()
            source = metadata.get('source', 'unknown')
            symbol = metadata.get('symbol', '')
            filename = f"{source}_{symbol}_{timestamp}.{format.lower()}"

        # 确保文件名有正确的扩展名
        if not filename.endswith(f".{format.lower()}"):
            filename = f"{filename}.{format.lower()}"

        # 构建完整路径
        filepath = self.export_dir / filename

        try:
            # 调用相应的导出函数
            export_func = self.supported_formats[format.lower()]
            export_func(data_model, filepath, include_metadata, **kwargs)

            # 记录导出历史
            self._record_export(data_model, format, filepath)

            logger.info(f"Data exported to {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            raise DataLoaderError(f"Failed to export data: {e}")

    def get_supported_formats(self) -> List[str]:
        """获取支持的导出格式列表"""
        return list(self.supported_formats.keys())

    def export_to_buffer(
        self,
        data_model: IDataModel,
        format: str,
        include_metadata: bool = True,
        **kwargs
    ) -> bytes:
        """
        导出数据到内存缓冲区

        Args:
            data_model: 数据模型
            format: 导出格式
            include_metadata: 是否包含元数据
            **kwargs: 其他导出选项

        Returns:
            bytes: 导出的数据字节

        Raises:
            DataLoaderError: 如果导出失败
        """
        import io
        
        # 检查格式是否支持
        if format.lower() not in self.supported_formats:
            supported = ', '.join(self.supported_formats.keys())
            raise DataLoaderError(f"Unsupported export format: {format}. Supported formats: {supported}")

        try:
            # 创建内存缓冲区
            buffer = io.BytesIO()
            
            # 获取数据
            data = data_model.get_data()
            
            if format.lower() == 'csv':
                data.to_csv(buffer, index=True)
            elif format.lower() == 'json':
                if include_metadata:
                    export_data = {
                        'data': data.to_dict('records'),
                        'metadata': data_model.get_metadata()
                    }
                    buffer.write(json.dumps(export_data, indent=2).encode('utf-8'))
                else:
                    data.to_json(buffer, orient='records')
            elif format.lower() == 'excel':
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    data.to_excel(writer, sheet_name='data', index=True)
                    if include_metadata:
                        metadata_df = pd.DataFrame([data_model.get_metadata()])
                        metadata_df.to_excel(writer, sheet_name='metadata', index=False)
            elif format.lower() == 'pickle':
                import pickle
                if include_metadata:
                    export_data = {
                        'data': data,
                        'metadata': data_model.get_metadata()
                    }
                    pickle.dump(export_data, buffer)
                else:
                    pickle.dump(data, buffer)
            else:
                raise DataLoaderError(f"Buffer export not supported for format: {format}")
            
            buffer.seek(0)
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Failed to export to buffer: {e}")
            raise DataLoaderError(f"Failed to export to buffer: {e}")

    def export_multiple(
        self,
        data_models: List[IDataModel],
        format: str,
        zip_filename: Optional[str] = None,
        include_metadata: bool = True,
        **kwargs
    ) -> str:
        """
        导出多个数据模型到ZIP文件

        Args:
            data_models: 数据模型列表
            format: 导出格式
            zip_filename: ZIP文件名，如果为None则自动生成
            include_metadata: 是否包含元数据
            **kwargs: 其他导出选项

        Returns:
            str: ZIP文件路径

        Raises:
            DataLoaderError: 如果导出失败
        """
        # 检查格式是否支持
        if format.lower() not in self.supported_formats:
            supported = ', '.join(self.supported_formats.keys())
            raise DataLoaderError(f"Unsupported export format: {format}. Supported formats: {supported}")

        # 生成ZIP文件名
        if zip_filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            zip_filename = f"export_multiple_{timestamp}.zip"
        elif not zip_filename.endswith('.zip'):
            zip_filename = f"{zip_filename}.zip"

        # 构建完整路径
        zip_path = self.export_dir / zip_filename

        try:
            # 创建临时目录
            temp_dir = self.export_dir / "temp_export"
            temp_dir.mkdir(exist_ok=True)

            # 导出每个数据模型
            exported_files = []
            for i, data_model in enumerate(data_models):
                # 生成文件名
                metadata = data_model.get_metadata()
                source = metadata.get('source', 'unknown')
                symbol = metadata.get('symbol', '')
                filename = f"{source}_{symbol}_{i}.{format.lower()}"

                # 导出到临时目录
                temp_path = temp_dir / filename
                export_func = self.supported_formats[format.lower()]
                export_func(data_model, temp_path, include_metadata, **kwargs)
                exported_files.append(temp_path)

            # 创建ZIP文件
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for file in exported_files:
                    zipf.write(file, arcname=file.name)
                    
                    # 如果包含元数据，添加元数据文件
                    if include_metadata and format.lower() not in ['excel', 'json', 'pickle', 'hdf']:
                        metadata_file = file.with_suffix('.metadata.json')
                        if metadata_file.exists():
                            zipf.write(metadata_file, arcname=metadata_file.name)
            
            # 清理临时文件
            for file in exported_files:
                if file.exists():
                    file.unlink()
                metadata_file = file.with_suffix('.metadata.json')
                if metadata_file.exists():
                    metadata_file.unlink()
            
            if temp_dir.exists():
                temp_dir.rmdir()
            
            logger.info(f"Multiple data models exported to {zip_path}")
            return str(zip_path)
        except Exception as e:
            logger.error(f"Failed to export multiple data models: {e}")
            raise DataLoaderError(f"Failed to export multiple data models: {e}")
    
    def _export_csv(
        self,
        data_model: IDataModel,
        filepath: Path,
        include_metadata: bool = True,
        **kwargs
    ) -> None:
        """导出为CSV格式"""
        data_model.data.to_csv(filepath, **kwargs)
        if include_metadata:
            metadata_file = filepath.with_suffix('.metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(data_model.get_metadata(), f, indent=2)
    
    def _export_excel(
        self,
        data_model: IDataModel,
        filepath: Path,
        include_metadata: bool = True,
        **kwargs
    ) -> None:
        """导出为Excel格式"""
        with pd.ExcelWriter(filepath) as writer:
            data_model.data.to_excel(writer, sheet_name='Data', **kwargs)
            if include_metadata:
                pd.DataFrame([data_model.get_metadata()]).to_excel(
                    writer,
                    sheet_name='Metadata',
                    index=False
                )

    def _export_json(
        self,
        data_model: IDataModel,
        filepath: Path,
        include_metadata: bool = True,
        **kwargs
    ) -> None:
        """导出为JSON格式"""
        data = {
            'data': data_model.data.to_dict(orient='records'),
            'metadata': data_model.get_metadata() if include_metadata else None
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def _export_parquet(
        self,
        data_model: IDataModel,
        filepath: Path,
        include_metadata: bool = True,
        **kwargs
    ) -> None:
        """导出为Parquet格式"""
        data_model.data.to_parquet(filepath, **kwargs)
        if include_metadata:
            metadata_file = filepath.with_suffix('.metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(data_model.get_metadata(), f, indent=2)

    def _export_pickle(
        self,
        data_model: IDataModel,
        filepath: Path,
        include_metadata: bool = True,
        **kwargs
    ) -> None:
        """导出为Pickle格式"""
        data = {
            'data': data_model.data,
            'metadata': data_model.get_metadata() if include_metadata else None
        }
        pd.to_pickle(data, filepath)

    def _export_hdf(
        self,
        data_model: IDataModel,
        filepath: Path,
        include_metadata: bool = True,
        **kwargs
    ) -> None:
        """导出为HDF5格式"""
        with pd.HDFStore(filepath) as store:
            store.put('data', data_model.data, **kwargs)
            if include_metadata:
                store.put('metadata', pd.Series(data_model.get_metadata()))

    def get_export_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        获取导出历史记录

        Args:
            limit: 返回的记录数量限制

        Returns:
            List[Dict[str, Any]]: 导出历史记录
        """
        if limit is None:
            return self.history
        return self.history[-limit:]

    def clear_history(self) -> None:
        """清除导出历史记录"""
        self.history = []
        self._save_history()
