"""
数据准备阶段模块

负责数据收集、加载、质量检查和预处理
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass

from .base import PipelineStage
from ..exceptions import DataQualityException, StageExecutionException
from ..config import StageConfig


@dataclass
class DataQualityReport:
    """数据质量报告"""
    total_rows: int
    total_columns: int
    missing_values: Dict[str, int]
    missing_percentage: Dict[str, float]
    duplicate_rows: int
    outlier_counts: Dict[str, int]
    data_types: Dict[str, str]
    timestamp_range: Optional[Tuple[datetime, datetime]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "missing_values": self.missing_values,
            "missing_percentage": self.missing_percentage,
            "duplicate_rows": self.duplicate_rows,
            "outlier_counts": self.outlier_counts,
            "data_types": self.data_types,
            "timestamp_range": (
                (self.timestamp_range[0].isoformat(), self.timestamp_range[1].isoformat())
                if self.timestamp_range else None
            )
        }


class DataPreparationStage(PipelineStage):
    """
    数据准备阶段
    
    功能：
    - 从多个数据源收集数据
    - 数据加载和合并
    - 数据质量检查
    - 数据清洗和预处理
    - 数据版本管理
    """
    
    def __init__(self, config: Optional[StageConfig] = None):
        super().__init__("data_preparation", config)
        self._quality_report: Optional[DataQualityReport] = None
        self._raw_data: Optional[pd.DataFrame] = None
        self._processed_data: Optional[pd.DataFrame] = None
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行数据准备
        
        Args:
            context: 包含data_sources, date_range等配置
            
        Returns:
            包含processed_data, quality_report的输出
        """
        self.logger.info("开始数据准备阶段")
        
        # 获取配置
        data_sources = self.config.config.get("data_sources", ["market_data"])
        date_range = self.config.config.get("date_range", "last_90_days")
        quality_checks = self.config.config.get("quality_checks", True)
        
        # 1. 数据收集
        self.logger.info(f"从数据源收集数据: {data_sources}")
        raw_data = self._collect_data(data_sources, date_range, context)
        
        if raw_data is None or raw_data.empty:
            raise StageExecutionException(
                message="数据收集失败：没有获取到数据",
                stage_name=self.name
            )
        
        self._raw_data = raw_data
        self.logger.info(f"收集到 {len(raw_data)} 行数据")
        
        # 2. 数据质量检查
        if quality_checks:
            self.logger.info("执行数据质量检查")
            self._quality_report = self._check_quality(raw_data)
            
            # 检查质量阈值
            if not self._validate_quality(self._quality_report):
                raise DataQualityException(
                    message="数据质量检查未通过",
                    quality_issues=self._quality_report.to_dict()
                )
            
            self.logger.info("数据质量检查通过")
        
        # 3. 数据清洗和预处理
        self.logger.info("执行数据清洗")
        processed_data = self._clean_data(raw_data)
        self._processed_data = processed_data
        
        # 4. 数据验证
        self._validate_processed_data(processed_data)
        
        self.logger.info(f"数据准备完成，处理后数据: {len(processed_data)} 行")
        
        return {
            "raw_data_shape": raw_data.shape,
            "processed_data_shape": processed_data.shape,
            "processed_data": processed_data,
            "quality_report": self._quality_report.to_dict() if self._quality_report else None,
            "data_sources": data_sources,
            "date_range": date_range
        }
    
    def _collect_data(
        self,
        data_sources: List[str],
        date_range: str,
        context: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        从数据源收集数据
        
        Args:
            data_sources: 数据源列表
            date_range: 日期范围
            context: 上下文
            
        Returns:
            合并后的数据框
        """
        all_data = []
        
        # 解析日期范围
        end_date = datetime.now()
        if date_range == "last_90_days":
            start_date = end_date - timedelta(days=90)
        elif date_range == "last_30_days":
            start_date = end_date - timedelta(days=30)
        elif date_range == "last_1_year":
            start_date = end_date - timedelta(days=365)
        else:
            # 尝试从context获取
            start_date = context.get("start_date", end_date - timedelta(days=90))
            end_date = context.get("end_date", end_date)
        
        for source in data_sources:
            try:
                if source == "market_data":
                    data = self._load_market_data(start_date, end_date, context)
                elif source == "fundamental_data":
                    data = self._load_fundamental_data(start_date, end_date, context)
                elif source == "technical_data":
                    data = self._load_technical_data(start_date, end_date, context)
                else:
                    self.logger.warning(f"未知数据源: {source}")
                    continue
                
                if data is not None and not data.empty:
                    all_data.append(data)
                    self.logger.debug(f"从 {source} 加载 {len(data)} 行数据")
                    
            except Exception as e:
                self.logger.error(f"从 {source} 加载数据失败: {e}")
                raise StageExecutionException(
                    message=f"数据源 {source} 加载失败: {e}",
                    stage_name=self.name,
                    cause=e
                )
        
        if not all_data:
            return pd.DataFrame()
        
        # 合并数据
        if len(all_data) == 1:
            return all_data[0]
        
        # 多数据源合并（按时间戳）
        merged_data = all_data[0]
        for data in all_data[1:]:
            merged_data = pd.merge(
                merged_data, data,
                on="timestamp",
                how="outer"
            )
        
        return merged_data.sort_values("timestamp").reset_index(drop=True)
    
    def _load_market_data(
        self,
        start_date: datetime,
        end_date: datetime,
        context: Dict[str, Any]
    ) -> pd.DataFrame:
        """加载市场数据"""
        # 从context获取数据管理器
        data_manager = context.get("data_manager")
        
        if data_manager:
            # 使用数据管理器加载
            symbols = context.get("symbols", [])
            return data_manager.load_data(symbols, start_date, end_date)
        
        # 模拟数据（实际应从数据库加载）
        self.logger.warning("使用模拟市场数据")
        dates = pd.date_range(start=start_date, end=end_date, freq="1min")
        np.random.seed(42)
        
        data = pd.DataFrame({
            "timestamp": dates,
            "open": np.random.randn(len(dates)).cumsum() + 100,
            "high": np.random.randn(len(dates)).cumsum() + 101,
            "low": np.random.randn(len(dates)).cumsum() + 99,
            "close": np.random.randn(len(dates)).cumsum() + 100,
            "volume": np.random.randint(1000, 100000, len(dates))
        })
        
        return data
    
    def _load_fundamental_data(
        self,
        start_date: datetime,
        end_date: datetime,
        context: Dict[str, Any]
    ) -> pd.DataFrame:
        """加载基本面数据"""
        # 实际应从基本面数据库加载
        self.logger.warning("基本面数据加载暂未实现")
        return pd.DataFrame()
    
    def _load_technical_data(
        self,
        start_date: datetime,
        end_date: datetime,
        context: Dict[str, Any]
    ) -> pd.DataFrame:
        """加载技术指标数据"""
        # 实际应从技术指标数据库加载
        self.logger.warning("技术指标数据加载暂未实现")
        return pd.DataFrame()
    
    def _check_quality(self, data: pd.DataFrame) -> DataQualityReport:
        """
        检查数据质量
        
        Args:
            data: 原始数据
            
        Returns:
            质量报告
        """
        # 缺失值统计
        missing_values = data.isnull().sum().to_dict()
        missing_percentage = {
            col: (missing_values[col] / len(data) * 100) if len(data) > 0 else 0
            for col in data.columns
        }
        
        # 重复行
        duplicate_rows = data.duplicated().sum()
        
        # 异常值检测（使用IQR方法）
        outlier_counts = {}
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = data[
                (data[col] < Q1 - 1.5 * IQR) | 
                (data[col] > Q3 + 1.5 * IQR)
            ]
            outlier_counts[col] = len(outliers)
        
        # 数据类型
        data_types = {col: str(dtype) for col, dtype in data.dtypes.items()}
        
        # 时间范围
        timestamp_range = None
        if "timestamp" in data.columns:
            timestamp_range = (data["timestamp"].min(), data["timestamp"].max())
        
        return DataQualityReport(
            total_rows=len(data),
            total_columns=len(data.columns),
            missing_values=missing_values,
            missing_percentage=missing_percentage,
            duplicate_rows=duplicate_rows,
            outlier_counts=outlier_counts,
            data_types=data_types,
            timestamp_range=timestamp_range
        )
    
    def _validate_quality(self, report: DataQualityReport) -> bool:
        """
        验证数据质量是否满足要求
        
        Args:
            report: 质量报告
            
        Returns:
            是否通过验证
        """
        # 检查缺失值比例
        max_missing_threshold = self.config.config.get("max_missing_threshold", 10.0)
        for col, pct in report.missing_percentage.items():
            if pct > max_missing_threshold:
                self.logger.warning(f"列 {col} 缺失值比例 {pct:.2f}% 超过阈值 {max_missing_threshold}%")
                return False
        
        # 检查重复行比例
        if report.total_rows > 0:
            dup_pct = report.duplicate_rows / report.total_rows * 100
            if dup_pct > 5.0:
                self.logger.warning(f"重复行比例 {dup_pct:.2f}% 过高")
                return False
        
        # 检查是否有数据
        if report.total_rows < 100:
            self.logger.warning(f"数据量 {report.total_rows} 过少")
            return False
        
        return True
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        清洗数据
        
        Args:
            data: 原始数据
            
        Returns:
            清洗后的数据
        """
        df = data.copy()
        
        # 1. 处理重复行
        initial_rows = len(df)
        df = df.drop_duplicates()
        dropped = initial_rows - len(df)
        if dropped > 0:
            self.logger.info(f"删除 {dropped} 行重复数据")
        
        # 2. 处理缺失值
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # 数值列：使用前向填充+后向填充
        for col in numeric_cols:
            missing_before = df[col].isnull().sum()
            df[col] = df[col].fillna(method="ffill").fillna(method="bfill")
            missing_after = df[col].isnull().sum()
            if missing_before > 0:
                self.logger.info(f"列 {col} 填充 {missing_before - missing_after} 个缺失值")
        
        # 3. 处理异常值（可选）
        if self.config.config.get("handle_outliers", False):
            df = self._handle_outliers(df)
        
        # 4. 数据类型转换
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # 5. 排序
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)
        
        return df
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理异常值（使用winsorization）"""
        df = data.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            
            outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            if outliers > 0:
                df[col] = df[col].clip(lower, upper)
                self.logger.info(f"列 {col} 处理 {outliers} 个异常值")
        
        return df
    
    def _validate_processed_data(self, data: pd.DataFrame) -> None:
        """验证处理后的数据"""
        if data.empty:
            raise StageExecutionException(
                message="数据处理后为空",
                stage_name=self.name
            )
        
        if "timestamp" not in data.columns:
            raise StageExecutionException(
                message="处理后数据缺少timestamp列",
                stage_name=self.name
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取阶段指标"""
        metrics = {
            "raw_rows": len(self._raw_data) if self._raw_data is not None else 0,
            "processed_rows": len(self._processed_data) if self._processed_data is not None else 0,
            "columns": len(self._processed_data.columns) if self._processed_data is not None else 0
        }
        
        if self._quality_report:
            metrics["quality_score"] = self._calculate_quality_score(self._quality_report)
        
        return metrics
    
    def _calculate_quality_score(self, report: DataQualityReport) -> float:
        """计算数据质量分数"""
        # 基于缺失值、重复行等因素计算质量分数
        if report.total_rows == 0:
            return 0.0
        
        # 缺失值扣分
        missing_penalty = sum(report.missing_percentage.values()) / len(report.missing_percentage) * 0.5
        
        # 重复行扣分
        dup_penalty = (report.duplicate_rows / report.total_rows) * 100 * 0.3
        
        # 异常值扣分
        outlier_penalty = sum(report.outlier_counts.values()) / report.total_rows * 100 * 0.2 if report.total_rows > 0 else 0
        
        score = max(0, 100 - missing_penalty - dup_penalty - outlier_penalty)
        return round(score, 2)
    
    def rollback(self, context: Dict[str, Any]) -> bool:
        """回滚数据准备阶段"""
        self.logger.info("回滚数据准备阶段")
        self._raw_data = None
        self._processed_data = None
        self._quality_report = None
        return True
