# src/feature/feature_manager.py
import time
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import os
import torch
from sklearn.exceptions import NotFittedError
from sklearn.ensemble import RandomForestRegressor
from transformers import DistilBertTokenizer, AutoModelForSequenceClassification
from src.infrastructure.utils.exceptions import DataLoaderError
from .processors.feature_selector import FeatureSelector
from .processors.feature_standardizer import FeatureStandardizer
from src.models.model_manager import ModelManager
from .feature_metadata import FeatureMetadata
from .processors.feature_engineer import FeatureEngineer
from .technical.technical_processor import TechnicalProcessor
from .sentiment.sentiment_analyzer import SentimentAnalyzer
from src.infrastructure.utils.logger import get_logger
from typing import Dict, List
import joblib

logger = get_logger(__name__)  # 自动继承全局配置


class FeatureManager:
    """特征工程全流程处理

    属性:
        model_path (str): 模型存储路径
        stock_code (str): 股票代码标识
        scaler (StandardScaler): 数据标准化器
        selected_features (List[str]): 选择的特征列
    """

    def __init__(
            self,
            model_path: str,
            stock_code: str,
            model_manager: ModelManager,
            feature_selector=None,
            feature_selector_params: dict = None,
            preserve_features: List[str] = None  # 显式传递需要保留的特征
    ):
        """初始化特征管道"""
        self.model_path = Path(model_path)
        self.stock_code = stock_code

        self.model_path.mkdir(parents=True, exist_ok=True)
        self.target = None
        self.selector = None
        # 初始化处理器
        self.technical_processor = TechnicalProcessor(
            register_func=self._register_feature_config
        )
        self.sentiment_analyzer = SentimentAnalyzer(use_segmentation=True, use_gpu=False)
        self.model_manager = model_manager
        self.feature_selector = feature_selector
        self.model_name = f"{stock_code}_model"
        self.model_version = "v1.0"
        self.selected_features = []
        self.preserve_features = preserve_features or []  # 显式存储保留特征
        self.feature_engineer = FeatureEngineer(
            technical_processor=self.technical_processor,
            stock_code=stock_code
        )
        feature_selector_dir = self.model_path / f"{stock_code}_feature_selector"
        feature_selector_dir.mkdir(parents=True, exist_ok=True)

        default_selector_params = {
            "min_features_to_select": 20,
            "cv": 5,
            "selector_type": "rfecv"  # 保留在字典中以便动态调整
        }
        feature_selector_params = feature_selector_params or default_selector_params.copy()
        # 提取并移除 selector_type 参数
        selector_type = feature_selector_params.pop("selector_type", "rfecv")
        self.feature_selector = FeatureSelector(
            selector_type=selector_type,
            model_path=feature_selector_dir,
            preserve_features=self.preserve_features,
            **feature_selector_params
        )
        self.feature_standardizer = FeatureStandardizer(model_path=self.model_path)
        self.feature_standardizer.scaler_path = self.model_path / f"{stock_code}_feature_scaler.pkl"
        self.logger = logger

        self.feature_params = {}  # 或根据需求加载配置

        self.metadata = FeatureMetadata(
            feature_params=self.feature_params,
            data_source_version="v1.0",
            feature_list=[]
        )

        if not self.load_pipeline("v1.0"):
            self.logger.info("未找到历史特征管道，初始化新元数据")

    def update_features(self, new_data: pd.DataFrame):
        """增量特征更新逻辑，新增特征重要性反馈

        支持:
        - 技术指标的滚动计算
        - 情感特征的动态追加
        - 合并后触发特征选择更新
        - 特征重要性反馈循环
        """
        if not self.metadata:
            raise RuntimeError("特征元数据未初始化，无法执行增量更新")

        # 检查新数据是否包含元数据中的特征列
        required_features = self.metadata.feature_list
        missing_features = [feature for feature in required_features if feature not in new_data.columns]
        if missing_features:
            self.logger.warning(f"新数据缺少特征列: {missing_features}")

        # 验证新数据的特征维度与元数据一致
        if len(new_data.columns) != len(required_features):
            self.logger.error("特征维度不匹配：新数据特征数 %d，元数据特征数 %d",
                              len(new_data.columns), len(required_features))
            self._retrain_model(new_data)  # 调用重新训练方法
            raise RuntimeError("特征不兼容，已触发重新训练")  # 修改异常信息

        # 检查模型兼容性
        if not self.model_manager.validate_model(
                model_name=self.model_name,
                version=self.model_version,
                checks={'feature_columns': self.metadata.feature_list}):
            self.logger.warning("特征不兼容，将重新训练模型")
            self._retrain_model(new_data)  # 确保此处调用了 _retrain_model 方法
            raise RuntimeError("特征不兼容，已触发重新训练")  # 修改异常信息
        else:
            # 新增目标列提取逻辑
            if 'target' in new_data.columns:
                self.target = new_data['target']
            elif self.metadata and 'target' in self.metadata.feature_list:
                self.target = new_data[self.metadata.feature_list['target']]
            else:
                self.target = pd.Series(0, index=new_data.index)  # 默认目标值

            # 1. 更新技术指标（滚动计算）
            indicators = self.metadata.feature_params.get('technical_indicators', [])
            if hasattr(self, 'technical_features'):
                new_technical = self.technical_processor.calc_indicators(new_data, indicators)
                self.technical_features = pd.concat([self.technical_features, new_technical])
            else:
                self.technical_features = self.technical_processor.calc_indicators(new_data, indicators)

            # 2. 更新情感特征（动态追加）
            if hasattr(self, 'sentiment_features'):
                new_sentiment = self.feature_engineer.generate_sentiment_features(
                    news_data=new_data
                )
                self.sentiment_features = pd.concat([self.sentiment_features, new_sentiment])
            else:
                self.sentiment_features = self.feature_engineer.generate_sentiment_features(
                    news_data=new_data
                )

            # 3. 合并特征并更新选择器
            merged = self.feature_engineer.merge_features(
                stock_data=new_data,
                technical_features=self.technical_features,
                sentiment_features=self.sentiment_features,
                metadata=self.metadata  # 传递metadata参数
            )
            # 使用FeatureSelector进行特征选择
            merged_selected = self.feature_selection(merged, self.target, is_training=False)

            # 4. 调用FeatureStandardizer更新标准器
            normalized = self.feature_standardizer.fit_transform(
                features=merged_selected,
                is_training=False,  # 增量更新时通常不在训练模式
                metadata=self.metadata
            )

            # 5. 特征重要性反馈循环
            self._feedback_feature_importance(merged_selected, self.target)

            return normalized

    def _retrain_model(self, new_data: pd.DataFrame):
        """重新训练模型逻辑"""
        # 确保目标列存在
        if 'target' not in new_data.columns:
            new_data['target'] = 0.0  # 添加默认目标列
            self.logger.info("生成默认目标列")

        # 更新元数据
        self.metadata.update_feature_columns(new_data.columns.tolist())

        # 调用 model_manager 的 retrain_model 方法
        self.model_manager.retrain_model(new_data)

        # 确保 target 属性被正确初始化
        self.target = new_data['target']

    def _feedback_feature_importance(self, features: pd.DataFrame, target: pd.Series) -> None:
        """特征重要性反馈机制，动态调整特征生成逻辑"""
        try:
            # 检查输入数据是否有效
            if features.empty or target.empty:
                self.logger.warning("特征或目标数据为空，跳过反馈循环")
                return

            # 计算特征重要性
            importance = self._calculate_feature_importance(features, target)
            if importance.empty:
                self.logger.warning("特征重要性计算结果为空")
                return

            # 确定重要特征
            important_features = importance[importance > importance.mean()].index.tolist()

            # 如果没有重要特征，则强制设置 min_features_to_select=3
            if not important_features:
                # 日志：记录触发参数更新
                self.logger.info("触发反馈逻辑，调用update_selector_params")  # 新增

                # 明确更新特征选择器的参数
                self.feature_selector.update_selector_params(
                    min_features_to_select=3
                )

                # 日志：记录fit调用
                self.logger.info("重新拟合特征选择器")  # 新增

                # 确保重新初始化选择器并拟合
                self.feature_selector.fit(features, target, is_training=False)
            else:
                # 动态更新特征选择器参数
                self.feature_selector.update_selector_params(
                    estimator=RandomForestRegressor(n_estimators=50),
                    min_features_to_select=max(3, len(important_features))
                )
                # 重新拟合特征选择器
                self.feature_selector.fit(features, target, is_training=False)
        except Exception as e:
            self.logger.error(f"特征重要性反馈失败: {str(e)}")
            raise

    def _calculate_feature_importance(self, features: pd.DataFrame, target: pd.Series) -> pd.Series:
        """计算特征重要性"""
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(features, target)
        return pd.Series(model.feature_importances_, index=features.columns).sort_values(ascending=False)

    def save_pipeline(self, data_source_version: str) -> None:
        """保存特征处理管道状态及元数据"""
        try:
            # 保存逻辑
            if self.feature_standardizer.scaler is None:
                raise ValueError("特征标准化器未初始化")

            pipeline_path = os.path.join(self.model_path, f"{self.stock_code}_feature_pipeline.pkl")
            joblib.dump({
                'scaler': self.feature_standardizer.scaler,
                'features': self.selected_features,
                'metadata': FeatureMetadata(
                    feature_params=self.feature_params,
                    data_source_version=data_source_version,
                    feature_list=self.selected_features,
                    scaler_path=self.model_path / f"{self.stock_code}_feature_scaler.pkl",
                    selector_path=self.model_path / f"{self.stock_code}_feature_selector.pkl"
                )
            }, pipeline_path)
        except Exception as e:
            self.logger.error(f"特征处理管道保存失败: {str(e)}")
            raise

    def load_pipeline(self, data_source_version: str) -> bool:
        """加载特征处理管道状态及验证元数据"""
        pipeline_path = self.model_path / f"{self.stock_code}_feature_pipeline.pkl"
        if not pipeline_path.exists():
            return False
        try:
            state = joblib.load(pipeline_path)
            self.scaler = state['scaler']
            self.selected_features = state['features']
            self.metadata = state['metadata']

            # 验证数据源版本兼容性
            if self.metadata.data_source_version != data_source_version:
                warnings.warn(
                    "数据源版本不匹配，可能影响特征一致性",
                    UserWarning
                )
                self.logger.warning("数据源版本不匹配，可能影响特征一致性")
            return True
        except Exception as e:
            self.logger.error(f"特征管道加载失败: {str(e)}")
            try:
                pipeline_path.unlink(missing_ok=True)
                self.logger.info(f"已删除损坏文件: {pipeline_path}")
            except Exception as delete_error:
                self.logger.error(f"删除损坏文件失败: {str(delete_error)}")
            raise DataLoaderError(f"无法加载特征管道: {str(e)}") from e

    def execute_pipeline(
            self,
            stock_data: pd.DataFrame,
            news_data: pd.DataFrame,
            technical_params: Dict = None,
            sentiment_params: Dict = None,
    ) -> pd.DataFrame:
        """执行特征工程全流程"""
        if stock_data.empty:
            raise ValueError("输入数据为空，请确保数据不为空")

        # 如果新闻数据为空，创建一个空的DataFrame，使用stock_data的索引
        if news_data.empty:
            news_data = pd.DataFrame(index=stock_data.index)
            self.logger.info("新闻数据为空，将使用空的DataFrame进行处理")

        # 确保 stock_data 包含必要的列
        required_stock_columns = ["close", "high", "low", "volume"]
        missing_stock_columns = [col for col in required_stock_columns if col not in stock_data.columns]
        if missing_stock_columns:
            self.logger.error(f"股票数据缺少必要列: {missing_stock_columns}")
            raise ValueError(f"股票数据缺少必要列: {missing_stock_columns}")

        # 确保索引为 DatetimeIndex
        if not isinstance(stock_data.index, pd.DatetimeIndex):
            if 'date' in stock_data.columns:
                stock_data = stock_data.set_index('date')
                self.logger.info("股票数据已设置 'date' 为索引")
            else:
                raise KeyError("股票数据必须包含 'date' 列或索引名为 'date'")

        if not isinstance(news_data.index, pd.DatetimeIndex):
            if 'date' in news_data.columns:
                news_data = news_data.set_index('date')
                self.logger.info("新闻数据已设置 'date' 为索引")
            else:
                raise KeyError("新闻数据必须包含 'date' 列或索引名为 'date'")

        # 检查索引对齐
        if not stock_data.index.equals(news_data.index):
            self.logger.error(
                "索引不匹配：股票数据索引 %s，新闻数据索引 %s",
                stock_data.index.tolist(),
                news_data.index.tolist()
            )
            raise ValueError("特征索引不匹配")

        # 确保 'target' 列存在
        if 'target' not in stock_data.columns:
            stock_data['target'] = 0.0  # 添加默认目标列
            self.logger.info("股票数据中未找到 'target' 列，已添加默认目标列")

        # 1. 特征生成
        technical_features = self.feature_engineer.generate_technical_features(
            stock_data,
            technical_params.get('indicators') if technical_params else None
        )
        sentiment_features = self.feature_engineer.generate_sentiment_features(
            news_data,
            **sentiment_params if sentiment_params else {}
        )
        merged_features = self.feature_engineer.merge_features(
            stock_data.drop(columns=['target']),  # 移除目标列
            technical_features,
            sentiment_features,
            metadata=self.metadata
        )

        # 2. 特征标准化
        standardized_features = self.feature_standardizer.fit_transform(
            features=merged_features,
            is_training=True,
            metadata=self.metadata
        )

        # 3. 特征选择（新增调用feature_selection方法）
        selected_features = self.feature_selection(
            standardized_features,
            stock_data['target'],
            is_training=True
        )

        # 4. 执行特征选择器的transform并转换为DataFrame
        final_features = pd.DataFrame(
            selected_features,
            columns=self.feature_selector.selected_features,
            index=standardized_features.index
        )

        # 5. 保存元数据
        self.metadata.save(self.model_path / "feature_metadata.pkl")

        return final_features  # 确保返回的是DataFrame

    def _bert_sentiment(
            self,
            text: str,
            tokenizer: DistilBertTokenizer,
            model: AutoModelForSequenceClassification
    ) -> float:
        """BERT情感分析内部实现"""
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        return torch.nn.functional.softmax(outputs.logits, dim=-1)[0][1].item()

    def feature_selection(
            self,
            features: pd.DataFrame,
            target: pd.Series,
            is_training: bool = True
    ) -> pd.DataFrame:
        """执行特征选择（通过FeatureSelector类）"""
        # 预测模式下若目标变量为空，尝试使用默认值
        if not is_training and target is None:
            target = pd.Series(0, index=features.index)  # 生成默认目标值

        try:
            if is_training:
                self.feature_selector.fit(features, target, is_training=is_training)
            return self.feature_selector.transform(features)
        except (NotFittedError, ValueError, RuntimeError) as e:
            msg = f"特征选择失败：{str(e)}"
            self.logger.warning(msg)
            warnings.warn(msg, UserWarning)
            return features.copy()
        except Exception as e:
            self.logger.error(f"特征选择严重错误: {str(e)}")
            raise

    def merge_features(
            self,
            stock_data: pd.DataFrame,
            technical_features: pd.DataFrame,
            sentiment_features: pd.DataFrame,
            metadata: FeatureMetadata = None
    ) -> pd.DataFrame:
        """合并特征数据"""
        merged_features = self._merge_features(stock_data, technical_features, sentiment_features)

        # 更新元数据
        if metadata:
            metadata.update_feature_columns(list(merged_features.columns))
            metadata.last_updated = time.time()  # 统一更新时间戳

        # 合并特征后添加调试日志
        self.logger.info(f"合并后的特征列名: {merged_features.columns.tolist()}")
        self.logger.info(f"合并后的特征样例:\n{merged_features.head()}")

        return merged_features

    def _merge_features(self, stock_data, technical_features, sentiment_features):
        """按日期合并特征数据（直接通过索引合并)"""
        try:
            # 确保所有数据框的索引为 date
            def ensure_date_index(df, df_name):
                if df.index.name != 'date':
                    if 'date' in df.columns:
                        df = df.set_index('date')
                    else:
                        raise KeyError(f"{df_name} 必须包含 'date' 列或索引名为 'date'")
                return df

            # 确保索引正确
            stock_data = ensure_date_index(stock_data, "stock_data")
            technical_features = ensure_date_index(technical_features, "technical_features")
            sentiment_features = ensure_date_index(sentiment_features, "sentiment_features")

            # 检查索引是否唯一
            if not (
                    stock_data.index.is_unique and technical_features.index.is_unique and sentiment_features.index.is_unique):
                raise ValueError("特征索引存在重复日期")

            # 检查索引是否对齐
            if not (stock_data.index.equals(technical_features.index) and stock_data.index.equals(
                    sentiment_features.index)):
                raise ValueError("特征索引不匹配")

            # 添加前缀前确保情感特征非空
            if not sentiment_features.empty:
                sentiment_features = sentiment_features.add_prefix("senti_")
            else:
                # 生成默认情感特征列
                default_sentiment = pd.DataFrame(
                    0.5,  # 中性值填充
                    index=stock_data.index,
                    columns=["senti_snownlp_mean"]
                )
                sentiment_features = default_sentiment

            technical_features = technical_features.add_prefix("tech_")

            # 合并特征
            merged = stock_data.join([technical_features, sentiment_features], how='left')

            # 强制转换所有列为数值类型（处理可能的 object 类型残留）
            merged = merged.apply(pd.to_numeric, errors='coerce')
            merged = merged.dropna(how='all', axis=1)  # 移除全空列

            non_numeric = merged.select_dtypes(exclude=[np.number]).columns
            if not non_numeric.empty:
                self.logger.error(f"非数值列存在: {non_numeric.tolist()}")
                raise ValueError("特征矩阵包含非数值列")
            self._validate_alignment(merged)

            # 验证列名无重复
            assert len(merged.columns) == len(set(merged.columns)), "特征列名冲突"
            return merged
        except Exception as e:
            self.logger.error(f"特征合并失败: {str(e)}")
            raise

    def _generate_default_target(self, features: pd.DataFrame) -> pd.Series:
        """生成默认目标列（全零）"""
        default_target = pd.Series(0.0, index=features.index, name='target')
        return default_target

    def _register_feature_config(self, config) -> None:
        """特征配置注册方法"""
        if not hasattr(self, 'feature_engineer'):
            return
            
        # 延迟导入避免循环依赖
        from .feature_config import FeatureConfig
        if isinstance(config, FeatureConfig):
            self.feature_engineer.register_feature(config)
