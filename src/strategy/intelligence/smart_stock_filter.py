#!/usr/bin/env python3
"""
AI智能股票筛选服务 - 策略层实现

基于机器学习的股票池动态调整算法：
1. 股票重要性预测模型
2. 流动性评估模型
3. 市场波动敏感度模型
4. 动态池调整算法

架构位置：策略层 (Strategy Layer) - 符合业务流程驱动架构
职责：提供智能化的股票筛选决策支持，整合特征工程、模型训练、策略回测等服务
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import os

try:
    import sklearn
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, accuracy_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    sklearn = None

# 策略层统一的日志系统（符合架构设计：策略层统一日志接口）
try:
    from src.infrastructure.logging.core.unified_logger import get_unified_logger
    logger = get_unified_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# 策略层服务集成（符合架构设计：策略层通过适配器访问其他层服务）
_feature_engineering_service = None
_ml_training_service = None
_backtest_service = None

def _get_feature_engineering_service():
    """获取特征工程服务（通过策略层适配器访问特征层）"""
    global _feature_engineering_service
    if _feature_engineering_service is None:
        try:
            # 策略层通过网关层适配器访问特征工程服务
            from src.gateway.web.feature_engineering_service import get_feature_engine
            _feature_engineering_service = get_feature_engine()
            logger.info("特征工程服务已集成到AI智能筛选器")
        except Exception as e:
            logger.warning(f"无法获取特征工程服务: {e}")
            _feature_engineering_service = None
    return _feature_engineering_service

def _get_ml_training_service():
    """获取ML训练服务（通过策略层适配器访问ML层）"""
    global _ml_training_service
    if _ml_training_service is None:
        try:
            # 策略层通过网关层适配器访问ML训练服务
            from src.ml.core.ml_service import MLService
            _ml_training_service = MLService()
            logger.info("ML训练服务已集成到AI智能筛选器")
        except Exception as e:
            logger.warning(f"无法获取ML训练服务: {e}")
            _ml_training_service = None
    return _ml_training_service

def _get_backtest_service():
    """获取策略回测服务（策略层内部服务）"""
    global _backtest_service
    if _backtest_service is None:
        try:
            # 策略层直接访问回测服务
            from src.gateway.web.backtest_service import get_backtest_service
            _backtest_service = get_backtest_service()
            logger.info("策略回测服务已集成到AI智能筛选器")
        except Exception as e:
            logger.warning(f"无法获取策略回测服务: {e}")
            _backtest_service = None
    return _backtest_service


@dataclass
class StockFeatures:
    """股票特征数据"""
    code: str
    name: str
    price: float
    volume: float
    turnover: float
    volatility: float
    market_cap: float
    pe_ratio: float
    pb_ratio: float
    turnover_rate: float
    amplitude: float
    avg_volume_5d: float
    avg_turnover_5d: float
    momentum_5d: float
    momentum_20d: float


@dataclass
class MarketState:
    """市场状态"""
    volatility_index: float  # 市场波动率
    trading_volume: float    # 总成交量
    market_sentiment: float  # 市场情绪(-1到1)
    sector_rotation: Dict[str, float]  # 板块轮动
    timestamp: datetime


class SmartStockFilter:
    """
    AI智能股票筛选器 - 策略层核心组件

    基于多维度特征的机器学习模型，实现：
    1. 股票重要性评分预测
    2. 流动性评估
    3. 市场适应性调整
    4. 策略回测验证

    架构定位：策略层智能决策组件
    业务价值：为量化策略提供智能化的股票池选择支持
    """

    def __init__(self, model_path: str = "models/smart_filter"):
        """
        初始化AI智能筛选器

        Args:
            model_path: 模型存储路径
        """
        self.model_path = model_path
        self.models = {}
        self.scalers = {}
        self.feature_columns = [
            'price', 'volume', 'turnover', 'volatility', 'market_cap',
            'pe_ratio', 'pb_ratio', 'turnover_rate', 'amplitude',
            'avg_volume_5d', 'avg_turnover_5d', 'momentum_5d', 'momentum_20d'
        ]

        # 模型配置
        self.model_configs = {
            'importance': {
                'model_type': 'regression',
                'target': 'importance_score',
                'features': self.feature_columns
            },
            'liquidity': {
                'model_type': 'regression',
                'target': 'liquidity_score',
                'features': self.feature_columns
            },
            'volatility_sensitivity': {
                'model_type': 'classification',
                'target': 'high_volatility',
                'features': self.feature_columns
            }
        }

        # 确保模型目录存在
        os.makedirs(self.model_path, exist_ok=True)

        # 初始化策略层服务集成
        self._init_strategy_services()

        # 加载或训练模型
        self._load_or_train_models()

        logger.info("AI智能筛选器初始化完成，策略层服务集成完毕")

    def _init_strategy_services(self):
        """初始化策略层服务集成"""
        try:
            # 预初始化服务连接，避免运行时延迟
            _get_feature_engineering_service()
            _get_ml_training_service()
            _get_backtest_service()

            logger.info("策略层服务集成初始化完成")

        except Exception as e:
            logger.warning(f"策略层服务集成初始化失败: {e}")

    def _load_or_train_models(self):
        """加载或训练模型"""
        try:
            if not _SKLEARN_AVAILABLE:
                logger.warning("scikit-learn未安装，使用默认评分算法")
                return

            for model_name, config in self.model_configs.items():
                model_file = os.path.join(self.model_path, f"{model_name}_model.pkl")
                scaler_file = os.path.join(self.model_path, f"{model_name}_scaler.pkl")

                if os.path.exists(model_file) and os.path.exists(scaler_file):
                    # 加载已训练的模型
                    self._load_model(model_name)
                else:
                    # 训练新模型
                    logger.info(f"训练{model_name}模型...")
                    trained_model = self._train_model(model_name, config)
                    if trained_model is None:
                        logger.warning(f"{model_name}模型训练失败，将使用默认算法")

        except Exception as e:
            logger.error(f"模型加载/训练失败: {e}")
            logger.warning("将使用默认评分算法")

    def _load_model(self, model_name: str):
        """加载模型"""
        try:
            import joblib
            model_file = os.path.join(self.model_path, f"{model_name}_model.pkl")
            scaler_file = os.path.join(self.model_path, f"{model_name}_scaler.pkl")

            self.models[model_name] = joblib.load(model_file)
            self.scalers[model_name] = joblib.load(scaler_file)

            logger.info(f"成功加载{model_name}模型")

        except Exception as e:
            logger.error(f"加载{model_name}模型失败: {e}")

    def _train_model(self, model_name: str, config: Dict[str, Any]):
        """训练模型 - 使用专业的ML训练管道"""
        try:
            # 获取ML训练服务
            ml_service = _get_ml_training_service()

            if ml_service:
                # 使用专业的ML训练管道
                return self._train_with_ml_pipeline(model_name, config, ml_service)
            else:
                # 降级到原有的sklearn训练
                logger.warning(f"ML训练服务不可用，使用基础sklearn训练")
                return self._train_with_sklearn(model_name, config)

        except Exception as e:
            logger.error(f"训练{model_name}模型失败: {e}")
            return None

    def _train_with_ml_pipeline(self, model_name: str, config: Dict[str, Any], ml_service):
        """使用专业的ML训练管道训练模型"""
        try:
            # 生成训练数据
            X, y = self._generate_training_data(model_name, config)
            if X is None or len(X) == 0:
                logger.error(f"无法生成{model_name}模型的训练数据")
                return None

            # 准备训练数据
            training_data = {
                'features': X,
                'target': y,
                'feature_names': self.feature_columns
            }

            # 配置模型训练参数
            model_config = self._get_model_config_for_ml_pipeline(model_name, config)

            # 生成唯一的模型ID
            model_id = f"smart_filter_{model_name}_{int(time.time())}"

            # 使用ML训练服务训练模型
            success = ml_service.train_model(model_id, training_data, model_config)

            if success:
                # 获取训练结果
                model_info = ml_service.get_model_info(model_id)
                performance = ml_service.get_model_performance(model_id)

                # 保存模型信息
                self.models[model_name] = {
                    'model_id': model_id,
                    'ml_service': ml_service,
                    'model_info': model_info,
                    'performance': performance,
                    'feature_names': self.feature_columns
                }

                logger.info(f"使用ML管道成功训练{model_name}模型，性能: {performance}")
                return self.models[model_name]
            else:
                logger.error(f"ML管道训练{model_name}模型失败")
                return None

        except Exception as e:
            logger.error(f"ML管道训练失败: {e}")
            return None

    def _get_model_config_for_ml_pipeline(self, model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """获取ML管道的模型配置"""
        base_config = {
            'model_type': config.get('model_type', 'regression'),
            'hyperparameter_tuning': True,  # 启用超参数优化
            'feature_selection': True,     # 启用特征选择
            'cross_validation': True,      # 启用交叉验证
            'ensemble_methods': True,      # 启用集成方法
            'early_stopping': True,        # 启用早停
        }

        # 根据模型类型调整配置
        if model_name == 'importance':
            base_config.update({
                'algorithms': ['random_forest', 'xgboost', 'lightgbm', 'gradient_boosting'],
                'primary_metric': 'r2_score',
                'hyperparameter_search': 'bayesian'  # 贝叶斯优化
            })
        elif model_name == 'liquidity':
            base_config.update({
                'algorithms': ['random_forest', 'xgboost', 'lightgbm'],
                'primary_metric': 'neg_mean_squared_error',
                'hyperparameter_search': 'grid'  # 网格搜索
            })
        else:  # volatility_sensitivity
            base_config.update({
                'algorithms': ['gradient_boosting', 'xgboost', 'logistic_regression'],
                'primary_metric': 'f1_score',
                'hyperparameter_search': 'random'  # 随机搜索
            })

        return base_config

    def _train_with_sklearn(self, model_name: str, config: Dict[str, Any]):
        """使用基础sklearn训练模型（降级方案）"""
        try:
            if not _SKLEARN_AVAILABLE:
                logger.error("sklearn不可用，无法训练模型")
                return None

            # 生成训练数据
            X, y = self._generate_training_data(model_name, config)
            if X is None or len(X) == 0:
                logger.error(f"无法生成{model_name}模型的训练数据")
                return None

            # 数据标准化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # 创建模型
            if config['model_type'] == 'regression':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                model = GradientBoostingClassifier(n_estimators=100, random_state=42)

            # 训练模型
            model.fit(X_scaled, y)

            # 保存模型
            import joblib
            model_file = os.path.join(self.model_path, f"{model_name}_model.pkl")
            scaler_file = os.path.join(self.model_path, f"{model_name}_scaler.pkl")

            joblib.dump(model, model_file)
            joblib.dump(scaler, scaler_file)

            self.models[model_name] = model
            self.scalers[model_name] = scaler

            # 评估模型
            score = self._evaluate_model(model, X_scaled, y)
            logger.info(f"sklearn训练{model_name}模型完成，评分: {score:.3f}")

            return self.models[model_name]

        except Exception as e:
            logger.error(f"sklearn训练失败: {e}")
            return None

    def _generate_training_data(self, model_name: str, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """生成训练数据"""
        try:
            # 优先从已采集的数据集生成训练样本
            X, y = self._generate_training_data_from_real_stocks(model_name)

            # 如果数据库数据不足，使用合成数据作为补充
            min_samples = 200  # 最少需要的样本数量
            if len(X) < min_samples:
                logger.warning(f"数据库训练数据不足({len(X)})，使用合成数据补充到{min_samples}个样本")
                X_synth, y_synth = self._generate_synthetic_training_data(model_name, min_samples - len(X))
                if len(X) > 0:
                    X = np.vstack([X, X_synth])
                    y = np.concatenate([y, y_synth])
                else:
                    X, y = X_synth, y_synth

            logger.info(f"为{model_name}模型生成了{len(X)}个训练样本（优先使用已采集数据集）")
            return X, y

        except Exception as e:
            logger.error(f"生成训练数据失败: {e}")
            # 降级到纯合成数据
            logger.warning("降级使用合成训练数据")
            return self._generate_synthetic_training_data(model_name, 500)

    def _generate_training_data_from_real_stocks(self, model_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """从已采集的股票数据集生成训练数据"""
        try:
            from datetime import datetime, timedelta

            logger.info("从已采集的股票数据集生成训练数据...")

            # 获取数据库中已有的股票数据作为训练样本
            training_samples = self._extract_training_data_from_database()

            if not training_samples:
                logger.warning("数据库中没有足够的股票数据，使用合成数据作为备选")
                # 降级到合成数据获取，避免频繁调用AKShare接口
                return self._generate_synthetic_training_data(model_name, min_samples=50)

            # 转换为numpy数组
            X = np.array([list(sample['features'].values()) for sample in training_samples])

            # 生成目标变量
            y = self._generate_target_from_database_data(X, model_name, training_samples)

            logger.info(f"从已采集数据集生成了{len(X)}个训练样本")
            return X, y

        except Exception as e:
            logger.error(f"从已采集数据集生成训练数据失败: {e}")
            return np.array([]), np.array([])

    def _extract_training_data_from_database(self) -> List[Dict[str, Any]]:
        """从数据库提取训练数据"""
        try:
            from src.gateway.web.postgresql_persistence import query_stock_data_from_postgresql
            from datetime import datetime, timedelta

            training_samples = []

            # 获取最近30天的股票数据作为训练样本
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)

            # 选择最具代表性的股票作为训练样本（避免数据量过大）
            representative_symbols = [
                '000001', '000002', '600000', '600036', '600519',  # 大盘股
                '300001', '300002', '000858', '002594', '300750',  # 成长股
                '600276', '000568', '002415', '600887', '000661'   # 行业龙头
            ]

            logger.info(f"从数据库获取{len(representative_symbols)}只代表性股票的训练数据")

            # 从数据库查询数据
            stock_data = query_stock_data_from_postgresql(
                source_id="akshare_stock_a",  # 使用主要的A股数据源
                symbols=representative_symbols,
                start_date=start_date,
                end_date=end_date
            )

            # 处理每只股票的数据
            for symbol, df in stock_data.items():
                if df is not None and not df.empty:
                    try:
                        # 从DataFrame中提取特征
                        features = self._extract_features_from_dataframe(df)
                        if features:
                            training_samples.append({
                                'symbol': symbol,
                                'features': features,
                                'dataframe': df
                            })

                            # 限制每只股票的样本数量，避免数据过大
                            if len(training_samples) >= 200:  # 最多200个训练样本
                                break

                    except Exception as e:
                        logger.debug(f"处理股票{symbol}数据失败: {e}")
                        continue

            logger.info(f"成功从数据库提取了{len(training_samples)}个训练样本")
            return training_samples

        except Exception as e:
            logger.error(f"从数据库提取训练数据失败: {e}")
            return []

    def _extract_features_from_dataframe(self, df) -> Optional[Dict[str, float]]:
        """从股票DataFrame中提取特征 - 使用量化特征工程服务"""
        try:
            if df.empty or len(df) < 5:  # 至少需要5天的数据
                return None

            # 获取特征工程服务
            feature_service = _get_feature_engineering_service()

            if feature_service and hasattr(feature_service, 'extract_features'):
                # 使用量化特征工程服务提取高级特征
                try:
                    # 准备数据用于特征工程
                    stock_data = {
                        'ohlcv_data': df.to_dict('records'),  # OHLCV数据
                        'data_type': 'daily',  # 数据类型
                        'symbol': 'unknown'  # 符号（如果可用）
                    }

                    # 创建特征提取任务配置
                    feature_config = {
                        'task_type': 'comprehensive_features',  # 综合特征提取
                        'data': stock_data,
                        'feature_types': [
                            'technical_indicators',  # 技术指标 (RSI, MACD, Bollinger Bands等)
                            'price_patterns',       # 价格形态
                            'volume_analysis',      # 成交量分析
                            'volatility_measures',  # 波动率指标
                            'momentum_indicators',  # 动量指标
                            'trend_indicators',     # 趋势指标
                            'statistical_features'  # 统计特征
                        ],
                        'timeframes': ['daily', 'weekly'],  # 时间框架
                        'include_market_data': True  # 包含市场数据
                    }

                    # 调用特征工程服务
                    feature_result = feature_service.extract_features(feature_config)

                    if feature_result and 'features' in feature_result:
                        advanced_features = feature_result['features']
                        logger.debug(f"成功从量化特征工程服务提取{len(advanced_features)}个高级特征")

                        # 合并基础特征和高级特征
                        basic_features = self._extract_basic_features(df)
                        if basic_features:
                            advanced_features.update(basic_features)
                            return advanced_features
                        else:
                            return advanced_features

                except Exception as e:
                    logger.warning(f"量化特征工程服务调用失败，使用基础特征提取: {e}")

            # 降级到基础特征提取
            logger.info("使用基础特征提取方法")
            return self._extract_basic_features(df)

        except Exception as e:
            logger.debug(f"从DataFrame提取特征失败: {e}")
            return None

    def _extract_basic_features(self, df) -> Optional[Dict[str, float]]:
        """提取基础特征（降级方案）"""
        try:
            if df.empty or len(df) < 5:
                return None

            # 使用最近的数据点
            latest_data = df.iloc[-1]  # 最新一天的数据
            recent_data = df.tail(5)  # 最近5天的数据

            # 基本价格和成交信息
            close_price = float(latest_data.get('close_price', 0) or 0)
            volume = float(latest_data.get('volume', 0) or 0)
            amount = float(latest_data.get('amount', 0) or 0)

            if close_price <= 0 or volume <= 0:
                return None

            # 计算派生特征
            # 市值估算（基于成交量和价格的粗略估算）
            market_cap = close_price * volume * 0.01  # 假设流通比例1%

            # 换手率
            turnover_rate = float(latest_data.get('turnover_rate', 0) or 0)

            # 波动率（使用标准差）
            price_volatility = recent_data['close_price'].pct_change().std() * np.sqrt(252)  # 年化波动率

            # 振幅（日均振幅）
            amplitude = recent_data['amplitude'].mean() / 100 if 'amplitude' in recent_data.columns else 0.03

            # PE/PB估算（简化计算）
            pe_ratio = close_price / max(0.1, close_price * 0.05)  # 假设盈利能力
            pb_ratio = close_price / max(0.1, close_price * 0.8)   # 假设PB

            # 5日均量/额
            avg_volume_5d = recent_data['volume'].mean()
            avg_turnover_5d = recent_data['amount'].mean()

            # 动量指标
            momentum_5d = (close_price - recent_data['close_price'].iloc[0]) / recent_data['close_price'].iloc[0]
            momentum_20d = momentum_5d * 0.8  # 简化计算

            return {
                'price': close_price,
                'volume': volume,
                'turnover': amount,
                'volatility': price_volatility,
                'market_cap': market_cap,
                'pe_ratio': pe_ratio,
                'pb_ratio': pb_ratio,
                'turnover_rate': turnover_rate,
                'amplitude': amplitude,
                'avg_volume_5d': avg_volume_5d,
                'avg_turnover_5d': avg_turnover_5d,
                'momentum_5d': momentum_5d,
                'momentum_20d': momentum_20d
            }

        except Exception as e:
            logger.debug(f"基础特征提取失败: {e}")
            return None

    def _generate_target_from_database_data(self, X: np.ndarray, model_name: str, samples: List[Dict[str, Any]]) -> np.ndarray:
        """基于数据库数据生成目标变量"""
        try:
            if model_name == 'importance':
                # 重要性评分：基于市值、成交量、换手率的综合评分
                market_cap_scores = (X[:, 4] - X[:, 4].min()) / (X[:, 4].max() - X[:, 4].min() + 1e-6)
                volume_scores = (X[:, 1] - X[:, 1].min()) / (X[:, 1].max() - X[:, 1].min() + 1e-6)
                turnover_scores = (X[:, 7] - X[:, 7].min()) / (X[:, 7].max() - X[:, 7].min() + 1e-6)
                y = (market_cap_scores * 0.4 + volume_scores * 0.3 + turnover_scores * 0.3)

            elif model_name == 'liquidity':
                # 流动性评分：基于成交量、成交额和换手率
                volume_scores = (X[:, 1] - X[:, 1].min()) / (X[:, 1].max() - X[:, 1].min() + 1e-6)
                turnover_scores = (X[:, 7] - X[:, 7].min()) / (X[:, 7].max() - X[:, 7].min() + 1e-6)
                amount_scores = (X[:, 2] - X[:, 2].min()) / (X[:, 2].max() - X[:, 2].min() + 1e-6)
                y = (volume_scores * 0.4 + turnover_scores * 0.4 + amount_scores * 0.2)

            else:  # volatility_sensitivity
                # 波动敏感度：基于价格波动率分类
                volatility_threshold = np.percentile(X[:, 3], 70)  # 70分位数作为阈值
                y = (X[:, 3] > volatility_threshold).astype(int)

            return y.astype(float)

        except Exception as e:
            logger.error(f"生成数据库数据目标变量失败: {e}")
            return np.zeros(len(X))

    def _generate_training_data_from_live_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """从实时数据生成训练数据（降级方案）"""
        try:
            logger.info("使用实时数据作为训练数据备选方案...")

            # 获取实时股票数据作为训练样本
            import akshare as ak
            market_data = ak.stock_zh_a_spot_em()
            logger.info(f"获取到{len(market_data)}只股票的实时数据")

            training_samples = []

            for _, stock in market_data.iterrows():
                try:
                    # 提取特征
                    features = self._extract_real_stock_features(stock)
                    if features:
                        training_samples.append({
                            'symbol': str(stock.get('代码', '')),
                            'features': features
                        })

                        # 限制样本数量
                        if len(training_samples) >= 100:
                            break

                except Exception as e:
                    logger.debug(f"处理股票数据失败: {e}")
                    continue

            if not training_samples:
                logger.warning("未能从实时数据中提取到有效的训练样本")
                return np.array([]), np.array([])

            # 转换为numpy数组
            X = np.array([list(sample['features'].values()) for sample in training_samples])

            # 生成目标变量
            y = self._generate_target_from_live_data(X, training_samples)

            logger.info(f"从实时股票数据生成了{len(X)}个训练样本")
            return X, y

        except Exception as e:
            logger.error(f"从实时股票数据生成训练数据失败: {e}")
            return np.array([]), np.array([])

    def _extract_real_stock_features(self, stock_data) -> Optional[Dict[str, float]]:
        """从实时股票数据中提取特征"""
        try:
            # 从akshare数据中提取特征
            code = str(stock_data.get('代码', ''))
            if not code or len(code) != 6:
                return None

            # 基本价格和成交信息
            price = float(stock_data.get('最新价', 0) or 0)
            volume = float(str(stock_data.get('成交量(手)', '0')).replace('手', '')) * 100  # 转换为股数
            turnover = float(str(stock_data.get('成交额(万)', '0')).replace('万', '')) * 10000  # 转换为元

            if price <= 0 or volume <= 0:
                return None

            # 计算派生特征
            # 市值估算（基于成交量和价格的粗略估算）
            market_cap = price * volume * 0.01  # 假设流通比例1%

            # 换手率
            turnover_rate = turnover / max(market_cap, 1000000)  # 换手率

            # 波动率（使用振幅作为近似）
            volatility = abs(float(stock_data.get('涨跌幅', '0').replace('%', '')) / 100) * 0.5 + 0.02

            # 振幅
            amplitude = abs(float(stock_data.get('涨跌幅', '0').replace('%', '')) / 100)

            # PE/PB估算（简化计算）
            pe_ratio = price / max(0.1, price * 0.05)  # 假设盈利能力
            pb_ratio = price / max(0.1, price * 0.8)   # 假设PB

            # 5日均量/额（使用当前值作为近似）
            avg_volume_5d = volume * 0.8  # 假设5日均量稍低于当日
            avg_turnover_5d = turnover * 0.8

            # 动量指标（使用涨跌幅）
            momentum_5d = amplitude * 0.3
            momentum_20d = amplitude * 0.1

            return {
                'price': price,
                'volume': volume,
                'turnover': turnover,
                'volatility': volatility,
                'market_cap': market_cap,
                'pe_ratio': pe_ratio,
                'pb_ratio': pb_ratio,
                'turnover_rate': turnover_rate,
                'amplitude': amplitude,
                'avg_volume_5d': avg_volume_5d,
                'avg_turnover_5d': avg_turnover_5d,
                'momentum_5d': momentum_5d,
                'momentum_20d': momentum_20d
            }

        except Exception as e:
            logger.debug(f"提取股票特征失败: {e}")
            return None

    def _generate_target_from_live_data(self, X: np.ndarray, samples: List[Dict[str, Any]]) -> np.ndarray:
        """基于实时数据生成目标变量"""
        try:
            # 这里可以根据具体的模型需求调整目标变量计算
            # 目前使用简化的计算方法
            model_name = 'general'  # 通用计算

            if model_name == 'importance':
                market_cap_scores = (X[:, 4] - X[:, 4].min()) / (X[:, 4].max() - X[:, 4].min() + 1e-6)
                volume_scores = (X[:, 1] - X[:, 1].min()) / (X[:, 1].max() - X[:, 1].min() + 1e-6)
                y = market_cap_scores * 0.5 + volume_scores * 0.5
            else:
                y = np.zeros(len(X))

            return y.astype(float)

        except Exception as e:
            logger.error(f"生成实时数据目标变量失败: {e}")
            return np.zeros(len(X))

    def _generate_synthetic_training_data(self, model_name: str, n_samples: int = 50, min_samples: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """生成合成训练数据（降级方案）"""
        try:
            # 如果提供了min_samples，确保至少生成min_samples个样本
            if min_samples is not None and n_samples < min_samples:
                n_samples = min_samples
                
            logger.info(f"生成{n_samples}个合成训练样本用于{model_name}模型")

            np.random.seed(42)
            X = np.random.randn(n_samples, len(self.feature_columns))
            X = np.abs(X)  # 确保特征为正值

            # 根据模型类型生成目标变量
            if model_name == 'importance':
                y = (X[:, 4] * 0.4 + X[:, 1] * 0.3 + X[:, 8] * 0.3).astype(float)
            elif model_name == 'liquidity':
                y = (X[:, 1] * 0.6 + X[:, 7] * 0.4).astype(float)
            else:  # volatility_sensitivity
                y = (X[:, 3] > X[:, 3].mean()).astype(int)

            return X, y

        except Exception as e:
            logger.error(f"生成合成训练数据失败: {e}")
            return np.array([]), np.array([])

    def _evaluate_model(self, model, X: np.ndarray, y: np.ndarray) -> float:
        """评估模型性能"""
        try:
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X, y, cv=3, scoring='r2' if hasattr(model, 'predict') else 'accuracy')
            return scores.mean()
        except:
            return 0.0

    def predict_stock_importance(self, stocks_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        预测股票重要性评分

        Args:
            stocks_data: 股票数据列表

        Returns:
            股票代码到重要性评分的映射
        """
        try:
            if not stocks_data:
                return {}

            # 提取特征
            features_list = []
            stock_codes = []

            for stock in stocks_data:
                features = self._extract_features(stock)
                if features:
                    features_list.append([features[col] for col in self.feature_columns])
                    stock_codes.append(stock.get('code', ''))

            if not features_list:
                return {}

            X = np.array(features_list)

            # 使用模型预测
            predictions = self._predict_with_model('importance', X)

            # 归一化到0-1范围
            predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min() + 1e-6)

            return dict(zip(stock_codes, predictions.tolist()))

        except Exception as e:
            logger.error(f"股票重要性预测失败: {e}")
            return {}

    def predict_stock_liquidity(self, stocks_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        预测股票流动性评分

        Args:
            stocks_data: 股票数据列表

        Returns:
            股票代码到流动性评分的映射
        """
        try:
            if not stocks_data:
                return {}

            # 提取特征
            features_list = []
            stock_codes = []

            for stock in stocks_data:
                features = self._extract_features(stock)
                if features:
                    features_list.append([features[col] for col in self.feature_columns])
                    stock_codes.append(stock.get('code', ''))

            if not features_list:
                return {}

            X = np.array(features_list)

            # 使用模型预测
            predictions = self._predict_with_model('liquidity', X)

            # 归一化到0-1范围
            predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min() + 1e-6)

            return dict(zip(stock_codes, predictions.tolist()))

        except Exception as e:
            logger.error(f"股票流动性预测失败: {e}")
            return {}

    def _extract_features(self, stock_data: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """从股票数据中提取特征"""
        try:
            # 这里应该从实际的股票数据中提取特征
            # 目前使用简化的实现
            return {
                'price': float(stock_data.get('price', 0) or 0),
                'volume': float(stock_data.get('volume', 0) or 0),
                'turnover': float(stock_data.get('turnover', 0) or 0),
                'volatility': float(stock_data.get('volatility', 0.05) or 0.05),
                'market_cap': float(stock_data.get('market_cap', 0) or 0),
                'pe_ratio': float(stock_data.get('pe_ratio', 20) or 20),
                'pb_ratio': float(stock_data.get('pb_ratio', 1.5) or 1.5),
                'turnover_rate': float(stock_data.get('turnover_rate', 0.02) or 0.02),
                'amplitude': float(stock_data.get('amplitude', 0.03) or 0.03),
                'avg_volume_5d': float(stock_data.get('avg_volume_5d', 0) or 0),
                'avg_turnover_5d': float(stock_data.get('avg_turnover_5d', 0) or 0),
                'momentum_5d': float(stock_data.get('momentum_5d', 0) or 0),
                'momentum_20d': float(stock_data.get('momentum_20d', 0) or 0)
            }

        except Exception as e:
            logger.debug(f"提取股票特征失败: {e}")
            return None

    def _predict_with_model(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """使用训练好的模型进行预测"""
        try:
            model_info = self.models.get(model_name)

            if model_info and isinstance(model_info, dict) and 'ml_service' in model_info:
                # 使用ML服务进行预测
                ml_service = model_info['ml_service']
                model_id = model_info['model_id']

                # 准备推理请求
                inference_request = {
                    'model_id': model_id,
                    'features': X,
                    'feature_names': self.feature_columns
                }

                # 执行推理
                result = ml_service.infer(model_id, inference_request)
                if result and result.get('success'):
                    return np.array(result.get('predictions', []))
                else:
                    logger.warning(f"ML服务推理失败，使用默认算法")

            # 使用sklearn模型或默认算法
            if model_name in self.models and model_name in self.scalers:
                X_scaled = self.scalers[model_name].transform(X)
                model = self.models[model_name]
                if hasattr(model, 'predict'):
                    return model.predict(X_scaled)

            # 默认算法
            if model_name == 'importance':
                return self._default_importance_scoring(X)
            elif model_name == 'liquidity':
                return self._default_liquidity_scoring(X)
            else:
                return np.zeros(len(X))

        except Exception as e:
            logger.error(f"模型预测失败: {e}")
            return np.zeros(len(X))

    def _default_importance_scoring(self, X: np.ndarray) -> np.ndarray:
        """默认重要性评分算法"""
        # 基于市值(4)、成交量(1)、振幅(8)的加权评分
        return X[:, 4] * 0.4 + X[:, 1] * 0.3 + X[:, 8] * 0.3

    def _default_liquidity_scoring(self, X: np.ndarray) -> np.ndarray:
        """默认流动性评分算法"""
        # 基于成交量(1)和换手率(7)的评分
        return X[:, 1] * 0.6 + X[:, 7] * 0.4

    def select_optimal_stocks(self,
                             all_stocks: List[str],
                             strategy_config: Dict[str, Any],
                             market_state: Optional[MarketState] = None,
                             target_size: int = 100) -> List[str]:
        """
        根据策略配置和市场状态选择最优股票池（集成回测验证）

        Args:
            all_stocks: 所有可用股票
            strategy_config: 策略配置
            market_state: 当前市场状态
            target_size: 目标池大小

        Returns:
            选择的股票代码列表
        """
        try:
            if not all_stocks:
                return []

            strategy_type = strategy_config.get('strategy_id', 'multi_factor')

            # 获取股票基础数据
            stocks_data = self._get_stocks_data(all_stocks)

            # 计算AI模型评分
            importance_scores = self.predict_stock_importance(stocks_data)
            liquidity_scores = self.predict_stock_liquidity(stocks_data)

            # 策略特定的评分权重
            weights = self._get_strategy_weights(strategy_type)

            # 计算综合评分
            combined_scores = {}
            for stock_code in all_stocks:
                imp_score = importance_scores.get(stock_code, 0.5)
                liq_score = liquidity_scores.get(stock_code, 0.5)

                combined_score = (
                    imp_score * weights['importance'] +
                    liq_score * weights['liquidity']
                )

                # 市场状态调整
                if market_state:
                    combined_score = self._adjust_for_market_state(
                        combined_score, market_state, stock_code
                    )

                combined_scores[stock_code] = combined_score

            # 按评分排序并选择候选股票（选择更多用于回测验证）
            candidates_count = min(target_size * 2, len(all_stocks))
            sorted_stocks = sorted(combined_scores.items(),
                                 key=lambda x: x[1], reverse=True)

            candidate_stocks = [code for code, score in sorted_stocks[:candidates_count]]

            # 使用回测验证优化最终选择
            validate_with_backtest = (
                strategy_config.get('validate_with_backtest', False) or
                strategy_config.get('backtest_validation', False) or
                strategy_config.get('enable_backtest', False)
            )

            if validate_with_backtest and len(candidate_stocks) >= target_size:
                final_stocks = self._optimize_with_backtest_validation(
                    candidate_stocks, strategy_config, target_size
                )
            else:
                final_stocks = candidate_stocks[:target_size]

            logger.info(f"AI智能筛选：从{len(all_stocks)}只股票中选择了{len(final_stocks)}只，策略:{strategy_type}")

            return final_stocks

        except Exception as e:
            logger.error(f"AI智能股票选择失败: {e}")
            # 返回前N个作为fallback
            return all_stocks[:target_size]

    def _get_strategy_weights(self, strategy_type: str) -> Dict[str, float]:
        """获取策略特定的评分权重"""
        weights_map = {
            'hf_trading': {'importance': 0.3, 'liquidity': 0.7},  # 高频交易重视流动性
            'multi_factor': {'importance': 0.6, 'liquidity': 0.4},  # 多因子均衡考虑
            'market_making': {'importance': 0.4, 'liquidity': 0.6},  # 做市重视流动性
            'stat_arb': {'importance': 0.5, 'liquidity': 0.5},  # 统计套利均衡考虑
            'momentum': {'importance': 0.7, 'liquidity': 0.3}  # 动量策略重视重要性
        }

        return weights_map.get(strategy_type, {'importance': 0.5, 'liquidity': 0.5})

    def _adjust_for_market_state(self, score: float, market_state: MarketState, stock_code: str) -> float:
        """根据市场状态调整股票评分"""
        try:
            # 高波动市场：降低对波动敏感股票的评分
            if market_state.volatility_index > 0.05:  # 5%以上波动
                # 这里可以根据股票的波动敏感度模型进行调整
                score *= 0.9

            # 低迷市场：提高流动性好的股票评分
            if market_state.market_sentiment < -0.5:
                score *= 1.1  # 提高流动性权重

            return score

        except Exception as e:
            logger.warning(f"市场状态调整失败: {e}")
            return score

    def _get_stocks_data(self, stock_codes: List[str]) -> List[Dict[str, Any]]:
        """获取股票数据（简化实现）"""
        # 这里应该从实际的数据源获取股票的实时数据
        # 目前返回模拟数据
        stocks_data = []
        for code in stock_codes[:500]:  # 限制数量避免性能问题
            stocks_data.append({
                'code': code,
                'price': np.random.uniform(5, 200),
                'volume': np.random.uniform(10000, 1000000),
                'turnover': np.random.uniform(100000, 10000000),
                'volatility': np.random.uniform(0.01, 0.1),
                'market_cap': np.random.uniform(100000000, 10000000000),
                'pe_ratio': np.random.uniform(5, 50),
                'pb_ratio': np.random.uniform(0.5, 5),
                'turnover_rate': np.random.uniform(0.001, 0.1),
                'amplitude': np.random.uniform(0.01, 0.15),
                'avg_volume_5d': np.random.uniform(50000, 5000000),
                'avg_turnover_5d': np.random.uniform(500000, 50000000),
                'momentum_5d': np.random.uniform(-0.1, 0.1),
                'momentum_20d': np.random.uniform(-0.2, 0.2)
            })

        return stocks_data

    def _optimize_with_backtest_validation(self,
                                         candidate_stocks: List[str],
                                         strategy_config: Dict[str, Any],
                                         target_size: int) -> List[str]:
        """
        使用回测验证优化股票池选择

        Args:
            candidate_stocks: 候选股票列表
            strategy_config: 策略配置
            target_size: 目标池大小

        Returns:
            优化后的股票列表
        """
        try:
            backtest_service = _get_backtest_service()
            if not backtest_service:
                logger.warning("回测服务不可用，返回AI评分选择结果")
                return candidate_stocks[:target_size]

            # 生成多个投资组合进行回测比较
            portfolio_candidates = self._generate_portfolio_candidates(
                candidate_stocks, target_size
            )

            best_portfolio = None
            best_score = float('-inf')

            logger.info(f"开始回测验证，共测试{len(portfolio_candidates)}个组合")

            for i, portfolio in enumerate(portfolio_candidates):
                try:
                    # 执行回测
                    backtest_result = self._run_portfolio_backtest(
                        portfolio, strategy_config
                    )

                    if backtest_result:
                        # 评估回测结果
                        score = self._evaluate_backtest_result(backtest_result)

                        if score > best_score:
                            best_score = score
                            best_portfolio = portfolio

                        logger.debug(f"组合{i+1}回测完成，得分: {score:.3f}")

                except Exception as e:
                    logger.warning(f"组合{i+1}回测失败: {e}")
                    continue

            if best_portfolio:
                logger.info(f"回测优化完成，最佳组合得分: {best_score:.3f}")
                return best_portfolio
            else:
                logger.warning("回测优化失败，返回AI评分选择结果")
                return candidate_stocks[:target_size]

        except Exception as e:
            logger.error(f"回测验证优化失败: {e}")
            return candidate_stocks[:target_size]

    def _generate_portfolio_candidates(self, candidate_stocks: List[str], target_size: int) -> List[List[str]]:
        """生成多个投资组合用于回测比较"""
        try:
            import random
            random.seed(42)  # 确保结果可重现

            portfolios = []

            # 生成不同的组合
            for _ in range(min(5, len(candidate_stocks) // target_size)):  # 最多5个组合
                # 随机选择目标数量的股票
                portfolio = random.sample(candidate_stocks, min(target_size, len(candidate_stocks)))
                portfolios.append(portfolio)

            # 确保至少有一个组合（前N个）
            if not portfolios and candidate_stocks:
                portfolios.append(candidate_stocks[:target_size])

            return portfolios

        except Exception as e:
            logger.error(f"生成组合候选失败: {e}")
            return [candidate_stocks[:target_size]] if candidate_stocks else []

    def _run_portfolio_backtest(self, portfolio: List[str], strategy_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """运行投资组合回测"""
        try:
            backtest_service = _get_backtest_service()
            if not backtest_service:
                return None

            # 构建回测配置
            backtest_config = {
                "strategy": {
                    "name": "AI_Selected_Portfolio_Strategy",
                    "type": "portfolio_strategy",
                    "selected_stocks": portfolio,
                    "rebalancing_period": "weekly",
                    "risk_management": {
                        "max_position_size": 0.1,
                        "stop_loss": strategy_config.get('stop_loss', 0.05),
                        "take_profit": strategy_config.get('take_profit', 0.15)
                    }
                },
                "data": {
                    "start_date": "2023-01-01",
                    "end_date": "2024-01-01",
                    "benchmark": "000001"  # 上证指数作为基准
                },
                "capital": {
                    "initial_capital": 1000000,
                    "currency": "CNY"
                },
                "settings": {
                    "commission": 0.0003,  # 交易佣金
                    "slippage": 0.0001    # 滑点
                }
            }

            # 执行回测
            result = backtest_service.run_backtest(backtest_config)

            if result and result.get('success'):
                return result.get('data', {})
            else:
                logger.warning("回测执行失败")
                return None

        except Exception as e:
            logger.error(f"运行投资组合回测失败: {e}")
            return None

    def _evaluate_backtest_result(self, backtest_result: Dict[str, Any]) -> float:
        """评估回测结果质量"""
        try:
            performance = backtest_result.get('performance', {})
            risk = backtest_result.get('risk', {})

            # 计算综合评分
            sharpe_ratio = performance.get('sharpe_ratio', 0)
            max_drawdown = risk.get('max_drawdown', 1)
            annual_return = performance.get('annual_return', 0)
            win_rate = performance.get('win_rate', 0)

            # 归一化各项指标
            sharpe_score = min(max(sharpe_ratio / 2, 0), 1)  # Sharpe > 2 为满分
            drawdown_score = 1 - min(max_drawdown, 0.5) / 0.5  # 最大回撤 < 50% 为满分
            return_score = min(max(annual_return / 0.3, 0), 1)  # 年化收益 > 30% 为满分
            win_rate_score = win_rate  # 胜率直接使用

            # 加权综合评分
            total_score = (
                sharpe_score * 0.4 +      # 夏普比率权重最高
                drawdown_score * 0.3 +    # 最大回撤重要性次之
                return_score * 0.2 +      # 年化收益
                win_rate_score * 0.1      # 胜率
            )

            return total_score

        except Exception as e:
            logger.error(f"评估回测结果失败: {e}")
            return 0.0


# 全局单例实例
_smart_filter_instance = None

def get_smart_stock_filter() -> SmartStockFilter:
    """获取智能股票筛选器单例实例"""
    global _smart_filter_instance
    if _smart_filter_instance is None:
        _smart_filter_instance = SmartStockFilter()
    return _smart_filter_instance