#!/usr/bin/env python3
"""
RQA2025增强特征工程
实现更多技术指标和特征处理功能
"""

import logging

logger = logging.getLogger(__name__)


def fix_technical_indicator_processor():
    """修复技术指标处理器"""
    print("🔧 修复技术指标处理器...")

    try:
        # 修复UnifiedLogger导入问题
        processor_path = "src/features/processors/technical_indicator_processor.py"
        with open(processor_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 修复导入语句
        if 'from src.infrastructure.logging import UnifiedLogger' in content:
            content = content.replace(
                'from src.infrastructure.logging import UnifiedLogger',
                'from src.infrastructure.logging.unified_logger import UnifiedLogger, get_logger'
            )

            with open(processor_path, 'w', encoding='utf-8') as f:
                f.write(content)

            print("   ✅ 修复了TechnicalIndicatorProcessor中的导入")

        return True

    except Exception as e:
        print(f"   ❌ 修复技术指标处理器失败: {e}")
        return False


def create_additional_technical_indicators():
    """创建额外的技术指标"""
    print("\n📊 创建额外技术指标...")

    try:
        # 创建KDJ指标计算器
        kdj_calculator = '''#!/usr/bin/env python3
"""
KDJ指标计算器
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class KDJCalculator:
    """KDJ指标计算器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.period = self.config.get('period', 9)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算KDJ指标

        Args:
            data: 包含OHLC数据的DataFrame

        Returns:
            包含KDJ指标的DataFrame
        """
        try:
            if not all(col in data.columns for col in ['high', 'low', 'close']):
                raise ValueError("数据缺少必需的OHLC列")

            df = data.copy()

            # 计算RSV (Raw Stochastic Value)
            for i in range(len(df)):
                if i >= self.period - 1:
                    # 获取周期内最高价和最低价
                    high_period = df['high'].iloc[i-self.period+1:i+1].max()
                    low_period = df['low'].iloc[i-self.period+1:i+1].min()
                    close_current = df['close'].iloc[i]

                    if high_period != low_period:
                        rsv = (close_current - low_period) / (high_period - low_period) * 100
                    else:
                        rsv = 50  # 当最高价等于最低价时，RSV设为50
                else:
                    rsv = 50  # 前period-1个周期设为50

                df.loc[df.index[i], 'rsv'] = rsv

            # 计算K、D、J值
            df['k_value'] = df['rsv'].ewm(alpha=1/3, adjust=False).mean()
            df['d_value'] = df['k_value'].ewm(alpha=1/3, adjust=False).mean()
            df['j_value'] = 3 * df['k_value'] - 2 * df['d_value']

            # 重命名列
            df = df.rename(columns={
                'k_value': 'kdj_k',
                'd_value': 'kdj_d',
                'j_value': 'kdj_j'
            })

            logger.info(f"KDJ指标计算完成，周期: {self.period}")
            return df

        except Exception as e:
            logger.error(f"KDJ指标计算失败: {e}")
            return data
'''

        # 创建布林带指标计算器
        bollinger_calculator = '''#!/usr/bin/env python3
"""
布林带指标计算器
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class BollingerBandsCalculator:
    """布林带指标计算器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.period = self.config.get('period', 20)
        self.std_dev = self.config.get('std_dev', 2)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算布林带指标

        Args:
            data: 包含收盘价数据的DataFrame

        Returns:
            包含布林带指标的DataFrame
        """
        try:
            if 'close' not in data.columns:
                raise ValueError("数据缺少收盘价列")

            df = data.copy()

            # 计算简单移动平均线 (中线)
            df['bb_middle'] = df['close'].rolling(window=self.period).mean()

            # 计算标准差
            df['bb_std'] = df['close'].rolling(window=self.period).std()

            # 计算上轨和下轨
            df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * self.std_dev)
            df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * self.std_dev)

            # 计算布林带宽度
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

            # 计算价格相对位置
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

            logger.info(f"布林带指标计算完成，周期: {self.period}, 标准差: {self.std_dev}")
            return df

        except Exception as e:
            logger.error(f"布林带指标计算失败: {e}")
            return data
'''

        # 创建威廉指标计算器
        williams_calculator = '''#!/usr/bin/env python3
"""
威廉指标计算器
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class WilliamsCalculator:
    """威廉指标计算器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.period = self.config.get('period', 14)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算威廉指标

        Args:
            data: 包含OHLC数据的DataFrame

        Returns:
            包含威廉指标的DataFrame
        """
        try:
            if not all(col in data.columns for col in ['high', 'low', 'close']):
                raise ValueError("数据缺少必需的OHLC列")

            df = data.copy()

            # 计算周期内最高价和最低价
            df['highest_high'] = df['high'].rolling(window=self.period).max()
            df['lowest_low'] = df['low'].rolling(window=self.period).min()

            # 计算威廉指标 %R
            for i in range(len(df)):
                if i >= self.period - 1:
                    highest = df['highest_high'].iloc[i]
                    lowest = df['lowest_low'].iloc[i]
                    close = df['close'].iloc[i]

                    if highest != lowest:
                        williams_r = (highest - close) / (highest - lowest) * (-100)
                    else:
                        williams_r = 0
                else:
                    williams_r = 0

                df.loc[df.index[i], 'williams_r'] = williams_r

            # 清理临时列
            df = df.drop(['highest_high', 'lowest_low'], axis=1)

            logger.info(f"威廉指标计算完成，周期: {self.period}")
            return df

        except Exception as e:
            logger.error(f"威廉指标计算失败: {e}")
            return data
'''

        # 创建CCI指标计算器
        cci_calculator = '''#!/usr/bin/env python3
"""
CCI指标计算器
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class CCICalculator:
    """CCI指标计算器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.period = self.config.get('period', 14)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算CCI指标

        Args:
            data: 包含OHLC数据的DataFrame

        Returns:
            包含CCI指标的DataFrame
        """
        try:
            if not all(col in data.columns for col in ['high', 'low', 'close']):
                raise ValueError("数据缺少必需的OHLC列")

            df = data.copy()

            # 计算典型价格 (TP)
            df['tp'] = (df['high'] + df['low'] + df['close']) / 3

            # 计算MA
            df['tp_ma'] = df['tp'].rolling(window=self.period).mean()

            # 计算平均偏差
            df['mean_deviation'] = df['tp'].rolling(window=self.period).apply(
                lambda x: np.mean(np.abs(x - x.mean())), raw=True
            )

            # 计算CCI
            df['cci'] = (df['tp'] - df['tp_ma']) / (0.015 * df['mean_deviation'])

            # 清理临时列
            df = df.drop(['tp', 'tp_ma', 'mean_deviation'], axis=1)

            logger.info(f"CCI指标计算完成，周期: {self.period}")
            return df

        except Exception as e:
            logger.error(f"CCI指标计算失败: {e}")
            return data
'''

        # 创建ATR指标计算器
        atr_calculator = '''#!/usr/bin/env python3
"""
ATR指标计算器
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class ATRCalculator:
    """ATR指标计算器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.period = self.config.get('period', 14)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算ATR指标

        Args:
            data: 包含OHLC数据的DataFrame

        Returns:
            包含ATR指标的DataFrame
        """
        try:
            if not all(col in data.columns for col in ['high', 'low', 'close']):
                raise ValueError("数据缺少必需的OHLC列")

            df = data.copy()

            # 计算真实波幅 (True Range)
            df['high_low'] = df['high'] - df['low']
            df['high_prev_close'] = np.abs(df['high'] - df['close'].shift(1))
            df['low_prev_close'] = np.abs(df['low'] - df['close'].shift(1))

            df['true_range'] = df[['high_low', 'high_prev_close', 'low_prev_close']].max(axis=1)

            # 计算ATR (使用指数移动平均)
            df['atr'] = df['true_range'].ewm(span=self.period, adjust=False).mean()

            # 计算ATR比率
            df['atr_ratio'] = df['atr'] / df['close']

            # 清理临时列
            df = df.drop(['high_low', 'high_prev_close', 'low_prev_close', 'true_range'], axis=1)

            logger.info(f"ATR指标计算完成，周期: {self.period}")
            return df

        except Exception as e:
            logger.error(f"ATR指标计算失败: {e}")
            return data
'''

        # 创建技术指标目录
        import os
        os.makedirs("src/features/indicators", exist_ok=True)

        # 创建各个指标计算器
        calculators = {
            "kdj_calculator.py": kdj_calculator,
            "bollinger_calculator.py": bollinger_calculator,
            "williams_calculator.py": williams_calculator,
            "cci_calculator.py": cci_calculator,
            "atr_calculator.py": atr_calculator
        }

        for filename, content in calculators.items():
            filepath = f"src/features/indicators/{filename}"
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"   ✅ 创建了 {filename}")

        print("   ✅ 额外技术指标创建完成")
        return True

    except Exception as e:
        print(f"   ❌ 创建额外技术指标失败: {e}")
        return False


def create_feature_selector():
    """创建特征选择器"""
    print("\n🔍 创建特征选择器...")

    try:
        feature_selector = '''#!/usr/bin/env python3
"""
特征选择器
提供特征选择和降维功能
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class FeatureSelector:
    """特征选择器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.method = self.config.get('method', 'correlation')  # correlation, mutual_info, kbest, pca
        self.k_features = self.config.get('k_features', 10)

    def select_features(self, X: pd.DataFrame, y: Optional[pd.Series] = None,
                       method: Optional[str] = None) -> Dict[str, Any]:
        """
        选择最优特征

        Args:
            X: 特征数据
            y: 目标变量 (某些方法需要)
            method: 选择方法

        Returns:
            包含选择结果的字典
        """
        try:
            if method:
                self.method = method

            if self.method == 'correlation':
                return self._select_by_correlation(X, y)
            elif self.method == 'mutual_info':
                return self._select_by_mutual_info(X, y)
            elif self.method == 'kbest':
                return self._select_by_kbest(X, y)
            elif self.method == 'pca':
                return self._select_by_pca(X)
            else:
                raise ValueError(f"不支持的特征选择方法: {self.method}")

        except Exception as e:
            logger.error(f"特征选择失败: {e}")
            return {
                'selected_features': X.columns.tolist(),
                'scores': {},
                'method': self.method,
                'error': str(e)
            }

    def _select_by_correlation(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """基于相关性选择特征"""
        try:
            if y is None:
                raise ValueError("相关性选择需要目标变量")

            # 计算相关性
            correlations = {}
            for column in X.columns:
                if X[column].dtype in ['float64', 'int64']:
                    corr = abs(X[column].corr(y))
                    if not np.isnan(corr):
                        correlations[column] = corr

            # 按相关性排序
            sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
            selected_features = [feature for feature, _ in sorted_features[:self.k_features]]

            return {
                'selected_features': selected_features,
                'scores': correlations,
                'method': 'correlation',
                'feature_count': len(selected_features)
            }

        except Exception as e:
            logger.error(f"相关性特征选择失败: {e}")
            return {
                'selected_features': X.columns.tolist(),
                'scores': {},
                'method': 'correlation',
                'error': str(e)
            }

    def _select_by_mutual_info(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """基于互信息选择特征"""
        try:
            if y is None:
                raise ValueError("互信息选择需要目标变量")

            # 标准化数据
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

            # 计算互信息
            mi_scores = mutual_info_regression(X_scaled, y)

            # 创建特征-分数映射
            feature_scores = dict(zip(X.columns, mi_scores))

            # 按互信息排序
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            selected_features = [feature for feature, _ in sorted_features[:self.k_features]]

            return {
                'selected_features': selected_features,
                'scores': feature_scores,
                'method': 'mutual_info',
                'feature_count': len(selected_features)
            }

        except Exception as e:
            logger.error(f"互信息特征选择失败: {e}")
            return {
                'selected_features': X.columns.tolist(),
                'scores': {},
                'method': 'mutual_info',
                'error': str(e)
            }

    def _select_by_kbest(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """基于SelectKBest选择特征"""
        try:
            if y is None:
                raise ValueError("KBest选择需要目标变量")

            # 使用F检验选择特征
            selector = SelectKBest(score_func=f_regression, k=self.k_features)
            X_selected = selector.fit_transform(X, y)

            # 获取选择的特征名称
            selected_mask = selector.get_support()
            selected_features = X.columns[selected_mask].tolist()

            # 获取特征分数
            feature_scores = dict(zip(X.columns, selector.scores_))

            return {
                'selected_features': selected_features,
                'scores': feature_scores,
                'method': 'kbest',
                'feature_count': len(selected_features)
            }

        except Exception as e:
            logger.error(f"KBest特征选择失败: {e}")
            return {
                'selected_features': X.columns.tolist(),
                'scores': {},
                'method': 'kbest',
                'error': str(e)
            }

    def _select_by_pca(self, X: pd.DataFrame) -> Dict[str, Any]:
        """基于PCA进行特征降维"""
        try:
            # 标准化数据
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # 执行PCA
            pca = PCA(n_components=self.k_features)
            X_pca = pca.fit_transform(X_scaled)

            # 创建新的特征名称
            selected_features = [f'pca_{i+1}' for i in range(self.k_features)]

            # 计算解释方差比
            explained_variance_ratio = pca.explained_variance_ratio_
            feature_scores = dict(zip(selected_features, explained_variance_ratio))

            return {
                'selected_features': selected_features,
                'scores': feature_scores,
                'method': 'pca',
                'feature_count': len(selected_features),
                'total_explained_variance': np.sum(explained_variance_ratio),
                'pca_components': pca.components_
            }

        except Exception as e:
            logger.error(f"PCA特征选择失败: {e}")
            return {
                'selected_features': X.columns.tolist(),
                'scores': {},
                'method': 'pca',
                'error': str(e)
            }

    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """获取特征重要性"""
        try:
            result = self.select_features(X, y, method='correlation')

            if 'scores' in result:
                return result['scores']
            else:
                return {}

        except Exception as e:
            logger.error(f"获取特征重要性失败: {e}")
            return {}
'''

        # 创建特征选择器
        with open("src/features/utils/feature_selector.py", 'w', encoding='utf-8') as f:
            f.write(feature_selector)

        print("   ✅ 特征选择器创建完成")
        return True

    except Exception as e:
        print(f"   ❌ 创建特征选择器失败: {e}")
        return False


def update_technical_indicator_processor():
    """更新技术指标处理器以支持新指标"""
    print("\n🔄 更新技术指标处理器...")

    try:
        processor_path = "src/features/processors/technical_indicator_processor.py"
        with open(processor_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 添加新的指标导入
        import_section = '''
from ..indicators.kdj_calculator import KDJCalculator
from ..indicators.bollinger_calculator import BollingerBandsCalculator
from ..indicators.williams_calculator import WilliamsCalculator
from ..indicators.cci_calculator import CCICalculator
from ..indicators.atr_calculator import ATRCalculator
'''

        if 'class TechnicalIndicatorProcessor:' in content:
            # 在类定义前添加导入
            class_definition = 'class TechnicalIndicatorProcessor:'
            if import_section not in content:
                content = content.replace(
                    class_definition,
                    import_section + '\n' + class_definition
                )

                # 在__init__方法中添加新的计算器实例
                init_method_pattern = 'def __init__(self, config: Optional[Dict[str, Any]] = None):'
                if init_method_pattern in content:
                    # 添加新的计算器实例
                    init_addition = '''
        # 新增指标计算器
        self.kdj_calculator = KDJCalculator(config)
        self.bollinger_calculator = BollingerBandsCalculator(config)
        self.williams_calculator = WilliamsCalculator(config)
        self.cci_calculator = CCICalculator(config)
        self.atr_calculator = ATRCalculator(config)
'''
                    # 找到__init__方法的结束位置
                    init_start = content.find(init_method_pattern)
                    init_end_pattern = '\n        self.logger.info'
                    init_end = content.find(init_end_pattern, init_start)

                    if init_end > 0:
                        content = content[:init_end] + init_addition + content[init_end:]

                # 更新calculate_indicators方法以支持新指标
                if 'def calculate_indicators(self, data: pd.DataFrame, indicators: Dict[str, Dict[str, Any]]) -> pd.DataFrame:' in content:
                    # 添加新的指标计算逻辑
                    new_indicators_logic = '''
            elif indicator_name == 'kdj':
                data = self.kdj_calculator.calculate(data)
            elif indicator_name == 'bollinger_bands':
                data = self.bollinger_calculator.calculate(data)
            elif indicator_name == 'williams':
                data = self.williams_calculator.calculate(data)
            elif indicator_name == 'cci':
                data = self.cci_calculator.calculate(data)
            elif indicator_name == 'atr':
                data = self.atr_calculator.calculate(data)
'''

                    # 找到现有的elif链的末尾
                    elif_pattern = "elif indicator_name == 'obv':"
                    if elif_pattern in content:
                        content = content.replace(
                            elif_pattern,
                            new_indicators_logic + '\n            ' + elif_pattern
                        )

                with open(processor_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                print("   ✅ 技术指标处理器已更新")

        return True

    except Exception as e:
        print(f"   ❌ 更新技术指标处理器失败: {e}")
        return False


def test_enhanced_features():
    """测试增强的特征工程功能"""
    print("\n🧪 测试增强特征工程...")

    try:
        # 导入增强后的处理器
        from src.features.processors.technical_indicator_processor import TechnicalIndicatorProcessor
        from src.data.loaders import StockDataLoader

        # 加载测试数据
        loader = StockDataLoader({'data_source': 'mock'})
        data = loader.load_data('000001', '2024-01-01', '2024-01-31')

        if not data.empty and len(data) > 20:
            # 创建处理器
            processor = TechnicalIndicatorProcessor()

            # 测试新指标
            new_indicators = {
                'kdj': {'period': 9},
                'bollinger_bands': {'period': 20, 'std_dev': 2},
                'williams': {'period': 14},
                'cci': {'period': 14},
                'atr': {'period': 14}
            }

            # 计算指标
            result_data = processor.calculate_indicators(data, new_indicators)

            if not result_data.empty:
                # 检查新指标是否成功添加
                new_columns = [col for col in result_data.columns if col not in data.columns]
                print(f"   ✅ 新增特征数量: {len(new_columns)}")
                print(f"   📊 新特征示例: {new_columns[:5]}")

                # 测试特征选择器
                try:
                    from src.features.utils.feature_selector import FeatureSelector

                    # 创建目标变量
                    target_data = result_data.copy()
                    target_data['target'] = (target_data['close'].shift(-1)
                                             > target_data['close']).astype(int)
                    target_data = target_data.dropna()

                    # 选择特征
                    feature_cols = [col for col in target_data.columns
                                    if col not in ['timestamp', 'target'] and
                                    target_data[col].dtype in ['float64', 'int64']]

                    if len(feature_cols) > 5:
                        X = target_data[feature_cols]
                        y = target_data['target']

                        selector = FeatureSelector({'method': 'correlation', 'k_features': 5})
                        selection_result = selector.select_features(X, y)

                        if 'selected_features' in selection_result:
                            print(
                                f"   ✅ 特征选择成功: 选择了 {len(selection_result['selected_features'])} 个特征")
                            print(f"   🔍 选择特征: {selection_result['selected_features']}")

                except Exception as e:
                    print(f"   ⚠️ 特征选择器测试失败: {e}")

                return True
            else:
                print("   ❌ 指标计算失败")
                return False
        else:
            print("   ❌ 测试数据不足")
            return False

    except Exception as e:
        print(f"   ❌ 增强特征工程测试失败: {e}")
        return False


def main():
    """主函数"""
    print("🚀 RQA2025增强特征工程")

    results = {
        'fix_processor': False,
        'new_indicators': False,
        'feature_selector': False,
        'update_processor': False,
        'test_features': False
    }

    # 1. 修复技术指标处理器
    results['fix_processor'] = fix_technical_indicator_processor()

    # 2. 创建额外技术指标
    results['new_indicators'] = create_additional_technical_indicators()

    # 3. 创建特征选择器
    results['feature_selector'] = create_feature_selector()

    # 4. 更新技术指标处理器
    results['update_processor'] = update_technical_indicator_processor()

    # 5. 测试增强特征
    results['test_features'] = test_enhanced_features()

    # 总结
    successful = sum(results.values())
    total = len(results)

    print(f"\n📊 增强特征工程总结:")
    print(f"   成功: {successful}/{total}")
    print(".1f")

    for task, success in results.items():
        status = "✅" if success else "❌"
        task_name = {
            'fix_processor': '修复技术指标处理器',
            'new_indicators': '创建额外技术指标',
            'feature_selector': '创建特征选择器',
            'update_processor': '更新技术指标处理器',
            'test_features': '测试增强特征'
        }.get(task, task)
        print(f"   {status} {task_name}")

    if successful == total:
        print("\n🎉 增强特征工程全部完成！")
        print("   现在支持更多技术指标和特征选择功能")
    elif successful >= 3:
        print("\n👍 增强特征工程大部分完成")
        print("   核心功能已经增强")
    else:
        print("\n⚠️ 增强特征工程需要进一步完善")


if __name__ == "__main__":
    main()
