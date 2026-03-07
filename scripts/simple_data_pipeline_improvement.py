#!/usr/bin/env python3
"""
RQA2025简化数据管道改进
完善数据加载和特征处理功能
"""

import logging

logger = logging.getLogger(__name__)


def fix_data_loader_imports():
    """修复数据加载器中的导入错误"""
    print("🔧 修复数据加载器导入错误...")

    try:
        # 修复BaseDataLoader中的导入
        base_loader_path = "src/data/loaders/base_data_loader.py"
        with open(base_loader_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 修复导入语句
        if 'from src.infrastructure.logging import UnifiedLogger' in content:
            content = content.replace(
                'from src.infrastructure.logging import UnifiedLogger',
                'from src.infrastructure.logging.unified_logger import UnifiedLogger, get_logger'
            )

            with open(base_loader_path, 'w', encoding='utf-8') as f:
                f.write(content)

            print("   ✅ 修复了BaseDataLoader中的导入")

        return True

    except Exception as e:
        print(f"   ❌ 修复导入错误失败: {e}")
        return False


def create_data_source_manager():
    """创建数据源管理器"""
    print("\n📊 创建数据源管理器...")

    try:
        # 创建数据源目录
        import os
        os.makedirs("src/data/sources", exist_ok=True)

        data_source_manager = '''#!/usr/bin/env python3
"""
RQA2025数据源管理器
统一管理多个数据源的访问
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class DataSource(ABC):
    """数据源抽象基类"""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.is_available = False

    @abstractmethod
    def check_availability(self) -> bool:
        """检查数据源可用性"""
        pass

    @abstractmethod
    def fetch_data(self, symbol: str, start_date: str, end_date: str,
                   interval: str = '1d') -> pd.DataFrame:
        """获取数据"""
        pass

    def get_info(self) -> Dict[str, Any]:
        """获取数据源信息"""
        return {
            'name': self.name,
            'available': self.is_available,
            'config': self.config
        }

class YahooFinanceSource(DataSource):
    """Yahoo Finance数据源"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('Yahoo Finance', config)

    def check_availability(self) -> bool:
        try:
            import requests
            test_url = "https://query1.finance.yahoo.com/v8/finance/chart/AAPL"
            response = requests.head(test_url, timeout=5)
            self.is_available = response.status_code == 200
            return self.is_available
        except:
            self.is_available = False
            return False

    def fetch_data(self, symbol: str, start_date: str, end_date: str,
                   interval: str = '1d') -> pd.DataFrame:
        try:
            import requests

            start_timestamp = int(pd.Timestamp(start_date).timestamp())
            end_timestamp = int(pd.Timestamp(end_date).timestamp())

            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {
                'period1': start_timestamp,
                'period2': end_timestamp,
                'interval': interval,
                'includePrePost': 'false',
                'events': 'div,splits'
            }

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()

            data = response.json()

            if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                result = data['chart']['result'][0]
                timestamps = result['timestamp']
                quotes = result['indicators']['quote'][0]

                df_data = {
                    'timestamp': pd.to_datetime(timestamps, unit='s'),
                    'open': quotes.get('open', []),
                    'high': quotes.get('high', []),
                    'low': quotes.get('low', []),
                    'close': quotes.get('close', []),
                    'volume': quotes.get('volume', [])
                }

                df = pd.DataFrame(df_data)
                df = df.dropna(subset=['close'])
                df = df.sort_values('timestamp').reset_index(drop=True)

                logger.info(f"Yahoo Finance获取数据成功: {symbol}, {len(df)} 条记录")
                return df
            else:
                logger.warning(f"Yahoo Finance API返回数据格式错误: {symbol}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Yahoo Finance获取数据失败 {symbol}: {e}")
            return pd.DataFrame()

class DataSourceManager:
    """数据源管理器"""

    def __init__(self):
        self.sources = {}
        self._init_sources()

    def _init_sources(self):
        """初始化数据源"""
        self.sources['yahoo'] = YahooFinanceSource()

    def get_available_sources(self) -> List[str]:
        """获取可用的数据源"""
        available = []
        for name, source in self.sources.items():
            if source.check_availability():
                available.append(name)
        return available

    def fetch_data_with_fallback(self, symbol: str, start_date: str, end_date: str,
                                interval: str = '1d', preferred_source: str = None) -> pd.DataFrame:
        """使用后备机制获取数据"""
        sources_to_try = []

        if preferred_source and preferred_source in self.sources:
            sources_to_try.append(preferred_source)

        for name in ['yahoo']:
            if name != preferred_source and name in self.sources:
                sources_to_try.append(name)

        for source_name in sources_to_try:
            logger.info(f"尝试从 {source_name} 获取数据: {symbol}")
            source = self.sources[source_name]

            if source.check_availability():
                data = source.fetch_data(symbol, start_date, end_date, interval)
                if not data.empty:
                    logger.info(f"成功从 {source_name} 获取数据: {len(data)} 条记录")
                    return data
                else:
                    logger.warning(f"从 {source_name} 获取数据为空")
            else:
                logger.warning(f"数据源 {source_name} 不可用")

        logger.error(f"所有数据源都无法获取数据: {symbol}")
        return pd.DataFrame()
'''

        # 创建数据源管理器文件
        with open("src/data/sources/data_source_manager.py", 'w', encoding='utf-8') as f:
            f.write(data_source_manager)

        print("   ✅ 数据源管理器创建完成")
        return True

    except Exception as e:
        print(f"   ❌ 创建数据源管理器失败: {e}")
        return False


def create_data_preprocessor():
    """创建数据预处理器"""
    print("\n🧹 创建数据预处理器...")

    try:
        # 创建预处理目录
        import os
        os.makedirs("src/data/preprocessing", exist_ok=True)

        data_preprocessor = '''#!/usr/bin/env python3
"""
RQA2025数据预处理器
提供高级数据预处理和清洗功能
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """数据预处理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.preprocessing_stats = {
            'total_processed': 0,
            'missing_values_handled': 0,
            'outliers_handled': 0,
            'duplicates_removed': 0
        }

    def preprocess(self, data: pd.DataFrame,
                   steps: Optional[List[str]] = None) -> pd.DataFrame:
        """
        执行数据预处理

        Args:
            data: 原始数据
            steps: 预处理步骤列表

        Returns:
            预处理后的数据
        """
        if data is None or data.empty:
            logger.warning("输入数据为空")
            return data

        if steps is None:
            steps = ['validate', 'clean', 'normalize']

        processed_data = data.copy()
        original_shape = processed_data.shape

        for step in steps:
            try:
                if step == 'validate':
                    processed_data = self._validate_data(processed_data)
                elif step == 'clean':
                    processed_data = self._clean_data(processed_data)
                elif step == 'normalize':
                    processed_data = self._normalize_data(processed_data)

                logger.info(f"预处理步骤 {step} 完成")

            except Exception as e:
                logger.error(f"预处理步骤 {step} 失败: {e}")
                continue

        final_shape = processed_data.shape
        logger.info(f"数据预处理完成: {original_shape} -> {final_shape}")
        self.preprocessing_stats['total_processed'] += 1

        return processed_data

    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """验证数据结构"""
        required_columns = ['timestamp', 'open', 'high', 'low', 'close']

        # 检查必需列
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"缺少必需列: {missing_columns}")

        # 检查数据类型
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')

        numeric_columns = ['open', 'high', 'low', 'close']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        if 'volume' in data.columns:
            data['volume'] = pd.to_numeric(data['volume'], errors='coerce')

        # 删除无效行
        data = data.dropna(subset=numeric_columns)

        return data

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """清理数据"""
        original_count = len(data)

        # 1. 处理重复数据
        duplicates_before = len(data)
        data = data.drop_duplicates(subset=['timestamp'])
        duplicates_after = len(data)
        duplicates_removed = duplicates_before - duplicates_after

        if duplicates_removed > 0:
            logger.info(f"移除重复数据: {duplicates_removed} 条")
            self.preprocessing_stats['duplicates_removed'] += duplicates_removed

        # 2. 处理缺失值
        missing_before = data.isnull().sum().sum()
        data = self._handle_missing_values(data)
        missing_after = data.isnull().sum().sum()
        missing_handled = missing_before - missing_after

        if missing_handled > 0:
            logger.info(f"处理缺失值: {missing_handled} 个")
            self.preprocessing_stats['missing_values_handled'] += missing_handled

        # 3. 处理异常值
        data = self._handle_outliers(data)

        # 4. 排序和重置索引
        data = data.sort_values('timestamp').reset_index(drop=True)

        final_count = len(data)
        if original_count != final_count:
            logger.info(f"数据清理: {original_count} -> {final_count}")

        return data

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        # 价格数据的向前填充
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in data.columns:
                data[col] = data[col].fillna(method='ffill')

        # 成交量填充为0
        if 'volume' in data.columns:
            data['volume'] = data['volume'].fillna(0)

        return data

    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理异常值"""
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        outliers_handled = 0

        for col in numeric_columns:
            if col in data.columns:
                # 使用IQR方法检测异常值
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outlier_mask = (data[col] < lower_bound) | (data[col] > upper_bound)

                if outlier_mask.sum() > 0:
                    median_value = data[col].median()
                    data.loc[outlier_mask, col] = median_value
                    outliers_handled += outlier_mask.sum()

        if outliers_handled > 0:
            logger.info(f"处理异常值: {outliers_handled} 个")
            self.preprocessing_stats['outliers_handled'] += outliers_handled

        return data

    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """标准化数据"""
        # 价格数据标准化 (可选)
        if self.config.get('normalize_prices', False):
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in data.columns:
                    if 'close' in data.columns and col != 'close':
                        data[col] = data[col] / data['close']

        if self.config.get('normalize_volume', False) and 'volume' in data.columns:
            data['volume'] = np.log1p(data['volume'])

        return data

class DataQualityMonitor:
    """数据质量监控器"""

    def __init__(self):
        self.quality_metrics = {
            'total_records': 0,
            'missing_values': 0,
            'duplicates': 0,
            'outliers': 0,
            'data_quality_score': 0.0
        }

    def assess_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """评估数据质量"""
        metrics = {
            'total_records': len(data),
            'missing_values': data.isnull().sum().sum(),
            'duplicates': len(data) - len(data.drop_duplicates()),
            'completeness': 1 - (data.isnull().sum().sum() / (len(data) * len(data.columns))),
            'timestamp_monotonic': data['timestamp'].is_monotonic_increasing
        }

        # 计算数据质量评分
        completeness_score = metrics['completeness'] * 100
        monotonic_score = 100 if metrics['timestamp_monotonic'] else 0
        duplicate_penalty = min(metrics['duplicates'] / len(data) * 100, 20) if len(data) > 0 else 0

        quality_score = (completeness_score + monotonic_score - duplicate_penalty)
        metrics['data_quality_score'] = max(0, min(100, quality_score))

        self.quality_metrics.update(metrics)

        return metrics

    def generate_quality_report(self, data: pd.DataFrame) -> str:
        """生成质量报告"""
        metrics = self.assess_quality(data)

        report = f"""
数据质量评估报告
==================
总记录数: {metrics['total_records']}
缺失值数量: {metrics['missing_values']}
重复记录数: {metrics['duplicates']}
数据完整性: {metrics['completeness']:.1%}
时间序列单调性: {'✓' if metrics['timestamp_monotonic'] else '✗'}
数据质量评分: {metrics['data_quality_score']:.1f}/100

质量等级: {'优秀' if metrics['data_quality_score'] >= 90 else
           '良好' if metrics['data_quality_score'] >= 80 else
           '一般' if metrics['data_quality_score'] >= 60 else '较差'}
"""

        return report.strip()
'''

        # 创建数据预处理器文件
        with open("src/data/preprocessing/data_preprocessor.py", 'w', encoding='utf-8') as f:
            f.write(data_preprocessor)

        print("   ✅ 数据预处理器创建完成")
        return True

    except Exception as e:
        print(f"   ❌ 创建数据预处理器失败: {e}")
        return False


def update_base_data_loader():
    """更新BaseDataLoader以使用新的组件"""
    print("\n🔄 更新BaseDataLoader...")

    try:
        base_loader_path = "src/data/loaders/base_data_loader.py"
        with open(base_loader_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 添加预处理器导入
        import_section = '''from ..preprocessing.data_preprocessor import DataPreprocessor, DataQualityMonitor
'''
        if 'from src.infrastructure.cache import BaseCacheManager' in content:
            content = content.replace(
                'from src.infrastructure.cache import BaseCacheManager',
                import_section + 'from src.infrastructure.cache import BaseCacheManager'
            )

            # 添加预处理器实例
            init_section = '''
        # 初始化数据预处理器
        self.preprocessor = DataPreprocessor(self.config.get('preprocessing', {}))
        self.quality_monitor = DataQualityMonitor()
'''
            if 'self.logger.info(f"初始化 {self.__class__.__name__}")' in content:
                content = content.replace(
                    '        self.logger.info(f"初始化 {self.__class__.__name__}")',
                    init_section + '        self.logger.info(f"初始化 {self.__class__.__name__}")'
                )

            # 更新preprocess_data方法
            if 'def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:' in content:
                old_preprocess_method = '''    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理

        Args:
            data: 原始数据

        Returns:
            处理后的数据
        """
        try:
            # 使用新的预处理器
            processed_data = self.preprocessor.preprocess(data)

            # 生成质量报告
            quality_report = self.quality_monitor.generate_quality_report(processed_data)
            self.logger.info(f"数据质量报告:\\n{quality_report}")

            return processed_data

        except Exception as e:
            self.logger.error(f"数据预处理失败: {e}")
            return data'''

                if old_preprocess_method not in content:
                    # 如果方法不存在，添加它
                    content += '''

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理

        Args:
            data: 原始数据

        Returns:
            处理后的数据
        """
        try:
            # 使用新的预处理器
            processed_data = self.preprocessor.preprocess(data)

            # 生成质量报告
            quality_report = self.quality_monitor.generate_quality_report(processed_data)
            self.logger.info(f"数据质量报告:\\n{quality_report}")

            return processed_data

        except Exception as e:
            self.logger.error(f"数据预处理失败: {e}")
            return data'''

                with open(base_loader_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                print("   ✅ BaseDataLoader已更新以使用新的预处理器")

        return True

    except Exception as e:
        print(f"   ❌ 更新BaseDataLoader失败: {e}")
        return False


def test_data_pipeline():
    """测试数据管道"""
    print("\n🧪 测试数据管道...")

    try:
        from src.data.loaders import StockDataLoader

        loader = StockDataLoader({'data_source': 'mock'})
        data = loader.load_data('000001', '2024-01-01', '2024-01-31')

        if not data.empty:
            print(f"   ✅ 数据加载成功: {len(data)} 条记录")

            # 测试预处理器
            if hasattr(loader, 'preprocess_data'):
                processed_data = loader.preprocess_data(data)
                print(f"   ✅ 数据预处理成功: {len(processed_data)} 条记录")
            else:
                print("   ❌ 预处理器不可用")

            return True
        else:
            print("   ❌ 数据加载失败")
            return False

    except Exception as e:
        print(f"   ❌ 数据管道测试失败: {e}")
        return False


def main():
    """主函数"""
    print("🚀 RQA2025数据管道改进")

    results = {
        'fix_imports': False,
        'data_source_manager': False,
        'data_preprocessor': False,
        'update_loader': False,
        'test_pipeline': False
    }

    # 1. 修复导入错误
    results['fix_imports'] = fix_data_loader_imports()

    # 2. 创建数据源管理器
    results['data_source_manager'] = create_data_source_manager()

    # 3. 创建数据预处理器
    results['data_preprocessor'] = create_data_preprocessor()

    # 4. 更新BaseDataLoader
    results['update_loader'] = update_base_data_loader()

    # 5. 测试数据管道
    results['test_pipeline'] = test_data_pipeline()

    # 总结
    successful = sum(results.values())
    total = len(results)

    print(f"\n📊 数据管道改进总结:")
    print(f"   成功: {successful}/{total}")
    print(".1f")

    for task, success in results.items():
        status = "✅" if success else "❌"
        task_name = {
            'fix_imports': '修复导入错误',
            'data_source_manager': '数据源管理器',
            'data_preprocessor': '数据预处理器',
            'update_loader': '更新数据加载器',
            'test_pipeline': '管道测试'
        }.get(task, task)
        print(f"   {status} {task_name}")

    if successful == total:
        print("\n🎉 数据管道改进全部完成！")
    elif successful >= 3:
        print("\n👍 数据管道改进大部分完成")
    else:
        print("\n⚠️ 数据管道改进需要进一步完善")


if __name__ == "__main__":
    main()
