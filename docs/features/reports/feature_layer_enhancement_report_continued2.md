# RQA2025 特征层功能增强分析报告（续2）

## 2. 功能分析（续）

### 2.3 特征存储和复用（续）

#### 2.3.1 特征存储（续）

**实现建议**（续）：

```python
    def store_feature(
        self,
        name: str,
        feature_data: pd.DataFrame,
        source_data: pd.DataFrame,
        params: Dict,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        存储特征
        
        Args:
            name: 特征名称
            feature_data: 特征数据
            source_data: 源数据
            params: 特征参数
            description: 特征描述
            tags: 特征标签
            
        Returns:
            str: 特征ID
        """
        # 生成数据哈希
        data_hash = self._generate_data_hash(source_data)
        
        # 生成特征ID
        feature_id = self._generate_feature_id(name, params, data_hash)
        
        # 创建特征文件路径
        feature_file = os.path.join(self.store_dir, f"{feature_id}.pkl")
        
        # 存储特征数据
        try:
            with open(feature_file, 'wb') as f:
                pickle.dump(feature_data, f)
        except Exception as e:
            logger.error(f"Failed to store feature data: {e}")
            return None
        
        # 更新元数据
        self.metadata['features'][feature_id] = {
            'name': name,
            'params': params,
            'data_hash': data_hash,
            'description': description or '',
            'tags': tags or [],
            'created_at': datetime.now().isoformat(),
            'file_path': feature_file,
            'shape': feature_data.shape,
            'columns': list(feature_data.columns)
        }
        
        # 保存元数据
        self._save_metadata()
        
        return feature_id
    
    def load_feature(self, feature_id: str) -> Optional[pd.DataFrame]:
        """
        加载特征
        
        Args:
            feature_id: 特征ID
            
        Returns:
            Optional[pd.DataFrame]: 特征数据
        """
        # 检查特征是否存在
        if feature_id not in self.metadata['features']:
            logger.warning(f"Feature {feature_id} not found")
            return None
        
        # 获取特征文件路径
        feature_file = self.metadata['features'][feature_id]['file_path']
        
        # 加载特征数据
        try:
            with open(feature_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load feature data: {e}")
            return None
    
    def find_feature(
        self,
        name: str,
        params: Dict,
        source_data: pd.DataFrame
    ) -> Optional[str]:
        """
        查找特征
        
        Args:
            name: 特征名称
            params: 特征参数
            source_data: 源数据
            
        Returns:
            Optional[str]: 特征ID
        """
        # 生成数据哈希
        data_hash = self._generate_data_hash(source_data)
        
        # 生成特征ID
        feature_id = self._generate_feature_id(name, params, data_hash)
        
        # 检查特征是否存在
        if feature_id in self.metadata['features']:
            return feature_id
        else:
            return None
    
    def list_features(
        self,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        列出特征
        
        Args:
            name: 特征名称
            tags: 特征标签
            
        Returns:
            List[Dict]: 特征列表
        """
        features = []
        
        for feature_id, feature_info in self.metadata['features'].items():
            # 按名称过滤
            if name and feature_info['name'] != name:
                continue
            
            # 按标签过滤
            if tags and not all(tag in feature_info['tags'] for tag in tags):
                continue
            
            # 添加特征ID
            feature_info = feature_info.copy()
            feature_info['id'] = feature_id
            
            features.append(feature_info)
        
        return features
    
    def delete_feature(self, feature_id: str) -> bool:
        """
        删除特征
        
        Args:
            feature_id: 特征ID
            
        Returns:
            bool: 是否成功
        """
        # 检查特征是否存在
        if feature_id not in self.metadata['features']:
            logger.warning(f"Feature {feature_id} not found")
            return False
        
        # 获取特征文件路径
        feature_file = self.metadata['features'][feature_id]['file_path']
        
        # 删除特征文件
        try:
            if os.path.exists(feature_file):
                os.remove(feature_file)
        except Exception as e:
            logger.error(f"Failed to delete feature file: {e}")
            return False
        
        # 更新元数据
        del self.metadata['features'][feature_id]
        
        # 保存元数据
        self._save_metadata()
        
        return True
```

#### 2.3.2 特征复用

**现状分析**：
缺乏特征复用机制，导致相同特征被重复计算。

**实现建议**：
实现一个 `FeatureManager` 类，提供特征计算和复用功能：

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Callable
import logging
from datetime import datetime
import time

logger = logging.getLogger(__name__)

class FeatureManager:
    """特征管理器"""
    
    def __init__(
        self,
        feature_store: 'FeatureStore',
        parallel_processor: Optional['ParallelFeatureProcessor'] = None
    ):
        """
        初始化特征管理器
        
        Args:
            feature_store: 特征存储
            parallel_processor: 并行处理器
        """
        self.feature_store = feature_store
        self.parallel_processor = parallel_processor
        
        # 特征处理器注册表
        self.processors: Dict[str, Callable] = {}
        
        # 特征计算统计
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'computation_time': 0
        }
    
    def register_processor(
        self,
        name: str,
        processor_func: Callable
    ) -> None:
        """
        注册特征处理器
        
        Args:
            name: 处理器名称
            processor_func: 处理器函数
        """
        self.processors[name] = processor_func
    
    def compute_feature(
        self,
        name: str,
        data: pd.DataFrame,
        params: Dict,
        force_recompute: bool = False,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        计算特征
        
        Args:
            name: 特征名称
            data: 输入数据
            params: 特征参数
            force_recompute: 是否强制重新计算
            description: 特征描述
            tags: 特征标签
            
        Returns:
            pd.DataFrame: 特征数据
        """
        # 检查处理器是否存在
        if name not in self.processors:
            raise ValueError(f"Feature processor {name} not registered")
        
        # 如果不强制重新计算，则尝试从存储中加载
        if not force_recompute:
            feature_id = self.feature_store.find_feature(name, params, data)
            if feature_id:
                feature_data = self.feature_store.load_feature(feature_id)
                if feature_data is not None:
                    self.stats['cache_hits'] += 1
                    logger.info(f"Feature {name} loaded from store")
                    return feature_data
        
        self.stats['cache_misses'] += 1
        
        # 获取处理器函数
        processor_func = self.processors[name]
        
        # 计算特征
        start_time = time.time()
        
        if self.parallel_processor and hasattr(processor_func, 'parallel'):
            # 使用并行处理器
            feature_data = self.parallel_processor.process(
                data,
                processor_func,
                **params
            )
        else:
            # 使用串行处理
            feature_data = processor_func(data, **params)
        
        end_time = time.time()
        computation_time = end_time - start_time
        self.stats['computation_time'] += computation_time
        
        logger.info(f"Feature {name} computed in {computation_time:.2f}s")
        
        # 存储特征
        self.feature_store.store_feature(
            name,
            feature_data,
            data,
            params,
            description,
            tags
        )
        
        return feature_data
    
    def compute_multiple_features(
        self,
        feature_configs: List[Dict],
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        计算多个特征
        
        Args:
            feature_configs: 特征配置列表
            data: 输入数据
            
        Returns:
            pd.DataFrame: 特征数据
        """
        result = data.copy()
        
        for config in feature_configs:
            name = config['name']
            params = config.get('params', {})
            force_recompute = config.get('force_recompute', False)
            description = config.get('description')
            tags = config.get('tags')
            
            try:
                feature_data = self.compute_feature(
                    name,
                    data,
                    params,
                    force_recompute,
                    description,
                    tags
                )
                
                # 合并特征数据
                if isinstance(feature_data, pd.DataFrame):
                    # 如果是DataFrame，则合并所有列
                    for col in feature_data.columns:
                        if col not in result.columns:
                            result[col] = feature_data[col]
                elif isinstance(feature_data, pd.Series):
                    # 如果是Series，则添加为单列
                    col_name = config.get('output_name', name)
                    result[col_name] = feature_data
            
            except Exception as e:
                logger.error(f"Error computing feature {name}: {e}")
        
        return result
    
    def get_stats(self) -> Dict:
        """
        获取统计信息
        
        Returns:
            Dict: 统计信息
        """
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        cache_hit_rate = self.stats['cache_hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'cache_hit_rate': cache_hit_rate,
            'computation_time': self.stats['computation_time'],
            'total_requests': total_requests
        }
```

### 2.4 特征工程自动化

#### 2.4.1 自动特征生成

**现状分析**：
特征工程流程自动化程度不高，需要手动配置和调整特征。

**实现建议**：
实现一个 `AutoFeatureGenerator` 类，提供自动特征生成功能：

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

logger = logging.getLogger(__name__)

class AutoFeatureGenerator:
    """自动特征生成器"""
    
    def __init__(
        self,
        feature_manager: 'FeatureManager',
        random_state: int = 42
    ):
        """
        初始化自动特征生成器
        
        Args:
            feature_manager: 特征管理器
            random_state: 随机种子
        """
        self.feature_manager = feature_manager
        self.random_state = random_state
    
    def generate_time_features(
        self,
        data: pd.DataFrame,
        date_column: str
    ) -> pd.DataFrame:
        """
        生成时间特征
        
        Args:
            data: 输入数据
            date_column: 日期列名
            
        Returns:
            pd.DataFrame: 特征数据
        """
        # 确保日期列是datetime类型
        df = data.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column])
        
        # 提取日期特征
        date_df = pd.DataFrame(index=df.index)
        date_df['year'] = df[date_column].dt.year
        date_df['month'] = df[date_column].dt.month
        date_df['day'] = df[date_column].dt.day
        date_df['dayofweek'] = df[date_column].dt.dayofweek
        date_df['quarter'] = df[date_column].dt.quarter
        date_df['is_month_start'] = df[date_column].dt.is_month_start.astype(int)
        date_df['is_month_end'] = df[date_column].dt.is_month_end.astype(int)
        date_df['is_quarter_start'] = df[date_column].dt.is_quarter_start.astype(int)
        date_df['is_quarter_end'] = df[date_column].dt.is_quarter_end.astype(int)
        date_df['is_year_start'] = df[date_column].dt.is_year_start.astype(int)
        date_df['is_year_end'] = df[date_column].dt.is_year_end.astype(int)
        
        # 添加季节性特征
        date_df['sin_month'] = np.sin(2 * np.pi * df[date_column].dt.month / 12)
        date_df['cos_month'