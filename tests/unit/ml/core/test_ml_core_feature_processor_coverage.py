#!/usr/bin/env python3
"""
测试MLCore的特征处理器相关方法的覆盖率
补充fit_feature_processor, transform_features, get_feature_importance等方法的测试
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.ml.core.ml_core import MLCore
import sys
from pathlib import Path

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


@pytest.fixture
def ml_core():
    """创建MLCore实例"""
    return MLCore()


@pytest.fixture
def sample_data():
    """创建样本数据"""
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100)
    })


class TestMLCoreFeatureProcessor:
    """测试特征处理器相关方法"""
    
    def test_create_feature_processor_standard(self, ml_core):
        """测试创建标准特征处理器"""
        processor_id = ml_core.create_feature_processor('standard')
        
        assert processor_id is not None
        assert processor_id.startswith('processor_standard_')
        assert processor_id in ml_core.feature_processors
        
        processor_info = ml_core.feature_processors[processor_id]
        assert processor_info['type'] == 'standard'
        assert processor_info['processor'] is not None
    
    def test_create_feature_processor_robust(self, ml_core):
        """测试创建robust特征处理器"""
        processor_id = ml_core.create_feature_processor('robust')
        
        assert processor_id is not None
        assert processor_id.startswith('processor_robust_')
        assert processor_id in ml_core.feature_processors
    
    def test_create_feature_processor_minmax(self, ml_core):
        """测试创建minmax特征处理器"""
        processor_id = ml_core.create_feature_processor('minmax')
        
        assert processor_id is not None
        assert processor_id.startswith('processor_minmax_')
        assert processor_id in ml_core.feature_processors
    
    def test_create_feature_processor_invalid_type(self, ml_core):
        """测试创建无效类型的特征处理器"""
        with pytest.raises(ValueError, match="不支持的处理器类型"):
            ml_core.create_feature_processor('invalid_type')
    
    def test_create_feature_processor_with_config(self, ml_core):
        """测试使用配置创建特征处理器"""
        config = {'with_mean': False, 'with_std': True}
        processor_id = ml_core.create_feature_processor('standard', config=config)
        
        assert processor_id in ml_core.feature_processors
        assert ml_core.feature_processors[processor_id]['config'] == config
    
    def test_create_feature_processor_import_error(self, ml_core, monkeypatch):
        """测试sklearn导入失败时的错误处理"""
        # Mock sklearn.preprocessing导入失败
        original_import = __import__
        def mock_import(name, *args, **kwargs):
            if name == 'sklearn.preprocessing':
                raise ImportError("sklearn not available")
            return original_import(name, *args, **kwargs)
        
        monkeypatch.setattr('builtins.__import__', mock_import)
        
        # 重新加载ml_core模块以触发导入错误
        import importlib
        import src.ml.core.ml_core
        importlib.reload(src.ml.core.ml_core)
        
        # 创建一个新的MLCore实例，应该触发ImportError
        with pytest.raises(ImportError):
            ml_core.create_feature_processor('standard')
    
    def test_fit_feature_processor_success(self, ml_core, sample_data):
        """测试拟合特征处理器成功"""
        processor_id = ml_core.create_feature_processor('standard')
        ml_core.fit_feature_processor(processor_id, sample_data)
        
        # 验证处理器已拟合（可以转换数据）
        transformed = ml_core.transform_features(processor_id, sample_data)
        assert transformed.shape == sample_data.shape
    
    def test_fit_feature_processor_with_numpy_array(self, ml_core):
        """测试使用numpy数组拟合特征处理器"""
        X = np.random.randn(100, 3)
        processor_id = ml_core.create_feature_processor('standard')
        ml_core.fit_feature_processor(processor_id, X)
        
        # 验证可以转换
        transformed = ml_core.transform_features(processor_id, X)
        assert transformed.shape == X.shape
    
    def test_fit_feature_processor_not_found(self, ml_core, sample_data):
        """测试拟合不存在的特征处理器"""
        with pytest.raises(ValueError, match="特征处理器.*不存在"):
            ml_core.fit_feature_processor('nonexistent_processor', sample_data)
    
    def test_fit_feature_processor_exception(self, ml_core, sample_data, monkeypatch):
        """测试拟合时发生异常"""
        processor_id = ml_core.create_feature_processor('standard')
        
        # Mock fit方法抛出异常
        processor = ml_core.feature_processors[processor_id]['processor']
        original_fit = processor.fit
        processor.fit = MagicMock(side_effect=Exception("拟合失败"))
        
        with pytest.raises(Exception, match="拟合失败"):
            ml_core.fit_feature_processor(processor_id, sample_data)
        
        # 恢复原始方法
        processor.fit = original_fit
    
    def test_transform_features_success(self, ml_core, sample_data):
        """测试特征转换成功"""
        processor_id = ml_core.create_feature_processor('standard')
        ml_core.fit_feature_processor(processor_id, sample_data)
        
        transformed = ml_core.transform_features(processor_id, sample_data)
        
        assert isinstance(transformed, np.ndarray)
        assert transformed.shape == sample_data.shape
    
    def test_transform_features_with_numpy_array(self, ml_core):
        """测试使用numpy数组转换特征"""
        X = np.random.randn(100, 3)
        processor_id = ml_core.create_feature_processor('standard')
        ml_core.fit_feature_processor(processor_id, X)
        
        transformed = ml_core.transform_features(processor_id, X)
        
        assert isinstance(transformed, np.ndarray)
        assert transformed.shape == X.shape
    
    def test_transform_features_not_found(self, ml_core, sample_data):
        """测试转换不存在的特征处理器"""
        with pytest.raises(ValueError, match="特征处理器.*不存在"):
            ml_core.transform_features('nonexistent_processor', sample_data)
    
    def test_transform_features_without_fit(self, ml_core, sample_data):
        """测试未拟合就转换特征（应该失败）"""
        processor_id = ml_core.create_feature_processor('standard')
        
        # 尝试转换未拟合的处理器（sklearn会抛出异常）
        with pytest.raises(Exception):
            ml_core.transform_features(processor_id, sample_data)
    
    def test_transform_features_exception(self, ml_core, sample_data, monkeypatch):
        """测试转换时发生异常"""
        processor_id = ml_core.create_feature_processor('standard')
        ml_core.fit_feature_processor(processor_id, sample_data)
        
        # Mock transform方法抛出异常
        processor = ml_core.feature_processors[processor_id]['processor']
        original_transform = processor.transform
        processor.transform = MagicMock(side_effect=Exception("转换失败"))
        
        with pytest.raises(Exception, match="转换失败"):
            ml_core.transform_features(processor_id, sample_data)
        
        # 恢复原始方法
        processor.transform = original_transform


class TestMLCoreFeatureImportance:
    """测试特征重要性相关方法"""
    
    def test_get_feature_importance_with_feature_importances(self, ml_core, sample_data):
        """测试获取具有feature_importances_的模型的特征重要性"""
        from sklearn.ensemble import RandomForestRegressor
        
        # 训练一个随机森林模型
        y = pd.Series(np.random.randn(100))
        model_id = ml_core.train_model(sample_data, y, model_type='rf')
        
        # 获取特征重要性
        importance = ml_core.get_feature_importance(model_id)
        
        assert importance is not None
        assert isinstance(importance, dict)
        assert len(importance) == 3  # 3个特征
        assert all(isinstance(val, (int, float)) for val in importance.values())
    
    def test_get_feature_importance_with_coef(self, ml_core, sample_data):
        """测试获取具有coef_的模型的特征重要性"""
        from sklearn.linear_model import LinearRegression
        
        # 训练一个线性回归模型
        y = pd.Series(np.random.randn(100))
        model_id = ml_core.train_model(sample_data, y, model_type='linear')
        
        # 获取特征重要性
        importance = ml_core.get_feature_importance(model_id)
        
        assert importance is not None
        assert isinstance(importance, dict)
        assert len(importance) == 3  # 3个特征
        assert all(isinstance(val, (int, float)) for val in importance.values())
    
    def test_get_feature_importance_with_feature_names(self, ml_core, sample_data):
        """测试使用特征名称获取特征重要性"""
        from sklearn.ensemble import RandomForestRegressor
        
        # 训练模型
        y = pd.Series(np.random.randn(100))
        model_id = ml_core.train_model(
            sample_data, y,
            model_type='rf',
            feature_names=list(sample_data.columns)
        )
        
        # 获取特征重要性
        importance = ml_core.get_feature_importance(model_id)
        
        assert importance is not None
        # 验证特征名称正确
        assert 'feature1' in importance or 'feature_0' in importance
    
    def test_get_feature_importance_model_not_found(self, ml_core):
        """测试获取不存在的模型的特征重要性"""
        importance = ml_core.get_feature_importance('nonexistent_model')
        assert importance is None
    
    def test_get_feature_importance_no_importance_attr(self, ml_core, sample_data):
        """测试获取没有重要性属性的模型的特征重要性"""
        from sklearn.cluster import KMeans
        
        # 创建一个没有feature_importances_或coef_的模型
        model = KMeans(n_clusters=3)
        model.fit(sample_data.values)
        
        # 手动添加到models字典
        model_id = "test_kmeans_model"
        ml_core.models[model_id] = {
            'model': model,
            'model_type': 'kmeans',
            'feature_names': list(sample_data.columns),
            'created_at': pd.Timestamp.now()
        }
        
        # 获取特征重要性应该返回None
        importance = ml_core.get_feature_importance(model_id)
        assert importance is None
    
    def test_get_feature_importance_exception(self, ml_core):
        """测试获取特征重要性时发生异常"""
        # 创建一个会抛出异常的模型
        class ExceptionModel:
            @property
            def feature_importances_(self):
                raise Exception("访问属性失败")
        
        model_id = "test_model"
        ml_core.models[model_id] = {
            'model': ExceptionModel(),
            'model_type': 'test',
            'feature_names': []
        }
        
        # 获取特征重要性时应该捕获异常并返回None
        importance = ml_core.get_feature_importance(model_id)
        assert importance is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

