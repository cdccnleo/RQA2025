#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
长期优化测试套件

测试第三阶段长期优化的所有组件：
1. 移动端应用架构
2. 深度学习信号生成器
3. 跨市场数据整合

作者: AI Assistant
创建日期: 2026-02-21
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestMobileAppArchitecture:
    """测试移动端应用架构"""
    
    def test_mobile_project_structure(self):
        """测试移动端项目结构完整性"""
        mobile_dir = project_root / "mobile"
        
        # 检查关键文件和目录
        required_files = [
            "package.json",
            "tsconfig.json",
            "App.tsx",
            "README.md"
        ]
        
        required_dirs = [
            "src",
            "src/screens",
            "src/components",
            "src/services",
            "src/store",
            "src/utils"
        ]
        
        for file in required_files:
            assert (mobile_dir / file).exists(), f"缺少文件: {file}"
        
        for dir in required_dirs:
            assert (mobile_dir / dir).exists(), f"缺少目录: {dir}"
        
        print("✓ 移动端项目结构完整")
    
    def test_mobile_package_json(self):
        """测试移动端依赖配置"""
        import json
        
        package_path = project_root / "mobile" / "package.json"
        with open(package_path, 'r', encoding='utf-8') as f:
            package = json.load(f)
        
        # 检查关键依赖
        dependencies = package.get('dependencies', {})
        key_deps = [
            'react',
            'react-native',
            '@react-navigation/native',
            '@reduxjs/toolkit'
        ]
        
        for dep in key_deps:
            assert dep in dependencies, f"缺少依赖: {dep}"
        
        print("✓ 移动端依赖配置正确")
    
    def test_mobile_app_entry(self):
        """测试移动端入口文件"""
        app_path = project_root / "mobile" / "App.tsx"
        
        with open(app_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查关键组件
        assert 'NavigationContainer' in content, "缺少导航容器"
        assert 'Provider' in content, "缺少Redux Provider"
        assert 'SafeAreaProvider' in content, "缺少安全区域Provider"
        
        print("✓ 移动端入口文件配置正确")


class TestDeepLearningSignalGenerator:
    """测试深度学习信号生成器"""
    
    def test_model_initialization(self):
        """测试模型初始化"""
        from src.ml.deep_learning_signal_generator import DeepLearningSignalGenerator
        
        with patch('src.ml.deep_learning_signal_generator.LSTMModel'), \
             patch('src.ml.deep_learning_signal_generator.TransformerModel'), \
             patch('src.ml.deep_learning_signal_generator.ReinforcementLearningModel'):
            
            generator = DeepLearningSignalGenerator()
            
            assert generator.lstm_model is not None
            assert generator.transformer_model is not None
            assert generator.rl_model is not None
            assert generator.model_weights == {'lstm': 0.4, 'transformer': 0.3, 'rl': 0.3}
        
        print("✓ 深度学习模型初始化正确")
    
    def test_ensemble_logic(self):
        """测试模型集成逻辑"""
        from src.ml.deep_learning_signal_generator import DeepLearningSignalGenerator
        
        with patch('src.ml.deep_learning_signal_generator.LSTMModel'), \
             patch('src.ml.deep_learning_signal_generator.TransformerModel'), \
             patch('src.ml.deep_learning_signal_generator.ReinforcementLearningModel'):
            
            generator = DeepLearningSignalGenerator()
            
            # 模拟预测结果
            lstm_pred = {'signal': 'buy', 'confidence': 0.85}
            transformer_pred = {'signal': 'buy', 'confidence': 0.75}
            rl_pred = {'signal': 'hold', 'confidence': 0.60}
            
            result = generator._ensemble(lstm_pred, transformer_pred, rl_pred)
            
            assert 'signal' in result
            assert 'confidence' in result
            assert result['signal'] in ['buy', 'sell', 'hold']
            assert 0 <= result['confidence'] <= 1
        
        print("✓ 模型集成逻辑正确")
    
    def test_signal_generation(self):
        """测试信号生成"""
        from src.ml.deep_learning_signal_generator import DeepLearningSignalGenerator
        
        with patch('src.ml.deep_learning_signal_generator.LSTMModel') as mock_lstm, \
             patch('src.ml.deep_learning_signal_generator.TransformerModel') as mock_transformer, \
             patch('src.ml.deep_learning_signal_generator.ReinforcementLearningModel') as mock_rl:
            
            # 设置模拟返回值
            mock_lstm.return_value.predict.return_value = {'signal': 'buy', 'confidence': 0.8}
            mock_transformer.return_value.predict.return_value = {'signal': 'buy', 'confidence': 0.75}
            mock_rl.return_value.predict.return_value = {'signal': 'hold', 'confidence': 0.6}
            
            generator = DeepLearningSignalGenerator()
            
            # 创建模拟数据
            market_data = {
                'symbol': '000001.SZ',
                'prices': np.random.randn(100),
                'volumes': np.random.randint(1000, 10000, 100)
            }
            
            result = generator.generate_signal(market_data)
            
            assert 'signal' in result
            assert 'confidence' in result
            assert 'model_predictions' in result
            assert result['signal'] in ['buy', 'sell', 'hold']
        
        print("✓ 信号生成正确")


class TestCrossMarketDataIntegration:
    """测试跨市场数据整合"""
    
    def test_market_types(self):
        """测试市场类型定义"""
        from src.data.adapters.cross_market.cross_market_data_manager import MarketType
        
        assert MarketType.A_SHARE.value == 'CN'
        assert MarketType.HK_STOCK.value == 'HK'
        assert MarketType.US_STOCK.value == 'US'
        
        print("✓ 市场类型定义正确")
    
    def test_data_manager_initialization(self):
        """测试数据管理器初始化"""
        from src.data.adapters.cross_market.cross_market_data_manager import CrossMarketDataManager
        
        with patch('src.data.adapters.cross_market.cross_market_data_manager.HKStockDataSource'), \
             patch('src.data.adapters.cross_market.cross_market_data_manager.USStockDataSource'):
            
            manager = CrossMarketDataManager()
            
            assert manager is not None
            assert len(manager.data_sources) >= 2
        
        print("✓ 跨市场数据管理器初始化正确")
    
    def test_global_market_overview(self):
        """测试全球市场概览"""
        from src.data.adapters.cross_market.cross_market_data_manager import CrossMarketDataManager
        
        with patch('src.data.adapters.cross_market.cross_market_data_manager.HKStockDataSource'), \
             patch('src.data.adapters.cross_market.cross_market_data_manager.USStockDataSource'):
            
            manager = CrossMarketDataManager()
            
            # 模拟数据源
            manager.data_sources = {
                Mock(): Mock(),
                Mock(): Mock()
            }
            
            overview = asyncio.run(manager.get_global_market_overview())
            
            assert 'markets' in overview
            assert 'timestamp' in overview
        
        print("✓ 全球市场概览功能正确")
    
    def test_hk_stock_data_source(self):
        """测试港股数据源"""
        from src.data.adapters.cross_market.cross_market_data_manager import HKStockDataSource
        
        source = HKStockDataSource()
        
        assert hasattr(source, 'market_type')
        assert source.market_type == 'hk_stock'
        
        print("✓ 港股数据源配置正确")
    
    def test_us_stock_data_source(self):
        """测试美股数据源"""
        from src.data.adapters.cross_market.cross_market_data_manager import USStockDataSource
        
        source = USStockDataSource()
        
        assert hasattr(source, 'market_type')
        assert source.market_type == 'us_stock'
        
        print("✓ 美股数据源配置正确")


class TestLongTermOptimizationIntegration:
    """测试长期优化集成"""
    
    def test_all_components_exist(self):
        """测试所有组件存在"""
        components = [
            "mobile/App.tsx",
            "mobile/package.json",
            "src/ml/deep_learning_signal_generator.py",
            "src/data/adapters/cross_market/cross_market_data_manager.py"
        ]
        
        for component in components:
            path = project_root / component
            assert path.exists(), f"缺少组件: {component}"
        
        print("✓ 所有长期优化组件存在")
    
    def test_configuration_files(self):
        """测试配置文件"""
        # 检查移动端配置
        mobile_config = project_root / "mobile" / "package.json"
        assert mobile_config.exists()
        
        # 检查跨市场数据配置
        cross_market_dir = project_root / "src" / "data" / "adapters" / "cross_market"
        assert cross_market_dir.exists()
        
        print("✓ 配置文件正确")


def run_all_tests():
    """运行所有长期优化测试"""
    print("\n" + "="*60)
    print("开始长期优化测试")
    print("="*60 + "\n")
    
    test_classes = [
        TestMobileAppArchitecture(),
        TestDeepLearningSignalGenerator(),
        TestCrossMarketDataIntegration(),
        TestLongTermOptimizationIntegration()
    ]
    
    passed = 0
    failed = 0
    
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n{class_name}")
        print("-" * 40)
        
        for method_name in dir(test_class):
            if method_name.startswith('test_'):
                try:
                    method = getattr(test_class, method_name)
                    method()
                    passed += 1
                except Exception as e:
                    print(f"✗ {method_name}: {str(e)}")
                    failed += 1
    
    print("\n" + "="*60)
    print(f"测试结果: 通过 {passed}, 失败 {failed}")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
