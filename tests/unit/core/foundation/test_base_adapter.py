#!/usr/bin/env python3
"""
BaseAdapter单元测试

测试适配器基类的所有功能
"""

import pytest
from typing import Dict, Any

# 尝试导入所需模块
try:
    from src.core.foundation.base_adapter import BaseAdapter
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestAdapter(BaseAdapter[Dict[str, Any], Dict[str, Any]]):
    """测试用适配器类"""
    
    def __init__(self, name: str = "test", config: Dict[str, Any] = None, enable_cache: bool = False):
        super().__init__(name, config, enable_cache)
        self.adapt_count = 0
    
    def _do_adapt(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.adapt_count += 1
        return {
            'adapted': True,
            'original': data,
            'count': self.adapt_count
        }


class SimpleAdapter(BaseAdapter[str, str]):
    """简单字符串适配器"""
    
    def _do_adapt(self, data: str) -> str:
        return data.upper()


class TestBaseAdapter:
    """BaseAdapter测试套件"""
    
    def test_adapter_creation(self):
        """测试适配器创建"""
        adapter = TestAdapter(name="test_adapter")
        
        assert adapter.name == "test_adapter"
        assert adapter._status == AdapterStatus.READY
        assert adapter.enable_cache is False
        assert adapter._success_count == 0
        assert adapter._error_count == 0
    
    def test_adapter_basic_adapt(self):
        """测试基本适配功能"""
        adapter = TestAdapter()
        input_data = {'key': 'value'}
        
        result = adapter.adapt(input_data)
        
        assert result['adapted'] is True
        assert result['original'] == input_data
        assert adapter.adapt_count == 1
        assert adapter._success_count == 1
    
    def test_adapter_input_validation(self):
        """测试输入验证"""
        adapter = TestAdapter()
        
        # None输入应该失败
        with pytest.raises(ValueError) as exc_info:
            adapter.adapt(None)
        
        assert "无效的输入数据" in str(exc_info.value)
        assert adapter._error_count == 1
    
    def test_adapter_cache_functionality(self):
        """测试缓存功能"""
        adapter = TestAdapter(enable_cache=True)
        input_data = {'key': 'value'}
        
        # 第一次调用
        result1 = adapter.adapt(input_data)
        assert adapter.adapt_count == 1
        
        # 第二次调用（应该从缓存获取）
        result2 = adapter.adapt(input_data)
        assert result2 == result1
        assert adapter.adapt_count == 1  # 没有增加，说明用了缓存
    
    def test_adapter_cache_disabled(self):
        """测试缓存禁用"""
        adapter = TestAdapter(enable_cache=False)
        input_data = {'key': 'value'}
        
        # 多次调用
        result1 = adapter.adapt(input_data)
        result2 = adapter.adapt(input_data)
        
        assert adapter.adapt_count == 2  # 每次都执行
    
    def test_adapter_clear_cache(self):
        """测试清空缓存"""
        adapter = TestAdapter(enable_cache=True)
        adapter.adapt({'key': 'value'})
        
        assert len(adapter._cache) > 0
        
        adapter.clear_cache()
        assert len(adapter._cache) == 0
    
    def test_adapter_statistics(self):
        """测试统计功能"""
        adapter = TestAdapter()
        
        # 成功的调用
        adapter.adapt({'key': 'value'})
        adapter.adapt({'key2': 'value2'})
        
        stats = adapter.get_stats()
        
        assert stats['success_count'] == 2
        assert stats['error_count'] == 0
        assert stats['total_count'] == 2
        assert '100.00%' in stats['success_rate']
    
    def test_adapter_error_handling(self):
        """测试错误处理"""
        class ErrorAdapter(BaseAdapter[Dict, Dict]):
            def _do_adapt(self, data: Dict) -> Dict:
                raise ValueError("Adaptation error")
        
        adapter = ErrorAdapter("error_adapter")
        
        with pytest.raises(RuntimeError):
            adapter.adapt({'key': 'value'})
        
        assert adapter._error_count == 1
        assert adapter._last_error is not None
        assert adapter._status == AdapterStatus.ERROR
    
    def test_adapter_custom_error_recovery(self):
        """测试自定义错误恢复"""
        class RecoveryAdapter(BaseAdapter[Dict, Dict]):
            def _do_adapt(self, data: Dict) -> Dict:
                if data.get('error'):
                    raise ValueError("Error")
                return {'success': True}
            
            def _handle_error(self, data: Dict, error: Exception) -> Dict:
                # 自定义恢复逻辑
                return {'recovered': True, 'error': str(error)}
        
        adapter = RecoveryAdapter("recovery_adapter")
        result = adapter.adapt({'error': True})
        
        assert result['recovered'] is True
        assert 'Error' in result['error']
    
    def test_adapter_preprocess(self):
        """测试预处理"""
        class PreprocessAdapter(BaseAdapter[Dict, Dict]):
            def _do_adapt(self, data: Dict) -> Dict:
                return data
            
            def _preprocess(self, data: Dict) -> Dict:
                data['preprocessed'] = True
                return data
        
        adapter = PreprocessAdapter("preprocess")
        result = adapter.adapt({'original': True})
        
        assert result['preprocessed'] is True
        assert result['original'] is True
    
    def test_adapter_postprocess(self):
        """测试后处理"""
        class PostprocessAdapter(BaseAdapter[Dict, Dict]):
            def _do_adapt(self, data: Dict) -> Dict:
                return {'adapted': True}
            
            def _postprocess(self, data: Dict) -> Dict:
                data['postprocessed'] = True
                return data
        
        adapter = PostprocessAdapter("postprocess")
        result = adapter.adapt({'original': True})
        
        assert result['adapted'] is True
        assert result['postprocessed'] is True
    
    def test_adapter_health_check(self):
        """测试健康检查"""
        adapter = TestAdapter()
        
        # 新创建的适配器应该是健康的
        assert adapter.is_healthy() is True
        
        # 成功调用后仍然健康
        adapter.adapt({'key': 'value'})
        assert adapter.is_healthy() is True
        
        # 禁用后不健康
        adapter.disable()
        assert adapter.is_healthy() is False
    
    def test_adapter_enable_disable(self):
        """测试启用/禁用"""
        adapter = TestAdapter()
        
        assert adapter._status == AdapterStatus.READY
        
        adapter.disable()
        assert adapter._status == AdapterStatus.DISABLED
        
        adapter.enable()
        assert adapter._status == AdapterStatus.READY
    
    def test_adapter_reset_stats(self):
        """测试重置统计"""
        adapter = TestAdapter()
        adapter.adapt({'key': 'value'})
        
        assert adapter._success_count == 1
        
        adapter.reset_stats()
        assert adapter._success_count == 0
        assert adapter._error_count == 0
        assert adapter._last_error is None
    
    def test_adapter_decorator(self):
        """测试适配器装饰器"""
        @adapter("decorated_adapter", enable_cache=True)
        class DecoratedAdapter(BaseAdapter[str, str]):
            def _do_adapt(self, data: str) -> str:
                return data.upper()
        
        # 注意：装饰器修改了__init__
        decorated = DecoratedAdapter()
        assert decorated.enable_cache is True


class TestAdapterChain:
    """AdapterChain测试套件"""
    
    def test_chain_creation(self):
        """测试链创建"""
        chain = AdapterChain("test_chain")
        assert chain.name == "test_chain"
        assert len(chain._adapters) == 0
    
    def test_chain_add_adapter(self):
        """测试添加适配器"""
        chain = AdapterChain("test_chain")
        adapter1 = SimpleAdapter("adapter1")
        adapter2 = SimpleAdapter("adapter2")
        
        chain.add_adapter(adapter1).add_adapter(adapter2)
        
        assert len(chain._adapters) == 2
    
    def test_chain_execution(self):
        """测试链执行"""
        class UpperAdapter(BaseAdapter[str, str]):
            def _do_adapt(self, data: str) -> str:
                return data.upper()
        
        class AddPrefixAdapter(BaseAdapter[str, str]):
            def _do_adapt(self, data: str) -> str:
                return f"PREFIX_{data}"
        
        chain = AdapterChain("test_chain")
        chain.add_adapter(UpperAdapter("upper"))
        chain.add_adapter(AddPrefixAdapter("prefix"))
        
        result = chain.execute("test")
        
        assert result == "PREFIX_TEST"
    
    def test_chain_skip_unhealthy(self):
        """测试跳过不健康的适配器"""
        class UpperAdapter(BaseAdapter[str, str]):
            def _do_adapt(self, data: str) -> str:
                return data.upper()
        
        adapter1 = UpperAdapter("adapter1")
        adapter2 = UpperAdapter("adapter2")
        
        # 禁用第一个适配器
        adapter1.disable()
        
        chain = AdapterChain("test_chain")
        chain.add_adapter(adapter1)
        chain.add_adapter(adapter2)
        
        # 应该跳过adapter1，只执行adapter2
        result = chain.execute("test")
        assert result == "TEST"
    
    def test_chain_stats(self):
        """测试链统计"""
        adapter1 = SimpleAdapter("adapter1")
        adapter2 = SimpleAdapter("adapter2")
        
        chain = AdapterChain("test_chain")
        chain.add_adapter(adapter1).add_adapter(adapter2)
        
        stats = chain.get_chain_stats()
        
        assert stats['chain_name'] == "test_chain"
        assert stats['adapter_count'] == 2
        assert len(stats['adapters']) == 2


class TestAdapterIntegration:
    """适配器集成测试"""
    
    def test_end_to_end_adaptation(self):
        """测试端到端适配"""
        class DataNormalizer(BaseAdapter[Dict, Dict]):
            def _do_adapt(self, data: Dict) -> Dict:
                return {k.lower(): v for k, v in data.items()}
        
        class DataValidator(BaseAdapter[Dict, Dict]):
            def validate_input(self, data: Dict) -> bool:
                return 'required_field' in data
            
            def _do_adapt(self, data: Dict) -> Dict:
                data['validated'] = True
                return data
        
        class DataEnricher(BaseAdapter[Dict, Dict]):
            def _do_adapt(self, data: Dict) -> Dict:
                data['enriched'] = True
                data['timestamp'] = '2025-11-03'
                return data
        
        # 创建处理链
        chain = AdapterChain("data_processing")
        chain.add_adapter(DataNormalizer("normalizer"))
        chain.add_adapter(DataValidator("validator"))
        chain.add_adapter(DataEnricher("enricher"))
        
        # 执行
        input_data = {'REQUIRED_FIELD': 'value', 'OTHER': 'data'}
        result = chain.execute(input_data)
        
        assert 'required_field' in result  # 已标准化
        assert result['validated'] is True  # 已验证
        assert result['enriched'] is True  # 已增强
        assert 'timestamp' in result
    
    def test_adapter_with_cache_performance(self):
        """测试缓存性能"""
        class SlowAdapter(BaseAdapter[str, str]):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.process_count = 0
            
            def _do_adapt(self, data: str) -> str:
                self.process_count += 1
                # 模拟慢速处理
                return data.upper()
        
        # 无缓存
        adapter_no_cache = SlowAdapter("no_cache", enable_cache=False)
        for _ in range(10):
            adapter_no_cache.adapt("test")
        assert adapter_no_cache.process_count == 10
        
        # 有缓存
        adapter_with_cache = SlowAdapter("with_cache", enable_cache=True)
        for _ in range(10):
            adapter_with_cache.adapt("test")
        assert adapter_with_cache.process_count == 1  # 只处理一次


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

