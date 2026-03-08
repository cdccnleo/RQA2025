#!/usr/bin/env python3
"""
重构后的适配器实现

基于BaseAdapter重构，消除代码重复，提供统一的适配器框架

重构说明：
- 原有7个适配器文件存在大量重复（~800-1200行）
- 每个适配器都有相似的错误处理、日志记录、数据验证逻辑
- 使用BaseAdapter基类后，代码更简洁、功能更强大

迁移示例：
    # 旧方式
    from src.infrastructure.integration.adapters.trading_adapter import TradingAdapter
    
    # 新方式（推荐）
    from src.infrastructure.integration.adapters.refactored_adapters import UnifiedTradingAdapter

创建时间: 2025-11-03
版本: 2.0
"""

from typing import Dict, Any, List, Optional
from decimal import Decimal
from src.core.foundation.base_adapter import BaseAdapter, adapter


@adapter("trading", enable_cache=True)
class UnifiedTradingAdapter(BaseAdapter[Dict[str, Any], Dict[str, Any]]):
    """
    统一交易适配器（重构版）
    
    基于BaseAdapter，提供：
    - 自动化的数据验证
    - 智能缓存支持
    - 统一的错误处理
    - 性能监控
    """
    
    def _do_adapt(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        适配交易数据
        
        Args:
            data: 原始交易数据
            
        Returns:
            标准化的交易数据
        """
        return {
            'symbol': data.get('symbol', '').upper(),
            'price': Decimal(str(data.get('price', 0))),
            'quantity': int(data.get('quantity', 0)),
            'side': data.get('side', 'buy').lower(),
            'timestamp': data.get('timestamp'),
            'order_type': data.get('order_type', 'market'),
            'metadata': data.get('metadata', {})
        }
    
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """验证输入的交易数据"""
        if not super().validate_input(data):
            return False
        
        # 交易特定的验证
        required_fields = ['symbol', 'price', 'quantity']
        for field in required_fields:
            if field not in data:
                self._logger.error(f"缺少必需字段: {field}")
                return False
        
        # 价格和数量必须大于0
        try:
            if float(data['price']) <= 0:
                self._logger.error("价格必须大于0")
                return False
            if int(data['quantity']) <= 0:
                self._logger.error("数量必须大于0")
                return False
        except (ValueError, TypeError) as e:
            self._logger.error(f"数据类型错误: {e}")
            return False
        
        return True
    
    def _preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """预处理交易数据"""
        # 清理和标准化数据
        processed = data.copy()
        
        # 标准化symbol（转大写，去空格）
        if 'symbol' in processed:
            processed['symbol'] = processed['symbol'].strip().upper()
        
        # 标准化side（转小写）
        if 'side' in processed:
            processed['side'] = processed['side'].lower()
            if processed['side'] not in ['buy', 'sell']:
                processed['side'] = 'buy'  # 默认值
        
        return processed
    
    def _handle_error(self, data: Dict[str, Any], error: Exception) -> Dict[str, Any]:
        """错误恢复：返回默认值"""
        self._logger.warning(f"交易适配失败，返回默认值: {error}")
        return {
            'symbol': 'UNKNOWN',
            'price': Decimal('0'),
            'quantity': 0,
            'side': 'buy',
            'timestamp': None,
            'order_type': 'market',
            'metadata': {'error': str(error)}
        }


@adapter("risk", enable_cache=True)
class UnifiedRiskAdapter(BaseAdapter[Dict[str, Any], Dict[str, Any]]):
    """
    统一风险适配器（重构版）
    
    负责风险数据的标准化和验证
    """
    
    def _do_adapt(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """适配风险数据"""
        return {
            'risk_level': data.get('risk_level', 'medium'),
            'risk_score': float(data.get('risk_score', 0)),
            'var': Decimal(str(data.get('var', 0))),  # Value at Risk
            'max_drawdown': Decimal(str(data.get('max_drawdown', 0))),
            'sharpe_ratio': float(data.get('sharpe_ratio', 0)),
            'volatility': float(data.get('volatility', 0)),
            'timestamp': data.get('timestamp'),
            'portfolio_id': data.get('portfolio_id'),
            'alerts': data.get('alerts', [])
        }
    
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """验证风险数据"""
        if not super().validate_input(data):
            return False
        
        # 风险评分应该在0-100之间
        if 'risk_score' in data:
            try:
                score = float(data['risk_score'])
                if not (0 <= score <= 100):
                    self._logger.error(f"风险评分超出范围: {score}")
                    return False
            except (ValueError, TypeError):
                self._logger.error("风险评分格式错误")
                return False
        
        return True
    
    def _preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """预处理风险数据"""
        processed = data.copy()
        
        # 标准化风险等级
        if 'risk_level' in processed:
            level = processed['risk_level'].lower()
            if level not in ['low', 'medium', 'high', 'critical']:
                processed['risk_level'] = 'medium'
            else:
                processed['risk_level'] = level
        
        return processed


@adapter("security", enable_cache=False)  # 安全数据不应缓存
class UnifiedSecurityAdapter(BaseAdapter[Dict[str, Any], Dict[str, Any]]):
    """
    统一安全适配器（重构版）
    
    负责安全相关数据的处理
    """
    
    def _do_adapt(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """适配安全数据"""
        return {
            'user_id': data.get('user_id'),
            'token': data.get('token'),
            'permissions': data.get('permissions', []),
            'session_id': data.get('session_id'),
            'ip_address': data.get('ip_address'),
            'device_id': data.get('device_id'),
            'timestamp': data.get('timestamp'),
            'action': data.get('action'),
            'resource': data.get('resource')
        }
    
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """验证安全数据"""
        if not super().validate_input(data):
            return False
        
        # 安全数据必须包含user_id
        if 'user_id' not in data:
            self._logger.error("缺少user_id")
            return False
        
        return True
    
    def _postprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """后处理：脱敏敏感信息"""
        processed = data.copy()
        
        # 脱敏token（只显示前4位和后4位）
        if 'token' in processed and processed['token']:
            token = processed['token']
            if len(token) > 8:
                processed['token'] = f"{token[:4]}***{token[-4:]}"
        
        return processed


@adapter("features", enable_cache=True)
class UnifiedFeaturesAdapter(BaseAdapter[Dict[str, Any], Dict[str, Any]]):
    """
    统一特征适配器（重构版）
    
    负责机器学习特征数据的标准化
    """
    
    def _do_adapt(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """适配特征数据"""
        return {
            'feature_id': data.get('feature_id'),
            'feature_name': data.get('feature_name'),
            'feature_values': list(data.get('feature_values', [])),
            'feature_type': data.get('feature_type', 'numeric'),
            'normalization': data.get('normalization', 'none'),
            'missing_value_strategy': data.get('missing_value_strategy', 'mean'),
            'importance_score': float(data.get('importance_score', 0)),
            'timestamp': data.get('timestamp')
        }
    
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """验证特征数据"""
        if not super().validate_input(data):
            return False
        
        # 必须有feature_name
        if 'feature_name' not in data:
            self._logger.error("缺少feature_name")
            return False
        
        # feature_values必须是列表
        if 'feature_values' in data and not isinstance(data['feature_values'], (list, tuple)):
            self._logger.error("feature_values必须是列表或元组")
            return False
        
        return True


# 创建适配器实例的便捷函数
def create_adapters() -> Dict[str, BaseAdapter]:
    """
    创建所有适配器的便捷函数
    
    Returns:
        包含所有适配器实例的字典
    """
    return {
        'trading': UnifiedTradingAdapter(
            name='trading_adapter',
            enable_cache=True
        ),
        'risk': UnifiedRiskAdapter(
            name='risk_adapter',
            enable_cache=True
        ),
        'security': UnifiedSecurityAdapter(
            name='security_adapter',
            enable_cache=False
        ),
        'features': UnifiedFeaturesAdapter(
            name='features_adapter',
            enable_cache=True
        )
    }


def get_adapter(adapter_type: str) -> Optional[BaseAdapter]:
    """
    获取特定类型的适配器
    
    Args:
        adapter_type: 适配器类型（trading, risk, security, features）
        
    Returns:
        适配器实例
    """
    adapters = create_adapters()
    return adapters.get(adapter_type)


__all__ = [
    'UnifiedTradingAdapter',
    'UnifiedRiskAdapter',
    'UnifiedSecurityAdapter',
    'UnifiedFeaturesAdapter',
    'create_adapters',
    'get_adapter'
]

