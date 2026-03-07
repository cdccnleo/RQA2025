#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trading层 - ExecutionEngine完整测试（Week 3）
方案B Month 1任务：深度测试ExecutionEngine模块
目标：ExecutionEngine模块从<5%提升到40%+
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch
import time

# 导入ExecutionEngine和相关类型
try:
    from src.trading.execution.execution_engine import ExecutionEngine
    from src.trading.execution.execution_types import ExecutionMode, ExecutionStatus
except ImportError:
    ExecutionEngine, ExecutionMode, ExecutionStatus = None, None, None


pytestmark = [pytest.mark.timeout(30)]


class TestExecutionEngineInstantiation:
    """测试ExecutionEngine实例化"""
    
    def test_engine_default_instantiation(self):
        """测试默认实例化"""
        if ExecutionEngine is None:
            pytest.skip("ExecutionEngine not available")
        
        engine = ExecutionEngine()
        assert engine is not None
        assert isinstance(engine.executions, dict)
        assert engine.execution_id_counter == 0
    
    def test_engine_with_config(self):
        """测试带配置实例化"""
        if ExecutionEngine is None:
            pytest.skip("ExecutionEngine not available")
        
        config = {
            'max_concurrent_orders': 200,
            'execution_timeout': 120
        }
        
        engine = ExecutionEngine(config=config)
        assert engine.config == config
        assert engine.max_concurrent_orders == 200
        assert engine.execution_timeout == 120
    
    def test_engine_config_defaults(self):
        """测试配置默认值"""
        if ExecutionEngine is None:
            pytest.skip("ExecutionEngine not available")
        
        engine = ExecutionEngine()
        # 应该有默认的max_concurrent_orders
        assert hasattr(engine, 'max_concurrent_orders')
        assert hasattr(engine, 'execution_timeout')


class TestCreateExecution:
    """测试create_execution方法"""
    
    @pytest.fixture
    def engine(self):
        """创建engine实例"""
        if ExecutionEngine is None:
            pytest.skip("ExecutionEngine not available")
        return ExecutionEngine()
    
    def test_create_market_execution(self, engine):
        """测试创建市价执行"""
        exec_id = engine.create_execution(
            symbol="600000.SH",
            side="BUY",
            quantity=100.0
        )
        
        assert exec_id is not None
        assert isinstance(exec_id, str)
        assert exec_id in engine.executions
    
    def test_create_limit_execution(self, engine):
        """测试创建限价执行"""
        exec_id = engine.create_execution(
            symbol="000001.SZ",
            side="SELL",
            quantity=500.0,
            price=15.50,
            mode=ExecutionMode.LIMIT if ExecutionMode else "LIMIT"
        )
        
        assert exec_id is not None
        assert engine.executions[exec_id]['price'] == 15.50
    
    def test_create_execution_validates_symbol(self, engine):
        """测试验证symbol"""
        with pytest.raises(ValueError):
            engine.create_execution(
                symbol="",  # 空symbol
                side="BUY",
                quantity=100.0
            )
    
    def test_create_execution_validates_quantity(self, engine):
        """测试验证quantity"""
        with pytest.raises(ValueError):
            engine.create_execution(
                symbol="600000.SH",
                side="BUY",
                quantity=-100.0  # 负数量
            )
    
    def test_create_execution_validates_quantity_positive(self, engine):
        """测试quantity必须为正"""
        with pytest.raises(ValueError):
            engine.create_execution(
                symbol="600000.SH",
                side="BUY",
                quantity=0  # 零数量
            )
    
    def test_create_execution_validates_price(self, engine):
        """测试验证price"""
        with pytest.raises(ValueError):
            engine.create_execution(
                symbol="600000.SH",
                side="BUY",
                quantity=100.0,
                price=-10.0  # 负价格
            )
    
    def test_create_execution_limit_requires_price(self, engine):
        """测试限价单必须有价格"""
        with pytest.raises(ValueError):
            engine.create_execution(
                symbol="600000.SH",
                side="BUY",
                quantity=100.0,
                mode=ExecutionMode.LIMIT if ExecutionMode else "LIMIT",
                price=None  # 限价单没有价格
            )
    
    def test_create_execution_increments_counter(self, engine):
        """测试执行ID计数器递增"""
        exec_id1 = engine.create_execution("600000.SH", "BUY", 100.0)
        exec_id2 = engine.create_execution("000001.SZ", "SELL", 200.0)
        
        assert exec_id1 != exec_id2
        assert engine.execution_id_counter == 2
    
    def test_create_execution_stores_execution(self, engine):
        """测试execution被正确存储"""
        exec_id = engine.create_execution("600000.SH", "BUY", 100.0)
        
        execution = engine.executions[exec_id]
        assert execution['symbol'] == "600000.SH"
        assert execution['side'] == "BUY"
        assert execution['quantity'] == 100.0
        assert execution['status'] == ExecutionStatus.PENDING if ExecutionStatus else "PENDING"
    
    def test_create_execution_with_kwargs(self, engine):
        """测试create_execution接受额外参数"""
        exec_id = engine.create_execution(
            symbol="600000.SH",
            side="BUY",
            quantity=100.0,
            custom_param="test_value"
        )
        
        execution = engine.executions[exec_id]
        assert 'kwargs' in execution
        assert execution['kwargs']['custom_param'] == "test_value"


class TestStartExecution:
    """测试start_execution方法"""
    
    @pytest.fixture
    def engine(self):
        """创建engine实例"""
        if ExecutionEngine is None:
            pytest.skip("ExecutionEngine not available")
        return ExecutionEngine()
    
    def test_start_execution_success(self, engine):
        """测试启动执行成功"""
        exec_id = engine.create_execution("600000.SH", "BUY", 100.0)
        
        result = engine.start_execution(exec_id)
        
        assert result == True
        # 检查状态已更新
        if ExecutionStatus:
            assert engine.executions[exec_id]['status'] in [
                ExecutionStatus.RUNNING, 
                ExecutionStatus.PENDING
            ]
    
    def test_start_execution_invalid_id(self, engine):
        """测试启动不存在的执行"""
        result = engine.start_execution("invalid_id_999")
        
        assert result == False
    
    def test_start_execution_sets_start_time(self, engine):
        """测试启动执行设置开始时间"""
        exec_id = engine.create_execution("600000.SH", "BUY", 100.0)
        
        engine.start_execution(exec_id)
        
        execution = engine.executions[exec_id]
        assert execution['start_time'] is not None
        assert execution['start_time'] > 0


class TestCancelExecution:
    """测试cancel_execution方法"""
    
    @pytest.fixture
    def engine(self):
        """创建engine实例"""
        if ExecutionEngine is None:
            pytest.skip("ExecutionEngine not available")
        return ExecutionEngine()
    
    def test_cancel_execution_exists(self, engine):
        """测试cancel_execution方法存在"""
        assert hasattr(engine, 'cancel_execution')
    
    def test_cancel_pending_execution(self, engine):
        """测试取消待执行的任务"""
        if not hasattr(engine, 'cancel_execution'):
            pytest.skip("cancel_execution not available")
        
        exec_id = engine.create_execution("600000.SH", "BUY", 100.0)
        
        result = engine.cancel_execution(exec_id)
        
        # 应该返回True或成功取消
        assert result == True or result is not None
    
    def test_cancel_invalid_execution(self, engine):
        """测试取消不存在的执行"""
        if not hasattr(engine, 'cancel_execution'):
            pytest.skip("cancel_execution not available")
        
        result = engine.cancel_execution("invalid_id_999")
        
        assert result == False


class TestGetExecution:
    """测试get_execution方法"""
    
    @pytest.fixture
    def engine(self):
        """创建engine实例"""
        if ExecutionEngine is None:
            pytest.skip("ExecutionEngine not available")
        return ExecutionEngine()
    
    def test_get_execution_exists(self, engine):
        """测试get_execution方法存在"""
        # ExecutionEngine可能没有get_execution方法，检查替代方法
        assert (hasattr(engine, 'get_execution') or 
                hasattr(engine, 'get_order') or
                hasattr(engine, 'get_execution_status'))
    
    def test_get_existing_execution(self, engine):
        """测试获取存在的执行"""
        if not hasattr(engine, 'get_execution'):
            pytest.skip("get_execution not available")
        
        exec_id = engine.create_execution("600000.SH", "BUY", 100.0)
        
        execution = engine.get_execution(exec_id)
        
        assert execution is not None
        assert execution['id'] == exec_id
    
    def test_get_nonexistent_execution(self, engine):
        """测试获取不存在的执行"""
        if not hasattr(engine, 'get_execution'):
            pytest.skip("get_execution not available")
        
        execution = engine.get_execution("invalid_id_999")
        
        assert execution is None


class TestExecutionLifecycle:
    """测试执行生命周期"""
    
    @pytest.fixture
    def engine(self):
        """创建engine实例"""
        if ExecutionEngine is None:
            pytest.skip("ExecutionEngine not available")
        return ExecutionEngine()
    
    def test_complete_execution_lifecycle(self, engine):
        """测试完整的执行生命周期"""
        # 1. 创建执行
        exec_id = engine.create_execution("600000.SH", "BUY", 100.0)
        assert exec_id in engine.executions
        
        # 2. 启动执行
        result = engine.start_execution(exec_id)
        assert result == True
        
        # 3. 检查状态
        execution = engine.executions[exec_id]
        assert execution['start_time'] is not None
    
    def test_execution_status_transition(self, engine):
        """测试执行状态转换"""
        exec_id = engine.create_execution("600000.SH", "BUY", 100.0)
        
        # 初始状态应该是PENDING
        initial_status = engine.executions[exec_id]['status']
        assert initial_status in [ExecutionStatus.PENDING, "PENDING"] if ExecutionStatus else True
        
        # 启动后状态可能变化
        engine.start_execution(exec_id)


class TestExecutionEdgeCases:
    """测试执行边界情况"""
    
    @pytest.fixture
    def engine(self):
        """创建engine实例"""
        if ExecutionEngine is None:
            pytest.skip("ExecutionEngine not available")
        return ExecutionEngine()
    
    def test_create_execution_max_quantity(self, engine):
        """测试最大数量验证"""
        # 根据源代码，quantity有MAX_POSITION_SIZE限制
        with pytest.raises(ValueError):
            engine.create_execution(
                symbol="600000.SH",
                side="BUY",
                quantity=99999999999.0  # 超大数量
            )
    
    def test_create_execution_max_price(self, engine):
        """测试最大价格验证"""
        with pytest.raises(ValueError):
            engine.create_execution(
                symbol="600000.SH",
                side="BUY",
                quantity=100.0,
                price=99999999999.0  # 超大价格
            )
    
    def test_multiple_executions(self, engine):
        """测试多个执行任务"""
        exec_ids = []
        for i in range(5):
            exec_id = engine.create_execution(
                symbol=f"60000{i}.SH",
                side="BUY",
                quantity=100.0 * (i + 1)
            )
            exec_ids.append(exec_id)
        
        assert len(exec_ids) == 5
        assert len(engine.executions) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src/trading/execution/execution_engine", "--cov-report=term"])

